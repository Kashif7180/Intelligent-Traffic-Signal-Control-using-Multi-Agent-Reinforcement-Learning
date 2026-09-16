"""
Phase 8R Training Script: MAPPO + Temporal Transformer + GAT + Improved Causal Communication.

Trains the Phase 8R architecture on SAGE-Traffic:
- Temporal Transformer Encoder (43-D -> 64-D)
- Gated GAT with Dedicated Causal Pathway phi(C_ij) in R^8
- Causal Influence Estimator (directional C_ij in [0, 1] over 8 allowed edges)
- Multi-layer Gate MLP ([h_i, h_j, phi(C_ij)]) -> 64 -> 32 -> 1 -> Sigmoid
- Causal-Aware Message Modulation (m_ij = g_ij * alpha_ij * W h_j)
- Differentiable Communication Budget Regularization:
  L_comm_budget = lambda_budget * max(0, mean(g) - target_ratio)^2 + lambda_comm * mean(g)
- Decentralized Actor (68-D -> 4 actions)
- Centralized Critic (256-D -> 1 scalar value V(S_comm))

Setup:
- 40 episodes, 100 decision steps per episode (500s SUMO simulation time)
- Training Seeds 43 to 82 (base_seed=42 + ep)
- Evaluation seeds 1001-1010 remain untouched
- Baselines (Phase 4 MAPPO, Phase 6 GAT, Phase 7, Phase 8) remain frozen
- Checkpoints saved to models/checkpoints/mappo_causal_improved_*.pt
- Logs saved to experiments/mappo_causal_improved_training_log.csv
- Training plot saved to experiments/mappo_causal_improved_reward_curve.png
"""

import os
import sys
import csv
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.traffic_env import MultiAgentTrafficEnv
from models.causal_influence import ALLOWED_DIRECTED_EDGES
from models.causal_improved import MAPPOCausalImprovedController


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    p8r_cfg = config.get("phase8r", {})
    causal_cfg = p8r_cfg.get("causal", {})
    comm_cfg = p8r_cfg.get("communication", {})
    train_cfg = p8r_cfg.get("training", {})
    sim_cfg = config.get("simulation", {})
    agent_cfg = config.get("agent", {})
    tf_cfg = config.get("transformer", {})

    train_episodes = int(train_cfg.get("train_episodes", 40))
    steps_per_episode = int(train_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Output directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_causal_improved_training_log.csv"
    plot_file = "experiments/mappo_causal_improved_reward_curve.png"

    print("=" * 125)
    print("SAGE-Traffic Phase 8R: Improved Causal Communication Training (40 Episodes)")
    print("=" * 125)
    print("Architecture:               CTDE (Decentralized Actors, Centralized Critic)")
    print(f"Temporal History:           k={tf_cfg.get('history_length', 4)} timesteps (43-D -> 64-D)")
    print(f"Dedicated Causal Pathway:   phi(C_ij) in R^{causal_cfg.get('causal_embedding_dim', 8)} (MLP 1->16->8 + LayerNorm)")
    print(f"Communication Gate:         MLP([h_i, h_j, phi(C_ij)]) -> {comm_cfg.get('hidden_dims', [64, 32])} -> 1 -> Sigmoid")
    print(f"Message Modulation:         m_ij = g_ij * alpha_ij * W h_j (causal-aware)")
    print(f"Budget Regularization:      L_budget = {comm_cfg.get('budget_penalty_coef', 0.05)} * max(0, g - {comm_cfg.get('target_ratio', 0.50)})^2 + {comm_cfg.get('lambda_comm', 0.01)} * g")
    print(f"Training Horizon:           {train_episodes} episodes x {steps_per_episode} steps (500s SUMO simulation time)")
    print(f"Training Seeds:             {base_seed + 1} to {base_seed + train_episodes} (Evaluation seeds 1001-1010 untouched)")
    print(f"Actor LR / Critic LR:       {train_cfg.get('lr_actor', 0.0003)} / {train_cfg.get('lr_critic', 0.0005)}")
    print("-" * 125)

    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOCausalImprovedController(config_path=config_path)

    # CSV Header
    fieldnames = [
        "episode",
        "seed",
        "possible_communications",
        "actual_communications",
        "communication_ratio",
        "total_communication_cost",
        "task_return",
        "communication_adjusted_return",
        "delay",
        "queue",
        "throughput",
        "mean_C_ij",
        "min_C_ij",
        "max_C_ij",
        "mean_gate",
        "min_gate",
        "max_gate",
        "c_gate_corr",
        "policy_loss",
        "value_loss",
        "entropy",
        "comm_budget_loss",
        "causal_grad_norm"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"{'Ep':<4} | {'Seed':<5} | {'Task Ret':<10} | {'Adj Ret':<10} | {'Comm Cost':<10} | {'Comm Ratio':<11} | {'Mean C_ij':<10} | {'Mean Gate':<10} | {'Delay(s)':<9} | {'Queue':<7} | {'Thru':<6} | {'Grad_C':<8} | {'BudgetLoss':<10}")
    print("-" * 125)

    ep_history = []

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)
            ctrl.reset_history(obs, info)

            ep_task_rewards = {a: 0.0 for a in env.agents}
            ep_comm_costs = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_actual_comm = 0
            ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

            step_c_scores = []
            step_gate_values = []
            c_vals_flat = []
            g_vals_flat = []

            for step in range(steps_per_episode):
                # 1. Directional causal influence estimation scores C_ij
                c_scores = ctrl.compute_causal_context(info)
                for edge in ALLOWED_DIRECTED_EDGES:
                    c_val = c_scores.get(edge, 0.5)
                    step_c_scores.append(c_val)
                    c_vals_flat.append(c_val)

                # 2. Snapshot history window
                histories = ctrl.history_manager.get_all_histories()

                # 3. Action selection via improved causal gated communication
                actions, log_probs, comm_costs, actual_comm = ctrl.get_actions(c_scores, deterministic=False)
                ep_actual_comm += actual_comm

                # Track gate values
                for edge in ALLOWED_DIRECTED_EDGES:
                    g_val = ctrl.gated_gat.comm_gate.last_gate_values.get(edge, 0.5)
                    step_gate_values.append(g_val)
                    g_vals_flat.append(g_val)

                # 4. Centralized critic evaluation
                value = ctrl.get_centralized_value(c_scores)

                # 5. Environment step
                next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

                # 6. Store in buffer
                done = any(terminations.values()) or any(truncations.values())
                ctrl.buffer.add(
                    histories=histories,
                    context_dict=c_scores,
                    value=value,
                    actions=actions,
                    log_probs=log_probs,
                    task_rewards=task_rewards,
                    comm_costs=comm_costs,
                    actual_comm=actual_comm,
                    done=done
                )

                # Accumulate step metrics
                for a in env.agents:
                    ep_task_rewards[a] += task_rewards[a]
                    ep_comm_costs[a] += comm_costs[a]

                step_queues = [next_info[a]["metrics"]["queue_length"] for a in env.agents]
                step_waits = [next_info[a]["metrics"]["waiting_time"] for a in env.agents]
                ep_queues.append(np.mean(step_queues))
                ep_waiting.append(np.mean(step_waits))

                ctrl.update_history(next_obs)
                obs = next_obs
                info = next_info

            # Final step bootstrap value for GAE
            last_c_scores = ctrl.compute_causal_context(info)
            last_val = ctrl.get_centralized_value(last_c_scores)

            # MAPPO PPO update (incorporating budget loss directly)
            loss_dict = ctrl.update(last_val)

            # Episode summary metrics
            ep_throughput = sum(info[a]["metrics"]["vehicle_count"] for a in env.agents) // 4
            task_ret = float(sum(ep_task_rewards.values()))
            total_comm_cost = float(sum(ep_comm_costs.values()))
            adj_ret = task_ret - total_comm_cost
            mean_delay = float(np.mean(ep_waiting))
            mean_queue = float(np.mean(ep_queues))
            comm_ratio = float(ep_actual_comm / ep_possible_comm)
            mean_c = float(np.mean(step_c_scores)) if step_c_scores else 0.5
            min_c = float(np.min(step_c_scores)) if step_c_scores else 0.5
            max_c = float(np.max(step_c_scores)) if step_c_scores else 0.5
            mean_g = float(np.mean(step_gate_values)) if step_gate_values else 0.5
            min_g = float(np.min(step_gate_values)) if step_gate_values else 0.5
            max_g = float(np.max(step_gate_values)) if step_gate_values else 0.5

            # Correlation between C_ij and gate probability across steps
            if len(c_vals_flat) > 1 and np.std(c_vals_flat) > 1e-6 and np.std(g_vals_flat) > 1e-6:
                corr_matrix = np.corrcoef(c_vals_flat, g_vals_flat)
                c_g_corr = float(corr_matrix[0, 1])
                if np.isnan(c_g_corr):
                    c_g_corr = 0.0
            else:
                c_g_corr = 0.0

            row = {
                "episode": ep,
                "seed": ep_seed,
                "possible_communications": ep_possible_comm,
                "actual_communications": ep_actual_comm,
                "communication_ratio": comm_ratio,
                "total_communication_cost": total_comm_cost,
                "task_return": task_ret,
                "communication_adjusted_return": adj_ret,
                "delay": mean_delay,
                "queue": mean_queue,
                "throughput": ep_throughput,
                "mean_C_ij": mean_c,
                "min_C_ij": min_c,
                "max_C_ij": max_c,
                "mean_gate": mean_g,
                "min_gate": min_g,
                "max_gate": max_g,
                "c_gate_corr": c_g_corr,
                "policy_loss": loss_dict["policy_loss"],
                "value_loss": loss_dict["value_loss"],
                "entropy": loss_dict["entropy"],
                "comm_budget_loss": loss_dict["comm_budget_loss"],
                "causal_grad_norm": loss_dict["causal_grad_norm"]
            }

            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            ep_history.append(row)

            print(
                f"{ep:<4} | {ep_seed:<5} | {task_ret:<10.2f} | {adj_ret:<10.2f} | {total_comm_cost:<10.2f} | "
                f"{comm_ratio * 100:<10.2f}% | {mean_c:<10.3f} | {mean_g:<10.3f} | {mean_delay:<9.2f} | {mean_queue:<7.2f} | {ep_throughput:<6} | {loss_dict['causal_grad_norm']:<8.4f} | {loss_dict['comm_budget_loss']:<10.4f}"
            )

        # Save Phase 8R Checkpoints
        ctrl.save_checkpoints("models/checkpoints")
        print("\n" + "=" * 125)
        print("Phase 8R Checkpoints successfully saved to:")
        print("  - models/checkpoints/mappo_causal_improved_actor.pt")
        print("  - models/checkpoints/mappo_causal_improved_critic.pt")
        print("=" * 125)

    finally:
        env.close()

    # Generate Training Reward & Metric Curves
    if ep_history:
        episodes = [r["episode"] for r in ep_history]
        task_returns = [r["task_return"] for r in ep_history]
        adj_returns = [r["communication_adjusted_return"] for r in ep_history]
        delays = [r["delay"] for r in ep_history]
        queues = [r["queue"] for r in ep_history]
        comm_ratios = [r["communication_ratio"] * 100 for r in ep_history]
        mean_cs = [r["mean_C_ij"] for r in ep_history]
        mean_gs = [r["mean_gate"] for r in ep_history]
        causal_grads = [r["causal_grad_norm"] for r in ep_history]
        budget_losses = [r["comm_budget_loss"] for r in ep_history]

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

        # Panel 1: Task vs. Adjusted Return
        ax1.plot(episodes, task_returns, color="#1f77b4", linewidth=2.0, label="Task-Only Return")
        ax1.plot(episodes, adj_returns, color="#ff7f0e", linewidth=1.8, linestyle="--", label="Comm-Adjusted Return")
        ax1.set_title("Phase 8R: Return vs. Training Episodes", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Return")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        # Panel 2: Average Delay & Queue
        ax2.plot(episodes, delays, color="#d62728", linewidth=2.0, label="Delay (s)")
        ax2_q = ax2.twinx()
        ax2_q.plot(episodes, queues, color="#9467bd", linewidth=1.8, linestyle=":", label="Queue Length")
        ax2.set_title("Phase 8R: Delay and Queue vs. Training Episodes", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Delay (s)", color="#d62728")
        ax2_q.set_ylabel("Average Queue Length", color="#9467bd")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Communication Ratio vs Target Budget (50%)
        ax3.plot(episodes, comm_ratios, color="#2ca02c", linewidth=2.0, label="Comm Ratio (%)")
        ax3.axhline(50.0, color="#d62728", linestyle="--", linewidth=1.5, label="Target Budget (50%)")
        ax3.set_title("Phase 8R: Communication Ratio vs. Training Episodes", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Communication Ratio (%)")
        ax3.set_ylim(-5, 105)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        # Panel 4: Mean C_ij and Mean Gate Probability
        ax4.plot(episodes, mean_cs, color="#8c564b", linewidth=2.0, label="Mean C_ij")
        ax4.plot(episodes, mean_gs, color="#e377c2", linewidth=2.0, linestyle="--", label="Mean Gate Prob")
        ax4.set_title("Phase 8R: Mean C_ij & Gate Probability vs. Training Episodes", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Value [0, 1]")
        ax4.set_ylim(-0.05, 1.05)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper right")

        # Panel 5: Causal Pathway Gradient Norm
        ax5.plot(episodes, causal_grads, color="#17becf", linewidth=2.0, label="Causal Grad Norm")
        ax5.set_title("Phase 8R: Causal Pathway Gradient Norm vs. Episodes", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Gradient Norm")
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc="upper right")

        # Panel 6: Budget Loss
        ax6.plot(episodes, budget_losses, color="#bcbd22", linewidth=2.0, label="Comm Budget Loss")
        ax6.set_title("Phase 8R: Communication Budget Loss vs. Episodes", fontsize=12, fontweight="bold")
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Loss")
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Training visualization successfully saved to {plot_file}")


if __name__ == "__main__":
    train()
