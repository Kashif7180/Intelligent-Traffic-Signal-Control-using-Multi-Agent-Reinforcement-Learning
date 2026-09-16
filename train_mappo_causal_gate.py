"""
Phase 8 Training Script: MAPPO + Temporal Transformer + GAT + Causal Influence Gate.

Trains the complete architecture on SAGE-Traffic:
- Temporal Transformer Encoder (43-D -> 64-D)
- Gated Graph Attention Network (GAT) with Causal Influence Gate
- Causal Influence Estimator (directional C_ij in [0, 1] over 8 allowed edges)
- Decentralized Actor (68-D -> 4 actions)
- Centralized Critic (256-D -> 1 scalar value V(S_comm))

Conditions match Phase 7:
- 40 episodes, 100 decision steps per episode (500s SUMO simulation time)
- Seeds 43 to 82
- Normal demand profile
- Communication penalty lambda_comm * comm_cost included in reward
- Logs to experiments/mappo_causal_gate_training_log.csv
- Checkpoints saved to models/checkpoints/mappo_causal_gate_*.pt
- Real training plot saved to experiments/mappo_causal_gate_reward_curve.png
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
from models.communication_gate import MAPPOCausalGateController, ALLOWED_DIRECTED_EDGES


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    comm_cfg = config.get("communication", {})
    tf_cfg = config.get("transformer", {})
    gat_cfg = config.get("gat", {})
    ci_cfg = config.get("causal_influence", {})
    sim_cfg = config.get("simulation", {})
    agent_cfg = config.get("agent", {})

    train_episodes = int(comm_cfg.get("train_episodes", 40))
    steps_per_episode = int(comm_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_causal_gate_training_log.csv"
    plot_file = "experiments/mappo_causal_gate_reward_curve.png"

    print("=" * 115)
    print("SAGE-Traffic Phase 8: MAPPO + Temporal + GAT + Causal Influence Gate Training")
    print("=" * 115)
    print("Architecture:            CTDE (Decentralized Actors, Centralized Critic)")
    print(f"Temporal History:        k={tf_cfg.get('history_length', 4)} timesteps")
    print(f"Communication Graph:     8 Directed Edges over Physical Road Network")
    print(f"Causal Estimator Method: {ci_cfg.get('method', 'granger_style')} (window_size={ci_cfg.get('window_size', 15)})")
    print(f"Communication Gate:      MLP([h_i, h_j, C_ij]) -> Sigmoid (hidden_dim={comm_cfg.get('hidden_dim', 32)})")
    print(f"Communication Cost:      lambda_comm = {comm_cfg.get('cost', 0.01)} per transmitted message")
    print(f"Intersections:           {agent_cfg['agents']} (4 Signalized Intersections)")
    print(f"Training Episodes:       {train_episodes}")
    print(f"Steps per Episode:       {steps_per_episode} (500s SUMO simulation time)")
    print(f"Actor LR / Critic LR:    {comm_cfg.get('lr_actor', 0.0003)} / {comm_cfg.get('lr_critic', 0.0005)}")
    print("-" * 115)

    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOCausalGateController(config_path=config_path)

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
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"{'Ep':<4} | {'Seed':<5} | {'Task Ret':<10} | {'Adj Ret':<10} | {'Comm Cost':<10} | {'Comm Ratio':<12} | {'Mean C_ij':<10} | {'Delay (s)':<10} | {'Queue':<8} | {'Throughput':<10}")
    print("-" * 115)

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
            ep_throughput = 0
            ep_actual_comm = 0
            ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

            step_c_scores = []
            step_gate_values = []

            for step in range(steps_per_episode):
                # 1. Directional causal influence estimation scores C_ij
                c_scores = ctrl.compute_causal_context(info)
                step_c_scores.extend(list(c_scores.values()))

                # 2. Snapshot history window
                histories = ctrl.history_manager.get_all_histories()

                # 3. Action selection via causal gated communication
                actions, log_probs, comm_costs, actual_comm = ctrl.get_actions(c_scores, deterministic=False)
                ep_actual_comm += actual_comm

                # Track gate values
                step_gate_values.extend(list(ctrl.gated_gat.comm_gate.last_gate_values.values()))

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

            # MAPPO PPO update
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
                "policy_loss": loss_dict["policy_loss"],
                "value_loss": loss_dict["value_loss"],
                "entropy": loss_dict["entropy"],
                "approx_kl": loss_dict["approx_kl"]
            }

            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            ep_history.append(row)

            print(
                f"{ep:<4} | {ep_seed:<5} | {task_ret:<10.2f} | {adj_ret:<10.2f} | {total_comm_cost:<10.2f} | "
                f"{comm_ratio * 100:<11.2f}% | {mean_c:<10.3f} | {mean_delay:<10.2f} | {mean_queue:<8.2f} | {ep_throughput:<10}"
            )

        # Save Phase 8 Checkpoints
        ctrl.save_checkpoints("models/checkpoints")
        print("\nCheckpoints successfully saved to models/checkpoints/mappo_causal_gate_*.pt")

    finally:
        env.close()

    # Generate Training Reward & Metric Curve
    if ep_history:
        episodes = [r["episode"] for r in ep_history]
        task_returns = [r["task_return"] for r in ep_history]
        adj_returns = [r["communication_adjusted_return"] for r in ep_history]
        delays = [r["delay"] for r in ep_history]
        queues = [r["queue"] for r in ep_history]
        comm_ratios = [r["communication_ratio"] * 100 for r in ep_history]
        mean_cs = [r["mean_C_ij"] for r in ep_history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Task vs. Adjusted Return
        ax1.plot(episodes, task_returns, color="#1f77b4", linewidth=2.0, label="Task-Only Return")
        ax1.plot(episodes, adj_returns, color="#ff7f0e", linewidth=1.8, linestyle="--", label="Comm-Adjusted Return")
        ax1.set_title("Phase 8: Return vs. Training Episodes", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Return")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 2: Mean Delay
        ax2.plot(episodes, delays, color="#d62728", linewidth=2.0)
        ax2.set_title("Average Delay (seconds / vehicle)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Delay (s)")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Queue Length & Mean C_ij
        ax3.plot(episodes, queues, color="#2ca02c", linewidth=2.0, label="Mean Queue (veh)")
        ax3.set_title("Average Queue Length", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Queue (veh)")
        ax3.grid(True, alpha=0.3)

        # Panel 4: Communication Ratio & Mean C_ij
        ax4.plot(episodes, comm_ratios, color="#9467bd", linewidth=2.0, label="Comm Ratio (%)")
        ax4.plot(episodes, [c * 100 for c in mean_cs], color="#8c564b", linestyle=":", linewidth=2.0, label="Mean C_ij x 100")
        ax4.set_title("Communication Ratio & C_ij Dynamic", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Percentage (%)")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(plot_file, dpi=200)
        plt.close()
        print(f"Training reward curve saved to {plot_file}")

        # Also copy to artifacts directory
        artifact_plot = os.path.join(
            r"C:\Users\kashi\.gemini\antigravity-ide\brain\549c93f1-f3e5-4dbb-865e-10d0077ea528",
            "mappo_causal_gate_reward_curve.png"
        )
        try:
            import shutil
            shutil.copyfile(plot_file, artifact_plot)
        except Exception:
            pass


if __name__ == "__main__":
    train()
