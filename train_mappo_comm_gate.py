"""
Phase 7 Training Script: MAPPO + Temporal Transformer + GAT + Learnable Communication Gate.

Trains the complete architecture on SAGE-Traffic:
- Temporal Transformer Encoder (43-D -> 64-D)
- Gated Graph Attention Network (GAT) with Learnable Communication Gate
- Decentralized Actor (68-D -> 4 actions)
- Centralized Critic (256-D -> 1 scalar value V(S_comm))

Conditions match Phase 6:
- 40 episodes, 100 decision steps per episode (500s SUMO simulation time)
- Seeds 43 to 82
- Normal demand profile
- Communication penalty lambda_comm * comm_cost included in reward
- Logs to experiments/mappo_comm_gate_training_log.csv
- Checkpoints saved to models/checkpoints/mappo_comm_gate_*.pt
- Real training plot saved to experiments/mappo_comm_gate_reward_curve.png
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
from models.communication_gate import MAPPOGateController, ALLOWED_DIRECTED_EDGES


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    comm_cfg = config["communication"]
    tf_cfg = config["transformer"]
    gat_cfg = config["gat"]
    sim_cfg = config["simulation"]
    agent_cfg = config["agent"]

    train_episodes = int(comm_cfg.get("train_episodes", 40))
    steps_per_episode = int(comm_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_comm_gate_training_log.csv"
    plot_file = "experiments/mappo_comm_gate_reward_curve.png"

    print("=" * 105)
    print("SAGE-Traffic Phase 7: MAPPO + Temporal + GAT + Learnable Communication Gate Training")
    print("=" * 105)
    print("Architecture:            CTDE (Decentralized Actors, Centralized Critic)")
    print(f"Temporal History:        k={tf_cfg.get('history_length', 4)} timesteps")
    print(f"Communication Graph:     8 Directed Edges over Physical Road Network")
    print(f"Communication Gate:      MLP([h_i, h_j, c_ij]) -> Sigmoid (hidden_dim={comm_cfg.get('hidden_dim', 32)})")
    print(f"Communication Cost:      lambda_comm = {comm_cfg.get('cost', 0.01)} per transmitted message")
    print(f"Heuristic Context:       Normalized Queue/Pressure Difference (Non-Causal)")
    print(f"Intersections:           {agent_cfg['agents']} (4 Signalized Intersections)")
    print(f"Training Episodes:       {train_episodes}")
    print(f"Steps per Episode:       {steps_per_episode} (500s SUMO simulation time)")
    print(f"Actor LR / Critic LR:    {comm_cfg['lr_actor']} / {comm_cfg['lr_critic']}")
    print("-" * 105)

    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOGateController(config_path=config_path)

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
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"{'Ep':<4} | {'Seed':<5} | {'Task Ret':<10} | {'Adj Ret':<10} | {'Comm Cost':<10} | {'Comm Ratio':<12} | {'Delay (s)':<10} | {'Queue':<8} | {'Throughput':<10}")
    print("-" * 105)

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)
            ctrl.reset_history(obs)

            ep_task_rewards = {a: 0.0 for a in env.agents}
            ep_comm_costs = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0
            ep_actual_comm = 0
            ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

            for step in range(steps_per_episode):
                # 1. Compute heuristic context c_ij
                ctx = ctrl.compute_heuristic_context(info)

                # 2. Snapshot history window
                histories = ctrl.history_manager.get_all_histories()

                # 3. Action selection via gated communication
                actions, log_probs, comm_costs, actual_comm = ctrl.get_actions(ctx, deterministic=False)
                ep_actual_comm += actual_comm

                # 4. Centralized critic evaluation
                value = ctrl.get_centralized_value(ctx)

                # 5. Environment step
                next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

                # 6. Store in buffer
                done = any(terminations.values()) or any(truncations.values())
                ctrl.buffer.add(
                    histories=histories,
                    context_dict=ctx,
                    value=value,
                    actions=actions,
                    log_probs=log_probs,
                    task_rewards=task_rewards,
                    comm_costs=comm_costs,
                    actual_comm=actual_comm,
                    done=done
                )

                for a in env.agents:
                    ep_task_rewards[a] += task_rewards[a]
                    ep_comm_costs[a] += comm_costs[a]

                step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
                step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
                ep_queues.append(step_q)
                ep_waiting.append(step_w)
                ep_throughput += next_info["A"]["arrived_vehicles"]

                ctrl.update_history(next_obs)
                obs = next_obs
                info = next_info

                if done:
                    break

            # 7. End-of-episode PPO update
            last_ctx = ctrl.compute_heuristic_context(info)
            last_value = ctrl.get_centralized_value(last_ctx)
            loss_stats = ctrl.update(last_value)

            task_ret = sum(ep_task_rewards.values())
            tot_comm_cost = sum(ep_comm_costs.values())
            adj_ret = task_ret - tot_comm_cost
            comm_ratio = ep_actual_comm / ep_possible_comm
            m_queue = float(np.mean(ep_queues))
            m_delay = float(np.mean(ep_waiting))

            # Write to CSV
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "episode": ep,
                    "seed": ep_seed,
                    "possible_communications": ep_possible_comm,
                    "actual_communications": ep_actual_comm,
                    "communication_ratio": f"{comm_ratio:.4f}",
                    "total_communication_cost": f"{tot_comm_cost:.4f}",
                    "task_return": f"{task_ret:.2f}",
                    "communication_adjusted_return": f"{adj_ret:.2f}",
                    "delay": f"{m_delay:.2f}",
                    "queue": f"{m_queue:.2f}",
                    "throughput": ep_throughput,
                    "policy_loss": f"{loss_stats['policy_loss']:.4f}",
                    "value_loss": f"{loss_stats['value_loss']:.4f}",
                    "entropy": f"{loss_stats['entropy']:.4f}",
                    "approx_kl": f"{loss_stats['approx_kl']:.6f}"
                })

            print(f"{ep:<4} | {ep_seed:<5} | {task_ret:<10.1f} | {adj_ret:<10.1f} | {tot_comm_cost:<10.2f} | {comm_ratio*100:<10.1f}% | {m_delay:<10.2f} | {m_queue:<8.2f} | {ep_throughput:<10}")

        print("-" * 105)
        print("\nTraining complete! Saving Phase 7 checkpoints...")

        ctrl.save_checkpoints("models/checkpoints")
        print("  Saved:")
        print("    -> models/checkpoints/mappo_comm_gate_actor.pt")
        print("    -> models/checkpoints/mappo_comm_gate_critic.pt")

        print(f"\nGenerating real reward curve plot -> {plot_file}...")
        generate_plots(csv_file, plot_file)
        print("  Plot successfully saved.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


def generate_plots(csv_path: str, plot_path: str):
    """Generates training curves from real Phase 7 training CSV."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "episode": int(row["episode"]),
                "task_return": float(row["task_return"]),
                "adj_return": float(row["communication_adjusted_return"]),
                "comm_cost": float(row["total_communication_cost"]),
                "comm_ratio": float(row["communication_ratio"]),
                "delay": float(row["delay"]),
                "queue": float(row["queue"]),
                "value_loss": float(row["value_loss"])
            })

    episodes = [d["episode"] for d in data]
    task_returns = [d["task_return"] for d in data]
    adj_returns = [d["adj_return"] for d in data]
    comm_ratios = [d["comm_ratio"] * 100 for d in data]
    delays = [d["delay"] for d in data]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")

    for ax in axs.flat:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#585b70")

    # 1. Returns
    axs[0, 0].plot(episodes, task_returns, color="#89b4fa", lw=2, marker="o", ms=4, label="Task Return")
    axs[0, 0].plot(episodes, adj_returns, color="#f38ba8", lw=2, linestyle="--", label="Comm-Adjusted Return")
    axs[0, 0].set_title("Phase 7 Returns vs Episode", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Return")
    axs[0, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 2. Average Delay
    axs[0, 1].plot(episodes, delays, color="#fab387", lw=2, marker="s", ms=4, label="Average Delay (s)")
    axs[0, 1].set_title("Average Intersection Delay vs Episode", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Seconds / vehicle")
    axs[0, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 3. Communication Ratio
    axs[1, 0].plot(episodes, comm_ratios, color="#a6e3a1", lw=2, marker="^", ms=4, label="Communication Ratio (%)")
    axs[1, 0].axhline(100.0, color="#585b70", linestyle=":", label="100% (No Gating)")
    axs[1, 0].set_title("Inter-Agent Communication Ratio (%) vs Episode", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Actual / Possible (%)")
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 4. Total Communication Cost
    comm_costs = [d["comm_cost"] for d in data]
    axs[1, 1].plot(episodes, comm_costs, color="#cba6f7", lw=2, marker="d", ms=4, label="Comm Cost Penalty")
    axs[1, 1].set_title("Episode Communication Cost Penalty vs Episode", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel(r"$\lambda_{\mathrm{comm}} \times \mathrm{CommCost}$")
    axs[1, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train()
