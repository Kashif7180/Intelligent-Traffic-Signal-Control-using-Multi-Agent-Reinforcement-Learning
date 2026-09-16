"""
Phase 6 Training Script: MAPPO + Temporal Transformer + Graph Attention Network (GAT).

Trains the full end-to-end architecture on SAGE-Traffic:
- Temporal Transformer Encoder (43-D -> 64-D)
- Multi-head Graph Attention Network (GAT) layer over 4 intersections
- Decentralized GAT Actor (68-D -> 4 actions)
- Centralized GAT Critic (256-D -> 1 scalar value V(S_gat))

Conditions match previous phases:
- 40 episodes, 100 decision steps per episode (500s SUMO simulation time per episode)
- Seeds 43 to 82
- Normal demand profile
- Logs to experiments/mappo_transformer_gat_training_log.csv
- Checkpoints saved to models/checkpoints/mappo_transformer_gat_*.pt
- Real training plot saved to experiments/mappo_transformer_gat_reward_curve.png
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
from models.gat_network import MAPPOGATController


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    gat_cfg = config["gat"]
    tf_cfg = config["transformer"]
    sim_cfg = config["simulation"]
    agent_cfg = config["agent"]

    train_episodes = int(gat_cfg.get("train_episodes", 40))
    steps_per_episode = int(gat_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_transformer_gat_training_log.csv"
    plot_file = "experiments/mappo_transformer_gat_reward_curve.png"

    print("=" * 100)
    print("SAGE-Traffic Phase 6: MAPPO + Temporal Transformer + GAT Training")
    print("=" * 100)
    print("Architecture:            CTDE (Decentralized Actors, Centralized Critic)")
    print(f"Temporal History:        k={tf_cfg.get('history_length', 4)} timesteps (repeated initial obs)")
    print(f"Temporal Transformer:    Encoder (2 layers, 4 heads, d_model=64)")
    print(f"Graph Attention (GAT):   4 heads, 64-D, Physical Adjacency + Self-Loops, concat_heads=False")
    print(f"Decentralized Actor:     64-D GAT node embedding + 4-D agent ID = 68-D -> 4 action logits")
    print(f"Centralized Critic:      4 x 64-D GAT node embeddings = 256-D -> 1 scalar V(S_gat)")
    print(f"Intersections:           {agent_cfg['agents']} (4 Signalized Intersections)")
    print(f"Training Episodes:       {train_episodes}")
    print(f"Steps per Episode:       {steps_per_episode} (500s SUMO simulation time)")
    print(f"Total Env Steps:         {train_episodes * steps_per_episode:,}")
    print(f"Total Transitions:       {train_episodes * steps_per_episode * 4:,}")
    print(f"Actor LR / Critic LR:    {gat_cfg['lr_actor']} / {gat_cfg['lr_critic']}")
    print(f"Discount (gamma):        {gat_cfg['gamma']}")
    print(f"GAE Lambda:              {gat_cfg['gae_lambda']}")
    print(f"Clip Ratio:              {gat_cfg['clip_ratio']}")
    print(f"Minibatch Size:          {gat_cfg['minibatch_size']}")
    print(f"PPO Epochs:              {gat_cfg['epochs']}")
    print("-" * 100)

    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOGATController(config_path=config_path)

    # CSV Header
    fieldnames = [
        "episode",
        "seed",
        "aggregate_return",
        "mean_delay",
        "mean_queue",
        "throughput",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "return_A",
        "return_B",
        "return_C",
        "return_D"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"{'Ep':<4} | {'Seed':<5} | {'Agg Return':<12} | {'A / B / C / D Returns':<32} | {'Mean Q':<8} | {'Mean Delay':<10} | {'Throughput':<10}")
    print("-" * 100)

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)

            # Initialize temporal history buffer
            ctrl.reset_history(obs)

            ep_rewards = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0

            for step in range(steps_per_episode):
                # 1. Snapshot current history window for buffer storage
                histories = ctrl.history_manager.get_all_histories()

                # 2. Action Selection via GAT message passing + Decentralized Actors
                actions, log_probs = ctrl.get_actions(deterministic=False)

                # 3. Centralized Critic Evaluation (training only)
                value = ctrl.get_centralized_value()

                # 4. Environment Step
                next_obs, rewards, terminations, truncations, next_info = env.step(actions)

                # 5. Store in CTDE Rollout Buffer
                done = any(terminations.values()) or any(truncations.values())
                ctrl.buffer.add(
                    histories=histories,
                    value=value,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    done=done
                )

                for a in env.agents:
                    ep_rewards[a] += rewards[a]

                # Step metric tracking
                step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
                step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
                ep_queues.append(step_q)
                ep_waiting.append(step_w)
                ep_throughput += next_info["A"]["arrived_vehicles"]

                # 6. Slide history window forward
                ctrl.update_history(next_obs)
                obs = next_obs

                if done:
                    break

            # 7. End-to-end PPO update at end of rollout
            last_value = ctrl.get_centralized_value()
            loss_stats = ctrl.update(last_value)

            agg_ret = sum(ep_rewards.values())
            m_queue = float(np.mean(ep_queues))
            m_delay = float(np.mean(ep_waiting))

            # Write to CSV
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "episode": ep,
                    "seed": ep_seed,
                    "aggregate_return": f"{agg_ret:.2f}",
                    "mean_delay": f"{m_delay:.2f}",
                    "mean_queue": f"{m_queue:.2f}",
                    "throughput": ep_throughput,
                    "policy_loss": f"{loss_stats['policy_loss']:.4f}",
                    "value_loss": f"{loss_stats['value_loss']:.4f}",
                    "entropy": f"{loss_stats['entropy']:.4f}",
                    "approx_kl": f"{loss_stats['approx_kl']:.6f}",
                    "return_A": f"{ep_rewards['A']:.2f}",
                    "return_B": f"{ep_rewards['B']:.2f}",
                    "return_C": f"{ep_rewards['C']:.2f}",
                    "return_D": f"{ep_rewards['D']:.2f}"
                })

            ret_str = f"{ep_rewards['A']:6.1f} / {ep_rewards['B']:6.1f} / {ep_rewards['C']:6.1f} / {ep_rewards['D']:6.1f}"
            print(f"{ep:<4} | {ep_seed:<5} | {agg_ret:<12.1f} | {ret_str:<32} | {m_queue:<8.2f} | {m_delay:<10.2f} | {ep_throughput:<10}")

        print("-" * 100)
        print("\nTraining complete! Saving checkpoints...")

        # Save MAPPO+Transformer+GAT checkpoints
        ctrl.save_checkpoints("models/checkpoints")
        print("  Checkpoints successfully saved:")
        print("    -> models/checkpoints/mappo_transformer_gat_actor.pt")
        print("    -> models/checkpoints/mappo_transformer_gat_critic.pt")

        # Generate training plot
        print(f"\nGenerating real reward curve plot -> {plot_file}...")
        generate_plots(csv_file, plot_file)
        print("  Plot successfully saved.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


def generate_plots(csv_path: str, plot_path: str):
    """Generate training curves dynamically from real MAPPO+Transformer+GAT training log CSV."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "episode": int(row["episode"]),
                "agg_return": float(row["aggregate_return"]),
                "mean_queue": float(row["mean_queue"]),
                "mean_delay": float(row["mean_delay"]),
                "value_loss": float(row["value_loss"]),
                "policy_loss": float(row["policy_loss"])
            })

    episodes = [d["episode"] for d in data]
    agg_returns = [d["agg_return"] for d in data]
    mean_queues = [d["mean_queue"] for d in data]
    mean_delays = [d["mean_delay"] for d in data]
    val_losses = [d["value_loss"] for d in data]

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

    # 1. Aggregate Return
    axs[0, 0].plot(episodes, agg_returns, color="#89b4fa", lw=2, marker="o", ms=4, label="Aggregate Return")
    axs[0, 0].set_title("MAPPO + Temporal + GAT Aggregate Return vs Episode", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Sum of Rewards")
    axs[0, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 2. Average Delay
    axs[0, 1].plot(episodes, mean_delays, color="#f38ba8", lw=2, marker="s", ms=4, label="Mean Delay (s)")
    axs[0, 1].set_title("Average Intersection Delay vs Episode", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Seconds / vehicle")
    axs[0, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 3. Average Queue Length
    axs[1, 0].plot(episodes, mean_queues, color="#fab387", lw=2, marker="^", ms=4, label="Mean Queue Length")
    axs[1, 0].set_title("Average Queue Length vs Episode", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Vehicles")
    axs[1, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 4. Centralized Critic Value Loss
    axs[1, 1].plot(episodes, val_losses, color="#a6e3a1", lw=2, marker="d", ms=4, label="Critic Value Loss")
    axs[1, 1].set_title("Centralized GAT Critic Value Loss vs Episode", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("MSE Loss")
    axs[1, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train()
