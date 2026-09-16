"""
Phase 5 Training Script: MAPPO + Temporal Transformer on SAGE-Traffic.
Trains a genuine PyTorch Temporal Transformer Encoder end-to-end with MAPPO (CTDE).
Conditions match Phase 4 exactly:
- 40 episodes, 100 decision steps per episode (500s SUMO simulation time per episode)
- Seeds 43 to 82
- Normal demand profile
- Logs to experiments/mappo_transformer_training_log.csv
- Checkpoints saved to models/checkpoints/mappo_transformer_*.pt
- Real training plot saved to experiments/mappo_transformer_reward_curve.png
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
from models.transformer_network import MAPPOTemporalController


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tf_cfg = config["transformer"]
    sim_cfg = config["simulation"]
    agent_cfg = config["agent"]

    train_episodes = int(tf_cfg.get("train_episodes", 40))
    steps_per_episode = int(tf_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_transformer_training_log.csv"
    plot_file = "experiments/mappo_transformer_reward_curve.png"

    print("=" * 95)
    print("SAGE-Traffic Phase 5: MAPPO + Temporal Transformer Training")
    print("=" * 95)
    print(f"Architecture:            Centralized Training, Decentralized Execution (CTDE)")
    print(f"Temporal Encoder:        PyTorch TransformerEncoder (2 layers, 4 heads, d_model=64)")
    print(f"History Window:          k={tf_cfg['history_length']} timesteps (Initial repeated [o0,o0,o0,o0])")
    print(f"Decentralized Actor:     64-D temporal embedding + 4-D agent ID = 68-D -> 4 action logits")
    print(f"Centralized Critic:      4 x 64-D temporal embeddings = 256-D -> 1 scalar value V(S_temporal)")
    print(f"Intersections:           {agent_cfg['agents']} (4 Signalized Intersections)")
    print(f"Training Episodes:       {train_episodes}")
    print(f"Steps per Episode:       {steps_per_episode} (500s SUMO simulation time)")
    print(f"Total Env Steps:         {train_episodes * steps_per_episode:,}")
    print(f"Total Transitions:       {train_episodes * steps_per_episode * 4:,}")
    print(f"Actor LR / Critic LR:    {tf_cfg['lr_actor']} / {tf_cfg['lr_critic']}")
    print(f"Discount (gamma):        {tf_cfg['gamma']}")
    print(f"GAE Lambda:              {tf_cfg['gae_lambda']}")
    print(f"Clip Ratio:              {tf_cfg['clip_ratio']}")
    print(f"Minibatch Size:          {tf_cfg['minibatch_size']}")
    print(f"PPO Epochs:              {tf_cfg['epochs']}")
    print("-" * 95)

    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOTemporalController(config_path=config_path)

    # CSV Header
    fieldnames = [
        "episode",
        "return_A",
        "return_B",
        "return_C",
        "return_D",
        "aggregate_return",
        "mean_queue",
        "mean_waiting_time",
        "throughput",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "seed"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"{'Ep':<4} | {'Agg Return':<12} | {'A / B / C / D Returns':<32} | {'Mean Q':<8} | {'Mean Wait':<10} | {'Throughput':<10}")
    print("-" * 95)

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)

            # Initialize history by repeating initial observation k times
            ctrl.reset_history(obs)

            ep_rewards = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0

            for step in range(steps_per_episode):
                # 1. Snapshot current history window for buffer storage
                histories = ctrl.history_manager.get_all_histories()

                # 2. Decentralized Action Selection (local history + agent ID only)
                actions, log_probs = ctrl.get_actions(deterministic=False)

                # 3. Centralized Critic Evaluation (all 4 agents' temporal histories concatenated to 256-D)
                value = ctrl.get_centralized_value()

                # 4. Environment Step
                next_obs, rewards, terminations, truncations, next_info = env.step(actions)

                # 5. Store in CTDE Temporal Rollout Buffer
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

                # 6. Slide history window forward with newly received observation
                ctrl.update_history(next_obs)
                obs = next_obs

                if done:
                    break

            # 7. End-to-end PPO Update at end of rollout
            last_value = ctrl.get_centralized_value()
            loss_stats = ctrl.update(last_value)

            agg_ret = sum(ep_rewards.values())
            m_queue = float(np.mean(ep_queues))
            m_wait = float(np.mean(ep_waiting))

            # Write to CSV
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({
                    "episode": ep,
                    "return_A": f"{ep_rewards['A']:.2f}",
                    "return_B": f"{ep_rewards['B']:.2f}",
                    "return_C": f"{ep_rewards['C']:.2f}",
                    "return_D": f"{ep_rewards['D']:.2f}",
                    "aggregate_return": f"{agg_ret:.2f}",
                    "mean_queue": f"{m_queue:.2f}",
                    "mean_waiting_time": f"{m_wait:.2f}",
                    "throughput": ep_throughput,
                    "policy_loss": f"{loss_stats['policy_loss']:.4f}",
                    "value_loss": f"{loss_stats['value_loss']:.4f}",
                    "entropy": f"{loss_stats['entropy']:.4f}",
                    "approx_kl": f"{loss_stats['approx_kl']:.6f}",
                    "seed": ep_seed
                })

            ret_str = f"{ep_rewards['A']:6.1f} / {ep_rewards['B']:6.1f} / {ep_rewards['C']:6.1f} / {ep_rewards['D']:6.1f}"
            print(f"{ep:<4} | {agg_ret:<12.1f} | {ret_str:<32} | {m_queue:<8.2f} | {m_wait:<10.2f} | {ep_throughput:<10}")

        print("-" * 95)
        print("\nTraining complete! Saving checkpoints...")

        # Save MAPPO+Transformer checkpoints
        ctrl.save_checkpoints("models/checkpoints")
        print("  Checkpoints successfully saved:")
        print("    -> models/checkpoints/mappo_transformer_actor.pt")
        print("    -> models/checkpoints/mappo_transformer_critic.pt")

        # Generate real training plot
        print(f"\nGenerating real reward curve plot -> {plot_file}...")
        generate_plots(csv_file, plot_file)
        print("  Plot successfully saved.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


def generate_plots(csv_path: str, plot_path: str):
    """Generate training curves dynamically from real MAPPO+Transformer training log CSV."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "episode": int(row["episode"]),
                "agg_return": float(row["aggregate_return"]),
                "mean_queue": float(row["mean_queue"]),
                "mean_wait": float(row["mean_waiting_time"]),
                "value_loss": float(row["value_loss"]),
                "policy_loss": float(row["policy_loss"])
            })

    episodes = [d["episode"] for d in data]
    agg_returns = [d["agg_return"] for d in data]
    mean_queues = [d["mean_queue"] for d in data]
    mean_waits = [d["mean_wait"] for d in data]
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
    axs[0, 0].set_title("MAPPO + Temporal Transformer Aggregate Return vs Episode", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Sum of Rewards")
    axs[0, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 2. Average Delay (Waiting Time)
    axs[0, 1].plot(episodes, mean_waits, color="#f38ba8", lw=2, marker="s", ms=4, label="Mean Waiting Time (s)")
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
    axs[1, 1].set_title("Centralized Temporal Critic Value Loss vs Episode", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("MSE Loss")
    axs[1, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train()
