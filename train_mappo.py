"""
Training Script for Multi-Agent PPO (MAPPO - Phase 4) on SAGE-Traffic.
Centralized Training with Decentralized Execution (CTDE):
- 1 Centralized Critic (172 -> 128 -> 64 -> 1) trained on global state S_t
- Decentralized Actor(s) (47 -> 64 -> 64 -> 4) conditioned strictly on local observation
- 40 training episodes, 100 decision steps per episode (500s SUMO time per episode)
- Seeds 43-82 (matching Phase 3 training seeds)
- Logs to experiments/mappo_training_log.csv
- Saves checkpoints to models/checkpoints/mappo_*.pt
- Generates real curve plot to experiments/mappo_reward_curve.png
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
from models.mappo_network import MAPPOController


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mappo_cfg = config["mappo"]
    sim_cfg = config["simulation"]
    agent_cfg = config["agent"]

    train_episodes = int(mappo_cfg.get("train_episodes", 40))
    steps_per_episode = int(mappo_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/mappo_training_log.csv"
    plot_file = "experiments/mappo_reward_curve.png"

    print("=" * 85)
    print("SAGE-Traffic Phase 4: Multi-Agent PPO (MAPPO - CTDE) Training")
    print("=" * 85)
    print(f"Architecture:        Centralized Training with Decentralized Execution (CTDE)")
    print(f"Centralized Critic:  172 -> 128 -> 64 -> 1 (Consumes global state S_t)")
    print(f"Decentralized Actor: 47 -> 64 -> 64 -> 4 (Local obs 43 + 4-dim Agent ID)")
    print(f"Policy Sharing:      {mappo_cfg.get('share_policy', True)}")
    print(f"Intersections:       {agent_cfg['agents']} (4 Signalized Intersections)")
    print(f"Training Episodes:   {train_episodes}")
    print(f"Steps per Episode:   {steps_per_episode} (500s SUMO simulation time)")
    print(f"Total Env Steps:     {train_episodes * steps_per_episode:,}")
    print(f"Total Transitions:   {train_episodes * steps_per_episode * 4:,}")
    print(f"Actor LR / Critic LR:{mappo_cfg['lr_actor']} / {mappo_cfg['lr_critic']}")
    print(f"Discount (gamma):    {mappo_cfg['gamma']}")
    print(f"GAE Lambda:          {mappo_cfg['gae_lambda']}")
    print(f"Clip Ratio:          {mappo_cfg['clip_ratio']}")
    print(f"Minibatch Size:      {mappo_cfg['minibatch_size']}")
    print(f"PPO Epochs:          {mappo_cfg['epochs']}")
    print("-" * 85)

    # Initialize environment and controller
    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOController(config_path=config_path)

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
    print("-" * 88)

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)

            ep_rewards = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0

            for step in range(steps_per_episode):
                # 1. Decentralized Action Selection (local obs + id only)
                actions, log_probs = ctrl.get_actions(obs, deterministic=False)

                # 2. Centralized Critic Evaluation (global state S_t in R^172)
                global_state = ctrl.construct_global_state(obs)
                value = ctrl.get_value(global_state)

                # 3. Environment Step
                next_obs, rewards, terminations, truncations, next_info = env.step(actions)

                # 4. Store in CTDE Rollout Buffer
                done = any(terminations.values()) or any(truncations.values())
                ctrl.buffer.add(
                    global_state=global_state,
                    value=value,
                    local_obs=obs,
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

                obs = next_obs
                if done:
                    break

            # 5. MAPPO PPO Update at end of rollout
            last_global_state = ctrl.construct_global_state(obs)
            loss_stats = ctrl.update(last_global_state)

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

        print("-" * 88)
        print("\nTraining complete! Saving checkpoints...")

        # Save MAPPO checkpoints
        ctrl.save_checkpoints("models/checkpoints")
        print("  Checkpoints successfully saved:")
        print("    -> models/checkpoints/mappo_critic.pt")
        print("    -> models/checkpoints/mappo_actor_shared.pt")

        # Generate real training plot
        print(f"\nGenerating real reward curve plot -> {plot_file}...")
        generate_plots(csv_file, plot_file)
        print("  Plot successfully saved.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


def generate_plots(csv_path: str, plot_path: str):
    """Generate training curves directly from real MAPPO training log CSV."""
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "episode": int(row["episode"]),
                "agg_return": float(row["aggregate_return"]),
                "return_A": float(row["return_A"]),
                "return_B": float(row["return_B"]),
                "return_C": float(row["return_C"]),
                "return_D": float(row["return_D"]),
                "mean_queue": float(row["mean_queue"]),
                "mean_wait": float(row["mean_waiting_time"]),
                "throughput": int(row["throughput"]),
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
    axs[0, 0].set_title("MAPPO Aggregate Return vs Episode", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Sum of Rewards")
    axs[0, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 2. Average Delay (Waiting Time)
    axs[0, 1].plot(episodes, mean_waits, color="#f38ba8", lw=2, marker="s", ms=4, label="Mean Waiting Time (s)")
    axs[0, 1].set_title("MAPPO Average Intersection Delay vs Episode", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Seconds / vehicle")
    axs[0, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 3. Average Queue Length
    axs[1, 0].plot(episodes, mean_queues, color="#fab387", lw=2, marker="^", ms=4, label="Mean Queue Length")
    axs[1, 0].set_title("MAPPO Average Queue Length vs Episode", fontsize=12, fontweight="bold")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Vehicles")
    axs[1, 0].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    # 4. Centralized Critic Value Loss
    axs[1, 1].plot(episodes, val_losses, color="#a6e3a1", lw=2, marker="d", ms=4, label="Critic Value Loss")
    axs[1, 1].set_title("Centralized Critic Value Loss vs Episode", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("MSE Loss")
    axs[1, 1].legend(facecolor="#181825", edgecolor="#45475a", labelcolor="#cdd6f4")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train()
