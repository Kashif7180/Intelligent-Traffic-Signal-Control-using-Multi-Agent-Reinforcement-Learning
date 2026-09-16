"""
Training script for Independent PPO (IPPO) on SAGE-Traffic (Phase 3).
Trains 4 completely independent PPO agents (one per intersection) for 40 episodes.
Logs metrics to experiments/training_log.csv, saves models to models/checkpoints/,
and generates experiments/ppo_reward_curve.png from real training output.
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
from models.ppo_network import IndependentPPOAgent


def train():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ppo_cfg = config["ppo"]
    sim_cfg = config["simulation"]
    agent_cfg = config["agent"]

    train_episodes = int(ppo_cfg.get("train_episodes", 40))
    steps_per_episode = int(ppo_cfg.get("steps_per_episode", 100))
    base_seed = int(sim_cfg.get("seed", 42))

    # Directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    csv_file = "experiments/training_log.csv"
    plot_file = "experiments/ppo_reward_curve.png"

    print("=" * 80)
    print("SAGE-Traffic Phase 3: Independent PPO (IPPO) Baseline Training")
    print("=" * 80)
    print(f"Algorithm:           Independent PPO (IPPO) - Strictly Independent Learners")
    print(f"Intersections:       {agent_cfg['agents']} (4 Separate Agents)")
    print(f"Training Episodes:   {train_episodes}")
    print(f"Steps per Episode:   {steps_per_episode} (500s SUMO simulation time)")
    print(f"Total Env Steps:     {train_episodes * steps_per_episode:,}")
    print(f"Total Transitions:   {train_episodes * steps_per_episode * 4:,}")
    print(f"Learning Rate:       {ppo_cfg['lr']}")
    print(f"Discount (gamma):    {ppo_cfg['gamma']}")
    print(f"GAE Lambda:          {ppo_cfg['gae_lambda']}")
    print(f"Clip Ratio:          {ppo_cfg['clip_ratio']}")
    print(f"Minibatch Size:      {ppo_cfg['minibatch_size']}")
    print(f"PPO Epochs:          {ppo_cfg['epochs']}")
    print("-" * 80)

    # Initialize environment
    env = MultiAgentTrafficEnv(config_path=config_path)

    # Initialize 4 completely separate PPO agents
    obs_dim = env.observation_space("A").shape[0]
    act_dim = env.action_space("A").n
    agents = {
        agent_id: IndependentPPOAgent(
            agent_id=agent_id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            ppo_cfg=ppo_cfg
        )
        for agent_id in env.agents
    }

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
        "mean_policy_loss",
        "mean_value_loss",
        "mean_entropy",
        "seed"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    history = {
        "episode": [],
        "return_A": [],
        "return_B": [],
        "return_C": [],
        "return_D": [],
        "aggregate_return": [],
        "mean_queue": [],
        "mean_waiting_time": [],
        "throughput": []
    }

    print(f"{'Ep':<4} | {'Agg Return':<12} | {'A / B / C / D Returns':<32} | {'Mean Q':<8} | {'Mean Wait':<10} | {'Throughput':<10}")
    print("-" * 86)

    try:
        for ep in range(1, train_episodes + 1):
            ep_seed = base_seed + ep
            obs, info = env.reset(seed=ep_seed)

            ep_rewards = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0

            for step in range(steps_per_episode):
                actions = {}
                log_probs = {}
                values = {}

                # Independent action selection per agent
                for a in env.agents:
                    act, lp, val = agents[a].get_action_and_value(obs[a])
                    actions[a] = act
                    log_probs[a] = lp
                    values[a] = val

                next_obs, rewards, terminations, truncations, next_info = env.step(actions)

                # Store transitions in independent rollout buffers
                for a in env.agents:
                    done = terminations[a] or truncations[a]
                    agents[a].buffer.add(
                        state=obs[a],
                        action=actions[a],
                        log_prob=log_probs[a],
                        reward=rewards[a],
                        value=values[a],
                        done=done
                    )
                    ep_rewards[a] += rewards[a]

                # Step metric tracking
                step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
                step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
                ep_queues.append(step_q)
                ep_waiting.append(step_w)
                ep_throughput += next_info["A"]["arrived_vehicles"]

                obs = next_obs
                if all(terminations.values()) or all(truncations.values()):
                    break

            # PPO Policy Update after rollout
            pol_losses = []
            val_losses = []
            entropies = []
            for a in env.agents:
                last_val = agents[a].get_value(obs[a])
                loss_stats = agents[a].update(last_val)
                pol_losses.append(loss_stats["policy_loss"])
                val_losses.append(loss_stats["value_loss"])
                entropies.append(loss_stats["entropy"])

            agg_ret = sum(ep_rewards.values())
            m_queue = float(np.mean(ep_queues))
            m_wait = float(np.mean(ep_waiting))
            m_pol_loss = float(np.mean(pol_losses))
            m_val_loss = float(np.mean(val_losses))
            m_ent = float(np.mean(entropies))

            # Record history
            history["episode"].append(ep)
            history["return_A"].append(ep_rewards["A"])
            history["return_B"].append(ep_rewards["B"])
            history["return_C"].append(ep_rewards["C"])
            history["return_D"].append(ep_rewards["D"])
            history["aggregate_return"].append(agg_ret)
            history["mean_queue"].append(m_queue)
            history["mean_waiting_time"].append(m_wait)
            history["throughput"].append(ep_throughput)

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
                    "mean_policy_loss": f"{m_pol_loss:.4f}",
                    "mean_value_loss": f"{m_val_loss:.4f}",
                    "mean_entropy": f"{m_ent:.4f}",
                    "seed": ep_seed
                })

            ret_str = f"{ep_rewards['A']:6.1f} / {ep_rewards['B']:6.1f} / {ep_rewards['C']:6.1f} / {ep_rewards['D']:6.1f}"
            print(f"{ep:<4} | {agg_ret:<12.1f} | {ret_str:<32} | {m_queue:<8.2f} | {m_wait:<10.2f} | {ep_throughput:<10}")

        print("-" * 86)
        print("\nTraining complete! Saving checkpoints...")

        # Save model checkpoints
        for a in env.agents:
            ckpt_path = f"models/checkpoints/agent_{a}.pt"
            agents[a].save_checkpoint(ckpt_path)
            print(f"  Saved checkpoint for Agent {a} -> {ckpt_path}")

        # Generate real training plot
        print(f"\nGenerating real reward curve plot -> {plot_file}...")
        generate_plots(csv_file, plot_file)
        print("  Plot successfully saved.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


def generate_plots(csv_path: str, plot_path: str):
    """Generate training curve figure directly from real training log CSV data."""
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
                "throughput": int(row["throughput"])
            })

    episodes = [d["episode"] for d in data]
    agg_returns = [d["agg_return"] for d in data]
    mean_queues = [d["mean_queue"] for d in data]
    mean_waits = [d["mean_wait"] for d in data]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle("SAGE-Traffic Phase 3: Independent PPO (IPPO) Real Training Curves (40 Episodes)", fontsize=14, fontweight="bold")

    # 1. Aggregate Return
    axs[0, 0].plot(episodes, agg_returns, color="#1f77b4", alpha=0.4, label="Raw Aggregate Return")
    if len(agg_returns) >= 5:
        # 5-episode moving average
        kernel = np.ones(5) / 5
        smoothed = np.convolve(agg_returns, kernel, mode="valid")
        axs[0, 0].plot(episodes[4:], smoothed, color="#1f77b4", linewidth=2.5, label="5-Ep Moving Avg")
    axs[0, 0].set_title("Aggregate Episode Return (Sum of 4 Agents)")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Return (Negative Delay + Queue)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. Per-Agent Returns
    axs[0, 1].plot(episodes, [d["return_A"] for d in data], label="Agent A", color="#d62728", alpha=0.8)
    axs[0, 1].plot(episodes, [d["return_B"] for d in data], label="Agent B", color="#2ca02c", alpha=0.8)
    axs[0, 1].plot(episodes, [d["return_C"] for d in data], label="Agent C", color="#9467bd", alpha=0.8)
    axs[0, 1].plot(episodes, [d["return_D"] for d in data], label="Agent D", color="#ff7f0e", alpha=0.8)
    axs[0, 1].set_title("Per-Agent Returns (Local Intersections)")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Return")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. Mean Waiting Time
    axs[1, 0].plot(episodes, mean_waits, color="#e377c2", alpha=0.4, label="Raw Mean Wait")
    if len(mean_waits) >= 5:
        smoothed_wait = np.convolve(mean_waits, np.ones(5)/5, mode="valid")
        axs[1, 0].plot(episodes[4:], smoothed_wait, color="#e377c2", linewidth=2.5, label="5-Ep Moving Avg")
    axs[1, 0].set_title("Average Waiting Time per Intersection (s)")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Waiting Time (s)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Mean Queue Length & Throughput
    ax4 = axs[1, 1]
    ax4_twin = ax4.twinx()
    p1 = ax4.plot(episodes, mean_queues, color="#17becf", linewidth=2, label="Mean Queue (veh)")
    p2 = ax4_twin.plot(episodes, [d["throughput"] for d in data], color="#8c564b", linestyle="--", label="Throughput (veh)")
    ax4.set_title("Mean Queue Length & Completed Vehicle Throughput")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Queue Length (veh)", color="#17becf")
    ax4_twin.set_ylabel("Throughput (veh)", color="#8c564b")
    ax4.grid(True, alpha=0.3)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    train()
