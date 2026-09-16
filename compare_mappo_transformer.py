"""
Fair Comparative Evaluation: Frozen Phase 4 MAPPO Baseline vs. Phase 5 MAPPO + Temporal Transformer.
Evaluates both controllers across seeds 1001-1010 under identical simulation conditions:
- Same 2x2 grid network
- Same normal demand configuration & dynamic route generation
- Same 100 decision steps (500 SUMO simulation seconds per episode)
- Same TraCI environment and metric definitions
- Frozen Phase 4 MAPPO uses models/checkpoints/mappo_actor_shared.pt
- Phase 5 MAPPO+Transformer uses models/checkpoints/mappo_transformer_actor.pt
- Deterministic greedy action selection (argmax)
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
from models.transformer_network import MAPPOTemporalController


def evaluate_mappo_baseline(env, seeds, steps_per_episode, mappo_ctrl):
    results = []
    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            actions, _ = mappo_ctrl.get_actions(obs, deterministic=True)
            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                break

        results.append({
            "episode": ep_idx,
            "seed": seed,
            "controller": "MAPPO (Frozen Baseline)",
            "return_A": ep_rewards["A"],
            "return_B": ep_rewards["B"],
            "return_C": ep_rewards["C"],
            "return_D": ep_rewards["D"],
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_waiting": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        })
    return results


def evaluate_transformer_model(env, seeds, steps_per_episode, tf_ctrl):
    results = []
    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        tf_ctrl.reset_history(obs)

        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            actions, _ = tf_ctrl.get_actions(deterministic=True)
            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            tf_ctrl.update_history(next_obs)
            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                break

        results.append({
            "episode": ep_idx,
            "seed": seed,
            "controller": "MAPPO + Transformer",
            "return_A": ep_rewards["A"],
            "return_B": ep_rewards["B"],
            "return_C": ep_rewards["C"],
            "return_D": ep_rewards["D"],
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_waiting": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        })
    return results


def plot_comparison(mappo_res, tf_res, output_path):
    controllers = ["MAPPO (Frozen Baseline)", "MAPPO + Transformer"]
    colors = ["#89b4fa", "#fab387"]

    metrics = [
        ("mean_waiting", "Average Delay (s / veh)", [r["mean_waiting"] for r in mappo_res], [r["mean_waiting"] for r in tf_res]),
        ("mean_queue", "Average Queue Length (veh)", [r["mean_queue"] for r in mappo_res], [r["mean_queue"] for r in tf_res]),
        ("throughput", "Throughput (Arrived Vehicles)", [r["throughput"] for r in mappo_res], [r["throughput"] for r in tf_res]),
        ("agg_return", "Aggregate Return", [r["agg_return"] for r in mappo_res], [r["agg_return"] for r in tf_res])
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")

    for ax, (key, title, m_data, tf_data) in zip(axs.flat, metrics):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#585b70", axis="y")

        means = [np.mean(m_data), np.mean(tf_data)]
        stds = [np.std(m_data), np.std(tf_data)]
        x_pos = np.arange(len(controllers))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="#cdd6f4", lw=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(controllers, fontsize=11, fontweight="bold", color="#cdd6f4")
        ax.set_title(title, fontsize=12, fontweight="bold")

        for bar, m, s in zip(bars, means, stds):
            h = bar.get_height()
            y_text = h + (s if h >= 0 else -s - abs(h)*0.08)
            va = "bottom" if h >= 0 else "top"
            ax.annotate(f"{m:.2f} ± {s:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, y_text),
                        xytext=(0, 3 if h >= 0 else -8),
                        textcoords="offset points",
                        ha="center", va=va,
                        color="#ffffff", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_episodes = 10
    steps_per_episode = int(config["transformer"].get("steps_per_episode", 100))
    eval_seeds = [1000 + i for i in range(1, eval_episodes + 1)]

    csv_file = "experiments/evaluation_comparison_transformer.csv"
    plot_file = "experiments/controller_comparison_transformer.png"

    print("=" * 95)
    print("SAGE-TRAFFIC PHASE 5: HEAD-TO-HEAD COMPARATIVE EVALUATION")
    print("Frozen Phase 4 MAPPO Baseline vs. Phase 5 MAPPO + Temporal Transformer")
    print("=" * 95)
    print(f"Evaluation Seeds:        {eval_seeds} (10 Paired Seeds)")
    print(f"Steps per Episode:       {steps_per_episode} (500s SUMO simulation time per episode)")
    print(f"MAPPO Checkpoint:        models/checkpoints/mappo_actor_shared.pt")
    print(f"Transformer Checkpoint:  models/checkpoints/mappo_transformer_actor.pt")
    print("-" * 95)

    env = MultiAgentTrafficEnv(config_path=config_path)

    # 1. Load frozen MAPPO baseline
    mappo_ctrl = MAPPOController(config_path=config_path)
    mappo_ctrl.load_checkpoints("models/checkpoints")
    print("[1/2] Frozen Phase 4 MAPPO baseline checkpoint loaded successfully.")

    # 2. Load MAPPO + Temporal Transformer model
    tf_ctrl = MAPPOTemporalController(config_path=config_path)
    tf_ctrl.load_checkpoints("models/checkpoints")
    print("[2/2] Phase 5 MAPPO + Temporal Transformer checkpoint loaded successfully.")

    try:
        # Run evaluations
        print("\n[EVAL 1/2] Evaluating Frozen MAPPO Baseline (Seeds 1001-1010)...")
        mappo_results = evaluate_mappo_baseline(env, eval_seeds, steps_per_episode, mappo_ctrl)
        print("  Frozen MAPPO baseline evaluation completed.")

        print("\n[EVAL 2/2] Evaluating MAPPO + Temporal Transformer (Seeds 1001-1010)...")
        tf_results = evaluate_transformer_model(env, eval_seeds, steps_per_episode, tf_ctrl)
        print("  MAPPO + Temporal Transformer evaluation completed.")

        # Save to CSV
        os.makedirs("experiments", exist_ok=True)
        fieldnames = [
            "seed", "controller", "return_A", "return_B", "return_C", "return_D",
            "agg_return", "mean_queue", "mean_waiting", "throughput"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in mappo_results + tf_results:
                writer.writerow({
                    "seed": r["seed"],
                    "controller": r["controller"],
                    "return_A": f"{r['return_A']:.2f}",
                    "return_B": f"{r['return_B']:.2f}",
                    "return_C": f"{r['return_C']:.2f}",
                    "return_D": f"{r['return_D']:.2f}",
                    "agg_return": f"{r['agg_return']:.2f}",
                    "mean_queue": f"{r['mean_queue']:.2f}",
                    "mean_waiting": f"{r['mean_waiting']:.2f}",
                    "throughput": r["throughput"]
                })
        print(f"\nSaved all 20 evaluation records to {csv_file}")

        # Per-seed comparison table
        print("\n" + "=" * 95)
        print(f"{'Seed':<6} | {'Metric':<14} | {'MAPPO (Baseline)':<22} | {'MAPPO + Transformer':<22}")
        print("-" * 95)
        for i, s in enumerate(eval_seeds):
            print(f"{s:<6} | {'Delay (s)':<14} | {mappo_results[i]['mean_waiting']:<22.2f} | {tf_results[i]['mean_waiting']:<22.2f}")
            print(f"{'':<6} | {'Queue (veh)':<14} | {mappo_results[i]['mean_queue']:<22.2f} | {tf_results[i]['mean_queue']:<22.2f}")
            print(f"{'':<6} | {'Throughput':<14} | {mappo_results[i]['throughput']:<22} | {tf_results[i]['throughput']:<22}")
            print(f"{'':<6} | {'Agg Return':<14} | {mappo_results[i]['agg_return']:<22.2f} | {tf_results[i]['agg_return']:<22.2f}")
            if i < len(eval_seeds) - 1:
                print("-" * 95)
        print("=" * 95)

        # Summary statistics
        m_waits = [r["mean_waiting"] for r in mappo_results]
        m_queues = [r["mean_queue"] for r in mappo_results]
        m_tps = [r["throughput"] for r in mappo_results]
        m_rets = [r["agg_return"] for r in mappo_results]

        tf_waits = [r["mean_waiting"] for r in tf_results]
        tf_queues = [r["mean_queue"] for r in tf_results]
        tf_tps = [r["throughput"] for r in tf_results]
        tf_rets = [r["agg_return"] for r in tf_results]

        def stats(arr):
            return np.mean(arr), np.std(arr)

        def pct(new, old):
            return ((new - old) / abs(old)) * 100.0

        print("\n" + "=" * 95)
        print("HEAD-TO-HEAD SUMMARY STATISTICS (Mean +/- Std Dev across 10 Seeds)")
        print("=" * 95)
        print(f"{'Metric':<24} | {'MAPPO (Baseline)':<24} | {'MAPPO + Transformer':<24} | {'Change (%)':<12}")
        print("-" * 95)
        print(f"{'Average Delay (s)':<24} | {stats(m_waits)[0]:6.2f} +/- {stats(m_waits)[1]:<14.2f} | {stats(tf_waits)[0]:6.2f} +/- {stats(tf_waits)[1]:<14.2f} | {pct(stats(tf_waits)[0], stats(m_waits)[0]):+6.2f}%")
        print(f"{'Average Queue (veh)':<24} | {stats(m_queues)[0]:6.2f} +/- {stats(m_queues)[1]:<14.2f} | {stats(tf_queues)[0]:6.2f} +/- {stats(tf_queues)[1]:<14.2f} | {pct(stats(tf_queues)[0], stats(m_queues)[0]):+6.2f}%")
        print(f"{'Throughput (veh)':<24} | {stats(m_tps)[0]:6.1f} +/- {stats(m_tps)[1]:<14.1f} | {stats(tf_tps)[0]:6.1f} +/- {stats(tf_tps)[1]:<14.1f} | {pct(stats(tf_tps)[0], stats(m_tps)[0]):+6.2f}%")
        print(f"{'Aggregate Return':<24} | {stats(m_rets)[0]:6.2f} +/- {stats(m_rets)[1]:<14.2f} | {stats(tf_rets)[0]:6.2f} +/- {stats(tf_rets)[1]:<14.2f} | {pct(stats(tf_rets)[0], stats(m_rets)[0]):+6.2f}%")
        print("-" * 95)

        # Scientific Verdict
        delay_diff = pct(stats(tf_waits)[0], stats(m_waits)[0])
        queue_diff = pct(stats(tf_queues)[0], stats(m_queues)[0])

        print("\nHONEST SCIENTIFIC CONCLUSION:")
        if delay_diff < -2.0 and queue_diff < -2.0:
            print("  TEMPORAL TRANSFORMER IMPROVED PERFORMANCE")
        elif abs(delay_diff) <= 2.0 and abs(queue_diff) <= 2.0:
            print("  TEMPORAL TRANSFORMER PERFORMED COMPARABLY")
        else:
            print("  TEMPORAL TRANSFORMER DEGRADED PERFORMANCE")

        # Plot
        print(f"\nGenerating comparison plot -> {plot_file}...")
        plot_comparison(mappo_results, tf_results, plot_file)
        print("  Plot saved successfully.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


if __name__ == "__main__":
    main()
