"""
Phase 6 Comparative Evaluation:
Frozen Phase 5 (MAPPO + Temporal Transformer) vs. Phase 6 (MAPPO + Temporal Transformer + GAT).

Conditions:
- Seeds: 1001 to 1010 (10 evaluation seeds)
- 100 decision steps per episode (500s SUMO simulation time)
- Same 2x2 grid network and normal demand
- Deterministic action selection (argmax)
- Centralized critic is NEVER used during evaluation
- Frozen Phase 5 loads: models/checkpoints/mappo_transformer_actor.pt
- Phase 6 loads: models/checkpoints/mappo_transformer_gat_actor.pt
- Produces:
  experiments/evaluation_comparison_gat.csv
  experiments/controller_comparison_gat.png
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
from models.gat_network import MAPPOGATController


def evaluate_transformer_baseline(env, seeds, steps_per_episode, tf_ctrl):
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
            "controller": "Phase 5: MAPPO + Transformer",
            "return_A": ep_rewards["A"],
            "return_B": ep_rewards["B"],
            "return_C": ep_rewards["C"],
            "return_D": ep_rewards["D"],
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_delay": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        })
    return results


def evaluate_gat_model(env, seeds, steps_per_episode, gat_ctrl):
    results = []
    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        gat_ctrl.reset_history(obs)

        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            actions, _ = gat_ctrl.get_actions(deterministic=True)
            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            gat_ctrl.update_history(next_obs)
            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                break

        results.append({
            "episode": ep_idx,
            "seed": seed,
            "controller": "Phase 6: MAPPO + Transformer + GAT",
            "return_A": ep_rewards["A"],
            "return_B": ep_rewards["B"],
            "return_C": ep_rewards["C"],
            "return_D": ep_rewards["D"],
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_delay": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        })
    return results


def plot_comparison(tf_res, gat_res, output_path):
    controllers = ["Phase 5: Transformer", "Phase 6: Transformer + GAT"]
    colors = ["#fab387", "#cba6f7"]

    metrics = [
        ("mean_delay", "Average Delay (s / veh)", [r["mean_delay"] for r in tf_res], [r["mean_delay"] for r in gat_res], True),
        ("mean_queue", "Average Queue Length (veh)", [r["mean_queue"] for r in tf_res], [r["mean_queue"] for r in gat_res], True),
        ("throughput", "Throughput (Arrived Vehicles)", [r["throughput"] for r in tf_res], [r["throughput"] for r in gat_res], False),
        ("agg_return", "Aggregate Return", [r["agg_return"] for r in tf_res], [r["agg_return"] for r in gat_res], False)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")

    for ax, (key, title, tf_data, gat_data, lower_is_better) in zip(axs.flat, metrics):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#585b70", axis="y")

        means = [np.mean(tf_data), np.mean(gat_data)]
        stds = [np.std(tf_data), np.std(gat_data)]
        x_pos = np.arange(len(controllers))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="#cdd6f4", lw=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(controllers, fontsize=10, fontweight="bold", color="#cdd6f4")
        ax.set_title(title, fontsize=11, fontweight="bold")

        for bar, m, s in zip(bars, means, stds):
            h = bar.get_height()
            y_text = h + (s if h >= 0 else -s - abs(h)*0.08)
            va = "bottom" if h >= 0 else "top"
            ax.annotate(f"{m:.2f} ± {s:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, y_text),
                        xytext=(0, 3 if h >= 0 else -8),
                        textcoords="offset points",
                        ha="center", va=va,
                        fontsize=9, fontweight="bold", color="#ffffff")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_comparison():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_seeds = list(range(1001, 1011))
    steps_per_episode = 100

    csv_path = "experiments/evaluation_comparison_gat.csv"
    plot_path = "experiments/controller_comparison_gat.png"

    print("=" * 105)
    print("SAGE-Traffic Phase 6 Fair Comparative Evaluation")
    print("Frozen Phase 5 (MAPPO + Temporal Transformer) vs. Phase 6 (MAPPO + Temporal Transformer + GAT)")
    print("=" * 105)
    print(f"Evaluation Seeds:    {eval_seeds}")
    print(f"Steps per Episode:   {steps_per_episode} (500s SUMO simulation time)")
    print(f"Action Mode:         Deterministic Greedy (argmax)")
    print(f"Critic Status:       Training-Only (NOT used during evaluation)")
    print("-" * 105)

    env = MultiAgentTrafficEnv(config_path=config_path)

    # 1. Load Frozen Phase 5 Model
    print("\n[1/2] Loading Frozen Phase 5 MAPPO + Temporal Transformer Checkpoint...")
    tf_ctrl = MAPPOTemporalController(config_path=config_path)
    tf_ctrl.load_checkpoints("models/checkpoints")
    print("      Running Phase 5 evaluation on seeds 1001-1010...")
    tf_res = evaluate_transformer_baseline(env, eval_seeds, steps_per_episode, tf_ctrl)
    print("      Phase 5 evaluation completed.")

    # 2. Load Phase 6 Model
    print("\n[2/2] Loading Phase 6 MAPPO + Temporal Transformer + GAT Checkpoint...")
    gat_ctrl = MAPPOGATController(config_path=config_path)
    gat_ctrl.load_checkpoints("models/checkpoints")
    print("      Running Phase 6 evaluation on seeds 1001-1010...")
    gat_res = evaluate_gat_model(env, eval_seeds, steps_per_episode, gat_ctrl)
    print("      Phase 6 evaluation completed.")

    env.close()

    # Save to CSV
    fieldnames = [
        "seed", "episode", "controller",
        "return_A", "return_B", "return_C", "return_D",
        "agg_return", "mean_delay", "mean_queue", "throughput"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in tf_res:
            writer.writerow(r)
        for r in gat_res:
            writer.writerow(r)

    print(f"\nSaved detailed evaluation log -> {csv_path}")

    # Generate comparative plot
    plot_comparison(tf_res, gat_res, plot_path)
    print(f"Saved comparative bar chart -> {plot_path}")

    # Print Summary Table
    print("\n" + "=" * 115)
    print(f"{'Seed':<6} | {'Controller':<32} | {'Delay (s)':<12} | {'Queue (veh)':<12} | {'Throughput':<12} | {'Agg Return':<12}")
    print("-" * 115)

    for r_tf, r_gat in zip(tf_res, gat_res):
        print(f"{r_tf['seed']:<6} | {'Phase 5 (Transformer)':<32} | {r_tf['mean_delay']:<12.2f} | {r_tf['mean_queue']:<12.2f} | {r_tf['throughput']:<12} | {r_tf['agg_return']:<12.2f}")
        print(f"{r_gat['seed']:<6} | {'Phase 6 (Transformer + GAT)':<32} | {r_gat['mean_delay']:<12.2f} | {r_gat['mean_queue']:<12.2f} | {r_gat['throughput']:<12} | {r_gat['agg_return']:<12.2f}")
        print("-" * 115)

    # Compute Aggregate Stats
    tf_delay_m, tf_delay_s = np.mean([r["mean_delay"] for r in tf_res]), np.std([r["mean_delay"] for r in tf_res])
    gat_delay_m, gat_delay_s = np.mean([r["mean_delay"] for r in gat_res]), np.std([r["mean_delay"] for r in gat_res])

    tf_queue_m, tf_queue_s = np.mean([r["mean_queue"] for r in tf_res]), np.std([r["mean_queue"] for r in tf_res])
    gat_queue_m, gat_queue_s = np.mean([r["mean_queue"] for r in gat_res]), np.std([r["mean_queue"] for r in gat_res])

    tf_tp_m, tf_tp_s = np.mean([r["throughput"] for r in tf_res]), np.std([r["throughput"] for r in tf_res])
    gat_tp_m, gat_tp_s = np.mean([r["throughput"] for r in gat_res]), np.std([r["throughput"] for r in gat_res])

    tf_ret_m, tf_ret_s = np.mean([r["agg_return"] for r in tf_res]), np.std([r["agg_return"] for r in tf_res])
    gat_ret_m, gat_ret_s = np.mean([r["agg_return"] for r in gat_res]), np.std([r["agg_return"] for r in gat_res])

    # Percentage change relative to Phase 5
    def calc_pct(p6, p5):
        return ((p6 - p5) / abs(p5)) * 100.0 if abs(p5) > 1e-8 else 0.0

    d_delay_pct = calc_pct(gat_delay_m, tf_delay_m)
    d_queue_pct = calc_pct(gat_queue_m, tf_queue_m)
    d_tp_pct = calc_pct(gat_tp_m, tf_tp_m)
    d_ret_pct = calc_pct(gat_ret_m, tf_ret_m)

    print("\n" + "=" * 90)
    print(f"{'Metric':<25} | {'Phase 5 mean±std':<22} | {'Phase 6 mean±std':<22} | {'Change (%)':<15}")
    print("-" * 90)
    print(f"{'Average Delay (s)':<25} | {tf_delay_m:6.2f} ± {tf_delay_s:5.2f}       | {gat_delay_m:6.2f} ± {gat_delay_s:5.2f}       | {d_delay_pct:+6.2f}%")
    print(f"{'Average Queue (veh)':<25} | {tf_queue_m:6.2f} ± {tf_queue_s:5.2f}       | {gat_queue_m:6.2f} ± {gat_queue_s:5.2f}       | {d_queue_pct:+6.2f}%")
    print(f"{'Throughput (vehicles)':<25} | {tf_tp_m:6.1f} ± {tf_tp_s:5.1f}       | {gat_tp_m:6.1f} ± {gat_tp_s:5.1f}       | {d_tp_pct:+6.2f}%")
    print(f"{'Aggregate Return':<25} | {tf_ret_m:7.2f} ± {tf_ret_s:6.2f}     | {gat_ret_m:7.2f} ± {gat_ret_s:6.2f}     | {d_ret_pct:+6.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    run_comparison()
