"""
Phase 7 Formal Comparative Evaluation:
Frozen Phase 6 (MAPPO + Temporal + GAT) vs. Phase 7 (MAPPO + Temporal + GAT + Learnable Comm Gate).

Evaluates both controllers across seeds 1001-1010 under identical simulation conditions:
- Same 2x2 grid network
- Same normal demand configuration & dynamic route generation
- Same 100 decision steps (500s SUMO simulation time per episode)
- Deterministic action selection (argmax)
- Centralized critic is NEVER used during evaluation
- Frozen Phase 6 loads: models/checkpoints/mappo_transformer_gat_actor.pt
- Phase 7 loads: models/checkpoints/mappo_comm_gate_actor.pt

Outputs:
- experiments/evaluation_comparison_comm_gate.csv
- experiments/controller_comparison_comm_gate.png
- experiments/communication_gate_log.csv
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
from models.communication_gate import MAPPOGateController, ALLOWED_DIRECTED_EDGES


def evaluate_phase6(env, seeds, steps_per_episode, p6_ctrl):
    results = []
    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        p6_ctrl.reset_history(obs)

        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            actions, _ = p6_ctrl.get_actions(deterministic=True)
            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            p6_ctrl.update_history(next_obs)
            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                break

        results.append({
            "episode": ep_idx,
            "seed": seed,
            "controller": "Phase 6: GAT (No Gating)",
            "mean_delay": float(np.mean(ep_waiting)),
            "mean_queue": float(np.mean(ep_queues)),
            "throughput": ep_throughput,
            "task_return": sum(ep_rewards.values()),
            "comm_cost": "N/A",
            "adjusted_return": "N/A",
            "possible_comm": "N/A",
            "actual_comm": "N/A",
            "comm_ratio": "N/A"
        })
    return results


def evaluate_phase7(env, seeds, steps_per_episode, p7_ctrl, comm_log_path):
    results = []
    comm_records = []

    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        p7_ctrl.reset_history(obs)

        ep_task_rewards = {a: 0.0 for a in env.agents}
        ep_comm_costs = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0
        ep_actual_comm = 0
        ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

        for step in range(steps_per_episode):
            ctx = p7_ctrl.compute_heuristic_context(info)
            actions, _, comm_costs, actual_comm = p7_ctrl.get_actions(ctx, deterministic=True)
            ep_actual_comm += actual_comm

            # Log edge-level gate values for detailed inspection
            gate_vals = p7_ctrl.gated_gat.comm_gate.last_gate_values
            for (src, tgt), g_val in gate_vals.items():
                is_comm = (g_val >= p7_ctrl.comm_threshold)
                cost = p7_ctrl.comm_cost_weight * (1.0 if is_comm else 0.0)
                comm_records.append({
                    "seed": seed,
                    "episode": ep_idx,
                    "decision_step": step,
                    "source": src,
                    "target": tgt,
                    "gate_value": f"{g_val:.4f}",
                    "communicated": is_comm,
                    "communication_cost": f"{cost:.4f}"
                })

            next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_task_rewards[a] += task_rewards[a]
                ep_comm_costs[a] += comm_costs[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            p7_ctrl.update_history(next_obs)
            obs = next_obs
            info = next_info
            if any(terminations.values()) or any(truncations.values()):
                break

        task_ret = sum(ep_task_rewards.values())
        tot_comm_cost = sum(ep_comm_costs.values())
        adj_ret = task_ret - tot_comm_cost
        comm_ratio = ep_actual_comm / ep_possible_comm

        results.append({
            "episode": ep_idx,
            "seed": seed,
            "controller": "Phase 7: GAT + Comm Gate",
            "mean_delay": float(np.mean(ep_waiting)),
            "mean_queue": float(np.mean(ep_queues)),
            "throughput": ep_throughput,
            "task_return": task_ret,
            "comm_cost": tot_comm_cost,
            "adjusted_return": adj_ret,
            "possible_comm": ep_possible_comm,
            "actual_comm": ep_actual_comm,
            "comm_ratio": comm_ratio
        })

    # Save detailed edge log
    if comm_log_path:
        with open(comm_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "episode", "decision_step", "source", "target",
                "gate_value", "communicated", "communication_cost"
            ])
            writer.writeheader()
            for r in comm_records:
                writer.writerow(r)

    return results


def plot_comparison(p6_res, p7_res, output_path):
    controllers = ["Phase 6 (GAT)", "Phase 7 (GAT + Comm Gate)"]
    colors = ["#cba6f7", "#a6e3a1"]

    metrics = [
        ("mean_delay", "Average Delay (s)", [r["mean_delay"] for r in p6_res], [r["mean_delay"] for r in p7_res]),
        ("mean_queue", "Average Queue Length (veh)", [r["mean_queue"] for r in p6_res], [r["mean_queue"] for r in p7_res]),
        ("throughput", "Throughput (Arrived veh)", [r["throughput"] for r in p6_res], [r["throughput"] for r in p7_res]),
        ("task_return", "Task-Only Return", [r["task_return"] for r in p6_res], [r["task_return"] for r in p7_res])
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")

    for ax, (key, title, p6_data, p7_data) in zip(axs.flat, metrics):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#585b70", axis="y")

        means = [np.mean(p6_data), np.mean(p7_data)]
        stds = [np.std(p6_data), np.std(p7_data)]
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


def run_evaluation():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_seeds = list(range(1001, 1011))
    steps_per_episode = 100

    csv_path = "experiments/evaluation_comparison_comm_gate.csv"
    plot_path = "experiments/controller_comparison_comm_gate.png"
    comm_log_path = "experiments/communication_gate_log.csv"

    print("=" * 115)
    print("SAGE-Traffic Phase 7 Formal Comparative Evaluation")
    print("Frozen Phase 6 (GAT) vs. Phase 7 (GAT + Learnable Communication Gate)")
    print("=" * 115)
    print(f"Evaluation Seeds:    {eval_seeds}")
    print(f"Steps per Episode:   {steps_per_episode} (500s SUMO simulation time)")
    print(f"Action Mode:         Deterministic Greedy (argmax)")
    print(f"Gate Mode:           Deterministic Hard Threshold (>= {config['communication']['threshold']})")
    print("-" * 115)

    env = MultiAgentTrafficEnv(config_path=config_path)

    # 1. Evaluate Frozen Phase 6 Baseline
    print("\n[1/2] Loading Frozen Phase 6 Checkpoint (GAT)...")
    p6_ctrl = MAPPOGATController(config_path=config_path)
    p6_ctrl.load_checkpoints("models/checkpoints")
    print("      Running Phase 6 evaluation across seeds 1001-1010...")
    p6_res = evaluate_phase6(env, eval_seeds, steps_per_episode, p6_ctrl)
    print("      Phase 6 evaluation completed.")

    # 2. Evaluate Phase 7 Model
    print("\n[2/2] Loading Phase 7 Checkpoint (GAT + Comm Gate)...")
    p7_ctrl = MAPPOGateController(config_path=config_path)
    p7_ctrl.load_checkpoints("models/checkpoints")
    print("      Running Phase 7 evaluation across seeds 1001-1010...")
    p7_res = evaluate_phase7(env, eval_seeds, steps_per_episode, p7_ctrl, comm_log_path)
    print("      Phase 7 evaluation completed.")

    env.close()

    # Save summary CSV
    fieldnames = [
        "seed", "episode", "controller",
        "mean_delay", "mean_queue", "throughput",
        "task_return", "comm_cost", "adjusted_return",
        "possible_comm", "actual_comm", "comm_ratio"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in p6_res:
            writer.writerow(r)
        for r in p7_res:
            writer.writerow(r)

    print(f"\nSaved evaluation CSV -> {csv_path}")
    print(f"Saved detailed edge communication log -> {comm_log_path}")

    # Generate comparative plot
    plot_comparison(p6_res, p7_res, plot_path)
    print(f"Saved comparative bar chart -> {plot_path}")

    # Print Detailed Per-Seed Table
    print("\n" + "=" * 130)
    print(f"{'Seed':<5} | {'Controller':<26} | {'Delay(s)':<10} | {'Queue':<8} | {'Throughput':<11} | {'Task Ret':<11} | {'Comm Cost':<10} | {'Comm Ratio':<11}")
    print("-" * 130)
    for r6, r7 in zip(p6_res, p7_res):
        print(f"{r6['seed']:<5} | {'Phase 6 (GAT)':<26} | {r6['mean_delay']:<10.2f} | {r6['mean_queue']:<8.2f} | {r6['throughput']:<11} | {r6['task_return']:<11.2f} | {'N/A':<10} | {'N/A':<11}")
        c_ratio_str = f"{r7['comm_ratio']*100:.1f}% ({r7['actual_comm']}/{r7['possible_comm']})"
        print(f"{r7['seed']:<5} | {'Phase 7 (GAT+Gate)':<26} | {r7['mean_delay']:<10.2f} | {r7['mean_queue']:<8.2f} | {r7['throughput']:<11} | {r7['task_return']:<11.2f} | {r7['comm_cost']:<10.2f} | {c_ratio_str:<11}")
        print("-" * 130)

    # Compute Summary Statistics
    p6_delay_m, p6_delay_s = np.mean([r["mean_delay"] for r in p6_res]), np.std([r["mean_delay"] for r in p6_res])
    p7_delay_m, p7_delay_s = np.mean([r["mean_delay"] for r in p7_res]), np.std([r["mean_delay"] for r in p7_res])

    p6_queue_m, p6_queue_s = np.mean([r["mean_queue"] for r in p6_res]), np.std([r["mean_queue"] for r in p6_res])
    p7_queue_m, p7_queue_s = np.mean([r["mean_queue"] for r in p7_res]), np.std([r["mean_queue"] for r in p7_res])

    p6_tp_m, p6_tp_s = np.mean([r["throughput"] for r in p6_res]), np.std([r["throughput"] for r in p6_res])
    p7_tp_m, p7_tp_s = np.mean([r["throughput"] for r in p7_res]), np.std([r["throughput"] for r in p7_res])

    p6_ret_m, p6_ret_s = np.mean([r["task_return"] for r in p6_res]), np.std([r["task_return"] for r in p6_res])
    p7_ret_m, p7_ret_s = np.mean([r["task_return"] for r in p7_res]), np.std([r["task_return"] for r in p7_res])

    p7_comm_ratio_m, p7_comm_ratio_s = np.mean([r["comm_ratio"] for r in p7_res]), np.std([r["comm_ratio"] for r in p7_res])
    p7_comm_act_m = np.mean([r["actual_comm"] for r in p7_res])
    p7_comm_cost_m = np.mean([r["comm_cost"] for r in p7_res])

    def pct_chg(p7, p6):
        return ((p7 - p6) / abs(p6)) * 100.0 if abs(p6) > 1e-8 else 0.0

    print("\n" + "=" * 90)
    print(f"{'Metric':<25} | {'Phase 6 (mean±std)':<22} | {'Phase 7 (mean±std)':<22} | {'Change (%)':<15}")
    print("-" * 90)
    print(f"{'Avg Delay (s)':<25} | {p6_delay_m:6.2f} ± {p6_delay_s:5.2f}       | {p7_delay_m:6.2f} ± {p7_delay_s:5.2f}       | {pct_chg(p7_delay_m, p6_delay_m):+6.2f}%")
    print(f"{'Avg Queue (veh)':<25} | {p6_queue_m:6.2f} ± {p6_queue_s:5.2f}       | {p7_queue_m:6.2f} ± {p7_queue_s:5.2f}       | {pct_chg(p7_queue_m, p6_queue_m):+6.2f}%")
    print(f"{'Throughput (veh)':<25} | {p6_tp_m:6.1f} ± {p6_tp_s:5.1f}       | {p7_tp_m:6.1f} ± {p7_tp_s:5.1f}       | {pct_chg(p7_tp_m, p6_tp_m):+6.2f}%")
    print(f"{'Task-only Return':<25} | {p6_ret_m:7.2f} ± {p6_ret_s:6.2f}     | {p7_ret_m:7.2f} ± {p7_ret_s:6.2f}     | {pct_chg(p7_ret_m, p6_ret_m):+6.2f}%")
    print(f"{'Communication Ratio':<25} | {'N/A':<22} | {p7_comm_ratio_m*100:5.1f}% ± {p7_comm_ratio_s*100:4.1f}%     | {'N/A':<15}")
    print(f"{'Actual Communications':<25} | {'N/A':<22} | {p7_comm_act_m:5.1f} / 800         | {'N/A':<15}")
    print(f"{'Communication Cost':<25} | {'N/A':<22} | {p7_comm_cost_m:6.2f}               | {'N/A':<15}")
    print("=" * 90)


if __name__ == "__main__":
    run_evaluation()
