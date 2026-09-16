"""
Phase 8 Comparative Evaluation Script: Phase 7 (GAT + Comm Gate) vs Phase 8 (GAT + Causal Gate).

Runs a strictly controlled, deterministic 10-seed evaluation (seeds 1001-1010):
- Evaluates Phase 7 using frozen checkpoint models/checkpoints/mappo_comm_gate_actor.pt
- Evaluates Phase 8 using trained checkpoint models/checkpoints/mappo_causal_gate_actor.pt
- Both controllers evaluated under identical environment seeds, normal demand, and 500s horizon.
- Logs step-by-step causal influence estimates and decisions to experiments/causal_influence_log.csv
- Logs seed-level summary to experiments/evaluation_comparison_causal_gate.csv
- Generates comparative bar charts saved to experiments/controller_comparison_causal_gate.png
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
from models.communication_gate import (
    MAPPOGateController,
    MAPPOCausalGateController,
    ALLOWED_DIRECTED_EDGES
)
from models.causal_influence import CORRIDOR_INCOMING_LANES


def run_evaluation():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_seeds = list(range(1001, 1011))  # 1001 to 1010
    steps_per_episode = 100
    os.makedirs("experiments", exist_ok=True)

    summary_csv = "experiments/evaluation_comparison_causal_gate.csv"
    causal_log_csv = "experiments/causal_influence_log.csv"
    plot_file = "experiments/controller_comparison_causal_gate.png"

    print("=" * 115)
    print("SAGE-Traffic Phase 8 Evaluation: Phase 7 vs Phase 8 (Causal Influence Gate)")
    print("=" * 115)
    print(f"Evaluation Seeds:        {eval_seeds}")
    print(f"Horizon per Episode:     {steps_per_episode} steps (500s SUMO simulation time)")
    print("Action Selection:        Deterministic Greedy argmax")
    print("Gate Decision:           Hard Threshold g_ij >= 0.5")
    print("-" * 115)

    # 1. Initialize Controllers
    ctrl_p7 = MAPPOGateController(config_path=config_path)
    ctrl_p7.load_checkpoints("models/checkpoints")

    ctrl_p8 = MAPPOCausalGateController(config_path=config_path)
    ctrl_p8.load_checkpoints("models/checkpoints")

    # CSV Writers
    summary_fields = [
        "seed",
        "episode",
        "controller",
        "mean_delay",
        "mean_queue",
        "throughput",
        "task_return",
        "comm_cost",
        "adjusted_return",
        "possible_comm",
        "actual_comm",
        "comm_ratio",
        "mean_C_ij"
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

    causal_fields = [
        "seed",
        "episode",
        "decision_step",
        "source",
        "target",
        "C_ij",
        "gate_value",
        "communicated",
        "queue_src",
        "wait_src",
        "demand_src",
        "corridor_queue",
        "queue_tgt",
        "wait_tgt"
    ]
    with open(causal_log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=causal_fields)
        writer.writeheader()

    controllers = [
        ("Phase 7: GAT + Comm Gate", ctrl_p7, False),
        ("Phase 8: GAT + Causal Gate", ctrl_p8, True)
    ]

    all_results = {name: [] for name, _, _ in controllers}

    env = MultiAgentTrafficEnv(config_path=config_path)

    try:
        for ctrl_name, controller, is_phase8 in controllers:
            print(f"\nEvaluating: {ctrl_name}")
            print(f"{'Seed':<6} | {'Delay (s)':<10} | {'Queue':<8} | {'Throughput':<10} | {'Task Ret':<10} | {'Comm Cost':<10} | {'Comm Ratio':<12} | {'Mean C_ij':<10}")
            print("-" * 95)

            for ep_idx, seed in enumerate(eval_seeds, start=1):
                obs, info = env.reset(seed=seed)
                if is_phase8:
                    controller.reset_history(obs, info)
                else:
                    controller.reset_history(obs)

                ep_task_rewards = {a: 0.0 for a in env.agents}
                ep_comm_costs = {a: 0.0 for a in env.agents}
                ep_queues = []
                ep_waiting = []
                ep_actual_comm = 0
                ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800
                step_c_scores = []

                for step in range(steps_per_episode):
                    if is_phase8:
                        # Phase 8 Causal context C_ij
                        context = controller.compute_causal_context(info)
                        step_c_scores.extend(list(context.values()))
                    else:
                        # Phase 7 Heuristic context c_ij
                        context = controller.compute_heuristic_context(info)

                    # Action selection
                    actions, log_probs, comm_costs, actual_comm = controller.get_actions(context, deterministic=True)
                    ep_actual_comm += actual_comm

                    # If Phase 8, log step-by-step causal influence metrics
                    if is_phase8:
                        with open(causal_log_csv, "a", newline="", encoding="utf-8") as cf:
                            cwriter = csv.DictWriter(cf, fieldnames=causal_fields)
                            for (src, tgt), c_val in context.items():
                                g_val = controller.gated_gat.comm_gate.last_gate_values.get((src, tgt), 0.5)
                                is_comm = controller.gated_gat.comm_gate.last_comm_decisions.get((src, tgt), True)

                                q_src = info[src]["metrics"]["queue_length"]
                                w_src = info[src]["metrics"]["waiting_time"]
                                d_src = info[src]["metrics"]["vehicle_count"]
                                q_tgt = info[tgt]["metrics"]["queue_length"]
                                w_tgt = info[tgt]["metrics"]["waiting_time"]

                                tgt_lanes = CORRIDOR_INCOMING_LANES.get((src, tgt), [])
                                l_details = info[tgt]["metrics"].get("lane_details", {})
                                c_queue = sum(l_details.get(l, {}).get("queue_length", 0.0) for l in tgt_lanes)

                                cwriter.writerow({
                                    "seed": seed,
                                    "episode": ep_idx,
                                    "decision_step": step,
                                    "source": src,
                                    "target": tgt,
                                    "C_ij": c_val,
                                    "gate_value": g_val,
                                    "communicated": int(is_comm),
                                    "queue_src": q_src,
                                    "wait_src": w_src,
                                    "demand_src": d_src,
                                    "corridor_queue": c_queue,
                                    "queue_tgt": q_tgt,
                                    "wait_tgt": w_tgt
                                })

                    # Advance environment
                    next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

                    for a in env.agents:
                        ep_task_rewards[a] += task_rewards[a]
                        ep_comm_costs[a] += comm_costs[a]

                    step_queues = [next_info[a]["metrics"]["queue_length"] for a in env.agents]
                    step_waits = [next_info[a]["metrics"]["waiting_time"] for a in env.agents]
                    ep_queues.append(np.mean(step_queues))
                    ep_waiting.append(np.mean(step_waits))

                    controller.update_history(next_obs)
                    obs = next_obs
                    info = next_info

                ep_throughput = sum(info[a]["metrics"]["vehicle_count"] for a in env.agents) // 4
                task_ret = float(sum(ep_task_rewards.values()))
                total_comm_cost = float(sum(ep_comm_costs.values()))
                adj_ret = task_ret - total_comm_cost
                mean_delay = float(np.mean(ep_waiting))
                mean_queue = float(np.mean(ep_queues))
                comm_ratio = float(ep_actual_comm / ep_possible_comm)
                mean_c = float(np.mean(step_c_scores)) if step_c_scores else None

                res_row = {
                    "seed": seed,
                    "episode": ep_idx,
                    "controller": ctrl_name,
                    "mean_delay": mean_delay,
                    "mean_queue": mean_queue,
                    "throughput": ep_throughput,
                    "task_return": task_ret,
                    "comm_cost": total_comm_cost,
                    "adjusted_return": adj_ret,
                    "possible_comm": ep_possible_comm,
                    "actual_comm": ep_actual_comm,
                    "comm_ratio": comm_ratio,
                    "mean_C_ij": mean_c if mean_c is not None else "N/A"
                }

                all_results[ctrl_name].append(res_row)

                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=summary_fields)
                    writer.writerow(res_row)

                c_str = f"{mean_c:<10.3f}" if mean_c is not None else f"{'N/A':<10}"
                print(
                    f"{seed:<6} | {mean_delay:<10.2f} | {mean_queue:<8.2f} | {ep_throughput:<10} | "
                    f"{task_ret:<10.2f} | {total_comm_cost:<10.2f} | {comm_ratio * 100:<11.1f}% | {c_str}"
                )

    finally:
        env.close()

    # Generate Comparison Plot
    p7_delays = [r["mean_delay"] for r in all_results["Phase 7: GAT + Comm Gate"]]
    p8_delays = [r["mean_delay"] for r in all_results["Phase 8: GAT + Causal Gate"]]

    p7_queues = [r["mean_queue"] for r in all_results["Phase 7: GAT + Comm Gate"]]
    p8_queues = [r["mean_queue"] for r in all_results["Phase 8: GAT + Causal Gate"]]

    p7_thru = [r["throughput"] for r in all_results["Phase 7: GAT + Comm Gate"]]
    p8_thru = [r["throughput"] for r in all_results["Phase 8: GAT + Causal Gate"]]

    p7_ret = [r["task_return"] for r in all_results["Phase 7: GAT + Comm Gate"]]
    p8_ret = [r["task_return"] for r in all_results["Phase 8: GAT + Causal Gate"]]

    p7_ratio = [r["comm_ratio"] * 100 for r in all_results["Phase 7: GAT + Comm Gate"]]
    p8_ratio = [r["comm_ratio"] * 100 for r in all_results["Phase 8: GAT + Causal Gate"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(eval_seeds))
    width = 0.35

    # Panel 1: Delay
    axes[0, 0].bar(x - width/2, p7_delays, width, label="Phase 7 (Heuristic Gate)", color="#7f7f7f")
    axes[0, 0].bar(x + width/2, p8_delays, width, label="Phase 8 (Causal Gate)", color="#1f77b4")
    axes[0, 0].set_title("Average Delay (seconds / vehicle)", fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(s) for s in eval_seeds])
    axes[0, 0].set_ylabel("Delay (s)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel 2: Queue
    axes[0, 1].bar(x - width/2, p7_queues, width, label="Phase 7 (Heuristic Gate)", color="#7f7f7f")
    axes[0, 1].bar(x + width/2, p8_queues, width, label="Phase 8 (Causal Gate)", color="#2ca02c")
    axes[0, 1].set_title("Average Queue Length (vehicles)", fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(s) for s in eval_seeds])
    axes[0, 1].set_ylabel("Queue (veh)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel 3: Throughput
    axes[1, 0].bar(x - width/2, p7_thru, width, label="Phase 7 (Heuristic Gate)", color="#7f7f7f")
    axes[1, 0].bar(x + width/2, p8_thru, width, label="Phase 8 (Causal Gate)", color="#ff7f0e")
    axes[1, 0].set_title("Completed Throughput (vehicles)", fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([str(s) for s in eval_seeds])
    axes[1, 0].set_ylabel("Vehicles")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel 4: Communication Ratio (%)
    axes[1, 1].bar(x - width/2, p7_ratio, width, label="Phase 7 (Heuristic Gate)", color="#7f7f7f")
    axes[1, 1].bar(x + width/2, p8_ratio, width, label="Phase 8 (Causal Gate)", color="#9467bd")
    axes[1, 1].set_title("Communication Ratio (% of 800 possible)", fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([str(s) for s in eval_seeds])
    axes[1, 1].set_ylabel("Communication Ratio (%)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"\nComparative evaluation plot saved to {plot_file}")

    # Copy to artifacts directory
    artifact_plot = os.path.join(
        r"C:\Users\kashi\.gemini\antigravity-ide\brain\549c93f1-f3e5-4dbb-865e-10d0077ea528",
        "controller_comparison_causal_gate.png"
    )
    try:
        import shutil
        shutil.copyfile(plot_file, artifact_plot)
    except Exception:
        pass


if __name__ == "__main__":
    run_evaluation()
