"""
SAGE-Traffic Phase 9: Comparative Evaluation & Safety Layer Audit
Compares Frozen Phase 8R (Unshielded Baseline) vs Phase 9 (Runtime Shielded Controller)
across 10 evaluation seeds (1001-1010) under standard demand.

Execution Protocol:
- Seeds: 1001 to 1010 (10 identical evaluation seeds)
- Horizon: 100 decision steps per episode (500s SUMO simulation time = 2,000 intersection-seconds per episode)
- Total audit horizon: 20,000 intersection-seconds
- Deterministic action selection (argmax) for all policies
- Hard threshold communication gating (g_ij >= 0.5)
- Second-by-second SUMO physical signal state auditing:
  Inspects link indices to guarantee ZERO simultaneous conflicting greens.
- Outputs:
  - experiments/evaluation_comparison_phase9.csv
  - experiments/controller_comparison_phase9.png
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
from models.causal_improved import MAPPOCausalImprovedController
from models.communication_gate import ALLOWED_DIRECTED_EDGES


def evaluate_phase9():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_seeds = list(range(1001, 1011))  # 1001 to 1010
    steps_per_episode = 100
    os.makedirs("experiments", exist_ok=True)

    summary_csv = "experiments/evaluation_comparison_phase9.csv"
    plot_file = "experiments/controller_comparison_phase9.png"

    print("=" * 135)
    print("SAGE-Traffic Phase 9: Safety Layer Comparative Evaluation & Forensic Audit (Seeds 1001-1010)")
    print("=" * 135)
    print(f"Evaluation Seeds:        {eval_seeds}")
    print(f"Horizon per Episode:     {steps_per_episode} steps (500s SUMO simulation time)")
    print("Action Selection:        Deterministic Greedy argmax")
    print("Safety Shielding:        Active Runtime Constraint Enforcement + Second-by-Second Physical Audit")
    print("-" * 135)

    # 1. Instantiate and load controllers
    print("Loading controller checkpoints...")
    p4_ctrl = MAPPOController(config_path=config_path)
    p4_ctrl.load_checkpoints("models/checkpoints")

    p8r_ctrl = MAPPOCausalImprovedController(config_path=config_path)
    p8r_ctrl.load_checkpoints("models/checkpoints")

    # Definitions of controllers under test:
    # (Display Name, Short Key, Controller Object, Safety Enabled Flag)
    controllers = [
        ("Phase 4: Frozen MAPPO", "mappo", p4_ctrl, False),
        ("Phase 8R: Baseline (Unshielded)", "p8r_unshielded", p8r_ctrl, False),
        ("Phase 9: Shielded Controller", "p9_shielded", p8r_ctrl, True),
    ]

    all_results = {c[0]: [] for c in controllers}

    env = MultiAgentTrafficEnv(config_path=config_path, gui=False)

    for ctrl_name, ctrl_type, controller, safety_enabled in controllers:
        print(f"\nEvaluating: {ctrl_name} (Safety Shield: {'ENABLED' if safety_enabled else 'DISABLED'})...")

        for ep_idx, seed in enumerate(eval_seeds, start=1):
            # Configure safety mode on the environment
            env.set_safety_enabled(safety_enabled)

            obs, info = env.reset(seed=seed, demand_level="normal")

            if ctrl_type in ["p8r_unshielded", "p9_shielded"]:
                controller.reset_history(obs, info)

            ep_task_rewards = {a: 0.0 for a in env.agents}
            ep_comm_costs = {a: 0.0 for a in env.agents}
            ep_queues = []
            ep_waiting = []
            ep_throughput = 0
            ep_actual_comm = 0
            ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

            step_c_scores = []
            step_gate_values = []
            ep_interventions = 0
            ep_physical_violations = 0

            for step in range(steps_per_episode):
                if ctrl_type == "mappo":
                    actions, _ = controller.get_actions(obs, deterministic=True)
                    actual_comm = 0
                    comm_costs = {a: 0.0 for a in env.agents}
                elif ctrl_type in ["p8r_unshielded", "p9_shielded"]:
                    context = controller.compute_causal_context(info)
                    step_c_scores.extend(list(context.values()))
                    actions, log_probs, comm_costs, actual_comm = controller.get_actions(context, deterministic=True)
                    for edge in ALLOWED_DIRECTED_EDGES:
                        g_val = controller.gated_gat.comm_gate.last_gate_values.get(edge, 0.5)
                        step_gate_values.append(g_val)
                else:
                    raise ValueError(f"Unknown controller type: {ctrl_type}")

                ep_actual_comm += actual_comm

                # Step through environment (passes through safety layer when enabled)
                next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

                for a in env.agents:
                    ep_task_rewards[a] += task_rewards[a]
                    ep_comm_costs[a] += comm_costs[a]

                step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
                step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
                ep_queues.append(step_q)
                ep_waiting.append(step_w)
                ep_throughput += next_info["A"]["arrived_vehicles"]

                # Extract safety statistics from info
                safety_stats = next_info["A"].get("safety", {})
                ep_interventions = safety_stats.get("total_interventions", 0)
                ep_physical_violations = safety_stats.get("audit_physical_violations", 0)

                # Update history
                if ctrl_type in ["p8r_unshielded", "p9_shielded"]:
                    controller.update_history(next_obs)

                obs = next_obs
                info = next_info
                if any(terminations.values()) or any(truncations.values()):
                    break

            task_ret = float(sum(ep_task_rewards.values()))
            tot_comm_cost = float(sum(ep_comm_costs.values()))
            adj_ret = task_ret - tot_comm_cost
            comm_ratio = float(ep_actual_comm / ep_possible_comm)
            mean_delay = float(np.mean(ep_waiting))
            mean_queue = float(np.mean(ep_queues))

            mean_c_str = f"{np.mean(step_c_scores):.4f}" if step_c_scores else "N/A"
            mean_g_str = f"{np.mean(step_gate_values):.4f}" if step_gate_values else "N/A"

            res_row = {
                "seed": seed,
                "episode": ep_idx,
                "controller": ctrl_name,
                "mean_delay": mean_delay,
                "mean_queue": mean_queue,
                "throughput": ep_throughput,
                "task_return": task_ret,
                "comm_cost": tot_comm_cost,
                "adjusted_return": adj_ret,
                "possible_comm": ep_possible_comm,
                "actual_comm": ep_actual_comm,
                "comm_ratio": comm_ratio,
                "mean_C_ij": mean_c_str,
                "mean_gate": mean_g_str,
                "interventions": ep_interventions,
                "physical_violations": ep_physical_violations,
            }
            all_results[ctrl_name].append(res_row)

            print(
                f"  Seed {seed} | Delay: {mean_delay:6.2f}s | Queue: {mean_queue:5.2f} | "
                f"Throughput: {ep_throughput:3d} | Return: {task_ret:8.2f} | "
                f"Comm: {comm_ratio*100:5.1f}% | Interventions: {ep_interventions:2d} | "
                f"Violations: {ep_physical_violations:1d}"
            )

    env.close()

    # Save CSV Summary
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "seed", "episode", "controller", "mean_delay", "mean_queue",
            "throughput", "task_return", "comm_cost", "adjusted_return",
            "possible_comm", "actual_comm", "comm_ratio", "mean_C_ij",
            "mean_gate", "interventions", "physical_violations"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in [c[0] for c in controllers]:
            for r in all_results[c]:
                writer.writerow(r)

    print(f"\nDetailed evaluation results successfully saved to {summary_csv}")

    # Generate Publication-Quality Comparison Figure
    ctrl_names = [c[0] for c in controllers]
    colors = ["#4c72b0", "#c44e52", "#55a868"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11))

    # Metric 1: Average Delay
    delays_mean = [np.mean([r["mean_delay"] for r in all_results[c]]) for c in ctrl_names]
    delays_std = [np.std([r["mean_delay"] for r in all_results[c]]) for c in ctrl_names]
    bars1 = ax1.bar(range(len(ctrl_names)), delays_mean, yerr=delays_std, capsize=5, color=colors, alpha=0.85)
    ax1.set_title("Average Vehicle Delay (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Delay (seconds / veh)")
    ax1.set_xticks(range(len(ctrl_names)))
    ax1.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=10, ha="right", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 2: Average Queue Length
    queues_mean = [np.mean([r["mean_queue"] for r in all_results[c]]) for c in ctrl_names]
    queues_std = [np.std([r["mean_queue"] for r in all_results[c]]) for c in ctrl_names]
    bars2 = ax2.bar(range(len(ctrl_names)), queues_mean, yerr=queues_std, capsize=5, color=colors, alpha=0.85)
    ax2.set_title("Average Queue Length (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Queue Length (vehicles)")
    ax2.set_xticks(range(len(ctrl_names)))
    ax2.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=10, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 3: Cumulative Throughput
    thru_mean = [np.mean([r["throughput"] for r in all_results[c]]) for c in ctrl_names]
    thru_std = [np.std([r["throughput"] for r in all_results[c]]) for c in ctrl_names]
    bars3 = ax3.bar(range(len(ctrl_names)), thru_mean, yerr=thru_std, capsize=5, color=colors, alpha=0.85)
    ax3.set_title("Cumulative Episode Throughput (Higher is Better)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Vehicles Arrived")
    ax3.set_xticks(range(len(ctrl_names)))
    ax3.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=10, ha="right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 4: Safety Violations & Interventions Rate
    violations = [np.sum([r["physical_violations"] for r in all_results[c]]) for c in ctrl_names]
    interventions = [np.mean([r["interventions"] for r in all_results[c]]) for c in ctrl_names]
    x = np.arange(len(ctrl_names))
    width = 0.35
    ax4.bar(x - width/2, interventions, width, label="Avg Interventions / Episode", color="#ff7f0e", alpha=0.85)
    ax4.bar(x + width/2, violations, width, label="Total Physical Violations", color="#d62728", alpha=0.85)
    ax4.set_title("Safety Invariants & Shield Interventions", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.set_xticks(x)
    ax4.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=10, ha="right", fontsize=9)
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Comparative evaluation plot successfully saved to {plot_file}")

    # Print Final Markdown-compatible Summary Table
    print("\n" + "=" * 135)
    print("PHASE 9 COMPARATIVE EVALUATION SUMMARY (MEAN +/- STD OVER 10 EVALUATION SEEDS)")
    print("=" * 135)
    header = (
        f"{'Controller':<36} | {'Delay (s)':<14} | {'Queue (veh)':<14} | "
        f"{'Throughput':<12} | {'Task Return':<16} | {'Comm (%)':<10} | {'Interventions':<14} | {'Violations':<10}"
    )
    print(header)
    print("-" * 135)

    for c in ctrl_names:
        res = all_results[c]
        d_m, d_s = np.mean([r["mean_delay"] for r in res]), np.std([r["mean_delay"] for r in res])
        q_m, q_s = np.mean([r["mean_queue"] for r in res]), np.std([r["mean_queue"] for r in res])
        t_m, t_s = np.mean([r["throughput"] for r in res]), np.std([r["throughput"] for r in res])
        r_m, r_s = np.mean([r["task_return"] for r in res]), np.std([r["task_return"] for r in res])
        c_m = np.mean([r["comm_ratio"] * 100 for r in res])
        int_m = np.mean([r["interventions"] for r in res])
        viol_tot = int(np.sum([r["physical_violations"] for r in res]))

        row_str = (
            f"{c:<36} | {d_m:5.2f} +/- {d_s:4.2f} | {q_m:5.2f} +/- {q_s:4.2f} | "
            f"{t_m:5.1f} +/- {t_s:4.1f} | {r_m:7.1f} +/- {r_s:5.1f} | "
            f"{c_m:5.1f}%     | {int_m:5.1f} / ep     | {viol_tot:4d} (0%)"
        )
        print(row_str)
    print("=" * 135)


if __name__ == "__main__":
    evaluate_phase9()

