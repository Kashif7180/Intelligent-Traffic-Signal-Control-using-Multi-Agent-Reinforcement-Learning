"""
Comprehensive 10-Seed Comparative Evaluation for Phase 8R.

Evaluates and compares 5 controllers under strictly identical conditions:
1. Phase 4 MAPPO (Frozen Baseline)
2. Phase 6 GAT (Frozen Baseline)
3. Phase 7 Communication Gate (Frozen Baseline)
4. Phase 8 Causal Gate (Archived Baseline)
5. Phase 8R Improved Causal Communication (Newly Trained)

Experimental Protocol:
- Seeds: 1001 to 1010 (10 evaluation seeds)
- Horizon: 100 decision steps per episode (500s SUMO simulation time)
- Same 2x2 grid road network & normal demand profile
- Deterministic action selection (argmax) for all RL controllers
- Hard threshold decision (g_ij >= 0.5) for communication channels
- Centralized critic is NEVER used during action selection or evaluation
- Outputs:
  - experiments/evaluation_comparison_phase8r.csv
  - experiments/controller_comparison_phase8r.png
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
from models.gat_network import MAPPOGATController
from models.communication_gate import MAPPOGateController, MAPPOCausalGateController, ALLOWED_DIRECTED_EDGES
from models.causal_improved import MAPPOCausalImprovedController


def evaluate_all():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_seeds = list(range(1001, 1011))  # 1001 to 1010
    steps_per_episode = 100
    os.makedirs("experiments", exist_ok=True)

    summary_csv = "experiments/evaluation_comparison_phase8r.csv"
    plot_file = "experiments/controller_comparison_phase8r.png"

    print("=" * 135)
    print("SAGE-Traffic Phase 8R: 5-Controller Comparative Evaluation (10 Evaluation Seeds: 1001-1010)")
    print("=" * 135)
    print(f"Evaluation Seeds:        {eval_seeds}")
    print(f"Horizon per Episode:     {steps_per_episode} steps (500s SUMO simulation time)")
    print("Action Selection:        Deterministic Greedy argmax")
    print("Gate Decision:           Hard Threshold g_ij >= 0.5")
    print("-" * 135)

    # 1. Instantiate and load all 5 controllers
    print("Loading controller checkpoints...")
    p4_ctrl = MAPPOController(config_path=config_path)
    p4_ctrl.load_checkpoints("models/checkpoints")

    p6_ctrl = MAPPOGATController(config_path=config_path)
    p6_ctrl.load_checkpoints("models/checkpoints")

    p7_ctrl = MAPPOGateController(config_path=config_path)
    p7_ctrl.load_checkpoints("models/checkpoints")

    p8_ctrl = MAPPOCausalGateController(config_path=config_path)
    p8_ctrl.load_checkpoints("models/checkpoints")

    p8r_ctrl = MAPPOCausalImprovedController(config_path=config_path)
    p8r_ctrl.load_checkpoints("models/checkpoints")
    print("All 5 controllers successfully instantiated and loaded.")

    controllers = [
        ("Phase 4: Frozen MAPPO", p4_ctrl, "mappo"),
        ("Phase 6: Frozen GAT", p6_ctrl, "gat"),
        ("Phase 7: Frozen Comm Gate", p7_ctrl, "p7_gate"),
        ("Phase 8: Original Causal Gate", p8_ctrl, "p8_causal"),
        ("Phase 8R: Improved Causal Comm", p8r_ctrl, "p8r_improved")
    ]

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
        "mean_C_ij",
        "mean_gate"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

    all_results = {name: [] for name, _, _ in controllers}
    env = MultiAgentTrafficEnv(config_path=config_path)

    try:
        for ctrl_name, controller, ctrl_type in controllers:
            print(f"\nEvaluating: {ctrl_name}")
            print(f"{'Seed':<6} | {'Delay (s)':<10} | {'Queue':<8} | {'Throughput':<10} | {'Task Ret':<10} | {'Comm Cost':<10} | {'Comm Ratio':<12} | {'Mean C_ij':<10} | {'Mean Gate':<10}")
            print("-" * 115)

            for ep_idx, seed in enumerate(eval_seeds, start=1):
                obs, info = env.reset(seed=seed)

                # Reset histories appropriately
                if ctrl_type in ["p8_causal", "p8r_improved"]:
                    controller.reset_history(obs, info)
                elif ctrl_type in ["gat", "p7_gate"]:
                    controller.reset_history(obs)

                ep_task_rewards = {a: 0.0 for a in env.agents}
                ep_comm_costs = {a: 0.0 for a in env.agents}
                ep_queues = []
                ep_waiting = []
                ep_throughput = 0
                ep_actual_comm = 0
                ep_possible_comm = steps_per_episode * len(ALLOWED_DIRECTED_EDGES)  # 800

                step_c_scores = []
                step_gate_values = []

                for step in range(steps_per_episode):
                    if ctrl_type == "mappo":
                        # Decentralized action selection (strictly local obs + agent ID)
                        actions, _ = controller.get_actions(obs, deterministic=True)
                        actual_comm = 0
                        comm_costs = {a: 0.0 for a in env.agents}
                    elif ctrl_type == "gat":
                        # GAT full physical mesh communication (all 8 physical channels active)
                        actions, _ = controller.get_actions(deterministic=True)
                        actual_comm = 8
                        # 0.01 per channel * 8 channels = 0.08 per step, divided among 4 agents
                        comm_costs = {a: (0.01 * 8) / 4.0 for a in env.agents}
                        step_gate_values.extend([1.0] * 8)
                    elif ctrl_type == "p7_gate":
                        # Phase 7 Heuristic context c_ij
                        context = controller.compute_heuristic_context(info)
                        actions, log_probs, comm_costs, actual_comm = controller.get_actions(context, deterministic=True)
                        for edge in ALLOWED_DIRECTED_EDGES:
                            g_val = controller.gated_gat.comm_gate.last_gate_values.get(edge, 0.5)
                            step_gate_values.append(g_val)
                    elif ctrl_type == "p8_causal":
                        # Phase 8 Causal context C_ij
                        context = controller.compute_causal_context(info)
                        step_c_scores.extend(list(context.values()))
                        actions, log_probs, comm_costs, actual_comm = controller.get_actions(context, deterministic=True)
                        for edge in ALLOWED_DIRECTED_EDGES:
                            g_val = controller.gated_gat.comm_gate.last_gate_values.get(edge, 0.5)
                            step_gate_values.append(g_val)
                    elif ctrl_type == "p8r_improved":
                        # Phase 8R Improved Causal Communication
                        context = controller.compute_causal_context(info)
                        step_c_scores.extend(list(context.values()))
                        actions, log_probs, comm_costs, actual_comm = controller.get_actions(context, deterministic=True)
                        for edge in ALLOWED_DIRECTED_EDGES:
                            g_val = controller.gated_gat.comm_gate.last_gate_values.get(edge, 0.5)
                            step_gate_values.append(g_val)
                    else:
                        raise ValueError(f"Unknown controller type: {ctrl_type}")

                    ep_actual_comm += actual_comm

                    # Environment transition
                    next_obs, task_rewards, terminations, truncations, next_info = env.step(actions)

                    for a in env.agents:
                        ep_task_rewards[a] += task_rewards[a]
                        ep_comm_costs[a] += comm_costs[a]

                    step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
                    step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
                    ep_queues.append(step_q)
                    ep_waiting.append(step_w)
                    ep_throughput += next_info["A"]["arrived_vehicles"]

                    # Update history
                    if ctrl_type in ["gat", "p7_gate", "p8_causal", "p8r_improved"]:
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
                    "mean_gate": mean_g_str
                }

                with open(summary_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=summary_fields)
                    writer.writerow(res_row)

                all_results[ctrl_name].append(res_row)

                print(
                    f"{seed:<6} | {mean_delay:<10.2f} | {mean_queue:<8.2f} | {ep_throughput:<10} | {task_ret:<10.2f} | "
                    f"{tot_comm_cost:<10.2f} | {comm_ratio * 100:<11.2f}% | {mean_c_str:<10} | {mean_g_str:<10}"
                )

    finally:
        env.close()

    # Generate Comparative Visualization
    ctrl_names = [c[0] for c in controllers]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Metric 1: Average Delay (s)
    delays_mean = [np.mean([r["mean_delay"] for r in all_results[c]]) for c in ctrl_names]
    delays_std = [np.std([r["mean_delay"] for r in all_results[c]]) for c in ctrl_names]
    bars1 = ax1.bar(range(len(ctrl_names)), delays_mean, yerr=delays_std, capsize=5, color=colors, alpha=0.85)
    ax1.set_title("Average Vehicle Delay (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Delay (seconds / veh)")
    ax1.set_xticks(range(len(ctrl_names)))
    ax1.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=15, ha="right", fontsize=9)
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
    ax2.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=15, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Metric 3: Task Return vs Adjusted Return
    ret_mean = [np.mean([r["task_return"] for r in all_results[c]]) for c in ctrl_names]
    ret_std = [np.std([r["task_return"] for r in all_results[c]]) for c in ctrl_names]
    bars3 = ax3.bar(range(len(ctrl_names)), ret_mean, yerr=ret_std, capsize=5, color=colors, alpha=0.85)
    ax3.set_title("Aggregate Task Return (Higher is Better)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Task Return")
    ax3.set_xticks(range(len(ctrl_names)))
    ax3.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=15, ha="right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval - 25, f"{yval:.1f}", ha="center", va="top", fontsize=8, fontweight="bold")

    # Metric 4: Communication Ratio (%)
    comm_mean = [np.mean([r["comm_ratio"] * 100 for r in all_results[c]]) for c in ctrl_names]
    bars4 = ax4.bar(range(len(ctrl_names)), comm_mean, color=colors, alpha=0.85)
    ax4.set_title("Communication Ratio (%) (Lower is More Efficient)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Communication Ratio (%)")
    ax4.set_ylim(-5, 115)
    ax4.set_xticks(range(len(ctrl_names)))
    ax4.set_xticklabels([c.split(":")[0] for c in ctrl_names], rotation=15, ha="right", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\nComparative evaluation plot successfully saved to {plot_file}")


if __name__ == "__main__":
    evaluate_all()
