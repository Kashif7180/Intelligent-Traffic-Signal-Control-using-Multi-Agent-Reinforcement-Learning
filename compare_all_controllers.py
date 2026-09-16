"""
Fair, Rigorous 3-Way Comparative Evaluation for SAGE-Traffic (Phase 4):
Fixed-Time Baseline vs. Independent PPO (IPPO) vs. Multi-Agent PPO (MAPPO).

Evaluates all three controllers under identical conditions across seeds 1001-1010:
- Same 2x2 grid network
- Same normal demand profile & dynamic route generation
- Same 100 decision steps (500 SUMO seconds per episode)
- Same TraCI environment & metric definitions
- IPPO uses existing Phase 3 checkpoints (agent_A.pt, agent_B.pt, agent_C.pt, agent_D.pt)
- MAPPO uses trained Phase 4 checkpoints (mappo_actor_shared.pt, decentralized execution)
- Deterministic action selection (greedy argmax) for both RL controllers
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
from models.fixed_time_controller import FixedTimeController
from models.ppo_network import IndependentPPOAgent
from models.mappo_network import MAPPOController


def evaluate_controller(
    env: MultiAgentTrafficEnv,
    controller_type: str,
    seeds: list,
    steps_per_episode: int,
    ippo_agents=None,
    mappo_ctrl=None
):
    """
    Evaluate a controller over seeds 1001-1010 under identical conditions.
    """
    fixed_time_ctrl = FixedTimeController("config.yaml") if controller_type == "Fixed-Time" else None
    episode_results = []

    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        if fixed_time_ctrl:
            fixed_time_ctrl.reset()

        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            # Controller action selection
            if controller_type == "Fixed-Time":
                actions = fixed_time_ctrl.get_actions(obs, info)
            elif controller_type == "IPPO":
                actions = {}
                for a in env.agents:
                    act, _, _ = ippo_agents[a].get_action_and_value(obs[a], deterministic=True)
                    actions[a] = act
            elif controller_type == "MAPPO":
                # Decentralized action selection (strictly local obs + agent ID)
                actions, _ = mappo_ctrl.get_actions(obs, deterministic=True)
            else:
                raise ValueError(f"Unknown controller type: {controller_type}")

            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            # Metric tracking
            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            obs = next_obs
            info = next_info
            if any(terminations.values()) or any(truncations.values()):
                break

        ep_data = {
            "episode": ep_idx,
            "seed": seed,
            "controller": controller_type,
            "return_A": ep_rewards["A"],
            "return_B": ep_rewards["B"],
            "return_C": ep_rewards["C"],
            "return_D": ep_rewards["D"],
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_waiting": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        }
        episode_results.append(ep_data)

    return episode_results


def plot_3way_comparison(all_results: dict, output_path: str):
    """Generate high-resolution 3-way comparative plots."""
    controllers = ["Fixed-Time", "IPPO", "MAPPO"]
    colors = ["#f38ba8", "#89b4fa", "#a6e3a1"]

    metrics = [
        ("mean_waiting", "Average Delay (Waiting Time / veh)", "Seconds", [all_results[c]["waits"] for c in controllers]),
        ("mean_queue", "Average Queue Length", "Vehicles", [all_results[c]["queues"] for c in controllers]),
        ("throughput", "Throughput (Arrived Vehicles)", "Vehicles", [all_results[c]["throughputs"] for c in controllers]),
        ("agg_return", "Aggregate Return", "Sum of Rewards", [all_results[c]["returns"] for c in controllers])
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")

    for ax, (key, title, ylabel, data_lists) in zip(axs.flat, metrics):
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#585b70", axis="y")

        means = [np.mean(d) for d in data_lists]
        stds = [np.std(d) for d in data_lists]
        x_pos = np.arange(len(controllers))

        bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="#cdd6f4", lw=1.2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(controllers, fontsize=11, fontweight="bold", color="#cdd6f4")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Label bars with value
        for bar, m, s in zip(bars, means, stds):
            h = bar.get_height()
            y_text = h + (s if h >= 0 else -s - abs(h)*0.08)
            va = "bottom" if h >= 0 else "top"
            ax.annotate(f"{m:.1f} ± {s:.1f}",
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
    steps_per_episode = int(config["ppo"].get("steps_per_episode", 100))
    eval_seeds = [1000 + i for i in range(1, eval_episodes + 1)]

    csv_file = "experiments/evaluation_comparison_3way.csv"
    plot_file = "experiments/controller_comparison_3way.png"

    print("=" * 95)
    print("SAGE-TRAFFIC PHASE 4: RIGOROUS 3-WAY COMPARATIVE EVALUATION")
    print("Fixed-Time Baseline vs. Independent PPO (IPPO) vs. Multi-Agent PPO (MAPPO)")
    print("=" * 95)
    print(f"Evaluation Seeds:      {eval_seeds} (10 Paired Seeds)")
    print(f"Steps per Episode:     {steps_per_episode} (500s SUMO simulation time per episode)")
    print(f"Traffic Demand:        {config['demand']['active_demand']} (Normal demand profile)")
    print(f"IPPO Checkpoints:      models/checkpoints/agent_{{A,B,C,D}}.pt")
    print(f"MAPPO Checkpoints:     models/checkpoints/mappo_actor_shared.pt")
    print("-" * 95)

    env = MultiAgentTrafficEnv(config_path=config_path)

    # 1. Load trained IPPO agents (Phase 3)
    obs_dim = env.observation_space("A").shape[0]
    act_dim = env.action_space("A").n
    ippo_agents = {}
    for a in env.agents:
        agent = IndependentPPOAgent(agent_id=a, obs_dim=obs_dim, act_dim=act_dim, ppo_cfg=config["ppo"])
        ckpt_path = f"models/checkpoints/agent_{a}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing IPPO checkpoint: {ckpt_path}")
        agent.load_checkpoint(ckpt_path)
        ippo_agents[a] = agent
    print("[1/3] Phase 3 IPPO checkpoints loaded successfully.")

    # 2. Load trained MAPPO controller (Phase 4)
    mappo_ctrl = MAPPOController(config_path=config_path)
    mappo_ckpt = "models/checkpoints/mappo_actor_shared.pt"
    if not os.path.exists(mappo_ckpt):
        raise FileNotFoundError(f"Missing MAPPO checkpoint: {mappo_ckpt}")
    mappo_ctrl.load_checkpoints("models/checkpoints")
    print("[2/3] Phase 4 MAPPO checkpoints loaded successfully.")

    try:
        # Run evaluations
        print("\n[EVAL 1/3] Evaluating Fixed-Time Baseline (Seeds 1001-1010)...")
        ft_results = evaluate_controller(env, "Fixed-Time", eval_seeds, steps_per_episode)
        print("  Fixed-Time evaluation completed.")

        print("\n[EVAL 2/3] Evaluating Independent PPO (Seeds 1001-1010)...")
        ippo_results = evaluate_controller(env, "IPPO", eval_seeds, steps_per_episode, ippo_agents=ippo_agents)
        print("  IPPO evaluation completed.")

        print("\n[EVAL 3/3] Evaluating Multi-Agent PPO (Seeds 1001-1010)...")
        mappo_results = evaluate_controller(env, "MAPPO", eval_seeds, steps_per_episode, mappo_ctrl=mappo_ctrl)
        print("  MAPPO evaluation completed.")

        # Save to CSV
        os.makedirs("experiments", exist_ok=True)
        fieldnames = [
            "seed", "controller", "return_A", "return_B", "return_C", "return_D",
            "agg_return", "mean_queue", "mean_waiting", "throughput"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in ft_results + ippo_results + mappo_results:
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
        print(f"\nSaved all 30 evaluation episode records to {csv_file}")

        # Aggregate Statistics
        results_by_ctrl = {
            "Fixed-Time": {
                "queues": [r["mean_queue"] for r in ft_results],
                "waits": [r["mean_waiting"] for r in ft_results],
                "throughputs": [r["throughput"] for r in ft_results],
                "returns": [r["agg_return"] for r in ft_results]
            },
            "IPPO": {
                "queues": [r["mean_queue"] for r in ippo_results],
                "waits": [r["mean_waiting"] for r in ippo_results],
                "throughputs": [r["throughput"] for r in ippo_results],
                "returns": [r["agg_return"] for r in ippo_results]
            },
            "MAPPO": {
                "queues": [r["mean_queue"] for r in mappo_results],
                "waits": [r["mean_waiting"] for r in mappo_results],
                "throughputs": [r["throughput"] for r in mappo_results],
                "returns": [r["agg_return"] for r in mappo_results]
            }
        }

        # Print per-seed table
        print("\n" + "=" * 95)
        print(f"{'Seed':<6} | {'Metric':<14} | {'Fixed-Time':<18} | {'IPPO (Phase 3)':<18} | {'MAPPO (Phase 4)':<18}")
        print("-" * 95)
        for i, s in enumerate(eval_seeds):
            print(f"{s:<6} | {'Delay (s)':<14} | {ft_results[i]['mean_waiting']:<18.2f} | {ippo_results[i]['mean_waiting']:<18.2f} | {mappo_results[i]['mean_waiting']:<18.2f}")
            print(f"{'':<6} | {'Queue (veh)':<14} | {ft_results[i]['mean_queue']:<18.2f} | {ippo_results[i]['mean_queue']:<18.2f} | {mappo_results[i]['mean_queue']:<18.2f}")
            print(f"{'':<6} | {'Throughput':<14} | {ft_results[i]['throughput']:<18} | {ippo_results[i]['throughput']:<18} | {mappo_results[i]['throughput']:<18}")
            print(f"{'':<6} | {'Agg Return':<14} | {ft_results[i]['agg_return']:<18.2f} | {ippo_results[i]['agg_return']:<18.2f} | {mappo_results[i]['agg_return']:<18.2f}")
            if i < len(eval_seeds) - 1:
                print("-" * 95)
        print("=" * 95)

        # Summary Metrics
        stats = {}
        for c in ["Fixed-Time", "IPPO", "MAPPO"]:
            stats[c] = {
                "wait_mean": np.mean(results_by_ctrl[c]["waits"]),
                "wait_std": np.std(results_by_ctrl[c]["waits"]),
                "queue_mean": np.mean(results_by_ctrl[c]["queues"]),
                "queue_std": np.std(results_by_ctrl[c]["queues"]),
                "tp_mean": np.mean(results_by_ctrl[c]["throughputs"]),
                "tp_std": np.std(results_by_ctrl[c]["throughputs"]),
                "ret_mean": np.mean(results_by_ctrl[c]["returns"]),
                "ret_std": np.std(results_by_ctrl[c]["returns"])
            }

        print("\n" + "=" * 95)
        print("HEAD-TO-HEAD SUMMARY STATISTICS (Mean +/- Std Dev across 10 Seeds)")
        print("=" * 95)
        print(f"{'Metric':<24} | {'Fixed-Time':<20} | {'IPPO (Phase 3)':<20} | {'MAPPO (Phase 4)':<20}")
        print("-" * 95)
        print(f"{'Average Delay (s)':<24} | {stats['Fixed-Time']['wait_mean']:6.2f} +/- {stats['Fixed-Time']['wait_std']:<11.2f} | {stats['IPPO']['wait_mean']:6.2f} +/- {stats['IPPO']['wait_std']:<11.2f} | {stats['MAPPO']['wait_mean']:6.2f} +/- {stats['MAPPO']['wait_std']:<11.2f}")
        print(f"{'Average Queue (veh)':<24} | {stats['Fixed-Time']['queue_mean']:6.2f} +/- {stats['Fixed-Time']['queue_std']:<11.2f} | {stats['IPPO']['queue_mean']:6.2f} +/- {stats['IPPO']['queue_std']:<11.2f} | {stats['MAPPO']['queue_mean']:6.2f} +/- {stats['MAPPO']['queue_std']:<11.2f}")
        print(f"{'Throughput (veh)':<24} | {stats['Fixed-Time']['tp_mean']:6.1f} +/- {stats['Fixed-Time']['tp_std']:<11.1f} | {stats['IPPO']['tp_mean']:6.1f} +/- {stats['IPPO']['tp_std']:<11.1f} | {stats['MAPPO']['tp_mean']:6.1f} +/- {stats['MAPPO']['tp_std']:<11.1f}")
        print(f"{'Aggregate Return':<24} | {stats['Fixed-Time']['ret_mean']:6.2f} +/- {stats['Fixed-Time']['ret_std']:<11.2f} | {stats['IPPO']['ret_mean']:6.2f} +/- {stats['IPPO']['ret_std']:<11.2f} | {stats['MAPPO']['ret_mean']:6.2f} +/- {stats['MAPPO']['ret_std']:<11.2f}")
        print("-" * 95)

        # Percentage Changes
        def pct_diff(new_val, old_val):
            return ((new_val - old_val) / abs(old_val)) * 100.0

        print("\nPERCENTAGE COMPARISONS:")
        print("-" * 95)
        print("1. IPPO vs. Fixed-Time Baseline:")
        print(f"   - Average Delay:       {pct_diff(stats['IPPO']['wait_mean'], stats['Fixed-Time']['wait_mean']):+6.2f}%")
        print(f"   - Average Queue:       {pct_diff(stats['IPPO']['queue_mean'], stats['Fixed-Time']['queue_mean']):+6.2f}%")
        print(f"   - Throughput:          {pct_diff(stats['IPPO']['tp_mean'], stats['Fixed-Time']['tp_mean']):+6.2f}%")
        print(f"   - Aggregate Return:    {pct_diff(stats['IPPO']['ret_mean'], stats['Fixed-Time']['ret_mean']):+6.2f}%")

        print("\n2. MAPPO vs. Fixed-Time Baseline:")
        print(f"   - Average Delay:       {pct_diff(stats['MAPPO']['wait_mean'], stats['Fixed-Time']['wait_mean']):+6.2f}%")
        print(f"   - Average Queue:       {pct_diff(stats['MAPPO']['queue_mean'], stats['Fixed-Time']['queue_mean']):+6.2f}%")
        print(f"   - Throughput:          {pct_diff(stats['MAPPO']['tp_mean'], stats['Fixed-Time']['tp_mean']):+6.2f}%")
        print(f"   - Aggregate Return:    {pct_diff(stats['MAPPO']['ret_mean'], stats['Fixed-Time']['ret_mean']):+6.2f}%")

        print("\n3. MAPPO vs. IPPO Baseline:")
        print(f"   - Average Delay:       {pct_diff(stats['MAPPO']['wait_mean'], stats['IPPO']['wait_mean']):+6.2f}%")
        print(f"   - Average Queue:       {pct_diff(stats['MAPPO']['queue_mean'], stats['IPPO']['queue_mean']):+6.2f}%")
        print(f"   - Throughput:          {pct_diff(stats['MAPPO']['tp_mean'], stats['IPPO']['tp_mean']):+6.2f}%")
        print(f"   - Aggregate Return:    {pct_diff(stats['MAPPO']['ret_mean'], stats['IPPO']['ret_mean']):+6.2f}%")
        print("-" * 95)

        # Honest Final Verdict
        delay_vs_ippo = pct_diff(stats['MAPPO']['wait_mean'], stats['IPPO']['wait_mean'])
        queue_vs_ippo = pct_diff(stats['MAPPO']['queue_mean'], stats['IPPO']['queue_mean'])
        print("\nHONEST SCIENTIFIC VERDICT:")
        if delay_vs_ippo < -2.0 and queue_vs_ippo < -2.0:
            print("  MAPPO outperforms Independent PPO (IPPO) across key traffic metrics (lower delay and queues).")
        elif delay_vs_ippo > 2.0 and queue_vs_ippo > 2.0:
            print("  Independent PPO (IPPO) outperforms MAPPO under this training budget and configuration.")
        else:
            print("  MAPPO and Independent PPO exhibit comparable traffic performance within variance.")

        # Plot
        print(f"\nGenerating 3-way comparison plot -> {plot_file}...")
        plot_3way_comparison(results_by_ctrl, plot_file)
        print("  Comparison plot saved successfully.")

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


if __name__ == "__main__":
    main()
