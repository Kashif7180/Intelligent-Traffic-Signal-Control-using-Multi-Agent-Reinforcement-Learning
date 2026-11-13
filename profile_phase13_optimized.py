"""
SAGE-Traffic Phase 13 Step 3: Post-Optimization Performance Profiling Suite
===========================================================================
Measures environment performance and TraCI overhead post-optimization.
Saves comparison to logs/phase13_performance/optimized_profile.json.
"""

import os
import sys
import json
import time
import numpy as np
import torch

from network.topology import PHYSICAL_INTERSECTIONS
from environment.traffic_env import MultiAgentTrafficEnv
from visualize_phase10_gui import Phase10UnifiedGUIDashboard
import tkinter as tk

OUTPUT_JSON = "logs/phase13_performance/optimized_profile.json"


def profile_optimized_sumo(num_steps=50):
    """Profile SUMO TraCI environment step and component breakdown post-optimization."""
    print("\n--- PROFILING OPTIMIZED SUMO ENVIRONMENT (50 steps) ---")
    env = MultiAgentTrafficEnv("config.yaml", gui=False)
    obs, info = env.reset(seed=1001, options={"safety_enabled": True})

    total_step_times = []
    simulation_step_times = []
    obs_calc_times = []
    reward_calc_times = []
    metrics_query_times = []
    actions = {a: 0 for a in PHYSICAL_INTERSECTIONS}

    for s in range(num_steps):
        t_start = time.perf_counter()

        val_actions = env.safety_layer.validate_actions(actions, env.signals, env.current_step * 5.0, env.current_step)
        for a, act in val_actions.items():
            env.signals[a].apply_action(act)

        # Simulation advancement
        adv0 = time.perf_counter()
        for _ in range(env.step_duration):
            for sig in env.signals.values():
                sig.update_second()
            import traci
            traci.simulationStep()
            env.safety_layer.audit_all_physical_states(traci.simulation.getTime())
        adv1 = time.perf_counter()
        simulation_step_times.append((adv1 - adv0) * 1000.0)

        env.current_step += 1

        # Optimized single-pass query & calculation
        m0 = time.perf_counter()
        agent_metrics = {a: env.signals[a].get_metrics() for a in env.agents}
        m1 = time.perf_counter()
        metrics_query_times.append((m1 - m0) * 1000.0)

        o0 = time.perf_counter()
        for a in env.agents:
            _ = env.signals[a].compute_observation(env.features, env.normalization, metrics=agent_metrics[a])
        o1 = time.perf_counter()
        obs_calc_times.append((o1 - o0) * 1000.0)

        r0 = time.perf_counter()
        for a in env.agents:
            _ = env.signals[a].compute_reward(env.reward_cfg, metrics=agent_metrics[a])
        r1 = time.perf_counter()
        reward_calc_times.append((r1 - r0) * 1000.0)

        t_end = time.perf_counter()
        total_step_times.append((t_end - t_start) * 1000.0)

    env.close()

    mean_total = float(np.mean(total_step_times))
    mean_sim = float(np.mean(simulation_step_times))
    mean_obs = float(np.mean(obs_calc_times))
    mean_rew = float(np.mean(reward_calc_times))
    mean_metrics = float(np.mean(metrics_query_times))

    steps_per_sec = 1000.0 / mean_total
    sim_sec_per_sec = steps_per_sec * 5.0

    results = {
        "mean_step_time_ms": round(mean_total, 3),
        "steps_per_second": round(steps_per_sec, 2),
        "simulated_seconds_per_wall_second": round(sim_sec_per_sec, 2),
        "breakdown": {
            "traci_advancement_and_physics_ms": round(mean_sim, 3),
            "single_pass_metrics_query_ms": round(mean_metrics, 3),
            "reused_observation_generation_ms": round(mean_obs, 3),
            "reused_reward_calculation_ms": round(mean_rew, 3),
            "total_query_portions_ms": round(mean_metrics + mean_obs + mean_rew, 3)
        }
    }

    print(f"  Optimized Step Latency: {mean_total:.2f} ms | SPS: {steps_per_sec:.1f} steps/s ({sim_sec_per_sec:.1f}x real-time)")
    print(f"  --> TraCI advancement (5 sim sec): {mean_sim:.2f} ms ({mean_sim/mean_total*100:.1f}%)")
    print(f"  --> Single-pass TraCI Metrics query: {mean_metrics:.2f} ms ({mean_metrics/mean_total*100:.1f}%)")
    print(f"  --> In-memory Obs computation (0 TraCI IPC): {mean_obs:.4f} ms ({mean_obs/mean_total*100:.2f}%)")
    print(f"  --> In-memory Reward computation (0 TraCI IPC): {mean_rew:.4f} ms ({mean_rew/mean_total*100:.2f}%)")
    print(f"  --> TOTAL Query portion: {mean_metrics + mean_obs + mean_rew:.2f} ms ({(mean_metrics + mean_obs + mean_rew)/mean_total*100:.1f}%)")

    return results


def profile_optimized_gui(num_steps=20):
    """Profile GUI decision step latency post-optimization."""
    print("\n--- PROFILING OPTIMIZED GUI & XAI PIPELINE (20 steps) ---")
    root = tk.Tk()
    root.withdraw()

    app = Phase10UnifiedGUIDashboard(
        root=root,
        config_path="config.yaml",
        initial_controller_id="C9",
        initial_seed=1001,
        max_steps=num_steps,
        use_sumo_gui=False,
        smoke_test=True
    )

    total_gui_times = []
    for s in range(num_steps):
        t0 = time.perf_counter()
        app.execute_decision_step()
        t1 = time.perf_counter()
        total_gui_times.append((t1 - t0) * 1000.0)

    app.env.close()
    root.destroy()

    mean_gui = float(np.mean(total_gui_times))
    p50_gui = float(np.percentile(total_gui_times, 50))
    p95_gui = float(np.percentile(total_gui_times, 95))

    results = {
        "mean_gui_step_ms": round(mean_gui, 3),
        "p50_ms": round(p50_gui, 3),
        "p95_ms": round(p95_gui, 3)
    }
    print(f"  Optimized GUI Step: Mean={mean_gui:.2f} ms | P50={p50_gui:.2f} ms | P95={p95_gui:.2f} ms")
    return results


def main():
    with open("logs/phase13_performance/baseline_profile.json", "r") as f:
        baseline = json.load(f)

    opt_sumo = profile_optimized_sumo()
    opt_gui = profile_optimized_gui()

    base_step = baseline["sumo_simulation"]["mean_step_time_ms"]
    opt_step = opt_sumo["mean_step_time_ms"]
    step_speedup = (base_step - opt_step) / base_step * 100.0

    base_queries = baseline["sumo_simulation"]["breakdown"]["sum_of_query_portions_ms"]
    opt_queries = opt_sumo["breakdown"]["total_query_portions_ms"]
    query_speedup = (base_queries - opt_queries) / base_queries * 100.0

    base_gui = baseline["gui_and_xai"]["mean_gui_step_ms"]
    opt_gui_lat = opt_gui["mean_gui_step_ms"]
    gui_speedup = (base_gui - opt_gui_lat) / base_gui * 100.0

    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step_latency_ms": {
            "baseline": base_step,
            "optimized": opt_step,
            "reduction_percent": round(step_speedup, 2)
        },
        "query_overhead_ms": {
            "baseline": base_queries,
            "optimized": opt_queries,
            "reduction_percent": round(query_speedup, 2)
        },
        "gui_step_latency_ms": {
            "baseline": base_gui,
            "optimized": opt_gui_lat,
            "reduction_percent": round(gui_speedup, 2)
        },
        "steps_per_second": {
            "baseline": baseline["sumo_simulation"]["steps_per_second"],
            "optimized": opt_sumo["steps_per_second"]
        },
        "optimized_sumo": opt_sumo,
        "optimized_gui": opt_gui
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n================================================================================")
    print("EMPIRICAL PERFORMANCE COMPARISON SUMMARY:")
    print("================================================================================")
    print(f"  Step Latency:        {base_step:.2f} ms -> {opt_step:.2f} ms ({step_speedup:.1f}% reduction)")
    print(f"  TraCI Query Portions: {base_queries:.2f} ms -> {opt_queries:.2f} ms ({query_speedup:.1f}% reduction)")
    print(f"  GUI Step Latency:    {base_gui:.2f} ms -> {opt_gui_lat:.2f} ms ({gui_speedup:.1f}% reduction)")
    print(f"  Throughput (SPS):    {baseline['sumo_simulation']['steps_per_second']:.1f} SPS -> {opt_sumo['steps_per_second']:.1f} SPS")
    print("================================================================================")
    print(f"Saved comparison to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

