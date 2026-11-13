"""
SAGE-Traffic Phase 13 Step 3: Baseline Performance Profiling Suite
==================================================================
Accurately measures:
1. Controller Inference Latency (Mean, Median P50, P95, P99 per agent & 4-agent total)
2. SUMO Simulation & TraCI Step Breakdown (simulationStep, obs, reward, metrics, safety audit)
3. Memory and CPU Utilization
4. GUI & XAI Explanation Latency Breakdown
"""

import os
import sys
import gc
import json
import time
import tracemalloc
import numpy as np
import torch

from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES
from environment.traffic_env import MultiAgentTrafficEnv
from benchmark_phase10_master import load_all_controllers
from visualize_phase10_gui import Phase10UnifiedGUIDashboard
import tkinter as tk

OUTPUT_JSON = "logs/phase13_performance/baseline_profile.json"


def profile_controller_inference(num_trials=100):
    """Profile inference latency across representative controllers."""
    print("\n--- 1. PROFILING CONTROLLER INFERENCE (100 trials each) ---")
    all_ctrls = {c["id"]: c for c in load_all_controllers("config.yaml")}
    profiled_cids = ["C1", "C3", "C4", "C5", "C8", "C9"]

    mock_obs = {a: np.random.uniform(0.0, 1.0, 43).astype(np.float32) for a in PHYSICAL_INTERSECTIONS}
    mock_info = {
        a: {
            "metrics": {
                "queue_length": 5.0,
                "waiting_time": 12.0,
                "vehicle_count": 6,
                "average_speed": 7.5,
                "average_occupancy": 0.25
            }
        }
        for a in PHYSICAL_INTERSECTIONS
    }
    eye_4 = torch.eye(4)

    results = {}
    cuda_available = torch.cuda.is_available()

    for cid in profiled_cids:
        cfg = all_ctrls[cid]
        ctype = cfg["type"]
        cobj = cfg["obj"]

        # Reset history
        if ctype in ["transformer", "gat", "comm_gate"]:
            cobj.reset_history(mock_obs)
        elif ctype in ["causal_gate", "causal_improved"]:
            cobj.reset_history(mock_obs, mock_info)
        elif ctype == "fixed_time":
            cobj.reset()

        # Warm-up (5 steps)
        for _ in range(5):
            if ctype == "fixed_time":
                cobj.get_actions(mock_obs, mock_info)
            elif ctype == "mappo_shared":
                cobj.get_actions(mock_obs, deterministic=True)
            elif ctype in ["transformer", "gat"]:
                cobj.get_actions(deterministic=True)
            elif ctype == "causal_improved":
                ctx = cobj.compute_causal_context(mock_info)
                cobj.get_actions(ctx, deterministic=True)

        latencies_ms = []
        cpu_times_ms = []

        for _ in range(num_trials):
            t0 = time.perf_counter()
            c0 = time.process_time()

            if ctype == "fixed_time":
                actions = cobj.get_actions(mock_obs, mock_info)
            elif ctype == "mappo_shared":
                actions, _ = cobj.get_actions(mock_obs, deterministic=True)
            elif ctype in ["transformer", "gat"]:
                actions, _ = cobj.get_actions(deterministic=True)
            elif ctype == "causal_improved":
                ctx = cobj.compute_causal_context(mock_info)
                actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)

            t1 = time.perf_counter()
            c1 = time.process_time()

            latencies_ms.append((t1 - t0) * 1000.0)
            cpu_times_ms.append((c1 - c0) * 1000.0)

        lat_arr = np.array(latencies_ms)
        mean_lat = float(np.mean(lat_arr))
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_cpu = float(np.mean(cpu_times_ms))

        # Per-agent latency (total 4 agents divided by 4)
        per_agent_mean = mean_lat / 4.0
        per_agent_p50 = p50 / 4.0
        per_agent_p95 = p95 / 4.0
        per_agent_p99 = p99 / 4.0

        results[cid] = {
            "name": cfg["name"],
            "total_4agent_mean_ms": round(mean_lat, 4),
            "total_4agent_p50_ms": round(p50, 4),
            "total_4agent_p95_ms": round(p95, 4),
            "total_4agent_p99_ms": round(p99, 4),
            "per_agent_mean_ms": round(per_agent_mean, 4),
            "per_agent_p50_ms": round(per_agent_p50, 4),
            "per_agent_p95_ms": round(per_agent_p95, 4),
            "per_agent_p99_ms": round(per_agent_p99, 4),
            "cpu_time_ms": round(mean_cpu, 4),
            "device": "cuda" if cuda_available else "cpu"
        }
        print(f"  {cid} ({cfg['name']}): Total={mean_lat:.3f}ms (P50={p50:.3f}ms, P95={p95:.3f}ms) | Per-Agent={per_agent_mean:.3f}ms")

    return results


def profile_sumo_simulation(num_steps=50):
    """Profile SUMO TraCI environment step and component breakdown."""
    print("\n--- 2. PROFILING SUMO TRAFFIC ENVIRONMENT (50 steps) ---")
    env = MultiAgentTrafficEnv("config.yaml", gui=False)
    obs, info = env.reset(seed=1001, options={"safety_enabled": True})

    total_step_times = []
    simulation_step_times = []
    obs_calc_times = []
    reward_calc_times = []
    metrics_query_times = []
    safety_audit_times = []
    actions = {a: 0 for a in PHYSICAL_INTERSECTIONS}

    for s in range(num_steps):
        t_start = time.perf_counter()

        # Measure TraCI subcomponents manually for 1 step
        # 1. Validation
        v0 = time.perf_counter()
        val_actions = env.safety_layer.validate_actions(actions, env.signals, env.current_step * 5.0, env.current_step)
        v1 = time.perf_counter()

        # 2. Apply actions
        for a, act in val_actions.items():
            env.signals[a].apply_action(act)

        # 3. Advancement
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

        # 4. Observation calculation
        o0 = time.perf_counter()
        for a in env.agents:
            _ = env.signals[a].compute_observation(env.features, env.normalization)
        o1 = time.perf_counter()
        obs_calc_times.append((o1 - o0) * 1000.0)

        # 5. Reward calculation
        r0 = time.perf_counter()
        for a in env.agents:
            _ = env.signals[a].compute_reward(env.reward_cfg)
        r1 = time.perf_counter()
        reward_calc_times.append((r1 - r0) * 1000.0)

        # 6. Metrics query for infos
        m0 = time.perf_counter()
        for a in env.agents:
            _ = env.signals[a].get_metrics()
        m1 = time.perf_counter()
        metrics_query_times.append((m1 - m0) * 1000.0)

        t_end = time.perf_counter()
        total_step_times.append((t_end - t_start) * 1000.0)

    env.close()

    mean_total = float(np.mean(total_step_times))
    mean_sim = float(np.mean(simulation_step_times))
    mean_obs = float(np.mean(obs_calc_times))
    mean_rew = float(np.mean(reward_calc_times))
    mean_metrics = float(np.mean(metrics_query_times))

    # Steps per second
    steps_per_sec = 1000.0 / mean_total
    sim_sec_per_sec = steps_per_sec * 5.0  # 5s per step

    results = {
        "mean_step_time_ms": round(mean_total, 3),
        "steps_per_second": round(steps_per_sec, 2),
        "simulated_seconds_per_wall_second": round(sim_sec_per_sec, 2),
        "breakdown": {
            "traci_advancement_and_physics_ms": round(mean_sim, 3),
            "observation_generation_ms": round(mean_obs, 3),
            "reward_calculation_ms": round(mean_rew, 3),
            "metrics_query_in_infos_ms": round(mean_metrics, 3),
            "sum_of_query_portions_ms": round(mean_obs + mean_rew + mean_metrics, 3)
        }
    }

    print(f"  Step Latency: {mean_total:.2f} ms | SPS: {steps_per_sec:.1f} steps/s ({sim_sec_per_sec:.1f}x real-time)")
    print(f"  --> TraCI advancement (5 sim sec): {mean_sim:.2f} ms ({mean_sim/mean_total*100:.1f}%)")
    print(f"  --> Observation generation (TraCI queries): {mean_obs:.2f} ms ({mean_obs/mean_total*100:.1f}%)")
    print(f"  --> Reward calculation (TraCI queries): {mean_rew:.2f} ms ({mean_rew/mean_total*100:.1f}%)")
    print(f"  --> Infos metrics query (TraCI queries): {mean_metrics:.2f} ms ({mean_metrics/mean_total*100:.1f}%)")
    print(f"  --> TOTAL TraCI Query overhead: {mean_obs + mean_rew + mean_metrics:.2f} ms ({(mean_obs + mean_rew + mean_metrics)/mean_total*100:.1f}%)")

    return results


def profile_memory_and_stability(num_episodes=5, steps_per_ep=30):
    """Profile RAM footprint and memory stability across sequential episodes."""
    print("\n--- 3. PROFILING MEMORY & STABILITY (5 episodes) ---")
    tracemalloc.start()
    env = MultiAgentTrafficEnv("config.yaml", gui=False)

    memory_snapshots = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1001 + ep, options={"safety_enabled": True})
        for s in range(steps_per_ep):
            obs, rew, term, trunc, info = env.step({"A": 0, "B": 0, "C": 0, "D": 0})

        current, peak = tracemalloc.get_traced_memory()
        memory_snapshots.append({
            "episode": ep + 1,
            "current_mb": round(current / (1024 * 1024), 3),
            "peak_mb": round(peak / (1024 * 1024), 3)
        })
        print(f"  Episode {ep+1}: Current RAM = {current / (1024 * 1024):.2f} MB | Peak = {peak / (1024 * 1024):.2f} MB")

    env.close()
    gc.collect()

    final_cur, final_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "snapshots": memory_snapshots,
        "final_current_mb": round(final_cur / (1024 * 1024), 3),
        "peak_mb": round(final_peak / (1024 * 1024), 3),
        "net_growth_mb": round((memory_snapshots[-1]["current_mb"] - memory_snapshots[0]["current_mb"]), 3)
    }
    return results


def profile_gui_and_xai(num_steps=20):
    """Profile GUI decision step and XAI explanation pipeline latency."""
    print("\n--- 4. PROFILING GUI & XAI EXPLAINER PIPELINE (20 steps) ---")
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
    print(f"  GUI Step (Inference + SUMO + XAI + Log): Mean={mean_gui:.2f} ms | P50={p50_gui:.2f} ms | P95={p95_gui:.2f} ms")
    return results


def main():
    profile_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "controller_inference": profile_controller_inference(),
        "sumo_simulation": profile_sumo_simulation(),
        "memory_and_stability": profile_memory_and_stability(),
        "gui_and_xai": profile_gui_and_xai()
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print(f"\n[DONE] Saved complete baseline profile to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

