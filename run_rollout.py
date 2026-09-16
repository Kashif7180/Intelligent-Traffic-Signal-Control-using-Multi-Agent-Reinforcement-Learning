"""
Minimal verification script for Phase 1 of SAGE-Traffic.
Runs a 100-step random-action rollout across the 2x2 grid network
and logs per-intersection metrics (queue length, waiting time, vehicle count).
"""

import sys
import random
import yaml
from environment.traffic_env import TrafficEnv


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("SAGE-Traffic Phase 1: 100-Step Random Action Rollout Verification")
    print("=" * 70)
    print(f"Network: {config['network']['net_file']}")
    print(f"Intersections: {config['network']['intersections']}")
    print(f"Active Demand Profile: {config['demand']['active_demand']}")
    print(f"Step Duration: {config['simulation']['step_duration']}s per decision step")
    print(f"Total Control Steps: {config['simulation']['sim_max_steps']}")
    print(f"Yellow Clearance Duration: {config['simulation']['yellow_duration']}s")
    print("-" * 70)

    # Initialize environment
    env = TrafficEnv(config_path=config_path)
    rng = random.Random(config["simulation"]["seed"])

    try:
        # Reset environment
        print("Resetting environment and launching SUMO via TraCI...")
        initial_states = env.reset(seed=config["simulation"]["seed"])
        print("Environment initialized successfully. Beginning rollout...\n")

        print(f"{'Step':<6} | {'SimTime':<8} | {'Inter A (Q/W/V)':<18} | {'Inter B (Q/W/V)':<18} | {'Inter C (Q/W/V)':<18} | {'Inter D (Q/W/V)':<18}")
        print("-" * 105)

        total_steps = config["simulation"]["sim_max_steps"]
        step_records = []

        for step_idx in range(1, total_steps + 1):
            # Sample random action (0 or 1) for each intersection
            actions = {ts_id: rng.choice([0, 1]) for ts_id in env.intersections}

            states, rewards, terminated, truncated, info = env.step(actions)
            sim_time = info["A"]["sim_time"]
            active_veh = info["A"]["active_vehicles"]

            # Record metrics for summary
            step_data = {
                "step": step_idx,
                "sim_time": sim_time,
                "active_vehicles": active_veh,
                "intersections": {
                    ts_id: {
                        "queue": info[ts_id]["metrics"]["queue_length"],
                        "waiting": info[ts_id]["metrics"]["waiting_time"],
                        "count": info[ts_id]["metrics"]["vehicle_count"],
                        "speed": info[ts_id]["metrics"]["average_speed"]
                    }
                    for ts_id in env.intersections
                }
            }
            step_records.append(step_data)

            # Print every 10 steps, plus step 1, 50, and 100
            if step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps:
                fmt_a = f"{info['A']['metrics']['queue_length']}/{info['A']['metrics']['waiting_time']:.0f}/{info['A']['metrics']['vehicle_count']}"
                fmt_b = f"{info['B']['metrics']['queue_length']}/{info['B']['metrics']['waiting_time']:.0f}/{info['B']['metrics']['vehicle_count']}"
                fmt_c = f"{info['C']['metrics']['queue_length']}/{info['C']['metrics']['waiting_time']:.0f}/{info['C']['metrics']['vehicle_count']}"
                fmt_d = f"{info['D']['metrics']['queue_length']}/{info['D']['metrics']['waiting_time']:.0f}/{info['D']['metrics']['vehicle_count']}"
                print(f"{step_idx:<6} | {sim_time:<8.0f} | {fmt_a:<18} | {fmt_b:<18} | {fmt_c:<18} | {fmt_d:<18}")

            if any(terminated.values()) or any(truncated.values()):
                break

        print("-" * 105)
        print("\n" + "=" * 70)
        print("ROLLOUT COMPLETE: Summary Statistics (Averaged Over 100 Steps)")
        print("=" * 70)
        print(f"Total Simulation Time Elapsed: {step_records[-1]['sim_time']:.0f} seconds")
        print(f"Active Vehicles on Network at Step 100: {step_records[-1]['active_vehicles']}")
        print(f"Sample Final Step Reward (Agent A): {rewards['A']:.3f} (Delay + Queue Multi-Objective)")
        print("-" * 70)

        print(f"{'Intersection':<14} | {'Mean Queue (veh)':<18} | {'Mean Waiting Time (s)':<22} | {'Mean Veh Count':<16}")
        print("-" * 70)
        for ts_id in env.intersections:
            mean_q = sum(r["intersections"][ts_id]["queue"] for r in step_records) / len(step_records)
            mean_w = sum(r["intersections"][ts_id]["waiting"] for r in step_records) / len(step_records)
            mean_v = sum(r["intersections"][ts_id]["count"] for r in step_records) / len(step_records)
            print(f"{ts_id:<14} | {mean_q:<18.2f} | {mean_w:<22.2f} | {mean_v:<16.2f}")
        print("=" * 70)
        print("[SUCCESS] TraCI bidirectional control and metric extraction verified.")

    finally:
        env.close()
        print("TraCI environment closed cleanly.\n")


if __name__ == "__main__":
    main()
