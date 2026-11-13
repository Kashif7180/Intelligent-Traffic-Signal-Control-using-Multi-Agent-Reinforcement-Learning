"""
Phase 13 Stress Benchmark Runner: Evaluates C1, C3, C5, C8, C9 across stress scenarios.
Records detailed telemetry to logs/phase13_stress/phase13_stress_benchmark.csv.
"""

import os
import sys
import csv
import time
import numpy as np
import torch

from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES
from environment.traffic_env import MultiAgentTrafficEnv
from benchmark_phase10_master import load_all_controllers

OUTPUT_CSV = "logs/phase13_stress/phase13_stress_benchmark.csv"
STRESS_ROUTE_DIR = "network/route_cache_phase13_stress"

def run_stress_benchmarks():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    all_ctrls = {c["id"]: c for c in load_all_controllers("config.yaml")}
    tested_cids = ["C1", "C3", "C5", "C8", "C9"]

    scenarios = [
        ("Empty", "routes_empty.rou.xml", [4001], 10),
        ("Extreme_HyperSaturation", "routes_extreme_seed_{seed}.rou.xml", [4001, 4002], 20),
        ("Asymmetric_Bottleneck", "routes_asymmetric_seed_{seed}.rou.xml", [4001, 4002], 20),
        ("Dynamic_Shock_Surge", "routes_surge_seed_{seed}.rou.xml", [4001], 30),
        ("Gridlock_Dissipation", "routes_dissipation_seed_{seed}.rou.xml", [4001], 30)
    ]

    fieldnames = [
        "scenario", "controller_id", "controller_name", "seed", "steps",
        "mean_delay", "mean_queue", "throughput", "active_vehicles",
        "total_interventions", "physical_conflicts", "mean_entropy", "safety_enabled"
    ]

    results = []

    print("================================================================================")
    print("SAGE-TRAFFIC PHASE 13 STEP 2: CONTROLLED STRESS BENCHMARK RUNNER")
    print("================================================================================")

    for scen_name, r_template, seeds, steps in scenarios:
        for seed in seeds:
            r_filename = r_template.format(seed=seed)
            r_path = os.path.join(STRESS_ROUTE_DIR, r_filename)
            assert os.path.exists(r_path), f"Route file not found: {r_path}"

            for cid in tested_cids:
                cfg = all_ctrls[cid]
                ctype = cfg["type"]
                cobj = cfg["obj"]
                safety_enabled = (cid == "C9")

                env = MultiAgentTrafficEnv("config.yaml", gui=False)
                try:
                    obs, info = env.reset(seed=seed, options={"route_file": r_path, "safety_enabled": safety_enabled})

                    if ctype in ["transformer", "gat", "comm_gate"]:
                        cobj.reset_history(obs)
                    elif ctype in ["causal_gate", "causal_improved"]:
                        cobj.reset_history(obs, info)
                    elif ctype == "fixed_time":
                        cobj.reset()

                    step_entropies = []
                    eye_4 = torch.eye(4)

                    for step in range(steps):
                        # Infer actions
                        entropy = 0.0
                        if ctype == "fixed_time":
                            actions = cobj.get_actions(obs, info)
                            entropy = 0.0
                        elif ctype == "mappo_shared":
                            actions, _ = cobj.get_actions(obs, deterministic=True)
                            with torch.no_grad():
                                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                                    x = np.concatenate([obs[a], eye_4[i].numpy()], axis=-1)
                                    dist = cobj.shared_actor(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
                                    entropy += float(dist.entropy().item())
                            entropy /= 4.0
                        elif ctype == "gat":
                            actions, _ = cobj.get_actions(deterministic=True)
                            with torch.no_grad():
                                hist_list = [cobj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                                hist_t = torch.as_tensor(np.stack(hist_list), dtype=torch.float32)
                                emb = cobj.transformer_encoder(hist_t)
                                gat_emb = cobj.gat(emb.unsqueeze(0)).squeeze(0)
                                dist = cobj.actor(gat_emb, eye_4)
                                entropy = float(dist.entropy().mean().item())
                        elif ctype == "causal_improved":
                            ctx = cobj.compute_causal_context(info)
                            actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)
                            with torch.no_grad():
                                hist_list = [cobj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                                hist_t = torch.as_tensor(np.stack(hist_list), dtype=torch.float32)
                                emb = cobj.transformer_encoder(hist_t)
                                gat_res = cobj.gated_gat(emb.unsqueeze(0), ctx)
                                gat_emb = gat_res[0].squeeze(0)
                                dist = cobj.actor(gat_emb, eye_4)
                                entropy = float(dist.entropy().mean().item())

                        step_entropies.append(entropy)
                        obs, rew, term, trunc, info = env.step(actions)

                        if ctype in ["transformer", "gat", "causal_improved"]:
                            cobj.update_history(obs)

                    mean_delay = float(np.mean([info[a]["metrics"]["waiting_time"] for a in PHYSICAL_INTERSECTIONS]))
                    mean_queue = float(np.mean([info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS]))
                    throughput = int(info["A"].get("arrived_vehicles", 0))
                    active_veh = int(info["A"].get("active_vehicles", 0))
                    interventions = int(info["A"]["safety"]["total_interventions"])
                    conflicts = int(info["A"]["safety"]["audit_physical_violations"])
                    mean_entropy = float(np.mean(step_entropies)) if step_entropies else 0.0

                    row = {
                        "scenario": scen_name,
                        "controller_id": cid,
                        "controller_name": cfg["name"],
                        "seed": seed,
                        "steps": steps,
                        "mean_delay": round(mean_delay, 2),
                        "mean_queue": round(mean_queue, 2),
                        "throughput": throughput,
                        "active_vehicles": active_veh,
                        "total_interventions": interventions,
                        "physical_conflicts": conflicts,
                        "mean_entropy": round(mean_entropy, 4),
                        "safety_enabled": safety_enabled
                    }
                    results.append(row)
                    print(f"[{scen_name} | Seed {seed}] {cid}: Delay={mean_delay:.2f}s | Queue={mean_queue:.2f} | ActVeh={active_veh} | Interv={interventions} | Conflicts={conflicts}")

                finally:
                    env.close()

    # Write results
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[DONE] Recorded {len(results)} stress benchmark episodes to {OUTPUT_CSV}")
    return results

if __name__ == "__main__":
    run_stress_benchmarks()

