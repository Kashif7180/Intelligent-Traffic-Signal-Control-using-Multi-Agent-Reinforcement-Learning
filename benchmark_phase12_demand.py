"""
SAGE-Traffic Phase 12 Step 1: Robustness Testing Under Different Traffic Demands
Evaluates 5 representative controllers across 3 traffic demand levels (LOW, MEDIUM, HIGH)
and 5 reproducible evaluation seeds (1001-1005), yielding 75 evaluation episodes.

Demand Levels:
- LOW:    Light traffic flow (profile: 'low', insertion prob: 0.083, ~655 veh/1000s, ~41% baseline)
- MEDIUM: Baseline urban flow (profile: 'normal', insertion prob: 0.200, ~1604 veh/1000s, 100% baseline)
- HIGH:   Heavy peak-hour flow (profile: 'peak', insertion prob: 0.350, ~2814 veh/1000s, ~175% baseline)

Representative Controllers:
- C1: Fixed-Time Baseline (Periodic 30s Green Rule-Based) [Unshielded]
- C3: Phase 4 MAPPO Shared Policy (CTDE MAPPO Shared MLP 47->64->64->4) [Unshielded]
- C5: Phase 6 GAT-MAPPO (Transformer + Dense GAT + MLP 68->4) [Unshielded]
- C8: Phase 8R Causal Improved (Transformer + Gated GAT + phi(C_ij) + MLP 68->4) [Unshielded]
- C9: Phase 9 Shielded (SAGE-Traffic Complete Architecture + Runtime Safety Shield) [Shielded]

Evaluation Matrix:
5 Controllers x 3 Demand Levels x 5 Seeds = 75 Evaluation Episodes.
Each episode: 100 decision steps (500s SUMO simulation time).
"""

import os
import sys
import shutil
import hashlib
import time
import csv
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.topology import (
    PHYSICAL_INTERSECTIONS,
    ALLOWED_DIRECTED_EDGES,
    CORRIDOR_INCOMING_LANES
)
from environment.demand_generator import generate_routes
from environment.traffic_env import MultiAgentTrafficEnv
from models.fixed_time_controller import FixedTimeController
from models.mappo_network import MAPPOController
from models.gat_network import MAPPOGATController
from models.causal_improved import MAPPOCausalImprovedController

# Authentic SHA256 Checksums for all 20 Checkpoints
AUTHENTIC_CHECKPOINT_HASHES = {
    "agent_A.pt": "140d4e66664357da66d1abe2f4a9c391a88790ab5f10a770529705a5643db88e",
    "agent_B.pt": "89b433b52b6e5f20249947aea02bb34a0b13f04dcc5337a2a08a251d383d7d51",
    "agent_C.pt": "3bf905fcfc2f605157c77046e20f650fc29771dad8d4a00cb3c524c370992c0b",
    "agent_D.pt": "78f691f3568c51354a2d2a9d52279482af0708d98de3c824024b2f1d95769577",
    "mappo_actor_A.pt": "f4d6bb870580681bb0f75e1d60bd0c09ef7395771f25aff442a039551fd3419c",
    "mappo_actor_B.pt": "30307d36fd8ae27bb9143ddc701efa8de6c13408b6ce9c1a5b6d3c6a5a709650",
    "mappo_actor_C.pt": "aa0dbe3b2c9356caa9108062cc4f1bca12576c549b5c0619c24cd6fdbd6f8e33",
    "mappo_actor_D.pt": "a23a5ad4cedd057fff83673a6db63493fed99f5e6981bde029e2c43215a194f9",
    "mappo_actor_shared.pt": "aa3be0f91b8ed2a710ba4b3d14668b7093eea97d2e7da4225a02c55c796f85b3",
    "mappo_causal_gate_actor.pt": "81c2e33fac002a24ac843187c296150afa918468ca3063893d35b6cb057075a0",
    "mappo_causal_gate_critic.pt": "213a00da324511f277a4e6c523dfcf1c2da71931c5f836316dc4674528d6ec0e",
    "mappo_causal_improved_actor.pt": "9a30cb38568abca26e96205efdfba0dfece6dc974bca4ff218041f284d811846",
    "mappo_causal_improved_critic.pt": "8b6ba5ff9d08eccf207095d860ea641ade828e8fe003b6aed629984d39d55a19",
    "mappo_comm_gate_actor.pt": "4759869bca999d18035e27195d4621b3b49305dcedca1df38a743924621076f8",
    "mappo_comm_gate_critic.pt": "0070b5f013eeff5a2f0ba75954a00b5cb262fd9b5dce24b24f565a65602aeb0b",
    "mappo_critic.pt": "44f576d4b2295c04187ec5fce62df6d48eb281aa7dd701044c5c81621e5a2d7d",
    "mappo_transformer_actor.pt": "f14ff6b672f6743446b9a326e9c5991092709c85d0e9ead7a05285018572301f",
    "mappo_transformer_critic.pt": "027ffa3cd2894c02a906f12bb7d29e4c73033616a2caed90ca0f10af7bb44cda",
    "mappo_transformer_gat_actor.pt": "ad669be67c535cf8064ab85a10c8373ffc7f8fd68089115f9dc8c7a5241603a7",
    "mappo_transformer_gat_critic.pt": "a10bde1b0fc676c70e5725c7e0f76ae67faf8c65321c1c93d1cd62a3494fc7e4",
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def verify_all_checkpoints():
    """Verify all 20 checkpoints strictly match authentic SHA256 hashes."""
    for filename, expected_hash in AUTHENTIC_CHECKPOINT_HASHES.items():
        path = os.path.join("models/checkpoints", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} missing at {path}")
        actual_hash = compute_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint integrity violation for {filename}: {actual_hash} != {expected_hash}")
    print("[PASS] Checkpoint integrity verification: 20/20 files strictly verified.")


def freeze_phase12_routes(demand_scenarios, seeds, config_path="config.yaml"):
    """Pre-generate and freeze route cache for Phase 12 demand levels."""
    cache_dir = "network/route_cache_phase12"
    os.makedirs(cache_dir, exist_ok=True)
    manifest = {}

    print(f"\n--- Freezing Route Cache for Phase 12 Demands ({list(demand_scenarios.keys())}) and Seeds {seeds} ---")
    for d_name, d_profile in demand_scenarios.items():
        manifest[d_name] = {}
        for s in seeds:
            cache_path = os.path.join(cache_dir, f"routes_{d_name.lower()}_seed_{s}.rou.xml")
            if not os.path.exists(cache_path):
                out_path = generate_routes(config_path, demand_level=d_profile, seed=s)
                shutil.copy(out_path, cache_path)
            
            with open(cache_path, "r", encoding="utf-8") as f:
                veh_count = f.read().count("<vehicle ")
            
            sha = compute_sha256(cache_path)
            manifest[d_name][s] = {
                "path": cache_path,
                "sha256": sha,
                "vehicles": veh_count
            }
            print(f"  [{d_name}] Seed {s}: {veh_count} veh | SHA256: {sha[:16]}... | {os.path.basename(cache_path)}")
    print(f"[PASS] All {len(demand_scenarios) * len(seeds)} route files frozen and verified.")
    return manifest


def install_frozen_route(demand_name, seed, manifest, target_path="network/routes.rou.xml"):
    """Copy cached route file to active target and verify SHA256."""
    info = manifest[demand_name][seed]
    shutil.copy(info["path"], target_path)
    actual_sha = compute_sha256(target_path)
    if actual_sha != info["sha256"]:
        raise RuntimeError(f"Route installation hash mismatch for {demand_name} seed {seed}: {actual_sha} != {info['sha256']}")


def load_phase12_controllers(config_path="config.yaml"):
    """Load and instantiate the 5 representative controllers."""
    ckpt_dir = "models/checkpoints"

    # C1: Fixed-Time
    c1_ctrl = FixedTimeController(config_path)

    # C3: MAPPO Shared
    c3_ctrl = MAPPOController(config_path=config_path)
    c3_ctrl.load_checkpoints(ckpt_dir)

    # C5: GAT-MAPPO
    c5_ctrl = MAPPOGATController(config_path=config_path)
    c5_ctrl.load_checkpoints(ckpt_dir)

    # C8 & C9: Phase 8R Causal Improved & Phase 9 Shielded
    c8_ctrl = MAPPOCausalImprovedController(config_path=config_path)
    c8_ctrl.load_checkpoints(ckpt_dir)

    controllers = [
        {"id": "C1", "name": "Fixed-Time Baseline", "safety_enabled": False, "type": "fixed_time", "obj": c1_ctrl},
        {"id": "C3", "name": "Phase 4 MAPPO", "safety_enabled": False, "type": "mappo_shared", "obj": c3_ctrl},
        {"id": "C5", "name": "Phase 6 GAT-MAPPO", "safety_enabled": False, "type": "gat", "obj": c5_ctrl},
        {"id": "C8", "name": "Phase 8R Causal Improved", "safety_enabled": False, "type": "causal_improved", "obj": c8_ctrl},
        {"id": "C9", "name": "Phase 9 Shielded (SAGE-Traffic)", "safety_enabled": True, "type": "causal_improved", "obj": c8_ctrl},
    ]
    return controllers


def evaluate_episode(env, cfg, seed, demand_name, demand_profile, veh_count, steps_per_episode=100):
    """Execute one evaluation episode under specified demand and seed."""
    ctrl_type = cfg["type"]
    ctrl_obj = cfg["obj"]
    safety_enabled = cfg["safety_enabled"]

    obs, info = env.reset(seed=seed, options={"safety_enabled": safety_enabled}, demand_level=demand_profile)

    # Verify safety mode isolation
    if env.safety_enabled != safety_enabled:
        raise RuntimeError(f"Safety mode mismatch! Expected {safety_enabled}, got {env.safety_enabled}")
    if env.safety_layer.enabled != safety_enabled:
        raise RuntimeError(f"Safety layer state mismatch! Expected {safety_enabled}, got {env.safety_layer.enabled}")

    # Reset controller state
    if ctrl_type == "fixed_time":
        ctrl_obj.reset()
    elif ctrl_type == "mappo_shared":
        pass
    elif ctrl_type == "gat":
        ctrl_obj.reset_history(obs)
    elif ctrl_type == "causal_improved":
        ctrl_obj.reset_history(obs, info)

    ep_rewards = {a: 0.0 for a in PHYSICAL_INTERSECTIONS}
    ep_queues = []
    ep_waiting = []
    ep_throughput = 0
    ep_actual_comm = 0
    ep_possible_comm = steps_per_episode * 8

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    step_entropies = []
    eye_4 = torch.eye(4)

    for step in range(steps_per_episode):
        actual_comm_step = 0
        step_entropy = 0.0

        if ctrl_type == "fixed_time":
            actions = ctrl_obj.get_actions(obs, info)
            step_entropy = 0.0

        elif ctrl_type == "mappo_shared":
            actions, _ = ctrl_obj.get_actions(obs, deterministic=True)
            with torch.no_grad():
                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                    id_vec = eye_4[i].numpy()
                    x = np.concatenate([obs[a], id_vec], axis=-1)
                    x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                    dist = ctrl_obj.shared_actor(x_t)
                    step_entropy += float(dist.entropy().item())
            step_entropy /= 4.0

        elif ctrl_type == "gat":
            actions, _ = ctrl_obj.get_actions(deterministic=True)
            actual_comm_step = 8
            with torch.no_grad():
                hist_list = [ctrl_obj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                hist_tensor = torch.as_tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
                emb = ctrl_obj.transformer_encoder(hist_tensor)
                gat_emb = ctrl_obj.gat(emb.unsqueeze(0)).squeeze(0)
                dist = ctrl_obj.actor(gat_emb, eye_4)
                step_entropy = float(dist.entropy().mean().item())

        elif ctrl_type == "causal_improved":
            ctx = ctrl_obj.compute_causal_context(info)
            actions, _, comm_costs, actual_comm_step = ctrl_obj.get_actions(ctx, deterministic=True)
            with torch.no_grad():
                hist_list = [ctrl_obj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                hist_tensor = torch.as_tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
                emb = ctrl_obj.transformer_encoder(hist_tensor)
                gat_res = ctrl_obj.gated_gat(emb.unsqueeze(0), ctx)
                gat_emb = gat_res[0] if isinstance(gat_res, tuple) else gat_res
                dist = ctrl_obj.actor(gat_emb.squeeze(0), eye_4)
                step_entropy = float(dist.entropy().mean().item())

        for a in PHYSICAL_INTERSECTIONS:
            act = actions.get(a, 0)
            action_counts[act] = action_counts.get(act, 0) + 1

        ep_actual_comm += actual_comm_step
        step_entropies.append(step_entropy)

        next_obs, rewards, terminations, truncations, next_info = env.step(actions)

        if ctrl_type in ["gat", "causal_improved"]:
            ctrl_obj.update_history(next_obs)

        for a in PHYSICAL_INTERSECTIONS:
            ep_rewards[a] += rewards.get(a, 0.0)

        step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS])
        step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in PHYSICAL_INTERSECTIONS])
        ep_queues.append(step_q)
        ep_waiting.append(step_w)
        ep_throughput += next_info["A"]["arrived_vehicles"]

        obs = next_obs
        info = next_info

        if any(terminations.values()) or any(truncations.values()):
            break

    safety_stats = env.safety_layer.get_statistics()
    total_actions = sum(action_counts.values())
    comm_ratio = (ep_actual_comm / ep_possible_comm) if ep_possible_comm > 0 else 0.0
    comm_cost = ep_actual_comm * 0.01

    record = {
        "demand_level": demand_name,
        "demand_profile": demand_profile,
        "vehicle_count": veh_count,
        "controller_id": cfg["id"],
        "controller_name": cfg["name"],
        "safety_enabled": safety_enabled,
        "seed": seed,
        "mean_delay": float(np.mean(ep_waiting)),
        "mean_queue": float(np.mean(ep_queues)),
        "throughput": int(ep_throughput),
        "task_return": float(sum(ep_rewards.values())),
        "actual_comm": int(ep_actual_comm),
        "possible_comm": int(ep_possible_comm),
        "comm_ratio": float(comm_ratio),
        "comm_cost": float(comm_cost),
        "action_keep_pct": float(action_counts[0] / total_actions * 100.0),
        "action_switch_pct": float(action_counts[1] / total_actions * 100.0),
        "action_extend_pct": float(action_counts[2] / total_actions * 100.0),
        "action_reduce_pct": float(action_counts[3] / total_actions * 100.0),
        "policy_entropy": float(np.mean(step_entropies)),
        "total_interventions": int(safety_stats.get("total_interventions", 0)),
        "min_green_enforced": int(safety_stats.get("min_green_enforced", 0)),
        "max_green_enforced": int(safety_stats.get("max_green_enforced", 0)),
        "min_red_enforced": int(safety_stats.get("min_red_enforced", 0)),
        "yellow_clearance_clamps": int(safety_stats.get("yellow_clearance_clamps", 0)),
        "all_red_clearance_clamps": int(safety_stats.get("all_red_clearance_clamps", 0)),
        "physical_conflicts": int(safety_stats.get("physical_conflicts", 0)),
        "completion_status": "COMPLETED"
    }
    return record


def run_demand_benchmark():
    config_path = "config.yaml"
    seeds = [1001, 1002, 1003, 1004, 1005]
    demand_scenarios = {
        "LOW": "low",
        "MEDIUM": "normal",
        "HIGH": "peak"
    }

    print("=" * 100)
    print("PHASE 12 STEP 1: ROBUSTNESS EVALUATION UNDER MULTI-DEMAND REGIMES")
    print("=" * 100)

    # 1. Pre-Benchmark Checkpoint Verification
    print("\n[Step 1/6] Verifying Checkpoint Hashes Before Benchmark...")
    verify_all_checkpoints()

    # 2. Freeze Route Cache for Demands
    print("\n[Step 2/6] Freezing Multi-Demand Route Cache...")
    route_manifest = freeze_phase12_routes(demand_scenarios, seeds, config_path)

    # 3. Load Controllers
    print("\n[Step 3/6] Instantiating 5 Representative Controllers...")
    controllers = load_phase12_controllers(config_path)
    for c in controllers:
        print(f"  Loaded {c['id']}: {c['name']} (Safety: {c['safety_enabled']})")

    # 4. Initialize Multi-Agent Environment
    print("\n[Step 4/6] Initializing MultiAgentTrafficEnv...")
    env = MultiAgentTrafficEnv(config_path=config_path, gui=False)

    all_benchmark_records = []
    total_evals = len(demand_scenarios) * len(controllers) * len(seeds)
    eval_idx = 0

    print(f"\n[Step 5/6] Executing {total_evals} Evaluation Episodes (3 Demands x 5 Controllers x 5 Seeds)...")
    start_time = time.time()

    for d_name, d_profile in demand_scenarios.items():
        print(f"\n>>> Running Demand Regime: {d_name} (profile: {d_profile}) <<<")
        for s in seeds:
            # Install frozen route file for this demand and seed
            install_frozen_route(d_name, s, route_manifest)
            veh_count = route_manifest[d_name][s]["vehicles"]

            for ctrl in controllers:
                eval_idx += 1
                rec = evaluate_episode(
                    env=env,
                    cfg=ctrl,
                    seed=s,
                    demand_name=d_name,
                    demand_profile=d_profile,
                    veh_count=veh_count,
                    steps_per_episode=100
                )
                all_benchmark_records.append(rec)
                print(f"  [{eval_idx:02d}/{total_evals:02d}] {d_name:6s} | Seed {s} | {ctrl['id']} ({ctrl['name'][:18]:18s}) | "
                      f"Delay: {rec['mean_delay']:6.2f}s | Queue: {rec['mean_queue']:5.2f} | "
                      f"Throughput: {rec['throughput']:4d} | Interventions: {rec['total_interventions']:3d} | "
                      f"Conflicts: {rec['physical_conflicts']}")

    env.close()
    elapsed = time.time() - start_time
    print(f"\n[PASS] All {total_evals} evaluation runs completed in {elapsed:.2f}s ({elapsed/total_evals:.2f}s per episode).")

    # 5. Save Raw Benchmark Results
    raw_csv = "logs/phase12_demand_benchmark.csv"
    os.makedirs("logs", exist_ok=True)
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_benchmark_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_benchmark_records)
    print(f"[SAVED] Raw benchmark dataset (75 rows): {raw_csv}")

    # 6. Compute Aggregate Summary Statistics
    summary_records = []
    for d_name in demand_scenarios.keys():
        for ctrl in controllers:
            cid = ctrl["id"]
            subset = [r for r in all_benchmark_records if r["demand_level"] == d_name and r["controller_id"] == cid]
            delays = [r["mean_delay"] for r in subset]
            queues = [r["mean_queue"] for r in subset]
            throughputs = [r["throughput"] for r in subset]
            returns = [r["task_return"] for r in subset]
            interventions = [r["total_interventions"] for r in subset]
            conflicts = [r["physical_conflicts"] for r in subset]

            summary_records.append({
                "demand_level": d_name,
                "controller_id": cid,
                "controller_name": ctrl["name"],
                "safety_enabled": ctrl["safety_enabled"],
                "num_seeds": len(subset),
                # Delay
                "delay_mean": float(np.mean(delays)),
                "delay_pop_sd": float(np.std(delays, ddof=0)),
                "delay_sample_sd": float(np.std(delays, ddof=1)),
                "delay_median": float(np.median(delays)),
                "delay_min": float(np.min(delays)),
                "delay_max": float(np.max(delays)),
                # Queue
                "queue_mean": float(np.mean(queues)),
                "queue_pop_sd": float(np.std(queues, ddof=0)),
                "queue_sample_sd": float(np.std(queues, ddof=1)),
                "queue_median": float(np.median(queues)),
                # Throughput
                "throughput_mean": float(np.mean(throughputs)),
                "throughput_pop_sd": float(np.std(throughputs, ddof=0)),
                "throughput_sample_sd": float(np.std(throughputs, ddof=1)),
                "throughput_median": float(np.median(throughputs)),
                # Return
                "return_mean": float(np.mean(returns)),
                "return_pop_sd": float(np.std(returns, ddof=0)),
                "return_sample_sd": float(np.std(returns, ddof=1)),
                # Interventions & Conflicts
                "interventions_mean": float(np.mean(interventions)),
                "conflicts_sum": int(sum(conflicts))
            })

    summary_csv = "logs/phase12_demand_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[SAVED] Aggregate summary dataset (15 rows): {summary_csv}")

    # 7. Generate Comparison Plot
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/phase12_demand_comparison.png"
    generate_phase12_plots(summary_records, plot_path)
    print(f"[SAVED] High-resolution comparative plot: {plot_path}")

    # 8. Post-Benchmark Checkpoint Verification
    print("\n[Step 6/6] Verifying Checkpoint Hashes After Benchmark...")
    verify_all_checkpoints()

    # Restore default route file
    try:
        shutil.copy("network/route_cache/routes_seed_1001.rou.xml", "network/routes.rou.xml")
    except Exception:
        pass
    print("\n[SUCCESS] Phase 12 Demand Benchmark fully executed and verified!")


def generate_phase12_plots(summaries, output_path):
    """Generate 6-panel high-resolution comparison plot across demand levels."""
    plt.style.use("default")
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), dpi=200)
    fig.patch.set_facecolor("#0f111a")

    demands = ["LOW", "MEDIUM", "HIGH"]
    controllers = ["C1", "C3", "C5", "C8", "C9"]
    ctrl_colors = {
        "C1": "#6c7086",  # Gray
        "C3": "#74c7ec",  # Cyan Blue
        "C5": "#cba6f7",  # Purple
        "C8": "#89dceb",  # Sky Blue
        "C9": "#a6e3a1",  # Mint Green
    }
    x = np.arange(len(demands))
    bar_width = 0.16

    # 1. Delay Bar Chart
    ax1 = axs[0, 0]
    ax1.set_facecolor("#181926")
    for i, cid in enumerate(controllers):
        means = [next(s["delay_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        errs = [next(s["delay_pop_sd"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax1.bar(x + i * bar_width, means, bar_width, yerr=errs, capsize=3, label=cid, color=ctrl_colors[cid], alpha=0.9, edgecolor="#cad3f5")
    ax1.set_xticks(x + 2 * bar_width)
    ax1.set_xticklabels(demands, color="#cad3f5", fontweight="bold")
    ax1.set_title("Mean Delay (s) [Lower is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Seconds", color="#cad3f5")
    ax1.tick_params(colors="#cad3f5")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax1.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 2. Queue Length Bar Chart
    ax2 = axs[0, 1]
    ax2.set_facecolor("#181926")
    for i, cid in enumerate(controllers):
        means = [next(s["queue_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        errs = [next(s["queue_pop_sd"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax2.bar(x + i * bar_width, means, bar_width, yerr=errs, capsize=3, label=cid, color=ctrl_colors[cid], alpha=0.9, edgecolor="#cad3f5")
    ax2.set_xticks(x + 2 * bar_width)
    ax2.set_xticklabels(demands, color="#cad3f5", fontweight="bold")
    ax2.set_title("Average Queue Length (veh) [Lower is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Vehicles", color="#cad3f5")
    ax2.tick_params(colors="#cad3f5")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")

    # 3. Throughput Bar Chart
    ax3 = axs[0, 2]
    ax3.set_facecolor("#181926")
    for i, cid in enumerate(controllers):
        means = [next(s["throughput_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        errs = [next(s["throughput_pop_sd"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax3.bar(x + i * bar_width, means, bar_width, yerr=errs, capsize=3, label=cid, color=ctrl_colors[cid], alpha=0.9, edgecolor="#cad3f5")
    ax3.set_xticks(x + 2 * bar_width)
    ax3.set_xticklabels(demands, color="#cad3f5", fontweight="bold")
    ax3.set_title("Throughput (Arrived Vehicles) [Higher is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Arrived Vehicles", color="#cad3f5")
    ax3.tick_params(colors="#cad3f5")
    ax3.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")

    # 4. Episode Task Return
    ax4 = axs[1, 0]
    ax4.set_facecolor("#181926")
    for i, cid in enumerate(controllers):
        means = [next(s["return_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        errs = [next(s["return_pop_sd"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax4.bar(x + i * bar_width, means, bar_width, yerr=errs, capsize=3, label=cid, color=ctrl_colors[cid], alpha=0.9, edgecolor="#cad3f5")
    ax4.set_xticks(x + 2 * bar_width)
    ax4.set_xticklabels(demands, color="#cad3f5", fontweight="bold")
    ax4.set_title("Episode Cumulative Return [Higher is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Reward Sum", color="#cad3f5")
    ax4.tick_params(colors="#cad3f5")
    ax4.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")

    # 5. Safety Interventions
    ax5 = axs[1, 1]
    ax5.set_facecolor("#181926")
    for i, cid in enumerate(controllers):
        means = [next(s["interventions_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax5.bar(x + i * bar_width, means, bar_width, label=cid, color=ctrl_colors[cid], alpha=0.9, edgecolor="#cad3f5")
    ax5.set_xticks(x + 2 * bar_width)
    ax5.set_xticklabels(demands, color="#cad3f5", fontweight="bold")
    ax5.set_title("Safety Shield Interventions (Clamps / ep)", color="#cad3f5", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Interventions Count", color="#cad3f5")
    ax5.tick_params(colors="#cad3f5")
    ax5.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")

    # 6. Delay Scaling Trajectory (Line Plot)
    ax6 = axs[1, 2]
    ax6.set_facecolor("#181926")
    for cid in controllers:
        means = [next(s["delay_mean"] for s in summaries if s["demand_level"] == d and s["controller_id"] == cid) for d in demands]
        ax6.plot(demands, means, marker="o", linewidth=2.5, label=cid, color=ctrl_colors[cid])
    ax6.set_title("Delay Sensitivity to Traffic Congestion", color="#cad3f5", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Mean Delay (s)", color="#cad3f5")
    ax6.set_xlabel("Traffic Demand Level", color="#cad3f5")
    ax6.tick_params(colors="#cad3f5")
    ax6.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax6.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    plt.suptitle("SAGE-Traffic Phase 12: Robustness Evaluation Across LOW, MEDIUM, and HIGH Traffic Demands",
                 color="#cad3f5", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


if __name__ == "__main__":
    run_demand_benchmark()

