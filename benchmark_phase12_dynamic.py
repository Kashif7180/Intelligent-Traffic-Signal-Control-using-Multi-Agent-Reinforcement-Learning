"""
SAGE-Traffic Phase 12 Step 2: Dynamic Demand & Sudden Traffic Spike Robustness
Evaluates 5 representative controllers under sudden dynamic changes in traffic demand:
- Scenario A: Sudden Traffic Surge (MEDIUM -> 2x Peak at t=200s)
- Scenario B: Sudden Traffic Reduction (HIGH -> MEDIUM at t=200s)
Across 5 new reproducible seeds: 2001-2005 (50 evaluation episodes).
"""

import os
import sys
import shutil
import hashlib
import time
import csv
import random
import yaml
import numpy as np
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES
from environment.demand_generator import get_all_routes
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
    """Verify all 20 checkpoints strictly match authentic SHA256 checksums."""
    for filename, expected_hash in AUTHENTIC_CHECKPOINT_HASHES.items():
        path = os.path.join("models/checkpoints", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} missing at {path}")
        actual_hash = compute_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint integrity violation for {filename}: {actual_hash} != {expected_hash}")
    print("[PASS] Checkpoint integrity verification: 20/20 files strictly verified.")


def generate_dynamic_routes(
    scenario_type: str,
    seed: int,
    transition_time: int = 200,
    sim_duration: int = 1000,
    config_path: str = "config.yaml",
    cache_dir: str = "network/route_cache_phase12"
) -> Tuple[str, str, int, int]:
    """
    Generate dynamic route file with transition at transition_time (200s).
    Scenario A (SURGE): p=0.200 (Medium) -> p=0.400 (2x Baseline)
    Scenario B (REDUCTION): p=0.350 (High) -> p=0.200 (Medium)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if scenario_type == "SURGE":
        p_pre, p_post = 0.200, 0.400
    elif scenario_type == "REDUCTION":
        p_pre, p_post = 0.350, 0.200
    else:
        raise ValueError(f"Unknown scenario {scenario_type}")

    rng = random.Random(seed)
    routes_by_entry = get_all_routes()
    car_cfg = config["demand"]["vehicle_types"]["car"]

    root = ET.Element("routes")
    ET.SubElement(
        root, "vType",
        id="car", accel=str(car_cfg["accel"]), decel=str(car_cfg["decel"]),
        sigma=str(car_cfg["sigma"]), length=str(car_cfg["length"]),
        minGap=str(car_cfg["minGap"]), maxSpeed=str(car_cfg["maxSpeed"])
    )

    route_id_counter = 0
    route_map = {}
    for entry, path_list in routes_by_entry.items():
        for path in path_list:
            path_str = " ".join(path)
            if path_str not in route_map:
                r_id = f"r_{route_id_counter}"
                route_id_counter += 1
                route_map[path_str] = r_id
                ET.SubElement(root, "route", id=r_id, edges=path_str)

    veh_count = 0
    sched_in_window = 0
    for t in range(sim_duration):
        p_now = p_pre if t < transition_time else p_post
        for entry, path_list in routes_by_entry.items():
            if not path_list:
                continue
            if rng.random() < p_now:
                chosen_path = rng.choice(path_list)
                path_str = " ".join(chosen_path)
                r_id = route_map[path_str]
                ET.SubElement(
                    root, "vehicle",
                    id=f"veh_{veh_count}", type="car", route=r_id,
                    depart=str(t), departLane="best", departSpeed="max"
                )
                veh_count += 1
                if t < 500:
                    sched_in_window += 1

    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="    ")
    cleaned_xml = "\n".join([line for line in xml_str.splitlines() if line.strip()])

    out_path = os.path.join(cache_dir, f"routes_dynamic_{scenario_type.lower()}_seed_{seed}.rou.xml")
    os.makedirs(cache_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_xml)

    sha = hashlib.sha256(cleaned_xml.encode("utf-8")).hexdigest()
    return out_path, sha, veh_count, sched_in_window


def load_phase12_controllers(config_path="config.yaml"):
    """Load representative controllers."""
    ckpt_dir = "models/checkpoints"

    c1_ctrl = FixedTimeController(config_path)
    c3_ctrl = MAPPOController(config_path=config_path)
    c3_ctrl.load_checkpoints(ckpt_dir)
    c5_ctrl = MAPPOGATController(config_path=config_path)
    c5_ctrl.load_checkpoints(ckpt_dir)
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


def evaluate_dynamic_episode(
    env,
    cfg,
    seed,
    scenario_type,
    route_file,
    sched_in_window,
    transition_step=40,  # step 40 = t=200s (40 * 5s)
    steps_per_episode=100
):
    """Execute dynamic evaluation episode and compute pre/post transition metrics."""
    ctrl_type = cfg["type"]
    ctrl_obj = cfg["obj"]
    safety_enabled = cfg["safety_enabled"]

    obs, info = env.reset(seed=seed, options={"safety_enabled": safety_enabled, "route_file": route_file})

    if ctrl_type == "fixed_time":
        ctrl_obj.reset()
    elif ctrl_type == "mappo_shared":
        pass
    elif ctrl_type == "gat":
        ctrl_obj.reset_history(obs)
    elif ctrl_type == "causal_improved":
        ctrl_obj.reset_history(obs, info)

    ep_rewards = {a: 0.0 for a in PHYSICAL_INTERSECTIONS}
    step_queues = []
    step_delays = []
    throughput_at_200 = 0
    total_throughput = 0

    eye_4 = torch.eye(4)

    for step in range(steps_per_episode):
        if ctrl_type == "fixed_time":
            actions = ctrl_obj.get_actions(obs, info)
        elif ctrl_type == "mappo_shared":
            actions, _ = ctrl_obj.get_actions(obs, deterministic=True)
        elif ctrl_type == "gat":
            actions, _ = ctrl_obj.get_actions(deterministic=True)
        elif ctrl_type == "causal_improved":
            ctx = ctrl_obj.compute_causal_context(info)
            actions, _, _, _ = ctrl_obj.get_actions(ctx, deterministic=True)

        next_obs, rewards, terminations, truncations, next_info = env.step(actions)

        if ctrl_type in ["gat", "causal_improved"]:
            ctrl_obj.update_history(next_obs)

        for a in PHYSICAL_INTERSECTIONS:
            ep_rewards[a] += rewards.get(a, 0.0)

        step_q = float(np.mean([next_info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS]))
        step_w = float(np.mean([next_info[a]["metrics"]["waiting_time"] for a in PHYSICAL_INTERSECTIONS]))
        step_queues.append(step_q)
        step_delays.append(step_w)

        arrived = next_info["A"]["arrived_vehicles"]
        total_throughput += arrived
        if step == transition_step - 1:
            throughput_at_200 = total_throughput

        obs = next_obs
        info = next_info

        if any(terminations.values()) or any(truncations.values()):
            break

    # Pre-transition (t < 200s, steps 0-39) vs Post-transition (t >= 200s, steps 40-99)
    pre_delays = step_delays[:transition_step]
    post_delays = step_delays[transition_step:]
    pre_queues = step_queues[:transition_step]
    post_queues = step_queues[transition_step:]

    pre_delay = float(np.mean(pre_delays)) if pre_delays else 0.0
    post_delay = float(np.mean(post_delays)) if post_delays else 0.0
    pre_queue = float(np.mean(pre_queues)) if pre_queues else 0.0
    post_queue = float(np.mean(post_queues)) if post_queues else 0.0
    peak_queue = float(np.max(step_queues)) if step_queues else 0.0

    pre_throughput = throughput_at_200
    post_throughput = total_throughput - throughput_at_200
    pending_vehicles = max(0, sched_in_window - total_throughput)

    delay_spike_ratio = float(post_delay / max(0.001, pre_delay))

    # Recovery Time Estimation for Reduction scenario:
    # Criterion: Time (in seconds) after t=200s until queue drops below 1.25x baseline Medium queue (8.04 veh for MARL, 14.66 for C1)
    # for at least 3 consecutive decision steps.
    baseline_q = 14.66 if ctrl_type == "fixed_time" else (9.87 if safety_enabled else 8.04)
    target_recovery_threshold = 1.25 * baseline_q
    recovery_time_s = -1.0  # -1 indicates unrecovered within window

    if scenario_type == "REDUCTION":
        consec = 0
        for i, q in enumerate(post_queues):
            if q <= target_recovery_threshold:
                consec += 1
                if consec >= 3:
                    # step duration is 5s
                    recovery_time_s = float((i - 2) * 5)
                    break
            else:
                consec = 0

    safety_stats = env.safety_layer.get_statistics()

    record = {
        "scenario": scenario_type,
        "scenario_name": "Sudden Traffic Surge (Medium -> 2x Peak)" if scenario_type == "SURGE" else "Sudden Traffic Reduction (Peak -> Medium)",
        "controller_id": cfg["id"],
        "controller_name": cfg["name"],
        "safety_enabled": safety_enabled,
        "seed": seed,
        "scheduled_in_window": sched_in_window,
        "mean_delay": float(np.mean(step_delays)),
        "pre_delay": pre_delay,
        "post_delay": post_delay,
        "delay_spike_ratio": delay_spike_ratio,
        "mean_queue": float(np.mean(step_queues)),
        "pre_queue": pre_queue,
        "post_queue": post_queue,
        "peak_queue": peak_queue,
        "total_throughput": int(total_throughput),
        "pre_throughput": int(pre_throughput),
        "post_throughput": int(post_throughput),
        "pending_vehicles": int(pending_vehicles),
        "recovery_time_s": recovery_time_s,
        "task_return": float(sum(ep_rewards.values())),
        "total_interventions": int(safety_stats.get("total_interventions", 0)),
        "physical_conflicts": int(safety_stats.get("physical_conflicts", 0)),
        "completion_status": "COMPLETED"
    }
    return record, step_delays, step_queues


def run_dynamic_benchmark():
    config_path = "config.yaml"
    seeds = [2001, 2002, 2003, 2004, 2005]
    scenarios = ["SURGE", "REDUCTION"]

    print("=" * 100)
    print("PHASE 12 STEP 2: DYNAMIC DEMAND & SUDDEN TRAFFIC SPIKE ROBUSTNESS BENCHMARK")
    print("=" * 100)

    # 1. Pre-Benchmark Checkpoint Verification
    print("\n[Step 1/6] Verifying Checkpoint Hashes Before Benchmark...")
    verify_all_checkpoints()

    # 2. Pre-generate and Freeze Dynamic Routes
    print("\n[Step 2/6] Generating and Freezing Dynamic Route Cache for Seeds 2001-2005...")
    route_manifest = {}
    for sc in scenarios:
        route_manifest[sc] = {}
        for s in seeds:
            rf, sha, tot, in_win = generate_dynamic_routes(sc, s, transition_time=200, config_path=config_path)
            route_manifest[sc][s] = {
                "path": rf,
                "sha256": sha,
                "total_vehs": tot,
                "in_window": in_win
            }
            print(f"  [{sc}] Seed {s}: in_window={in_win} veh | total={tot} | SHA256={sha[:16]}... | {os.path.basename(rf)}")
    print(f"[PASS] All {len(scenarios) * len(seeds)} dynamic route files generated and frozen.")

    # 3. Load Controllers
    print("\n[Step 3/6] Instantiating 5 Representative Controllers...")
    controllers = load_phase12_controllers(config_path)

    # 4. Initialize MultiAgentTrafficEnv
    print("\n[Step 4/6] Initializing MultiAgentTrafficEnv...")
    env = MultiAgentTrafficEnv(config_path=config_path, gui=False)

    all_records = []
    trajectory_data = {sc: {c["id"]: [] for c in controllers} for sc in scenarios}
    total_evals = len(scenarios) * len(controllers) * len(seeds)
    eval_idx = 0

    print(f"\n[Step 5/6] Executing {total_evals} Evaluation Episodes (2 Scenarios x 5 Controllers x 5 Seeds)...")
    start_time = time.time()

    for sc in scenarios:
        print(f"\n>>> Running Dynamic Scenario: {sc} <<<")
        for s in seeds:
            rf_info = route_manifest[sc][s]
            rf_path = rf_info["path"]
            in_win = rf_info["in_window"]

            for ctrl in controllers:
                eval_idx += 1
                rec, delays, queues = evaluate_dynamic_episode(
                    env=env,
                    cfg=ctrl,
                    seed=s,
                    scenario_type=sc,
                    route_file=rf_path,
                    sched_in_window=in_win,
                    transition_step=40,
                    steps_per_episode=100
                )
                all_records.append(rec)
                trajectory_data[sc][ctrl["id"]].append(queues)

                recov_str = f"{rec['recovery_time_s']:.0f}s" if rec['recovery_time_s'] >= 0 else "N/A"
                print(f"  [{eval_idx:02d}/{total_evals:02d}] {sc:9s} | Seed {s} | {ctrl['id']} ({ctrl['name'][:18]:18s}) | "
                      f"Pre-Delay: {rec['pre_delay']:5.2f}s -> Post: {rec['post_delay']:6.2f}s (Ratio: {rec['delay_spike_ratio']:4.1f}x) | "
                      f"Pre-Q: {rec['pre_queue']:4.1f} -> Post-Q: {rec['post_queue']:4.1f} | Peak-Q: {rec['peak_queue']:4.1f} | "
                      f"Throughput: {rec['total_throughput']:3d} | Pending: {rec['pending_vehicles']:4d} | Recovery: {recov_str} | Conflicts: {rec['physical_conflicts']}")

    env.close()
    elapsed = time.time() - start_time
    print(f"\n[PASS] All {total_evals} dynamic evaluation runs completed in {elapsed:.2f}s ({elapsed/total_evals:.2f}s per episode).")

    # 5. Save Raw Benchmark CSV
    raw_csv = "logs/phase12_dynamic_benchmark.csv"
    os.makedirs("logs", exist_ok=True)
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[SAVED] Raw dynamic benchmark dataset (50 rows): {raw_csv}")

    # 6. Compute Aggregate Summary Statistics
    summary_records = []
    for sc in scenarios:
        for ctrl in controllers:
            cid = ctrl["id"]
            subset = [r for r in all_records if r["scenario"] == sc and r["controller_id"] == cid]
            
            delays = [r["mean_delay"] for r in subset]
            pre_delays = [r["pre_delay"] for r in subset]
            post_delays = [r["post_delay"] for r in subset]
            spike_ratios = [r["delay_spike_ratio"] for r in subset]

            queues = [r["mean_queue"] for r in subset]
            pre_queues = [r["pre_queue"] for r in subset]
            post_queues = [r["post_queue"] for r in subset]
            peak_queues = [r["peak_queue"] for r in subset]

            throughputs = [r["total_throughput"] for r in subset]
            pending = [r["pending_vehicles"] for r in subset]
            returns = [r["task_return"] for r in subset]
            interventions = [r["total_interventions"] for r in subset]
            conflicts = [r["physical_conflicts"] for r in subset]

            recov_vals = [r["recovery_time_s"] for r in subset if r["recovery_time_s"] >= 0]
            mean_recov = float(np.mean(recov_vals)) if recov_vals else -1.0

            summary_records.append({
                "scenario": sc,
                "scenario_name": subset[0]["scenario_name"],
                "controller_id": cid,
                "controller_name": ctrl["name"],
                "safety_enabled": ctrl["safety_enabled"],
                "num_seeds": len(subset),
                # Delay Metrics
                "delay_mean": float(np.mean(delays)),
                "delay_pop_sd": float(np.std(delays, ddof=0)),
                "delay_sample_sd": float(np.std(delays, ddof=1)),
                "pre_delay_mean": float(np.mean(pre_delays)),
                "post_delay_mean": float(np.mean(post_delays)),
                "delay_spike_ratio_mean": float(np.mean(spike_ratios)),
                # Queue Metrics
                "queue_mean": float(np.mean(queues)),
                "queue_pop_sd": float(np.std(queues, ddof=0)),
                "pre_queue_mean": float(np.mean(pre_queues)),
                "post_queue_mean": float(np.mean(post_queues)),
                "peak_queue_mean": float(np.mean(peak_queues)),
                # Throughput & Pending
                "throughput_mean": float(np.mean(throughputs)),
                "pending_mean": float(np.mean(pending)),
                "return_mean": float(np.mean(returns)),
                "recovery_time_mean_s": mean_recov,
                # Safety
                "interventions_mean": float(np.mean(interventions)),
                "conflicts_sum": int(sum(conflicts))
            })

    summary_csv = "logs/phase12_dynamic_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[SAVED] Aggregate dynamic summary dataset (10 rows): {summary_csv}")

    # 7. Generate Comparison Plot
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/phase12_dynamic_comparison.png"
    generate_dynamic_plots(summary_records, trajectory_data, plot_path)
    print(f"[SAVED] High-resolution comparative plot: {plot_path}")

    # 8. Post-Benchmark Checkpoint Verification
    print("\n[Step 6/6] Verifying Checkpoint Hashes After Benchmark...")
    verify_all_checkpoints()

    # Restore default route file
    try:
        shutil.copy("network/route_cache/routes_seed_1001.rou.xml", "network/routes.rou.xml")
    except Exception:
        pass
    print("\n[SUCCESS] Phase 12 Step 2 Dynamic Benchmark fully executed and verified!")


def generate_dynamic_plots(summaries, trajectory_data, output_path):
    """Generate 6-panel high-resolution comparison plot for dynamic scenarios."""
    plt.style.use("default")
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), dpi=200)
    fig.patch.set_facecolor("#0f111a")

    controllers = ["C1", "C3", "C5", "C8", "C9"]
    ctrl_colors = {
        "C1": "#6c7086",  # Gray
        "C3": "#74c7ec",  # Cyan Blue
        "C5": "#cba6f7",  # Purple
        "C8": "#89dceb",  # Sky Blue
        "C9": "#a6e3a1",  # Mint Green
    }
    steps_x = np.arange(100) * 5  # simulation seconds (0 to 495s)

    # 1. Surge Queue Trajectory over Time (Panel 1)
    ax1 = axs[0, 0]
    ax1.set_facecolor("#181926")
    for cid in controllers:
        arr = np.mean(trajectory_data["SURGE"][cid], axis=0)
        ax1.plot(steps_x, arr, label=cid, color=ctrl_colors[cid], linewidth=2.2)
    ax1.axvline(200, color="#f38ba8", linestyle="--", linewidth=1.8, label="Surge Spike (t=200s)")
    ax1.set_title("Surge: Queue Trajectory over Time [Lower is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Queue Length (veh)", color="#cad3f5")
    ax1.set_xlabel("Simulation Time (seconds)", color="#cad3f5")
    ax1.tick_params(colors="#cad3f5")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax1.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 2. Reduction Queue Trajectory over Time (Panel 2)
    ax2 = axs[0, 1]
    ax2.set_facecolor("#181926")
    for cid in controllers:
        arr = np.mean(trajectory_data["REDUCTION"][cid], axis=0)
        ax2.plot(steps_x, arr, label=cid, color=ctrl_colors[cid], linewidth=2.2)
    ax2.axvline(200, color="#a6e3a1", linestyle="--", linewidth=1.8, label="Demand Drop (t=200s)")
    ax2.set_title("Reduction: Queue Dissipation over Time", color="#cad3f5", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Queue Length (veh)", color="#cad3f5")
    ax2.set_xlabel("Simulation Time (seconds)", color="#cad3f5")
    ax2.tick_params(colors="#cad3f5")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax2.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 3. Pre vs Post Delay Comparison (Bar Chart)
    ax3 = axs[0, 2]
    ax3.set_facecolor("#181926")
    bar_width = 0.35
    idx = np.arange(len(controllers))
    pre_d = [next(s["pre_delay_mean"] for s in summaries if s["scenario"] == "SURGE" and s["controller_id"] == cid) for cid in controllers]
    post_d = [next(s["post_delay_mean"] for s in summaries if s["scenario"] == "SURGE" and s["controller_id"] == cid) for cid in controllers]
    ax3.bar(idx - bar_width/2, pre_d, bar_width, label="Pre-Surge (t<200s)", color="#89b4fa", alpha=0.9, edgecolor="#cad3f5")
    ax3.bar(idx + bar_width/2, post_d, bar_width, label="Post-Surge (t>=200s)", color="#f38ba8", alpha=0.9, edgecolor="#cad3f5")
    ax3.set_xticks(idx)
    ax3.set_xticklabels(controllers, color="#cad3f5", fontweight="bold")
    ax3.set_title("Surge: Pre vs Post Delay Impact (s)", color="#cad3f5", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Delay (seconds)", color="#cad3f5")
    ax3.tick_params(colors="#cad3f5")
    ax3.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax3.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 4. Total Throughput vs Pending Vehicles
    ax4 = axs[1, 0]
    ax4.set_facecolor("#181926")
    thru = [next(s["throughput_mean"] for s in summaries if s["scenario"] == "SURGE" and s["controller_id"] == cid) for cid in controllers]
    pend = [next(s["pending_mean"] for s in summaries if s["scenario"] == "SURGE" and s["controller_id"] == cid) for cid in controllers]
    ax4.bar(idx - bar_width/2, thru, bar_width, label="Arrived Vehicles", color="#a6e3a1", alpha=0.9, edgecolor="#cad3f5")
    ax4.bar(idx + bar_width/2, pend, bar_width, label="Pending in Network", color="#fab387", alpha=0.9, edgecolor="#cad3f5")
    ax4.set_xticks(idx)
    ax4.set_xticklabels(controllers, color="#cad3f5", fontweight="bold")
    ax4.set_title("Surge: Throughput vs Residual Congestion", color="#cad3f5", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Vehicle Count", color="#cad3f5")
    ax4.tick_params(colors="#cad3f5")
    ax4.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax4.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 5. Reduction Delay Dissipation (Pre vs Post)
    ax5 = axs[1, 1]
    ax5.set_facecolor("#181926")
    pre_red = [next(s["pre_delay_mean"] for s in summaries if s["scenario"] == "REDUCTION" and s["controller_id"] == cid) for cid in controllers]
    post_red = [next(s["post_delay_mean"] for s in summaries if s["scenario"] == "REDUCTION" and s["controller_id"] == cid) for cid in controllers]
    ax5.bar(idx - bar_width/2, pre_red, bar_width, label="Peak Demand (t<200s)", color="#f38ba8", alpha=0.9, edgecolor="#cad3f5")
    ax5.bar(idx + bar_width/2, post_red, bar_width, label="Post-Drop (t>=200s)", color="#a6e3a1", alpha=0.9, edgecolor="#cad3f5")
    ax5.set_xticks(idx)
    ax5.set_xticklabels(controllers, color="#cad3f5", fontweight="bold")
    ax5.set_title("Reduction: Pre vs Post Delay Dissipation (s)", color="#cad3f5", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Delay (seconds)", color="#cad3f5")
    ax5.tick_params(colors="#cad3f5")
    ax5.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax5.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    # 6. Overall Cumulative Return across Scenarios
    ax6 = axs[1, 2]
    ax6.set_facecolor("#181926")
    ret_surge = [next(s["return_mean"] for s in summaries if s["scenario"] == "SURGE" and s["controller_id"] == cid) for cid in controllers]
    ret_reduc = [next(s["return_mean"] for s in summaries if s["scenario"] == "REDUCTION" and s["controller_id"] == cid) for cid in controllers]
    ax6.bar(idx - bar_width/2, ret_surge, bar_width, label="Surge Return", color="#f9e2af", alpha=0.9, edgecolor="#cad3f5")
    ax6.bar(idx + bar_width/2, ret_reduc, bar_width, label="Reduction Return", color="#cba6f7", alpha=0.9, edgecolor="#cad3f5")
    ax6.set_xticks(idx)
    ax6.set_xticklabels(controllers, color="#cad3f5", fontweight="bold")
    ax6.set_title("Task Return (Reward Sum) [Higher is Better]", color="#cad3f5", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Cumulative Reward", color="#cad3f5")
    ax6.tick_params(colors="#cad3f5")
    ax6.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    ax6.legend(facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=8)

    plt.suptitle("SAGE-Traffic Phase 12 Step 2: Dynamic Demand Response & Traffic Spike Robustness",
                 color="#cad3f5", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


if __name__ == "__main__":
    run_dynamic_benchmark()

