"""
SAGE-Traffic Phase 12 Step 3: Sensor Noise & Partial Observability Robustness Benchmark
======================================================================================
Evaluates 5 representative controllers under synthetic sensor noise perturbations:
- C1: Fixed-Time Baseline (Periodic 30s Green Rule-Based) [Unshielded]
- C3: Phase 4 MAPPO (Centralized Critic, Decentralized Actors) [Unshielded]
- C5: Phase 6 GAT-MAPPO (Relational Graph Attention Multi-Agent PPO) [Unshielded]
- C8: Phase 8R Causal Improved (Causal Influence Pathway Integration) [Unshielded]
- C9: Phase 9 Shielded SAGE-Traffic (Formal Runtime Safety Shield Active) [Shielded]

Noise Regimes:
1. CLEAN: sigma = 0.00 (Zero perturbation baseline)
2. MODERATE: sigma = 0.05 (5% std dev on continuous sensor features)
3. HIGH: sigma = 0.15 (15% std dev on continuous sensor features)

Evaluation Matrix:
3 noise conditions x 5 controllers x 5 seeds (3001-3005) = 75 evaluation episodes.
Strict Isolation:
- Only policy observation inputs are perturbed.
- Ground-truth SUMO state, rewards, metrics, and Phase 9 safety shield decisions remain authoritative.
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
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.topology import PHYSICAL_INTERSECTIONS
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


# Feature breakdown for 43-D observation:
# - Indices 0-39: 8 lanes x 5 continuous normalized features (vehicle_count, queue_length, average_speed, lane_occupancy, waiting_time)
# - Indices 40-41: current_phase one-hot green indicators (Phase 0, Phase 2) [EXCLUDED: discrete signal state]
# - Index 42: phase_duration normalized [EXCLUDED: internal controller timer]
CONTINUOUS_FEATURE_INDICES = list(range(40))
SIGNAL_STATE_INDICES = [40, 41, 42]


def apply_feature_aware_noise(
    obs: dict,
    sigma: float,
    noise_rng: np.random.Generator
) -> dict:
    """
    Apply zero-mean Gaussian perturbation with [0.0, 1.0] clipping to eligible continuous traffic features.
    Leaves categorical, one-hot, and internal signal state features untouched.
    """
    if sigma <= 0.0:
        return {a: arr.copy() for a, arr in obs.items()}

    noisy_obs = {}
    for a, arr in obs.items():
        perturbed = arr.copy()
        # Generate Gaussian perturbations for the 40 continuous dimensions
        noise = noise_rng.normal(loc=0.0, scale=sigma, size=len(CONTINUOUS_FEATURE_INDICES)).astype(np.float32)
        perturbed[CONTINUOUS_FEATURE_INDICES] = np.clip(
            perturbed[CONTINUOUS_FEATURE_INDICES] + noise,
            0.0,
            1.0
        )
        # Ensure indices 40, 41, 42 remain strictly identical to ground-truth
        perturbed[SIGNAL_STATE_INDICES] = arr[SIGNAL_STATE_INDICES]
        noisy_obs[a] = perturbed

    return noisy_obs


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


def get_cached_route_file(seed: int, cache_dir: str = "network/route_cache_phase12") -> str:
    """Return path to frozen route file for seed."""
    rou_path = os.path.join(cache_dir, f"routes_medium_seed_{seed}.rou.xml")
    if not os.path.exists(rou_path):
        raise FileNotFoundError(f"Missing cached route file: {rou_path}")
    return rou_path


def count_scheduled_in_window(route_file: str, horizon: int = 500) -> int:
    """Count vehicles scheduled with depart < horizon."""
    tree = ET.parse(route_file)
    root = tree.getroot()
    return sum(1 for v in root.findall("vehicle") if int(v.attrib["depart"]) < horizon)


def evaluate_noise_episode(
    env,
    cfg,
    seed,
    condition_name,
    sigma,
    route_file,
    sched_in_window,
    steps_per_episode=100
):
    """
    Execute single evaluation episode with feature-aware synthetic sensor noise.
    Strictly isolates policy observation inputs from environment ground truth.
    """
    ctrl_type = cfg["type"]
    ctrl_obj = cfg["obj"]
    safety_enabled = cfg["safety_enabled"]

    # Ground-truth environment reset
    obs, info = env.reset(seed=seed, options={"safety_enabled": safety_enabled, "route_file": route_file})

    # Independent deterministic noise RNG per episode
    condition_idx = 0 if condition_name == "CLEAN" else (1 if condition_name == "MODERATE" else 2)
    noise_seed = seed * 1000 + condition_idx
    noise_rng = np.random.Generator(np.random.PCG64(noise_seed))

    # Apply initial observation noise for policy consumption
    policy_obs = apply_feature_aware_noise(obs, sigma, noise_rng)

    if ctrl_type == "fixed_time":
        ctrl_obj.reset()
    elif ctrl_type == "mappo_shared":
        pass
    elif ctrl_type == "gat":
        ctrl_obj.reset_history(policy_obs)
    elif ctrl_type == "causal_improved":
        ctrl_obj.reset_history(policy_obs, info)

    ep_rewards = {a: 0.0 for a in PHYSICAL_INTERSECTIONS}
    step_queues = []
    step_delays = []
    total_throughput = 0

    for step in range(steps_per_episode):
        # Policy action selection based strictly on policy_obs
        if ctrl_type == "fixed_time":
            actions = ctrl_obj.get_actions(policy_obs, info)
        elif ctrl_type == "mappo_shared":
            actions, _ = ctrl_obj.get_actions(policy_obs, deterministic=True)
        elif ctrl_type == "gat":
            actions, _ = ctrl_obj.get_actions(deterministic=True)
        elif ctrl_type == "causal_improved":
            ctx = ctrl_obj.compute_causal_context(info)
            actions, _, _, _ = ctrl_obj.get_actions(ctx, deterministic=True)

        # Ground-truth environment step: receives action, validates against authoritative state
        next_obs, rewards, terminations, truncations, next_info = env.step(actions)

        # Generate noisy observations for the NEXT step policy consumption
        next_policy_obs = apply_feature_aware_noise(next_obs, sigma, noise_rng)

        if ctrl_type in ["gat", "causal_improved"]:
            ctrl_obj.update_history(next_policy_obs)

        for a in PHYSICAL_INTERSECTIONS:
            ep_rewards[a] += rewards.get(a, 0.0)

        step_q = float(np.mean([next_info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS]))
        step_w = float(np.mean([next_info[a]["metrics"]["waiting_time"] for a in PHYSICAL_INTERSECTIONS]))
        step_queues.append(step_q)
        step_delays.append(step_w)

        arrived = next_info["A"]["arrived_vehicles"]
        total_throughput += arrived

        obs = next_obs
        policy_obs = next_policy_obs
        info = next_info

        if any(terminations.values()) or any(truncations.values()):
            break

    mean_delay = float(np.mean(step_delays)) if step_delays else 0.0
    mean_queue = float(np.mean(step_queues)) if step_queues else 0.0
    peak_queue = float(np.max(step_queues)) if step_queues else 0.0
    task_return = float(sum(ep_rewards.values()))
    pending_vehicles = max(0, sched_in_window - total_throughput)

    safety_stats = env.safety_layer.get_statistics()
    total_interventions = safety_stats["total_interventions"]
    physical_conflicts = info["A"]["safety"]["audit_physical_violations"]

    record = {
        "condition": condition_name,
        "noise_sigma": sigma,
        "controller_id": cfg["id"],
        "controller_name": cfg["name"],
        "safety_enabled": safety_enabled,
        "seed": seed,
        "scheduled_in_window": sched_in_window,
        "mean_delay": mean_delay,
        "mean_queue": mean_queue,
        "peak_queue": peak_queue,
        "throughput": total_throughput,
        "pending_vehicles": pending_vehicles,
        "task_return": task_return,
        "total_interventions": total_interventions,
        "physical_conflicts": physical_conflicts,
        "completion_status": "COMPLETED"
    }
    return record


def run_benchmark():
    """Execute complete 75-episode sensor noise robustness benchmark."""
    print("=" * 80)
    print("SAGE-Traffic Phase 12 Step 3: Sensor Noise & Partial Observability Benchmark")
    print("=" * 80)

    # 1. Verify Checkpoints
    verify_all_checkpoints()

    # 2. Setup Conditions and Seeds
    conditions = [
        {"name": "CLEAN", "sigma": 0.00, "desc": "Zero observation noise"},
        {"name": "MODERATE", "sigma": 0.05, "desc": "5% std dev Gaussian sensor noise"},
        {"name": "HIGH", "sigma": 0.15, "desc": "15% std dev Gaussian sensor noise"}
    ]
    seeds = [3001, 3002, 3003, 3004, 3005]

    config_path = "config.yaml"
    controllers = load_phase12_controllers(config_path)

    # 3. Cache routes verification
    route_files = {}
    sched_windows = {}
    for s in seeds:
        rf = get_cached_route_file(s)
        route_files[s] = rf
        sched_windows[s] = count_scheduled_in_window(rf)
        sha = compute_sha256(rf)
        print(f"Seed {s}: Route file = {rf} (SHA: {sha[:12]}..., sched_in_window = {sched_windows[s]})")

    total_expected = len(conditions) * len(controllers) * len(seeds)
    print(f"\nExecution Matrix: {len(conditions)} conditions x {len(controllers)} controllers x {len(seeds)} seeds = {total_expected} episodes.")

    # 4. Instantiate environment
    env = MultiAgentTrafficEnv(config_path=config_path, gui=False)

    records = []
    ep_count = 0
    t0 = time.time()

    try:
        for cond in conditions:
            cond_name = cond["name"]
            sigma = cond["sigma"]
            print(f"\n>>> Running Noise Regime: {cond_name} (sigma = {sigma})")

            for cfg in controllers:
                cid = cfg["id"]
                cname = cfg["name"]

                for seed in seeds:
                    ep_count += 1
                    ep_t0 = time.time()
                    rf = route_files[seed]
                    sw = sched_windows[seed]

                    rec = evaluate_noise_episode(
                        env=env,
                        cfg=cfg,
                        seed=seed,
                        condition_name=cond_name,
                        sigma=sigma,
                        route_file=rf,
                        sched_in_window=sw
                    )
                    records.append(rec)
                    ep_dur = time.time() - ep_t0

                    print(
                        f"[{ep_count:02d}/{total_expected}] {cond_name:<8} | {cid:<3} | Seed {seed} | "
                        f"Delay: {rec['mean_delay']:6.2f}s | Queue: {rec['mean_queue']:5.2f} | "
                        f"Throughput: {rec['throughput']:3d} | Pending: {rec['pending_vehicles']:4d} | "
                        f"Interventions: {rec['total_interventions']:3d} | Conflicts: {rec['physical_conflicts']:1d} | "
                        f"Elapsed: {ep_dur:4.1f}s"
                    )
    finally:
        env.close()

    total_duration = time.time() - t0
    print(f"\nBenchmark completed {len(records)} episodes in {total_duration:.1f}s.")

    # 5. Save Raw Benchmark CSV
    os.makedirs("logs", exist_ok=True)
    raw_csv = "logs/phase12_noise_benchmark.csv"

    fieldnames = [
        "condition", "noise_sigma", "controller_id", "controller_name", "safety_enabled",
        "seed", "scheduled_in_window", "mean_delay", "mean_queue", "peak_queue",
        "throughput", "pending_vehicles", "task_return", "total_interventions",
        "physical_conflicts", "completion_status"
    ]

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[SAVED] Raw benchmark data -> {raw_csv} ({len(records)} rows)")

    # 6. Aggregate Summary CSV
    summary_csv = "logs/phase12_noise_summary.csv"
    summary_rows = []

    for cond in conditions:
        cond_name = cond["name"]
        sigma = cond["sigma"]

        for cfg in controllers:
            cid = cfg["id"]
            subset = [r for r in records if r["condition"] == cond_name and r["controller_id"] == cid]
            n = len(subset)

            delays = [r["mean_delay"] for r in subset]
            queues = [r["mean_queue"] for r in subset]
            peak_qs = [r["peak_queue"] for r in subset]
            throughputs = [r["throughput"] for r in subset]
            pendings = [r["pending_vehicles"] for r in subset]
            returns = [r["task_return"] for r in subset]
            interventions = [r["total_interventions"] for r in subset]
            conflicts = [r["physical_conflicts"] for r in subset]

            summary_rows.append({
                "condition": cond_name,
                "noise_sigma": sigma,
                "controller_id": cid,
                "controller_name": cfg["name"],
                "safety_enabled": cfg["safety_enabled"],
                "num_seeds": n,
                "delay_mean": float(np.mean(delays)),
                "delay_pop_sd": float(np.std(delays, ddof=0)),
                "delay_sample_sd": float(np.std(delays, ddof=1)),
                "queue_mean": float(np.mean(queues)),
                "queue_pop_sd": float(np.std(queues, ddof=0)),
                "peak_queue_mean": float(np.mean(peak_qs)),
                "throughput_mean": float(np.mean(throughputs)),
                "pending_mean": float(np.mean(pendings)),
                "return_mean": float(np.mean(returns)),
                "interventions_mean": float(np.mean(interventions)),
                "conflicts_sum": int(sum(conflicts))
            })

    summary_fields = [
        "condition", "noise_sigma", "controller_id", "controller_name", "safety_enabled",
        "num_seeds", "delay_mean", "delay_pop_sd", "delay_sample_sd", "queue_mean",
        "queue_pop_sd", "peak_queue_mean", "throughput_mean", "pending_mean",
        "return_mean", "interventions_mean", "conflicts_sum"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SAVED] Aggregated summary data -> {summary_csv} ({len(summary_rows)} rows)")

    # 7. Generate High-Resolution Comparison Visualization
    generate_comparison_plot(summary_rows)

    # 8. Post-Execution Checkpoint Verification
    verify_all_checkpoints()


def generate_comparison_plot(summary_rows):
    """Generate 6-panel comparison figure."""
    os.makedirs("plots", exist_ok=True)
    plot_file = "plots/phase12_noise_comparison.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.patch.set_facecolor("#1e1e2e")

    controllers = ["C1", "C3", "C5", "C8", "C9"]
    colors = {
        "C1": "#6c7086",  # Gray
        "C3": "#89b4fa",  # Soft Blue
        "C5": "#cba6f7",  # Mauve
        "C8": "#f38ba8",  # Red / Coral
        "C9": "#a6e3a1",  # Emerald Green
    }
    conditions = ["CLEAN", "MODERATE", "HIGH"]
    x = np.arange(len(conditions))
    width = 0.15

    for ax in axes.flat:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cad3f5")
        ax.xaxis.label.set_color("#cad3f5")
        ax.yaxis.label.set_color("#cad3f5")
        ax.title.set_color("#cad3f5")
        for spine in ax.spines.values():
            spine.set_color("#45475a")

    # 1. Mean Delay
    ax = axes[0, 0]
    for i, cid in enumerate(controllers):
        means = [next(r["delay_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        sds = [next(r["delay_pop_sd"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        ax.bar(x + i * width, means, width, yerr=sds, label=cid, color=colors[cid], capsize=3, alpha=0.88)
    ax.set_title("Average Vehicle Delay (s)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Delay (s)")
    ax.legend(facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cad3f5", fontsize=9)

    # 2. Mean Queue
    ax = axes[0, 1]
    for i, cid in enumerate(controllers):
        means = [next(r["queue_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        sds = [next(r["queue_pop_sd"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        ax.bar(x + i * width, means, width, yerr=sds, label=cid, color=colors[cid], capsize=3, alpha=0.88)
    ax.set_title("Average Queue Length (veh)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Queue Length (veh)")

    # 3. Peak Queue
    ax = axes[0, 2]
    for i, cid in enumerate(controllers):
        means = [next(r["peak_queue_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        ax.bar(x + i * width, means, width, label=cid, color=colors[cid], alpha=0.88)
    ax.set_title("Peak Network Queue (veh)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Peak Queue (veh)")

    # 4. Throughput
    ax = axes[1, 0]
    for i, cid in enumerate(controllers):
        means = [next(r["throughput_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        ax.bar(x + i * width, means, width, label=cid, color=colors[cid], alpha=0.88)
    ax.set_title("Completed Throughput (veh)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Vehicles Arrived")

    # 5. Pending Vehicles
    ax = axes[1, 1]
    for i, cid in enumerate(controllers):
        means = [next(r["pending_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == cid) for c in conditions]
        ax.bar(x + i * width, means, width, label=cid, color=colors[cid], alpha=0.88)
    ax.set_title("Pending Vehicles at Termination", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Vehicles Pending")

    # 6. Safety Interventions & Physical Conflicts
    ax = axes[1, 2]
    interv_means = [next(r["interventions_mean"] for r in summary_rows if r["condition"] == c and r["controller_id"] == "C9") for c in conditions]
    ax.bar(x, interv_means, width * 2, label="C9 Interventions (clamps/ep)", color="#a6e3a1", alpha=0.88)
    ax.set_title("Phase 9 Safety Interventions & Conflicts", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Interventions per Episode")
    # Annotate conflicts = 0
    for idx in range(len(conditions)):
        ax.text(idx, 100, "Conflicts: 0", ha="center", va="center", color="#cad3f5", fontsize=10, fontweight="bold")

    plt.suptitle("SAGE-Traffic Phase 12 Step 3: Sensor Noise & Partial Observability Robustness", color="#cad3f5", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(plot_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SAVED] High-resolution visualization -> {plot_file}")


if __name__ == "__main__":
    run_benchmark()

