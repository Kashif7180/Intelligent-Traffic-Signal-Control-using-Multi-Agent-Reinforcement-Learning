"""
SAGE-Traffic Phase 10: Master 11-Configuration Empirical Benchmark
Validates all 9 primary architectures and 2 ablations across 10 evaluation seeds (1001-1010)
under strictly frozen and reproducible simulation demand.

Evaluation Matrix (11 Configurations x 10 Seeds = 110 Evaluations):
Primary Controllers:
- C1: Fixed-Time Baseline (Periodic 30s Green Rule-Based) [Unshielded]
- C2: Phase 3 IPPO (Decentralized IPPO MLPs 43->64->64->4) [Unshielded]
- C3: Phase 4 MAPPO Shared Policy (CTDE MAPPO Shared MLP 47->64->64->4) [Unshielded]
- C4: Phase 5 Transformer-MAPPO (Temporal Transformer k=4 + MLP 68->4) [Unshielded]
- C5: Phase 6 GAT-MAPPO (Transformer + Dense GAT + MLP 68->4) [Unshielded]
- C6: Phase 7 Comm-Gated MAPPO (Transformer + Gated GAT + Comm Gate + MLP 68->4) [Unshielded]
- C7: Phase 8 Causal-Gated MAPPO (Transformer + Gated GAT + Causal Gate + MLP 68->4) [Unshielded]
- C8: Phase 8R Causal Improved (Transformer + Gated GAT + phi(C_ij) + MLP 68->4) [Unshielded]
- C9: Phase 9 Shielded (SAGE-Traffic Complete Architecture + Runtime Safety Shield) [Shielded]

Ablation Evaluations:
- A1: Phase 4 Individual Actors (MAPPO Individual Actor MLPs per agent) [Unshielded]
- A2: Phase 9 Safety Disabled (Phase 9 Complete Architecture without Safety Shield) [Unshielded]

Protocol:
- Seeds: 1001 - 1010
- Horizon: 100 decision steps per episode (500s SUMO simulation time)
- Demand: Normal traffic profile, pre-generated and frozen with SHA256
- Action Selection: Deterministic Greedy (argmax)
- Gate Decision: Hard threshold g_ij >= 0.5
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
from models.ppo_network import IndependentPPOAgent
from models.mappo_network import MAPPOController, DecentralizedActorNetwork
from models.transformer_network import MAPPOTemporalController
from models.gat_network import MAPPOGATController
from models.communication_gate import MAPPOGateController, MAPPOCausalGateController
from models.causal_improved import MAPPOCausalImprovedController

# Master Authentic SHA256 Checksums for all 20 Checkpoints
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
    """Verify all 20 checkpoints strictly match authentic hashes."""
    for filename, expected_hash in AUTHENTIC_CHECKPOINT_HASHES.items():
        path = os.path.join("models/checkpoints", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} missing at {path}")
        actual_hash = compute_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint integrity violation for {filename}: {actual_hash} != {expected_hash}")
    print("[PASS] Checkpoint integrity verification: 20/20 files strictly verified.")


def freeze_route_cache(seeds, config_path="config.yaml"):
    """Pre-generate and freeze route cache for all evaluation seeds."""
    cache_dir = "network/route_cache"
    os.makedirs(cache_dir, exist_ok=True)
    manifest = {}

    print(f"\n--- Stage B: Generating & Freezing Route Cache for Seeds {seeds} ---")
    for s in seeds:
        cache_path = os.path.join(cache_dir, f"routes_seed_{s}.rou.xml")
        if not os.path.exists(cache_path):
            out_path = generate_routes(config_path, demand_level="normal", seed=s)
            shutil.copy(out_path, cache_path)
        sha = compute_sha256(cache_path)
        manifest[s] = (cache_path, sha)
        print(f"  Seed {s}: {os.path.basename(cache_path)} | SHA256: {sha[:16]}...")
    print(f"[PASS] Route cache frozen and verified for all {len(seeds)} seeds.")
    return manifest


def install_frozen_route(seed, route_manifest, target_path="network/routes.rou.xml"):
    """Install cached route file to active target and verify SHA256 identity."""
    cache_path, expected_sha = route_manifest[seed]
    shutil.copy(cache_path, target_path)
    actual_sha = compute_sha256(target_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Route installation hash mismatch for seed {seed}: {actual_sha} != {expected_sha}")


def load_all_controllers(config_path="config.yaml"):
    """Load and instantiate all 11 evaluation configurations."""
    ckpt_dir = "models/checkpoints"

    # C1: Fixed-Time
    c1_ctrl = FixedTimeController(config_path)

    # C2: IPPO
    c2_agents = {}
    for a in PHYSICAL_INTERSECTIONS:
        agent = IndependentPPOAgent(a, obs_dim=43, act_dim=4)
        data = torch.load(os.path.join(ckpt_dir, f"agent_{a}.pt"), map_location="cpu", weights_only=False)
        agent.actor.load_state_dict(data["actor_state_dict"], strict=True)
        agent.critic.load_state_dict(data["critic_state_dict"], strict=True)
        c2_agents[a] = agent

    # C3: MAPPO Shared
    c3_ctrl = MAPPOController(config_path=config_path)
    c3_ctrl.load_checkpoints(ckpt_dir)

    # C4: Transformer-MAPPO
    c4_ctrl = MAPPOTemporalController(config_path=config_path)
    c4_ctrl.load_checkpoints(ckpt_dir)

    # C5: GAT-MAPPO
    c5_ctrl = MAPPOGATController(config_path=config_path)
    c5_ctrl.load_checkpoints(ckpt_dir)

    # C6: Comm-Gated MAPPO
    c6_ctrl = MAPPOGateController(config_path=config_path)
    c6_ctrl.load_checkpoints(ckpt_dir)

    # C7: Causal-Gated MAPPO
    c7_ctrl = MAPPOCausalGateController(config_path=config_path)
    c7_ctrl.load_checkpoints(ckpt_dir)

    # C8 / C9 / A2: Phase 8R / Phase 9 Causal Improved
    c8_ctrl = MAPPOCausalImprovedController(config_path=config_path)
    c8_ctrl.load_checkpoints(ckpt_dir)

    # A1: Phase 4 Individual Actors
    a1_actors = {}
    for a in PHYSICAL_INTERSECTIONS:
        actor = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
        data = torch.load(os.path.join(ckpt_dir, f"mappo_actor_{a}.pt"), map_location="cpu", weights_only=False)
        actor.load_state_dict(data["actor_state_dict"], strict=True)
        a1_actors[a] = actor

    controllers = [
        # Primary (C1 - C9)
        {"id": "C1", "name": "Fixed-Time Baseline", "category": "Primary", "arch": "Periodic 30s Green Rule-Based", "safety_enabled": False, "type": "fixed_time", "obj": c1_ctrl},
        {"id": "C2", "name": "Phase 3 IPPO", "category": "Primary", "arch": "Decentralized IPPO MLP (43->64->64->4)", "safety_enabled": False, "type": "ippo", "obj": c2_agents},
        {"id": "C3", "name": "Phase 4 MAPPO (Shared Policy)", "category": "Primary", "arch": "CTDE MAPPO Shared MLP (47->64->64->4)", "safety_enabled": False, "type": "mappo_shared", "obj": c3_ctrl},
        {"id": "C4", "name": "Phase 5 Transformer-MAPPO", "category": "Primary", "arch": "Temporal Transformer (k=4, 43->64) + MLP (68->4)", "safety_enabled": False, "type": "transformer", "obj": c4_ctrl},
        {"id": "C5", "name": "Phase 6 GAT-MAPPO", "category": "Primary", "arch": "Transformer (64) + Dense GAT (64) + MLP (68->4)", "safety_enabled": False, "type": "gat", "obj": c5_ctrl},
        {"id": "C6", "name": "Phase 7 Comm-Gated MAPPO", "category": "Primary", "arch": "Transformer (64) + Gated GAT + Comm Gate + MLP (68->4)", "safety_enabled": False, "type": "comm_gate", "obj": c6_ctrl},
        {"id": "C7", "name": "Phase 8 Causal-Gated MAPPO", "category": "Primary", "arch": "Transformer (64) + Gated GAT + Causal Gate + MLP (68->4)", "safety_enabled": False, "type": "causal_gate", "obj": c7_ctrl},
        {"id": "C8", "name": "Phase 8R Causal Improved", "category": "Primary", "arch": "Transformer (64) + Improved Gated GAT + phi(C_ij) + MLP (68->4)", "safety_enabled": False, "type": "causal_improved", "obj": c8_ctrl},
        {"id": "C9", "name": "Phase 9 Shielded (SAGE-Traffic)", "category": "Primary", "arch": "Phase 8R Architecture + Runtime Safety Shield", "safety_enabled": True, "type": "causal_improved", "obj": c8_ctrl},
        # Ablations (A1 - A2)
        {"id": "A1", "name": "Phase 4 Individual Actors", "category": "Ablation", "arch": "MAPPO Individual Actor MLPs (47->64->64->4 per agent)", "safety_enabled": False, "type": "mappo_individual", "obj": a1_actors},
        {"id": "A2", "name": "Phase 9 Safety Disabled", "category": "Ablation", "arch": "Phase 9 Deployment without Safety Shield", "safety_enabled": False, "type": "causal_improved", "obj": c8_ctrl},
    ]

    return controllers


def evaluate_episode(env, cfg, seed, steps_per_episode=100):
    """
    Execute one complete evaluation episode for a given controller and seed.
    Strictly records per-seed metrics: delay, queue, throughput, return, comm, entropy, interventions.
    """
    ctrl_type = cfg["type"]
    ctrl_obj = cfg["obj"]
    safety_enabled = cfg["safety_enabled"]

    # Gymnasium options dictionary explicitly controls runtime safety shielding
    obs, info = env.reset(seed=seed, options={"safety_enabled": safety_enabled})

    # Verify that safety status matches configuration
    if env.safety_enabled != safety_enabled:
        raise RuntimeError(f"Safety mode mismatch! Expected {safety_enabled}, got {env.safety_enabled}")
    if env.safety_layer.enabled != safety_enabled:
        raise RuntimeError(f"Safety layer state mismatch! Expected {safety_enabled}, got {env.safety_layer.enabled}")

    # Controller reset
    if ctrl_type == "fixed_time":
        ctrl_obj.reset()
    elif ctrl_type == "ippo":
        pass  # stateless
    elif ctrl_type == "mappo_shared":
        pass  # stateless
    elif ctrl_type == "mappo_individual":
        pass  # stateless
    elif ctrl_type in ["transformer", "gat", "comm_gate"]:
        ctrl_obj.reset_history(obs)
    elif ctrl_type == "causal_gate":
        ctrl_obj.reset_history(obs, info)
    elif ctrl_type == "causal_improved":
        ctrl_obj.reset_history(obs, info)

    ep_rewards = {a: 0.0 for a in PHYSICAL_INTERSECTIONS}
    ep_queues = []
    ep_waiting = []
    ep_throughput = 0
    ep_actual_comm = 0
    possible_comm_per_step = 8
    ep_possible_comm = steps_per_episode * possible_comm_per_step  # 800

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    step_entropies = []
    eye_4 = torch.eye(4)

    for step in range(steps_per_episode):
        comm_cost_step = 0.0
        actual_comm_step = 0
        step_entropy = 0.0

        if ctrl_type == "fixed_time":
            actions = ctrl_obj.get_actions(obs, info)
            step_entropy = 0.0

        elif ctrl_type == "ippo":
            actions = {}
            for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                obs_t = torch.as_tensor(obs[a], dtype=torch.float32).unsqueeze(0)
                dist = ctrl_obj[a].actor(obs_t)
                actions[a] = int(torch.argmax(dist.probs, dim=-1).item())
                step_entropy += float(dist.entropy().item())
            step_entropy /= 4.0

        elif ctrl_type == "mappo_shared":
            actions, _ = ctrl_obj.get_actions(obs, deterministic=True)
            # Compute policy entropy
            with torch.no_grad():
                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                    id_vec = eye_4[i].numpy()
                    x = np.concatenate([obs[a], id_vec], axis=-1)
                    x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                    dist = ctrl_obj.shared_actor(x_t)
                    step_entropy += float(dist.entropy().item())
            step_entropy /= 4.0

        elif ctrl_type == "mappo_individual":
            actions = {}
            with torch.no_grad():
                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                    id_vec = eye_4[i].numpy()
                    x = np.concatenate([obs[a], id_vec], axis=-1)
                    x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                    dist = ctrl_obj[a](x_t)
                    actions[a] = int(torch.argmax(dist.probs, dim=-1).item())
                    step_entropy += float(dist.entropy().item())
            step_entropy /= 4.0

        elif ctrl_type == "transformer":
            actions, _ = ctrl_obj.get_actions(deterministic=True)
            step_entropy = 0.0
            with torch.no_grad():
                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                    hist_a = ctrl_obj.history_manager.get_agent_history(a)
                    hist_t = torch.as_tensor(hist_a, dtype=torch.float32).unsqueeze(0)
                    emb = ctrl_obj.transformer_encoder(hist_t)
                    dist = ctrl_obj.actor(emb, eye_4[i:i+1])
                    step_entropy += float(dist.entropy().item())
            step_entropy /= 4.0

        elif ctrl_type == "gat":
            actions, _ = ctrl_obj.get_actions(deterministic=True)
            actual_comm_step = 8  # Dense attention over all 8 physical edges
            comm_cost_step = actual_comm_step * 0.01
            step_entropy = 0.0
            with torch.no_grad():
                hist_list = [ctrl_obj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                hist_tensor = torch.as_tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
                emb = ctrl_obj.transformer_encoder(hist_tensor)
                gat_emb = ctrl_obj.gat(emb.unsqueeze(0)).squeeze(0)
                dist = ctrl_obj.actor(gat_emb, eye_4)
                step_entropy = float(dist.entropy().mean().item())

        elif ctrl_type == "comm_gate":
            ctx = ctrl_obj.compute_heuristic_context(info)
            actions, _, comm_costs, actual_comm_step = ctrl_obj.get_actions(ctx, deterministic=True)
            comm_cost_step = sum(comm_costs.values())
            step_entropy = 0.0
            with torch.no_grad():
                hist_list = [ctrl_obj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
                hist_tensor = torch.as_tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
                emb = ctrl_obj.transformer_encoder(hist_tensor)
                gat_res = ctrl_obj.gated_gat(emb.unsqueeze(0), ctx)
                gat_emb = gat_res[0] if isinstance(gat_res, tuple) else gat_res
                dist = ctrl_obj.actor(gat_emb.squeeze(0), eye_4)
                step_entropy = float(dist.entropy().mean().item())

        elif ctrl_type in ["causal_gate", "causal_improved"]:
            ctx = ctrl_obj.compute_causal_context(info)
            actions, _, comm_costs, actual_comm_step = ctrl_obj.get_actions(ctx, deterministic=True)
            comm_cost_step = sum(comm_costs.values())
            step_entropy = 0.0
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

        # Update controller history if applicable
        if ctrl_type in ["transformer", "gat", "comm_gate", "causal_gate", "causal_improved"]:
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

    # Extract safety statistics
    safety_stats = env.safety_layer.get_statistics()
    total_actions = sum(action_counts.values())

    comm_ratio = (ep_actual_comm / ep_possible_comm) if ep_possible_comm > 0 else 0.0
    comm_cost = ep_actual_comm * 0.01

    record = {
        "controller_id": cfg["id"],
        "controller_name": cfg["name"],
        "category": cfg["category"],
        "architecture": cfg["arch"],
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
    }
    return record


def run_benchmark():
    config_path = "config.yaml"
    seeds = list(range(1001, 1011))  # 1001 to 1010
    steps_per_episode = 100

    print("=" * 140)
    print("SAGE-TRAFFIC PHASE 10: MASTER 11-CONFIGURATION EMPIRICAL BENCHMARK")
    print("=" * 140)

    # Stage A: Validate Checkpoint Hashes
    print("\n--- Stage A: Pre-Benchmark Checkpoint Integrity Verification ---")
    verify_all_checkpoints()

    # Stage B: Freeze Route Cache
    route_manifest = freeze_route_cache(seeds, config_path)

    # Load All Controllers
    print("\n--- Stage A.2: Instantiating and Loading All 11 Configurations ---")
    controllers = load_all_controllers(config_path)
    print(f"Successfully loaded all {len(controllers)} controller configurations.")

    env = MultiAgentTrafficEnv(config_path=config_path, gui=False)

    # Stage C: Smoke Tests (1 episode, 2 steps each)
    print("\n--- Stage C: Running Smoke Tests Across All 11 Configurations ---")
    for cfg in controllers:
        install_frozen_route(1001, route_manifest)
        rec = evaluate_episode(env, cfg, seed=1001, steps_per_episode=2)
        print(f"  [SMOKE PASS] {cfg['id']:<3} | {cfg['name']:<35} | Safety: {str(rec['safety_enabled']):<5} | Delay: {rec['mean_delay']:.2f}s")
    print("[PASS] All 11 smoke tests completed cleanly.")

    # Stage D: Complete Benchmark Execution (110 Episodes)
    print("\n--- Stage D: Executing Full 110-Episode Benchmark Matrix ---")
    all_records = []
    start_time = time.time()

    for cfg in controllers:
        print(f"\nEvaluating {cfg['id']}: {cfg['name']} ({cfg['category']}) | Shield: {'ENABLED' if cfg['safety_enabled'] else 'DISABLED'}")
        print(f"{'Seed':<6} | {'Delay (s)':<10} | {'Queue':<8} | {'Throughput':<11} | {'Return':<10} | {'Comm Ratio':<11} | {'Entropy':<8} | {'Interv':<7} | Conflicts")
        print("-" * 95)

        for seed in seeds:
            # Re-install and verify frozen route before every episode
            install_frozen_route(seed, route_manifest)

            rec = evaluate_episode(env, cfg, seed, steps_per_episode=steps_per_episode)
            all_records.append(rec)

            print(f"{seed:<6} | {rec['mean_delay']:<10.2f} | {rec['mean_queue']:<8.2f} | {rec['throughput']:<11} | {rec['task_return']:<10.1f} | {rec['comm_ratio']:<11.1%} | {rec['policy_entropy']:<8.3f} | {rec['total_interventions']:<7} | {rec['physical_conflicts']}")

    env.close()
    elapsed = time.time() - start_time
    print(f"\n[PASS] All 110 benchmark episodes finished in {elapsed:.1f}s ({elapsed/110:.2f}s per episode).")

    # Stage E: Post-Benchmark Checkpoint Verification & Artifact Generation
    print("\n--- Stage E: Post-Benchmark Verification & Artifact Generation ---")
    verify_all_checkpoints()

    # Save detailed CSV
    os.makedirs("logs", exist_ok=True)
    benchmark_csv = "logs/phase10_master_benchmark.csv"
    fieldnames = list(all_records[0].keys())

    with open(benchmark_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[SAVED] Detailed benchmark dataset: {benchmark_csv} ({len(all_records)} rows)")

    # Aggregate Statistics
    summary_records = []
    for cfg in controllers:
        cid = cfg["id"]
        c_recs = [r for r in all_records if r["controller_id"] == cid]

        delays = [r["mean_delay"] for r in c_recs]
        queues = [r["mean_queue"] for r in c_recs]
        throughputs = [r["throughput"] for r in c_recs]
        returns = [r["task_return"] for r in c_recs]
        comm_ratios = [r["comm_ratio"] for r in c_recs]
        entropies = [r["policy_entropy"] for r in c_recs]
        interventions = [r["total_interventions"] for r in c_recs]
        conflicts = [r["physical_conflicts"] for r in c_recs]

        summary_records.append({
            "id": cid,
            "name": cfg["name"],
            "category": cfg["category"],
            "architecture": cfg["arch"],
            "safety_enabled": cfg["safety_enabled"],
            "delay_mean": float(np.mean(delays)),
            "delay_std": float(np.std(delays)),
            "queue_mean": float(np.mean(queues)),
            "queue_std": float(np.std(queues)),
            "throughput_mean": float(np.mean(throughputs)),
            "throughput_std": float(np.std(throughputs)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "comm_ratio_mean": float(np.mean(comm_ratios)),
            "comm_ratio_std": float(np.std(comm_ratios)),
            "entropy_mean": float(np.mean(entropies)),
            "interventions_mean": float(np.mean(interventions)),
            "conflicts_sum": int(sum(conflicts))
        })

    summary_csv = "logs/phase10_master_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[SAVED] Aggregate summary statistics: {summary_csv}")

    # Generate Publication Comparison Figure
    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/phase10_architecture_comparison.png"
    generate_comparison_plots(summary_records, plot_path)
    print(f"[SAVED] High-resolution comparative figure: {plot_path}")

    # Restore network/routes.rou.xml from HEAD
    try:
        shutil.copy("network/route_cache/routes_seed_1001.rou.xml", "network/routes.rou.xml")
    except Exception:
        pass


def generate_comparison_plots(summaries, output_path):
    """Generate high-resolution multi-panel architecture comparison figure."""
    plt.style.use("default")
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    fig.patch.set_facecolor("#0f111a")

    # Filter Primary Controllers C1-C9
    primaries = [s for s in summaries if s["category"] == "Primary"]
    labels = [s["id"] for s in primaries]
    names = [s["name"] for s in primaries]

    # Colors: C1 Gray, C2-C7 Gradient Blues/Purples, C8 Cyan, C9 Emerald Green
    colors = [
        "#6c7086",  # C1
        "#89b4fa",  # C2
        "#74c7ec",  # C3
        "#b4befe",  # C4
        "#cba6f7",  # C5
        "#f5c2e7",  # C6
        "#fab387",  # C7
        "#89dceb",  # C8
        "#a6e3a1",  # C9
    ]

    # 1. Delay
    ax = axs[0, 0]
    ax.set_facecolor("#181926")
    means = [s["delay_mean"] for s in primaries]
    stds = [s["delay_std"] for s in primaries]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#cad3f5", alpha=0.9)
    ax.set_title("Mean Delay per Vehicle (s) [Lower is Better]", color="#cad3f5", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Seconds", color="#cad3f5", fontsize=10)
    ax.tick_params(colors="#cad3f5")
    ax.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.5, f"{m:.1f}s", ha="center", va="bottom", color="#cad3f5", fontsize=8, fontweight="bold")

    # 2. Queue Length
    ax = axs[0, 1]
    ax.set_facecolor("#181926")
    means = [s["queue_mean"] for s in primaries]
    stds = [s["queue_std"] for s in primaries]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#cad3f5", alpha=0.9)
    ax.set_title("Average Queue Length (veh) [Lower is Better]", color="#cad3f5", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Vehicles", color="#cad3f5", fontsize=10)
    ax.tick_params(colors="#cad3f5")
    ax.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.15, f"{m:.2f}", ha="center", va="bottom", color="#cad3f5", fontsize=8, fontweight="bold")

    # 3. Throughput
    ax = axs[1, 0]
    ax.set_facecolor("#181926")
    means = [s["throughput_mean"] for s in primaries]
    stds = [s["throughput_std"] for s in primaries]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#cad3f5", alpha=0.9)
    ax.set_title("Throughput (Arrived Vehicles) [Higher is Better]", color="#cad3f5", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Arrived Vehicles", color="#cad3f5", fontsize=10)
    ax.tick_params(colors="#cad3f5")
    ax.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 1.0, f"{m:.1f}", ha="center", va="bottom", color="#cad3f5", fontsize=8, fontweight="bold")

    # 4. Communication Ratio
    ax = axs[1, 1]
    ax.set_facecolor("#181926")
    means = [s["comm_ratio_mean"] * 100.0 for s in primaries]
    stds = [s["comm_ratio_std"] * 100.0 for s in primaries]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#cad3f5", alpha=0.9)
    ax.set_title("Communication Channel Utilization (%) [Lower is More Efficient]", color="#cad3f5", fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Active Channels (%)", color="#cad3f5", fontsize=10)
    ax.tick_params(colors="#cad3f5")
    ax.grid(True, linestyle="--", alpha=0.2, color="#cad3f5")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 1.0, f"{m:.1f}%", ha="center", va="bottom", color="#cad3f5", fontsize=8, fontweight="bold")

    # Overall Figure Title and Legend
    plt.suptitle("SAGE-Traffic: Comprehensive Architectural Benchmark (C1 - C9 Across 10 Seeds)", color="#cad3f5", fontsize=15, fontweight="bold", y=0.99)
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=f"{l}: {n}") for l, n, c in zip(labels, names, colors)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), facecolor="#181926", edgecolor="#cad3f5", labelcolor="#cad3f5", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_benchmark()
