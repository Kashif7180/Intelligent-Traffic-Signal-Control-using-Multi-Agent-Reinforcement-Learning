"""
Comprehensive Forensic Audit Script for Phase 6.
Tests all 20 audit criteria specified by user without modifying any source files.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import time
import csv
import yaml
import torch
import numpy as np

from environment.traffic_env import MultiAgentTrafficEnv
from models.gat_network import MAPPOGATController, GraphAttentionLayer, DecentralizedGATActor, CentralizedGATCritic
from models.transformer_network import MAPPOTemporalController
from models.mappo_network import MAPPOController


def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 100)
    print("PHASE 6 FORENSIC AUDIT EXECUTION")
    print("=" * 100)

    # -------------------------------------------------------------
    # AUDIT 1 & 2 & 3: Checkpoint paths, SHA256 hashes, and differences
    # -------------------------------------------------------------
    print("\n--- [AUDIT 1, 2, 3] Checkpoints and Hashes ---")
    p4_actor_path = "models/checkpoints/mappo_actor_shared.pt"
    p4_critic_path = "models/checkpoints/mappo_critic.pt"
    p5_actor_path = "models/checkpoints/mappo_transformer_actor.pt"
    p5_critic_path = "models/checkpoints/mappo_transformer_critic.pt"
    p6_actor_path = "models/checkpoints/mappo_transformer_gat_actor.pt"
    p6_critic_path = "models/checkpoints/mappo_transformer_gat_critic.pt"

    for p in [p4_actor_path, p4_critic_path, p5_actor_path, p5_critic_path, p6_actor_path, p6_critic_path]:
        assert os.path.exists(p), f"Missing {p}"
        print(f"{p:<55} | Size: {os.path.getsize(p):<8} | SHA256: {sha256(p)}")

    p4_act_hash = sha256(p4_actor_path)
    p6_act_hash = sha256(p6_actor_path)
    print(f"\nPhase 6 Actor hash == Phase 4 Actor hash? {p6_act_hash == p4_act_hash}")
    assert p6_act_hash != p4_act_hash, "CRITICAL: Phase 6 actor hash matches Phase 4!"

    # Inspect checkpoint keys
    p6_ckpt = torch.load(p6_actor_path, map_location="cpu", weights_only=False)
    print(f"Phase 6 Checkpoint keys: {list(p6_ckpt.keys())}")
    print(f"GAT state dict keys: {list(p6_ckpt['gat_state_dict'].keys())}")
    print(f"Transformer state dict keys (sample): {list(p6_ckpt['transformer_state_dict'].keys())[:3]}")
    print(f"Actor state dict keys: {list(p6_ckpt['actor_state_dict'].keys())}")

    # -------------------------------------------------------------
    # AUDIT 4 & 5: Model class instantiated and GAT construction
    # -------------------------------------------------------------
    print("\n--- [AUDIT 4, 5] Model Classes and GAT Instantiation ---")
    ctrl = MAPPOGATController(config_path="config.yaml")
    ctrl.load_checkpoints("models/checkpoints")

    print(f"Controller type:         {type(ctrl)}")
    print(f"Transformer type:        {type(ctrl.transformer_encoder)}")
    print(f"GAT type:                {type(ctrl.gat)}")
    print(f"Actor type:              {type(ctrl.actor)}")
    print(f"Critic type:             {type(ctrl.critic)}")
    print(f"GAT buffer adj shape:    {ctrl.gat.adj.shape}")
    print(f"GAT buffer adj matrix:\n{ctrl.gat.adj.cpu().numpy()}")

    # -------------------------------------------------------------
    # AUDIT 6, 7, 8, 9, 10, 11, 13, 14, 15:
    # Run seed 1001 with instrumentation
    # -------------------------------------------------------------
    print("\n--- [AUDIT 6, 7, 8, 9, 10, 11, 13, 14, 15] Step-by-Step Execution on Seed 1001 ---")

    # Load all three controllers for direct comparison
    p4_ctrl = MAPPOController(config_path="config.yaml")
    p4_ctrl.load_checkpoints("models/checkpoints")

    p5_ctrl = MAPPOTemporalController(config_path="config.yaml")
    p5_ctrl.load_checkpoints("models/checkpoints")

    # Hook / instrument GAT forward pass
    gat_call_count = 0
    original_gat_forward = ctrl.gat.forward

    def instrumented_gat_forward(*args, **kwargs):
        nonlocal gat_call_count
        gat_call_count += 1
        return original_gat_forward(*args, **kwargs)

    ctrl.gat.forward = instrumented_gat_forward

    env = MultiAgentTrafficEnv(config_path="config.yaml")
    obs, info = env.reset(seed=1001)

    ctrl.reset_history(obs)
    p5_ctrl.reset_history(obs)

    target_steps = [0, 10, 25, 50, 75, 99]
    step_data = {}
    p6_action_seq = []
    p4_action_seq = []
    p5_action_seq = []

    # Also keep track of GAT embeddings across steps
    gat_embeddings_by_step = {}
    gat_attention_by_step = {}

    for step in range(100):
        # 1. Evaluate Phase 6
        # Capture raw tensors before step
        with torch.no_grad():
            hist_list = [ctrl.history_manager.get_agent_history(a) for a in ctrl.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32).unsqueeze(0)
            flat_hist = hist_tensor.view(4, ctrl.history_length, ctrl.obs_dim)
            temp_embs = ctrl.transformer_encoder(flat_hist)  # [4, 64]
            gat_input = temp_embs.unsqueeze(0)               # [1, 4, 64]
            gat_embs = ctrl.gat(gat_input).squeeze(0)        # [4, 64]

            # Attention weights
            att_weights = ctrl.gat.get_attention_weights()

            # Per agent actor evaluation
            actor_outputs = {}
            for i, a in enumerate(ctrl.agents):
                node_emb = gat_embs[i:i+1]
                agent_id = ctrl.one_hot_tensor[i:i+1]
                inp_68 = torch.cat([node_emb, agent_id], dim=-1)
                dist = ctrl.actor(node_emb, agent_id)
                probs = dist.probs.squeeze(0).cpu().numpy()
                logits = dist.logits.squeeze(0).cpu().numpy()
                act = int(torch.argmax(dist.probs, dim=-1).item())

                actor_outputs[a] = {
                    "logits": logits,
                    "probs": probs,
                    "action": act,
                    "temp_norm": float(torch.norm(temp_embs[i]).item()),
                    "gat_norm": float(torch.norm(gat_embs[i]).item()),
                    "input_norm": float(torch.norm(inp_68).item())
                }

        # Also get P5 outputs on same history
        with torch.no_grad():
            p5_acts, _ = p5_ctrl.get_actions(deterministic=True)
            # P4 outputs on current raw obs
            p4_acts, _ = p4_ctrl.get_actions(obs, deterministic=True)

        p6_acts = {a: actor_outputs[a]["action"] for a in ctrl.agents}
        p6_action_seq.append(p6_acts)
        p5_action_seq.append(p5_acts)
        p4_action_seq.append(p4_acts)

        gat_embeddings_by_step[step] = gat_embs.cpu().numpy()
        gat_attention_by_step[step] = att_weights

        if step in target_steps:
            step_data[step] = {
                "P6_actor": actor_outputs,
                "P5_acts": p5_acts,
                "P4_acts": p4_acts,
                "attention": att_weights
            }

        # Step env with Phase 6 actions
        next_obs, rewards, terminations, truncations, next_info = env.step(p6_acts)
        ctrl.update_history(next_obs)
        p5_ctrl.update_history(next_obs)
        obs = next_obs

    env.close()

    print(f"\nTotal GAT forward calls during 100 decision steps: {gat_call_count}")

    # Print Step Data for Audit 7 & 8
    print("\n" + "=" * 90)
    print("DETAILED STEP INSPECTION FOR SEED 1001 (Audit 7 & 8)")
    print("=" * 90)
    for step in target_steps:
        print(f"\n>>> STEP {step} <<<")
        d = step_data[step]
        print(f"P4 Actions: {d['P4_acts']}")
        print(f"P5 Actions: {d['P5_acts']}")
        print(f"P6 Actions: { {a: d['P6_actor'][a]['action'] for a in ctrl.agents} }")
        for a in ctrl.agents:
            ao = d["P6_actor"][a]
            print(f"  Agent {a}:")
            print(f"    Logits:       {np.round(ao['logits'], 4)}")
            print(f"    Probs:        {np.round(ao['probs'], 4)}")
            print(f"    Action:       {ao['action']}")
            print(f"    TempEmb Norm: {ao['temp_norm']:.4f}")
            print(f"    GatEmb Norm:  {ao['gat_norm']:.4f}")
            print(f"    Input68 Norm: {ao['input_norm']:.4f}")

    # Audit 9, 10, 11: Compare Action Sequences
    print("\n" + "=" * 90)
    print("ACTION SEQUENCE COMPARISON (Audit 9, 10, 11)")
    print("=" * 90)
    p6_all_ones = all(all(p6_action_seq[s][a] == 1 for a in ctrl.agents) for s in range(100))
    p4_all_ones = all(all(p4_action_seq[s][a] == 1 for a in ctrl.agents) for s in range(100))
    p6_p4_identical = (p6_action_seq == p4_action_seq)

    print(f"Are all Phase 6 actions identically 1 (switch) at all 100 steps? {p6_all_ones}")
    print(f"Are all Phase 4 actions identically 1 (switch) at all 100 steps? {p4_all_ones}")
    print(f"Is Phase 6 action sequence IDENTICAL to Phase 4 action sequence?  {p6_p4_identical}")

    # Print first 10 actions of Phase 6 and Phase 4
    print("\nFirst 10 steps of Phase 6 actions:")
    for s in range(10):
        print(f"  Step {s:2d}: P6={p6_action_seq[s]} | P4={p4_action_seq[s]} | P5={p5_action_seq[s]}")

    # Audit 14 & 15: Do attention and GAT embeddings change across timesteps?
    print("\n" + "=" * 90)
    print("TEMPORAL VARIATION OF GAT EMBEDDINGS & ATTENTION (Audit 14 & 15)")
    print("=" * 90)
    emb_diff_0_10 = np.linalg.norm(gat_embeddings_by_step[10] - gat_embeddings_by_step[0])
    emb_diff_10_50 = np.linalg.norm(gat_embeddings_by_step[50] - gat_embeddings_by_step[10])
    emb_diff_50_99 = np.linalg.norm(gat_embeddings_by_step[99] - gat_embeddings_by_step[50])

    print(f"Norm diff between GAT embeddings at step 0 and 10:   {emb_diff_0_10:.6f}")
    print(f"Norm diff between GAT embeddings at step 10 and 50:  {emb_diff_10_50:.6f}")
    print(f"Norm diff between GAT embeddings at step 50 and 99:  {emb_diff_50_99:.6f}")
    print(f"Did GAT embeddings change across timesteps? {emb_diff_0_10 > 1e-4 and emb_diff_10_50 > 1e-4}")

    # Check attention changes
    att_map_0 = {(e["source"], e["target"]): e["weight"] for e in gat_attention_by_step[0]["aggregated"]}
    att_map_50 = {(e["source"], e["target"]): e["weight"] for e in gat_attention_by_step[50]["aggregated"]}
    att_map_99 = {(e["source"], e["target"]): e["weight"] for e in gat_attention_by_step[99]["aggregated"]}

    att_diff_0_50 = sum(abs(att_map_50[k] - att_map_0[k]) for k in att_map_0)
    att_diff_50_99 = sum(abs(att_map_99[k] - att_map_50[k]) for k in att_map_50)
    print(f"\nSum of abs attention weight diffs between step 0 and 50:   {att_diff_0_50:.6f}")
    print(f"Sum of abs attention weight diffs between step 50 and 99:  {att_diff_50_99:.6f}")
    print(f"Did attention weights change across timesteps? {att_diff_0_50 > 1e-6}")

    # Audit 13: Report GAT attention at steps 0, 10, 25, 50, 75, 99
    print("\n" + "=" * 90)
    print("GAT ATTENTION WEIGHTS ACROSS STEPS FOR SEED 1001 (Audit 13)")
    print("=" * 90)
    edges = [("A", "A"), ("B", "A"), ("C", "A"),
             ("A", "B"), ("B", "B"), ("D", "B"),
             ("A", "C"), ("C", "C"), ("D", "C"),
             ("B", "D"), ("C", "D"), ("D", "D")]

    header = f"{'Edge (src->tgt)':<16} | " + " | ".join([f"Step {s:<4}" for s in target_steps])
    print(header)
    print("-" * len(header))
    for src, tgt in edges:
        vals = [f"{dict({(e['source'], e['target']): e['weight'] for e in step_data[s]['attention']['aggregated']})[(src, tgt)]:.4f}" for s in target_steps]
        print(f"{src} -> {tgt:<10} | " + " | ".join(vals))

    # Audit 12: CSV inspect
    print("\n" + "=" * 90)
    print("INSPECTING SAVED GAT ATTENTION CSV (Audit 12)")
    print("=" * 90)
    csv_file = "experiments/gat_attention_weights.csv"
    print(f"File path: {os.path.abspath(csv_file)}")
    print(f"File size: {os.path.getsize(csv_file)} bytes")
    print(f"Last modified: {time.ctime(os.path.getmtime(csv_file))}")
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        print(f"Number of rows in CSV: {len(reader)}")
        print(f"Seeds in CSV: {set(r['seed'] for r in reader)}")
        print(f"Steps in CSV: {set(r['decision_step'] for r in reader)}")
        print(f"Heads in CSV: {set(r['head'] for r in reader)}")
        sample_row = reader[0]
        print(f"Sample row: {sample_row}")

    # Audit 16 & 19: Run one fresh independent evaluation for seed 1001
    print("\n" + "=" * 90)
    print("FRESH INDEPENDENT RUN FOR SEED 1001 (Audit 16 & 19)")
    print("=" * 90)
    fresh_ctrl = MAPPOGATController(config_path="config.yaml")
    fresh_ctrl.load_checkpoints("models/checkpoints")

    fresh_env = MultiAgentTrafficEnv(config_path="config.yaml")
    f_obs, _ = fresh_env.reset(seed=1001)
    fresh_ctrl.reset_history(f_obs)

    f_rewards = {a: 0.0 for a in fresh_env.agents}
    f_queues = []
    f_waiting = []
    f_tp = 0

    for step in range(100):
        f_actions, _ = fresh_ctrl.get_actions(deterministic=True)
        f_next_obs, r, term, trunc, f_info = fresh_env.step(f_actions)
        for a in fresh_env.agents:
            f_rewards[a] += r[a]
        f_queues.append(np.mean([f_info[a]["metrics"]["queue_length"] for a in fresh_env.agents]))
        f_waiting.append(np.mean([f_info[a]["metrics"]["waiting_time"] for a in fresh_env.agents]))
        f_tp += f_info["A"]["arrived_vehicles"]

        fresh_ctrl.update_history(f_next_obs)
        if any(term.values()) or any(trunc.values()):
            break

    fresh_env.close()

    fresh_agg_ret = sum(f_rewards.values())
    fresh_mean_q = float(np.mean(f_queues))
    fresh_mean_w = float(np.mean(f_waiting))

    print(f"Fresh Run Seed 1001 Results:")
    print(f"  Delay:      {fresh_mean_w:.2f} s")
    print(f"  Queue:      {fresh_mean_q:.2f} veh")
    print(f"  Throughput: {f_tp}")
    print(f"  Return:     {fresh_agg_ret:.2f}")

    # Compare with evaluation_comparison_gat.csv row for 1001
    with open("experiments/evaluation_comparison_gat.csv", "r", encoding="utf-8") as f:
        e_rows = list(csv.DictReader(f))
        p6_1001 = [r for r in e_rows if r["seed"] == "1001" and "Phase 6" in r["controller"]][0]

    print(f"\nReported in CSV for Seed 1001:")
    print(f"  Delay:      {float(p6_1001['mean_delay']):.2f} s")
    print(f"  Queue:      {float(p6_1001['mean_queue']):.2f} veh")
    print(f"  Throughput: {p6_1001['throughput']}")
    print(f"  Return:     {float(p6_1001['agg_return']):.2f}")

    delay_match = abs(fresh_mean_w - float(p6_1001['mean_delay'])) < 1e-4
    queue_match = abs(fresh_mean_q - float(p6_1001['mean_queue'])) < 1e-4
    tp_match = (f_tp == int(p6_1001['throughput']))
    ret_match = abs(fresh_agg_ret - float(p6_1001['agg_return'])) < 1e-2

    print(f"Does fresh run EXACTLY reproduce reported Phase 6 numbers? {delay_match and queue_match and tp_match and ret_match}")

    # Check why Phase 6 matches Phase 4:
    # Does Phase 6 choose action 1 (switch) deterministically?
    print("\n" + "=" * 90)
    print("ROOT CAUSE ANALYSIS: WHY DOES PHASE 6 PRODUCE THE SAME METRICS AS PHASE 4?")
    print("=" * 90)
    # Check actor weights and logits on typical input
    dummy_gat = torch.randn(1, 64)
    dummy_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    dummy_dist = fresh_ctrl.actor(dummy_gat, dummy_id)
    print(f"Actor logits on random input: {dummy_dist.logits.detach().numpy()}")
    print(f"Actor argmax action on random input: {torch.argmax(dummy_dist.probs, dim=-1).item()}")

    # Check actor final layer weights and biases
    last_linear = fresh_ctrl.actor.network[-1]
    print(f"Actor last layer weights shape: {last_linear.weight.shape}")
    print(f"Actor last layer bias: {last_linear.bias.detach().numpy()}")


if __name__ == "__main__":
    main()
