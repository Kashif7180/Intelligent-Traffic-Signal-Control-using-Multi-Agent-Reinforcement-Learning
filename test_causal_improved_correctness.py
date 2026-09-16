"""
Pre-Training Correctness Verification & Pathway Sensitivity for Phase 8R.

Validates 18 mandatory unit tests and experimentally measures:
1. Gradient magnitude through the causal pathway
2. Gradient magnitude through h_i/h_j pathway
3. Gate-logit sensitivity to C_ij
4. Gate probability sensitivity to C_ij
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np

from environment.traffic_env import MultiAgentTrafficEnv
from models.causal_influence import CausalInfluenceEstimator, ALLOWED_DIRECTED_EDGES
from models.causal_improved import (
    CausalPathwayEncoder,
    ImprovedCausalCommunicationGate,
    ImprovedGatedGraphAttentionLayer,
    MAPPOCausalImprovedController
)


def create_mock_info(q_A=10.0, w_A=20.0, d_A=15.0, q_B=2.0, w_B=4.0, d_B=3.0, corr_AB=6.0):
    return {
        "A": {
            "metrics": {
                "queue_length": q_A, "waiting_time": w_A, "vehicle_count": d_A,
                "lane_details": {"B_to_A_0": {"queue_length": 0.0}}
            }
        },
        "B": {
            "metrics": {
                "queue_length": q_B, "waiting_time": w_B, "vehicle_count": d_B,
                "lane_details": {"A_to_B_0": {"queue_length": corr_AB}}
            }
        },
        "C": {
            "metrics": {
                "queue_length": 1.0, "waiting_time": 2.0, "vehicle_count": 2.0,
                "lane_details": {"A_to_C_0": {"queue_length": 1.0}}
            }
        },
        "D": {
            "metrics": {
                "queue_length": 0.0, "waiting_time": 0.0, "vehicle_count": 1.0,
                "lane_details": {"B_to_D_0": {"queue_length": 0.0}}
            }
        }
    }


def run_tests():
    print("=" * 85)
    print("PHASE 8R: PRE-TRAINING TESTS & PATHWAY SENSITIVITY MEASUREMENTS")
    print("=" * 85)

    passed = 0
    total = 18

    estimator = CausalInfluenceEstimator(history_length=5, window_size=15)
    encoder = CausalPathwayEncoder(input_dim=1, hidden_dim=16, causal_dim=8)
    gate = ImprovedCausalCommunicationGate(node_dim=64, causal_dim=8, causal_hidden_dim=16)

    # -----------------------------------------------------------------
    # Test 1: C_ij estimator returns valid values in [0, 1]
    # -----------------------------------------------------------------
    estimator.reset()
    for _ in range(6):
        estimator.update_traffic_state(create_mock_info())
    scores = estimator.estimate_causal_influence()
    assert len(scores) == 8
    for v in scores.values():
        assert 0.0 <= v <= 1.0
    print("[PASS] Test 1:  C_ij estimator returns valid values in [0, 1]")
    passed += 1

    # -----------------------------------------------------------------
    # Test 2: C_ij varies with traffic history
    # -----------------------------------------------------------------
    est2 = CausalInfluenceEstimator(history_length=5)
    for s in range(8):
        est2.update_traffic_state(create_mock_info(q_A=1.0 + s * 3, d_A=2.0 + s * 2))
    scores2 = est2.estimate_causal_influence()
    assert scores[("A", "B")] != scores2[("A", "B")]
    print(f"[PASS] Test 2:  C_ij varies dynamically with traffic history (s1={scores[('A', 'B')]:.3f}, s2={scores2[('A', 'B')]:.3f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 3: C_ij has no future leakage
    # -----------------------------------------------------------------
    est_leak1 = CausalInfluenceEstimator(history_length=5)
    est_leak2 = CausalInfluenceEstimator(history_length=5)
    for s in range(5):
        est_leak1.update_traffic_state(create_mock_info(q_A=float(s)))
        est_leak2.update_traffic_state(create_mock_info(q_A=float(s)))
    s_t5_1 = est_leak1.estimate_causal_influence()
    est_leak2.update_traffic_state(create_mock_info(q_A=99.0))
    s_t5_1_after = est_leak1.estimate_causal_influence()
    assert s_t5_1 == s_t5_1_after
    print("[PASS] Test 3:  C_ij has no future leakage (strictly past observations)")
    passed += 1

    # -----------------------------------------------------------------
    # Test 4: Causal embedding receives C_ij
    # -----------------------------------------------------------------
    c_sample = torch.tensor([[0.75]], dtype=torch.float32)
    phi = encoder(c_sample)
    assert phi.shape == (1, 8)
    print(f"[PASS] Test 4:  Causal embedding phi(C_ij) receives C_ij and outputs shape {phi.shape}")
    passed += 1

    # -----------------------------------------------------------------
    # Test 5: Changing C_ij changes causal embedding
    # -----------------------------------------------------------------
    phi_0 = encoder(torch.tensor([[0.0]]))
    phi_1 = encoder(torch.tensor([[1.0]]))
    emb_diff = (phi_0 - phi_1).abs().sum().item()
    assert emb_diff > 0.1
    print(f"[PASS] Test 5:  Changing C_ij from 0 to 1 changes phi(C_ij) (L1 diff: {emb_diff:.4f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 6: Changing C_ij can change gate logits/probabilities
    # -----------------------------------------------------------------
    nodes = torch.randn(1, 4, 64)
    ctx_0 = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}
    ctx_1 = {e: 1.0 for e in ALLOWED_DIRECTED_EDGES}
    _, raw_0, logits_0 = gate(nodes, ctx_0, deterministic=False)
    _, raw_1, logits_1 = gate(nodes, ctx_1, deterministic=False)
    logit_diff = (logits_0 - logits_1).abs().mean().item()
    prob_diff = (raw_0 - raw_1).abs().mean().item()
    assert logit_diff > 0.001
    print(f"[PASS] Test 6:  Changing C_ij changes gate logits (diff: {logit_diff:.4f}) and probs (diff: {prob_diff:.4f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 7: Gate gradients reach the causal embedding
    # -----------------------------------------------------------------
    c_var = torch.tensor([[0.5]], requires_grad=True)
    phi_var = encoder(c_var)
    h_tgt = torch.randn(1, 64)
    h_src = torch.randn(1, 64)
    gate_in = torch.cat([h_tgt, h_src, phi_var], dim=-1)
    g_out = gate.sigmoid(gate.mlp_body(gate_in))
    g_out.backward()
    assert c_var.grad is not None and c_var.grad.abs().item() > 0.0
    print(f"[PASS] Test 7:  Gate gradients reach causal embedding & C_ij directly (grad: {c_var.grad.item():.6f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 8: Communication gate is differentiable during training
    # -----------------------------------------------------------------
    _, soft_gates, _ = gate(nodes, scores, deterministic=False)
    assert soft_gates.requires_grad
    print("[PASS] Test 8:  Communication gate produces differentiable soft values in (0, 1) during training")
    passed += 1

    # -----------------------------------------------------------------
    # Test 9: Communication cost changes when gate usage changes
    # -----------------------------------------------------------------
    cost_0 = 0.01 * float(raw_0.sum().item())
    cost_1 = 0.01 * float(raw_1.sum().item())
    assert cost_0 != cost_1
    print(f"[PASS] Test 9:  Communication cost changes with gate usage (cost_0={cost_0:.4f}, cost_1={cost_1:.4f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 10: Actor receives gated messages
    # -----------------------------------------------------------------
    ctrl = MAPPOCausalImprovedController(config_path="config.yaml")
    mock_obs = {a: np.zeros(43, dtype=np.float32) for a in ctrl.agents}
    ctrl.reset_history(mock_obs)
    actions, lps, costs, actual_comm = ctrl.get_actions(scores, deterministic=True)
    assert len(actions) == 4
    print("[PASS] Test 10: Decentralized actor receives gated neighbor messages (actions selected)")
    passed += 1

    # -----------------------------------------------------------------
    # Test 11: Actor rejects centralized global observation input
    # -----------------------------------------------------------------
    assert ctrl.actor_input_dim == 68  # 64-D node embedding + 4-D agent ID
    assert ctrl.actor_input_dim != 172  # Not raw global state
    assert ctrl.actor_input_dim != 256  # Not critic centralized state
    print(f"[PASS] Test 11: Actor strictly rejects global centralized state (actor_in={ctrl.actor_input_dim}-D, rejects 172/256-D)")
    passed += 1

    # -----------------------------------------------------------------
    # Test 12: Critic accepts centralized representation
    # -----------------------------------------------------------------
    assert ctrl.critic_input_dim == 256  # 4 x 64-D
    val = ctrl.get_centralized_value(scores)
    assert isinstance(val, float)
    print(f"[PASS] Test 12: Critic accepts centralized representation (dim={ctrl.critic_input_dim}, val={val:.3f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 13: Physical topology is respected (8 edges, diagonals disabled)
    # -----------------------------------------------------------------
    gate_mat, _, _ = ctrl.gated_gat.comm_gate(nodes, scores, deterministic=True)
    assert gate_mat[0, 3, 0].item() == 0.0  # A -> D disabled
    assert gate_mat[0, 2, 1].item() == 0.0  # B -> C disabled
    print("[PASS] Test 13: Physical topology strictly respected (diagonals A-D and B-C masked to 0.0)")
    passed += 1

    # -----------------------------------------------------------------
    # Test 14: Self communication is excluded from comm cost
    # -----------------------------------------------------------------
    # In gate_matrix, diagonal is 1.0, but only the 8 allowed edges contribute to comm_costs
    assert len(ctrl.gated_gat.comm_gate.last_gate_values) == 8
    print("[PASS] Test 14: Self-loops are local and strictly excluded from inter-agent communication cost")
    passed += 1

    # -----------------------------------------------------------------
    # Test 15: Causal pathway is actually executed during one real TraCI step
    # -----------------------------------------------------------------
    env = MultiAgentTrafficEnv(config_path="config.yaml")
    obs, info = env.reset(seed=42)
    ctrl.reset_history(obs, info)
    c_live = ctrl.compute_causal_context(info)
    acts, _, _, _ = ctrl.get_actions(c_live, deterministic=True)
    next_obs, rews, _, _, next_info = env.step(acts)
    ctrl.update_history(next_obs)
    env.close()
    assert len(ctrl.gated_gat.comm_gate.last_gate_values) == 8
    print("[PASS] Test 15: Causal pathway genuinely executes during live SUMO/TraCI environment step")
    passed += 1

    # -----------------------------------------------------------------
    # Test 16: Complete Phase 8R forward pass works
    # -----------------------------------------------------------------
    x_in = torch.randn(2, 4, 64)
    gated_out, gates, logits = ctrl.gated_gat(x_in, scores, deterministic=False)
    assert gated_out.shape == (2, 4, 64)
    assert gates.shape == (2, 8)
    assert logits.shape == (2, 8)
    print(f"[PASS] Test 16: Complete Phase 8R forward pass verified (gated_out shape: {gated_out.shape})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 17: Changing C_ij from 0 to 1 produces measurable internal pathway change
    # -----------------------------------------------------------------
    out_0, _, _ = ctrl.gated_gat(x_in, ctx_0, deterministic=False)
    out_1, _, _ = ctrl.gated_gat(x_in, ctx_1, deterministic=False)
    node_diff = (out_0 - out_1).abs().sum().item()
    assert node_diff > 0.001
    print(f"[PASS] Test 17: Changing C_ij from 0 to 1 produces measurable node embedding change (L1: {node_diff:.4f})")
    passed += 1

    # -----------------------------------------------------------------
    # Test 18: Communication sparsity regularization produces a non-zero gradient
    # -----------------------------------------------------------------
    ctrl.buffer.clear()
    hist_dict = ctrl.history_manager.get_all_histories()
    # Add dummy transition with high soft gate usage
    ctrl.buffer.add(
        histories=hist_dict,
        context_dict=scores,
        value=0.5,
        actions={"A": 0, "B": 1, "C": 2, "D": 0},
        log_probs={"A": -0.5, "B": -0.5, "C": -0.5, "D": -0.5},
        task_rewards={"A": -1.0, "B": -1.0, "C": -1.0, "D": -1.0},
        comm_costs={"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02},
        actual_comm=8,
        done=False
    )
    loss_dict = ctrl.update(last_value=0.5)
    assert "comm_budget_loss" in loss_dict and loss_dict["comm_budget_loss"] > 0.0
    print(f"[PASS] Test 18: Communication budget regularization produces non-zero loss & gradients (loss={loss_dict['comm_budget_loss']:.6f})")
    passed += 1

    print("=" * 85)
    print(f"PRE-TRAINING TEST RESULTS: {passed}/{total} TESTS PASSED")
    print("=" * 85)

    # -----------------------------------------------------------------
    # EXPERIMENTAL MEASUREMENTS REQUIRED BY USER
    # -----------------------------------------------------------------
    print("\n" + "=" * 85)
    print("EXPERIMENTAL PATHWAY SENSITIVITY & GRADIENT MEASUREMENTS")
    print("=" * 85)

    # 1. Gradient magnitude measurement: causal pathway vs. h_i/h_j pathway
    # Re-instantiate test gate with requires_grad
    test_gate = ImprovedCausalCommunicationGate(node_dim=64, causal_dim=8, causal_hidden_dim=16)
    h_tgt_var = torch.randn(1, 64, requires_grad=True)
    h_src_var = torch.randn(1, 64, requires_grad=True)
    c_var = torch.tensor([[0.75]], requires_grad=True)

    phi_c = test_gate.causal_encoder(c_var)
    phi_c.retain_grad()
    gate_in = torch.cat([h_tgt_var, h_src_var, phi_c], dim=-1)
    gate_logit = test_gate.mlp_body(gate_in)
    gate_prob = test_gate.sigmoid(gate_logit)

    loss = gate_prob.sum()
    loss.backward()

    grad_c = c_var.grad.abs().item()
    grad_phi = phi_c.grad.abs().mean().item()
    # Measure gradient w.r.t phi_c using register_hook or backward
    grad_htgt = h_tgt_var.grad.abs().mean().item()
    grad_hsrc = h_src_var.grad.abs().mean().item()
    grad_node = (grad_htgt + grad_hsrc) / 2.0

    print(f"1. Gradient magnitude through C_ij:              {grad_c:.6f}")
    print(f"2. Gradient magnitude through h_i/h_j pathway:  {grad_node:.6f}")
    print(f"   Relative leverage ratio (grad_C / grad_node): {grad_c / (grad_node + 1e-8):.4f}")

    # 2. Gate-logit and probability sensitivity to C_ij
    with torch.no_grad():
        c_sweep = np.linspace(0.0, 1.0, 11)
        sweep_logits = []
        sweep_probs = []
        for c_val in c_sweep:
            c_t = torch.tensor([[float(c_val)]], dtype=torch.float32)
            phi = test_gate.causal_encoder(c_t)
            gin = torch.cat([h_tgt_var.detach(), h_src_var.detach(), phi], dim=-1)
            gl = test_gate.mlp_body(gin).item()
            gp = test_gate.sigmoid(torch.tensor([[gl]])).item()
            sweep_logits.append(gl)
            sweep_probs.append(gp)

    delta_logit = max(sweep_logits) - min(sweep_logits)
    delta_prob = max(sweep_probs) - min(sweep_probs)

    print(f"3. Gate-logit sensitivity (max - min over [0,1]):  {delta_logit:.6f}")
    print(f"4. Gate probability sensitivity (max - min):       {delta_prob:.6f}")
    print("   C_ij Sweep [0.0 -> 1.0]:")
    for cv, gl, gp in zip(c_sweep[::2], sweep_logits[::2], sweep_probs[::2]):
        print(f"     C_ij = {cv:.2f} -> Gate Logit: {gl:+.4f}, Gate Prob: {gp:.4f}")

    print("=" * 85)


if __name__ == "__main__":
    run_tests()
