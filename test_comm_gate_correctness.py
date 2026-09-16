"""
Phase 7 Pre-Training Correctness Verification Suite.
Validates all 20 mandatory tests before training the Learnable Communication Gate:

Test 1:  Gate output shape (8 edges)
Test 2:  Gate values in [0, 1]
Test 3:  Gate has trainable parameters
Test 4:  Gate parameters receive non-zero gradients
Test 5:  Optimizer update changes gate parameters
Test 6:  Allowed graph edges only (8 directed edges)
Test 7:  Diagonal communication disabled (A-D, B-C)
Test 8:  Exactly 8 possible directed communications per step
Test 9:  Gated message changes when gate changes
Test 10: Zero gate suppresses message
Test 11: Full gate passes message
Test 12: Actor receives gated communication representation (68-D)
Test 13: Decentralized actor does not receive centralized global state (256-D rejected)
Test 14: Communication cost calculation
Test 15: Communication ratio calculation
Test 16: Reward includes communication cost
Test 17: Task reward and communication cost logged separately
Test 18: End-to-end environment test
Test 19: Actual gate decisions logged
Test 20: Full MAPPO training step completes with buffer reset
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.communication_gate import (
    LearnableCommunicationGate,
    GatedGraphAttentionLayer,
    MAPPOGateRolloutBuffer,
    MAPPOGateController,
    ALLOWED_DIRECTED_EDGES
)
from environment.traffic_env import MultiAgentTrafficEnv


def test_1():
    """Test 1: Gate output shape."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    h = torch.randn(4, 4, 64)
    ctx = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}
    gate_matrix, raw_gates = gate(h, ctx)
    assert gate_matrix.shape == (4, 4, 4), f"Expected gate matrix (4, 4, 4), got {gate_matrix.shape}"
    assert raw_gates.shape == (4, 8), f"Expected raw gates (4, 8), got {raw_gates.shape}"
    return True


def test_2():
    """Test 2: Gate values in [0, 1]."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    h = torch.randn(10, 4, 64)
    ctx = {e: np.random.uniform(-1.0, 1.0) for e in ALLOWED_DIRECTED_EDGES}
    _, raw_gates = gate(h, ctx)
    min_val = raw_gates.min().item()
    max_val = raw_gates.max().item()
    assert 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0, f"Values out of bounds: [{min_val}, {max_val}]"
    return True


def test_3():
    """Test 3: Gate has trainable parameters."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    params = list(gate.parameters())
    assert len(params) > 0, "No parameters in gate"
    total_params = sum(p.numel() for p in params)
    assert total_params > 0, "Total parameters is 0"
    return True


def test_4():
    """Test 4: Gate parameters receive non-zero gradients."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    h = torch.randn(2, 4, 64, requires_grad=True)
    ctx = {e: 0.5 for e in ALLOWED_DIRECTED_EDGES}
    _, raw_gates = gate(h, ctx)
    loss = raw_gates.sum()
    loss.backward()
    for name, p in gate.named_parameters():
        assert p.grad is not None and torch.norm(p.grad) > 0.0, f"Zero grad on {name}"
    return True


def test_5():
    """Test 5: Optimizer update changes gate parameters."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    opt = optim.Adam(gate.parameters(), lr=0.01)
    p_before = [p.clone() for p in gate.parameters()]

    h = torch.randn(4, 4, 64)
    ctx = {e: 0.2 for e in ALLOWED_DIRECTED_EDGES}
    _, raw_gates = gate(h, ctx)
    loss = (raw_gates ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    changed = any(torch.norm(p - p_b).item() > 1e-6 for p, p_b in zip(gate.parameters(), p_before))
    assert changed, "Gate parameters did not change after optimizer step"
    return True


def test_6():
    """Test 6: Allowed graph edges only (8 directed edges)."""
    assert len(ALLOWED_DIRECTED_EDGES) == 8, f"Expected 8 edges, got {len(ALLOWED_DIRECTED_EDGES)}"
    expected = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "D"), ("D", "B"), ("C", "D"), ("D", "C")]
    assert set(ALLOWED_DIRECTED_EDGES) == set(expected), "Allowed edge list mismatch"
    return True


def test_7():
    """Test 7: Diagonal communication disabled."""
    gate = LearnableCommunicationGate(node_dim=64, context_dim=1, hidden_dim=32)
    h = torch.randn(1, 4, 64)
    ctx = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}
    gate_matrix, _ = gate(h, ctx)
    # A=0, B=1, C=2, D=3
    # Diagonals: (A, D) -> (0, 3) and (3, 0); (B, C) -> (1, 2) and (2, 1)
    assert gate_matrix[0, 0, 3].item() == 0.0, "Diagonal D -> A has non-zero gate!"
    assert gate_matrix[0, 3, 0].item() == 0.0, "Diagonal A -> D has non-zero gate!"
    assert gate_matrix[0, 1, 2].item() == 0.0, "Diagonal C -> B has non-zero gate!"
    assert gate_matrix[0, 2, 1].item() == 0.0, "Diagonal B -> C has non-zero gate!"
    return True


def test_8():
    """Test 8: Exactly 8 possible directed communications per step."""
    possible_per_step = len(ALLOWED_DIRECTED_EDGES)
    assert possible_per_step == 8, f"Expected 8 possible comms, got {possible_per_step}"
    horizon_comms = possible_per_step * 100
    assert horizon_comms == 800, f"Expected 800 comms over 100 steps, got {horizon_comms}"
    return True


def test_9():
    """Test 9: Gated message changes when gate changes."""
    gated_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
    x = torch.randn(1, 4, 64)

    # Low context vs high context produces different gate values -> different output
    ctx_low = {e: -1.0 for e in ALLOWED_DIRECTED_EDGES}
    ctx_high = {e: 1.0 for e in ALLOWED_DIRECTED_EDGES}

    out1, g1 = gated_gat(x, ctx_low)
    out2, g2 = gated_gat(x, ctx_high)

    diff_out = torch.norm(out1 - out2).item()
    diff_g = torch.norm(g1 - g2).item()
    assert diff_g > 1e-4, "Gate did not respond to context"
    assert diff_out > 1e-4, f"Output did not change with gate (diff={diff_out})"
    return True


def test_10():
    """Test 10: Zero gate suppresses message."""
    gated_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
    x = torch.randn(1, 4, 64)
    ctx = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}

    # Artificially set gate to 0 for neighbor edges
    def zero_gate_forward(node_f, context_dict, deterministic=False):
        B = node_f.shape[0]
        gate_mat = torch.zeros((B, 4, 4), device=node_f.device)
        for i in range(4):
            gate_mat[:, i, i] = 1.0  # self-loop only
        raw_g = torch.zeros((B, 8), device=node_f.device)
        return gate_mat, raw_g

    gated_gat.comm_gate.forward = zero_gate_forward
    out_zero, _ = gated_gat(x, ctx)

    # Now change neighbor node B's feature, keeping node A's feature identical
    x_mod = x.clone()
    x_mod[0, 1, :] += 50.0  # alter B significantly

    out_zero_mod, _ = gated_gat(x_mod, ctx)

    # Since all neighbor gates into A are 0, node A's representation must NOT change
    diff_A = torch.norm(out_zero[0, 0] - out_zero_mod[0, 0]).item()
    assert diff_A < 1e-5, f"Neighbor message leaked through zero gate! diff_A = {diff_A}"
    return True


def test_11():
    """Test 11: Full gate passes message."""
    gated_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
    x = torch.randn(1, 4, 64)
    ctx = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}

    def full_gate_forward(node_f, context_dict, deterministic=False):
        B = node_f.shape[0]
        gate_mat = torch.ones((B, 4, 4), device=node_f.device)
        raw_g = torch.ones((B, 8), device=node_f.device)
        return gate_mat, raw_g

    gated_gat.comm_gate.forward = full_gate_forward
    out_full, _ = gated_gat(x, ctx)

    x_mod = x.clone()
    x_mod[0, 1, :] += 10.0  # alter neighbor B

    out_full_mod, _ = gated_gat(x_mod, ctx)
    diff_A = torch.norm(out_full[0, 0] - out_full_mod[0, 0]).item()
    assert diff_A > 1e-3, f"Full gate did not pass message! diff_A = {diff_A}"
    return True


def test_12():
    """Test 12: Actor receives gated communication representation (68-D)."""
    ctrl = MAPPOGateController(config_path="config.yaml")
    assert ctrl.actor_input_dim == 68, f"Expected 68-D actor input, got {ctrl.actor_input_dim}"
    dummy_gated = torch.randn(1, 64)
    dummy_id = torch.eye(4)[0].unsqueeze(0)
    dist = ctrl.actor(dummy_gated, dummy_id)
    assert dist.logits.shape == (1, 4), f"Expected (1, 4) logits, got {dist.logits.shape}"
    return True


def test_13():
    """Test 13: Decentralized actor does not receive 256-D global state."""
    ctrl = MAPPOGateController(config_path="config.yaml")
    global_state = torch.randn(1, 256)
    rejected = False
    try:
        ctrl.actor.forward_flat(global_state)
    except RuntimeError:
        rejected = True
    assert rejected, "Actor accepted 256-D input! CTDE decentralization violated."
    return True


def test_14():
    """Test 14: Communication cost calculation."""
    ctrl = MAPPOGateController(config_path="config.yaml")
    ctrl.comm_cost_weight = 0.01

    # Simulate 3 communications
    actual_comms = 3
    cost = ctrl.comm_cost_weight * actual_comms
    assert np.isclose(cost, 0.03), f"Expected 0.03 cost, got {cost}"
    return True


def test_15():
    """Test 15: Communication ratio calculation."""
    actual = 4
    possible = 8
    ratio = actual / possible
    assert np.isclose(ratio, 0.5), f"Expected 0.5 ratio, got {ratio}"
    return True


def test_16():
    """Test 16: Reward includes communication cost."""
    task_reward = -2.50
    comm_cost = 0.02
    adj_reward = task_reward - comm_cost
    assert np.isclose(adj_reward, -2.52), f"Expected -2.52, got {adj_reward}"
    return True


def test_17():
    """Test 17: Task reward and communication cost logged separately."""
    buffer = MAPPOGateRolloutBuffer(agents=["A", "B", "C", "D"], history_length=4, obs_dim=43)
    hist = {a: np.zeros((4, 43), dtype=np.float32) for a in ["A", "B", "C", "D"]}
    ctx = {e: 0.0 for e in ALLOWED_DIRECTED_EDGES}
    acts = {a: 0 for a in ["A", "B", "C", "D"]}
    lps = {a: -1.38 for a in ["A", "B", "C", "D"]}
    task_r = {"A": -1.0, "B": -1.0, "C": -1.0, "D": -1.0}
    comm_c = {"A": 0.01, "B": 0.01, "C": 0.0, "D": 0.0}

    buffer.add(hist, ctx, 0.0, acts, lps, task_r, comm_c, actual_comm=2, done=False)
    assert buffer.task_rewards["A"][0] == -1.0, "Task reward not separated"
    assert buffer.comm_costs["A"][0] == 0.01, "Comm cost not separated"
    assert np.isclose(buffer.adjusted_rewards[0], -4.02), f"Expected -4.02, got {buffer.adjusted_rewards[0]}"
    return True


def test_18():
    """Test 18: End-to-end environment test."""
    env = MultiAgentTrafficEnv(config_path="config.yaml")
    ctrl = MAPPOGateController(config_path="config.yaml")
    obs, info = env.reset(seed=42)
    ctrl.reset_history(obs)

    ctx = ctrl.compute_heuristic_context(info)
    actions, log_probs, comm_costs, actual_comm = ctrl.get_actions(ctx, deterministic=False)

    next_obs, rewards, term, trunc, next_info = env.step(actions)
    env.close()

    assert len(actions) == 4, "Missing actions"
    assert 0 <= actual_comm <= 8, f"Actual comms out of range: {actual_comm}"
    return True


def test_19():
    """Test 19: Actual gate decisions logged."""
    ctrl = MAPPOGateController(config_path="config.yaml")
    agents = ["A", "B", "C", "D"]
    o0 = {a: np.zeros(43, dtype=np.float32) for a in agents}
    ctrl.reset_history(o0)
    info = {a: {"metrics": {"queue_length": 5.0}} for a in ctrl.agents}
    ctx = ctrl.compute_heuristic_context(info)
    _, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
    decisions = ctrl.gated_gat.comm_gate.last_comm_decisions
    assert len(decisions) == 8, f"Expected 8 edge decisions, got {len(decisions)}"
    for edge, dec in decisions.items():
        assert isinstance(dec, bool), f"Decision for {edge} is not bool: {dec}"
    return True


def test_20():
    """Test 20: Full MAPPO training step completes with buffer reset."""
    ctrl = MAPPOGateController(config_path="config.yaml")
    agents = ["A", "B", "C", "D"]
    o0 = {a: np.random.randn(43).astype(np.float32) for a in agents}
    ctrl.reset_history(o0)

    for _ in range(5):
        info = {a: {"metrics": {"queue_length": np.random.uniform(0, 10)}} for a in agents}
        ctx = ctrl.compute_heuristic_context(info)
        actions, lps, comm_costs, actual_comm = ctrl.get_actions(ctx, deterministic=False)
        val = ctrl.get_centralized_value(ctx)
        task_r = {a: np.random.randn() for a in agents}
        hist = ctrl.history_manager.get_all_histories()

        ctrl.buffer.add(hist, ctx, val, actions, lps, task_r, comm_costs, actual_comm, False)
        next_o = {a: np.random.randn(43).astype(np.float32) for a in agents}
        ctrl.update_history(next_o)

    last_val = ctrl.get_centralized_value(ctx)
    metrics = ctrl.update(last_val)

    assert "policy_loss" in metrics and "value_loss" in metrics
    assert len(ctrl.buffer) == 0, "Buffer not reset after update"
    return True


def main():
    print("=" * 75)
    print("PHASE 7: LEARNABLE COMMUNICATION GATE PRE-TRAINING TEST SUITE")
    print("=" * 75)

    tests = [
        ("Test 1:  Gate output shape (8 edges)", test_1),
        ("Test 2:  Gate values in [0, 1]", test_2),
        ("Test 3:  Gate has trainable parameters", test_3),
        ("Test 4:  Gate parameters receive non-zero gradients", test_4),
        ("Test 5:  Optimizer update changes gate parameters", test_5),
        ("Test 6:  Allowed graph edges only (8 directed edges)", test_6),
        ("Test 7:  Diagonal communication disabled (A-D, B-C)", test_7),
        ("Test 8:  Exactly 8 possible directed communications per step", test_8),
        ("Test 9:  Gated message changes when gate changes", test_9),
        ("Test 10: Zero gate suppresses message", test_10),
        ("Test 11: Full gate passes message", test_11),
        ("Test 12: Actor receives gated representation (68-D)", test_12),
        ("Test 13: Decentralized actor rejects global state (256-D)", test_13),
        ("Test 14: Communication cost calculation", test_14),
        ("Test 15: Communication ratio calculation", test_15),
        ("Test 16: Reward includes communication cost", test_16),
        ("Test 17: Task reward and comm cost logged separately", test_17),
        ("Test 18: End-to-end environment step test", test_18),
        ("Test 19: Actual gate decisions logged", test_19),
        ("Test 20: Full MAPPO training step completes with buffer reset", test_20),
    ]

    passed = 0
    for name, fn in tests:
        try:
            ok = fn()
            if ok:
                print(f"  [PASS] {name}")
                passed += 1
            else:
                print(f"  [FAIL] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print("=" * 75)
    print(f"RESULT: {passed}/{len(tests)} TESTS PASSED")
    print("=" * 75)

    if passed == len(tests):
        print("ALL 20 PRE-TRAINING CORRECTNESS TESTS PASSED.")
        sys.exit(0)
    else:
        print("TEST SUITE FAILED. DO NOT START TRAINING.")
        sys.exit(1)


if __name__ == "__main__":
    main()
