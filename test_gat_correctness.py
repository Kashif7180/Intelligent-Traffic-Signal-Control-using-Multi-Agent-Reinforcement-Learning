"""
Pre-Training Correctness Verification Suite for Phase 6:
MAPPO + Temporal Transformer + Graph Attention Network (GAT).

Validates all 16 mandatory tests defined in Phase 6 requirements:
Test 1:  Graph topology & physical adjacency (4 nodes, A-B, A-C, B-D, C-D, no diagonals)
Test 2:  GAT input dimension [B, 4, 64] accepted
Test 3:  GAT output dimension [B, 4, 64] produced
Test 4:  Attention coefficients exist for every allowed edge
Test 5:  Attention normalization: sum_j alpha_ij == 1.0 for each target node i
Test 6:  Physical adjacency masking: non-neighbor edges receive alpha_ij == 0.0
Test 7:  Gradient flow: all GAT trainable parameters receive non-zero gradients
Test 8:  Optimizer update: GAT parameters change after an optimizer step
Test 9:  Actor input dimension: 64 + 4 = 68-D
Test 10: Critic input dimension: 4 * 64 = 256-D
Test 11: Decentralization: actor receives only local 68-D, not 256-D global state
Test 12: End-to-end temporal + GAT pipeline: 43-D -> [4, 64] -> [4, 64] -> 68-D -> 4
Test 13: Attention inspectability: extractable without altering model weights
Test 14: Attention changes with input: dynamic response to different node inputs
Test 15: Causal temporal history: no future leakage [o(t-k+1), ..., o(t)]
Test 16: Complete forward/backward PPO update across all components
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.gat_network import (
    build_physical_adjacency_matrix,
    GraphAttentionLayer,
    DecentralizedGATActor,
    CentralizedGATCritic,
    MAPPOGATRolloutBuffer,
    MAPPOGATController
)
from models.transformer_network import TemporalHistoryManager, TemporalTransformerEncoder


def run_test_1():
    """Test 1: Graph topology & physical adjacency."""
    adj = build_physical_adjacency_matrix(agents=["A", "B", "C", "D"], add_self_loops=True)
    assert adj.shape == (4, 4), f"Expected shape (4, 4), got {adj.shape}"

    # Self loops
    for i in range(4):
        assert adj[i, i].item() == 1.0, f"Missing self-loop at index {i}"

    # Expected edges (target i, source j)
    # A (0): A(0), B(1), C(2)
    assert adj[0, 0] == 1 and adj[0, 1] == 1 and adj[0, 2] == 1 and adj[0, 3] == 0, "Node A edges incorrect"
    # B (1): B(1), A(0), D(3)
    assert adj[1, 0] == 1 and adj[1, 1] == 1 and adj[1, 2] == 0 and adj[1, 3] == 1, "Node B edges incorrect"
    # C (2): C(2), A(0), D(3)
    assert adj[2, 0] == 1 and adj[2, 1] == 0 and adj[2, 2] == 1 and adj[2, 3] == 1, "Node C edges incorrect"
    # D (3): D(3), B(1), C(2)
    assert adj[3, 0] == 0 and adj[3, 1] == 1 and adj[3, 2] == 1 and adj[3, 3] == 1, "Node D edges incorrect"

    # Verify diagonals A-D and B-C are strictly zero
    assert adj[0, 3] == 0.0 and adj[3, 0] == 0.0, "Diagonal A-D exists in graph!"
    assert adj[1, 2] == 0.0 and adj[2, 1] == 0.0, "Diagonal B-C exists in graph!"
    return True


def run_test_2():
    """Test 2: GAT input dimensions [B, 4, 64] accepted."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(8, 4, 64)
    out = gat(x)
    assert out is not None, "GAT forward pass failed on [8, 4, 64]"
    return True


def run_test_3():
    """Test 3: GAT output dimensions [B, 4, 64] produced."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(5, 4, 64)
    out = gat(x)
    assert out.shape == (5, 4, 64), f"Expected shape (5, 4, 64), got {out.shape}"
    return True


def run_test_4():
    """Test 4: Attention coefficients exist for every allowed edge."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(2, 4, 64)
    _ = gat(x)
    weights = gat.get_attention_weights()

    assert "per_head" in weights and "aggregated" in weights, "Missing per_head or aggregated attention"
    # 4 self loops + 8 directional edges = 12 allowed edges
    assert len(weights["aggregated"]) == 12, f"Expected 12 allowed edges, got {len(weights['aggregated'])}"
    for edge in weights["aggregated"]:
        assert edge["weight"] > 0.0, f"Edge {edge['source']}->{edge['target']} has zero or negative weight: {edge['weight']}"
    return True


def run_test_5():
    """Test 5: Attention normalization: sum_j alpha_ij == 1.0 for each target node."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(3, 4, 64)
    _ = gat(x)
    alpha = gat.last_attention_weights  # [B, K, N, N]
    # Sum across source nodes (dim=-1)
    sums = alpha.sum(dim=-1)  # [B, K, N]
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=1e-5), f"Attention weights do not sum to 1.0: max diff = {torch.max(torch.abs(sums - ones)).item()}"
    return True


def run_test_6():
    """Test 6: Physical adjacency masking: non-neighbor nodes receive alpha_ij == 0.0."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(2, 4, 64)
    _ = gat(x)
    alpha = gat.last_attention_weights  # [B, K, N, N]

    # Indices: A:0, B:1, C:2, D:3
    # Non-neighbors: (A, D) -> (0, 3) and (3, 0); (B, C) -> (1, 2) and (2, 1)
    non_edges = [(0, 3), (3, 0), (1, 2), (2, 1)]
    for i, j in non_edges:
        weights = alpha[:, :, i, j]
        max_val = weights.abs().max().item()
        assert max_val < 1e-6, f"Non-edge ({i}, {j}) has non-zero attention weight: {max_val}"
    return True


def run_test_7():
    """Test 7: Gradient flow: all GAT trainable parameters receive non-zero gradients."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = gat(x)
    loss = out.sum()
    loss.backward()

    assert gat.W.grad is not None and torch.norm(gat.W.grad) > 0.0, "Zero grad on W"
    assert gat.a_src.grad is not None and torch.norm(gat.a_src.grad) > 0.0, "Zero grad on a_src"
    assert gat.a_dst.grad is not None and torch.norm(gat.a_dst.grad) > 0.0, "Zero grad on a_dst"
    return True


def run_test_8():
    """Test 8: Optimizer update: at least one GAT parameter changes after an optimizer step."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    optimizer = optim.Adam(gat.parameters(), lr=0.01)

    w_before = gat.W.clone()
    a_src_before = gat.a_src.clone()

    x = torch.randn(4, 4, 64)
    out = gat(x)
    loss = (out ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    diff_w = torch.norm(gat.W - w_before).item()
    diff_src = torch.norm(gat.a_src - a_src_before).item()

    assert diff_w > 1e-6, f"W did not change after optimizer step (diff={diff_w})"
    assert diff_src > 1e-6, f"a_src did not change after optimizer step (diff={diff_src})"
    return True


def run_test_9():
    """Test 9: Actor receives 64 + 4 = 68-D."""
    actor = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
    assert actor.input_dim == 68, f"Expected actor input_dim 68, got {actor.input_dim}"

    gat_emb = torch.randn(3, 64)
    agent_id = torch.eye(4)[0].unsqueeze(0).repeat(3, 1)  # [3, 4]
    dist = actor(gat_emb, agent_id)
    assert dist.logits.shape == (3, 4), f"Expected logits shape (3, 4), got {dist.logits.shape}"
    return True


def run_test_10():
    """Test 10: Critic receives 4 * 64 = 256-D."""
    critic = CentralizedGATCritic(num_agents=4, embedding_dim=64)
    assert critic.input_dim == 256, f"Expected critic input_dim 256, got {critic.input_dim}"

    global_gat = torch.randn(5, 256)
    val = critic(global_gat)
    assert val.shape == (5, 1), f"Expected critic output (5, 1), got {val.shape}"
    return True


def run_test_11():
    """Test 11: Decentralization: actor cannot directly receive the 256-D global state."""
    actor = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
    global_state = torch.randn(1, 256)

    # Passing 256-D directly to actor forward_flat should raise RuntimeError (shape mismatch)
    try:
        _ = actor.forward_flat(global_state)
        passed = False
    except RuntimeError:
        passed = True

    assert passed, "Actor accepted 256-D input without error! Decentralization violation."
    return True


def run_test_12():
    """Test 12: Full pipeline: 43-D -> history -> 64-D -> GAT -> 64-D -> actor 68-D -> 4."""
    k = 4
    obs_dim = 43
    encoder = TemporalTransformerEncoder(obs_dim=obs_dim, history_length=k, embedding_dim=64)
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    actor = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)

    # 4 agents with history [k, 43]
    agent_histories = torch.randn(4, k, obs_dim)
    temporal_embs = encoder(agent_histories)  # [4, 64]
    assert temporal_embs.shape == (4, 64)

    gat_embs = gat(temporal_embs.unsqueeze(0)).squeeze(0)  # [4, 64]
    assert gat_embs.shape == (4, 64)

    agent_id_one_hot = torch.eye(4)  # [4, 4]
    for i in range(4):
        dist = actor(gat_embs[i:i+1], agent_id_one_hot[i:i+1])
        assert dist.logits.shape == (1, 4)
    return True


def run_test_13():
    """Test 13: Attention inspectability without modifying model weights."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)
    w_copy = gat.W.clone()

    x = torch.randn(1, 4, 64)
    _ = gat(x)
    weights = gat.get_attention_weights()

    assert len(weights["summary"]) == 12
    # Verify model weights remained untouched
    assert torch.equal(gat.W, w_copy), "Weights changed during attention extraction"
    return True


def run_test_14():
    """Test 14: Attention changes with input: dynamic response to different node inputs."""
    gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4, concat_heads=False)

    x1 = torch.randn(1, 4, 64) * 0.1
    _ = gat(x1)
    weights1 = gat.last_attention_weights.clone()

    x2 = torch.randn(1, 4, 64) * 10.0
    _ = gat(x2)
    weights2 = gat.last_attention_weights.clone()

    diff = torch.norm(weights1 - weights2).item()
    assert diff > 1e-4, f"Attention weights did not respond to changed inputs (diff={diff})"
    return True


def run_test_15():
    """Test 15: Causal temporal history: no future leakage."""
    agents = ["A", "B", "C", "D"]
    mgr = TemporalHistoryManager(agents=agents, history_length=4, obs_dim=43)

    # Initial reset: repeats o0 4 times
    o0 = {a: np.full(43, 0.0, dtype=np.float32) for a in agents}
    mgr.reset(o0)

    for a in agents:
        h = mgr.get_agent_history(a)
        assert np.all(h == 0.0), "Initial history not properly repeated"

    # Step 1: feed o1
    o1 = {a: np.full(43, 1.0, dtype=np.float32) for a in agents}
    mgr.update(o1)
    for a in agents:
        h = mgr.get_agent_history(a)
        # Should be [0, 0, 0, 1]
        assert np.all(h[0] == 0.0) and np.all(h[1] == 0.0) and np.all(h[2] == 0.0) and np.all(h[3] == 1.0)

    # Step 2: feed o2
    o2 = {a: np.full(43, 2.0, dtype=np.float32) for a in agents}
    mgr.update(o2)
    for a in agents:
        h = mgr.get_agent_history(a)
        # Should be [0, 0, 1, 2]
        assert np.all(h[0] == 0.0) and np.all(h[1] == 0.0) and np.all(h[2] == 1.0) and np.all(h[3] == 2.0)
    return True


def run_test_16():
    """Test 16: Complete forward/backward PPO update across all components."""
    controller = MAPPOGATController(config_path="config.yaml")

    # Simulate 5 steps in buffer
    agents = ["A", "B", "C", "D"]
    initial_obs = {a: np.random.randn(43).astype(np.float32) for a in agents}
    controller.reset_history(initial_obs)

    for step in range(5):
        actions, log_probs = controller.get_actions(deterministic=False)
        val = controller.get_centralized_value()
        rewards = {a: np.random.randn() for a in agents}

        hist_dict = controller.history_manager.get_all_histories()
        controller.buffer.add(
            histories=hist_dict,
            value=val,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            done=False
        )

        next_obs = {a: np.random.randn(43).astype(np.float32) for a in agents}
        controller.update_history(next_obs)

    # Execute update
    last_val = controller.get_centralized_value()
    metrics = controller.update(last_val)

    assert "policy_loss" in metrics and "value_loss" in metrics
    assert not np.isnan(metrics["policy_loss"]), "policy_loss is NaN"
    assert not np.isnan(metrics["value_loss"]), "value_loss is NaN"
    assert len(controller.buffer) == 0, "Buffer not cleared after update"
    return True


def main():
    print("=" * 70)
    print("PHASE 6: GAT PRE-TRAINING CORRECTNESS VERIFICATION SUITE")
    print("=" * 70)

    tests = [
        ("Test 1:  Graph topology & physical adjacency", run_test_1),
        ("Test 2:  GAT input dimension [B, 4, 64] accepted", run_test_2),
        ("Test 3:  GAT output dimension [B, 4, 64] produced", run_test_3),
        ("Test 4:  Attention coefficients exist for every allowed edge", run_test_4),
        ("Test 5:  Attention normalization: sum_j alpha_ij == 1.0", run_test_5),
        ("Test 6:  Physical adjacency masking: non-neighbors get 0.0", run_test_6),
        ("Test 7:  Gradient flow to all GAT trainable parameters", run_test_7),
        ("Test 8:  Optimizer update: GAT parameters change after step", run_test_8),
        ("Test 9:  Actor input dimension: 64 + 4 = 68-D", run_test_9),
        ("Test 10: Critic input dimension: 4 * 64 = 256-D", run_test_10),
        ("Test 11: Decentralization: actor rejects 256-D global state", run_test_11),
        ("Test 12: End-to-end temporal + GAT pipeline", run_test_12),
        ("Test 13: Attention inspectability without weight alteration", run_test_13),
        ("Test 14: Attention changes dynamically with input", run_test_14),
        ("Test 15: Causal temporal history: no future leakage", run_test_15),
        ("Test 16: Complete forward/backward PPO update", run_test_16),
    ]

    passed_count = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                print(f"  [PASS] {name}")
                passed_count += 1
            else:
                print(f"  [FAIL] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print("=" * 70)
    print(f"RESULT: {passed_count}/{len(tests)} TESTS PASSED")
    print("=" * 70)

    if passed_count == len(tests):
        print("ALL PRE-TRAINING CORRECTNESS TESTS PASSED.")
        sys.exit(0)
    else:
        print("PRE-TRAINING VERIFICATION FAILED. DO NOT START TRAINING.")
        sys.exit(1)


if __name__ == "__main__":
    main()
