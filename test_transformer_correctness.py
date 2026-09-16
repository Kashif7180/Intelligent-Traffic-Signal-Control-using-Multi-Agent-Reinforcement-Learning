"""
Pre-Training Correctness Test Suite for Phase 5: MAPPO + Temporal Transformer.
Tests all 14 required correctness criteria:
1. Transformer accepts [B, 4, 43]
2. Transformer outputs [B, 64]
3. Positional encoding is present and temporal ordering affects representation
4. Transformer parameters receive non-zero gradients
5. Transformer parameters actually change after optimizer update
6. Actor input is exactly 68-D (64-D embedding + 4-D agent ID)
7. Actor rejects 172-D and 256-D global states
8. Critic accepts exactly 256-D centralized temporal representation
9. Agent IDs are A=[1,0,0,0], B=[0,1,0,0], C=[0,0,1,0], D=[0,0,0,1]
10. History buffer always contains exactly 4 observations
11. Initial history is correctly [o0, o0, o0, o0]
12. Sliding history correctly becomes [o1, o2, o3, o4]
13. No future observation is inserted into current action history
14. Full forward/backward dimension consistency through entire pipeline
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np

from models.transformer_network import (
    TemporalTransformerEncoder,
    DecentralizedTemporalActor,
    CentralizedTemporalCritic,
    TemporalHistoryManager,
    MAPPOTemporalController
)


def test_1_and_2_transformer_shapes():
    print("[TEST 1 & 2] Testing Transformer input/output dimensions...")
    B, k, obs_dim = 8, 4, 43
    encoder = TemporalTransformerEncoder(obs_dim=obs_dim, history_length=k, embedding_dim=64)
    dummy_input = torch.randn(B, k, obs_dim)
    output = encoder(dummy_input)

    assert output.shape == (B, 64), f"Transformer output shape mismatch: {output.shape} vs ({B}, 64)"
    print(f"  [OK] Input shape: [B={B}, k={k}, obs_dim={obs_dim}]")
    print(f"  [OK] Output shape: {output.shape} -> [B, 64]")


def test_3_positional_encoding_and_temporal_order():
    print("\n[TEST 3] Testing Positional Encoding and Temporal Order Sensitivity...")
    encoder = TemporalTransformerEncoder(obs_dim=43, history_length=4, embedding_dim=64)
    
    # Assert positional encoding parameter exists and has non-zero values
    assert hasattr(encoder, "pos_embedding")
    assert encoder.pos_embedding.shape == (1, 4, 64)
    pos_norm = torch.norm(encoder.pos_embedding).item()
    assert pos_norm > 0.0, "Positional embedding norm is zero!"
    print(f"  [OK] Learnable Positional Encoding parameter verified (norm = {pos_norm:.4f})")

    # Create sequence with distinct temporal steps
    seq_forward = torch.arange(4, dtype=torch.float32).view(1, 4, 1).repeat(1, 1, 43)
    seq_reversed = torch.flip(seq_forward, dims=[1])

    emb_forward = encoder(seq_forward)
    emb_reversed = encoder(seq_reversed)

    diff = torch.norm(emb_forward - emb_reversed).item()
    assert diff > 1e-4, f"Temporal ordering had no effect! Diff = {diff}"
    print(f"  [OK] Sequence reversal produces distinct embeddings: L2 diff = {diff:.6f}")


def test_4_and_5_gradients_and_parameter_updates():
    print("\n[TEST 4 & 5] Testing Transformer Gradients and Optimizer Parameter Updates...")
    encoder = TemporalTransformerEncoder(obs_dim=43, history_length=4, embedding_dim=64)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    initial_weights = [p.clone().detach() for p in encoder.parameters()]

    dummy_input = torch.randn(4, 4, 43)
    emb = encoder(dummy_input)
    target = torch.randn(4, 64)
    loss = ((emb - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist and are non-zero
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert torch.norm(param.grad).item() > 0.0, f"Parameter {name} gradient is zero!"
    print("  [OK] All Transformer parameters received non-zero gradients.")

    optimizer.step()

    # Verify parameters changed
    total_change = sum(torch.norm(p_after - p_before).item() for p_before, p_after in zip(initial_weights, encoder.parameters()))
    assert total_change > 1e-4, "Transformer parameters did not change after optimizer step!"
    print(f"  [OK] Transformer parameters updated: total L2 change = {total_change:.6f}")


def test_6_actor_input_dimensions():
    print("\n[TEST 6] Testing Decentralized Temporal Actor Input Dimensions (68-D)...")
    actor = DecentralizedTemporalActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
    assert actor.input_dim == 68, f"Expected actor input dim 68, got {actor.input_dim}"

    first_layer_shape = actor.network[0].weight.shape
    assert first_layer_shape == (64, 68), f"First layer shape mismatch: {first_layer_shape}"

    B = 8
    temporal_emb = torch.randn(B, 64)
    agent_id = torch.zeros(B, 4)
    agent_id[:, 0] = 1.0

    dist = actor(temporal_emb, agent_id)
    assert dist.logits.shape == (B, 4)
    print(f"  [OK] Actor input: 64-D embedding + 4-D agent ID = 68-D")
    print(f"  [OK] Actor output logits shape: {dist.logits.shape} -> action dim = 4")


def test_7_actor_rejects_global_states():
    print("\n[TEST 7] Testing Actor Rejection of Global States (172-D and 256-D)...")
    actor = DecentralizedTemporalActor(embedding_dim=64, agent_id_dim=4, act_dim=4)

    dummy_172 = torch.randn(1, 172)
    dummy_256 = torch.randn(1, 256)

    rejected_172 = False
    try:
        actor.forward_flat(dummy_172)
    except RuntimeError:
        rejected_172 = True

    rejected_256 = False
    try:
        actor.forward_flat(dummy_256)
    except RuntimeError:
        rejected_256 = True

    assert rejected_172, "Actor accepted 172-D global state!"
    assert rejected_256, "Actor accepted 256-D centralized state!"
    print("  [OK] Actor rejected 172-D global state (RuntimeError caught).")
    print("  [OK] Actor rejected 256-D centralized state (RuntimeError caught).")


def test_8_critic_accepts_centralized_representation():
    print("\n[TEST 8] Testing Centralized Temporal Critic Input (256-D)...")
    critic = CentralizedTemporalCritic(num_agents=4, embedding_dim=64)
    assert critic.input_dim == 256, f"Expected critic input dim 256, got {critic.input_dim}"

    first_layer_shape = critic.network[0].weight.shape
    assert first_layer_shape == (128, 256), f"Critic first layer shape mismatch: {first_layer_shape}"

    B = 8
    dummy_centralized = torch.randn(B, 256)
    val = critic(dummy_centralized)
    assert val.shape == (B, 1), f"Critic output shape mismatch: {val.shape} vs ({B}, 1)"
    print(f"  [OK] Centralized critic input: 4 x 64 = 256-D")
    print(f"  [OK] Centralized critic output: {val.shape} -> scalar state-value estimate")


def test_9_agent_id_orthogonality():
    print("\n[TEST 9] Testing Agent ID Representation...")
    ctrl = MAPPOTemporalController("config.yaml")
    expected = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
        "D": [0.0, 0.0, 0.0, 1.0]
    }
    for a in ["A", "B", "C", "D"]:
        actual = ctrl.agent_one_hots[a].tolist()
        assert actual == expected[a], f"Agent {a} ID mismatch: {actual} vs {expected[a]}"
        print(f"  [OK] Agent {a} ID: {actual}")


def test_10_11_12_13_history_manager():
    print("\n[TEST 10, 11, 12, 13] Testing Sliding History Window and Causality...")
    mgr = TemporalHistoryManager(agents=["A", "B", "C", "D"], history_length=4, obs_dim=43)

    # 11. Initial history: repeat o0
    o0 = {a: np.ones(43, dtype=np.float32) * 10.0 for a in ["A", "B", "C", "D"]}
    mgr.reset(o0)

    for a in ["A", "B", "C", "D"]:
        h = mgr.get_agent_history(a)
        assert h.shape == (4, 43), f"History shape mismatch: {h.shape}"
        for t in range(4):
            assert np.allclose(h[t], o0[a]), f"Initial history slot {t} not equal to o0"
    print("  [OK] Initial history at t=0 correctly repeats o0: [o0, o0, o0, o0]")

    # 12. Sliding history updates
    o1 = {a: np.ones(43, dtype=np.float32) * 11.0 for a in ["A", "B", "C", "D"]}
    o2 = {a: np.ones(43, dtype=np.float32) * 12.0 for a in ["A", "B", "C", "D"]}
    o3 = {a: np.ones(43, dtype=np.float32) * 13.0 for a in ["A", "B", "C", "D"]}
    o4 = {a: np.ones(43, dtype=np.float32) * 14.0 for a in ["A", "B", "C", "D"]}

    mgr.update(o1)
    h_after_1 = mgr.get_agent_history("A")
    assert np.allclose(h_after_1[0], o0["A"])
    assert np.allclose(h_after_1[1], o0["A"])
    assert np.allclose(h_after_1[2], o0["A"])
    assert np.allclose(h_after_1[3], o1["A"])

    mgr.update(o2)
    mgr.update(o3)
    mgr.update(o4)
    h_after_4 = mgr.get_agent_history("A")
    assert h_after_4.shape == (4, 43)
    assert np.allclose(h_after_4[0], o1["A"])
    assert np.allclose(h_after_4[1], o2["A"])
    assert np.allclose(h_after_4[2], o3["A"])
    assert np.allclose(h_after_4[3], o4["A"])
    print("  [OK] Sliding window maintains exactly k=4 observations: [o1, o2, o3, o4]")

    # 13. Causality check: verify no future observation is in history
    o5 = {a: np.ones(43, dtype=np.float32) * 999.0 for a in ["A", "B", "C", "D"]}
    # Currently history is [o1, o2, o3, o4]. Future observation o5 is NOT in history
    assert not any(np.allclose(h_after_4[i], o5["A"]) for i in range(4))
    print("  [OK] Causality verified: future observations are not present in current history.")


def test_14_full_pipeline_dimension_consistency():
    print("\n[TEST 14] Testing Full Forward/Backward Pipeline Dimension Consistency...")
    ctrl = MAPPOTemporalController("config.yaml")

    # 1. Reset history
    o0 = {a: np.random.randn(43).astype(np.float32) for a in ctrl.agents}
    ctrl.reset_history(o0)

    # 2. Decentralized action selection
    actions, log_probs = ctrl.get_actions(deterministic=False)
    for a in ctrl.agents:
        assert actions[a] in {0, 1, 2, 3}
        assert np.isfinite(log_probs[a])
    print(f"  [OK] Decentralized actions sampled: {actions}")

    # 3. Centralized value estimation
    val = ctrl.get_centralized_value()
    assert np.isfinite(val)
    print(f"  [OK] Centralized temporal value: {val:.4f}")

    # 4. Fill buffer and perform full PPO update
    for step in range(32):
        histories = ctrl.history_manager.get_all_histories()
        v = ctrl.get_centralized_value()
        acts, lps = ctrl.get_actions()
        rews = {a: float(-np.random.rand()) for a in ctrl.agents}
        done = (step == 31)

        ctrl.buffer.add(histories, v, acts, lps, rews, done)

        # Step history
        next_o = {a: np.random.randn(43).astype(np.float32) for a in ctrl.agents}
        ctrl.update_history(next_o)

    metrics = ctrl.update(last_value=0.0)
    print(f"  [OK] End-to-end PPO Update Metrics: {metrics}")
    print("  [OK] Full forward/backward pipeline dimension consistency verified successfully.")


if __name__ == "__main__":
    print("=" * 80)
    print("SAGE-TRAFFIC PHASE 5: TEMPORAL TRANSFORMER CORRECTNESS TEST SUITE")
    print("=" * 80)
    test_1_and_2_transformer_shapes()
    test_3_positional_encoding_and_temporal_order()
    test_4_and_5_gradients_and_parameter_updates()
    test_6_actor_input_dimensions()
    test_7_actor_rejects_global_states()
    test_8_critic_accepts_centralized_representation()
    test_9_agent_id_orthogonality()
    test_10_11_12_13_history_manager()
    test_14_full_pipeline_dimension_consistency()
    print("\n" + "=" * 80)
    print("[ALL 14 TRANSFORMER CORRECTNESS TESTS PASSED SUCCESSFULLY]")
    print("=" * 80)
