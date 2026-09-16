"""
Unit Test Suite for SAGE-Traffic Phase 4: MAPPO (CTDE) Architecture Correctness.

Verifies:
1. Actor input shape = 47 when shared (43 local obs + 4-dim one-hot agent ID)
2. Actor output = 4 action logits
3. Centralized critic input = 172 (concatenated 4 * 43 local obs, no agent ID)
4. Centralized critic output = 1 (scalar global value estimate V(S_t))
5. Actors cannot access global state during execution (structural decentralization)
6. Critic actually receives 172-dim global state during training
7. GAE and returns dimensions are correct (100-step trajectory)
8. PPO update changes parameters (both Centralized Critic and Decentralized Actor)
"""

import sys
import numpy as np
import torch
from models.mappo_network import (
    CentralizedCriticNetwork,
    DecentralizedActorNetwork,
    MAPPORolloutBuffer,
    MAPPOController
)


def test_1_and_2_actor_shared_shapes():
    print("[TEST 1 & 2] Verifying Decentralized Actor input/output shapes (shared mode)...")
    batch_size = 8
    actor_input_dim = 47  # 43 local obs + 4-dim agent ID
    act_dim = 4

    actor = DecentralizedActorNetwork(obs_dim=actor_input_dim, act_dim=act_dim, hidden_dims=[64, 64])
    
    # Check weight dimensions
    first_layer_weights = actor.network[0].weight.shape
    last_layer_weights = actor.network[-1].weight.shape
    assert first_layer_weights == (64, 47), f"Expected first layer weights (64, 47), got {first_layer_weights}"
    assert last_layer_weights == (4, 64), f"Expected last layer weights (4, 64), got {last_layer_weights}"

    dummy_input = torch.randn(batch_size, actor_input_dim)
    dist = actor(dummy_input)

    assert dist.logits.shape == (batch_size, 4), f"Actor logits shape mismatch: {dist.logits.shape} vs ({batch_size}, 4)"
    print(f"  [OK] Actor first layer weight shape: {first_layer_weights} -> input dim = 47")
    print(f"  [OK] Actor output logits shape: {dist.logits.shape} -> action dim = 4")


def test_3_and_4_critic_shapes():
    print("\n[TEST 3 & 4] Verifying Centralized Critic input/output shapes...")
    batch_size = 8
    global_dim = 172  # 4 * 43 concatenated local observations (NO agent ID)

    critic = CentralizedCriticNetwork(global_dim=global_dim, hidden_dims=[128, 64])

    # Check weight dimensions
    first_layer_weights = critic.network[0].weight.shape
    last_layer_weights = critic.network[-1].weight.shape
    assert first_layer_weights == (128, 172), f"Expected first layer (128, 172), got {first_layer_weights}"
    assert last_layer_weights == (1, 64), f"Expected last layer (1, 64), got {last_layer_weights}"

    dummy_global = torch.randn(batch_size, global_dim)
    val = critic(dummy_global)

    assert val.shape == (batch_size, 1), f"Critic shape mismatch: {val.shape} vs ({batch_size}, 1)"
    print(f"  [OK] Centralized Critic first layer weight shape: {first_layer_weights} -> input dim = 172")
    print(f"  [OK] Centralized Critic output shape: {val.shape} -> scalar value V(S_t) = 1")


def test_5_actors_cannot_access_global_state_during_execution():
    print("\n[TEST 5] Verifying Decentralized Actors cannot access global state during execution...")
    ctrl = MAPPOController("config.yaml")

    # Local observations only (43 dims each)
    obs_dict = {a: np.random.randn(43).astype(np.float32) for a in ctrl.agents}

    # Verify execution action selection
    actions, log_probs = ctrl.get_actions(obs_dict, deterministic=False)
    for a in ctrl.agents:
        assert actions[a] in {0, 1, 2, 3}, f"Invalid action: {actions[a]}"
        assert np.isfinite(log_probs[a]), f"Invalid log prob: {log_probs[a]}"
    print("  [OK] Actions sampled decentralized using strictly local observations (plus agent ID).")

    # Verify structural impossibility of passing 172-dim global state to actor
    global_state = ctrl.construct_global_state(obs_dict)
    assert global_state.shape == (172,)

    try:
        # Attempt to pass 172-dim global state directly to actor
        ctrl.shared_actor(torch.tensor(global_state, dtype=torch.float32).unsqueeze(0))
        assert False, "Actor accepted 172-dim global state! Structural decentralization violated!"
    except RuntimeError as e:
        print(f"  [OK] Structural check confirmed: Actor rejected 172-dim global state as expected ({e.__class__.__name__}).")


def test_6_critic_receives_global_state_during_training():
    print("\n[TEST 6] Verifying Centralized Critic receives strictly 172-dim global state during training...")
    buffer = MAPPORolloutBuffer(agents=["A", "B", "C", "D"])

    # Simulate 50 steps
    for t in range(50):
        obs_dict = {a: np.random.randn(43).astype(np.float32) for a in ["A", "B", "C", "D"]}
        g_s = np.concatenate([obs_dict[a] for a in ["A", "B", "C", "D"]])
        assert g_s.shape == (172,)
        val = float(np.random.randn())
        acts = {a: np.random.randint(0, 4) for a in ["A", "B", "C", "D"]}
        lps = {a: float(-np.random.rand()) for a in ["A", "B", "C", "D"]}
        rews = {a: float(-np.random.rand()) for a in ["A", "B", "C", "D"]}
        done = (t == 49)
        buffer.add(g_s, val, obs_dict, acts, lps, rews, done)

    buffer.compute_gae(last_value=0.0)

    # Check critic minibatch generator
    minibatch_gen = buffer.get_critic_minibatch_generator(minibatch_size=16)
    batch_count = 0
    for g_states, returns in minibatch_gen:
        batch_count += 1
        assert g_states.shape[1] == 172, f"Critic input dimension mismatch: {g_states.shape[1]} vs 172"
        assert returns.shape[1] == 1, f"Critic target dimension mismatch: {returns.shape[1]} vs 1"

    assert batch_count > 0
    print(f"  [OK] Critic minibatch verified: input shape = {g_states.shape} (strictly 172 dims, no agent ID).")


def test_7_gae_and_returns_dimensions():
    print("\n[TEST 7] Verifying GAE and returns dimensions for 100-step trajectory...")
    buffer = MAPPORolloutBuffer(agents=["A", "B", "C", "D"])
    num_steps = 100

    for t in range(num_steps):
        obs_dict = {a: np.random.randn(43).astype(np.float32) for a in ["A", "B", "C", "D"]}
        g_s = np.concatenate([obs_dict[a] for a in ["A", "B", "C", "D"]])
        val = float(-10.0 + np.random.randn())
        acts = {a: np.random.randint(0, 4) for a in ["A", "B", "C", "D"]}
        lps = {a: float(-0.5) for a in ["A", "B", "C", "D"]}
        rews = {a: float(-0.2) for a in ["A", "B", "C", "D"]}
        done = (t == num_steps - 1)
        buffer.add(g_s, val, obs_dict, acts, lps, rews, done)

    last_val = -10.0
    buffer.compute_gae(last_val, gamma=0.99, gae_lambda=0.95)

    assert buffer.advantages.shape == (100,), f"Advantages shape mismatch: {buffer.advantages.shape}"
    assert buffer.returns.shape == (100,), f"Returns shape mismatch: {buffer.returns.shape}"

    # Verify Bellman consistency: returns[t] - advantages[t] == values[t]
    diff = np.max(np.abs((buffer.returns - buffer.advantages) - np.array(buffer.values, dtype=np.float32)))
    assert diff < 1e-5, f"Bellman consistency violated: diff = {diff}"
    print(f"  [OK] GAE advantages shape: {buffer.advantages.shape}")
    print(f"  [OK] Centralized returns shape: {buffer.returns.shape}")
    print(f"  [OK] Bellman consistency verified (max diff = {diff:.2e})")


def test_8_ppo_update_changes_parameters():
    print("\n[TEST 8] Verifying PPO update genuine parameter changes...")
    
    # 8a: Shared policy mode
    print("  [8a] Testing Shared Policy MAPPO updates...")
    ctrl_shared = MAPPOController("config.yaml")
    ctrl_shared.share_policy = True

    actor_weights_before = [p.clone().detach() for p in ctrl_shared.shared_actor.parameters()]
    critic_weights_before = [p.clone().detach() for p in ctrl_shared.critic.parameters()]

    for t in range(32):
        o_dict = {a: np.random.randn(43).astype(np.float32) for a in ctrl_shared.agents}
        g_s = ctrl_shared.construct_global_state(o_dict)
        val = ctrl_shared.get_value(g_s)
        acts, lps = ctrl_shared.get_actions(o_dict)
        rews = {a: float(-np.random.rand()) for a in ctrl_shared.agents}
        done = (t == 31)
        ctrl_shared.buffer.add(g_s, val, o_dict, acts, lps, rews, done)

    metrics_shared = ctrl_shared.update(g_s)
    
    actor_diff = sum(torch.norm(p_after - p_before).item() for p_before, p_after in zip(actor_weights_before, ctrl_shared.shared_actor.parameters()))
    critic_diff = sum(torch.norm(p_after - p_before).item() for p_before, p_after in zip(critic_weights_before, ctrl_shared.critic.parameters()))

    assert actor_diff > 1e-6, "Shared actor parameters did not update!"
    assert critic_diff > 1e-6, "Centralized critic parameters did not update!"
    print(f"    [OK] Shared Actor L2 param change:       {actor_diff:.6f}")
    print(f"    [OK] Centralized Critic L2 param change: {critic_diff:.6f}")
    print(f"    [OK] Metrics: {metrics_shared}")

    # 8b: Individual policy mode
    print("  [8b] Testing Individual Policy MAPPO updates...")
    ctrl_indiv = MAPPOController("config.yaml")
    # Switch to individual mode
    ctrl_indiv.share_policy = False
    ctrl_indiv.actor_input_dim = 43
    ctrl_indiv.shared_actor = None
    ctrl_indiv.individual_actors = {
        a: DecentralizedActorNetwork(obs_dim=43, act_dim=4, hidden_dims=[64, 64])
        for a in ctrl_indiv.agents
    }
    ctrl_indiv.actor_optimizers = {
        a: torch.optim.Adam(ctrl_indiv.individual_actors[a].parameters(), lr=0.0003)
        for a in ctrl_indiv.agents
    }

    indiv_actors_before = {
        a: [p.clone().detach() for p in ctrl_indiv.individual_actors[a].parameters()]
        for a in ctrl_indiv.agents
    }

    for t in range(32):
        o_dict = {a: np.random.randn(43).astype(np.float32) for a in ctrl_indiv.agents}
        g_s = ctrl_indiv.construct_global_state(o_dict)
        val = ctrl_indiv.get_value(g_s)
        acts, lps = ctrl_indiv.get_actions(o_dict)
        rews = {a: float(-np.random.rand()) for a in ctrl_indiv.agents}
        done = (t == 31)
        ctrl_indiv.buffer.add(g_s, val, o_dict, acts, lps, rews, done)

    metrics_indiv = ctrl_indiv.update(g_s)

    for a in ctrl_indiv.agents:
        diff_a = sum(
            torch.norm(p_after - p_before).item()
            for p_before, p_after in zip(indiv_actors_before[a], ctrl_indiv.individual_actors[a].parameters())
        )
        assert diff_a > 1e-6, f"Individual actor {a} parameters did not update!"
        print(f"    [OK] Individual Actor {a} L2 param change: {diff_a:.6f}")
    print(f"    [OK] Individual mode metrics: {metrics_indiv}")


if __name__ == "__main__":
    print("=" * 80)
    print("SAGE-TRAFFIC PHASE 4: MAPPO ARCHITECTURE & CORRECTNESS UNIT TESTS")
    print("=" * 80)
    test_1_and_2_actor_shared_shapes()
    test_3_and_4_critic_shapes()
    test_5_actors_cannot_access_global_state_during_execution()
    test_6_critic_receives_global_state_during_training()
    test_7_gae_and_returns_dimensions()
    test_8_ppo_update_changes_parameters()
    print("\n" + "=" * 80)
    print("[ALL 8 MAPPO ARCHITECTURE & CORRECTNESS TESTS PASSED CLEANLY]")
    print("=" * 80)
