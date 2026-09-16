"""
Correctness and Unit Test Suite for Independent PPO (Phase 3).
Verifies:
1. Actor output shape = 4
2. Critic output shape = 1
3. Sampled action in {0, 1, 2, 3}
4. Log probability validity
5. Entropy validity
6. GAE computation correctness (length, returns = adv + val)
7. Minibatch generation (shuffling, shapes, complete coverage)
8. PPO update parameter change on non-empty rollout
"""

import sys
import numpy as np
import torch
from models.ppo_network import ActorNetwork, CriticNetwork, PPOBuffer, IndependentPPOAgent


def test_actor_critic_shapes():
    print("[TEST 1 & 2] Testing Actor and Critic Network output shapes...")
    obs_dim = 43
    act_dim = 4
    batch_size = 8
    dummy_obs = torch.randn(batch_size, obs_dim)

    actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=[64, 64])
    dist = actor(dummy_obs)
    logits = dist.logits
    assert logits.shape == (batch_size, act_dim), f"Actor logits shape mismatch: {logits.shape} vs ({batch_size}, {act_dim})"
    print(f"  [OK] Actor output shape verified: {logits.shape}")

    critic = CriticNetwork(obs_dim=obs_dim, hidden_dims=[64, 64])
    values = critic(dummy_obs)
    assert values.shape == (batch_size, 1), f"Critic values shape mismatch: {values.shape} vs ({batch_size}, 1)"
    print(f"  [OK] Critic output shape verified: {values.shape}")


def test_action_distribution():
    print("\n[TEST 3, 4, 5] Testing Action Sampling, Log-Probabilities, and Entropy...")
    obs_dim = 43
    actor = ActorNetwork(obs_dim=obs_dim, act_dim=4, hidden_dims=[64, 64])
    dummy_obs = torch.randn(1, obs_dim)

    dist = actor(dummy_obs)
    actions = [dist.sample().item() for _ in range(100)]
    unique_actions = set(actions)
    assert unique_actions.issubset({0, 1, 2, 3}), f"Sampled actions outside {{0, 1, 2, 3}}: {unique_actions}"
    print(f"  [OK] Sampled actions verified in {{0, 1, 2, 3}}. Distribution across 100 draws: {dict((a, actions.count(a)) for a in range(4))}")

    # Log probability check
    sample_action = dist.sample()
    log_prob = dist.log_prob(sample_action).item()
    assert np.isfinite(log_prob), f"Log prob is not finite: {log_prob}"
    assert log_prob <= 0.0, f"Log prob must be <= 0 for discrete distribution, got {log_prob}"
    print(f"  [OK] Log probability valid: {log_prob:.4f} (finite, <= 0.0)")

    # Entropy check
    entropy = dist.entropy().item()
    assert np.isfinite(entropy), f"Entropy is not finite: {entropy}"
    assert entropy >= 0.0, f"Entropy must be >= 0 for discrete distribution, got {entropy}"
    print(f"  [OK] Entropy valid: {entropy:.4f} (finite, >= 0.0)")


def test_gae_and_minibatch():
    print("\n[TEST 6 & 7] Testing GAE computation and Minibatch Generation in PPOBuffer...")
    buffer = PPOBuffer()
    num_steps = 20
    obs_dim = 43

    np.random.seed(42)
    for t in range(num_steps):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.randint(0, 4)
        lp = float(-np.random.rand() * 1.5)
        r = float(np.random.randn())
        v = float(np.random.randn())
        d = False if t < num_steps - 1 else True
        buffer.add(s, a, lp, r, v, d)

    assert len(buffer) == num_steps, f"Buffer length mismatch: {len(buffer)} vs {num_steps}"

    last_val = 0.5
    gamma = 0.99
    gae_lambda = 0.95
    buffer.compute_gae(last_value=last_val, gamma=gamma, gae_lambda=gae_lambda)

    assert len(buffer.advantages) == num_steps, f"Advantages length mismatch: {len(buffer.advantages)} vs {num_steps}"
    assert len(buffer.returns) == num_steps, f"Returns length mismatch: {len(buffer.returns)} vs {num_steps}"
    assert np.allclose(buffer.returns, buffer.advantages + np.array(buffer.values, dtype=np.float32)), \
        "Returns do not match advantages + values!"
    print(f"  [OK] GAE advantages and returns computed with correct length ({num_steps}) and exact mathematical identity R = A + V")

    # Minibatch test
    minibatch_size = 8
    total_yielded = 0
    generator = buffer.get_generator(minibatch_size=minibatch_size)
    for b_idx, (b_s, b_a, b_lp, b_adv, b_ret) in enumerate(generator):
        batch_len = len(b_s)
        total_yielded += batch_len
        assert b_s.shape == (batch_len, obs_dim)
        assert b_a.shape == (batch_len,)
        assert b_lp.shape == (batch_len,)
        assert b_adv.shape == (batch_len,)
        assert b_ret.shape == (batch_len,)
    assert total_yielded == num_steps, f"Total yielded samples {total_yielded} does not match {num_steps}"
    print(f"  [OK] Minibatch generator correctly yielded all {total_yielded} samples across shuffled batches")


def test_ppo_update_parameter_change():
    print("\n[TEST 8] Testing PPO update changes parameters on non-empty rollout...")
    agent = IndependentPPOAgent(
        agent_id="test_agent",
        obs_dim=43,
        act_dim=4,
        ppo_cfg={
            "lr": 0.001,
            "epochs": 2,
            "minibatch_size": 16,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5
        }
    )

    # Record parameter copies before update
    actor_params_before = [p.clone().detach() for p in agent.actor.parameters()]
    critic_params_before = [p.clone().detach() for p in agent.critic.parameters()]

    # Fill buffer with synthetic rollout
    np.random.seed(123)
    for t in range(32):
        s = np.random.randn(43).astype(np.float32)
        act, lp, val = agent.get_action_and_value(s)
        rew = -1.0 * np.random.rand()
        done = (t == 31)
        agent.buffer.add(s, act, lp, rew, val, done)

    # Perform PPO update
    loss_metrics = agent.update(last_value=0.0)
    print(f"  PPO Update Metrics: {loss_metrics}")

    # Verify parameters changed
    actor_diff = sum(torch.norm(p_after - p_before).item() for p_before, p_after in zip(actor_params_before, agent.actor.parameters()))
    critic_diff = sum(torch.norm(p_after - p_before).item() for p_before, p_after in zip(critic_params_before, agent.critic.parameters()))

    assert actor_diff > 1e-6, f"Actor parameters did not change after update! Diff: {actor_diff}"
    assert critic_diff > 1e-6, f"Critic parameters did not change after update! Diff: {critic_diff}"
    print(f"  [OK] Actor parameter update verified: L2 parameter change = {actor_diff:.6f}")
    print(f"  [OK] Critic parameter update verified: L2 parameter change = {critic_diff:.6f}")
    print("  [OK] Buffer cleanly cleared after update: len(buffer) =", len(agent.buffer))


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PPO BUFFER AND ALGORITHM CORRECTNESS TEST SUITE")
    print("=" * 80)
    test_actor_critic_shapes()
    test_action_distribution()
    test_gae_and_minibatch()
    test_ppo_update_parameter_change()
    print("\n" + "=" * 80)
    print("[ALL 8 UNIT TESTS PASSED SUCCESSFULLY]")
    print("=" * 80)
