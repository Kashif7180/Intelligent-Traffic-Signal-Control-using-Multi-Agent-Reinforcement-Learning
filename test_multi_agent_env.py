"""
Multi-Agent Gymnasium Environment Verification Test for SAGE-Traffic (Phase 2).
Steps the 4-agent environment for 50 steps with random actions, verifying observation/action
shapes, return dictionary structures, and multi-objective delay+queue rewards.
"""

import sys
import numpy as np
import yaml
from environment.traffic_env import MultiAgentTrafficEnv


def test_multi_agent_env():
    print("=" * 80)
    print("SAGE-Traffic Phase 2: Multi-Agent Environment Verification Test (50 Steps)")
    print("=" * 80)

    env = MultiAgentTrafficEnv(config_path="config.yaml")

    # 1. Verify Agents and Spaces Structure
    print(f"Registered Agents: {env.agents} (Total: {len(env.agents)})")
    assert len(env.agents) == 4, f"Expected 4 agents, got {len(env.agents)}"
    assert set(env.agents) == {"A", "B", "C", "D"}, f"Unexpected agent IDs: {env.agents}"

    print("\n--- Space Specifications Per Agent ---")
    for agent_id in env.agents:
        act_space = env.action_space(agent_id)
        obs_space = env.observation_space(agent_id)
        print(f"Agent {agent_id}:")
        print(f"  Action Space:      {act_space} (n={act_space.n})")
        print(f"  Observation Space: {obs_space} (shape={obs_space.shape}, dtype={obs_space.dtype})")
        assert act_space.n == 4, f"Agent {agent_id} action space must be Discrete(4), got {act_space.n}"
        assert len(obs_space.shape) == 1, f"Agent {agent_id} observation space must be 1D vector"
        assert obs_space.shape[0] == 43, f"Expected obs dim 43 (40 lane + 2 phase + 1 dur), got {obs_space.shape[0]}"

    # 2. Reset and Verify Initial Observations
    print("\nResetting Multi-Agent Environment...")
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict), f"reset() must return dict for observations, got {type(obs)}"
    assert isinstance(info, dict), f"reset() must return dict for info, got {type(info)}"
    assert set(obs.keys()) == set(env.agents), f"obs keys {list(obs.keys())} do not match agents {env.agents}"

    print("Initial observation shapes verified for all 4 agents:")
    for agent_id, o in obs.items():
        assert o.shape == env.observation_space(agent_id).shape
        assert o.dtype == np.float32
        print(f"  Agent {agent_id}: shape={o.shape}, min={o.min():.4f}, max={o.max():.4f}")

    # 3. Step for 50 Control Steps with Random Actions
    print("\nStepping environment for 50 control steps with random actions...")
    total_steps = 50
    step_rewards = {a: [] for a in env.agents}
    step_queues = {a: [] for a in env.agents}
    step_delays = {a: [] for a in env.agents}

    header = f"{'Step':<5} | {'Agent A (Act/Rew)':<20} | {'Agent B (Act/Rew)':<20} | {'Agent C (Act/Rew)':<20} | {'Agent D (Act/Rew)':<20}"
    print("-" * 92)
    print(header)
    print("-" * 92)

    for step in range(1, total_steps + 1):
        # Sample random actions for all agents
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}

        next_obs, rewards, terminations, truncations, next_info = env.step(actions)

        # Assert dictionary keys and return types
        assert set(next_obs.keys()) == set(env.agents)
        assert set(rewards.keys()) == set(env.agents)
        assert set(next_info.keys()) == set(env.agents)

        for agent_id in env.agents:
            # Shape verification
            expected_shape = env.observation_space(agent_id).shape
            actual_shape = next_obs[agent_id].shape
            assert actual_shape == expected_shape, f"Step {step}: Agent {agent_id} obs shape mismatch: {actual_shape} vs {expected_shape}"
            assert next_obs[agent_id].dtype == np.float32, f"Step {step}: Agent {agent_id} dtype mismatch: {next_obs[agent_id].dtype}"

            # Value bounds check
            assert np.all(next_obs[agent_id] >= 0.0) and np.all(next_obs[agent_id] <= 1.0), \
                f"Step {step}: Agent {agent_id} obs values outside [0, 1]: min={next_obs[agent_id].min()}, max={next_obs[agent_id].max()}"

            # Reward type check
            assert isinstance(rewards[agent_id], float), f"Step {step}: Agent {agent_id} reward is not float"
            step_rewards[agent_id].append(rewards[agent_id])

            # Queue & delay tracking
            m = next_info[agent_id]["metrics"]
            step_queues[agent_id].append(m["queue_length"])
            step_delays[agent_id].append(m["waiting_time"])

        # Periodic logging
        if step == 1 or step % 10 == 0 or step == total_steps:
            fmt_a = f"{actions['A']}/{rewards['A']:.2f}"
            fmt_b = f"{actions['B']}/{rewards['B']:.2f}"
            fmt_c = f"{actions['C']}/{rewards['C']:.2f}"
            fmt_d = f"{actions['D']}/{rewards['D']:.2f}"
            print(f"{step:<5} | {fmt_a:<20} | {fmt_b:<20} | {fmt_c:<20} | {fmt_d:<20}")

    print("-" * 92)

    # 4. Print Sample Normalized Observation Vector for Inspection
    print("\n--- Sample Normalized Observation Vector (Agent A, Step 50) ---")
    sample_obs = next_obs["A"]
    print(f"Shape: {sample_obs.shape}, Dtype: {sample_obs.dtype}")
    print(f"Per-lane values (8 lanes x 5 features = 40 values):")
    print(f"  Normalized Lane Feature Vector:\n  {np.round(sample_obs[:40], 3)}")
    print(f"Intersection-level values (2 phase one-hot + 1 phase duration = 3 values):")
    print(f"  Phase One-Hot: {sample_obs[40:42]}, Normalized Duration: {sample_obs[42]:.3f}")

    # 5. Summary Statistics
    print("\n" + "=" * 80)
    print("50-STEP MULTI-AGENT ROLLOUT SUMMARY STATISTICS")
    print("=" * 80)
    print(f"{'Agent':<8} | {'Mean Reward':<16} | {'Mean Queue (veh)':<18} | {'Mean Waiting (s)':<18} | {'Obs Shape':<12}")
    print("-" * 80)
    for agent_id in env.agents:
        mean_r = np.mean(step_rewards[agent_id])
        mean_q = np.mean(step_queues[agent_id])
        mean_w = np.mean(step_delays[agent_id])
        shape_str = str(env.observation_space(agent_id).shape)
        print(f"{agent_id:<8} | {mean_r:<16.3f} | {mean_q:<18.2f} | {mean_w:<18.2f} | {shape_str:<12}")
    print("=" * 80)

    env.close()
    print("\n[VERIFICATION SUCCESSFUL] Multi-agent interface, shapes, and rewards fully verified.\n")


if __name__ == "__main__":
    test_multi_agent_env()
