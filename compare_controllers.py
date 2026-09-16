"""
Rigorous, Fair Comparative Evaluation: Independent PPO (IPPO) vs. Fixed-Time Baseline.
Evaluates both controllers over 10 evaluation episodes under identical conditions
(same seeds, network, traffic demand, episode duration, and metric calculations).
"""

import os
import sys
import yaml
import numpy as np

from environment.traffic_env import MultiAgentTrafficEnv
from models.ppo_network import IndependentPPOAgent
from models.fixed_time_controller import FixedTimeController


def evaluate_controller(env, controller_type: str, seeds: list, steps_per_episode: int, agents_dict=None):
    """
    Evaluate a controller over a list of seeds under identical conditions.
    """
    fixed_time_ctrl = FixedTimeController("config.yaml") if controller_type == "Fixed-Time" else None

    episode_results = []

    for ep_idx, seed in enumerate(seeds, 1):
        obs, info = env.reset(seed=seed)
        if fixed_time_ctrl:
            fixed_time_ctrl.reset()

        ep_rewards = {a: 0.0 for a in env.agents}
        ep_queues = []
        ep_waiting = []
        ep_throughput = 0

        for step in range(steps_per_episode):
            if controller_type == "Fixed-Time":
                actions = fixed_time_ctrl.get_actions(obs, info)
            elif controller_type == "IPPO":
                actions = {}
                for a in env.agents:
                    act, _, _ = agents_dict[a].get_action_and_value(obs[a], deterministic=True)
                    actions[a] = act
            else:
                raise ValueError(f"Unknown controller: {controller_type}")

            next_obs, rewards, terminations, truncations, next_info = env.step(actions)

            for a in env.agents:
                ep_rewards[a] += rewards[a]

            step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
            step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
            ep_queues.append(step_q)
            ep_waiting.append(step_w)
            ep_throughput += next_info["A"]["arrived_vehicles"]

            obs = next_obs
            info = next_info
            if any(terminations.values()) or any(truncations.values()):
                break

        ep_data = {
            "episode": ep_idx,
            "seed": seed,
            "rewards": ep_rewards,
            "agg_return": sum(ep_rewards.values()),
            "mean_queue": float(np.mean(ep_queues)),
            "mean_waiting": float(np.mean(ep_waiting)),
            "throughput": ep_throughput
        }
        episode_results.append(ep_data)

    return episode_results


def main():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_episodes = int(config.get("fixed_time", {}).get("eval_episodes", 10))
    steps_per_episode = int(config["ppo"].get("steps_per_episode", 100))
    eval_seeds = [1000 + i for i in range(1, eval_episodes + 1)]

    print("=" * 85)
    print("SAGE-Traffic Phase 3: Fair Comparative Evaluation (IPPO vs Fixed-Time)")
    print("=" * 85)
    print(f"Evaluation Budget:      {eval_episodes} Episodes per Controller")
    print(f"Steps per Episode:      {steps_per_episode} (500s SUMO simulation time each)")
    print(f"Evaluation Seeds:       {eval_seeds}")
    print(f"Traffic Demand:         {config['demand']['active_demand']}")
    print("-" * 85)

    env = MultiAgentTrafficEnv(config_path=config_path)

    # Load trained IPPO agents
    obs_dim = env.observation_space("A").shape[0]
    act_dim = env.action_space("A").n
    ppo_agents = {}
    for a in env.agents:
        agent = IndependentPPOAgent(agent_id=a, obs_dim=obs_dim, act_dim=act_dim, ppo_cfg=config["ppo"])
        ckpt_path = f"models/checkpoints/agent_{a}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train_ippo.py first!")
        agent.load_checkpoint(ckpt_path)
        ppo_agents[a] = agent
    print("Trained Independent PPO checkpoints loaded successfully.")

    try:
        # 1. Evaluate Fixed-Time
        print("\n[1/2] Running Fixed-Time Baseline evaluation (10 episodes)...")
        ft_results = evaluate_controller(env, "Fixed-Time", eval_seeds, steps_per_episode)
        print("  Fixed-Time evaluation completed.")

        # 2. Evaluate IPPO
        print("\n[2/2] Running Independent PPO (IPPO) evaluation (10 episodes)...")
        ippo_results = evaluate_controller(env, "IPPO", eval_seeds, steps_per_episode, agents_dict=ppo_agents)
        print("  Independent PPO evaluation completed.")

        # Aggregate Statistics
        ft_queues = [r["mean_queue"] for r in ft_results]
        ft_waits = [r["mean_waiting"] for r in ft_results]
        ft_returns = [r["agg_return"] for r in ft_results]
        ft_throughputs = [r["throughput"] for r in ft_results]

        ippo_queues = [r["mean_queue"] for r in ippo_results]
        ippo_waits = [r["mean_waiting"] for r in ippo_results]
        ippo_returns = [r["agg_return"] for r in ippo_results]
        ippo_throughputs = [r["throughput"] for r in ippo_results]

        # Per-agent mean returns
        agents = env.agents
        ft_agent_returns = {a: np.mean([r["rewards"][a] for r in ft_results]) for a in agents}
        ippo_agent_returns = {a: np.mean([r["rewards"][a] for r in ippo_results]) for a in agents}

        print("\n" + "=" * 85)
        print("HEAD-TO-HEAD COMPARATIVE EVALUATION RESULTS (10 EPISODES MEAN +/- STD)")
        print("=" * 85)
        print(f"{'Metric':<35} | {'Fixed-Time':<20} | {'Independent PPO':<20} | {'Delta (%)':<10}")
        print("-" * 85)

        # Waiting time (lower is better)
        m_ft_wait = np.mean(ft_waits)
        m_ippo_wait = np.mean(ippo_waits)
        wait_delta = ((m_ippo_wait - m_ft_wait) / m_ft_wait) * 100
        print(f"{'Average Waiting Time (s)':<35} | {m_ft_wait:<6.2f} +/- {np.std(ft_waits):<6.2f}    | {m_ippo_wait:<6.2f} +/- {np.std(ippo_waits):<6.2f}    | {wait_delta:+6.2f}%")

        # Queue length (lower is better)
        m_ft_q = np.mean(ft_queues)
        m_ippo_q = np.mean(ippo_queues)
        q_delta = ((m_ippo_q - m_ft_q) / m_ft_q) * 100
        print(f"{'Average Queue Length (veh)':<35} | {m_ft_q:<6.2f} +/- {np.std(ft_queues):<6.2f}    | {m_ippo_q:<6.2f} +/- {np.std(ippo_queues):<6.2f}    | {q_delta:+6.2f}%")

        # Throughput (higher is better)
        m_ft_tp = np.mean(ft_throughputs)
        m_ippo_tp = np.mean(ippo_throughputs)
        tp_delta = ((m_ippo_tp - m_ft_tp) / m_ft_tp) * 100
        print(f"{'Total Throughput (completed veh)':<35} | {m_ft_tp:<6.1f} +/- {np.std(ft_throughputs):<6.1f}    | {m_ippo_tp:<6.1f} +/- {np.std(ippo_throughputs):<6.1f}    | {tp_delta:+6.2f}%")

        # Aggregate Return (higher / less negative is better)
        m_ft_ret = np.mean(ft_returns)
        m_ippo_ret = np.mean(ippo_returns)
        ret_delta = ((m_ippo_ret - m_ft_ret) / abs(m_ft_ret)) * 100
        print(f"{'Aggregate Return (all 4 agents)':<35} | {m_ft_ret:<6.1f} +/- {np.std(ft_returns):<6.1f}    | {m_ippo_ret:<6.1f} +/- {np.std(ippo_returns):<6.1f}    | {ret_delta:+6.2f}%")

        print("-" * 85)
        print("Per-Agent Mean Episode Returns:")
        for a in agents:
            diff = ippo_agent_returns[a] - ft_agent_returns[a]
            print(f"  Agent {a}: Fixed-Time = {ft_agent_returns[a]:8.2f} | IPPO = {ippo_agent_returns[a]:8.2f} | Diff = {diff:+8.2f}")

        print("=" * 85)
        # Explicit, honest verdict
        print("\n--- HONEST VERDICT (Rule 1 Compliance) ---")
        if m_ippo_wait < m_ft_wait and m_ippo_q < m_ft_q:
            verdict = "YES - Independent PPO outperformed Fixed-Time on delay and queue length."
        elif m_ippo_wait > m_ft_wait and m_ippo_q > m_ft_q:
            verdict = "NO - Independent PPO performed worse than Fixed-Time on delay and queue length."
        else:
            verdict = "MIXED - Independent PPO and Fixed-Time performed comparably across different metrics."

        print(f"Does Independent PPO beat Fixed-Time? -> {verdict}")
        print("Detailed explanation and factors documented in Phase 3 DONE report.\n")

    finally:
        env.close()


if __name__ == "__main__":
    main()
