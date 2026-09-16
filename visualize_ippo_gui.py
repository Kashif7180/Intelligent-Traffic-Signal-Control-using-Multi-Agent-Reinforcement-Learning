"""
Live GUI Rollout Visualization for Trained Phase 3 IPPO Model.
Runs 100 decision steps (500s simulation time) in SUMO-GUI with TraCI,
applying a configurable real-time delay (default 0.5s) after each TraCI simulation step.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import traci

from environment.traffic_env import MultiAgentTrafficEnv
from models.ppo_network import IndependentPPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Live SUMO-GUI Rollout for Trained IPPO Model")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=1001, help="Evaluation seed (default: 1001)")
    parser.add_argument("--steps", type=int, default=100, help="Number of decision steps to run (default: 100)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds after each TraCI simulation step (default: 0.5)")
    return parser.parse_args()


def run_gui_rollout():
    args = parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    action_names = config["agent"]["actions"]  # {0: keep, 1: switch, 2: extend, 3: reduce}

    print("=" * 95)
    print("SAGE-Traffic: Live SUMO-GUI Rollout of Trained Independent PPO Model")
    print("=" * 95)
    print(f"Configuration:             {config_path}")
    print(f"Evaluation Seed:           {args.seed}")
    print(f"Decision Steps:            {args.steps} (500s SUMO simulation time)")
    print(f"Delay Per Simulation Step: {args.delay}s (applied after each TraCI simulationStep)")
    print(f"Action Selection:          Deterministic Greedy (torch.argmax(dist.probs))")
    print(f"Checkpoints:               models/checkpoints/agent_{{A,B,C,D}}.pt")
    print("-" * 95)

    # 1. Initialize environment with GUI enabled
    env = MultiAgentTrafficEnv(config_path=config_path, gui=True)

    # 2. Load trained IPPO agents
    obs_dim = env.observation_space("A").shape[0]
    act_dim = env.action_space("A").n
    ppo_agents = {}
    for a in env.agents:
        ckpt_path = f"models/checkpoints/agent_{a}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Trained checkpoint not found: {ckpt_path}. Run train_ippo.py first!")
        agent = IndependentPPOAgent(agent_id=a, obs_dim=obs_dim, act_dim=act_dim, ppo_cfg=config["ppo"])
        agent.load_checkpoint(ckpt_path)
        ppo_agents[a] = agent
        print(f"  Loaded trained weights for Agent {a} from {ckpt_path}")

    print("\nStarting SUMO-GUI via TraCI...")
    obs, info = env.reset(seed=args.seed)
    print("SUMO-GUI launched and connected. Running live visualization...\n")

    # Table Header
    header = f"{'Step':<5} | {'SimTime':<8} | {'Agent Actions (A / B / C / D)':<32} | {'Queues (A/B/C/D)':<20} | {'Waits (A/B/C/D)':<20}"
    print("-" * 95)
    print(header)
    print("-" * 95)

    ep_rewards = {a: 0.0 for a in env.agents}
    total_throughput = 0
    sim_time = 0.0

    try:
        for step_idx in range(1, args.steps + 1):
            actions = {}
            for a in env.agents:
                # Deterministic action selection: torch.argmax(dist.probs)
                act, _, _ = ppo_agents[a].get_action_and_value(obs[a], deterministic=True)
                actions[a] = act

            # 1. Apply actions to signals
            for agent_id, action in actions.items():
                if agent_id in env.signals:
                    env.signals[agent_id].apply_action(action)

            # 2. Advance simulation second-by-second for step_duration with real-time delay
            for sec in range(env.step_duration):
                for sig in env.signals.values():
                    sig.update_second()
                traci.simulationStep()
                if args.delay > 0:
                    time.sleep(args.delay)

            env.current_step += 1

            # 3. Query state and metrics
            sim_ended = traci.simulation.getMinExpectedNumber() <= 0
            truncated_flag = env.current_step >= env.sim_max_steps
            arrived_this_step = traci.simulation.getArrivedNumber()
            total_throughput += arrived_this_step

            next_obs = {}
            rewards = {}
            next_info = {}
            for agent_id in env.agents:
                sig = env.signals[agent_id]
                next_obs[agent_id] = sig.compute_observation(env.features, env.normalization)
                r = sig.compute_reward(env.reward_cfg)
                rewards[agent_id] = r
                ep_rewards[agent_id] += r
                next_info[agent_id] = {
                    "metrics": sig.get_metrics(),
                    "current_step": env.current_step,
                    "sim_time": traci.simulation.getTime(),
                    "active_vehicles": traci.vehicle.getIDCount(),
                    "arrived_vehicles": arrived_this_step
                }

            sim_time = next_info["A"]["sim_time"]

            # Per-intersection queues, waiting times, and actions
            q_str = "/".join(str(next_info[a]["metrics"]["queue_length"]) for a in env.agents)
            w_str = "/".join(f"{next_info[a]['metrics']['waiting_time']:.0f}" for a in env.agents)
            act_str = "/".join(f"{a}:{action_names[actions[a]]}" for a in env.agents)

            print(f"{step_idx:<5} | {sim_time:<8.0f} | {act_str:<32} | {q_str:<20} | {w_str:<20}")

            obs = next_obs

            if sim_ended or truncated_flag:
                break

        print("-" * 95)
        print("\n" + "=" * 95)
        print("GUI ROLLOUT SUMMARY")
        print("=" * 95)
        print(f"Simulation Duration:                   {sim_time:.0f} seconds ({env.current_step} decision steps)")
        print(f"Total Completed Vehicles (Throughput): {total_throughput}")
        print("Per-Agent Cumulative Returns:")
        for a in env.agents:
            print(f"  Agent {a}: {ep_rewards[a]:.2f}")
        print(f"Aggregate Return:                      {sum(ep_rewards.values()):.2f}")
        print("=" * 95)

    finally:
        env.close()
        print("\nTraCI connection closed cleanly.")


if __name__ == "__main__":
    run_gui_rollout()
