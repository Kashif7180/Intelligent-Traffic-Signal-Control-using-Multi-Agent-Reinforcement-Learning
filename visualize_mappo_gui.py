"""
VISUALIZATION ONLY: Live SUMO-GUI Rollout of Trained Phase 4 MAPPO Model.
DISCLAIMER: This script is for interactive visualization and sanity-checking only.
It is NOT a formal evaluation script.

Requirements:
- Loads Phase 4 checkpoint: models/checkpoints/mappo_actor_shared.pt
- 43-dim local observation + 4-dim one-hot agent ID = 47-dim actor input
- Centralized critic is NOT used for action selection
- Deterministic greedy argmax action selection
- Seed 1001, 100 decision steps (500s SUMO simulation time)
- 0.5s visualization delay per decision step
- Logs raw vs effective actions, current phase, queue, and waiting time
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
from models.mappo_network import MAPPOController


def parse_args():
    parser = argparse.ArgumentParser(description="Live SUMO-GUI Rollout for Trained MAPPO Model (VISUALIZATION ONLY)")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=1001, help="Evaluation seed (default: 1001)")
    parser.add_argument("--steps", type=int, default=100, help="Number of decision steps to run (default: 100)")
    parser.add_argument("--delay", type=float, default=0.5, help="Visualization delay in seconds per decision step (default: 0.5s)")
    return parser.parse_args()


def run_mappo_gui_rollout():
    args = parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    action_names = config["agent"]["actions"]  # {0: keep, 1: switch, 2: extend, 3: reduce}

    print("=" * 105)
    print("SAGE-TRAFFIC: LIVE SUMO-GUI ROLLOUT FOR TRAINED MAPPO MODEL")
    print(">>> VISUALIZATION ONLY - NOT A FORMAL EVALUATION RUN <<<")
    print("=" * 105)
    print(f"Configuration:             {config_path}")
    print(f"Evaluation Seed:           {args.seed}")
    print(f"Decision Steps:            {args.steps} (500s SUMO simulation time)")
    print(f"Visualization Delay:       {args.delay}s per decision step")
    print(f"Action Selection:          Deterministic Greedy (torch.argmax(dist.probs))")
    print(f"Decentralized Actor Input: 43-D local observation + 4-D agent ID = 47 dimensions")
    print(f"Centralized Critic:        NOT USED during execution / action selection")
    print(f"Checkpoint Loaded:         models/checkpoints/mappo_actor_shared.pt")
    print("-" * 105)

    # 1. Initialize environment with SUMO-GUI enabled
    env = MultiAgentTrafficEnv(config_path=config_path, gui=True)

    # 2. Initialize and load MAPPO controller
    mappo_ctrl = MAPPOController(config_path=config_path)
    ckpt_path = "models/checkpoints/mappo_actor_shared.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MAPPO checkpoint not found: {ckpt_path}. Run train_mappo.py first!")
    mappo_ctrl.load_checkpoints("models/checkpoints")
    print("Phase 4 MAPPO shared actor loaded successfully.")

    print("\nStarting SUMO-GUI via TraCI...")
    obs, info = env.reset(seed=args.seed)
    print("SUMO-GUI launched and connected. Running live visualization...\n")

    # Table Header
    header = (
        f"{'Step':<5} | {'Time':<6} | "
        f"{'Actions [Raw -> Effective] (A / B / C / D)':<46} | "
        f"{'Phase':<10} | {'Queues':<12} | {'Waits':<14}"
    )
    print("-" * 105)
    print(header)
    print("-" * 105)

    ep_rewards = {a: 0.0 for a in env.agents}
    all_queues = []
    all_waits = []
    total_throughput = 0
    sim_time = 0.0

    try:
        for step_idx in range(1, args.steps + 1):
            # 1. Decentralized Action Selection (local obs + one-hot agent ID only)
            # Centralized critic is NOT called.
            actions, _ = mappo_ctrl.get_actions(obs, deterministic=True)

            # Record state before applying action to determine effective action
            pre_states = {
                a: {
                    "time_on_phase": env.signals[a].time_on_phase,
                    "is_yellow": env.signals[a].is_yellow,
                    "min_green": env.signals[a].min_green
                }
                for a in env.agents
            }

            # 2. Apply actions to signals
            for agent_id, action in actions.items():
                if agent_id in env.signals:
                    env.signals[agent_id].apply_action(action)

            # 3. Advance simulation second-by-second for step_duration
            for sec in range(env.step_duration):
                for sig in env.signals.values():
                    sig.update_second()
                traci.simulationStep()

            # 4. Visualization delay per decision step
            if args.delay > 0:
                time.sleep(args.delay)

            env.current_step += 1

            # 5. Query state and metrics
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

            # Format effective actions
            # Determine if raw switch command was executed or held due to min_green
            action_strs = []
            for a in env.agents:
                raw_act = actions[a]
                raw_name = action_names[raw_act].upper()
                if raw_act == 1:
                    if pre_states[a]["is_yellow"]:
                        eff = "YEL"
                    elif pre_states[a]["time_on_phase"] < pre_states[a]["min_green"]:
                        eff = "HOLD"  # suppressed by min_green constraint
                    else:
                        eff = "SW"    # switch initiated
                else:
                    eff = raw_name[:4]
                action_strs.append(f"{a}:{raw_name[:2]}->{eff}")

            act_display = " ".join(action_strs)
            phase_display = "/".join(str(next_info[a]["metrics"]["current_phase"]) for a in env.agents)
            q_display = "/".join(str(next_info[a]["metrics"]["queue_length"]) for a in env.agents)
            w_display = "/".join(f"{next_info[a]['metrics']['waiting_time']:.0f}" for a in env.agents)

            step_mean_q = float(np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents]))
            step_mean_w = float(np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents]))
            all_queues.append(step_mean_q)
            all_waits.append(step_mean_w)

            print(f"{step_idx:<5} | {sim_time:<6.0f} | {act_display:<46} | {phase_display:<10} | {q_display:<12} | {w_display:<14}")

            obs = next_obs

            if sim_ended or truncated_flag:
                break

        print("-" * 105)
        print("\n" + "=" * 105)
        print("MAPPO SUMO-GUI VISUALIZATION SUMMARY (VISUALIZATION ONLY)")
        print("=" * 105)
        print(f"Total Simulated Time:                  {sim_time:.0f} seconds ({env.current_step} decision steps)")
        print(f"Total Completed Vehicles (Throughput): {total_throughput}")
        print("Per-Agent Cumulative Episode Returns:")
        for a in env.agents:
            print(f"  Agent {a}: {ep_rewards[a]:.2f}")
        print(f"Aggregate Episode Return:              {sum(ep_rewards.values()):.2f}")
        print(f"Mean Queue Length:                     {np.mean(all_queues):.2f} vehicles")
        print(f"Mean Waiting Time (Average Delay):     {np.mean(all_waits):.2f} seconds / vehicle")
        print("=" * 105)
        print("DISCLAIMER: Visualization run completed cleanly. Existing formal evaluation results remain untouched.")
        print("=" * 105)

    finally:
        env.close()
        print("\nEnvironment closed cleanly.")


if __name__ == "__main__":
    run_mappo_gui_rollout()
