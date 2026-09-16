import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import yaml
from environment.traffic_env import MultiAgentTrafficEnv
from models.ppo_network import IndependentPPOAgent
from models.mappo_network import MAPPOController

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

env = MultiAgentTrafficEnv("config.yaml")

ippo_agents = {}
for a in ["A", "B", "C", "D"]:
    agent = IndependentPPOAgent(agent_id=a, obs_dim=43, act_dim=4, ppo_cfg=cfg["ppo"])
    agent.load_checkpoint(f"models/checkpoints/agent_{a}.pt")
    ippo_agents[a] = agent

mappo_ctrl = MAPPOController("config.yaml")
mappo_ctrl.load_checkpoints("models/checkpoints")

# Run IPPO for 10 steps
obs, _ = env.reset(seed=1001)
ippo_acts_list = []
for step in range(10):
    acts = {a: ippo_agents[a].get_action_and_value(obs[a], deterministic=True)[0] for a in env.agents}
    ippo_acts_list.append(acts)
    obs, _, _, _, _ = env.step(acts)

# Run MAPPO for 10 steps
obs, _ = env.reset(seed=1001)
mappo_acts_list = []
for step in range(10):
    acts, _ = mappo_ctrl.get_actions(obs, deterministic=True)
    mappo_acts_list.append(acts)
    obs, _, _, _, _ = env.step(acts)

env.close()

print("Step | IPPO Actions                  | MAPPO Actions")
print("-" * 60)
for i in range(10):
    print(f"{i:4d} | {str(ippo_acts_list[i]):<28} | {str(mappo_acts_list[i])}")
