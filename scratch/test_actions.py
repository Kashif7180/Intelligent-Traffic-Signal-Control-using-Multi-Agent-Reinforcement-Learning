import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
import yaml
from models.ppo_network import IndependentPPOAgent
from models.mappo_network import MAPPOController

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ippo_agents = {}
for a in ["A", "B", "C", "D"]:
    agent = IndependentPPOAgent(agent_id=a, obs_dim=43, act_dim=4, ppo_cfg=cfg["ppo"])
    agent.load_checkpoint(f"models/checkpoints/agent_{a}.pt")
    ippo_agents[a] = agent

mappo_ctrl = MAPPOController("config.yaml")
mappo_ctrl.load_checkpoints("models/checkpoints")

dummy_obs = {a: np.ones(43, dtype=np.float32) * 0.5 for a in ["A", "B", "C", "D"]}

ippo_acts = {a: ippo_agents[a].get_action_and_value(dummy_obs[a], deterministic=True)[0] for a in ["A", "B", "C", "D"]}
mappo_acts, _ = mappo_ctrl.get_actions(dummy_obs, deterministic=True)

print("IPPO Actions:", ippo_acts)
print("MAPPO Actions:", mappo_acts)

# Check weights
print("MAPPO Actor layer 0 weight norm:", torch.norm(mappo_ctrl.shared_actor.network[0].weight).item())
print("IPPO Agent A layer 0 weight norm:", torch.norm(ippo_agents["A"].actor.network[0].weight).item())
