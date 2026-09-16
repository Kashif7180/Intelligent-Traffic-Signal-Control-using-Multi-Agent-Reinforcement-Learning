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

for a in ["A", "B", "C", "D"]:
    # IPPO distribution
    with torch.no_grad():
        ippo_dist = ippo_agents[a].actor(torch.tensor(dummy_obs[a], dtype=torch.float32).unsqueeze(0))
        inp = np.concatenate([dummy_obs[a], mappo_ctrl.agent_one_hots[a]])
        mappo_dist = mappo_ctrl.shared_actor(torch.tensor(inp, dtype=torch.float32).unsqueeze(0))
    print(f"Agent {a}:")
    print(f"  IPPO  Action Probs: {ippo_dist.probs.numpy()[0]}")
    print(f"  MAPPO Action Probs: {mappo_dist.probs.numpy()[0]}")
