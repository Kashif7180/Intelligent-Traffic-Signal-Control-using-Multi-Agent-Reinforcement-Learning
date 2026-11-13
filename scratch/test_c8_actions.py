import sys
sys.path.insert(0, ".")
import numpy as np
from environment.traffic_env import MultiAgentTrafficEnv
from models.causal_improved import MAPPOCausalImprovedController

env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
obs, info = env.reset(seed=1001, options={"safety_enabled": False})

p8r = MAPPOCausalImprovedController(config_path="config.yaml")
p8r.load_checkpoints("models/checkpoints")
p8r.reset_history(obs, info)

actions_history = []
for step in range(10):
    ctx = p8r.compute_causal_context(info)
    actions, _, _, comm_count = p8r.get_actions(ctx, deterministic=True)
    actions_history.append((step, actions, comm_count))
    obs, rewards, term, trunc, info = env.step(actions)
env.close()

for step, act, comm in actions_history:
    print(f"Step {step}: Actions={act}, Comm={comm}")

