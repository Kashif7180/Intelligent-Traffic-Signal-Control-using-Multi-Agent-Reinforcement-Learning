import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from environment.traffic_env import MultiAgentTrafficEnv
from models.transformer_network import MAPPOTemporalController

env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
obs, info = env.reset(seed=1001, options={"safety_enabled": False})

c4 = MAPPOTemporalController(config_path="config.yaml")
c4.load_checkpoints("models/checkpoints")
c4.reset_history(obs)

action_dist = {0: 0, 1: 0, 2: 0, 3: 0}
delays = []
queues = []

for step in range(100):
    actions, log_probs = c4.get_actions(deterministic=True)
    for a, act in actions.items():
        action_dist[act] += 1
    next_obs, rewards, term, trunc, next_info = env.step(actions)
    c4.update_history(next_obs)
    
    step_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in env.agents])
    step_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in env.agents])
    delays.append(step_w)
    queues.append(step_q)
    obs = next_obs
env.close()

print(f"C4 100-step Evaluation:")
print(f"  Mean Delay: {np.mean(delays):.2f}s, Mean Queue: {np.mean(queues):.2f} veh")
print(f"  Action Distribution: {action_dist}")
total_acts = sum(action_dist.values())
for act, count in action_dist.items():
    print(f"    Action {act}: {count}/{total_acts} ({count/total_acts*100:.1f}%)")

