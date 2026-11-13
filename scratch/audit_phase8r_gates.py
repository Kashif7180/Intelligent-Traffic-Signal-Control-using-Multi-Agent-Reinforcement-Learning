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

print("Phase 8R Runtime Gate Inspection on Seed 1001:")
for step in range(5):
    ctx = p8r.compute_causal_context(info)
    actions, log_probs, comm_costs, comm_count = p8r.get_actions(ctx, deterministic=True)
    gate_vals = p8r.gated_gat.comm_gate.last_gate_values
    decisions = p8r.gated_gat.comm_gate.last_comm_decisions
    print(f"Step {step}: comm_count={comm_count}")
    for edge in list(gate_vals.keys())[:3]:
        print(f"   Edge {edge}: C_ij={ctx[edge]:.4f}, gate_val={gate_vals[edge]:.6f}, decision={decisions[edge]}")
    obs, rewards, term, trunc, info = env.step(actions)
env.close()

