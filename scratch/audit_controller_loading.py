import sys
sys.path.insert(0, ".")
import os
import torch
from benchmark_phase10_master import load_all_controllers

controllers = load_all_controllers("config.yaml")

print(f"Total controllers loaded: {len(controllers)}")
for c in controllers:
    cid = c["id"]
    name = c["name"]
    category = c["category"]
    ctype = c["type"]
    obj = c["obj"]
    
    # Check details
    details = ""
    if ctype == "fixed_time":
        details = f"FixedTimeController (green={obj.green_time}s)"
    elif ctype == "ippo":
        num_agents = len(obj)
        param_sample = next(obj["A"].actor.parameters())[0, :3].detach().numpy()
        details = f"{num_agents} IPPO agents (A-D), param[0,:3]={param_sample}"
    elif ctype == "mappo_shared":
        param_sample = next(obj.shared_actor.parameters())[0, :3].detach().numpy()
        details = f"MAPPO Shared (share_policy={obj.share_policy}), param[0,:3]={param_sample}"
    elif ctype == "mappo_individual":
        num_actors = len(obj)
        param_sample = next(obj["A"].parameters())[0, :3].detach().numpy()
        details = f"{num_actors} Individual actors (A-D), param_A[0,:3]={param_sample}"
    elif ctype == "transformer":
        t_param = next(obj.transformer_encoder.parameters())[0, :3].detach().numpy()
        details = f"MAPPOTemporalController, enc_param={t_param}"
    elif ctype == "gat":
        gat_param = next(obj.gat.parameters())[0, :3].detach().numpy()
        details = f"MAPPOGATController, gat_param={gat_param}"
    elif ctype == "comm_gate":
        gate_param = next(obj.gated_gat.comm_gate.parameters())[0, :3].detach().numpy()
        details = f"MAPPOGateController, gate_param={gate_param}"
    elif ctype in ["causal_gate", "causal_improved"]:
        if hasattr(obj.gated_gat, "causal_encoder"):
            c_param = next(obj.gated_gat.causal_encoder.parameters())[0, :3].detach().numpy()
            details = f"MAPPOCausalImprovedController (Phase 8R/9), causal_enc_param={c_param}"
        else:
            gate_param = next(obj.gated_gat.comm_gate.parameters())[0, :3].detach().numpy()
            details = f"MAPPOCausalGateController (Phase 8), gate_param={gate_param}"

    print(f"[{cid}] {name:<35} | Type: {ctype:<16} | {details}")

