import sys
sys.path.insert(0, ".")
import torch
import numpy as np

from models.ppo_network import IndependentPPOAgent
from models.mappo_network import MAPPOController, DecentralizedActorNetwork
from models.transformer_network import MAPPOTemporalController
from models.gat_network import MAPPOGATController
from models.communication_gate import MAPPOGateController, MAPPOCausalGateController
from models.causal_improved import MAPPOCausalImprovedController

ckpt_dir = "models/checkpoints"

# Test observation: all zeros or normal observation
obs = np.zeros(43, dtype=np.float32)
obs_47 = np.concatenate([obs, [1, 0, 0, 0]])

# 1. IPPO
ippo_a = IndependentPPOAgent('A', obs_dim=43, act_dim=4)
ippo_a.actor.load_state_dict(torch.load(f"{ckpt_dir}/agent_A.pt", map_location="cpu", weights_only=False)["actor_state_dict"])
with torch.no_grad():
    dist_ippo = ippo_a.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    print("IPPO probs:          ", dist_ippo.probs.numpy())

# 2. MAPPO Shared
mappo_shared = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
mappo_shared.load_state_dict(torch.load(f"{ckpt_dir}/mappo_actor_shared.pt", map_location="cpu", weights_only=False)["actor_state_dict"])
with torch.no_grad():
    dist_mappo = mappo_shared(torch.as_tensor(obs_47, dtype=torch.float32).unsqueeze(0))
    print("MAPPO Shared probs:  ", dist_mappo.probs.numpy())

# 3. Transformer
tf_ctrl = MAPPOTemporalController()
tf_ctrl.load_checkpoints(ckpt_dir)
hist_t = torch.zeros(1, 4, 43)
with torch.no_grad():
    emb_tf = tf_ctrl.transformer_encoder(hist_t)
    dist_tf = tf_ctrl.actor(emb_tf, torch.tensor([[1., 0., 0., 0.]]))
    print("Transformer probs:   ", dist_tf.probs.numpy())

# 4. GAT
gat_ctrl = MAPPOGATController()
gat_ctrl.load_checkpoints(ckpt_dir)
hist_all = torch.zeros(4, 4, 43)
with torch.no_grad():
    emb_gat = gat_ctrl.transformer_encoder(hist_all)
    gat_embs = gat_ctrl.gat(emb_gat.unsqueeze(0)).squeeze(0)
    dist_gat = gat_ctrl.actor(gat_embs, torch.eye(4))
    print("GAT probs (agent A): ", dist_gat.probs[0].numpy())

# 5. Comm Gate
cg_ctrl = MAPPOGateController()
cg_ctrl.load_checkpoints(ckpt_dir)
ctx = {e: 0.5 for e in [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "D"), ("D", "B"), ("C", "D"), ("D", "C")]}
with torch.no_grad():
    gat_res = cg_ctrl.gated_gat(emb_gat.unsqueeze(0), ctx)
    dist_cg = cg_ctrl.actor(gat_res[0].squeeze(0), torch.eye(4))
    print("Comm Gate probs (A): ", dist_cg.probs[0].numpy())

# 6. Causal Improved (Phase 8R)
p8r_ctrl = MAPPOCausalImprovedController()
p8r_ctrl.load_checkpoints(ckpt_dir)
with torch.no_grad():
    gat_res_p8r = p8r_ctrl.gated_gat(emb_gat.unsqueeze(0), ctx)
    dist_p8r = p8r_ctrl.actor(gat_res_p8r[0].squeeze(0), torch.eye(4))
    print("Phase 8R probs (A):  ", dist_p8r.probs[0].numpy())
