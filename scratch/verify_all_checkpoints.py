import os
import sys
import hashlib
sys.path.insert(0, ".")
import torch
from models.ppo_network import IndependentPPOAgent
from models.mappo_network import DecentralizedActorNetwork, CentralizedCriticNetwork
from models.transformer_network import TemporalTransformerEncoder, DecentralizedTemporalActor, CentralizedTemporalCritic
from models.gat_network import DecentralizedGATActor, CentralizedGATCritic, GraphAttentionLayer
from models.communication_gate import GatedGraphAttentionLayer
from models.causal_improved import ImprovedGatedGraphAttentionLayer

ckpt_dir = "models/checkpoints"
files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])

results = []
for f in files:
    p = os.path.join(ckpt_dir, f)
    with open(p, "rb") as fp:
        sha = hashlib.sha256(fp.read()).hexdigest()
    data = torch.load(p, map_location="cpu", weights_only=False)

    missing = []
    unexpected = []
    arch = ""
    status = "UNKNOWN"

    try:
        if f.startswith("agent_"):
            agent_id = f.replace("agent_", "").replace(".pt", "")
            agent = IndependentPPOAgent(agent_id=agent_id, obs_dim=43, act_dim=4)
            r_act = agent.actor.load_state_dict(data["actor_state_dict"], strict=True)
            r_crit = agent.critic.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "IPPO Actor (43->64->64->4) + Critic (43->64->64->1)"
            missing = r_act.missing_keys + r_crit.missing_keys
            unexpected = r_act.unexpected_keys + r_crit.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_actor_shared.pt":
            act = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
            r = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "MAPPO Shared Actor (47->64->64->4)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f.startswith("mappo_actor_") and len(f) == 16:
            act = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
            r = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = f"MAPPO Individual Actor {f[12]} (47->64->64->4)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_critic.pt":
            crit = CentralizedCriticNetwork(global_dim=172)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "MAPPO Centralized Critic (172->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_transformer_actor.pt":
            t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
            act = DecentralizedTemporalActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
            r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
            r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "Transformer Enc (43->64) + Temporal Actor (68->64->64->4)"
            missing = r_t.missing_keys + r_a.missing_keys
            unexpected = r_t.unexpected_keys + r_a.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_transformer_critic.pt":
            crit = CentralizedTemporalCritic(embedding_dim=64, num_agents=4)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "Transformer Centralized Critic (256->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_transformer_gat_actor.pt":
            t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
            gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
            act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
            r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
            r_g = gat.load_state_dict(data["gat_state_dict"], strict=True)
            r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "Transformer (64) + GAT (64) + GAT Actor (68->4)"
            missing = r_t.missing_keys + r_g.missing_keys + r_a.missing_keys
            unexpected = r_t.unexpected_keys + r_g.unexpected_keys + r_a.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_transformer_gat_critic.pt":
            crit = CentralizedGATCritic(embedding_dim=64, num_agents=4)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "GAT Centralized Critic (256->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_comm_gate_actor.pt":
            t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
            g_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
            act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
            r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
            r_g = g_gat.load_state_dict(data["gated_gat_state_dict"], strict=True)
            r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "Transformer (64) + Gated GAT + Comm Gate + Actor (68->4)"
            missing = r_t.missing_keys + r_g.missing_keys + r_a.missing_keys
            unexpected = r_t.unexpected_keys + r_g.unexpected_keys + r_a.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_comm_gate_critic.pt":
            crit = CentralizedGATCritic(embedding_dim=64, num_agents=4)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "Comm Gate Centralized Critic (256->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_causal_gate_actor.pt":
            t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
            g_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
            act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
            r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
            r_g = g_gat.load_state_dict(data["gated_gat_state_dict"], strict=True)
            r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "Transformer (64) + Gated GAT + Causal Gate + Actor (68->4)"
            missing = r_t.missing_keys + r_g.missing_keys + r_a.missing_keys
            unexpected = r_t.unexpected_keys + r_g.unexpected_keys + r_a.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_causal_gate_critic.pt":
            crit = CentralizedGATCritic(embedding_dim=64, num_agents=4)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "Causal Gate Centralized Critic (256->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_causal_improved_actor.pt":
            t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
            g_gat = ImprovedGatedGraphAttentionLayer(input_dim=64, hidden_dim=64, causal_dim=8, causal_hidden_dim=16, gate_hidden_dims=[64, 32], num_heads=4)
            act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
            r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
            r_g = g_gat.load_state_dict(data["gated_gat_state_dict"], strict=True)
            r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
            arch = "Transformer (64) + Improved Gated GAT + phi(C_ij) + Actor (68->4)"
            missing = r_t.missing_keys + r_g.missing_keys + r_a.missing_keys
            unexpected = r_t.unexpected_keys + r_g.unexpected_keys + r_a.unexpected_keys
            status = "STRICT_PASS"

        elif f == "mappo_causal_improved_critic.pt":
            crit = CentralizedGATCritic(embedding_dim=64, num_agents=4)
            r = crit.load_state_dict(data["critic_state_dict"], strict=True)
            arch = "Phase 8R Centralized Critic (256->128->64->1)"
            missing, unexpected = r.missing_keys, r.unexpected_keys
            status = "STRICT_PASS"

    except Exception as e:
        status = f"FAILED: {e}"

    results.append({
        "file": f,
        "sha256": sha,
        "arch": arch,
        "status": status,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected
    })

print(f"{'Filename':<33} | {'Status':<12} | {'Missing':<7} | {'Unexp':<6} | {'SHA256 (First 12)':<17} | Architecture")
print("-" * 125)
for r in results:
    print(f"{r['file']:<33} | {r['status']:<12} | {r['missing_count']:<7} | {r['unexpected_count']:<6} | {r['sha256'][:12]}... | {r['arch']}")
