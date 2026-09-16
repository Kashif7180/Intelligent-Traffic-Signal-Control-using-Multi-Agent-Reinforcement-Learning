"""
Phase 8R: Improved Causal Communication Architecture for SAGE-Traffic.

Key Architectural Improvements over Phase 8:
1. Dedicated Causal Pathway (phi(C_ij) in R^{causal_dim}):
   Projects the scalar causal influence estimate C_ij into an 8-dimensional learnable
   embedding space using an MLP + LayerNorm to increase representational leverage.
2. Improved Communication Gate:
   MLP([h_i, h_j, phi(C_ij)]) with dimensions (64 + 64 + 8) -> 64 -> 32 -> 1 -> Sigmoid.
3. Causal-Aware Message Modulation:
   Directly scales neighbor messages m_ij = g_ij * alpha_ij * W h_j in the GAT layer,
   ensuring C_ij directly governs neighbor information flow.
4. Differentiable Communication Budget Regularization:
   Loss includes:
   L_comm_budget = lambda_budget * max(0, mean(g) - target_ratio)^2 + lambda_comm * mean(g)
   participating directly in PPO actor backpropagation.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Dict, List, Tuple, Optional, Any

from models.transformer_network import TemporalTransformerEncoder, TemporalHistoryManager, layer_init
from models.gat_network import build_physical_adjacency_matrix, DecentralizedGATActor, CentralizedGATCritic
from models.causal_influence import CausalInfluenceEstimator, ALLOWED_DIRECTED_EDGES, CORRIDOR_INCOMING_LANES
from models.communication_gate import MAPPOGateRolloutBuffer


class CausalPathwayEncoder(nn.Module):
    """
    Dedicated Causal Pathway: Projects scalar C_ij into a multi-dimensional
    normalized embedding space to increase the representational leverage of C_ij.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, causal_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.causal_dim = causal_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, causal_dim)),
            nn.LayerNorm(causal_dim)
        )

    def forward(self, c_ij: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_ij: [B, 1] scalar causal influence estimates.
        Returns:
            phi_c: [B, causal_dim] causal embedding.
        """
        if c_ij.dim() == 1:
            c_ij = c_ij.unsqueeze(-1)
        return self.net(c_ij)


class ImprovedCausalCommunicationGate(nn.Module):
    """
    Multi-layer communication gate receiving [h_i, h_j, phi(C_ij)].
    Dimensions: (64 + 64 + causal_dim) -> 64 -> 32 -> 1 -> Sigmoid.
    """

    def __init__(
        self,
        node_dim: int = 64,
        causal_dim: int = 8,
        causal_hidden_dim: int = 16,
        hidden_dims: List[int] = [64, 32],
        hard_threshold: float = 0.5,
        agents: List[str] = ["A", "B", "C", "D"]
    ):
        super().__init__()
        self.node_dim = node_dim
        self.causal_dim = causal_dim
        self.hard_threshold = hard_threshold
        self.agents = agents
        self.agent_to_idx = {a: i for i, a in enumerate(agents)}
        self.allowed_edges = ALLOWED_DIRECTED_EDGES
        self.num_allowed_edges = len(self.allowed_edges)

        # 1. Dedicated Causal Pathway Encoder
        self.causal_encoder = CausalPathwayEncoder(
            input_dim=1,
            hidden_dim=causal_hidden_dim,
            causal_dim=causal_dim
        )

        # 2. Gate MLP: [h_tgt (64), h_src (64), phi_c (causal_dim)]
        self.gate_in_dim = 2 * node_dim + causal_dim  # e.g. 64 + 64 + 8 = 136
        
        layers = []
        in_d = self.gate_in_dim
        for h_d in hidden_dims:
            layers.append(layer_init(nn.Linear(in_d, h_d)))
            layers.append(nn.ReLU())
            in_d = h_d
        layers.append(layer_init(nn.Linear(in_d, 1), std=1.0))
        # Note: Sigmoid is applied separately so we can access pre-sigmoid logits
        self.mlp_body = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        # Cached outputs for inspection
        self.last_gate_logits: Dict[Tuple[str, str], float] = {}
        self.last_gate_values: Dict[Tuple[str, str], float] = {}
        self.last_comm_decisions: Dict[Tuple[str, str], bool] = {}

    def forward(
        self,
        node_features: torch.Tensor,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute communication gate values for all 8 allowed physical channels.

        Args:
            node_features: [B, 4, node_dim] or [4, node_dim]
            context_dict: mapping (src, tgt) -> float C_ij
            deterministic: If True, uses hard_threshold for decisions

        Returns:
            gate_matrix: [B, 4, 4] tensor for spatial message routing.
            raw_gates: [B, 8] continuous soft gate tensor.
            raw_logits: [B, 8] pre-sigmoid gate logit tensor.
        """
        is_2d = (node_features.dim() == 2)
        if is_2d:
            node_features = node_features.unsqueeze(0)

        B, N, D = node_features.shape
        device = node_features.device

        # Initialize gate matrix [B, 4, 4]
        # Diagonal (self-loops) is always 1.0 (local representation, not inter-agent comm)
        gate_matrix = torch.zeros((B, 4, 4), device=device, dtype=torch.float32)
        for i in range(4):
            gate_matrix[:, i, i] = 1.0

        raw_gate_list = []
        raw_logit_list = []
        edge_logits_step = {}
        edge_values_step = {}
        edge_decisions_step = {}

        for src, tgt in self.allowed_edges:
            src_idx = self.agent_to_idx[src]
            tgt_idx = self.agent_to_idx[tgt]

            h_tgt = node_features[:, tgt_idx, :]  # [B, 64]
            h_src = node_features[:, src_idx, :]  # [B, 64]

            c_val = context_dict.get((src, tgt), 0.5)
            c_tensor = torch.full((B, 1), float(c_val), device=device, dtype=torch.float32)

            # 1. Project through dedicated causal pathway
            phi_c = self.causal_encoder(c_tensor)  # [B, causal_dim]

            # 2. Gate input: [h_tgt, h_src, phi_c] = [B, 136]
            gate_in = torch.cat([h_tgt, h_src, phi_c], dim=-1)

            # 3. Gate logit and soft probability
            gate_logit = self.mlp_body(gate_in)     # [B, 1]
            g_soft = self.sigmoid(gate_logit)       # [B, 1] in (0, 1)

            if deterministic:
                # Discrete thresholding for evaluation
                g_eff = (g_soft >= self.hard_threshold).float()
            else:
                # Fully differentiable soft gate for training
                g_eff = g_soft

            gate_matrix[:, tgt_idx, src_idx] = g_eff.squeeze(-1)
            raw_gate_list.append(g_soft)
            raw_logit_list.append(gate_logit)

            logit_val = float(gate_logit[0, 0].item())
            prob_val = float(g_soft[0, 0].item())
            edge_logits_step[(src, tgt)] = logit_val
            edge_values_step[(src, tgt)] = prob_val
            edge_decisions_step[(src, tgt)] = bool(prob_val >= self.hard_threshold)

        self.last_gate_logits = edge_logits_step
        self.last_gate_values = edge_values_step
        self.last_comm_decisions = edge_decisions_step

        raw_gates = torch.cat(raw_gate_list, dim=-1)    # [B, 8]
        raw_logits = torch.cat(raw_logit_list, dim=-1)  # [B, 8]

        if is_2d:
            gate_matrix = gate_matrix.squeeze(0)

        return gate_matrix, raw_gates, raw_logits


class ImprovedGatedGraphAttentionLayer(nn.Module):
    """
    Improved GAT layer with dedicated causal pathway and causal-aware message modulation.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        causal_dim: int = 8,
        causal_hidden_dim: int = 16,
        gate_hidden_dims: List[int] = [64, 32],
        num_heads: int = 4,
        concat_heads: bool = False,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        hard_threshold: float = 0.5,
        agents: List[str] = ["A", "B", "C", "D"]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope
        self.agents = agents
        self.num_nodes = len(agents)

        # Physical road adjacency buffer
        adj = build_physical_adjacency_matrix(agents=agents, add_self_loops=True)
        self.register_buffer("adj", adj)

        # GAT learnable projections
        self.W = nn.Parameter(torch.empty(num_heads, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

        self.a_src = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.ELU()

        # Improved Communication Gate
        self.comm_gate = ImprovedCausalCommunicationGate(
            node_dim=input_dim,
            causal_dim=causal_dim,
            causal_hidden_dim=causal_hidden_dim,
            hidden_dims=gate_hidden_dims,
            hard_threshold=hard_threshold,
            agents=agents
        )

        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_gate_matrix: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal-aware gated message passing.

        Args:
            x: [B, N, input_dim] node embeddings from Temporal Transformer
            context_dict: mapping (src, tgt) -> float C_ij
            deterministic: True for evaluation, False for training

        Returns:
            out: [B, N, hidden_dim] modulated node representations
            raw_gates: [B, 8] continuous soft gate values
            raw_logits: [B, 8] gate pre-sigmoid logits
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0)

        B, N, _ = x.shape
        Wh = torch.einsum("bni, kih -> bknh", x, self.W)  # [B, K, N, H]

        attn_dst = torch.einsum("bknh, kho -> bkno", Wh, self.a_dst)  # [B, K, N, 1]
        attn_src = torch.einsum("bknh, kho -> bkno", Wh, self.a_src)  # [B, K, N, 1]
        attn_src = attn_src.transpose(2, 3)                           # [B, K, 1, N]

        logits = self.leaky_relu(attn_dst + attn_src)
        mask = self.adj.unsqueeze(0).unsqueeze(0)

        # 1. Compute Improved Communication Gate: [B, N, N] (tgt, src)
        gate_matrix, raw_gates, raw_logits = self.comm_gate(x, context_dict, deterministic=deterministic)
        self.last_gate_matrix = gate_matrix.detach()

        # 2. Gate modulation: Expand gate matrix to attention heads [B, 1, N, N]
        gate_expanded = gate_matrix.unsqueeze(1)
        gate_log = torch.log(torch.clamp(gate_expanded, min=1e-8, max=1.0))

        # Effective attention logit combines raw GAT logit + log(gate)
        gated_logits = logits + gate_log
        is_masked = (mask == 0.0) | (gate_expanded < 1e-6)
        masked_logits = gated_logits.masked_fill(is_masked, -1e9)

        alpha = torch.softmax(masked_logits, dim=-1)  # [B, K, N, N]
        alpha_dropped = self.dropout(alpha)
        self.last_attention_weights = alpha.detach()

        # 3. Causal-Aware Message Modulation:
        # Scale messages by soft gate g_ij:
        # message_ij = g_ij * alpha_ij * Wh_j
        # (self-loops have g_ii = 1.0, so local state is fully preserved)
        modulated_alpha = alpha_dropped * gate_expanded
        out_k = torch.einsum("bkij, bkjh -> bkih", modulated_alpha, Wh)

        if self.concat_heads:
            out = out_k.permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.hidden_dim)
        else:
            out = out_k.mean(dim=1)

        out = self.activation(out)

        if is_2d:
            out = out.squeeze(0)

        return out, raw_gates, raw_logits


class MAPPOCausalImprovedController:
    """
    Phase 8R Controller: MAPPO + Temporal Transformer + GAT + Dedicated Causal Pathway Gate.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.t_cfg = config.get("transformer", {})
        self.gat_cfg = config.get("gat", {})
        self.p8r_cfg = config.get("phase8r", {})
        self.causal_cfg = self.p8r_cfg.get("causal", {})
        self.comm_cfg = self.p8r_cfg.get("communication", {})
        self.train_cfg = self.p8r_cfg.get("training", {})
        self.norm_cfg = config.get("observation", {}).get("normalization", {})
        self.agents = config["agent"]["agents"]
        self.num_agents = len(self.agents)

        # Dimensions & parameters
        self.obs_dim = 43
        self.history_length = int(self.t_cfg.get("history_length", 4))
        self.embedding_dim = int(self.gat_cfg.get("input_dim", 64))
        self.gat_hidden_dim = int(self.gat_cfg.get("hidden_dim", 64))
        
        self.causal_dim = int(self.causal_cfg.get("causal_embedding_dim", 8))
        self.causal_hidden_dim = int(self.causal_cfg.get("hidden_dim", 16))
        self.gate_hidden_dims = list(self.comm_cfg.get("hidden_dims", [64, 32]))
        
        self.hard_threshold = float(self.comm_cfg.get("hard_threshold", 0.5))
        self.lambda_comm = float(self.comm_cfg.get("lambda_comm", 0.01))
        self.budget_penalty_coef = float(self.comm_cfg.get("budget_penalty_coef", 0.05))
        self.target_ratio = float(self.comm_cfg.get("target_ratio", 0.50))

        self.agent_id_dim = 4
        self.act_dim = 4
        self.actor_input_dim = self.gat_hidden_dim + self.agent_id_dim  # 68
        self.critic_input_dim = self.num_agents * self.gat_hidden_dim   # 256

        # Hyperparameters
        self.lr_actor = float(self.train_cfg.get("lr_actor", 0.0003))
        self.lr_critic = float(self.train_cfg.get("lr_critic", 0.0005))
        self.gamma = float(config.get("mappo", {}).get("gamma", 0.99))
        self.gae_lambda = float(config.get("mappo", {}).get("gae_lambda", 0.95))
        self.clip_ratio = float(config.get("mappo", {}).get("clip_ratio", 0.2))
        self.value_coef = float(config.get("mappo", {}).get("value_coef", 0.5))
        self.entropy_coef = float(config.get("mappo", {}).get("entropy_coef", 0.01))
        self.max_grad_norm = float(config.get("mappo", {}).get("max_grad_norm", 0.5))
        self.epochs = int(config.get("mappo", {}).get("epochs", 4))
        self.minibatch_size = int(config.get("mappo", {}).get("minibatch_size", 32))

        # One-hot agent IDs
        self.agent_one_hots = {
            a: np.eye(self.num_agents, dtype=np.float32)[i]
            for i, a in enumerate(self.agents)
        }
        self.one_hot_tensor = torch.tensor(
            np.stack([self.agent_one_hots[a] for a in self.agents], axis=0),
            dtype=torch.float32
        )

        # 1. Temporal Transformer Encoder
        self.transformer_encoder = TemporalTransformerEncoder(
            obs_dim=self.obs_dim,
            history_length=self.history_length,
            embedding_dim=self.embedding_dim,
            num_heads=int(self.t_cfg.get("num_heads", 4)),
            num_layers=int(self.t_cfg.get("num_layers", 2)),
            dim_feedforward=int(self.t_cfg.get("dim_feedforward", 128)),
            dropout=float(self.t_cfg.get("dropout", 0.0))
        )

        # 2. Improved Gated GAT (with Dedicated Causal Pathway)
        self.gated_gat = ImprovedGatedGraphAttentionLayer(
            input_dim=self.embedding_dim,
            hidden_dim=self.gat_hidden_dim,
            causal_dim=self.causal_dim,
            causal_hidden_dim=self.causal_hidden_dim,
            gate_hidden_dims=self.gate_hidden_dims,
            num_heads=int(self.gat_cfg.get("num_heads", 4)),
            concat_heads=bool(self.gat_cfg.get("concat_heads", False)),
            dropout=float(self.gat_cfg.get("dropout", 0.0)),
            hard_threshold=self.hard_threshold,
            agents=self.agents
        )

        # 3. Decentralized Actor
        self.actor = DecentralizedGATActor(
            embedding_dim=self.gat_hidden_dim,
            agent_id_dim=self.agent_id_dim,
            act_dim=self.act_dim,
            hidden_dims=[64, 64]
        )

        # 4. Centralized Critic
        self.critic = CentralizedGATCritic(
            num_agents=self.num_agents,
            embedding_dim=self.gat_hidden_dim,
            hidden_dims=[128, 64]
        )

        # 5. Causal Influence Estimator
        self.causal_estimator = CausalInfluenceEstimator(
            method="granger_style",
            history_length=5,
            window_size=int(self.causal_cfg.get("history_window", 15)),
            ridge_lambda=float(self.causal_cfg.get("ridge_lambda", 0.001)),
            max_queue=float(self.norm_cfg.get("max_queue_length", 40.0)),
            max_wait=float(self.norm_cfg.get("max_waiting_time", 120.0)),
            max_veh=float(self.norm_cfg.get("max_vehicle_count", 40.0)),
            agents=self.agents
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.transformer_encoder.parameters()) +
            list(self.gated_gat.parameters()) +
            list(self.actor.parameters()),
            lr=self.lr_actor,
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            eps=1e-5
        )

        # History manager & rollout buffer
        self.history_manager = TemporalHistoryManager(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )
        self.buffer = MAPPOGateRolloutBuffer(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )
        self.last_causal_scores: Dict[Tuple[str, str], float] = {}

    def compute_causal_context(self, info_dict: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """Compute directional C_ij scores from observable traffic history."""
        self.causal_estimator.update_traffic_state(info_dict)
        scores = self.causal_estimator.estimate_causal_influence()
        self.last_causal_scores = scores.copy()
        return scores

    def reset_history(self, initial_obs_dict: Dict[str, np.ndarray], info_dict: Optional[Dict[str, Any]] = None):
        self.history_manager.reset(initial_obs_dict)
        self.causal_estimator.reset()
        if info_dict is not None:
            self.causal_estimator.update_traffic_state(info_dict)
        self.last_causal_scores = self.causal_estimator.estimate_causal_influence()

    def update_history(self, next_obs_dict: Dict[str, np.ndarray]):
        self.history_manager.update(next_obs_dict)

    def get_actions(
        self,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float], int]:
        """
        Decentralized Action Selection via Improved Gated Communication.
        """
        self.transformer_encoder.eval()
        self.gated_gat.eval()
        self.actor.eval()

        with torch.no_grad():
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)

            temp_embs = self.transformer_encoder(hist_tensor)
            gated_embs, raw_gates, raw_logits = self.gated_gat(
                temp_embs.unsqueeze(0),
                context_dict=context_dict,
                deterministic=deterministic
            )

            flat_gated = gated_embs.view(self.num_agents, self.gat_hidden_dim)
            flat_ids = self.one_hot_tensor

            dist = self.actor(flat_gated, flat_ids)

            if deterministic:
                actions_tensor = torch.argmax(dist.probs, dim=-1)
            else:
                actions_tensor = dist.sample()

            log_probs_tensor = dist.log_prob(actions_tensor)

        actions = {a: int(actions_tensor[i].item()) for i, a in enumerate(self.agents)}
        log_probs = {a: float(log_probs_tensor[i].item()) for i, a in enumerate(self.agents)}

        comm_costs = {a: 0.0 for a in self.agents}
        actual_comm_count = 0

        for (src, tgt), g_val in self.gated_gat.comm_gate.last_gate_values.items():
            is_comm = (g_val >= self.hard_threshold)
            if is_comm:
                actual_comm_count += 1

            if deterministic:
                cost = self.lambda_comm * (1.0 if is_comm else 0.0)
            else:
                cost = self.lambda_comm * g_val

            comm_costs[tgt] += float(cost)

        return actions, log_probs, comm_costs, actual_comm_count

    def get_centralized_value(self, context_dict: Dict[Tuple[str, str], float]) -> float:
        """Centralized Critic Value Estimation (Training Only)."""
        self.transformer_encoder.eval()
        self.gated_gat.eval()
        self.critic.eval()

        with torch.no_grad():
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)

            temp_embs = self.transformer_encoder(hist_tensor)
            gated_embs, _, _ = self.gated_gat(
                temp_embs.unsqueeze(0),
                context_dict=context_dict,
                deterministic=False
            )

            centralized_state = gated_embs.view(1, self.critic_input_dim)
            val = self.critic(centralized_state).squeeze()
            return float(val.item())

    def update(self, last_value: float) -> Dict[str, float]:
        """
        PPO Update with Communication Budget Regularization:
        L_actor_total = L_PPO_surr - c_ent * H(pi) + L_comm_budget
        where:
        L_comm_budget = lambda_budget * max(0, mean(g) - target_ratio)^2 + lambda_comm * mean(g)
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "comm_budget_loss": 0.0}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        self.transformer_encoder.train()
        self.gated_gat.train()
        self.actor.train()
        self.critic.train()

        critic_losses = []
        actor_losses = []
        budget_losses = []
        entropies = []
        causal_grad_norms = []

        for _ in range(self.epochs):
            # 1. Update Centralized Critic
            for hist_batch, b_contexts, returns in self.buffer.get_critic_minibatch_generator(self.minibatch_size):
                B = hist_batch.shape[0]
                with torch.no_grad():
                    flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                    temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)
                    mean_context = {
                        k: float(np.mean([ctx[k] for ctx in b_contexts]))
                        for k in ALLOWED_DIRECTED_EDGES
                    }
                    gated_embs, _, _ = self.gated_gat(temp_embs, context_dict=mean_context, deterministic=False)
                    centralized_input = gated_embs.view(B, self.critic_input_dim)

                vals = self.critic(centralized_input)
                v_loss = 0.5 * ((vals - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

            # 2. Update Transformer + Gated GAT + Actor + Causal Pathway (End-to-End)
            for hist_batch, b_contexts, acts, old_lps, advs in self.buffer.get_actor_minibatch_generator(self.minibatch_size):
                B = hist_batch.shape[0]

                flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)

                mean_context = {
                    k: float(np.mean([ctx[k] for ctx in b_contexts]))
                    for k in ALLOWED_DIRECTED_EDGES
                }
                gated_embs, raw_gates, raw_logits = self.gated_gat(temp_embs, context_dict=mean_context, deterministic=False)

                flat_gated = gated_embs.view(B * self.num_agents, self.gat_hidden_dim)
                flat_ids = self.one_hot_tensor.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * self.num_agents, 4)
                flat_acts = acts.view(B * self.num_agents)
                flat_old_lps = old_lps.view(B * self.num_agents)
                flat_advs = advs.expand(-1, self.num_agents).contiguous().view(B * self.num_agents)

                dist = self.actor(flat_gated, flat_ids)
                new_lps = dist.log_prob(flat_acts)
                entropy = dist.entropy().mean()

                log_ratio = new_lps - flat_old_lps
                ratio = torch.exp(log_ratio)

                surr1 = ratio * flat_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * flat_advs
                pol_loss = -torch.min(surr1, surr2).mean()

                # Communication budget penalty participating directly in loss
                # Mean soft gate across all batch samples and edges
                mean_soft_gate = raw_gates.mean()
                excess_comm = torch.relu(mean_soft_gate - self.target_ratio)
                budget_loss = self.budget_penalty_coef * (excess_comm ** 2) + self.lambda_comm * mean_soft_gate

                # Total actor loss
                total_actor_loss = pol_loss - self.entropy_coef * entropy + budget_loss

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()

                # Track causal pathway gradient norm before clipping
                c_grads = [p.grad.detach() for p in self.gated_gat.comm_gate.causal_encoder.parameters() if p.grad is not None]
                if c_grads:
                    c_norm = torch.norm(torch.stack([torch.norm(g) for g in c_grads])).item()
                    causal_grad_norms.append(float(c_norm))

                trainable_params = (
                    list(self.transformer_encoder.parameters()) +
                    list(self.gated_gat.parameters()) +
                    list(self.actor.parameters())
                )
                nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.actor_optimizer.step()

                actor_losses.append(pol_loss.item())
                budget_losses.append(budget_loss.item())
                entropies.append(entropy.item())

        self.buffer.clear()

        return {
            "policy_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "comm_budget_loss": float(np.mean(budget_losses)) if budget_losses else 0.0,
            "causal_grad_norm": float(np.mean(causal_grad_norms)) if causal_grad_norms else 0.0
        }

    def save_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Save separate Phase 8R checkpoints."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        actor_path = os.path.join(checkpoint_dir, "mappo_causal_improved_actor.pt")
        torch.save({
            "transformer_state_dict": self.transformer_encoder.state_dict(),
            "gated_gat_state_dict": self.gated_gat.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim,
            "gat_hidden_dim": self.gat_hidden_dim,
            "causal_dim": self.causal_dim,
            "hard_threshold": self.hard_threshold
        }, actor_path)

        critic_path = os.path.join(checkpoint_dir, "mappo_causal_improved_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "input_dim": self.critic_input_dim
        }, critic_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load Phase 8R checkpoints."""
        actor_path = os.path.join(checkpoint_dir, "mappo_causal_improved_actor.pt")
        if os.path.exists(actor_path):
            ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
            self.transformer_encoder.load_state_dict(ckpt["transformer_state_dict"])
            self.gated_gat.load_state_dict(ckpt["gated_gat_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])

        critic_path = os.path.join(checkpoint_dir, "mappo_causal_improved_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
            self.critic.load_state_dict(ckpt["critic_state_dict"])
