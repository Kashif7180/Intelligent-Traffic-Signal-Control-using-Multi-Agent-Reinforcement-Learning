"""
Phase 7: MAPPO + Temporal Transformer + Graph Attention Network + Learnable Communication Gate.

Architecture Pipeline:
1. Temporal Transformer Encoder: [B, k, 43] -> [B, 64].
2. Physical Communication Graph:
   Preserves Phase 6 physical topology (A <-> B, A <-> C, B <-> D, C <-> D).
   No diagonal communication (A <-> D, B <-> C disabled).
   Exactly 8 directed communication edges per step.
   Self-loops are local and NOT counted as inter-agent communication.
3. Learnable Communication Gate:
   For every allowed directed edge j -> i (sender j -> receiver i):
   g_ij = sigmoid(MLP([h_i, h_j, c_ij])) in (0, 1)
   c_ij = normalized queue/traffic pressure difference heuristic (queue_j - queue_i) / 40.0.
   Note: Phase 7 uses a heuristic context; causal influence estimation is deferred to Phase 8.
4. Gated Graph Attention Layer:
   Modulates neighbor messages by g_ij:
   h'_i = sigma(m_ii + sum_{j in N(i), j != i} g_ij * m_ij) in R^64.
   Zero gate suppresses neighbor message; full gate passes neighbor message.
5. Decentralized Actor:
   Receives 64-D gated embedding + 4-D agent ID = 68-D -> 4 action logits.
6. Centralized Critic (Training-Only):
   Receives concatenation of all 4 gated embeddings = 256-D -> 1 scalar V(S_comm).
7. Communication Cost & Rewards:
   Task reward R_i, communication cost lambda_comm * sum_j g_ij, adjusted reward R'_i = R_i - cost_i.
   Tracked and logged separately.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Dict, List, Tuple, Optional, Generator, Any

from models.transformer_network import TemporalTransformerEncoder, TemporalHistoryManager, layer_init
from models.gat_network import build_physical_adjacency_matrix, DecentralizedGATActor, CentralizedGATCritic
from models.causal_influence import CausalInfluenceEstimator


# Exactly 8 allowed directed inter-agent communication edges
ALLOWED_DIRECTED_EDGES = [
    ("A", "B"), ("B", "A"),
    ("A", "C"), ("C", "A"),
    ("B", "D"), ("D", "B"),
    ("C", "D"), ("D", "C")
]


class LearnableCommunicationGate(nn.Module):
    """
    Trainable MLP Communication Gate for inter-agent message passing.
    
    For each allowed directed edge j -> i:
    g_ij = sigmoid(MLP([h_i, h_j, c_ij]))
    
    where:
    - h_i in R^64 (receiver representation)
    - h_j in R^64 (sender representation)
    - c_ij in R^1  (heuristic context: normalized pressure difference)
    Input dimension: 64 + 64 + 1 = 129
    """

    def __init__(
        self,
        node_dim: int = 64,
        context_dim: int = 1,
        hidden_dim: int = 32,
        threshold: float = 0.5,
        agents: List[str] = ["A", "B", "C", "D"]
    ):
        super().__init__()
        self.node_dim = node_dim
        self.context_dim = context_dim
        self.input_dim = 2 * node_dim + context_dim  # 129
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.agents = agents
        self.agent_to_idx = {a: i for i, a in enumerate(agents)}

        # Edge list and index mappings
        self.allowed_edges = ALLOWED_DIRECTED_EDGES
        self.num_allowed_edges = len(self.allowed_edges)  # 8

        # Trainable Gate MLP: Linear(129, hidden_dim) -> Tanh -> Linear(hidden_dim, 1) -> Sigmoid
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            nn.Sigmoid()
        )

        # Cached evaluation values
        self.last_gate_values: Dict[Tuple[str, str], float] = {}
        self.last_comm_decisions: Dict[Tuple[str, str], bool] = {}

    def forward(
        self,
        node_features: torch.Tensor,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute communication gate values for all 8 allowed edges.
        
        Args:
            node_features: [B, 4, 64] or [4, 64]
            context_dict: Mapping (src, tgt) -> float heuristic context
            deterministic: If True, uses hard threshold for comm decisions
            
        Returns:
            gate_matrix: [B, 4, 4] tensor where gate_matrix[b, tgt, src] = g_{src -> tgt}
                         for allowed edges; 1.0 for self-loops; 0.0 for diagonals.
            raw_gates: [B, 8] tensor of gate values for the 8 allowed edges.
        """
        is_2d = (node_features.dim() == 2)
        if is_2d:
            node_features = node_features.unsqueeze(0)  # [1, 4, 64]

        B, N, D = node_features.shape
        device = node_features.device

        # Initialize gate matrix [B, 4, 4] with 1.0 on diagonal (self-loops) and 0.0 elsewhere
        gate_matrix = torch.zeros((B, 4, 4), device=device, dtype=torch.float32)
        for i in range(4):
            gate_matrix[:, i, i] = 1.0  # Self-loops always pass, not inter-agent communication

        raw_gate_list = []
        edge_values_step = {}
        edge_decisions_step = {}

        # Compute gate for each allowed directed edge (src -> tgt)
        for src, tgt in self.allowed_edges:
            src_idx = self.agent_to_idx[src]
            tgt_idx = self.agent_to_idx[tgt]

            # Receiver h_tgt, Sender h_src
            h_tgt = node_features[:, tgt_idx, :]  # [B, 64]
            h_src = node_features[:, src_idx, :]  # [B, 64]

            # Context c_{src -> tgt}
            c_val = context_dict.get((src, tgt), 0.0)
            c_tensor = torch.full((B, 1), c_val, device=device, dtype=torch.float32)

            # Concatenate [h_tgt, h_src, c_ij] = [B, 129]
            mlp_in = torch.cat([h_tgt, h_src, c_tensor], dim=-1)
            g_soft = self.mlp(mlp_in)  # [B, 1] in (0, 1)

            if deterministic:
                # Hard thresholding for evaluation
                g_eff = (g_soft >= self.threshold).float()
            else:
                # Soft differentiable gate during training
                g_eff = g_soft

            gate_matrix[:, tgt_idx, src_idx] = g_eff.squeeze(-1)
            raw_gate_list.append(g_soft)

            # Store inspection record for batch index 0
            val = float(g_soft[0, 0].item())
            edge_values_step[(src, tgt)] = val
            edge_decisions_step[(src, tgt)] = bool(val >= self.threshold)

        self.last_gate_values = edge_values_step
        self.last_comm_decisions = edge_decisions_step

        raw_gates = torch.cat(raw_gate_list, dim=-1)  # [B, 8]

        if is_2d:
            gate_matrix = gate_matrix.squeeze(0)

        return gate_matrix, raw_gates


class GatedGraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with Learnable Communication Gating.
    
    Formulation:
    Messages from neighbor j to target i are modulated by gate g_ij:
    m_{ij}^k = alpha_{ij}^k W^k h_j
    tilde{m}_{ij}^k = g_{ij} * m_{ij}^k  (for j != i)
    tilde{m}_{ii}^k = m_{ii}^k            (for j == i, self-loop)
    
    Output for node i:
    h'_i = sigma((1/K) sum_{k=1}^K sum_{j in N(i)} tilde{m}_{ij}^k)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        concat_heads: bool = False,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        gate_threshold: float = 0.5,
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

        # Physical road adjacency
        adj = build_physical_adjacency_matrix(agents=agents, add_self_loops=True)
        self.register_buffer("adj", adj)

        # GAT weights
        self.W = nn.Parameter(torch.empty(num_heads, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

        self.a_src = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.ELU()

        # Learnable Communication Gate
        self.comm_gate = LearnableCommunicationGate(
            node_dim=input_dim,
            context_dim=1,
            hidden_dim=32,
            threshold=gate_threshold,
            agents=agents
        )

        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_gate_matrix: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated message passing.
        
        Args:
            x: [B, N, input_dim]
            context_dict: Heuristic context features
            deterministic: Deterministic evaluation mode
            
        Returns:
            out: [B, N, hidden_dim] gated representations
            raw_gates: [B, 8] raw gate values
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
        masked_logits = logits.masked_fill(mask == 0.0, -1e9)
        alpha = torch.softmax(masked_logits, dim=-1)  # [B, K, N, N]
        alpha_dropped = self.dropout(alpha)
        self.last_attention_weights = alpha.detach()

        # Compute Communication Gate Matrix: [B, N, N] (tgt, src)
        gate_matrix, raw_gates = self.comm_gate(x, context_dict, deterministic=deterministic)
        self.last_gate_matrix = gate_matrix.detach()

        # Modulate attention logits by physical adjacency and communication gate:
        # Gate matrix expanded to heads: [B, 1, N, N]
        gate_expanded = gate_matrix.unsqueeze(1)
        gate_log = torch.log(torch.clamp(gate_expanded, min=1e-8, max=1.0))

        # Effective logit combines raw GAT logit + log(gate)
        gated_logits = logits + gate_log
        # Mask out physically non-adjacent edges and fully closed gates
        is_masked = (mask == 0.0) | (gate_expanded < 1e-6)
        masked_logits = gated_logits.masked_fill(is_masked, -1e9)

        alpha = torch.softmax(masked_logits, dim=-1)  # [B, K, N, N]
        alpha_dropped = self.dropout(alpha)
        self.last_attention_weights = alpha.detach()

        # Aggregate messages: out_k [B, K, N, H]
        out_k = torch.einsum("bkij, bkjh -> bkih", alpha_dropped, Wh)

        if self.concat_heads:
            out = out_k.permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.hidden_dim)
        else:
            out = out_k.mean(dim=1)

        out = self.activation(out)

        if is_2d:
            out = out.squeeze(0)

        return out, raw_gates


class MAPPOGateRolloutBuffer:
    """
    CTDE Rollout Buffer for Phase 7.
    Stores transitions, task rewards, communication costs, adjusted returns, and gate decisions.
    """

    def __init__(self, agents: List[str], history_length: int = 4, obs_dim: int = 43):
        self.agents = agents
        self.num_agents = len(agents)
        self.k = history_length
        self.obs_dim = obs_dim

        self.histories: Dict[str, List[np.ndarray]] = {a: [] for a in agents}
        self.contexts: List[Dict[Tuple[str, str], float]] = []
        self.actions: Dict[str, List[int]] = {a: [] for a in agents}
        self.log_probs: Dict[str, List[float]] = {a: [] for a in agents}
        self.values: List[float] = []

        # Separated rewards
        self.task_rewards: Dict[str, List[float]] = {a: [] for a in agents}
        self.comm_costs: Dict[str, List[float]] = {a: [] for a in agents}
        self.adjusted_rewards: List[float] = []  # global adjusted reward for critic GAE
        self.dones: List[bool] = []

        # Gate stats
        self.actual_comms: List[int] = []
        self.possible_comms: List[int] = []

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        histories: Dict[str, np.ndarray],
        context_dict: Dict[Tuple[str, str], float],
        value: float,
        actions: Dict[str, int],
        log_probs: Dict[str, float],
        task_rewards: Dict[str, float],
        comm_costs: Dict[str, float],
        actual_comm: int,
        done: bool
    ):
        """Store transitions for step t."""
        for a in self.agents:
            self.histories[a].append(histories[a].copy())
            self.actions[a].append(actions[a])
            self.log_probs[a].append(log_probs[a])
            self.task_rewards[a].append(task_rewards[a])
            self.comm_costs[a].append(comm_costs[a])

        self.contexts.append(context_dict.copy())
        self.values.append(value)

        # Adjusted reward = task_reward - comm_cost
        adj_step_reward = sum(task_rewards.values()) - sum(comm_costs.values())
        self.adjusted_rewards.append(float(adj_step_reward))
        self.dones.append(done)

        self.actual_comms.append(actual_comm)
        self.possible_comms.append(len(ALLOWED_DIRECTED_EDGES))  # 8

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages on adjusted rewards."""
        num_steps = len(self.values)
        adv = np.zeros(num_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = self.values[t + 1]

            delta = self.adjusted_rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae

        self.advantages = adv
        self.returns = adv + np.array(self.values, dtype=np.float32)

    def get_critic_minibatch_generator(self, minibatch_size: int) -> Generator:
        """Yields minibatches of all 4 agents' histories, contexts, and target returns."""
        num_steps = len(self.values)
        indices = np.random.permutation(num_steps)

        all_hist = []
        for t in range(num_steps):
            step_h = np.stack([self.histories[a][t] for a in self.agents], axis=0)
            all_hist.append(step_h)

        hist_t = torch.tensor(np.array(all_hist), dtype=torch.float32)
        returns_t = torch.tensor(self.returns, dtype=torch.float32).unsqueeze(-1)

        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            b_contexts = [self.contexts[i] for i in b_idx]
            yield hist_t[b_idx], b_contexts, returns_t[b_idx]

    def get_actor_minibatch_generator(self, minibatch_size: int) -> Generator:
        """Yields minibatches preserving graph structure [B, 4, k, 43] for end-to-end training."""
        num_steps = len(self.values)
        indices = np.random.permutation(num_steps)
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        all_hist = []
        all_acts = []
        all_lps = []
        for t in range(num_steps):
            step_h = np.stack([self.histories[a][t] for a in self.agents], axis=0)
            step_a = np.array([self.actions[a][t] for a in self.agents], dtype=np.int64)
            step_lp = np.array([self.log_probs[a][t] for a in self.agents], dtype=np.float32)
            all_hist.append(step_h)
            all_acts.append(step_a)
            all_lps.append(step_lp)

        hist_t = torch.tensor(np.array(all_hist), dtype=torch.float32)
        acts_t = torch.tensor(np.array(all_acts), dtype=torch.long)
        old_lp_t = torch.tensor(np.array(all_lps), dtype=torch.float32)
        adv_t = torch.tensor(adv_norm, dtype=torch.float32).unsqueeze(-1)

        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            b_contexts = [self.contexts[i] for i in b_idx]
            yield hist_t[b_idx], b_contexts, acts_t[b_idx], old_lp_t[b_idx], adv_t[b_idx]

    def clear(self):
        """Reset buffer."""
        for a in self.agents:
            self.histories[a].clear()
            self.actions[a].clear()
            self.log_probs[a].clear()
            self.task_rewards[a].clear()
            self.comm_costs[a].clear()
        self.contexts.clear()
        self.values.clear()
        self.adjusted_rewards.clear()
        self.dones.clear()
        self.actual_comms.clear()
        self.possible_comms.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.values)


class MAPPOGateController:
    """
    Phase 7 Controller: MAPPO + Temporal Transformer + GAT + Learnable Communication Gate.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.t_cfg = config.get("transformer", {})
        self.gat_cfg = config.get("gat", {})
        self.comm_cfg = config.get("communication", {})
        self.agents = config["agent"]["agents"]
        self.num_agents = len(self.agents)

        # Dimensions & parameters
        self.obs_dim = 43
        self.history_length = int(self.t_cfg.get("history_length", 4))
        self.embedding_dim = int(self.gat_cfg.get("input_dim", 64))
        self.gat_hidden_dim = int(self.gat_cfg.get("hidden_dim", 64))
        self.comm_hidden_dim = int(self.comm_cfg.get("hidden_dim", 32))
        self.comm_threshold = float(self.comm_cfg.get("threshold", 0.5))
        self.comm_cost_weight = float(self.comm_cfg.get("cost", 0.01))

        self.agent_id_dim = 4
        self.act_dim = 4
        self.actor_input_dim = self.gat_hidden_dim + self.agent_id_dim  # 68
        self.critic_input_dim = self.num_agents * self.gat_hidden_dim   # 256

        # Hyperparameters
        self.lr_actor = float(self.comm_cfg.get("lr_actor", 0.0003))
        self.lr_critic = float(self.comm_cfg.get("lr_critic", 0.0005))
        self.gamma = float(self.comm_cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.comm_cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.comm_cfg.get("clip_ratio", 0.2))
        self.value_coef = float(self.comm_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(self.comm_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.comm_cfg.get("max_grad_norm", 0.5))
        self.epochs = int(self.comm_cfg.get("epochs", 4))
        self.minibatch_size = int(self.comm_cfg.get("minibatch_size", 32))

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

        # 2. Gated Graph Attention Layer (includes Learnable Communication Gate)
        self.gated_gat = GatedGraphAttentionLayer(
            input_dim=self.embedding_dim,
            hidden_dim=self.gat_hidden_dim,
            num_heads=int(self.gat_cfg.get("num_heads", 4)),
            concat_heads=bool(self.gat_cfg.get("concat_heads", False)),
            dropout=float(self.gat_cfg.get("dropout", 0.0)),
            gate_threshold=self.comm_threshold,
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

        # History manager & buffer
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

    def compute_heuristic_context(self, info_dict: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """
        Compute normalized traffic pressure / queue difference heuristic context:
        c_ij = (queue_j - queue_i) / 40.0 in [-1, 1].
        Deterministic, available online. Causal estimation is deferred to Phase 8.
        """
        contexts = {}
        for src, tgt in ALLOWED_DIRECTED_EDGES:
            q_src = info_dict.get(src, {}).get("metrics", {}).get("queue_length", 0.0)
            q_tgt = info_dict.get(tgt, {}).get("metrics", {}).get("queue_length", 0.0)
            # Pressure difference normalized by 40.0
            diff = (q_src - q_tgt) / 40.0
            contexts[(src, tgt)] = float(np.clip(diff, -1.0, 1.0))
        return contexts

    def reset_history(self, initial_obs_dict: Dict[str, np.ndarray]):
        self.history_manager.reset(initial_obs_dict)

    def update_history(self, next_obs_dict: Dict[str, np.ndarray]):
        self.history_manager.update(next_obs_dict)

    def get_actions(
        self,
        context_dict: Dict[Tuple[str, str], float],
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float], int]:
        """
        Decentralized Action Selection via Gated Communication:
        1. Encodes history to 64-D via Temporal Transformer.
        2. Gated GAT message passing with learnable gate g_ij.
        3. Decentralized actors receive local 68-D input -> action.
        
        Returns:
            actions: Dict[agent, action]
            log_probs: Dict[agent, log_prob]
            comm_costs: Dict[agent, cost]
            actual_comm_count: int
        """
        actions = {}
        log_probs = {}

        self.transformer_encoder.eval()
        self.gated_gat.eval()
        self.actor.eval()

        with torch.no_grad():
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32).unsqueeze(0)  # [1, 4, k, 43]

            flat_hist = hist_tensor.view(4, self.history_length, self.obs_dim)
            temp_embs = self.transformer_encoder(flat_hist)  # [4, 64]

            # Gated GAT forward pass
            gated_embs, _ = self.gated_gat(
                temp_embs.unsqueeze(0),
                context_dict=context_dict,
                deterministic=deterministic
            )
            gated_embs = gated_embs.squeeze(0)  # [4, 64]

            # Actor decisions
            for i, a in enumerate(self.agents):
                node_emb = gated_embs[i:i+1]
                agent_id = self.one_hot_tensor[i:i+1]
                dist = self.actor(node_emb, agent_id)

                if deterministic:
                    act = torch.argmax(dist.probs, dim=-1)
                else:
                    act = dist.sample()

                lp = dist.log_prob(act)
                actions[a] = int(act.item())
                log_probs[a] = float(lp.item())

        # Compute communication costs per agent based on incoming messages
        comm_costs = {a: 0.0 for a in self.agents}
        actual_comm_count = 0

        for (src, tgt), g_val in self.gated_gat.comm_gate.last_gate_values.items():
            is_comm = (g_val >= self.comm_threshold)
            if is_comm:
                actual_comm_count += 1

            if deterministic:
                cost = self.comm_cost_weight * (1.0 if is_comm else 0.0)
            else:
                cost = self.comm_cost_weight * g_val

            comm_costs[tgt] += float(cost)

        return actions, log_probs, comm_costs, actual_comm_count

    def get_centralized_value(self, context_dict: Dict[Tuple[str, str], float]) -> float:
        """Centralized Critic Value Estimation (TRAINING ONLY)."""
        self.transformer_encoder.eval()
        self.gated_gat.eval()
        self.critic.eval()

        with torch.no_grad():
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)

            temp_embs = self.transformer_encoder(hist_tensor)
            gated_embs, _ = self.gated_gat(temp_embs.unsqueeze(0), context_dict=context_dict, deterministic=False)

            centralized_state = gated_embs.view(1, self.critic_input_dim)
            val = self.critic(centralized_state).squeeze()
            return float(val.item())

    def update(self, last_value: float) -> Dict[str, float]:
        """PPO update with communication cost & gated backprop."""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        self.transformer_encoder.train()
        self.gated_gat.train()
        self.actor.train()
        self.critic.train()

        critic_losses = []
        actor_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.epochs):
            # 1. Update Centralized Critic
            for hist_batch, b_contexts, returns in self.buffer.get_critic_minibatch_generator(self.minibatch_size):
                B = hist_batch.shape[0]
                with torch.no_grad():
                    flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                    temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)
                    # Use mean context of batch for critic state
                    mean_context = {
                        k: float(np.mean([ctx[k] for ctx in b_contexts]))
                        for k in ALLOWED_DIRECTED_EDGES
                    }
                    gated_embs, _ = self.gated_gat(temp_embs, context_dict=mean_context, deterministic=False)
                    centralized_input = gated_embs.view(B, self.critic_input_dim)

                vals = self.critic(centralized_input)
                v_loss = 0.5 * ((vals - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

            # 2. Update Transformer + Gated GAT + Actor (End-to-End)
            for hist_batch, b_contexts, acts, old_lps, advs in self.buffer.get_actor_minibatch_generator(self.minibatch_size):
                B = hist_batch.shape[0]

                flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)

                mean_context = {
                    k: float(np.mean([ctx[k] for ctx in b_contexts]))
                    for k in ALLOWED_DIRECTED_EDGES
                }
                gated_embs, _ = self.gated_gat(temp_embs, context_dict=mean_context, deterministic=False)

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

                loss = pol_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                loss.backward()

                trainable_params = (
                    list(self.transformer_encoder.parameters()) +
                    list(self.gated_gat.parameters()) +
                    list(self.actor.parameters())
                )
                nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.actor_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)

                actor_losses.append(pol_loss.item())
                entropies.append(entropy.item())

        self.buffer.clear()

        return {
            "policy_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0
        }

    def save_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Save Phase 7 checkpoints separately."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        actor_path = os.path.join(checkpoint_dir, "mappo_comm_gate_actor.pt")
        torch.save({
            "transformer_state_dict": self.transformer_encoder.state_dict(),
            "gated_gat_state_dict": self.gated_gat.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim,
            "gat_hidden_dim": self.gat_hidden_dim,
            "comm_threshold": self.comm_threshold
        }, actor_path)

        critic_path = os.path.join(checkpoint_dir, "mappo_comm_gate_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "input_dim": self.critic_input_dim
        }, critic_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load Phase 7 checkpoints."""
        actor_path = os.path.join(checkpoint_dir, "mappo_comm_gate_actor.pt")
        if os.path.exists(actor_path):
            ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
            self.transformer_encoder.load_state_dict(ckpt["transformer_state_dict"])
            self.gated_gat.load_state_dict(ckpt["gated_gat_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])

        critic_path = os.path.join(checkpoint_dir, "mappo_comm_gate_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
            self.critic.load_state_dict(ckpt["critic_state_dict"])


class MAPPOCausalGateController(MAPPOGateController):
    """
    Phase 8 Controller: MAPPO + Temporal Transformer + GAT + Causal Influence Gate.
    
    Replaces the Phase 7 heuristic queue-difference context with directional causal influence
    estimation scores C_ij in [0, 1] computed by CausalInfluenceEstimator.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path=config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ci_cfg = config.get("causal_influence", {})
        norm_cfg = config.get("observation", {}).get("normalization", {})

        self.causal_estimator = CausalInfluenceEstimator(
            method=ci_cfg.get("method", "granger_style"),
            history_length=int(ci_cfg.get("history_length", 5)),
            window_size=int(ci_cfg.get("window_size", 15)),
            score_min=float(ci_cfg.get("score_min", 0.0)),
            score_max=float(ci_cfg.get("score_max", 1.0)),
            warmup_value=float(ci_cfg.get("warmup_value", 0.5)),
            ridge_lambda=float(ci_cfg.get("ridge_lambda", 0.001)),
            max_queue=float(norm_cfg.get("max_queue_length", 40.0)),
            max_wait=float(norm_cfg.get("max_waiting_time", 120.0)),
            max_veh=float(norm_cfg.get("max_vehicle_count", 40.0)),
            agents=self.agents
        )
        self.last_causal_scores: Dict[Tuple[str, str], float] = {}

    def compute_causal_context(self, info_dict: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """
        Compute directional causal influence estimation scores C_ij in [0, 1]
        for all 8 allowed edges j -> i.
        """
        self.causal_estimator.update_traffic_state(info_dict)
        scores = self.causal_estimator.estimate_causal_influence()
        self.last_causal_scores = scores.copy()
        return scores

    def reset_history(self, initial_obs_dict: Dict[str, np.ndarray], info_dict: Optional[Dict[str, Any]] = None):
        super().reset_history(initial_obs_dict)
        self.causal_estimator.reset()
        if info_dict is not None:
            self.causal_estimator.update_traffic_state(info_dict)
        self.last_causal_scores = self.causal_estimator.estimate_causal_influence()

    def save_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Save Phase 8 checkpoints separately."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        actor_path = os.path.join(checkpoint_dir, "mappo_causal_gate_actor.pt")
        torch.save({
            "transformer_state_dict": self.transformer_encoder.state_dict(),
            "gated_gat_state_dict": self.gated_gat.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim,
            "gat_hidden_dim": self.gat_hidden_dim,
            "comm_threshold": self.comm_threshold
        }, actor_path)

        critic_path = os.path.join(checkpoint_dir, "mappo_causal_gate_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "input_dim": self.critic_input_dim
        }, critic_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load Phase 8 checkpoints."""
        actor_path = os.path.join(checkpoint_dir, "mappo_causal_gate_actor.pt")
        if os.path.exists(actor_path):
            ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
            self.transformer_encoder.load_state_dict(ckpt["transformer_state_dict"])
            self.gated_gat.load_state_dict(ckpt["gated_gat_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])

        critic_path = os.path.join(checkpoint_dir, "mappo_causal_gate_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
            self.critic.load_state_dict(ckpt["critic_state_dict"])
