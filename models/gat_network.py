"""
Phase 6: MAPPO + Temporal Transformer + Graph Attention Network (GAT) for SAGE-Traffic.

Architecture Pipeline:
1. Temporal History: Sliding window of k=4 observations [k, 43] per intersection.
2. Temporal Transformer Encoder: PyTorch Transformer processing [B, k, 43] -> [B, 64].
3. Graph Attention Network (GAT) Layer:
   Processes [B, 4, 64] node representations over the physical road topology graph:
   A <-> B, A <-> C, B <-> D, C <-> D.
   Diagonals (A-D, B-C) are masked with -inf so non-neighbors cannot receive attention.
   Self-loops are included and distinguished from neighbor attention.
   Computes normalized attention coefficients alpha_ij (sum_j alpha_ij = 1.0).
   Produces GAT-enhanced node embeddings [B, 4, 64].
4. Decentralized GAT Actor:
   Receives ONLY its own node's 64-D GAT embedding + 4-D agent ID = 68-D.
   Network: 68 -> 64 -> 64 -> 4 action logits.
5. Centralized GAT Critic (Training-Only):
   Receives concatenation of all 4 nodes' 64-D GAT embeddings = 256-D.
   Network: 256 -> 128 -> 64 -> 1 scalar value V(S_gat).
6. MAPPOGATRolloutBuffer: Multi-agent rollout buffer for end-to-end PPO training.
7. MAPPOGATController: Orchestrates decentralized execution, centralized training,
   and attention weight extraction.
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


def build_physical_adjacency_matrix(
    agents: List[str] = ["A", "B", "C", "D"],
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Build the adjacency matrix strictly from the physical road network.
    
    Road Topology (from network/build_network.py):
    A: (0, 300)      B: (300, 300)
    C: (0, 0)        D: (300, 0)
    
    Physical Bidirectional Edges:
    - A <-> B (internal horizontal edge)
    - A <-> C (internal vertical edge)
    - B <-> D (internal vertical edge)
    - C <-> D (internal horizontal edge)
    
    Diagonals A <-> D and B <-> C do NOT exist physically.
    
    Adjacency Matrix (row = target i, col = source j, message j -> i):
    With self-loops:
      A: [1, 1, 1, 0]  (self, B, C)
      B: [1, 1, 0, 1]  (A, self, D)
      C: [1, 0, 1, 1]  (A, self, D)
      D: [0, 1, 1, 1]  (B, C, self)
    """
    idx_map = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    adj = torch.zeros((n, n), dtype=torch.float32)

    # Physical road connections
    physical_edges = [
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "D"), ("D", "B"),
        ("C", "D"), ("D", "C"),
    ]

    for u, v in physical_edges:
        if u in idx_map and v in idx_map:
            adj[idx_map[u], idx_map[v]] = 1.0

    if add_self_loops:
        for i in range(n):
            adj[i, i] = 1.0

    return adj


class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Network (GAT) Layer.
    
    Formulation:
    Given node features H in R^{B x N x F_in} (where N=4, F_in=64):
    For each head k in {1, ..., K}:
      1. Node projection: z_i^k = W^k h_i in R^{F_out}
      2. Attention logit for edge j -> i:
         e_{ij}^k = LeakyReLU(a_dst^{k T} z_i^k + a_src^{k T} z_j^k)  if (i, j) in E
         e_{ij}^k = -1e9                                                if (i, j) not in E
      3. Attention normalization:
         alpha_{ij}^k = exp(e_{ij}^k) / sum_{l in N(i)} exp(e_{il}^k)
      4. Message aggregation:
         h_i'^k = sum_{j in N(i)} alpha_{ij}^k z_j^k
    
    If concat_heads == False:
      Averages representations across heads:
      h_i' = (1/K) sum_{k=1}^K h_i'^k in R^{F_out} (64-D)
    
    Exposes:
      self.last_attention_weights: [B, K, N, N]
      self.last_aggregated_weights: [B, N, N]
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        concat_heads: bool = False,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        negative_slope: float = 0.2,
        agents: List[str] = ["A", "B", "C", "D"]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.add_self_loops = add_self_loops
        self.negative_slope = negative_slope
        self.agents = agents
        self.num_nodes = len(agents)

        # Build physical adjacency matrix
        adj = build_physical_adjacency_matrix(agents=agents, add_self_loops=add_self_loops)
        self.register_buffer("adj", adj)

        # Trainable parameters
        # Learnable node projection W: [num_heads, input_dim, hidden_dim]
        self.W = nn.Parameter(torch.empty(num_heads, input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

        # Attention weight vectors: a_src and a_dst of shape [num_heads, hidden_dim, 1]
        self.a_src = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, hidden_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.ELU()

        # Cached attention weights for interpretability
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_aggregated_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: [B, N, input_dim] or [N, input_dim]
        Output: [B, N, hidden_dim]
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0)  # [1, N, input_dim]

        B, N, _ = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        # 1. Project node features for all heads: [B, K, N, hidden_dim]
        # einsum: B: batch, N: nodes, I: input_dim, K: heads, H: hidden_dim
        Wh = torch.einsum("bni, kih -> bknh", x, self.W)

        # 2. Compute attention logits
        # attn_dst: [B, K, N, 1] -> target node i
        # attn_src: [B, K, 1, N] -> source node j
        attn_dst = torch.einsum("bknh, kho -> bkno", Wh, self.a_dst)  # [B, K, N, 1]
        attn_src = torch.einsum("bknh, kho -> bkno", Wh, self.a_src)  # [B, K, N, 1]
        attn_src = attn_src.transpose(2, 3)                           # [B, K, 1, N]

        # Raw attention scores before masking: e_{ij} = LeakyReLU(a_dst^T z_i + a_src^T z_j)
        # Broadcasting: [B, K, N, 1] + [B, K, 1, N] -> [B, K, N, N]
        logits = self.leaky_relu(attn_dst + attn_src)

        # 3. Mask non-adjacent edges with large negative value (-1e9)
        # adj shape: [N, N], 1 where edge j -> i exists, 0 otherwise
        mask = self.adj.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        masked_logits = logits.masked_fill(mask == 0.0, -1e9)

        # 4. Softmax normalization over source nodes (dim=-1, sum_j alpha_ij = 1.0)
        alpha = torch.softmax(masked_logits, dim=-1)  # [B, K, N, N]
        alpha_dropped = self.dropout(alpha)

        # Store attention weights for inspection (target i, source j)
        self.last_attention_weights = alpha.detach()
        self.last_aggregated_weights = alpha.mean(dim=1).detach()

        # 5. Message aggregation: h_i'^k = sum_j alpha_{ij}^k Wh_j^k
        # einsum: alpha [B, K, N, N], Wh [B, K, N, H] -> out_k [B, K, N, H]
        out_k = torch.einsum("bkij, bkjh -> bkih", alpha_dropped, Wh)

        # 6. Multi-head combination
        if self.concat_heads:
            # [B, N, K * H]
            out = out_k.permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.hidden_dim)
        else:
            # Average across heads: [B, N, H]
            out = out_k.mean(dim=1)

        # 7. Nonlinear activation
        out = self.activation(out)

        if is_2d:
            out = out.squeeze(0)

        return out

    def get_attention_weights(self) -> Dict[str, Any]:
        """
        Extract explicit, interpretable attention coefficients from the most recent forward pass.
        
        Returns:
            Dictionary containing:
            - 'per_head': {head_idx: [{'source': src, 'target': tgt, 'weight': float, 'is_self_loop': bool}]}
            - 'aggregated': [{'source': src, 'target': tgt, 'weight': float, 'is_self_loop': bool}]
            - 'summary': {f"{src} -> {tgt}": float}
        """
        if self.last_attention_weights is None or self.last_aggregated_weights is None:
            raise RuntimeError("No forward pass has been executed yet. Cannot extract attention weights.")

        # Use batch index 0 for inspection
        alpha = self.last_attention_weights[0].cpu().numpy()     # [K, N, N]
        agg_alpha = self.last_aggregated_weights[0].cpu().numpy() # [N, N]

        per_head = {}
        for k in range(self.num_heads):
            head_edges = []
            for i, tgt in enumerate(self.agents):
                for j, src in enumerate(self.agents):
                    if self.adj[i, j].item() > 0:
                        head_edges.append({
                            "source": src,
                            "target": tgt,
                            "weight": float(alpha[k, i, j]),
                            "is_self_loop": (src == tgt)
                        })
            per_head[k] = head_edges

        aggregated = []
        summary = {}
        for i, tgt in enumerate(self.agents):
            for j, src in enumerate(self.agents):
                if self.adj[i, j].item() > 0:
                    weight = float(agg_alpha[i, j])
                    aggregated.append({
                        "source": src,
                        "target": tgt,
                        "weight": weight,
                        "is_self_loop": (src == tgt)
                    })
                    summary[f"{src} -> {tgt}"] = weight

        return {
            "per_head": per_head,
            "aggregated": aggregated,
            "summary": summary
        }


class DecentralizedGATActor(nn.Module):
    """
    Decentralized Actor Network for Phase 6:
    Input: 64-D GAT node embedding + 4-D agent ID one-hot = 68-D.
    Output: Categorical distribution over 4 discrete actions.
    
    Architecture:
    Linear(68, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 4)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        agent_id_dim: int = 4,
        act_dim: int = 4,
        hidden_dims: List[int] = [64, 64]
    ):
        super().__init__()
        self.input_dim = embedding_dim + agent_id_dim
        self.act_dim = act_dim

        layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, act_dim), std=0.01))
        self.network = nn.Sequential(*layers)

    def forward(self, gat_embedding: torch.Tensor, agent_id_one_hot: torch.Tensor) -> Categorical:
        """
        gat_embedding:     [B, 64]
        agent_id_one_hot:  [B, 4]
        """
        x = torch.cat([gat_embedding, agent_id_one_hot], dim=-1)
        logits = self.network(x)
        return Categorical(logits=logits)

    def forward_flat(self, x: torch.Tensor) -> Categorical:
        """Direct evaluation on 68-D tensor."""
        logits = self.network(x)
        return Categorical(logits=logits)


class CentralizedGATCritic(nn.Module):
    """
    Centralized Critic Network for Phase 6:
    Input: concatenation of all 4 agents' 64-D GAT embeddings = 4 * 64 = 256-D.
    Output: scalar state-value estimate V(S_gat).
    
    Architecture:
    Linear(256, 128) -> Tanh -> Linear(128, 64) -> Tanh -> Linear(64, 1)
    """

    def __init__(
        self,
        num_agents: int = 4,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        self.input_dim = num_agents * embedding_dim  # 4 * 64 = 256

        layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, centralized_gat_state: torch.Tensor) -> torch.Tensor:
        """Input: [B, 256] -> Output: [B, 1]"""
        return self.network(centralized_gat_state)


class MAPPOGATRolloutBuffer:
    """
    Multi-Agent Rollout Buffer for MAPPO + Temporal Transformer + GAT.
    Stores historical observation windows, centralized values, actions, rewards, and dones.
    Computes GAE on centralized values and generates minibatches for end-to-end training.
    """

    def __init__(self, agents: List[str], history_length: int = 4, obs_dim: int = 43):
        self.agents = agents
        self.num_agents = len(agents)
        self.k = history_length
        self.obs_dim = obs_dim

        self.histories: Dict[str, List[np.ndarray]] = {a: [] for a in agents}
        self.actions: Dict[str, List[int]] = {a: [] for a in agents}
        self.log_probs: Dict[str, List[float]] = {a: [] for a in agents}
        self.values: List[float] = []
        self.global_rewards: List[float] = []
        self.dones: List[bool] = []

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        histories: Dict[str, np.ndarray],
        value: float,
        actions: Dict[str, int],
        log_probs: Dict[str, float],
        rewards: Dict[str, float],
        done: bool
    ):
        """Store transitions for step t."""
        for a in self.agents:
            self.histories[a].append(histories[a].copy())
            self.actions[a].append(actions[a])
            self.log_probs[a].append(log_probs[a])

        self.values.append(value)
        self.global_rewards.append(float(sum(rewards.values())))
        self.dones.append(done)

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and discounted returns using centralized critic values."""
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

            delta = self.global_rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae

        self.advantages = adv
        self.returns = adv + np.array(self.values, dtype=np.float32)

    def get_critic_minibatch_generator(self, minibatch_size: int) -> Generator:
        """
        Yield minibatches of all 4 agents' histories and target returns for Centralized Critic.
        """
        num_steps = len(self.values)
        indices = np.random.permutation(num_steps)

        # Shape: [num_steps, 4, k, 43]
        all_hist = []
        for t in range(num_steps):
            step_h = np.stack([self.histories[a][t] for a in self.agents], axis=0)
            all_hist.append(step_h)

        hist_t = torch.tensor(np.array(all_hist), dtype=torch.float32)
        returns_t = torch.tensor(self.returns, dtype=torch.float32).unsqueeze(-1)

        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield hist_t[b_idx], returns_t[b_idx]

    def get_actor_minibatch_generator(
        self,
        minibatch_size: int
    ) -> Generator:
        """
        Yield minibatches of all 4 agents' stacked histories, actions, old log_probs, and advantages.
        Preserves graph structure [B, 4, k, 43] so GAT can operate over physical neighbors during backprop.
        """
        num_steps = len(self.values)
        indices = np.random.permutation(num_steps)
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Stack over timesteps: [num_steps, 4, k, 43]
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

        hist_t = torch.tensor(np.array(all_hist), dtype=torch.float32)     # [num_steps, 4, k, 43]
        acts_t = torch.tensor(np.array(all_acts), dtype=torch.long)         # [num_steps, 4]
        old_lp_t = torch.tensor(np.array(all_lps), dtype=torch.float32)     # [num_steps, 4]
        adv_t = torch.tensor(adv_norm, dtype=torch.float32).unsqueeze(-1)  # [num_steps, 1]

        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield hist_t[b_idx], acts_t[b_idx], old_lp_t[b_idx], adv_t[b_idx]

    def clear(self):
        """Reset buffer contents."""
        for a in self.agents:
            self.histories[a].clear()
            self.actions[a].clear()
            self.log_probs[a].clear()
        self.values.clear()
        self.global_rewards.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.values)


class MAPPOGATController:
    """
    Centralized Training, Decentralized Execution (CTDE) Controller with Temporal Transformer + GAT.
    
    Components:
    1. TemporalTransformerEncoder: [B, k, 43] -> [B, 64]
    2. GraphAttentionLayer: [B, 4, 64] -> [B, 4, 64]
    3. DecentralizedGATActor: 64-D GAT embedding + 4-D agent ID = 68-D -> 4 action logits
    4. CentralizedGATCritic: 4 * 64-D = 256-D -> 1 scalar value V(S_gat)
    5. TemporalHistoryManager: rolling window of exactly k=4 observations
    6. MAPPOGATRolloutBuffer: CTDE rollout buffer supporting end-to-end backprop
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.t_cfg = config.get("transformer", {})
        self.gat_cfg = config.get("gat", {})
        self.agents = config["agent"]["agents"]
        self.num_agents = len(self.agents)

        # Dimensions
        self.obs_dim = 43
        self.history_length = int(self.t_cfg.get("history_length", 4))
        self.embedding_dim = int(self.gat_cfg.get("input_dim", 64))
        self.gat_hidden_dim = int(self.gat_cfg.get("hidden_dim", 64))
        self.gat_num_heads = int(self.gat_cfg.get("num_heads", 4))
        self.gat_concat_heads = bool(self.gat_cfg.get("concat_heads", False))
        self.gat_dropout = float(self.gat_cfg.get("dropout", 0.0))
        self.gat_add_self_loops = bool(self.gat_cfg.get("add_self_loops", True))

        self.agent_id_dim = 4
        self.act_dim = 4
        self.actor_input_dim = self.gat_hidden_dim + self.agent_id_dim  # 68
        self.critic_input_dim = self.num_agents * self.gat_hidden_dim   # 256

        # Hyperparameters
        self.lr_actor = float(self.gat_cfg.get("lr_actor", 0.0003))
        self.lr_critic = float(self.gat_cfg.get("lr_critic", 0.0005))
        self.gamma = float(self.gat_cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.gat_cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.gat_cfg.get("clip_ratio", 0.2))
        self.value_coef = float(self.gat_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(self.gat_cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.gat_cfg.get("max_grad_norm", 0.5))
        self.epochs = int(self.gat_cfg.get("epochs", 4))
        self.minibatch_size = int(self.gat_cfg.get("minibatch_size", 32))

        # One-hot agent IDs
        self.agent_one_hots = {
            a: np.eye(self.num_agents, dtype=np.float32)[i]
            for i, a in enumerate(self.agents)
        }
        self.one_hot_tensor = torch.tensor(
            np.stack([self.agent_one_hots[a] for a in self.agents], axis=0),
            dtype=torch.float32
        )  # [4, 4]

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

        # 2. Graph Attention Network (GAT) Layer
        self.gat = GraphAttentionLayer(
            input_dim=self.embedding_dim,
            hidden_dim=self.gat_hidden_dim,
            num_heads=self.gat_num_heads,
            concat_heads=self.gat_concat_heads,
            dropout=self.gat_dropout,
            add_self_loops=self.gat_add_self_loops,
            agents=self.agents
        )

        # 3. Decentralized GAT Actor (68 -> 64 -> 64 -> 4)
        self.actor = DecentralizedGATActor(
            embedding_dim=self.gat_hidden_dim,
            agent_id_dim=self.agent_id_dim,
            act_dim=self.act_dim,
            hidden_dims=[64, 64]
        )

        # 4. Centralized GAT Critic (256 -> 128 -> 64 -> 1)
        self.critic = CentralizedGATCritic(
            num_agents=self.num_agents,
            embedding_dim=self.gat_hidden_dim,
            hidden_dims=[128, 64]
        )

        # Optimizers (End-to-End Training)
        self.actor_optimizer = optim.Adam(
            list(self.transformer_encoder.parameters()) +
            list(self.gat.parameters()) +
            list(self.actor.parameters()),
            lr=self.lr_actor,
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            eps=1e-5
        )

        # 5. History Manager & Rollout Buffer
        self.history_manager = TemporalHistoryManager(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )
        self.buffer = MAPPOGATRolloutBuffer(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )

    def reset_history(self, initial_obs_dict: Dict[str, np.ndarray]):
        """Populate history buffer by repeating the initial observation k times."""
        self.history_manager.reset(initial_obs_dict)

    def update_history(self, next_obs_dict: Dict[str, np.ndarray]):
        """Slide window: append newly observed state and discard oldest."""
        self.history_manager.update(next_obs_dict)

    def get_actions(
        self,
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Decentralized Action Selection via Graph Message Passing:
        1. Encodes each agent's [k, 43] local history to 64-D via Temporal Transformer.
        2. Passes 4 node representations through GAT over physical road adjacency graph.
        3. Each decentralized actor receives ONLY its own 64-D GAT node embedding + 4-D agent ID = 68-D.
        Centralized critic is NEVER used for action selection.
        """
        actions = {}
        log_probs = {}

        self.transformer_encoder.eval()
        self.gat.eval()
        self.actor.eval()

        with torch.no_grad():
            # Gather all 4 histories: [1, 4, k, 43]
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32).unsqueeze(0)  # [1, 4, k, 43]

            # Flatten to pass through Transformer: [4, k, 43] -> [4, 64]
            flat_hist = hist_tensor.view(4, self.history_length, self.obs_dim)
            temporal_embs = self.transformer_encoder(flat_hist)  # [4, 64]

            # GAT pass: [1, 4, 64] -> [1, 4, 64]
            gat_input = temporal_embs.unsqueeze(0)  # [1, 4, 64]
            gat_embs = self.gat(gat_input).squeeze(0)  # [4, 64]

            # Evaluate each actor independently on its local GAT embedding + agent ID
            for i, a in enumerate(self.agents):
                node_emb = gat_embs[i:i+1]  # [1, 64]
                agent_id = self.one_hot_tensor[i:i+1]  # [1, 4]
                dist = self.actor(node_emb, agent_id)

                if deterministic:
                    act = torch.argmax(dist.probs, dim=-1)
                else:
                    act = dist.sample()

                lp = dist.log_prob(act)
                actions[a] = int(act.item())
                log_probs[a] = float(lp.item())

        return actions, log_probs

    def get_centralized_value(self) -> float:
        """
        Centralized Critic Value Estimation (TRAINING ONLY):
        Encodes all 4 agents' local histories, passes through GAT, concatenates to 256-D, evaluates V(S_gat).
        """
        self.transformer_encoder.eval()
        self.gat.eval()
        self.critic.eval()

        with torch.no_grad():
            hist_list = [self.history_manager.get_agent_history(a) for a in self.agents]
            hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)  # [4, k, 43]

            temporal_embs = self.transformer_encoder(hist_tensor)  # [4, 64]
            gat_embs = self.gat(temporal_embs.unsqueeze(0))         # [1, 4, 64]

            # Concatenate 4 * 64 = 256-D
            centralized_state = gat_embs.view(1, self.critic_input_dim)
            val = self.critic(centralized_state).squeeze()
            return float(val.item())

    def get_attention_weights(self) -> Dict[str, Any]:
        """Expose explicit attention coefficients from the GAT layer."""
        return self.gat.get_attention_weights()

    def update(self, last_value: float) -> Dict[str, float]:
        """
        PPO update for Transformer Encoder, GAT Layer, Decentralized Actor, and Centralized Critic.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        self.transformer_encoder.train()
        self.gat.train()
        self.actor.train()
        self.critic.train()

        critic_losses = []
        actor_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.epochs):
            # 1. Update Centralized Critic on 256-D GAT-enhanced state representations
            for hist_batch, returns in self.buffer.get_critic_minibatch_generator(self.minibatch_size):
                # hist_batch: [B, 4, k, 43]
                B = hist_batch.shape[0]
                with torch.no_grad():
                    flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                    temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)
                    gat_embs = self.gat(temp_embs)  # [B, 4, 64]
                    centralized_input = gat_embs.view(B, self.critic_input_dim)  # [B, 256]

                vals = self.critic(centralized_input)
                v_loss = 0.5 * ((vals - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

            # 2. Update Transformer Encoder + GAT Layer + Decentralized Actor (End-to-End)
            for hist_batch, acts, old_lps, advs in self.buffer.get_actor_minibatch_generator(self.minibatch_size):
                # hist_batch: [B, 4, k, 43]
                # acts: [B, 4], old_lps: [B, 4], advs: [B, 1]
                B = hist_batch.shape[0]

                # End-to-end forward pass:
                # 1. Temporal Transformer: [B * 4, k, 43] -> [B * 4, 64] -> [B, 4, 64]
                flat_hist = hist_batch.view(B * self.num_agents, self.history_length, self.obs_dim)
                temp_embs = self.transformer_encoder(flat_hist).view(B, self.num_agents, self.embedding_dim)

                # 2. GAT message passing: [B, 4, 64] -> [B, 4, 64]
                gat_embs = self.gat(temp_embs)  # [B, 4, 64]

                # 3. Decentralized Actor evaluations for all 4 agents
                # Flatten agents for batch processing: [B * 4, 64]
                flat_gat = gat_embs.view(B * self.num_agents, self.gat_hidden_dim)
                # Expand one-hot agent IDs: [B, 4, 4] -> [B * 4, 4]
                flat_ids = self.one_hot_tensor.unsqueeze(0).expand(B, -1, -1).contiguous().view(B * self.num_agents, 4)
                flat_acts = acts.view(B * self.num_agents)
                flat_old_lps = old_lps.view(B * self.num_agents)
                flat_advs = advs.expand(-1, self.num_agents).contiguous().view(B * self.num_agents)

                dist = self.actor(flat_gat, flat_ids)  # [B * 4, 68] -> Categorical(4)
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

                # Clip gradients across Transformer encoder, GAT layer, and Actor
                trainable_params = (
                    list(self.transformer_encoder.parameters()) +
                    list(self.gat.parameters()) +
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
        """Save Phase 6 MAPPO+Transformer+GAT checkpoints."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        actor_path = os.path.join(checkpoint_dir, "mappo_transformer_gat_actor.pt")
        torch.save({
            "transformer_state_dict": self.transformer_encoder.state_dict(),
            "gat_state_dict": self.gat.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim,
            "gat_hidden_dim": self.gat_hidden_dim,
            "gat_num_heads": self.gat_num_heads
        }, actor_path)

        critic_path = os.path.join(checkpoint_dir, "mappo_transformer_gat_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "input_dim": self.critic_input_dim
        }, critic_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load Phase 6 MAPPO+Transformer+GAT checkpoints."""
        actor_path = os.path.join(checkpoint_dir, "mappo_transformer_gat_actor.pt")
        if os.path.exists(actor_path):
            ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
            self.transformer_encoder.load_state_dict(ckpt["transformer_state_dict"])
            self.gat.load_state_dict(ckpt["gat_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])

        critic_path = os.path.join(checkpoint_dir, "mappo_transformer_gat_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
            self.critic.load_state_dict(ckpt["critic_state_dict"])
