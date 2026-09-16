"""
Phase 5: MAPPO with Temporal Transformer Encoder for SAGE-Traffic.
Implements:
1. TemporalTransformerEncoder: PyTorch Transformer processing [B, k, 43] -> [B, 64].
2. DecentralizedTemporalActor: Decentralized policy taking 64-D temporal embedding + 4-D agent ID = 68-D.
3. CentralizedTemporalCritic: Centralized value network taking 4 * 64-D = 256-D temporal global state.
4. TemporalHistoryManager: Rolling history buffer maintaining exactly k=4 observations, repeating initial obs at t=0.
5. MAPPOTemporalRolloutBuffer: Multi-agent CTDE buffer supporting end-to-end backpropagation.
6. MAPPOTemporalController: Controller orchestrating decentralized execution and centralized training.
"""

import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Dict, List, Tuple, Optional, Generator


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class TemporalTransformerEncoder(nn.Module):
    """
    Trainable PyTorch Transformer Encoder for temporal traffic observation history.
    Processes [Batch, history_length, 43] -> [Batch, embedding_dim].
    
    Architecture:
    1. Linear projection: 43 -> embedding_dim (64)
    2. Learnable positional encoding: Parameter(1, history_length, embedding_dim)
    3. nn.TransformerEncoder with num_layers (2) and num_heads (4)
    4. Sequence reduction: extracts final timestep representation [:, -1, :] -> [Batch, embedding_dim]
    """

    def __init__(
        self,
        obs_dim: int = 43,
        history_length: int = 4,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.embedding_dim = embedding_dim

        # 1. Feature projection per timestep
        self.input_proj = layer_init(nn.Linear(obs_dim, embedding_dim))

        # 2. Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, history_length, embedding_dim) * 0.02)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape [Batch, history_length, 43]
        Output: temporal embedding of shape [Batch, embedding_dim] (64)
        """
        # Linear projection: [B, k, 43] -> [B, k, 64]
        projected = self.input_proj(x)
        # Add positional encoding
        tokens = projected + self.pos_embedding
        # Pass through Transformer encoder
        encoded = self.transformer(tokens)
        # Reduction: select final timestep representation (current timestep context)
        current_step_embedding = encoded[:, -1, :]
        return current_step_embedding


class DecentralizedTemporalActor(nn.Module):
    """
    Decentralized Temporal Actor Network:
    Input: 64-D temporal embedding + 4-D agent ID one-hot = 68-D.
    Output: Categorical distribution over 4 discrete actions.
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

    def forward(self, temporal_embedding: torch.Tensor, agent_id_one_hot: torch.Tensor) -> Categorical:
        """
        temporal_embedding: [B, 64]
        agent_id_one_hot:   [B, 4]
        """
        x = torch.cat([temporal_embedding, agent_id_one_hot], dim=-1)
        logits = self.network(x)
        return Categorical(logits=logits)

    def forward_flat(self, x: torch.Tensor) -> Categorical:
        """Direct evaluation on 68-D tensor."""
        logits = self.network(x)
        return Categorical(logits=logits)


class CentralizedTemporalCritic(nn.Module):
    """
    Centralized Temporal Critic Network:
    Input: concatenation of all four agents' 64-D temporal embeddings = 4 * 64 = 256-D.
    Output: scalar state-value estimate V(S_temporal).
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

    def forward(self, centralized_temporal_state: torch.Tensor) -> torch.Tensor:
        """Input: [B, 256] -> Output: [B, 1]"""
        return self.network(centralized_temporal_state)


class TemporalHistoryManager:
    """
    Manages sliding observation history window of exactly k steps for all agents.
    At episode start (t=0), repeats initial observation k times: [o0, o0, o0, o0].
    At each step t, maintains sliding window: [o(t-k+1), ..., o(t)].
    Never uses future observations from t+1 or later.
    """

    def __init__(self, agents: List[str], history_length: int = 4, obs_dim: int = 43):
        self.agents = agents
        self.k = history_length
        self.obs_dim = obs_dim
        self.history: Dict[str, List[np.ndarray]] = {a: [] for a in agents}

    def reset(self, initial_obs_dict: Dict[str, np.ndarray]):
        """Populate history buffer by repeating the initial observation k times."""
        for a in self.agents:
            o0 = initial_obs_dict[a].astype(np.float32)
            self.history[a] = [o0.copy() for _ in range(self.k)]

    def update(self, next_obs_dict: Dict[str, np.ndarray]):
        """Slide window: append newly observed state and discard oldest."""
        for a in self.agents:
            ot = next_obs_dict[a].astype(np.float32)
            self.history[a].append(ot)
            if len(self.history[a]) > self.k:
                self.history[a].pop(0)

    def get_agent_history(self, agent: str) -> np.ndarray:
        """Return [k, 43] numpy array for a specific agent."""
        return np.array(self.history[agent], dtype=np.float32)

    def get_all_histories(self) -> Dict[str, np.ndarray]:
        """Return dict mapping each agent to its [k, 43] history array."""
        return {a: self.get_agent_history(a) for a in self.agents}


class MAPPOTemporalRolloutBuffer:
    """
    Multi-Agent Rollout Buffer for MAPPO with Temporal Transformer.
    Stores historical observation windows, centralized values, actions, rewards, and dones.
    Computes GAE on centralized values and generates minibatches for end-to-end training.
    """

    def __init__(self, agents: List[str], history_length: int = 4, obs_dim: int = 43):
        self.agents = agents
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
        minibatch_size: int,
        agent_one_hots: Dict[str, np.ndarray]
    ) -> Generator:
        """
        Yield minibatches of per-agent history, agent ID, action, old log_prob, and advantage
        for end-to-end training of Transformer + Actor.
        """
        num_steps = len(self.values)
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        all_histories = []
        all_agent_ids = []
        all_actions = []
        all_old_lps = []
        all_advs = []

        for a in self.agents:
            one_hot = agent_one_hots[a]
            for t in range(num_steps):
                all_histories.append(self.histories[a][t])
                all_agent_ids.append(one_hot)
                all_actions.append(self.actions[a][t])
                all_old_lps.append(self.log_probs[a][t])
                all_advs.append(adv_norm[t])

        total_samples = len(all_histories)
        indices = np.random.permutation(total_samples)

        hist_t = torch.tensor(np.array(all_histories), dtype=torch.float32)
        ids_t = torch.tensor(np.array(all_agent_ids), dtype=torch.float32)
        actions_t = torch.tensor(np.array(all_actions), dtype=torch.long)
        old_lp_t = torch.tensor(np.array(all_old_lps), dtype=torch.float32)
        adv_t = torch.tensor(np.array(all_advs), dtype=torch.float32)

        for start_idx in range(0, total_samples, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield hist_t[b_idx], ids_t[b_idx], actions_t[b_idx], old_lp_t[b_idx], adv_t[b_idx]

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


class MAPPOTemporalController:
    """
    Centralized Training, Decentralized Execution (CTDE) Controller with Temporal Transformer.
    
    Components:
    - TemporalTransformerEncoder: [B, 4, 43] -> [B, 64]
    - DecentralizedTemporalActor: 64-D embedding + 4-D agent ID = 68-D -> 4 action logits
    - CentralizedTemporalCritic: 4 * 64-D = 256-D -> 1 scalar value V(S_temporal)
    - TemporalHistoryManager: rolling window of exactly 4 observations
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.cfg = config.get("transformer", {})
        self.agents = config["agent"]["agents"]
        self.num_agents = len(self.agents)

        # Transformer & Network Dimensions
        self.obs_dim = 43
        self.history_length = int(self.cfg.get("history_length", 4))
        self.embedding_dim = int(self.cfg.get("embedding_dim", 64))
        self.num_heads = int(self.cfg.get("num_heads", 4))
        self.num_layers = int(self.cfg.get("num_layers", 2))
        self.dim_feedforward = int(self.cfg.get("dim_feedforward", 128))
        self.dropout = float(self.cfg.get("dropout", 0.0))

        self.agent_id_dim = 4
        self.act_dim = 4
        self.actor_input_dim = self.embedding_dim + self.agent_id_dim  # 68
        self.critic_input_dim = self.num_agents * self.embedding_dim   # 256

        # Hyperparameters
        self.lr_actor = float(self.cfg.get("lr_actor", 0.0003))
        self.lr_critic = float(self.cfg.get("lr_critic", 0.0005))
        self.gamma = float(self.cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.cfg.get("clip_ratio", 0.2))
        self.value_coef = float(self.cfg.get("value_coef", 0.5))
        self.entropy_coef = float(self.cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.cfg.get("max_grad_norm", 0.5))
        self.epochs = int(self.cfg.get("epochs", 4))
        self.minibatch_size = int(self.cfg.get("minibatch_size", 32))

        # Distinct One-Hot Agent IDs
        self.agent_one_hots = {
            a: np.eye(self.num_agents, dtype=np.float32)[i]
            for i, a in enumerate(self.agents)
        }

        # 1. Trainable PyTorch Temporal Transformer Encoder
        self.transformer_encoder = TemporalTransformerEncoder(
            obs_dim=self.obs_dim,
            history_length=self.history_length,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        # 2. Decentralized Temporal Actor (68 -> 64 -> 64 -> 4)
        self.actor = DecentralizedTemporalActor(
            embedding_dim=self.embedding_dim,
            agent_id_dim=self.agent_id_dim,
            act_dim=self.act_dim,
            hidden_dims=[64, 64]
        )

        # 3. Centralized Temporal Critic (256 -> 128 -> 64 -> 1)
        self.critic = CentralizedTemporalCritic(
            num_agents=self.num_agents,
            embedding_dim=self.embedding_dim,
            hidden_dims=[128, 64]
        )

        # Optimizers (End-to-End Training)
        self.actor_optimizer = optim.Adam(
            list(self.transformer_encoder.parameters()) + list(self.actor.parameters()),
            lr=self.lr_actor,
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr_critic,
            eps=1e-5
        )

        # 4. Rolling History Manager & Rollout Buffer
        self.history_manager = TemporalHistoryManager(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )
        self.buffer = MAPPOTemporalRolloutBuffer(
            agents=self.agents,
            history_length=self.history_length,
            obs_dim=self.obs_dim
        )

    def reset_history(self, initial_obs_dict: Dict[str, np.ndarray]):
        """Initialize history at start of episode by repeating initial observation k times."""
        self.history_manager.reset(initial_obs_dict)

    def update_history(self, next_obs_dict: Dict[str, np.ndarray]):
        """Update sliding window with newly received observations."""
        self.history_manager.update(next_obs_dict)

    def get_actions(
        self,
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Decentralized Action Selection:
        Each agent processes ONLY its own [k, 43] observation history + its own 4-D agent ID.
        Centralized critic is NEVER used.
        """
        actions = {}
        log_probs = {}

        self.transformer_encoder.eval()
        self.actor.eval()

        with torch.no_grad():
            for a in self.agents:
                # Local history only: [k, 43] -> [1, k, 43]
                hist = self.history_manager.get_agent_history(a)
                hist_t = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)

                # Transformer encodes local history to 64-D
                temporal_emb = self.transformer_encoder(hist_t)

                # Actor receives 64-D embedding + 4-D agent ID = 68-D
                agent_id_t = torch.tensor(self.agent_one_hots[a], dtype=torch.float32).unsqueeze(0)
                dist = self.actor(temporal_emb, agent_id_t)

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
        Encodes all four agents' local histories, concatenates to 256-D, evaluates V(S_temporal).
        """
        self.transformer_encoder.eval()
        self.critic.eval()

        with torch.no_grad():
            all_embs = []
            for a in self.agents:
                hist = self.history_manager.get_agent_history(a)
                hist_t = torch.tensor(hist, dtype=torch.float32).unsqueeze(0)
                emb = self.transformer_encoder(hist_t)
                all_embs.append(emb)

            # Concatenate 4 * 64 = 256-D
            centralized_state = torch.cat(all_embs, dim=-1)
            val = self.critic(centralized_state).squeeze()
            return float(val.item())

    def update(self, last_value: float) -> Dict[str, float]:
        """
        PPO update for Transformer Encoder, Decentralized Actor, and Centralized Critic.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        self.transformer_encoder.train()
        self.actor.train()
        self.critic.train()

        critic_losses = []
        actor_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.epochs):
            # 1. Update Centralized Critic on 256-D centralized temporal representations
            for hist_batch, returns in self.buffer.get_critic_minibatch_generator(self.minibatch_size):
                # hist_batch: [B, 4, k, 43]
                B = hist_batch.shape[0]
                with torch.no_grad():
                    embs = []
                    for i in range(self.num_agents):
                        agent_h = hist_batch[:, i, :, :]  # [B, k, 43]
                        e = self.transformer_encoder(agent_h)  # [B, 64]
                        embs.append(e)
                    centralized_input = torch.cat(embs, dim=-1)  # [B, 256]

                vals = self.critic(centralized_input)
                v_loss = 0.5 * ((vals - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

            # 2. Update Transformer Encoder + Decentralized Actor (End-to-End)
            for hists, ids, acts, old_lps, advs in self.buffer.get_actor_minibatch_generator(
                self.minibatch_size, self.agent_one_hots
            ):
                # End-to-end forward pass through Transformer encoder
                embs = self.transformer_encoder(hists)  # [B, 64]
                dist = self.actor(embs, ids)            # [B, 68] -> Categorical(4)

                new_lps = dist.log_prob(acts)
                entropy = dist.entropy().mean()

                log_ratio = new_lps - old_lps
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
                pol_loss = -torch.min(surr1, surr2).mean()

                loss = pol_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                loss.backward()
                # Clip gradients for both Transformer encoder and Actor
                nn.utils.clip_grad_norm_(
                    list(self.transformer_encoder.parameters()) + list(self.actor.parameters()),
                    self.max_grad_norm
                )
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
        """Save Phase 5 MAPPO+Transformer checkpoints."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save Actor + Transformer
        actor_path = os.path.join(checkpoint_dir, "mappo_transformer_actor.pt")
        torch.save({
            "transformer_state_dict": self.transformer_encoder.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "history_length": self.history_length,
            "embedding_dim": self.embedding_dim
        }, actor_path)

        # Save Centralized Critic
        critic_path = os.path.join(checkpoint_dir, "mappo_transformer_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "input_dim": self.critic_input_dim
        }, critic_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load Phase 5 MAPPO+Transformer checkpoints."""
        actor_path = os.path.join(checkpoint_dir, "mappo_transformer_actor.pt")
        if os.path.exists(actor_path):
            ckpt = torch.load(actor_path, map_location="cpu")
            self.transformer_encoder.load_state_dict(ckpt["transformer_state_dict"])
            self.actor.load_state_dict(ckpt["actor_state_dict"])

        critic_path = os.path.join(checkpoint_dir, "mappo_transformer_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu")
            self.critic.load_state_dict(ckpt["critic_state_dict"])
