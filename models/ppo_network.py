"""
Independent PPO (IPPO) Implementation for SAGE-Traffic.
Four completely separate PPO agents (one per intersection), strictly independent
with no parameter sharing, no centralized critic, and no inter-agent communication.
"""

import os
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


class ActorNetwork(nn.Module):
    """
    Independent Actor network: 43 -> 64 -> 64 -> 4 discrete action logits with Tanh activations.
    """

    def __init__(self, obs_dim: int = 43, act_dim: int = 4, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, act_dim), std=0.01))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.network(x)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """
    Independent Critic network: 43 -> 64 -> 64 -> 1 state value with Tanh activations.
    """

    def __init__(self, obs_dim: int = 43, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOBuffer:
    """
    Rollout buffer for a single independent PPO agent.
    Stores transitions, computes Generalized Advantage Estimation (GAE),
    and yields shuffled minibatches for policy optimization.
    """

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and discounted returns."""
        num_steps = len(self.rewards)
        self.advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_generator(self, minibatch_size: int) -> Generator:
        """Yield shuffled minibatches of tensors."""
        num_samples = len(self.states)
        indices = np.random.permutation(num_samples)

        states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.long)
        log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        advantages_t = torch.tensor(self.advantages, dtype=torch.float32)
        returns_t = torch.tensor(self.returns, dtype=torch.float32)

        # Normalize advantages across entire rollout
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std()
        advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)

        for start_idx in range(0, num_samples, minibatch_size):
            batch_indices = indices[start_idx : start_idx + minibatch_size]
            yield (
                states_t[batch_indices],
                actions_t[batch_indices],
                log_probs_t[batch_indices],
                advantages_t[batch_indices],
                returns_t[batch_indices]
            )

    def clear(self):
        """Reset storage buffers."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.states)


class IndependentPPOAgent:
    """
    Completely separate Independent PPO learner for a single intersection.
    Contains its own Actor, Critic, Optimizer, and Rollout Buffer.
    """

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 43,
        act_dim: int = 4,
        ppo_cfg: Optional[dict] = None
    ):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = ppo_cfg or {}

        # Hyperparameters (strictly from config)
        self.lr = float(self.cfg.get("lr", 0.0003))
        self.gamma = float(self.cfg.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.cfg.get("clip_ratio", 0.2))
        self.value_coef = float(self.cfg.get("value_coef", 0.5))
        self.entropy_coef = float(self.cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.cfg.get("max_grad_norm", 0.5))
        self.epochs = int(self.cfg.get("epochs", 4))
        self.minibatch_size = int(self.cfg.get("minibatch_size", 32))
        hidden_dims = self.cfg.get("hidden_dims", [64, 64])

        # Separate actor and critic networks
        self.actor = ActorNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims)
        self.critic = CriticNetwork(obs_dim=obs_dim, hidden_dims=hidden_dims)

        # Single optimizer for this agent's own parameters
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
            eps=1e-5
        )

        self.buffer = PPOBuffer()

    def get_action_and_value(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Given a local 43-dim state observation, returns (action, log_prob, value).
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = self.actor(obs_t)
            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(obs_t).squeeze(-1)

        return action.item(), log_prob.item(), value.item()

    def get_value(self, obs: np.ndarray) -> float:
        """Compute state value estimate."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            value = self.critic(obs_t).squeeze(-1)
        return value.item()

    def update(self, last_value: float) -> Dict[str, float]:
        """
        Run PPO clipped surrogate policy optimization and value function update.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # 1. Compute GAE and returns
        self.buffer.compute_gae(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []

        # 2. PPO Epochs over Minibatches
        for _ in range(self.epochs):
            for b_states, b_actions, b_old_log_probs, b_advantages, b_returns in self.buffer.get_generator(self.minibatch_size):
                # Actor forward
                dist = self.actor(b_states)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # Critic forward
                new_values = self.critic(b_states).squeeze(-1)

                # Ratio r_t(theta)
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backprop and gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        # Clear buffer after update
        self.buffer.clear()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls))
        }

    def save_checkpoint(self, filepath: str):
        """Save network weights and optimizer state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "agent_id": self.agent_id,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, filepath)

    def load_checkpoint(self, filepath: str):
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
