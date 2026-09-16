"""
Multi-Agent PPO (MAPPO) Architecture for SAGE-Traffic (Phase 4).
Centralized Training with Decentralized Execution (CTDE):
- Centralized Critic: conditioned on 172-dim global state [o_A, o_B, o_C, o_D] -> 128 -> 64 -> 1.
  Represents V(S_t), the global centralized state-value estimate.
- Decentralized Actor(s): conditioned strictly on local observations at execution time.
  When share_policy=True: shared actor with 43 local obs + 4 one-hot agent ID = 47 dims -> 64 -> 64 -> 4.
  When share_policy=False: four separate actors with 43 local obs -> 64 -> 64 -> 4.
- Actors cannot access global state at execution time.
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


class DecentralizedActorNetwork(nn.Module):
    """
    Decentralized Actor network:
    Input: local observation (43 dims) [+ optional 4-dim agent-ID one-hot = 47 dims when shared]
    Output: Categorical distribution over 4 discrete actions (keep, switch, extend, reduce).
    """

    def __init__(self, obs_dim: int = 47, act_dim: int = 4, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
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


class CentralizedCriticNetwork(nn.Module):
    """
    Centralized Critic network:
    Input: global state S_t (172 dims = 4 * 43 concatenated local observations).
           Consumes strictly the 172-dim global state. No agent ID appended.
    Output: scalar global state-value estimate V(S_t) (1 dim).
    """

    def __init__(self, global_dim: int = 172, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.global_dim = global_dim
        layers = []
        in_dim = global_dim
        for h in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h)))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(layer_init(nn.Linear(in_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MAPPORolloutBuffer:
    """
    Multi-Agent Rollout Buffer for CTDE.
    Stores:
    - global_states: list of S_t in R^172
    - values: list of V(S_t) in R (centralized value estimates)
    - global_rewards: list of r_global,t = sum_{a} r_{a,t} in R
    - dones: list of bool
    - local_obs: dict of lists of o_{a,t} in R^43
    - actions: dict of lists of a_{a,t} in {0, 1, 2, 3}
    - log_probs: dict of lists of log pi(a_{a,t})
    - rewards: dict of lists of per-agent rewards r_{a,t}

    Computes GAE advantages based on the single centralized value critic V(S_t)
    and global team reward r_global,t = sum_{a} r_{a,t}.
    """

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.global_states: List[np.ndarray] = []
        self.values: List[float] = []
        self.global_rewards: List[float] = []
        self.dones: List[bool] = []

        self.local_obs: Dict[str, List[np.ndarray]] = {a: [] for a in agents}
        self.actions: Dict[str, List[int]] = {a: [] for a in agents}
        self.log_probs: Dict[str, List[float]] = {a: [] for a in agents}
        self.rewards: Dict[str, List[float]] = {a: [] for a in agents}

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        global_state: np.ndarray,
        value: float,
        local_obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        log_probs: Dict[str, float],
        rewards: Dict[str, float],
        done: bool
    ):
        """Add one environment decision step to the rollout buffer."""
        self.global_states.append(global_state)
        self.values.append(value)
        # Global reward is aggregate sum across all intersections
        global_rew = float(sum(rewards.values()))
        self.global_rewards.append(global_rew)
        self.dones.append(done)

        for a in self.agents:
            self.local_obs[a].append(local_obs[a])
            self.actions[a].append(actions[a])
            self.log_probs[a].append(log_probs[a])
            self.rewards[a].append(rewards[a])

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Compute GAE advantages and discounted returns using the single centralized value critic V(S_t).
        delta_t = r_global,t + gamma * V(S_{t+1}) * (1 - done) - V(S_t)
        A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
        R_t = A_t + V(S_t)
        """
        num_steps = len(self.global_states)
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
        Yield minibatches of (global_state, return) for Centralized Critic training.
        Each sample: global_state in R^172, target_return in R^1.
        No agent ID appended to critic input.
        """
        num_steps = len(self.global_states)
        indices = np.random.permutation(num_steps)

        g_states_t = torch.tensor(np.array(self.global_states), dtype=torch.float32)
        returns_t = torch.tensor(self.returns, dtype=torch.float32).unsqueeze(-1)

        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield g_states_t[b_idx], returns_t[b_idx]

    def get_shared_actor_minibatch_generator(
        self,
        minibatch_size: int,
        agent_one_hots: Dict[str, np.ndarray]
    ) -> Generator:
        """
        Yield minibatches for the Shared Decentralized Actor.
        Input per sample: local_obs (43) + agent_one_hot (4) = 47 dims.
        Advantage: centralized advantage normalized across the batch.
        Total samples: num_steps * num_agents.
        """
        num_steps = len(self.global_states)
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        all_inputs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []

        for a in self.agents:
            one_hot = agent_one_hots[a]
            for t in range(num_steps):
                # Actor receives strictly local obs + agent id
                inp = np.concatenate([self.local_obs[a][t], one_hot])
                all_inputs.append(inp)
                all_actions.append(self.actions[a][t])
                all_log_probs.append(self.log_probs[a][t])
                all_advantages.append(adv_norm[t])

        total_samples = len(all_inputs)
        indices = np.random.permutation(total_samples)

        inputs_t = torch.tensor(np.array(all_inputs), dtype=torch.float32)
        actions_t = torch.tensor(np.array(all_actions), dtype=torch.long)
        old_lp_t = torch.tensor(np.array(all_log_probs), dtype=torch.float32)
        adv_t = torch.tensor(np.array(all_advantages), dtype=torch.float32)

        for start_idx in range(0, total_samples, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield inputs_t[b_idx], actions_t[b_idx], old_lp_t[b_idx], adv_t[b_idx]

    def get_individual_actor_minibatch_generator(
        self,
        agent: str,
        minibatch_size: int
    ) -> Generator:
        """
        Yield minibatches for an Individual Decentralized Actor.
        Input per sample: local_obs (43 dims).
        Advantage: centralized advantage normalized across the batch.
        Total samples: num_steps.
        """
        num_steps = len(self.global_states)
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        inputs_t = torch.tensor(np.array(self.local_obs[agent]), dtype=torch.float32)
        actions_t = torch.tensor(np.array(self.actions[agent]), dtype=torch.long)
        old_lp_t = torch.tensor(np.array(self.log_probs[agent]), dtype=torch.float32)
        adv_t = torch.tensor(adv_norm, dtype=torch.float32)

        indices = np.random.permutation(num_steps)
        for start_idx in range(0, num_steps, minibatch_size):
            b_idx = indices[start_idx : start_idx + minibatch_size]
            yield inputs_t[b_idx], actions_t[b_idx], old_lp_t[b_idx], adv_t[b_idx]

    def clear(self):
        """Reset storage buffers."""
        self.global_states.clear()
        self.values.clear()
        self.global_rewards.clear()
        self.dones.clear()
        for a in self.agents:
            self.local_obs[a].clear()
            self.actions[a].clear()
            self.log_probs[a].clear()
            self.rewards[a].clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.global_states)


class MAPPOController:
    """
    Centralized Training, Decentralized Execution (CTDE) Controller for MAPPO.
    Manages:
    1. One Centralized Critic: 172 -> 128 -> 64 -> 1. Consumes global state S_t.
    2. Decentralized Actor(s):
       - If share_policy=True: shared actor (47 -> 64 -> 64 -> 4) with 43 local obs + 4 one-hot agent ID.
       - If share_policy=False: four separate actors (43 -> 64 -> 64 -> 4).
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.cfg = config.get("mappo", {})
        self.agents = config["agent"]["agents"]
        self.num_agents = len(self.agents)

        # Policy sharing switch
        self.share_policy = bool(self.cfg.get("share_policy", True))
        self.agent_id_emb_dim = int(self.cfg.get("agent_id_emb_dim", 4))

        # Dimensions
        self.local_obs_dim = 43
        self.act_dim = 4
        # Concatenated global state [o_A, o_B, o_C, o_D] = 43 * 4 = 172
        self.global_state_dim = self.local_obs_dim * self.num_agents
        # Centralized critic receives strictly 172 dims
        self.critic_input_dim = self.global_state_dim

        if self.share_policy:
            # 43 local obs + 4 one-hot agent ID = 47
            self.actor_input_dim = self.local_obs_dim + self.agent_id_emb_dim
        else:
            # 43 local obs
            self.actor_input_dim = self.local_obs_dim

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

        hidden_actor = self.cfg.get("hidden_dims_actor", [64, 64])
        hidden_critic = self.cfg.get("hidden_dims_critic", [128, 64])

        # One-hot representation for agent IDs
        self.agent_one_hots = {
            a: np.eye(self.num_agents, dtype=np.float32)[i]
            for i, a in enumerate(self.agents)
        }

        # 1. One Centralized Critic (172 -> 128 -> 64 -> 1)
        self.critic = CentralizedCriticNetwork(
            global_dim=self.critic_input_dim,
            hidden_dims=hidden_critic
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

        # 2. Decentralized Actor(s)
        if self.share_policy:
            self.shared_actor = DecentralizedActorNetwork(
                obs_dim=self.actor_input_dim,
                act_dim=self.act_dim,
                hidden_dims=hidden_actor
            )
            self.actor_optimizer = optim.Adam(self.shared_actor.parameters(), lr=self.lr_actor, eps=1e-5)
            self.individual_actors = None
            self.actor_optimizers = None
        else:
            self.shared_actor = None
            self.actor_optimizer = None
            self.individual_actors = {
                a: DecentralizedActorNetwork(
                    obs_dim=self.actor_input_dim,
                    act_dim=self.act_dim,
                    hidden_dims=hidden_actor
                )
                for a in self.agents
            }
            self.actor_optimizers = {
                a: optim.Adam(self.individual_actors[a].parameters(), lr=self.lr_actor, eps=1e-5)
                for a in self.agents
            }

        # 3. Rollout Buffer
        self.buffer = MAPPORolloutBuffer(agents=self.agents)

    def get_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Decentralized Action Selection:
        Each agent selects action conditioned strictly on its local observation
        (with one-hot agent ID if shared).
        Actors receive ONLY local observation. No global state access.
        """
        actions = {}
        log_probs = {}

        with torch.no_grad():
            for a in self.agents:
                local_obs = obs_dict[a]
                if self.share_policy:
                    inp = np.concatenate([local_obs, self.agent_one_hots[a]])
                    inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                    dist = self.shared_actor(inp_t)
                else:
                    inp_t = torch.tensor(local_obs, dtype=torch.float32).unsqueeze(0)
                    dist = self.individual_actors[a](inp_t)

                if deterministic:
                    act = torch.argmax(dist.probs, dim=-1)
                else:
                    act = dist.sample()

                lp = dist.log_prob(act)
                actions[a] = act.item()
                log_probs[a] = lp.item()

        return actions, log_probs

    def get_value(self, global_state: np.ndarray) -> float:
        """
        Centralized Critic Value Estimation:
        Evaluates V(S_t) conditioned strictly on the 172-dim global state.
        No agent ID appended.
        """
        with torch.no_grad():
            inp_t = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)
            val = self.critic(inp_t).squeeze()
            return float(val.item())

    def construct_global_state(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate all agents' local observations into global state S_t in R^172."""
        return np.concatenate([obs_dict[a] for a in self.agents])

    def update(self, last_global_state: np.ndarray) -> Dict[str, float]:
        """
        PPO update for Centralized Critic and Decentralized Actor(s).
        Critic is updated on minibatches of (S_t, R_t).
        Actor(s) are updated on minibatches of (local_obs [+ id], action, old_lp, adv).
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # Terminal value for GAE
        last_val = self.get_value(last_global_state)
        self.buffer.compute_gae(last_val, self.gamma, self.gae_lambda)

        critic_losses = []
        actor_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.epochs):
            # 1. Update Centralized Critic on 172-dim global states
            for g_states, returns in self.buffer.get_critic_minibatch_generator(self.minibatch_size):
                vals = self.critic(g_states)
                v_loss = 0.5 * ((vals - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

            # 2. Update Decentralized Actor(s)
            if self.share_policy:
                for inps, acts, old_lps, advs in self.buffer.get_shared_actor_minibatch_generator(
                    self.minibatch_size, self.agent_one_hots
                ):
                    dist = self.shared_actor(inps)
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
                    nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)

                    actor_losses.append(pol_loss.item())
                    entropies.append(entropy.item())
            else:
                for a in self.agents:
                    for inps, acts, old_lps, advs in self.buffer.get_individual_actor_minibatch_generator(
                        a, self.minibatch_size
                    ):
                        actor = self.individual_actors[a]
                        opt = self.actor_optimizers[a]

                        dist = actor(inps)
                        new_lps = dist.log_prob(acts)
                        entropy = dist.entropy().mean()

                        log_ratio = new_lps - old_lps
                        ratio = torch.exp(log_ratio)

                        surr1 = ratio * advs
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
                        pol_loss = -torch.min(surr1, surr2).mean()

                        loss = pol_loss - self.entropy_coef * entropy

                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                        opt.step()

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
        """Save model checkpoints for centralized critic and decentralized actor(s)."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save Centralized Critic
        critic_path = os.path.join(checkpoint_dir, "mappo_critic.pt")
        torch.save({
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict()
        }, critic_path)

        # Save Decentralized Actor(s)
        if self.share_policy:
            actor_path = os.path.join(checkpoint_dir, "mappo_actor_shared.pt")
            torch.save({
                "actor_state_dict": self.shared_actor.state_dict(),
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
                "share_policy": True
            }, actor_path)
            # Also save per-agent copies for uniform evaluation referencing
            for a in self.agents:
                p = os.path.join(checkpoint_dir, f"mappo_actor_{a}.pt")
                torch.save({
                    "actor_state_dict": self.shared_actor.state_dict(),
                    "share_policy": True
                }, p)
        else:
            for a in self.agents:
                actor_path = os.path.join(checkpoint_dir, f"mappo_actor_{a}.pt")
                torch.save({
                    "actor_state_dict": self.individual_actors[a].state_dict(),
                    "optimizer_state_dict": self.actor_optimizers[a].state_dict(),
                    "share_policy": False
                }, actor_path)

    def load_checkpoints(self, checkpoint_dir: str = "models/checkpoints"):
        """Load trained MAPPO checkpoints."""
        critic_path = os.path.join(checkpoint_dir, "mappo_critic.pt")
        if os.path.exists(critic_path):
            ckpt = torch.load(critic_path, map_location="cpu")
            self.critic.load_state_dict(ckpt["critic_state_dict"])

        if self.share_policy:
            actor_path = os.path.join(checkpoint_dir, "mappo_actor_shared.pt")
            if not os.path.exists(actor_path):
                actor_path = os.path.join(checkpoint_dir, "mappo_actor_A.pt")
            ckpt = torch.load(actor_path, map_location="cpu")
            self.shared_actor.load_state_dict(ckpt["actor_state_dict"])
        else:
            for a in self.agents:
                actor_path = os.path.join(checkpoint_dir, f"mappo_actor_{a}.pt")
                ckpt = torch.load(actor_path, map_location="cpu")
                self.individual_actors[a].load_state_dict(ckpt["actor_state_dict"])
