# Phase 4 Implementation Plan: MAPPO (Centralized Training, Decentralized Execution)

Phase 4 implements a genuine **Multi-Agent PPO (MAPPO)** framework operating on the principle of Centralized Training with Decentralized Execution (CTDE), strictly distinct from the Phase 3 Independent PPO baseline.

---

## User Review Required

> [!IMPORTANT]
> **Key Architecture Decisions (CTDE)**:
> 1. **Centralized Critic (Training Time Only)**:
>    - Sees global state $S_t \in \mathbb{R}^{172}$, formed by concatenating all 4 agents' local observations: $[o_A, o_B, o_C, o_D]$ ($43 \times 4 = 172$ dimensions).
>    - Evaluates the global value function $V_i(S_t)$, eliminating the non-stationarity of independent critics.
> 2. **Decentralized Actors (Execution Time)**:
>    - Each agent selects actions conditioned solely on its own local observation ($o_i \in \mathbb{R}^{43}$).
>    - Zero inter-agent communication or global state access during rollout/inference.
> 3. **Shared vs. Individual Policy Option (Configurable in `config.yaml`)**:
>    - `share_policy: true` (Recommended Default): A single actor network parameterizes all intersections, with a 4-dimensional one-hot agent-ID embedding concatenated to local observation ($43 + 4 = 47$ dimensions). This enables transfer of signal coordination dynamics across intersections.
>    - `share_policy: false`: Four completely separate actor networks, each receiving purely its 43-dim local observation.
>    - Switchable cleanly via `config.yaml` without changing code.

---

## Proposed Changes

```
c:/Users/kashi/OneDrive/Desktop/major/
├── config.yaml                       # [MODIFY] Added mappo configuration section
├── models/
│   ├── mappo_network.py              # [NEW] CentralizedCritic, DecentralizedActor, MAPPORolloutBuffer, MAPPOController
│   └── checkpoints/                  # [NEW] Saved MAPPO weights (mappo_actor_*.pt, mappo_critic.pt)
├── experiments/
│   ├── mappo_training_log.csv        # [NEW] Real 40-episode MAPPO training logs
│   ├── mappo_reward_curve.png        # [NEW] Real MAPPO reward plot
│   └── controller_comparison.png     # [NEW] 3-way comparison bar chart (Fixed-Time vs IPPO vs MAPPO)
├── train_mappo.py                    # [NEW] 40-episode MAPPO training script
└── compare_all_controllers.py        # [NEW] Rigorous 3-way evaluation across 10 identical seeds
```

---

### Component Details

#### 1. Configuration: [MODIFY] [`config.yaml`](file:///c:/Users/kashi/OneDrive/Desktop/major/config.yaml)
Add `mappo:` configuration block per Rule 4:
```yaml
mappo:
  share_policy: true                  # true: parameter sharing with agent-ID embedding; false: individual actors
  agent_id_emb_dim: 4                 # 4-dim one-hot vector identifying intersection (A, B, C, D)
  hidden_dims_actor: [64, 64]         # MLP layers for decentralized actor
  hidden_dims_critic: [128, 64]       # Wider MLP layers for 172-dim centralized critic
  lr_actor: 0.0003                    # Learning rate for actor
  lr_critic: 0.0005                   # Learning rate for centralized critic
  gamma: 0.99                         # Discount factor
  gae_lambda: 0.95                    # GAE lambda
  clip_ratio: 0.2                     # PPO clipping epsilon
  value_coef: 0.5                     # Value loss weight
  entropy_coef: 0.01                  # Entropy bonus coefficient
  max_grad_norm: 0.5                  # Gradient norm clipping
  epochs: 4                           # PPO update epochs per rollout
  minibatch_size: 32                  # Minibatch size
  train_episodes: 40                  # Training budget matching Phase 3
  steps_per_episode: 100              # Steps per episode matching Phase 3
```

#### 2. Models: [NEW] [`models/mappo_network.py`](file:///c:/Users/kashi/OneDrive/Desktop/major/models/mappo_network.py)
- **`DecentralizedActorNetwork`**:
  - Input: local observation $\mathbb{R}^{43}$ (or $\mathbb{R}^{47}$ when agent-id embedding is active).
  - Architecture: Linear $\to$ Tanh $\to$ Linear $\to$ Tanh $\to$ Categorical logits over 4 discrete actions.
- **`CentralizedCriticNetwork`**:
  - Input: global state $S \in \mathbb{R}^{172}$ (+ agent ID vector $\mathbb{R}^{4}$).
  - Architecture: Linear(176, 128) $\to$ Tanh $\to$ Linear(128, 64) $\to$ Tanh $\to$ Linear(64, 1).
- **`MAPPORolloutBuffer`**:
  - Stores trajectories: global states $S_t$, per-agent local observations $o_{i, t}$, actions $a_{i, t}$, log-probs, values $V_{i, t}$, rewards $r_{i, t}$, and done flags.
  - Computes per-agent GAE advantages and discounted returns based on the centralized critic's estimates.
  - Normalizes advantages across minibatches.
- **`MAPPOController`**:
  - Manages optimization of the Centralized Critic and Decentralized Actor(s).
  - Handles checkpoint saving and loading (`models/checkpoints/mappo_*.pt`).

#### 3. Training Script: [NEW] [`train_mappo.py`](file:///c:/Users/kashi/OneDrive/Desktop/major/train_mappo.py)
- Executes 40 training episodes under the exact same simulation conditions as Phase 3 (seed sequence, episode duration, traffic demand).
- Logs episode returns, queue lengths, waiting times, policy loss, value loss, entropy to `experiments/mappo_training_log.csv`.
- Saves checkpoints upon completion.
- Generates real reward plot: `experiments/mappo_reward_curve.png`.

#### 4. 3-Way Comparative Evaluation: [NEW] [`compare_all_controllers.py`](file:///c:/Users/kashi/OneDrive/Desktop/major/compare_all_controllers.py)
- Evaluates:
  1. **Fixed-Time Controller** (deterministic pre-timed baseline)
  2. **Independent PPO** (Phase 3 trained checkpoints)
  3. **MAPPO** (Phase 4 trained checkpoints)
- Conducted over the exact same 10 evaluation seeds (`1001` through `1010`).
- Produces a comprehensive head-to-head comparison table:
  - Average Waiting Time / Delay (s)
  - Average Queue Length (veh)
  - Total Vehicle Throughput (completed veh)
  - Aggregate and Per-Agent Episode Returns
- Emits an honest verdict: whether MAPPO outperforms Independent PPO and Fixed-Time, explaining the cooperative dynamics observed.

---

## Verification Plan

### Automated Steps
1. **Unit Test**: Verify centralized critic (172/176 input $\to$ 1 output), decentralized actor (43/47 input $\to$ 4 output), GAE calculation, and parameter updates.
2. **Full MAPPO Training Run**:
   ```bash
   python train_mappo.py
   ```
   - Verify all 40 episodes complete cleanly.
   - Verify `experiments/mappo_training_log.csv` and `experiments/mappo_reward_curve.png` are produced from real data.
3. **3-Way Comparative Evaluation**:
   ```bash
   python compare_all_controllers.py
   ```
   - Run all 3 controllers across 10 evaluation seeds.
   - Print full comparison table and honest verdict.
4. **Phase 4 DONE Report**:
   - Provide complete results, plot embeds, and analysis.
