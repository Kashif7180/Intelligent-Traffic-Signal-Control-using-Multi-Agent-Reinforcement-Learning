# PHASE 8 FORENSIC AUDIT: CAUSAL INFLUENCE INTEGRATION

**Investigation Target**: Determine why Phase 8 evaluation produced metrics exactly identical to Phase 7 across all 10 evaluation seeds (delay: $28.70 \pm 3.80$s, queue: $7.77 \pm 0.89$, throughput: $52.2 \pm 4.3$, return: $-888.83 \pm 109.50$, comm ratio: $100.0\%$).

---

## 1. Executive Verdict

### **PARTIALLY VERIFIED**

**Summary of Findings**:
1. **The Causal Module is Real and Live**: The `CausalInfluenceEstimator` is genuinely instantiated, executed at every simulation step, and generates continuous directional scores $C_{ij} \in [0, 1]$ (768 unique values on Seed 1001, ranging from $0.025$ to $0.775$).
2. **Checkpoints and Weights are Genuinely Distinct**: Phase 7 and Phase 8 checkpoints are not copies. 100% of trainable parameter tensors (27 in Transformer, 4 in GAT, 6 in Actor) differ numerically between Phase 7 and Phase 8.
3. **C_ij is Physically Connected to the Gate**: The Gate MLP receives input dimension 129 ($[h_i, h_j, C_{ij}]$) where $C_{ij}$ occupies the 129th feature index without being zeroed, detached, or replaced by Phase 7 heuristics.
4. **Why Decisions and Metrics Are Identical (Root Cause)**:
   - **Dimensional Imbalance in Gate MLP**: The 128 embedding features $[h_i, h_j]$ (each with weight $\sim 0.10$) aggregate to $\sim 12.8$ in pre-activation magnitude, dwarfing the single scalar $C_{ij}$ feature (weight $0.12$). Sweeping $C_{ij}$ across its entire theoretical range $[0.0 \to 1.0]$ only shifts the continuous gate probability by **$\Delta g = 0.0023$** (from $0.7465$ to $0.7488$).
   - **Hard Threshold Saturation**: Both bounds ($0.7465$ and $0.7488$) reside far above the hard evaluation threshold of $0.5$. Consequently, the hard communication decision is deterministically **$1.0$ (communicated)** for 100% of edge evaluations across all ablations ($C=0, C=1, \text{shuffled}, \text{actual}$).
   - **Greedy Argmax Action Collapse**: Even though Phase 8 continuous actor probabilities differ from Phase 7 by up to **$0.0568$**, both policies assign overwhelmingly highest probability to Action 1 ("switch" phase, $P \approx 0.83$ in P8 vs $P \approx 0.88$ in P7). Deterministic greedy action selection (`argmax`) selects Action 1 at all 100 steps for both models, driving identical deterministic SUMO trajectories.

---

## 2. Checkpoint Evidence

| Model Role | File Path | File Size (bytes) | SHA256 Hash | Distinct from Phase 7? |
| :--- | :--- | :---: | :---: | :---: |
| **Phase 7 Actor** | `models/checkpoints/mappo_comm_gate_actor.pt` | 1,248,927 | `4759869bca999d18035e27195d4621b3b49305dcedca1df38a743924621076f8` | Baseline |
| **Phase 7 Critic** | `models/checkpoints/mappo_comm_gate_critic.pt` | 503,005 | `0070b5f013eeff5a2f0ba75954a00b5cb262fd9b5dce24b24f565a65602aeb0b` | Baseline |
| **Phase 8 Actor** | `models/checkpoints/mappo_causal_gate_actor.pt` | 1,249,325 | `81c2e33fac002a24ac843187c296150afa918468ca3063893d35b6cb057075a0` | **YES (Distinct)** |
| **Phase 8 Critic** | `models/checkpoints/mappo_causal_gate_critic.pt` | 503,065 | `213a00da324511f277a4e6c523dfcf1c2da71931c5f836316dc4674528d6ec0e` | **YES (Distinct)** |

### Parameter Difference Analysis
State dictionary tensor-by-tensor comparison between Phase 7 and Phase 8 actor checkpoints:
- **Transformer Encoder**: 27 / 27 parameter tensors differ (max absolute difference: `0.2522`, mean difference: `0.0831`).
- **Gated GAT Layer**: 4 / 4 parameter tensors differ (max absolute difference: `0.4716`, mean difference: `0.0537`).
- **Decentralized Actor**: 6 / 6 parameter tensors differ (max absolute difference: `0.3541`, mean difference: `0.0482`).
- **Conclusion**: Phase 8 does NOT accidentally load Phase 7 weights.

---

## 3. Architecture Evidence

The tensor and data flow through Phase 8 inference was traced from environment to action output:

```
[TraCI Traffic State info_dict]
       │
       ▼
[CausalInfluenceEstimator.update_traffic_state()]
  - Tracks rolling history W=15 of queue, wait, demand, corridor queues
       │
       ▼
[CausalInfluenceEstimator.estimate_causal_influence()]
  - Computes Ridge Regression: Restricted RSS_R vs Unrestricted RSS_U
  - Produces directional C_ij in [0, 1] for 8 allowed edges
       │
       ▼ context_dict = {(src, tgt): C_ij}
[LearnableCommunicationGate.forward(node_features, context_dict)]
  - Concatenation: [h_tgt (64), h_src (64), C_ij (1)] -> 129-D input tensor
  - Gate MLP: Linear(129, 32) -> Tanh -> Linear(32, 1) -> Sigmoid -> g_ij in (0, 1)
  - Hard Threshold (eval): g_eff = (g_ij >= 0.5).float()
       │
       ▼ gate_matrix [B, 4, 4]
[GatedGraphAttentionLayer.forward()]
  - Modulates attention logits: gated_logits = logits + log(clamp(gate, 1e-8, 1.0))
  - Produces 64-D gated embeddings h'_i for each agent
       │
       ▼ Concatenate with 4-D agent ID = 68-D
[DecentralizedGATActor.forward()]
  - Linear(68, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 4) -> Categorical dist
       │
       ▼
[argmax(dist.probs) -> Discrete Action {0, 1, 2, 3}]
```

- **Exact Input Dimension**: 129 ($64 + 64 + 1$).
- **Gate Layer Structure**: `Sequential(Linear(129, 32), Tanh(), Linear(32, 1), Sigmoid())`.
- **Heuristic Context Check**: The old heuristic $(q_j - q_i)/40.0$ is completely absent from the Phase 8 execution pipeline; $C_{ij}$ is passed exclusively.

---

## 4. Live Causal Estimation (Seed 1001)

During a fresh 100-step evaluation run on Seed 1001:
- **Total Decision Steps**: 100
- **Total C_ij Evaluations**: 800 ($8 \text{ directed edges} \times 100 \text{ steps}$)
- **Unique C_ij Values**: **768**
- **C_ij Minimum**: `0.025000`
- **C_ij Maximum**: `0.774836`
- **C_ij Mean**: `0.584867`
- **C_ij Standard Deviation**: `0.128722`
- **Communication Topology**: Exactly 8 directed edges ($A \leftrightarrow B, A \leftrightarrow C, B \leftrightarrow D, C \leftrightarrow D$). Diagonals ($A \leftrightarrow D, B \leftrightarrow C$) and self-loops were verified strictly excluded from inter-agent communication.

---

## 5. C_ij Sensitivity Analysis (Fixed Evaluation State)

Under the fixed real evaluation state at Seed 1001, Decision Step 50 ($t=250$s), an inference-only sensitivity test was performed by altering $C_{ij}$ inputs to the trained Phase 8 model:

| C Condition | Mean Gate Prob $\bar{g}$ | Gate ($A \to B$) | Gate ($B \to A$) | Active Comms ($g \ge 0.5$) | Action A Probs $[P_0, P_1, P_2, P_3]$ | Selected Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Actual $C_{ij}$** ($0.719$) | `0.7485` | `0.7626` | `0.7515` | **8 / 8** | `[0.0250, 0.8313, 0.0464, 0.0973]` | **1 (switch)** |
| **$C_{ij} = 0.0$** | `0.7465` | `0.7611` | `0.7484` | **8 / 8** | `[0.0250, 0.8313, 0.0464, 0.0973]` | **1 (switch)** |
| **$C_{ij} = 1.0$** | `0.7488` | `0.7627` | `0.7520` | **8 / 8** | `[0.0250, 0.8313, 0.0464, 0.0973]` | **1 (switch)** |
| **Shuffled $C_{ij}$** | `0.7485` | `0.7626` | `0.7512` | **8 / 8** | `[0.0250, 0.8313, 0.0464, 0.0973]` | **1 (switch)** |

### Sensitivity Findings
- **Gate Probability Shift**: Varying $C_{ij}$ from $0.0$ to $1.0$ changes the gate probability by only $\Delta g = 0.0023$ ($0.7465 \to 0.7488$).
- **Threshold Invariance**: Because both $0.7465$ and $0.7488$ are well above the $0.5$ decision threshold, the hard evaluation communication decision is $1.0$ in all conditions.
- **Actor Probability Invariance**: Because the deterministic threshold passes full neighbor information (gate = $1.0$) in all four conditions, the GAT layer outputs the identical representations, resulting in identical actor probabilities.

---

## 6. Phase 7 vs. Phase 8 Step-by-Step Comparison (Seed 1001)

| Metric / Aspect | Phase 7 (Heuristic Gate) | Phase 8 (Causal Gate) | Step-by-Step Difference |
| :--- | :---: | :---: | :---: |
| **Mean Gate Probability** | `0.8932` | `0.7405` | Mean abs diff: **`0.1467`** (Max diff: `0.2456`) |
| **Hard Comm Decisions** | 800 / 800 (100%) | 800 / 800 (100%) | **100% Identical** (all $\ge 0.5$) |
| **Actor Probability Probs** | Mean $P_1 = 0.8741$ | Mean $P_1 = 0.8313$ | Mean abs diff: **`0.0428`** (Max diff: `0.0568`) |
| **Action Sequence Identity** | Action 1 at all steps | Action 1 at all steps | **400 / 400 (100.0%) Identical** |
| **Total Task Return** | `-1087.95` | `-1087.95` | **0.00%** difference |
| **Total Communication Cost** | `8.00` | `8.00` | **0.00%** difference |
| **Mean Delay (s)** | `36.70` | `36.70` | **0.00%** difference |
| **Mean Queue (veh)** | `8.75` | `8.75` | **0.00%** difference |
| **Completed Throughput** | `54` | `54` | **0.00%** difference |

### Key Insight
Phase 7 and Phase 8 produce **numerically different internal representations and continuous probabilities** ($\Delta g \approx 0.15, \Delta P \approx 0.043$), but the discrete thresholding step ($g \ge 0.5$) and greedy action operator (`argmax`) collapse both continuous distributions to the identical discrete execution sequence.

---

## 7. Fresh Process Reproduction (Seed 1001)

An independent fresh evaluation of the saved Phase 8 checkpoint was executed on Seed 1001:
- Average Delay: **`36.70` s**
- Average Queue: **`8.75` veh**
- Completed Throughput: **`54` veh**
- Task Return: **`-1087.95`**
- Communication Ratio: **`100.0%` (800 / 800)**
- Communication Cost: **`8.00`**
- Mean $C_{ij}$: **`0.5849`**
- Result: **100% Deterministic Reproduction** verified.

---

## 8. Inference-Time Ablation Without Retraining (Seed 1001)

| Ablation Condition | Mean Delay (s) | Mean Queue (veh) | Throughput (veh) | Task Return | Comm Ratio (%) | Actual Comms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 8 Normal $C_{ij}$** | 36.70 | 8.75 | 54 | -1087.95 | 100.0% | 800 / 800 |
| **Phase 8 with $C_{ij} = 0.0$** | 36.70 | 8.75 | 54 | -1087.95 | 100.0% | 800 / 800 |
| **Phase 8 with $C_{ij} = 1.0$** | 36.70 | 8.75 | 54 | -1087.95 | 100.0% | 800 / 800 |
| **Phase 8 with Shuffled $C_{ij}$** | 36.70 | 8.75 | 54 | -1087.95 | 100.0% | 800 / 800 |

---

## 9. Dead-Code & Override Investigation

A systematic inspection of the codebase confirmed:
- **No Hardcoded Actions**: No static action arrays or hardcoded decision mappings exist.
- **No Evaluation Shortcuts**: The simulator executes all 100 steps through native TraCI second-by-second stepping.
- **No Accidental Fallbacks**: The Phase 7 heuristic queue difference is completely uncalled in `MAPPOCausalGateController`.
- **No Caching**: $C_{ij}$ is computed freshly on every step from current and lagged traffic buffers.

---

## 10. Critical Causal Influence Estimation Limitations

1. **Estimation, Not True Discovery**: This implementation performs causal influence estimation (a predictive statistical approximation), NOT ground-truth causal discovery.
2. **No Ground-Truth Causal Labels**: The traffic environment provides observational state variables without counterfactual labels.
3. **Predictive Association $\neq$ Physical Causation**: Granger-style predictive variance reduction measures whether past source observations linearly reduce target prediction errors; it does not prove physical vehicle transfer or causality.
4. **Network Confounding**: Coordinated traffic light cycles, common boundary arrival rates, and shared arterial corridors induce spurious temporal correlations.
5. **Score Interpretation**: $C_{ij} \in [0, 1]$ represents normalized predictive gain and directional queue pressure; it must NOT be interpreted as a probability of causation.
6. **No Interventional Proof**: Phase 8 does not prove that transmitting a packet along $j \to i$ causes traffic clearance at $i$.

---

## 11. Final Forensic Verdict

### **PARTIALLY VERIFIED**

`Phase 8 maintained the Phase 7 task performance but did NOT reduce communication.`

**Detailed Explanation**:
The Phase 8 causal influence estimation module is genuinely implemented, actively executes at every step, and connects mathematically into the communication gate MLP. However, because the 128 embedding features dominate the single scalar $C_{ij}$ feature in pre-activation magnitude, $C_{ij}$ has very weak leverage over the gate output ($\Delta g \approx 0.002$), which is completely absorbed by the hard $0.5$ evaluation threshold. As a consequence, all communication channels remain open at evaluation, and the greedy policy reproduces the exact Phase 7 discrete switching schedule.

---

```
PHASE 8 FORENSIC AUDIT COMPLETE — STOP
```
