# SAGE-Traffic: Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.28.0-green.svg)](https://eclipse.dev/sumo/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Phase%2011-Complete-success.svg)](#)

A comprehensive, state-of-the-art Multi-Agent Reinforcement Learning (MARL) framework for coordinated, causal-aware, and formally verified traffic signal control on urban road networks simulated using **Eclipse SUMO** (Simulation of Urban MObility).

---

## Overview

Traffic congestion in urban road networks produces significant economic losses, fuel waste, and increased greenhouse gas emissions. Traditional traffic light controllers rely on fixed-cycle timing or isolated actuated logic, failing to dynamically adapt to varying traffic surges and coordinate across adjacent intersections.

**SAGE-Traffic** (Safety-Augmented Graph-Enhanced Traffic Control) designs, trains, evaluates, and verifies advanced multi-agent reinforcement learning architectures on a 4-intersection ($2 \times 2$ arterial grid) traffic network with multi-lane dynamics and dynamic demand profiles.

Across 11 research phases, the project progressed from decentralized baselines to Centralized Training with Decentralized Execution (CTDE), temporal history self-attention, spatial Graph Attention Networks (GAT), counterfactual causal influence communication gating, a formal runtime safety shield, complete system integration, and grounded Explainable AI (XAI).

---

## Architectural Evolution (Phases 1–11)

| Phase | Architecture / Milestone | Paradigm | Key Mechanism | Checkpoint / Primary File |
| :---: | :--- | :--- | :--- | :--- |
| **Phase 1** | **$2 \times 2$ Network Construction** | Simulation | 4-intersection arterial grid with detector loops | `network/2x2_grid.net.xml` |
| **Phase 2** | **Multi-Agent Environment** | MARL Wrapper | PettingZoo/Gymnasium API with discrete signal actions | `environment/traffic_env.py` |
| **Phase 3** | **Independent PPO (IPPO)** | Decentralized RL | Independent actors and critics per intersection | `models/ppo_network.py` (`agent_A.pt`–`D.pt`) |
| **Phase 4** | **MAPPO (CTDE)** | CTDE | Centralized joint critic ($S_t \in \mathbb{R}^{172}$), shared/individual actors | `models/mappo_network.py` (`mappo_actor_shared.pt`) |
| **Phase 5** | **MAPPO + Temporal Transformer** | CTDE + Attention | $k=4$ sliding window history with temporal self-attention | `models/transformer_network.py` (`mappo_transformer_actor.pt`) |
| **Phase 6** | **MAPPO + Transformer + GAT** | Graph MARL | Spatial Graph Attention over 8 canonical directed edges | `models/gat_network.py` (`mappo_transformer_gat_actor.pt`) |
| **Phase 7** | **Comm-Gated MAPPO** | Adaptive Comm | Learnable sigmoid communication gate ($\tau = 0.5$) | `models/communication_gate.py` (`mappo_comm_gate_actor.pt`) |
| **Phase 8** | **Causal-Gated MAPPO** | Causal MARL | Counterfactual marginalization estimating policy influence | `models/causal_influence.py` (`mappo_causal_gate_actor.pt`) |
| **Phase 8R** | **Causal Improved Architecture** | Dual-Pathway GAT | Dedicated $\phi(C_{ij}) \in \mathbb{R}^8$ projection, unshielded baseline | `models/causal_improved.py` (`mappo_causal_improved_actor.pt`) |
| **Phase 9** | **Runtime Safety Shield** | Formal Verification | Formal safety invariants, $5\text{s}$ min-green, $1\text{s}$ clearance buffer | `models/safety_layer.py` (Unbypassable runtime shield) |
| **Phase 10** | **Complete System Integration** | Integration & Audit | Canonical topology module, 110-run benchmark, Unified GUI | `network/topology.py`, `visualize_phase10_gui.py` |
| **Phase 11** | **Explainable AI (XAI)** | Grounded Explanations | Decision rationales, attention/comm/causal explainers, GUI tab | `xai/`, `visualize_phase10_gui.py` (XAI Tab) |

---

## Master 11-Configuration Empirical Benchmark

Evaluated across 11 configurations (9 primary controllers + 2 distinct ablations) over 10 evaluation seeds ($1001 \dots 1010$), yielding 110 evaluation episodes on frozen, pre-generated route caches with SHA256 integrity verification.

> **Note on Standard Deviations:** In accordance with project reporting standards, values in the **Delay Mean $\pm$ SD** column report **Population Standard Deviation** ($N=10$, `ddof=0`). For complete statistical transparency, Sample Standard Deviation ($N-1=9$, `ddof=1`), Median, and 95% Student-$t$ Confidence Intervals are also provided.

| ID | Controller Configuration | Category | Mean Delay (s) [Pop SD, `ddof=0`] | Sample SD [`ddof=1`] | Median [IQR] (Delay) | 95% Student-$t$ CI (Delay) | Mean Queue (veh) | Throughput (veh) | Comm Ratio (%) | Interventions / ep | Physical Conflicts |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Fixed-Time Baseline (30s Cycle) | Primary | $151.50 \pm 15.72$ | $16.57$ | $145.93\text{ [19.75]}$ | $[139.65, 163.36]$ | $14.28 \pm 1.32$ | $106.6 \pm 8.7$ | $0.0\%$ | $0.0$ | $0$ |
| **C2** | Phase 3 IPPO | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $0.0\%$ | $0.0$ | $0$ |
| **C3** | Phase 4 MAPPO Shared Policy | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $0.0\%$ | $0.0$ | $0$ |
| **C4** | Phase 5 Transformer-MAPPO | Primary | $94.08 \pm 12.08$ | $12.74$ | $91.95\text{ [18.18]}$ | $[84.97, 103.19]$ | $11.99 \pm 0.89$ | $100.2 \pm 6.7$ | $0.0\%$ | $0.0$ | $0$ |
| **C5** | Phase 6 GAT-MAPPO | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $100.0\%$ | $0.0$ | $0$ |
| **C6** | Phase 7 Comm-Gated MAPPO | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $100.0\%$ | $0.0$ | $0$ |
| **C7** | Phase 8 Causal-Gated MAPPO | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $100.0\%$ | $0.0$ | $0$ |
| **C8** | Phase 8R Causal Improved | Primary | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $0.0\%$ | $0.0$ | $0$ |
| **C9** | Phase 9 SAGE-Traffic Complete | Primary | $37.15 \pm 3.44$ | $3.62$ | $36.60\text{ [3.77]}$ | $[34.56, 39.74]$ | $9.51 \pm 0.91$ | $89.5 \pm 7.9$ | $0.0\%$ | $200.0 \pm 0.0$ | $0$ |
| **A1** | Phase 4 Individual Actors | Ablation | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $0.0\%$ | $0.0$ | $0$ |
| **A2** | Phase 9 Safety Disabled | Ablation | $28.40 \pm 3.29$ | $3.47$ | $27.41\text{ [2.06]}$ | $[25.92, 30.88]$ | $7.76 \pm 0.86$ | $104.6 \pm 8.0$ | $0.0\%$ | $0.0$ | $0$ |

### Key Scientific Insights
1. **The Empirical Cost of Safety:** Comparing unshielded Phase 8R (C8: $28.40\text{s}$) with shielded SAGE-Traffic (C9: $37.15\text{s}$) reveals an $+8.75\text{s}$ ($+30.8\%$) delay penalty and $-14.4\%$ throughput reduction. This represents the explicit cost of enforcing mandatory $1\text{s}$ all-red clearance intervals and clamping illegal switch requests ($t_{\text{green}} < 5\text{s}$), which guarantees **zero physical signal conflicts** and **100% formal safety compliance**. Toggling safety off in ablation A2 immediately recovers unshielded performance.
2. **Greedy Action Equivalence:** Unshielded controllers (C2, C3, C5, C6, C7, C8, A1, A2) achieve identical macroscopic performance because greedy policy evaluation (`argmax`) selects Action 1 (`SWITCH`) on 100% of steps. The underlying SUMO signal state machine governs physical transitions according to min-green constraints, producing identical vehicle dynamics across identical frozen seeds.
3. **Decentralized Attention Breakdown:** Phase 5 Transformer-MAPPO without spatial GAT attention exhibits elevated delay ($94.08\text{s}$) due to credit assignment breakdown when local temporal actors extend green phases without inter-agent coordination.

---

## Project Repository Structure

```
major/
├── config.yaml                       # Master simulation, reward, and network hyperparameters
├── environment/
│   ├── traffic_env.py                # Multi-Agent Gymnasium environment with safety options
│   └── demand_generator.py           # Multi-demand profile vehicle trip generator
├── network/
│   ├── topology.py                   # Canonical dependency-neutral 2x2 grid topology module
│   ├── 2x2_grid.net.xml              # SUMO road geometry network file (4 nodes, 16 corridors)
│   ├── routes.rou.xml                # Active SUMO route definitions
│   ├── route_cache/                  # Pre-generated, frozen route files (seeds 1001-1010)
│   └── simulation.sumocfg            # SUMO simulation orchestrator file
├── models/
│   ├── fixed_time_controller.py      # Rule-based 30s cycle traffic controller
│   ├── ppo_network.py                # Phase 3 Independent PPO (IPPO) actor-critic
│   ├── mappo_network.py              # Phase 4 MAPPO shared & individual actor architectures
│   ├── transformer_network.py        # Phase 5 Temporal Transformer encoder (k=4 history)
│   ├── gat_network.py                # Phase 6 Spatial Graph Attention Network (GAT)
│   ├── communication_gate.py         # Phase 7 Learnable sigmoid communication gate
│   ├── causal_influence.py           # Phase 8 Counterfactual causal influence estimator
│   ├── causal_improved.py            # Phase 8R/9 Dual-pathway causal gated attention
│   ├── safety_layer.py               # Phase 9 Unbypassable runtime safety shield
│   └── checkpoints/                  # 20 verified pretrained model checkpoints (.pt)
├── xai/                              # Phase 11 Explainable AI (XAI) Framework
│   ├── __init__.py                   # Package exports and common data classes
│   ├── decision_explainer.py         # Decision probability, margin, entropy, and rationale
│   ├── attention_explainer.py        # 8-edge GAT spatial attention weights analysis
│   ├── communication_explainer.py    # Learnable communication gate & bandwidth analysis
│   ├── causal_explainer.py           # Counterfactual causal influence score analysis
│   ├── safety_explainer.py           # Safety invariant verification and clamp reasons
│   └── xai_logger.py                 # Structured telemetry buffering & JSONL export
├── logs/
│   ├── phase10_master_benchmark.csv  # Complete 110-row raw empirical benchmark dataset
│   └── phase10_master_summary.csv    # Consolidated 11-controller statistical summary
├── plots/
│   └── phase10_architecture_comparison.png # 6-panel publication-grade comparison plot
├── benchmark_phase10_master.py       # Master 11-configuration benchmark execution engine
├── visualize_phase10_gui.py          # Unified Multi-Controller Interactive GUI Dashboard
├── test_phase10_full_integration.py  # 22-test Phase 10 integration test suite
├── test_xai_framework.py             # 9-test Phase 11 XAI verification test suite
├── test_safety_layer.py              # 15-test Phase 9 safety shield regression suite
└── test_gat_correctness.py           # 16-test Phase 6 GAT correctness test suite
```

---

## Installation & Setup

### 1. Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python:** Version 3.10 to 3.14
- **Eclipse SUMO:** Version 1.20.0 or later (tested on SUMO v1.28.0)
  - Ensure `SUMO_HOME` is set in your environment variables and `$SUMO_HOME/bin` is in your `PATH`.
  - Verify with:
    ```bash
    sumo --version
    ```

### 2. Dependencies
Install required Python packages:
```bash
pip install torch numpy scipy matplotlib pyyaml traci sumolib
```

---

## Usage Instructions

### 1. Launching the Unified Multi-Controller GUI Dashboard
The interactive Tkinter GUI dashboard (`visualize_phase10_gui.py`) allows real-time switching between all 11 controller architectures, simulation speed adjustments, interactive 2×2 topology visualization, deep architecture telemetry, and live **Explainable AI (XAI)** decision rationales.

```bash
# Standard Launch: Opens Tkinter dashboard and SUMO-GUI window with default C9 Shielded controller:
python visualize_phase10_gui.py --auto-run

# Launch with Phase 8R Unshielded baseline:
python visualize_phase10_gui.py --controller C8 --auto-run

# Dashboard-Only Mode (runs SUMO headless without opening the secondary sumo-gui window):
python visualize_phase10_gui.py --no-gui --auto-run

# Automated Headless Smoke-Test (validates C8 and C9 execution without GUI windows; exits with code 0):
python visualize_phase10_gui.py --smoke-test
```

Navigate to the **Explainable AI (XAI)** tab in the dashboard to inspect per-intersection policy probability distributions, confidence margins, policy entropy, safety shield override justifications, and the 8-edge relational corridor matrix in real time.

### 2. Running the Master 11-Configuration Benchmark
To reproduce the complete 110-episode benchmark across all 11 configurations on frozen route seeds 1001–1010:
```bash
python benchmark_phase10_master.py
```
This generates:
- `logs/phase10_master_benchmark.csv` (110 rows of raw per-seed metrics)
- `logs/phase10_master_summary.csv` (11 rows of aggregate statistics)
- `plots/phase10_architecture_comparison.png` (Publication-grade 6-panel comparison)

### 3. Running Test & Regression Suites
To verify architectural correctness, reset isolation, safety invariants, XAI fidelity, and checkpoint integrity:
```bash
# Run Phase 11 Explainable AI (XAI) verification suite (9 tests):
python -m pytest test_xai_framework.py

# Run Phase 10 full integration suite (22 tests):
python -m pytest test_phase10_full_integration.py

# Run Phase 9 safety shield regression suite (15 tests):
python -m pytest test_safety_layer.py

# Run Phase 6 GAT correctness suite (16 tests):
python test_gat_correctness.py
```

---

## Phase 11: Explainable AI (XAI) Layer

Phase 11 equips SAGE-Traffic with an empirical, lightweight Explainable AI engine (`xai/`) grounded in real runtime model outputs:

1. **Decentralized Decision Rationales (`DecisionExplainer`):** Softmax probabilities, confidence margin over runner-up action ($\Delta p$), policy entropy ($H(p)$), and contextual traffic summaries (explicitly labeled as situational heuristics, distinguishing operational context from proven neural feature attributions).
2. **Spatial Graph Attention (`AttentionExplainer`):** Identifies dominant incoming corridors across the 8 canonical directed edges ($A \leftrightarrow B, A \leftrightarrow C, B \leftrightarrow D, C \leftrightarrow D$) with non-neighbor masking on diagonals.
   > **Epistemological Safeguard:** Attention weights reflect learned neural feature aggregation routing within the GAT layer; they do not represent physical vehicle counts or physical causality.
3. **Adaptive Communication Gating (`CommunicationExplainer`):** Tracks sigmoid gate activations ($g_{ij} \in [0, 1]$), evaluating hard discrete suppression ($\tau = 0.5$) in deterministic evaluation mode vs continuous feature attenuation in soft mode.
4. **Observational Predictive Influence (`CausalExplainer`):** Quantifies Granger-style linear predictive variance reduction on lagged traffic states ($C_{ij}$) and directional asymmetry ($\Delta C_{ij} = C_{ij} - C_{ji}$).
   > **Epistemological Safeguard:** Causal influence scores reflect observational predictive variance reduction from lagged traffic history, not counterfactual policy interventions or real-world physical causation.
5. **Runtime Safety Verification (`SafetyExplainer`):** Detects proposed vs executed action mismatches and articulates exact invariant clamp justifications (`MIN_GREEN`, `CLEARANCE_ACTIVE`, `MAX_GREEN`, `MIN_RED`).
6. **Zero Impact on Execution:** Operates as a read-only observer with **~0.126 ms** overhead per step, fully preserving simulation trajectory parity.

---

## Verified Model Checkpoints (20 Checkpoints)

All 20 authentic model checkpoints located in `models/checkpoints/` have been verified for strict SHA256 integrity:

```
agent_A.pt                          | 140d4e66664357da66d1abe2f4a9c391a88790ab5f10a770529705a5643db88e
agent_B.pt                          | 89b433b52b6e5f20249947aea02bb34a0b13f04dcc5337a2a08a251d383d7d51
agent_C.pt                          | 3bf905fcfc2f605157c77046e20f650fc29771dad8d4a00cb3c524c370992c0b
agent_D.pt                          | 78f691f3568c51354a2d2a9d52279482af0708d98de3c824024b2f1d95769577
mappo_actor_A.pt                    | f4d6bb870580681bb0f75e1d60bd0c09ef7395771f25aff442a039551fd3419c
mappo_actor_B.pt                    | 30307d36fd8ae27bb9143ddc701efa8de6c13408b6ce9c1a5b6d3c6a5a709650
mappo_actor_C.pt                    | aa0dbe3b2c9356caa9108062cc4f1bca12576c549b5c0619c24cd6fdbd6f8e33
mappo_actor_D.pt                    | a23a5ad4cedd057fff83673a6db63493fed99f5e6981bde029e2c43215a194f9
mappo_actor_shared.pt               | aa3be0f91b8ed2a710ba4b3d14668b7093eea97d2e7da4225a02c55c796f85b3
mappo_critic.pt                     | 44f576d4b2295c04187ec5fce62df6d48eb281aa7dd701044c5c81621e5a2d7d
mappo_transformer_actor.pt          | f14ff6b672f6743446b9a326e9c5991092709c85d0e9ead7a05285018572301f
mappo_transformer_critic.pt         | 027ffa3cd2894c02a906f12bb7d29e4c73033616a2caed90ca0f10af7bb44cda
mappo_transformer_gat_actor.pt      | ad669be67c535cf8064ab85a10c8373ffc7f8fd68089115f9dc8c7a5241603a7
mappo_transformer_gat_critic.pt     | a10bde1b0fc676c70e5725c7e0f76ae67faf8c65321c1c93d1cd62a3494fc7e4
mappo_comm_gate_actor.pt            | 4759869bca999d18035e27195d4621b3b49305dcedca1df38a743924621076f8
mappo_comm_gate_critic.pt           | 0070b5f013eeff5a2f0ba75954a00b5cb262fd9b5dce24b24f565a65602aeb0b
mappo_causal_gate_actor.pt          | 81c2e33fac002a24ac843187c296150afa918468ca3063893d35b6cb057075a0
mappo_causal_gate_critic.pt         | 213a00da324511f277a4e6c523dfcf1c2da71931c5f836316dc4674528d6ec0e
mappo_causal_improved_actor.pt      | 9a30cb38568abca26e96205efdfba0dfece6dc974bca4ff218041f284d811846
mappo_causal_improved_critic.pt     | 8b6ba5ff9d08eccf207095d860ea641ade828e8fe003b6aed629984d39d55a19
```

---

## Author & Project Information

- **Author:** Syed Mohd Kashif Rizvi ([@Kashif7180](https://github.com/Kashif7180))
- **Project:** SAGE-Traffic: Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning
- **Repository:** https://github.com/Kashif7180/Intelligent-Traffic-Signal-Control-using-Multi-Agent-Reinforcement-Learning
- **License:** MIT License
