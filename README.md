# Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning

A comprehensive, state-of-the-art Multi-Agent Reinforcement Learning (MARL) framework for coordinated traffic signal control on urban road networks simulated using **Eclipse SUMO** (Simulation of Urban MObility).

---

## Overview

Traffic congestion in urban road networks leads to substantial economic losses, fuel waste, and increased emissions. Traditional traffic light controllers rely on fixed-cycle timing or isolated actuated logic, failing to dynamically adapt to varying traffic surges and coordinate across adjacent intersections.

This project designs, implements, trains, and empirically validates advanced multi-agent reinforcement learning architectures on a 4-intersection ($2 \times 2$ grid) traffic network with complex multi-lane dynamics and dynamic demand profiles.

---

## Key Architectures & Controllers

| Architecture | Paradigm | Communication / Mechanism | Key File |
| :--- | :--- | :--- | :--- |
| **Fixed-Time Baseline** | Rule-Based | None (Fixed periodic phase transitions) | `models/fixed_time_controller.py` |
| **IPPO** | Decentralized RL | Independent actors and critics per intersection | `train_ippo.py`, `models/ppo_network.py` |
| **MAPPO (CTDE)** | Centralized Training, Decentralized Execution | Centralized critic observing global joint state ($S_t \in \mathbb{R}^{172}$); decentralized actors | `train_mappo.py`, `models/mappo_network.py` |
| **MAPPO + Transformer** | CTDE + Attention | Spatio-temporal self-attention across historical traffic states | `train_mappo_transformer.py`, `models/transformer_network.py` |
| **MAPPO + GAT** | CTDE + Graph RL | Graph Attention Networks learning dynamic edge weights between adjacent intersections | `visualize_gat_attention.py`, `models/gat_network.py` |
| **Communication Gate** | Adaptive Comm | Information-theoretic gating to balance message overhead vs. coordination gain | `train_mappo_comm_gate.py`, `models/communication_gate.py` |
| **Causal Influence Gate** | Counterfactual Comm | Counterfactual causal attribution measuring impact of agent messages on neighbor policies | `train_mappo_causal_improved.py`, `models/causal_improved.py` |

---

## Project Structure

```
major/
├── config.yaml                       # Global simulation, reward, and model hyperparameters
├── environment/
│   ├── traffic_env.py                # Multi-agent SUMO Gym-style environment wrapper
│   └── demand_generator.py           # Dynamic Poisson/Weibull trip generation & flow curves
├── network/
│   ├── 2x2_grid.net.xml              # SUMO road network geometry (4 intersections)
│   ├── routes.rou.xml                # Route definitions and demand distribution
│   └── simulation.sumocfg            # SUMO simulation configuration
├── models/
│   ├── fixed_time_controller.py      # Baseline fixed-cycle controller
│   ├── ppo_network.py                # IPPO actor-critic networks
│   ├── mappo_network.py              # Centralized critic & decentralized actor networks
│   ├── transformer_network.py        # Transformer temporal encoder
│   ├── gat_network.py                # Graph Attention Network (GAT) for MARL
│   ├── communication_gate.py         # Dynamic gating layer
│   ├── causal_improved.py            # Causal influence estimator & gated actor
│   └── checkpoints/                  # Pretrained model weights (.pt)
├── experiments/                      # Training logs, reward curves, and benchmark plots
│   ├── controller_comparison_phase8r.png
│   ├── gat_attention_visualization.png
│   └── ... (CSV metrics & plots)
├── test_*.py                         # Comprehensive unit & forensic test suites
├── train_*.py                        # End-to-end training pipelines for each architecture
├── compare_all_controllers_phase8r.py # Rigorous multi-seed benchmark & evaluation
└── visualize_phase8r_gui.py          # Real-time Pygame graphical dashboard & simulation playback
```

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.10+ (tested on Python 3.10–3.14)
- **Eclipse SUMO**: Install SUMO and ensure `SUMO_HOME` environment variable is set.
  ```bash
  # Verify SUMO installation
  sumo --version
  ```

### 2. Dependencies
Install required Python packages:
```bash
pip install torch numpy scipy matplotlib pyyaml pygame traci sumolib
```

---

## Usage

### 1. Training Agents
Train any of the supported controllers (hyperparameters defined in `config.yaml`):

```bash
# Train MAPPO baseline (Centralized Training, Decentralized Execution)
python train_mappo.py

# Train MAPPO with Graph Attention Networks (GAT)
python train_mappo_transformer_gat.py

# Train MAPPO with Causal Influence Communication Gating
python train_mappo_causal_improved.py
```

### 2. Comparative Evaluation & Benchmarking
Run an empirical evaluation across identical random seeds to measure queue lengths, waiting times, and network throughput:

```bash
python compare_all_controllers_phase8r.py
```

### 3. Real-Time Graphical Simulation (GUI)
Launch the interactive visualizer displaying traffic flow, queue lengths, current signal phases, and live telemetry across all 4 intersections:

```bash
python visualize_phase8r_gui.py
```

---

## Results Summary

Benchmarking against Fixed-Time and Independent PPO baselines demonstrates that coordinated MARL controllers achieve:
- **Significant reduction in average vehicle delay and queue lengths**.
- **Higher network throughput** during asymmetric peak-flow conditions.
- **Dynamic coordination** through graph attention weights and causal communication, directing green waves to heavily congested corridors.

---

## Author & License

- **Author**: Syed Mohd Kashif Rizvi ([@Kashif7180](https://github.com/Kashif7180))
- **Project**: Major Project — Multi-Agent Reinforcement Learning for Intelligent Transportation Systems
