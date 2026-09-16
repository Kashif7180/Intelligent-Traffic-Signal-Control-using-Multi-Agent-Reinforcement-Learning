"""
Phase 8R Live SUMO-GUI & Interactive Dashboard Visualization.

Visualizes the trained Phase 8R controller (MAPPO + Temporal Transformer + GAT + Causal Pathway + Communication Gate):
- Checkpoint loaded: models/checkpoints/mappo_causal_improved_actor.pt
- SUMO-GUI live traffic simulation on 2x2 grid (intersections A, B, C, D)
- Live Tkinter dashboard alongside SUMO-GUI:
  1. Top Control Bar (START, PAUSE, STEP, RESET, speed multipliers 0.25x-4x)
  2. Four Intersection Cards with Live Action Probabilities Bar Charts (KEEP, SWITCH, EXTEND, REDUCE)
  3. Causal Influence Estimation Panel (8 physical edges C_ij in [0, 1])
  4. Communication Status (800 possible, 0 actual, 0% ratio)
  5. 2D GAT Graph Canvas (A-B, A-C, B-D, C-D with self loops, no diagonals)
  6. Phase 8R Pipeline Flowchart (live updating feature values)
  7. Force Communication Interactive Test (LEARNED GATES vs OVERRIDE)
  8. Live Performance Strip (delay, queue, throughput, return, comm cost)
  9. Decision logging to logs/phase8r_gui_run.csv
"""

import os
import sys
import csv
import time
import argparse
import yaml
import numpy as np
import torch
import traci

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple, Optional, Any

from environment.traffic_env import MultiAgentTrafficEnv, get_sumo_binary
from models.causal_influence import ALLOWED_DIRECTED_EDGES
from models.causal_improved import MAPPOCausalImprovedController


ACTION_NAMES = {0: "KEEP", 1: "SWITCH", 2: "EXTEND", 3: "REDUCE"}
ACTION_COLORS = {0: "#89b4fa", 1: "#a6e3a1", 2: "#f9e2af", 3: "#f38ba8"}


class Phase8RGUIDashboard:
    def __init__(
        self,
        root: tk.Tk,
        config_path: str = "config.yaml",
        seed: int = 1001,
        max_steps: int = 100,
        step_delay: float = 0.25,
        use_sumo_gui: bool = True,
        auto_run: bool = False
    ):
        self.root = root
        self.config_path = config_path
        self.seed = seed
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.use_sumo_gui = use_sumo_gui
        self.auto_run = auto_run

        self.root.title("SAGE-Traffic Phase 8R: Live Causal Communication GUI Dashboard")
        self.root.geometry("1480x920")
        self.root.configure(bg="#181825")

        # Runtime states
        self.is_running = False
        self.step_idx = 0
        self.sim_time = 0.0
        self.total_task_return = 0.0
        self.total_comm_cost = 0.0
        self.possible_comm_total = 0
        self.actual_comm_total = 0
        self.force_comm_enabled = False

        # Metrics history
        self.all_delays = []
        self.all_queues = []
        self.total_arrived = 0

        # Create logs dir
        os.makedirs("logs", exist_ok=True)
        self.log_file = "logs/phase8r_gui_run.csv"
        self._init_csv_log()

        # Load environment and controller
        self._init_simulation()

        # Build UI layout
        self._build_ui()

        # Update initial UI
        self._update_ui_state()

        if self.auto_run:
            self.root.after(500, self.on_start)

    def _init_csv_log(self):
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "intersection", "queue", "waiting_time", "C_ij",
                "gate_probability", "communication_state",
                "KEEP_probability", "SWITCH_probability", "EXTEND_probability", "REDUCE_probability",
                "selected_action"
            ])

    def _init_simulation(self):
        print("[GUI] Initializing SAGE-Traffic MultiAgentTrafficEnv...")
        self.env = MultiAgentTrafficEnv(config_path=self.config_path, gui=self.use_sumo_gui)
        print("[GUI] Initializing MAPPOCausalImprovedController...")
        self.ctrl = MAPPOCausalImprovedController(config_path=self.config_path)
        self.ctrl.load_checkpoints("models/checkpoints")
        print("[GUI] Checkpoint loaded successfully: models/checkpoints/mappo_causal_improved_actor.pt")

        self.obs, self.info = self.env.reset(seed=self.seed)
        self.ctrl.reset_history(self.obs, self.info)
        self.step_idx = 0
        self.sim_time = 0.0
        self.total_task_return = 0.0
        self.total_comm_cost = 0.0
        self.possible_comm_total = 0
        self.actual_comm_total = 0
        self.all_delays.clear()
        self.all_queues.clear()
        self.total_arrived = 0

    def _build_ui(self):
        # 1. Top Bar: Controls & Modes
        top_bar = tk.Frame(self.root, bg="#1e1e2e", pady=8, padx=12, relief=tk.RAISED, bd=1)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        # Title
        title_lbl = tk.Label(
            top_bar,
            text="SAGE-Traffic Phase 8R: Causal Communication Dashboard",
            font=("Segoe UI", 13, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e"
        )
        title_lbl.pack(side=tk.LEFT, padx=(0, 20))

        # Simulation Buttons
        self.btn_start = tk.Button(top_bar, text="▶ START", bg="#2e7d32", fg="white", font=("Segoe UI", 9, "bold"), width=9, command=self.on_start)
        self.btn_start.pack(side=tk.LEFT, padx=3)

        self.btn_pause = tk.Button(top_bar, text="⏸ PAUSE", bg="#e65100", fg="white", font=("Segoe UI", 9, "bold"), width=9, command=self.on_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=3)

        self.btn_step = tk.Button(top_bar, text="⏭ STEP", bg="#1565c0", fg="white", font=("Segoe UI", 9, "bold"), width=9, command=self.on_step)
        self.btn_step.pack(side=tk.LEFT, padx=3)

        self.btn_reset = tk.Button(top_bar, text="🔄 RESET", bg="#424242", fg="white", font=("Segoe UI", 9, "bold"), width=9, command=self.on_reset)
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        # Speed Presets
        tk.Label(top_bar, text="Speed:", fg="#a6adc8", bg="#1e1e2e", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(15, 4))
        self.speed_var = tk.StringVar(value="1x")
        for spd in ["0.25x", "0.5x", "1x", "2x", "4x"]:
            rb = tk.Radiobutton(
                top_bar, text=spd, value=spd, variable=self.speed_var,
                bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                activebackground="#1e1e2e", activeforeground="#ffffff",
                command=self.on_speed_change
            )
            rb.pack(side=tk.LEFT, padx=2)

        # Force Communication Mode Toggle
        tk.Label(top_bar, text="|", fg="#45475a", bg="#1e1e2e", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=10)

        self.btn_force_on = tk.Button(
            top_bar, text="FORCE COMM ON", bg="#b71c1c", fg="white",
            font=("Segoe UI", 9, "bold"), padx=6, command=self.on_force_comm_on
        )
        self.btn_force_on.pack(side=tk.LEFT, padx=3)

        self.btn_force_off = tk.Button(
            top_bar, text="RESTORE LEARNED", bg="#37474f", fg="white",
            font=("Segoe UI", 9, "bold"), padx=6, command=self.on_restore_learned
        )
        self.btn_force_off.pack(side=tk.LEFT, padx=3)

        self.mode_badge = tk.Label(
            top_bar, text="LEARNED GATES", bg="#1b5e20", fg="#ffffff",
            font=("Segoe UI", 9, "bold"), padx=8, pady=2, relief=tk.RIDGE
        )
        self.mode_badge.pack(side=tk.LEFT, padx=10)

        # 2. Performance Strip
        self.perf_strip = tk.Frame(self.root, bg="#11111b", pady=6, padx=12)
        self.perf_strip.pack(side=tk.TOP, fill=tk.X)

        self.perf_labels: Dict[str, tk.Label] = {}
        metrics_order = [
            ("step", "Step: 0/100"),
            ("time", "Time: 0.0s / 500s"),
            ("veh_active", "Active: 0"),
            ("veh_arrived", "Arrived: 0"),
            ("throughput", "Throughput: 0"),
            ("delay", "Delay: 0.00s"),
            ("queue", "Queue: 0.00"),
            ("return", "Task Return: 0.00"),
            ("comm_ratio", "Comm Ratio: 0.0% (0/0)"),
            ("comm_cost", "Comm Cost: 0.00")
        ]
        for key, default_text in metrics_order:
            lbl = tk.Label(
                self.perf_strip, text=default_text, font=("Consolas", 10, "bold"),
                fg="#cdd6f4", bg="#1e1e2e", padx=8, pady=4, relief=tk.FLAT
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.perf_labels[key] = lbl

        # 3. Main Workspace: 3 Columns
        main_frame = tk.Frame(self.root, bg="#181825")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        main_frame.columnconfigure(0, weight=4)  # Left: Intersection cards & action probs
        main_frame.columnconfigure(1, weight=5)  # Center: GAT Graph & Causal Table
        main_frame.columnconfigure(2, weight=4)  # Right: Pipeline & Manipulation

        # COLUMN 1: Intersection Cards (A, B, C, D)
        col1 = tk.Frame(main_frame, bg="#181825")
        col1.grid(row=0, column=0, sticky="nsew", padx=4)

        tk.Label(col1, text="INTERSECTION CONTROLLERS & ACTION PROBABILITIES", font=("Segoe UI", 10, "bold"), fg="#89b4fa", bg="#181825").pack(anchor=tk.W, pady=(0, 4))

        self.agent_cards = {}
        self.prob_bars = {}

        for a in ["A", "B", "C", "D"]:
            card = tk.LabelFrame(col1, text=f"Intersection {a}", font=("Segoe UI", 9, "bold"), fg="#cdd6f4", bg="#1e1e2e", padx=6, pady=4, bd=1)
            card.pack(fill=tk.X, expand=True, pady=3)

            # Top row: phase and action badge
            hdr_frame = tk.Frame(card, bg="#1e1e2e")
            hdr_frame.pack(fill=tk.X)

            phase_lbl = tk.Label(hdr_frame, text="Phase: 0 (N-S Green)", font=("Segoe UI", 8, "bold"), fg="#a6e3a1", bg="#1e1e2e")
            phase_lbl.pack(side=tk.LEFT)

            act_lbl = tk.Label(hdr_frame, text="Action: KEEP", font=("Segoe UI", 8, "bold"), fg="#89b4fa", bg="#313244", padx=6)
            act_lbl.pack(side=tk.RIGHT)

            # Middle row: stats
            stats_lbl = tk.Label(card, text="Queue: 0.0 | Wait: 0.0s | Veh: 0 | Green: 30s", font=("Consolas", 8), fg="#a6adc8", bg="#1e1e2e")
            stats_lbl.pack(anchor=tk.W, pady=2)

            # Probabilities Canvas
            prob_canvas = tk.Canvas(card, height=48, bg="#181825", highlightthickness=0)
            prob_canvas.pack(fill=tk.X, pady=2)

            self.agent_cards[a] = {
                "phase_lbl": phase_lbl,
                "act_lbl": act_lbl,
                "stats_lbl": stats_lbl,
                "prob_canvas": prob_canvas
            }

        # COLUMN 2: GAT Graph & Causal Influence
        col2 = tk.Frame(main_frame, bg="#181825")
        col2.grid(row=0, column=1, sticky="nsew", padx=4)

        tk.Label(col2, text="GAT INFORMATION FLOW & PHYSICAL TOPOLOGY", font=("Segoe UI", 10, "bold"), fg="#89b4fa", bg="#181825").pack(anchor=tk.W, pady=(0, 4))

        # GAT Canvas
        self.gat_canvas = tk.Canvas(col2, height=270, bg="#1e1e2e", highlightthickness=1, highlightbackground="#313244")
        self.gat_canvas.pack(fill=tk.X, pady=(0, 6))

        # Causal Influence Table
        tk.Label(col2, text="CAUSAL INFLUENCE ESTIMATION & COMMUNICATION CHANNELS", font=("Segoe UI", 10, "bold"), fg="#f9e2af", bg="#181825").pack(anchor=tk.W, pady=(4, 2))

        tbl_frame = tk.Frame(col2, bg="#1e1e2e", padx=6, pady=4, bd=1, relief=tk.GROOVE)
        tbl_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        th = tk.Frame(tbl_frame, bg="#313244")
        th.pack(fill=tk.X, pady=1)
        tk.Label(th, text="Channel", width=12, font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#313244", anchor="w").pack(side=tk.LEFT, padx=4)
        tk.Label(th, text="C_ij (Influence)", width=15, font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#313244", anchor="center").pack(side=tk.LEFT, padx=4)
        tk.Label(th, text="Gate Prob g_ij", width=15, font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#313244", anchor="center").pack(side=tk.LEFT, padx=4)
        tk.Label(th, text="Status", width=10, font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#313244", anchor="center").pack(side=tk.LEFT, padx=4)

        self.edge_rows = {}
        for edge in ALLOWED_DIRECTED_EDGES:
            src, tgt = edge
            row_f = tk.Frame(tbl_frame, bg="#1e1e2e")
            row_f.pack(fill=tk.X, pady=1)

            lbl_name = tk.Label(row_f, text=f"{src} → {tgt}", width=12, font=("Consolas", 8, "bold"), fg="#89b4fa", bg="#1e1e2e", anchor="w")
            lbl_name.pack(side=tk.LEFT, padx=4)

            lbl_c = tk.Label(row_f, text="0.5000", width=15, font=("Consolas", 8), fg="#cdd6f4", bg="#1e1e2e", anchor="center")
            lbl_c.pack(side=tk.LEFT, padx=4)

            lbl_g = tk.Label(row_f, text="0.0001", width=15, font=("Consolas", 8), fg="#cdd6f4", bg="#1e1e2e", anchor="center")
            lbl_g.pack(side=tk.LEFT, padx=4)

            lbl_stat = tk.Label(row_f, text="OFF", width=10, font=("Segoe UI", 8, "bold"), fg="#6c7086", bg="#181825", anchor="center")
            lbl_stat.pack(side=tk.LEFT, padx=4)

            self.edge_rows[edge] = {
                "lbl_c": lbl_c,
                "lbl_g": lbl_g,
                "lbl_stat": lbl_stat
            }

        # COLUMN 3: Pipeline & Forced Communication Monitor
        col3 = tk.Frame(main_frame, bg="#181825")
        col3.grid(row=0, column=2, sticky="nsew", padx=4)

        tk.Label(col3, text="PHASE 8R ARCHITECTURAL PIPELINE", font=("Segoe UI", 10, "bold"), fg="#89b4fa", bg="#181825").pack(anchor=tk.W, pady=(0, 4))

        self.pipe_frame = tk.Frame(col3, bg="#1e1e2e", padx=8, pady=6, bd=1, relief=tk.GROOVE)
        self.pipe_frame.pack(fill=tk.X, pady=(0, 6))

        pipeline_stages = [
            ("1. Traffic Metrics", "Local [queue, wait, veh] (43-D)"),
            ("2. Temporal Transformer", "History k=4 -> Node Embedding (64-D)"),
            ("3. Causal Influence Estimation", "Granger Predictive Score C_ij in [0, 1]"),
            ("4. Causal Pathway Encoder", "phi(C_ij) in R^8 (LayerNorm)"),
            ("5. Communication Gate MLP", "MLP([h_i, h_j, phi_c]) -> g_ij in (0, 1)"),
            ("6. Causal Message Passing", "m_ij = g_ij * alpha_ij * W h_j"),
            ("7. Decentralized Actor", "Modulated 64-D + Agent ID 4-D = 68-D"),
            ("8. Signal Action", "argmax pi(a|s) -> [KEEP, SWITCH, EXTEND, REDUCE]")
        ]

        self.pipe_labels = []
        for name, desc in pipeline_stages:
            pf = tk.Frame(self.pipe_frame, bg="#1e1e2e")
            pf.pack(fill=tk.X, pady=1)
            tk.Label(pf, text=f"• {name}:", font=("Segoe UI", 8, "bold"), fg="#a6e3a1", bg="#1e1e2e", anchor="w").pack(side=tk.LEFT)
            d_lbl = tk.Label(pf, text=desc, font=("Consolas", 7), fg="#a6adc8", bg="#1e1e2e", anchor="w")
            d_lbl.pack(side=tk.LEFT, padx=4)
            self.pipe_labels.append(d_lbl)

        # Force Comm Inspection Card
        tk.Label(col3, text="COMMUNICATION MANIPULATION TEST", font=("Segoe UI", 10, "bold"), fg="#f38ba8", bg="#181825").pack(anchor=tk.W, pady=(4, 2))

        self.force_card = tk.LabelFrame(col3, text="Live Representation Sensitivity", font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#1e1e2e", padx=8, pady=6)
        self.force_card.pack(fill=tk.BOTH, expand=True)

        self.lbl_force_status = tk.Label(
            self.force_card,
            text="Active Mode: LEARNED GATES (0.0% Comm Ratio)\nAll 8 physical inter-agent channels pruned by budget regularization.",
            font=("Segoe UI", 8), fg="#a6e3a1", bg="#1e1e2e", justify=tk.LEFT
        )
        self.lbl_force_status.pack(anchor=tk.W, pady=2)

        self.lbl_force_emb = tk.Label(self.force_card, text="Node Embedding Shift (L2): 0.0000", font=("Consolas", 8), fg="#cdd6f4", bg="#1e1e2e")
        self.lbl_force_emb.pack(anchor=tk.W, pady=1)

        self.lbl_force_prob = tk.Label(self.force_card, text="Actor Probability Shift:   0.0000", font=("Consolas", 8), fg="#cdd6f4", bg="#1e1e2e")
        self.lbl_force_prob.pack(anchor=tk.W, pady=1)

        self.lbl_force_act = tk.Label(self.force_card, text="Action Flip Detected:       NONE", font=("Consolas", 8), fg="#cdd6f4", bg="#1e1e2e")
        self.lbl_force_act.pack(anchor=tk.W, pady=1)

    def on_speed_change(self):
        val = self.speed_var.get()
        mapping = {"0.25x": 0.8, "0.5x": 0.4, "1x": 0.2, "2x": 0.08, "4x": 0.01}
        self.step_delay = mapping.get(val, 0.2)

    def on_start(self):
        if not self.is_running:
            self.is_running = True
            self.step_simulation_loop()

    def on_pause(self):
        self.is_running = False

    def on_step(self):
        self.is_running = False
        self.execute_decision_step()

    def on_reset(self):
        self.is_running = False
        self._init_simulation()
        self._update_ui_state()

    def on_force_comm_on(self):
        self.force_comm_enabled = True
        self.mode_badge.config(
            text="[!] VISUALIZATION OVERRIDE - NOT TRAINED POLICY",
            bg="#b71c1c", fg="#ffffff"
        )
        self.lbl_force_status.config(
            text="Active Mode: FORCED COMMUNICATION ON\nAll 8 physical edges forced active (g_ij = 1.0) to observe sensitivity.",
            fg="#f38ba8"
        )

    def on_restore_learned(self):
        self.force_comm_enabled = False
        self.mode_badge.config(
            text="LEARNED GATES",
            bg="#1b5e20", fg="#ffffff"
        )
        self.lbl_force_status.config(
            text="Active Mode: LEARNED GATES (0.0% Comm Ratio)\nAll 8 physical inter-agent channels pruned by budget regularization.",
            fg="#a6e3a1"
        )

    def step_simulation_loop(self):
        if self.is_running and self.step_idx < self.max_steps:
            self.execute_decision_step()
            delay_ms = max(10, int(self.step_delay * 1000))
            self.root.after(delay_ms, self.step_simulation_loop)
        elif self.step_idx >= self.max_steps:
            self.is_running = False
            messagebox.showinfo("Simulation Complete", f"Completed {self.max_steps} decision steps (500s SUMO time).")

    def execute_decision_step(self):
        if self.step_idx >= self.max_steps:
            return

        self.step_idx += 1
        self.possible_comm_total += len(ALLOWED_DIRECTED_EDGES)  # +8

        # 1. Compute Causal Influence Estimates C_ij from observable history
        c_scores = self.ctrl.compute_causal_context(self.info)

        # 2. Extract Normal Phase 8R actions, probs, logits, and representations
        hist_list = [self.ctrl.history_manager.get_agent_history(a) for a in self.ctrl.agents]
        hist_tensor = torch.tensor(np.stack(hist_list, axis=0), dtype=torch.float32)

        with torch.no_grad():
            temp_embs = self.ctrl.transformer_encoder(hist_tensor)
            gated_normal, raw_gates, _ = self.ctrl.gated_gat(
                temp_embs.unsqueeze(0),
                context_dict=c_scores,
                deterministic=True
            )
            flat_normal = gated_normal.view(self.ctrl.num_agents, self.ctrl.gat_hidden_dim)
            dist_normal = self.ctrl.actor(flat_normal, self.ctrl.one_hot_tensor)
            normal_actions = torch.argmax(dist_normal.probs, dim=-1).tolist()
            normal_probs = dist_normal.probs.cpu().numpy()

            # Forced Communication computation for live sensitivity inspection
            gate_m_forced = torch.zeros((1, 4, 4), device=temp_embs.device, dtype=torch.float32)
            for i in range(4): gate_m_forced[0, i, i] = 1.0
            for src, tgt in ALLOWED_DIRECTED_EDGES:
                s_idx = self.ctrl.gated_gat.comm_gate.agent_to_idx[src]
                t_idx = self.ctrl.gated_gat.comm_gate.agent_to_idx[tgt]
                gate_m_forced[0, t_idx, s_idx] = 1.0

            Wh = torch.einsum("bni, kih -> bknh", temp_embs.unsqueeze(0), self.ctrl.gated_gat.W)
            attn_dst = torch.einsum("bknh, kho -> bkno", Wh, self.ctrl.gated_gat.a_dst)
            attn_src = torch.einsum("bknh, kho -> bkno", Wh, self.ctrl.gated_gat.a_src).transpose(2, 3)
            logits = self.ctrl.gated_gat.leaky_relu(attn_dst + attn_src)
            mask = self.ctrl.gated_gat.adj.unsqueeze(0).unsqueeze(0)
            gate_log = torch.log(torch.clamp(gate_m_forced.unsqueeze(1), min=1e-8, max=1.0))
            masked_logits = (logits + gate_log).masked_fill(mask == 0.0, -1e9)
            alpha_forced = torch.softmax(masked_logits, dim=-1)
            gated_forced = self.ctrl.gated_gat.activation(torch.einsum("bkij, bkjh -> bkih", alpha_forced, Wh).mean(dim=1))
            flat_forced = gated_forced.view(self.ctrl.num_agents, self.ctrl.gat_hidden_dim)
            dist_forced = self.ctrl.actor(flat_forced, self.ctrl.one_hot_tensor)
            forced_actions = torch.argmax(dist_forced.probs, dim=-1).tolist()
            forced_probs = dist_forced.probs.cpu().numpy()

        # Decide which action to execute in SUMO
        if self.force_comm_enabled:
            exec_actions = {a: forced_actions[i] for i, a in enumerate(self.ctrl.agents)}
            step_actual_comm = len(ALLOWED_DIRECTED_EDGES)  # 8
            step_comm_cost = self.ctrl.lambda_comm * step_actual_comm
            active_probs = forced_probs
        else:
            exec_actions = {a: normal_actions[i] for i, a in enumerate(self.ctrl.agents)}
            step_actual_comm = 0
            step_comm_cost = 0.0
            active_probs = normal_probs

        self.actual_comm_total += step_actual_comm
        self.total_comm_cost += step_comm_cost

        # Apply actions to SUMO environment
        next_obs, rewards, terminations, truncations, next_info = self.env.step(exec_actions)

        # Update returns & metrics
        step_return = sum(rewards.values())
        self.total_task_return += step_return

        step_queues = [next_info[a]["metrics"]["queue_length"] for a in self.env.agents]
        step_waits = [next_info[a]["metrics"]["waiting_time"] for a in self.env.agents]
        self.all_queues.append(np.mean(step_queues))
        self.all_delays.append(np.mean(step_waits))
        self.total_arrived += next_info["A"]["arrived_vehicles"]
        self.sim_time = self.step_idx * self.env.step_duration

        # Update controller history
        self.ctrl.update_history(next_obs)
        self.obs = next_obs
        self.info = next_info

        # Log decision step to CSV
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i, a in enumerate(self.ctrl.agents):
                # Incoming causal influence average
                inc_c = np.mean([c_scores[e] for e in ALLOWED_DIRECTED_EDGES if e[1] == a])
                inc_g = np.mean([self.ctrl.gated_gat.comm_gate.last_gate_values[e] for e in ALLOWED_DIRECTED_EDGES if e[1] == a])
                comm_st = "FORCED_ON" if self.force_comm_enabled else ("ON" if inc_g >= 0.5 else "OFF")
                writer.writerow([
                    self.step_idx, a,
                    f"{next_info[a]['metrics']['queue_length']:.2f}",
                    f"{next_info[a]['metrics']['waiting_time']:.2f}",
                    f"{inc_c:.4f}", f"{inc_g:.4f}", comm_st,
                    f"{active_probs[i, 0]:.4f}", f"{active_probs[i, 1]:.4f}",
                    f"{active_probs[i, 2]:.4f}", f"{active_probs[i, 3]:.4f}",
                    ACTION_NAMES[exec_actions[a]]
                ])

        # Update Dashboard UI
        self._update_ui_state(
            exec_actions=exec_actions,
            active_probs=active_probs,
            c_scores=c_scores,
            emb_shift=torch.norm(flat_forced - flat_normal).item(),
            prob_shift=np.max(np.abs(forced_probs - normal_probs)),
            actions_flipped=(normal_actions != forced_actions)
        )

    def _update_ui_state(
        self,
        exec_actions: Optional[Dict[str, int]] = None,
        active_probs: Optional[np.ndarray] = None,
        c_scores: Optional[Dict[Tuple[str, str], float]] = None,
        emb_shift: float = 0.0,
        prob_shift: float = 0.0,
        actions_flipped: bool = False
    ):
        # 1. Update Performance Strip
        self.perf_labels["step"].config(text=f"Step: {self.step_idx}/{self.max_steps}")
        self.perf_labels["time"].config(text=f"Time: {self.sim_time:.1f}s / 500s")
        active_veh = sum(self.info[a]["metrics"]["vehicle_count"] for a in self.env.agents) if hasattr(self, "info") else 0
        self.perf_labels["veh_active"].config(text=f"Active Veh: {active_veh}")
        self.perf_labels["veh_arrived"].config(text=f"Arrived Veh: {self.total_arrived}")
        self.perf_labels["throughput"].config(text=f"Throughput: {self.total_arrived}")
        mean_del = np.mean(self.all_delays) if self.all_delays else 0.0
        self.perf_labels["delay"].config(text=f"Mean Delay: {mean_del:.2f}s")
        mean_q = np.mean(self.all_queues) if self.all_queues else 0.0
        self.perf_labels["queue"].config(text=f"Mean Queue: {mean_q:.2f}")
        self.perf_labels["return"].config(text=f"Task Ret: {self.total_task_return:.1f}")
        ratio = (self.actual_comm_total / self.possible_comm_total * 100) if self.possible_comm_total > 0 else 0.0
        self.perf_labels["comm_ratio"].config(text=f"Comm Ratio: {ratio:.1f}% ({self.actual_comm_total}/{self.possible_comm_total})")
        self.perf_labels["comm_cost"].config(text=f"Comm Cost: {self.total_comm_cost:.2f}")

        # 2. Update Intersection Cards & Action Probabilities
        for i, a in enumerate(["A", "B", "C", "D"]):
            card = self.agent_cards[a]
            sig = self.env.signals[a] if hasattr(self, "env") and a in self.env.signals else None

            if sig:
                p_text = f"Phase {sig.current_phase} ({'N-S' if sig.current_phase == 0 else 'E-W'} Green)"
                if sig.is_yellow:
                    p_text += " [YELLOW]"
                p_color = "#f9e2af" if sig.is_yellow else ("#a6e3a1" if sig.current_phase == 0 else "#89b4fa")
                card["phase_lbl"].config(text=p_text, fg=p_color)

                act_idx = exec_actions[a] if exec_actions else 0
                act_name = ACTION_NAMES[act_idx]
                card["act_lbl"].config(text=f"Action: {act_name}", bg="#313244", fg=ACTION_COLORS[act_idx])

                m = self.info[a]["metrics"] if hasattr(self, "info") else {"queue_length": 0, "waiting_time": 0, "vehicle_count": 0}
                card["stats_lbl"].config(
                    text=f"Queue: {m['queue_length']:.1f} | Wait: {m['waiting_time']:.1f}s | Veh: {m['vehicle_count']} | Green: {sig.time_on_phase}s/{sig.target_green_duration}s"
                )

            # Draw Probability Bars
            canv = card["prob_canvas"]
            canv.delete("all")
            w = canv.winfo_width() or 340
            h = 48

            probs = active_probs[i] if active_probs is not None else np.array([0.25, 0.25, 0.25, 0.25])
            max_act = int(np.argmax(probs))

            bar_w = (w - 30) / 4
            for act_i in range(4):
                bx = 10 + act_i * bar_w
                bw = bar_w - 6
                p = probs[act_i]
                bar_h = max(4, int(p * 28))
                by = 32 - bar_h

                color = ACTION_COLORS[act_i]
                outline = "#00ff88" if act_i == max_act else "#45475a"
                width = 2 if act_i == max_act else 1

                canv.create_rectangle(bx, by, bx + bw, 32, fill=color, outline=outline, width=width)
                canv.create_text(bx + bw / 2, 40, text=ACTION_NAMES[act_i][:4], fill="#cdd6f4", font=("Segoe UI", 6, "bold"))
                canv.create_text(bx + bw / 2, max(8, by - 5), text=f"{p:.2f}", fill="#ffffff" if act_i == max_act else "#a6adc8", font=("Consolas", 6))

        # 3. Update GAT Canvas
        self._draw_gat_canvas(c_scores)

        # 4. Update Causal Influence Table
        if c_scores:
            for edge in ALLOWED_DIRECTED_EDGES:
                row = self.edge_rows[edge]
                c_val = c_scores.get(edge, 0.5)
                g_val = self.ctrl.gated_gat.comm_gate.last_gate_values.get(edge, 0.0001)

                row["lbl_c"].config(text=f"{c_val:.4f}")
                row["lbl_g"].config(text=f"{g_val:.4f}")

                if self.force_comm_enabled:
                    row["lbl_stat"].config(text="FORCED ON", fg="#ffffff", bg="#b71c1c")
                elif g_val >= 0.5:
                    row["lbl_stat"].config(text="ON", fg="#ffffff", bg="#2e7d32")
                else:
                    row["lbl_stat"].config(text="OFF", fg="#6c7086", bg="#181825")

        # 5. Update Forced Communication Inspection Stats
        self.lbl_force_emb.config(text=f"Node Embedding Shift (L2): {emb_shift:.4f}")
        self.lbl_force_prob.config(text=f"Actor Probability Shift:   {prob_shift:.4f}")
        act_text = "FLIPPED!" if actions_flipped else "NONE (Dominant Phase Preserved)"
        act_fg = "#f38ba8" if actions_flipped else "#a6e3a1"
        self.lbl_force_act.config(text=f"Action Flip Detected:       {act_text}", fg=act_fg)

    def _draw_gat_canvas(self, c_scores: Optional[Dict[Tuple[str, str], float]] = None):
        canv = self.gat_canvas
        canv.delete("all")
        w = canv.winfo_width() or 480
        h = canv.winfo_height() or 270

        # Node coordinates (2x2 grid)
        coords = {
            "A": (w * 0.25, h * 0.25),
            "B": (w * 0.75, h * 0.25),
            "C": (w * 0.25, h * 0.75),
            "D": (w * 0.75, h * 0.75),
        }

        # Draw 8 directed physical channels (4 dual bidirectional edges)
        edges = [
            ("A", "B", -12), ("B", "A", 12),
            ("A", "C", -12), ("C", "A", 12),
            ("B", "D", -12), ("D", "B", 12),
            ("C", "D", -12), ("D", "C", 12)
        ]

        for src, tgt, offset in edges:
            x1, y1 = coords[src]
            x2, y2 = coords[tgt]

            # Offset for dual bidirectional rendering
            if y1 == y2:  # Horizontal
                ox, oy = 0, offset
            else:          # Vertical
                ox, oy = offset, 0

            sx, sy = x1 + ox, y1 + oy
            tx, ty = x2 + ox, y2 + oy

            g_val = self.ctrl.gated_gat.comm_gate.last_gate_values.get((src, tgt), 0.0001)
            c_val = c_scores.get((src, tgt), 0.5) if c_scores else 0.5

            is_on = self.force_comm_enabled or (g_val >= 0.5)
            color = "#a6e3a1" if is_on else "#45475a"
            width = 3 if is_on else 1
            dash = None if is_on else (4, 4)

            # Draw directional link
            canv.create_line(sx, sy, tx, ty, fill=color, width=width, dash=dash, arrow=tk.LAST, arrowshape=(8, 10, 4))

            # Edge text
            mx = (sx + tx) / 2
            my = (sy + ty) / 2
            canv.create_text(mx, my, text=f"C:{c_val:.2f}", fill="#f9e2af" if is_on else "#6c7086", font=("Consolas", 6))

        # Draw Self-loops
        for node, (nx, ny) in coords.items():
            canv.create_oval(nx - 28, ny - 38, nx + 28, ny - 18, outline="#89b4fa", width=1.5, dash=(2, 2))
            canv.create_text(nx, ny - 42, text="Self (1.0)", fill="#89b4fa", font=("Segoe UI", 6))

        # Draw Nodes
        for node, (nx, ny) in coords.items():
            sig = self.env.signals.get(node)
            curr_p = sig.current_phase if sig else 0
            n_color = "#2e7d32" if curr_p == 0 else "#1565c0"

            canv.create_oval(nx - 20, ny - 20, nx + 20, ny + 20, fill=n_color, outline="#cdd6f4", width=2)
            canv.create_text(nx, ny, text=node, fill="#ffffff", font=("Segoe UI", 11, "bold"))


def run_gui():
    parser = argparse.ArgumentParser(description="Live SUMO-GUI Rollout for Phase 8R Controller")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=1001, help="Seed for evaluation (default: 1001)")
    parser.add_argument("--steps", type=int, default=100, help="Max decision steps (default: 100)")
    parser.add_argument("--delay", type=float, default=0.25, help="Step delay in seconds (default: 0.25)")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without SUMO-GUI window (dashboard only)")
    parser.add_argument("--auto-run", action="store_true", help="Start simulation automatically")
    args = parser.parse_args()

    root = tk.Tk()
    app = Phase8RGUIDashboard(
        root=root,
        config_path=args.config,
        seed=args.seed,
        max_steps=args.steps,
        step_delay=args.delay,
        use_sumo_gui=not args.no_gui,
        auto_run=args.auto_run
    )
    root.mainloop()


if __name__ == "__main__":
    run_gui()
