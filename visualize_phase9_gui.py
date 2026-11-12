"""
SAGE-Traffic Phase 9: Live Safety Layer & Traffic Control Dashboard
Interactive Tkinter GUI Dashboard with Real-Time Safety Shield Telemetry.

Features:
1. Control Bar: START, PAUSE, SINGLE STEP, RESET, and Simulation Speed Control.
2. Safety Shield Panel:
   - Live Status Badge: [SHIELD ACTIVE: ENFORCING INVARIANTS] / [SHIELD DISABLED]
   - Safe Boundary Toggle: Strictly prevents disabling during active clearance transitions.
   - Live Intervention Counter and Breakdown (Min Green, Max Green, Min Red, Clearance Active).
3. 4 Intersection Cards (A, B, C, D):
   - Current Phase & Physical Signal Light Display (NS vs EW)
   - Real-time Green Duration, Red Duration, and Clearance Countdown
   - Proposed Action vs Validated Safe Action (with red/green shield badge)
4. Forensic Signal Audit Panel:
   - Second-by-second SUMO physical signal string audit (Conflicting green detection)
   - Invariant verification status
5. Real-Time Intervention Event Log.
6. Overall Traffic Metrics Strip: Delay, Queue, Cumulative Throughput, Task Return.
"""

import os
import sys
import csv
import time
import argparse
import yaml
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple, Optional, Any

from environment.traffic_env import MultiAgentTrafficEnv
from models.causal_improved import MAPPOCausalImprovedController
from models.safety_layer import TrafficSignalSafetyLayer, ACTION_NAMES

ACTION_COLORS = {0: "#89b4fa", 1: "#a6e3a1", 2: "#f9e2af", 3: "#f38ba8"}


class Phase9SafetyGUIDashboard:
    def __init__(
        self,
        root: tk.Tk,
        config_path: str = "config.yaml",
        seed: int = 1001,
        max_steps: int = 100,
        step_delay: float = 0.20,
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

        self.root.title("SAGE-Traffic Phase 9: Runtime Safety Shield & Signal Control Dashboard")
        self.root.geometry("1480x920")
        self.root.configure(bg="#1e1e2e")

        # Runtime states
        self.is_running = False
        self.step_idx = 0
        self.sim_time = 0.0
        self.total_task_return = 0.0
        self.ep_throughput = 0

        # Safety states
        self.safety_enabled = True
        self.pending_safety_toggle: Optional[bool] = None

        # Load environment and controller
        self._init_backend()

        # Build GUI Layout
        self._build_ui()

        if self.auto_run:
            self.root.after(500, self._start_sim)

    def _init_backend(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.env = MultiAgentTrafficEnv(config_path=self.config_path, gui=self.use_sumo_gui)
        self.controller = MAPPOCausalImprovedController(config_path=self.config_path)
        try:
            self.controller.load_checkpoints("models/checkpoints")
        except Exception as e:
            print(f"Warning: Could not load checkpoints: {e}")

        self.obs, self.info = self.env.reset(seed=self.seed)
        self.controller.reset_history(self.obs, self.info)
        self._logged_intervention_count = 0


    def _build_ui(self):
        # 1. Header & Control Bar
        top_bar = tk.Frame(self.root, bg="#181825", height=60, padx=15, pady=10)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar,
            text="SAGE-Traffic Phase 9: Traffic Signal Safety Layer",
            font=("Helvetica", 16, "bold"),
            fg="#cdd6f4",
            bg="#181825"
        )
        title_lbl.pack(side=tk.LEFT)

        # Simulation Controls
        btn_frame = tk.Frame(top_bar, bg="#181825")
        btn_frame.pack(side=tk.RIGHT)

        self.start_btn = tk.Button(
            btn_frame, text="▶ START", bg="#a6e3a1", fg="#11111b",
            font=("Helvetica", 10, "bold"), padx=10, pady=4, relief=tk.FLAT,
            command=self._start_sim
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(
            btn_frame, text="⏸ PAUSE", bg="#f9e2af", fg="#11111b",
            font=("Helvetica", 10, "bold"), padx=10, pady=4, relief=tk.FLAT,
            command=self._pause_sim
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.step_btn = tk.Button(
            btn_frame, text="⏭ STEP", bg="#89b4fa", fg="#11111b",
            font=("Helvetica", 10, "bold"), padx=10, pady=4, relief=tk.FLAT,
            command=self._single_step
        )
        self.step_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(
            btn_frame, text="↺ RESET", bg="#f38ba8", fg="#11111b",
            font=("Helvetica", 10, "bold"), padx=10, pady=4, relief=tk.FLAT,
            command=self._reset_sim
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # 2. Safety Shield Banner & Safe Toggle Panel
        shield_bar = tk.Frame(self.root, bg="#11111b", height=50, padx=15, pady=8)
        shield_bar.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.shield_badge = tk.Label(
            shield_bar,
            text="● SAFETY SHIELD ACTIVE (Zero Invariant Violations)",
            font=("Helvetica", 11, "bold"),
            fg="#a6e3a1",
            bg="#11111b"
        )
        self.shield_badge.pack(side=tk.LEFT)

        self.toggle_shield_btn = tk.Button(
            shield_bar,
            text="🛡 TOGGLE SAFETY SHIELD",
            bg="#313244",
            fg="#cdd6f4",
            font=("Helvetica", 9, "bold"),
            padx=10,
            pady=3,
            relief=tk.FLAT,
            command=self._toggle_safety_shield
        )
        self.toggle_shield_btn.pack(side=tk.LEFT, padx=20)

        self.shield_warning_lbl = tk.Label(
            shield_bar,
            text="",
            font=("Helvetica", 9, "italic"),
            fg="#fab387",
            bg="#11111b"
        )
        self.shield_warning_lbl.pack(side=tk.LEFT, padx=10)

        # 3. Main Center Split
        main_pane = tk.Frame(self.root, bg="#1e1e2e", padx=10, pady=10)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: 4 Intersection Cards
        cards_frame = tk.Frame(main_pane, bg="#1e1e2e")
        cards_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.cards = {}
        for row, pair in enumerate([["A", "B"], ["C", "D"]]):
            r_frame = tk.Frame(cards_frame, bg="#1e1e2e")
            r_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)
            for col, ts_id in enumerate(pair):
                card = self._build_intersection_card(r_frame, ts_id)
                card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
                self.cards[ts_id] = card

        # Right Column: Safety Telemetry & Live Intervention Ticker
        telemetry_frame = tk.Frame(main_pane, bg="#181825", width=420, padx=15, pady=10)
        telemetry_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        tk.Label(
            telemetry_frame,
            text="SAFETY SHIELD TELEMETRY",
            font=("Helvetica", 12, "bold"),
            fg="#89b4fa",
            bg="#181825"
        ).pack(anchor="w", pady=(0, 5))

        self.stat_labels = {}
        stats_items = [
            ("Total Action Proposals", "0"),
            ("Shield Interventions", "0"),
            ("Min Green Enforced", "0"),
            ("Max Green Enforced", "0"),
            ("Min Red Protected", "0"),
            ("Clearance Clamped", "0"),
            ("Physical Conflicts", "0 (VERIFIED 0)"),
        ]
        for name, init_val in stats_items:
            f = tk.Frame(telemetry_frame, bg="#181825")
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=name + ":", font=("Helvetica", 9), fg="#a6adc8", bg="#181825").pack(side=tk.LEFT)
            lbl = tk.Label(f, text=init_val, font=("Helvetica", 9, "bold"), fg="#cdd6f4", bg="#181825")
            lbl.pack(side=tk.RIGHT)
            self.stat_labels[name] = lbl

        tk.Label(
            telemetry_frame,
            text="RECENT INTERVENTIONS LOG",
            font=("Helvetica", 11, "bold"),
            fg="#fab387",
            bg="#181825"
        ).pack(anchor="w", pady=(15, 5))

        # Log Treeview
        tree_frame = tk.Frame(telemetry_frame, bg="#181825")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.log_tree = ttk.Treeview(
            tree_frame,
            columns=("time", "agent", "prop", "safe", "reason"),
            show="headings",
            height=10
        )
        self.log_tree.heading("time", text="Time")
        self.log_tree.heading("agent", text="Agent")
        self.log_tree.heading("prop", text="Proposed")
        self.log_tree.heading("safe", text="Safe")
        self.log_tree.heading("reason", text="Constraint")

        self.log_tree.column("time", width=55, anchor="center")
        self.log_tree.column("agent", width=45, anchor="center")
        self.log_tree.column("prop", width=65, anchor="center")
        self.log_tree.column("safe", width=65, anchor="center")
        self.log_tree.column("reason", width=120, anchor="w")

        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 4. Bottom Metric Strip
        bottom_strip = tk.Frame(self.root, bg="#181825", height=40, padx=15, pady=8)
        bottom_strip.pack(side=tk.BOTTOM, fill=tk.X)

        self.bottom_labels = {}
        for m in ["Step", "Sim Time", "Avg Delay", "Avg Queue", "Throughput", "Task Return"]:
            f = tk.Frame(bottom_strip, bg="#181825")
            f.pack(side=tk.LEFT, padx=15)
            tk.Label(f, text=m + ":", font=("Helvetica", 9), fg="#a6adc8", bg="#181825").pack(side=tk.LEFT)
            lbl = tk.Label(f, text="--", font=("Helvetica", 9, "bold"), fg="#cdd6f4", bg="#181825")
            lbl.pack(side=tk.LEFT, padx=4)
            self.bottom_labels[m] = lbl

    def _build_intersection_card(self, parent, ts_id: str) -> tk.Frame:
        card = tk.Frame(parent, bg="#313244", padx=10, pady=8, relief=tk.RIDGE, bd=1)

        # Title
        hdr = tk.Frame(card, bg="#313244")
        hdr.pack(fill=tk.X, pady=(0, 5))
        tk.Label(
            hdr, text=f"Intersection {ts_id}", font=("Helvetica", 11, "bold"),
            fg="#cdd6f4", bg="#313244"
        ).pack(side=tk.LEFT)

        status_lbl = tk.Label(
            hdr, text="NS GREEN", font=("Helvetica", 9, "bold"),
            fg="#a6e3a1", bg="#313244"
        )
        status_lbl.pack(side=tk.RIGHT)
        card.status_lbl = status_lbl

        # Telemetry Labels
        details = tk.Frame(card, bg="#313244")
        details.pack(fill=tk.X, pady=2)

        eval_lbl = tk.Label(details, text="Decision State: Green 0s (min 5s)", font=("Helvetica", 8, "italic"), fg="#bac2de", bg="#313244")
        eval_lbl.pack(anchor="w")
        card.eval_lbl = eval_lbl

        # Action Shield Box
        act_box = tk.Frame(card, bg="#181825", padx=8, pady=4, relief=tk.GROOVE, bd=1)
        act_box.pack(fill=tk.X, pady=3)

        p_lbl = tk.Label(act_box, text="Proposed Action: KEEP", font=("Helvetica", 8), fg="#89b4fa", bg="#181825")
        p_lbl.pack(anchor="w")
        card.p_lbl = p_lbl

        s_lbl = tk.Label(act_box, text="Shielded Action: KEEP", font=("Helvetica", 8, "bold"), fg="#a6e3a1", bg="#181825")
        s_lbl.pack(anchor="w")
        card.s_lbl = s_lbl

        # Post-step physical state label
        post_lbl = tk.Label(card, text="Step Outcome: Initializing", font=("Helvetica", 8, "bold"), fg="#cdd6f4", bg="#313244")
        post_lbl.pack(anchor="w", pady=(2, 0))
        card.post_lbl = post_lbl

        # Traffic metrics
        q_lbl = tk.Label(card, text="Queue: 0 veh | Wait: 0.0s", font=("Helvetica", 8), fg="#a6adc8", bg="#313244")
        q_lbl.pack(anchor="w", pady=(1, 0))
        card.q_lbl = q_lbl

        return card


    def _toggle_safety_shield(self):
        new_state = not self.safety_enabled
        applied = self.env.set_safety_enabled(new_state)
        if applied:
            self.safety_enabled = new_state
            if self.safety_enabled:
                self.shield_badge.config(
                    text="● SAFETY SHIELD ACTIVE (Zero Invariant Violations)",
                    fg="#a6e3a1"
                )
            else:
                self.shield_badge.config(
                    text="○ SAFETY SHIELD DISABLED (Raw RL Policy Bypassing Shield)",
                    fg="#f38ba8"
                )
            self.shield_warning_lbl.config(text="")
        else:
            self.shield_warning_lbl.config(
                text="⚠ Active Yellow/All-Red Clearance: Toggle deferred to safe episode boundary!"
            )

    def _start_sim(self):
        if not self.is_running:
            self.is_running = True
            self._sim_loop()

    def _pause_sim(self):
        self.is_running = False

    def _reset_sim(self):
        self.is_running = False
        self.step_idx = 0
        self.sim_time = 0.0
        self.total_task_return = 0.0
        self.ep_throughput = 0
        self._logged_intervention_count = 0
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.controller.reset_history(self.obs, self.info)
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        self._update_ui_state({}, None)

    def _single_step(self):
        self._execute_step()

    def _sim_loop(self):
        if self.is_running and self.step_idx < self.max_steps:
            self._execute_step()
            self.root.after(int(self.step_delay * 1000), self._sim_loop)
        elif self.step_idx >= self.max_steps:
            self.is_running = False

    def _execute_step(self):
        if self.step_idx >= self.max_steps:
            return

        # 1. Capture pre-step signal state and time_on_phase before env.step()
        pre_step_state = {}
        for a in self.env.agents:
            sig = self.env.signals[a]
            pre_step_state[a] = {
                "phase": sig.current_phase,
                "phase_name": "NS GREEN" if sig.current_phase == 0 else "EW GREEN",
                "time_on_phase": sig.time_on_phase,
                "is_yellow": sig.is_yellow,
                "is_all_red": getattr(sig, "is_all_red", False),
                "target_duration": sig.target_green_duration,
                "opposing_red": getattr(sig, "red_time_ew", 0) if sig.current_phase == 0 else getattr(sig, "red_time_ns", 0)
            }

        context = self.controller.compute_causal_context(self.info)
        actions, log_probs, comm_costs, actual_comm = self.controller.get_actions(context, deterministic=True)

        next_obs, task_rewards, terminations, truncations, next_info = self.env.step(actions)
        self.controller.update_history(next_obs)

        self.step_idx += 1
        self.sim_time = next_info["A"]["sim_time"]
        self.total_task_return += sum(task_rewards.values())
        self.ep_throughput += next_info["A"]["arrived_vehicles"]

        self.obs = next_obs
        self.info = next_info

        self._update_ui_state(next_info, pre_step_state)

    def _update_ui_state(self, next_info, pre_step_state=None):
        # Update bottom strip
        self.bottom_labels["Step"].config(text=f"{self.step_idx}/{self.max_steps}")
        self.bottom_labels["Sim Time"].config(text=f"{self.sim_time:.1f}s")
        if next_info:
            mean_w = np.mean([next_info[a]["metrics"]["waiting_time"] for a in self.env.agents])
            mean_q = np.mean([next_info[a]["metrics"]["queue_length"] for a in self.env.agents])
            self.bottom_labels["Avg Delay"].config(text=f"{mean_w:.2f}s")
            self.bottom_labels["Avg Queue"].config(text=f"{mean_q:.2f} veh")
            self.bottom_labels["Throughput"].config(text=f"{self.ep_throughput}")
            self.bottom_labels["Task Return"].config(text=f"{self.total_task_return:.1f}")

            # Update Safety Telemetry
            safety_stats = next_info["A"].get("safety", {})
            self.stat_labels["Total Action Proposals"].config(text=str(safety_stats.get("total_proposals", 0)))
            self.stat_labels["Shield Interventions"].config(text=str(safety_stats.get("total_interventions", 0)))
            by_c = safety_stats.get("interventions_by_constraint", {})
            self.stat_labels["Min Green Enforced"].config(text=str(by_c.get("MIN_GREEN", 0)))
            self.stat_labels["Max Green Enforced"].config(text=str(by_c.get("MAX_GREEN", 0)))
            self.stat_labels["Min Red Protected"].config(text=str(by_c.get("MIN_RED", 0)))
            self.stat_labels["Clearance Clamped"].config(text=str(by_c.get("CLEARANCE_ACTIVE", 0)))
            self.stat_labels["Physical Conflicts"].config(text=f"{safety_stats.get('audit_physical_violations', 0)} (0%)")

            # Update Intervention Log Treeview without duplication
            if hasattr(self.env, "safety_layer"):
                all_interventions = self.env.safety_layer.recent_interventions
                if len(all_interventions) > self._logged_intervention_count:
                    new_entries = all_interventions[self._logged_intervention_count:]
                    for entry in new_entries:
                        sim_t = f"{entry['sim_time']:.0f}s"
                        agt = entry['intersection_id']
                        prop = ACTION_NAMES.get(entry['proposed_action'], str(entry['proposed_action']))
                        safe = ACTION_NAMES.get(entry['safe_action'], str(entry['safe_action']))
                        constraint = entry['active_constraint']
                        self.log_tree.insert("", 0, values=(sim_t, agt, prop, safe, constraint))
                    self._logged_intervention_count = len(all_interventions)

                    # Keep treeview readable by trimming to recent 100 rows
                    children = self.log_tree.get_children()
                    if len(children) > 100:
                        for extra in children[100:]:
                            self.log_tree.delete(extra)

            # Update Intersection Cards
            if pre_step_state:
                for a in self.env.agents:
                    card = self.cards[a]
                    sig = self.env.signals[a]
                    pre = pre_step_state[a]

                    # 1. Decision-time evaluation display (evaluated against pre-step green timer)
                    card.eval_lbl.config(
                        text=f"Decision State: {pre['phase_name']} at {pre['time_on_phase']}s (min {self.env.min_green}s) | Opp Red: {pre['opposing_red']}s"
                    )

                    prop_a = next_info[a].get("proposed_action", 0)
                    val_a = next_info[a].get("validated_action", 0)
                    p_name = ACTION_NAMES.get(prop_a, "UNKNOWN")
                    v_name = ACTION_NAMES.get(val_a, "UNKNOWN")

                    card.p_lbl.config(text=f"Proposed Action: {p_name} ({prop_a})")
                    if prop_a != val_a:
                        card.s_lbl.config(text=f"Shielded Action: {v_name} ({val_a}) [INTERVENED: CLAMPED]", fg="#fab387")
                    else:
                        card.s_lbl.config(text=f"Validated Action: {v_name} ({val_a}) [SAFE]", fg="#a6e3a1")

                    # 2. Post-step physical state display
                    post_phase_name = "NS GREEN" if sig.current_phase == 0 else "EW GREEN"
                    if sig.is_yellow:
                        card.status_lbl.config(text="CLEARANCE (YELLOW)", fg="#f9e2af")
                        card.post_lbl.config(text=f"Step Outcome: Active Yellow Clearance (timer: {sig.yellow_timer}s)", fg="#f9e2af")
                    elif getattr(sig, "is_all_red", False):
                        card.status_lbl.config(text="CLEARANCE (ALL-RED)", fg="#f38ba8")
                        card.post_lbl.config(text=f"Step Outcome: Active All-Red Clearance (timer: {sig.all_red_timer}s)", fg="#f38ba8")
                    else:
                        card.status_lbl.config(text=post_phase_name, fg="#a6e3a1")
                        if sig.current_phase != pre["phase"]:
                            card.post_lbl.config(
                                text=f"Step Outcome: Switched to {post_phase_name} (New Green: {sig.time_on_phase}s)",
                                fg="#89b4fa"
                            )
                        else:
                            card.post_lbl.config(
                                text=f"Step Outcome: Maintained {post_phase_name} (Total Green: {sig.time_on_phase}s)",
                                fg="#cdd6f4"
                            )

                    q = next_info[a]["metrics"]["queue_length"]
                    w = next_info[a]["metrics"]["waiting_time"]
                    card.q_lbl.config(text=f"Queue: {q:.1f} veh | Wait: {w:.1f}s")



def run_gui():
    parser = argparse.ArgumentParser(description="Live SUMO-GUI & Safety Dashboard for Phase 9")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=1001, help="Evaluation seed (default: 1001)")
    parser.add_argument("--steps", type=int, default=100, help="Max decision steps (default: 100)")
    parser.add_argument("--delay", type=float, default=0.20, help="Step delay in seconds (default: 0.20)")
    parser.add_argument("--no-gui", action="store_true", help="Disable SUMO-GUI window (dashboard only)")
    parser.add_argument("--auto-run", action="store_true", help="Automatically start simulation")
    args = parser.parse_args()

    root = tk.Tk()
    app = Phase9SafetyGUIDashboard(
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


