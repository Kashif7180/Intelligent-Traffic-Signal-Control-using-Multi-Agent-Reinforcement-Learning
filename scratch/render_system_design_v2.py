import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Professional Publication Theme
THEME = {
    "bg": "#f8fafc",           # Light slate background
    "card_bg": "#ffffff",      # Pure white cards
    "card_border": "#cbd5e1",  # Soft gray border
    "header_text": "#0f172a",  # Deep navy slate
    "sub_text": "#475569",     # Slate gray
    "body_text": "#1e293b",    # Dark readable text
    "blue": "#2563eb",         # Primary blue
    "blue_bg": "#eff6ff",      # Light blue fill
    "green": "#16a34a",        # Success green
    "green_bg": "#f0fdf4",     # Light green fill
    "purple": "#7c3aed",       # Purple accent
    "purple_bg": "#f5f3ff",    # Light purple fill
    "amber": "#d97706",        # Amber/Orange
    "amber_bg": "#fffbeb",     # Light amber fill
    "red": "#dc2626",          # Safety Red
    "red_bg": "#fef2f2",       # Light red fill
    "teal": "#0d9488",         # Teal
    "teal_bg": "#f0fdfa"       # Light teal fill
}

fig = plt.figure(figsize=(26, 15), dpi=300)
fig.patch.set_facecolor(THEME["bg"])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(THEME["bg"])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Helper to draw clean modern card
def draw_card(x, y, w, h, title, subtitle=None, header_col=THEME["blue"], bg_col=THEME["card_bg"], border_col=THEME["card_border"], lw=2.0, rx=1.2):
    shadow = patches.FancyBboxPatch((x+0.3, y-0.4), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor="#e2e8f0", edgecolor="none", zorder=1)
    ax.add_patch(shadow)
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=bg_col, edgecolor=border_col, lw=lw, zorder=2)
    ax.add_patch(card)
    if title:
        header_bar = patches.FancyBboxPatch((x, y + h - 4.5), w, 4.5, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=header_col, edgecolor="none", zorder=3)
        ax.add_patch(header_bar)
        ax.text(x + w/2, y + h - 2.2, title, color="#ffffff", fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
        if subtitle:
            ax.text(x + w/2, y + h - 6.2, subtitle, color=THEME["sub_text"], fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)
    return card

# Helper for clean labeled connection arrows
def draw_arrow(p1, p2, label=None, col=THEME["blue"], lw=2.8, rad=0.0):
    arrow = patches.FancyArrowPatch(p1, p2, arrowstyle="->,head_length=8,head_width=5", connectionstyle=f"arc3,rad={rad}", color=col, lw=lw, zorder=5)
    ax.add_patch(arrow)
    if label:
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2 + 1.6
        ax.text(mid_x, mid_y, label, color=col, fontsize=10, fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor=col, lw=1.2))

# Top Title
ax.text(50, 97.2, "Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning", 
        color=THEME["header_text"], fontsize=20, fontweight="bold", ha="center", va="center")
ax.text(50, 94.4, "System Design & Software Engineering Architecture: Layered Subsystems, IPC Dataflow & Verification Framework", 
        color=THEME["sub_text"], fontsize=12.5, fontweight="bold", ha="center", va="center")
ax.plot([3, 97], [92.5, 92.5], color=THEME["card_border"], lw=2.0)

# =========================================================================
# TIER 1: USER INTERACTION & ANALYTICS LAYER (Top Row: Y=74 to 90)
# =========================================================================
tier1_card = draw_card(3, 73, 94, 18, "TIER 1: PRESENTATION, GUI & BENCHMARKING SUBSYSTEM", 
                       "Interactive Desktop Dashboard, Real-Time Microscopic Viewport & Verification Suites", header_col=THEME["blue"])

# 1A: Tkinter GUI Dashboard
b1a = patches.FancyBboxPatch((4.5, 74.5), 29.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["blue_bg"], edgecolor=THEME["blue"], lw=1.2, zorder=3)
ax.add_patch(b1a)
ax.text(5.5, 83.5, "Interactive Tkinter GUI Dashboard", color=THEME["blue"], fontsize=11, fontweight="bold")
gui_bullets = [
    "• visualize_phase10_gui.py with Catppuccin HUD",
    "• 2x2 Network Topology Canvas with Live Signals",
    "• 4 Node Status Cards: Decision State & Telemetry",
    "• Dynamic Runtime Controller Switching (C1 - C9)"
]
for i, b in enumerate(gui_bullets):
    ax.text(5.5, 81.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 1B: SUMO-GUI Viewport
b1b = patches.FancyBboxPatch((35.5, 74.5), 28.0, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["blue_bg"], edgecolor=THEME["blue"], lw=1.2, zorder=3)
ax.add_patch(b1b)
ax.text(36.5, 83.5, "SUMO-GUI Microscopic Viewport", color=THEME["blue"], fontsize=11, fontweight="bold")
sumo_bullets = [
    "• Real-Time Vehicle Kinematics & Flow",
    "• Lane-Level Queueing & Discharging Behavior",
    "• Stopbar Traffic Light Head Signal Display",
    "• Synchronous Step Visualization (50ms Delay)"
]
for i, b in enumerate(sumo_bullets):
    ax.text(36.5, 81.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 1C: Benchmarking & CI
b1c = patches.FancyBboxPatch((65.0, 74.5), 30.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["blue_bg"], edgecolor=THEME["blue"], lw=1.2, zorder=3)
ax.add_patch(b1c)
ax.text(66.0, 83.5, "Benchmark & Verification Engine", color=THEME["blue"], fontsize=11, fontweight="bold")
bench_bullets = [
    "• Phase 10 Master Benchmark (110 Episodes)",
    "• Phase 12 Demand Scaling & Dynamic Surge Tests",
    "• Phase 13 Edge-Case & Gridlock Stress Suites",
    "• Continuous Integration: 91/91 Passed (100%)"
]
for i, b in enumerate(bench_bullets):
    weight = "bold" if "91/91" in b else "normal"
    col = THEME["blue"] if "91/91" in b else THEME["body_text"]
    ax.text(66.0, 81.2 - i*2.2, b, color=col, fontsize=9.2, fontweight=weight, zorder=4)

# Downward Arrow from Tier 1 to Tier 2
draw_arrow((50, 73), (50, 68), "Telemetry & Telemetry Feeds", col=THEME["blue"])

# =========================================================================
# TIER 2: EXPLAINABLE AI (XAI) LAYER (Middle-Upper: Y=50 to 67)
# =========================================================================
tier2_card = draw_card(3, 50, 94, 18, "TIER 2: EXPLAINABLE AI (XAI) & INTERPRETABILITY SUBSYSTEM", 
                       "Feature Attribution, Spatial Attention Routing, Gated Communication & Causal Sensitivity", header_col=THEME["teal"])

# 2A: Decision Explainer
b2a = patches.FancyBboxPatch((4.5, 51.5), 29.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["teal_bg"], edgecolor=THEME["teal"], lw=1.2, zorder=3)
ax.add_patch(b2a)
ax.text(5.5, 60.5, "Decision Explainer (xai/decision_explainer.py)", color=THEME["teal"], fontsize=10.5, fontweight="bold")
dec_bullets = [
    "• Policy Action Probability Distribution pi_i(a_i)",
    "• Action Selection Confidence Margin (+48.6%)",
    "• Policy Entropy Tracking (H = 0.9124)",
    "• Natural Language Decision Attribution Grounding"
]
for i, b in enumerate(dec_bullets):
    ax.text(5.5, 58.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 2B: Attention & Comm Explainer
b2b = patches.FancyBboxPatch((35.5, 51.5), 28.0, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["teal_bg"], edgecolor=THEME["teal"], lw=1.2, zorder=3)
ax.add_patch(b2b)
ax.text(36.5, 60.5, "Relational Attention & Gate Explainer", color=THEME["teal"], fontsize=10.5, fontweight="bold")
rel_bullets = [
    "• Spatial Graph Attention Matrices (alpha_ij)",
    "• Communication Channel Gates (g_ij > 0.50)",
    "• Message Transmission Counting & Pruning Rates",
    "• Dual-Pathway Correlation Synthesis (R^2 = 0.96)"
]
for i, b in enumerate(rel_bullets):
    ax.text(36.5, 58.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 2C: Causal & Safety Logger
b2c = patches.FancyBboxPatch((65.0, 51.5), 30.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["teal_bg"], edgecolor=THEME["teal"], lw=1.2, zorder=3)
ax.add_patch(b2c)
ax.text(66.0, 60.5, "Causal Sensitivity & Event Logger", color=THEME["teal"], fontsize=10.5, fontweight="bold")
causal_bullets = [
    "• Counterfactual Sensitivity Engine: C_j->i",
    "• Marginal Mean Perturbation Baseline (m_bar_j)",
    "• Structured JSONL Audit Logger (XAILogger)",
    "• Epistemological Distinction & Verification"
]
for i, b in enumerate(causal_bullets):
    ax.text(66.0, 58.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# Downward Arrow from Tier 2 to Tier 3
draw_arrow((50, 50), (50, 45), "Policy Actions & Embeddings", col=THEME["teal"])

# =========================================================================
# TIER 3: MULTI-AGENT CONTROLLERS & MODEL ZOO (Middle-Lower: Y=27 to 44)
# =========================================================================
tier3_card = draw_card(3, 27, 94, 18, "TIER 3: MULTI-AGENT CONTROL & MODEL ORCHESTRATION LAYER", 
                       "20 Frozen Checkpoints, 11 Model Architectures & Temporal Ring Buffers", header_col=THEME["purple"])

# 3A: Model Checkpoint Registry
b3a = patches.FancyBboxPatch((4.5, 28.5), 29.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["purple_bg"], edgecolor=THEME["purple"], lw=1.2, zorder=3)
ax.add_patch(b3a)
ax.text(5.5, 37.5, "Frozen Model Checkpoint Registry", color=THEME["purple"], fontsize=10.5, fontweight="bold")
ckpt_bullets = [
    "• 20 Pre-Trained Model Checkpoints (.pt)",
    "• Cryptographic SHA256 Integrity Verified",
    "• Storage: models/checkpoints/",
    "• Architectures: IPPO, MAPPO, GAT, Causal, C9"
]
for i, b in enumerate(ckpt_bullets):
    ax.text(5.5, 35.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 3B: Controller Orchestrator
b3b = patches.FancyBboxPatch((35.5, 28.5), 28.0, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["purple_bg"], edgecolor=THEME["purple"], lw=1.2, zorder=3)
ax.add_patch(b3b)
ax.text(36.5, 37.5, "Multi-Controller Orchestration", color=THEME["purple"], fontsize=10.5, fontweight="bold")
orch_bullets = [
    "• 11 Evaluated Controllers (C1 to C9 + A1, A2)",
    "• Synchronous Multi-Agent Decision Loop",
    "• Step Inference Latency: 1.48 ms across 4 Agents",
    "• Centralized Training with Decentralized Exec"
]
for i, b in enumerate(orch_bullets):
    ax.text(36.5, 35.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 3C: Temporal & History Buffers
b3c = patches.FancyBboxPatch((65.0, 28.5), 30.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["purple_bg"], edgecolor=THEME["purple"], lw=1.2, zorder=3)
ax.add_patch(b3c)
ax.text(66.0, 37.5, "History Buffers & Causal Masking", color=THEME["purple"], fontsize=10.5, fontweight="bold")
hist_bullets = [
    "• TemporalHistoryManager: Ring Buffer [4, 43]",
    "• Rolling Multi-Step History per Intersection",
    "• Dual-Pathway Gating: Sigmoid & Causal Masking",
    "• Zero Future Information Leakage [Verified]"
]
for i, b in enumerate(hist_bullets):
    ax.text(66.0, 35.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# Downward Arrow from Tier 3 to Tier 4
draw_arrow((50, 27), (50, 22), "Proposed Actions a_i^prop", col=THEME["purple"])

# =========================================================================
# TIER 4: SIMULATION, SAFETY SHIELD & SUMO ENGINE (Bottom Row: Y=4 to 21)
# =========================================================================
tier4_card = draw_card(3, 4, 94, 18, "TIER 4: SAFETY SHIELD, GYMNASIUM ENVIRONMENT & SUMO ENGINE", 
                       "Deterministic Invariant Verification, State Machine & Microscopic Traffic Simulation", header_col=THEME["green"])

# 4A: Safety Shield Layer
b4a = patches.FancyBboxPatch((4.5, 5.5), 29.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["red_bg"], edgecolor=THEME["red"], lw=1.2, zorder=3)
ax.add_patch(b4a)
ax.text(5.5, 14.5, "Runtime Safety Shield (Phase 9)", color=THEME["red"], fontsize=10.5, fontweight="bold")
safety_bullets = [
    "• 4 Deterministic Invariants Enforced:",
    "  - Min Green (< 5s), Max Green (>= 60s)",
    "  - Clearance Lockout (3s Yellow + 1s All-Red)",
    "• TraCI Conflict Audit: 0 Conflicts / 110,000s"
]
for i, b in enumerate(safety_bullets):
    weight = "bold" if "TraCI Conflict" in b else "normal"
    col = THEME["red"] if "TraCI Conflict" in b else THEME["body_text"]
    ax.text(5.5, 12.2 - i*2.2, b, color=col, fontsize=9.2, fontweight=weight, zorder=4)

# 4B: Gymnasium Environment Interface
b4b = patches.FancyBboxPatch((35.5, 5.5), 28.0, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["green_bg"], edgecolor=THEME["green"], lw=1.2, zorder=3)
ax.add_patch(b4b)
ax.text(36.5, 14.5, "Gymnasium Environment (traffic_env.py)", color=THEME["green"], fontsize=10.5, fontweight="bold")
env_bullets = [
    "• MultiAgentTrafficEnv Standardized API",
    "• Signal Timers & Target Phase Tracking",
    "• 43-D Feature Normalization Engine",
    "• Reward: R_i = -(0.1 Q + 0.05 W + 0.02 S)"
]
for i, b in enumerate(env_bullets):
    ax.text(36.5, 12.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# 4C: SUMO Simulation & TraCI
b4c = patches.FancyBboxPatch((65.0, 5.5), 30.5, 11.5, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["green_bg"], edgecolor=THEME["green"], lw=1.2, zorder=3)
ax.add_patch(b4c)
ax.text(66.0, 14.5, "SUMO Microscopic Engine & TraCI", color=THEME["green"], fontsize=10.5, fontweight="bold")
sumo_eng_bullets = [
    "• 2x2 Grid Road Network (4 Junctions, 8 Links)",
    "• 53 Deterministic Cached Route XML Files",
    "• Optimized Socket IPC (94.3 SPS Throughput)",
    "• Telemetry: Speed, Waiting, Queue, Counts"
]
for i, b in enumerate(sumo_eng_bullets):
    ax.text(66.0, 12.2 - i*2.2, b, color=THEME["body_text"], fontsize=9.2, zorder=4)

# Upward Feedback arrow from SUMO to Env
draw_arrow((65.0, 11.5), (34.0, 11.5), "Step State", col=THEME["green"])

# Global Bottom Caption
ax.text(50, 1.6, "Figure 11: Modular System Architecture & Dataflow — Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning",
        color=THEME["sub_text"], fontsize=11, fontweight="bold", ha="center", va="center")

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/11_system_design_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plots/11_system_design_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved redesigned system design diagram to report_images/11_system_design_diagram.png and plots/")
