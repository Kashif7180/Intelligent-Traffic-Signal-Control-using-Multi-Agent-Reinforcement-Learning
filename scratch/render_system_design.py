import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color Theme
THEME = {
    "bg": "#1e1e2e",
    "surface": "#181825",
    "card": "#24273a",
    "overlay": "#313244",
    "border": "#45475a",
    "text": "#cdd6f4",
    "subtext": "#a6adc8",
    "primary": "#89b4fa",   # Blue
    "success": "#a6e3a1",   # Green
    "warning": "#f9e2af",   # Yellow
    "danger": "#f38ba8",    # Red
    "peach": "#fab387",     # Orange
    "mauve": "#cba6f7",     # Purple
    "teal": "#94e2d5"       # Teal
}

fig = plt.figure(figsize=(22, 13), dpi=300)
fig.patch.set_facecolor(THEME["bg"])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(THEME["bg"])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Helper functions
def draw_box(x, y, w, h, title, subtitle=None, bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], lw=1.5, rx=0.8):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=bg, edgecolor=border, lw=lw, zorder=2)
    ax.add_patch(box)
    if title:
        ax.text(x + w/2, y + h - (1.8 if subtitle else h/2), title, color=title_col, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=3)
    if subtitle:
        ax.text(x + w/2, y + (h - 2.2)/2, subtitle, color=THEME["text"], fontsize=8, ha="center", va="center", zorder=3)
    return box

def draw_arrow(p1, p2, text=None, col=THEME["primary"], lw=1.8, style="->,head_length=5,head_width=3.5", rad=0.0):
    arrow = patches.FancyArrowPatch(p1, p2, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", color=col, lw=lw, zorder=4)
    ax.add_patch(arrow)
    if text:
        mid = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 + 1.2)
        ax.text(mid[0], mid[1], text, color=col, fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=THEME["surface"], edgecolor="none"))

# Top Master Title
ax.text(50, 97.2, "Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning", color=THEME["primary"], fontsize=17, fontweight="bold", ha="center", va="center")
ax.text(50, 94.6, "System Design & Software Engineering Architecture: Modular Component Integration, IPC Dataflow & Verification", color=THEME["subtext"], fontsize=11, ha="center", va="center")
ax.plot([4, 96], [93.0, 93.0], color=THEME["border"], lw=1.2)

# ==========================================
# 1. SUBSYSTEM: SIMULATION & NETWORK LAYER (Bottom-Left: X=4 to 26, Y=5 to 48)
# ==========================================
draw_box(4, 5, 23, 44, "1. SIMULATION & SUMO ENGINE", bg=THEME["surface"], border=THEME["primary"], title_col=THEME["primary"], rx=1.2)
ax.text(15.5, 45.5, "Eclipse SUMO Microscopic Engine\n2x2 Urban Grid & TraCI IPC Layer", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(5.5, 30, 20, 13, "Network & Topo Assets", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
net_text = (
    "• 2x2_grid.net.xml: 4 Nodes, 8 Links\n"
    "• simulation.sumocfg: XML Config\n"
    "• Topology Engine: network/topology.py\n"
    "  (4 Intersections, 8 Directed Corridors)"
)
ax.text(15.5, 35, net_text, color=THEME["text"], fontsize=7.5, ha="center", va="center")

draw_box(5.5, 17, 20, 11, "Demand & Route Generation", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
dem_text = (
    "• demand_generator.py: Poisson arrivals\n"
    "• 53 Pre-Generated Cached Route Files\n"
    "• Regimes: Low, Med, High, Surge, Stress"
)
ax.text(15.5, 21.5, dem_text, color=THEME["text"], fontsize=7.5, ha="center", va="center")

draw_box(5.5, 7, 20, 8, "TraCI Socket IPC & Cache", bg=THEME["card"], border=THEME["border"], title_col=THEME["success"], rx=0.6)
traci_text = "Optimized IPC (94.3 SPS, 320 socket queries saved)"
ax.text(15.5, 9.5, traci_text, color=THEME["text"], fontsize=7.2, ha="center", va="center")

# ==========================================
# 2. SUBSYSTEM: GYMNASIUM ENVIRONMENT INTERFACE (Top-Left: X=4 to 26, Y=51 to 91)
# ==========================================
draw_box(4, 51, 23, 40, "2. GYMNASIUM ENVIRONMENT", bg=THEME["surface"], border=THEME["mauve"], title_col=THEME["mauve"], rx=1.2)
ax.text(15.5, 87.5, "MultiAgentTrafficEnv Wrapper\nStandardized Agent-Environment Loop", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(5.5, 74, 20, 11, "TrafficSignal State Machine", bg=THEME["card"], border=THEME["border"], title_col=THEME["mauve"], rx=0.6)
sig_machine_text = (
    "• 4 Physical Signal Controllers (A, B, C, D)\n"
    "• Phase Timers & Targets (Green, Yellow, Red)\n"
    "• Clearance Progression: 3s Yellow + 1s All-Red"
)
ax.text(15.5, 78.5, sig_machine_text, color=THEME["text"], fontsize=7.5, ha="center", va="center")

draw_box(5.5, 62, 20, 10, "Feature Normalization", bg=THEME["card"], border=THEME["border"], title_col=THEME["mauve"], rx=0.6)
norm_text = (
    "• Vector Normalizer: $o_i \\in [0, 1]^{43}$\n"
    "• Queue (30), Wait (150s), Speed (13.89 m/s)"
)
ax.text(15.5, 66, norm_text, color=THEME["text"], fontsize=7.5, ha="center", va="center")

draw_box(5.5, 53, 20, 7.5, "Reward Calculator", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
rew_text = "$R_i = -(0.10 \\cdot Q_i + 0.05 \\cdot W_i + 0.02 \\cdot S_i)$"
ax.text(15.5, 55.5, rew_text, color=THEME["text"], fontsize=7.5, ha="center", va="center")

# Upward arrow from SUMO to Env
draw_arrow((15.5, 49), (15.5, 51), "TraCI Telemetry", col=THEME["primary"])

# ==========================================
# 3. SUBSYSTEM: MODEL ZOO & CONTROLLER ORCHESTRATION (Center: X=30 to 52, Y=5 to 91)
# ==========================================
draw_box(30, 5, 23, 86, "3. CONTROLLER & MODEL ZOO", bg=THEME["surface"], border=THEME["success"], title_col=THEME["success"], rx=1.2)
ax.text(41.5, 87.5, "11 Evaluated Architectures\nModel Registry & Inference Manager", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(31.5, 69, 20, 16, "Checkpoint Model Registry", bg=THEME["card"], border=THEME["border"], title_col=THEME["success"], rx=0.6)
ckpt_reg_text = (
    "20 Frozen Checkpoints (models/checkpoints/):\n"
    "• IPPO: agent_A.pt ... agent_D.pt\n"
    "• MAPPO: actor_shared.pt, critic.pt\n"
    "• Transformer: actor.pt, critic.pt\n"
    "• GAT: gat_actor.pt, gat_critic.pt\n"
    "• Comm-Gated: comm_gate_actor.pt ...\n"
    "• Causal-Gated: causal_gate_actor.pt ...\n"
    "• Dual-Pathway: causal_improved_*.pt"
)
ax.text(41.5, 76, ckpt_reg_text, color=THEME["text"], fontsize=7.2, ha="center", va="center")

draw_box(31.5, 47, 20, 20, "11 Controller Configurations", bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], rx=0.6)
ctrl_list_text = (
    "Primary Benchmarked Controllers:\n"
    "• C1: Fixed-Time Baseline (Periodic 30s)\n"
    "• C2: Phase 3 IPPO (Decentralized)\n"
    "• C3: Phase 4 Shared MAPPO\n"
    "• C4: Phase 5 Transformer-MAPPO\n"
    "• C5: Phase 6 Dense GAT-MAPPO\n"
    "• C6: Phase 7 Comm-Gated MAPPO\n"
    "• C7: Phase 8 Causal-Gated MAPPO\n"
    "• C8: Phase 8R Dual-Pathway (Unshielded)\n"
    "• C9: Phase 9 Shielded MARL Controller\n"
    "Ablations: A1 (Individual Actors), A2 (No Shield)"
)
ax.text(41.5, 56, ctrl_list_text, color=THEME["text"], fontsize=7.2, ha="center", va="center")

draw_box(31.5, 27, 20, 18, "History & Causal Engines", bg=THEME["card"], border=THEME["border"], title_col=THEME["peach"], rx=0.6)
hist_engine_text = (
    "• TemporalHistoryManager:\n"
    "  Sliding ring buffers [4, 43] per agent\n"
    "• Counterfactual Sensitivity Engine:\n"
    "  Marginal mean baseline calculation\n"
    "  Causal KL-Divergence computation\n"
    "• Dynamic Communication Router:\n"
    "  Sigmoid gating ($g_{ij} > 0.50$)"
)
ax.text(41.5, 35, hist_engine_text, color=THEME["text"], fontsize=7.2, ha="center", va="center")

draw_box(31.5, 7, 20, 18, "Multi-Agent Synchronizer", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
sync_text = (
    "• Synchronous 4-Agent Execution Loop\n"
    "• Step Inference Latency: 1.48 ms\n"
    "• Action Proposal Vector:\n"
    "  $a^{\\text{prop}} = [a_A, a_B, a_C, a_D]$\n"
    "• Dispatches proposals to Safety Shield"
)
ax.text(41.5, 15, sync_text, color=THEME["text"], fontsize=7.2, ha="center", va="center")

# Arrow from Env to Controller
draw_arrow((27, 71), (30, 71), "Obs Vector [43-D]", col=THEME["mauve"])

# ==========================================
# 4. SUBSYSTEM: RUNTIME SAFETY SHIELD & AUDITOR (Center-Right: X=56 to 74, Y=5 to 91)
# ==========================================
draw_box(56, 5, 19, 86, "4. SAFETY SHIELD LAYER", bg=THEME["surface"], border=THEME["danger"], title_col=THEME["danger"], rx=1.2)
ax.text(65.5, 87.5, "Phase 9 Safety Architecture\nDeterministic Invariant Enforcement", color=THEME["peach"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(57.5, 59, 16, 26, "Safety Invariant Engine", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
inv_text = (
    "Runtime Verification Rules:\n\n"
    "1. MIN_GREEN Clamp:\n"
    "   Blocks switch if $t < 5\\text{s}$\n\n"
    "2. MAX_GREEN Enforcement:\n"
    "   Forces switch if $t \\geq 60\\text{s}$\n\n"
    "3. CLEARANCE Lockout:\n"
    "   Clamps action during yellow\n\n"
    "4. MIN_RED Guarantee:\n"
    "   Protects opposing movement"
)
ax.text(65.5, 71, inv_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

draw_box(57.5, 33, 16, 24, "Action Validation Filter", bg=THEME["card"], border=THEME["border"], title_col=THEME["success"], rx=0.6)
filter_text = (
    "Deterministic Decision Filter:\n\n"
    "• IF safe $\\Rightarrow a^{\\text{exec}} = a^{\\text{prop}}$\n"
    "• IF unsafe $\\Rightarrow a^{\\text{exec}} = a^{\\text{safe}}$\n\n"
    "Intervention Logging:\n"
    "• Records timestamp, intersection,\n"
    "  proposed action, clamped action,\n"
    "  and constraint justification"
)
ax.text(65.5, 44, filter_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

draw_box(57.5, 7, 16, 24, "TraCI Physical Signal Auditor", bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], rx=0.6)
audit_box_text = (
    "TraCI Physical Invariant Check:\n\n"
    "• Queries live phase string:\n"
    "  e.g. 'GGGggrrrrrGGGggrrrrr'\n"
    "• Confirms orthogonal greens = 0\n"
    "• Verification Status:\n"
    "  100% Zero Conflict Guarantee\n"
    "• Sends validated actions to SUMO"
)
ax.text(65.5, 18, audit_box_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

# Arrow from Controller to Safety Shield
draw_arrow((53, 44), (56, 44), "Proposed a_i", col=THEME["danger"])
# Feedback arrow from Safety Shield back to SUMO
draw_arrow((65.5, 5), (15.5, 5), "Validated Action a_i^{exec} -> TraCI TrafficLight API", col=THEME["success"], rad=-0.15)

# ==========================================
# 5. SUBSYSTEM: EXPLAINABLE AI (XAI) & PRESENTATION (Right: X=78 to 96, Y=5 to 91)
# ==========================================
draw_box(78, 5, 18, 86, "5. XAI & ANALYTICS LAYER", bg=THEME["surface"], border=THEME["teal"], title_col=THEME["teal"], rx=1.2)
ax.text(87, 87.5, "Explainable AI (XAI) Engine\nTelemetry & Presentation Dashboards", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(79.5, 64, 15, 21, "Modular XAI Framework", bg=THEME["card"], border=THEME["border"], title_col=THEME["teal"], rx=0.6)
xai_framework_text = (
    "xai/ Subsystem:\n"
    "• DecisionExplainer: Margins\n"
    "• AttentionExplainer: $\\alpha_{ij}$\n"
    "• CommExplainer: $g_{ij}$ gates\n"
    "• CausalExplainer: $\\mathcal{C}_{j \\to i}$\n"
    "• SafetyExplainer: Invariants\n"
    "• XAILogger: JSONL Logs"
)
ax.text(87, 73.5, xai_framework_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

draw_box(79.5, 42, 15, 20, "Unified Tkinter Dashboard", bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], rx=0.6)
gui_text = (
    "visualize_phase10_gui.py:\n"
    "• Live 2x2 Interaction Canvas\n"
    "• 4 Node Status Cards\n"
    "• Live XAI Attribution HUD\n"
    "• Safety Invariant Badges\n"
    "• Dynamic Controller Switcher"
)
ax.text(87, 51, gui_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

draw_box(79.5, 23, 15, 17, "Automated Benchmarks", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
bench_text = (
    "• Phase 10: 110-Episode Master\n"
    "• Phase 12: Demand Scaling\n"
    "• Phase 12: Dynamic Traffic Surge\n"
    "• Phase 12: Sensor Noise Bench\n"
    "• Phase 13: Edge-Case Stress"
)
ax.text(87, 30.5, bench_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

draw_box(79.5, 7, 15, 14, "CI Verification Suite", bg=THEME["card"], border=THEME["border"], title_col=THEME["success"], rx=0.6)
ci_text = (
    "6 Comprehensive Test Suites:\n"
    "• Integration & XAI (31 Tests)\n"
    "• Safety & GAT (31 Tests)\n"
    "• End-to-End & Stress (29 Tests)\n"
    "Result: 91/91 Passed (100%)"
)
ax.text(87, 13, ci_text, color=THEME["text"], fontsize=7.3, ha="center", va="center")

# Arrow from Safety Shield to XAI
draw_arrow((75, 44), (78, 44), "Events & States", col=THEME["teal"])

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/11_system_design_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plots/11_system_design_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved system design diagram to report_images/11_system_design_diagram.png and plots/")
