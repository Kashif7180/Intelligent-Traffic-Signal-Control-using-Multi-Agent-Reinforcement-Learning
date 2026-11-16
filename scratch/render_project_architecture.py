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

# Helper function to draw rounded boxes
def draw_box(x, y, w, h, title, subtitle=None, bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], lw=1.5, rx=0.8):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=bg, edgecolor=border, lw=lw, zorder=2)
    ax.add_patch(box)
    if title:
        ax.text(x + w/2, y + h - (1.8 if subtitle else h/2), title, color=title_col, fontsize=10, fontweight="bold", ha="center", va="center", zorder=3)
    if subtitle:
        ax.text(x + w/2, y + (h - 2.2)/2, subtitle, color=THEME["text"], fontsize=8, ha="center", va="center", zorder=3)
    return box

def draw_arrow(p1, p2, text=None, col=THEME["primary"], lw=2.0, style="->,head_length=6,head_width=4", rad=0.0):
    arrow = patches.FancyArrowPatch(p1, p2, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", color=col, lw=lw, zorder=4)
    ax.add_patch(arrow)
    if text:
        mid = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 + 1.2)
        ax.text(mid[0], mid[1], text, color=col, fontsize=8, fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=THEME["surface"], edgecolor="none"))

# Top Header Title
ax.text(50, 97.2, "Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning", color=THEME["primary"], fontsize=17, fontweight="bold", ha="center", va="center")
ax.text(50, 94.6, "End-to-End Neural Policy & System Execution Architecture (Temporal Transformer + Dual-Pathway GAT + CTDE + Safety Shield)", color=THEME["subtext"], fontsize=11, ha="center", va="center")
ax.plot([4, 96], [93.0, 93.0], color=THEME["border"], lw=1.2)

# ==========================================
# 1. SECTION: RAW OBSERVATION SPACE (Left: X=4 to 22)
# ==========================================
draw_box(4, 18, 18, 72, "1. OBSERVATION SPACE", bg=THEME["surface"], border=THEME["primary"], title_col=THEME["primary"], rx=1.2)
ax.text(13, 85.5, "Per-Intersection Local State\n$o_i(t) \\in \\mathbb{R}^{43}$ for $i \\in \\{A, B, C, D\\}$", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

# Incoming Lanes Details
draw_box(5.5, 47, 15, 35, "Lane Metrics (8 Lanes x 5)", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
lane_metrics_text = (
    "8 Incoming Traffic Lanes:\n"
    "• North, South, East, West\n"
    "• 2 Lanes per approach\n\n"
    "5 Continuous Features / Lane:\n"
    "1. Queue Length $Q$ [0-30 veh]\n"
    "2. Waiting Time $W$ [0-150s]\n"
    "3. Average Speed $v$ [0-14 m/s]\n"
    "4. Lane Occupancy $\\rho$ [0-1.0]\n"
    "5. Vehicle Count $N$ [0-35 veh]\n\n"
    "Total Lane Features: 40-D"
)
ax.text(13, 62, lane_metrics_text, color=THEME["text"], fontsize=8, ha="center", va="center")

# Intersection State Details
draw_box(5.5, 21, 15, 23, "Signal State (3-D)", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
sig_state_text = (
    "Current Phase (One-Hot):\n"
    "• Phase 0: NS Green [1, 0]\n"
    "• Phase 2: EW Green [0, 1]\n\n"
    "Phase Duration Target:\n"
    "• Elapsed Green Time $t_{\\text{green}}$ [0-60s]\n\n"
    "Total Vector: 40 + 3 = 43-D"
)
ax.text(13, 31, sig_state_text, color=THEME["text"], fontsize=8, ha="center", va="center")

# ==========================================
# 2. SECTION: TEMPORAL TRANSFORMER (X=26 to 43)
# ==========================================
draw_box(26, 18, 18, 72, "2. TEMPORAL PATHWAY", bg=THEME["surface"], border=THEME["mauve"], title_col=THEME["mauve"], rx=1.2)
ax.text(35, 85.5, "Sliding Temporal Window Buffer\n$K=4$ Frames: $[o(t-3), ..., o(t)]$", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

draw_box(27.5, 61, 15, 21, "Temporal Buffer $K=4$", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
buf_text = (
    "Input Tensor Shape:\n$[B, K=4, D_{\\text{in}}=43]$\n\n"
    "• Frame $t-3$: History lag 15s\n"
    "• Frame $t-2$: History lag 10s\n"
    "• Frame $t-1$: History lag 5s\n"
    "• Frame $t$: Current frame\n"
    "Zero Future Leakage [Causal]"
)
ax.text(35, 70, buf_text, color=THEME["text"], fontsize=8, ha="center", va="center")

draw_box(27.5, 24, 15, 33, "Temporal Transformer", bg=THEME["card"], border=THEME["border"], title_col=THEME["mauve"], rx=0.6)
trans_text = (
    "Transformer Architecture:\n"
    "• Linear Projection: $43 \\to 64$\n"
    "• Positional Encoding (Learned)\n"
    "• Multi-Head Self-Attention:\n"
    "  Heads = 4, $d_k = 16$\n"
    "  Causal Triangular Mask\n"
    "• Feed-Forward MLP: $64 \\to 128 \\to 64$\n"
    "• LayerNorm + Residual Add\n\n"
    "Output Node Embedding:\n"
    "$h_i(t) \\in \\mathbb{R}^{64}$ for each node"
)
ax.text(35, 38.5, trans_text, color=THEME["text"], fontsize=8, ha="center", va="center")

# Arrow from Obs to Temporal
draw_arrow((22, 54), (26, 54), "History Window", col=THEME["mauve"])

# ==========================================
# 3. SECTION: SPATIAL GAT & DUAL-PATHWAY (X=47 to 69)
# ==========================================
draw_box(47, 18, 22, 72, "3. SPATIAL GAT & DUAL GATING", bg=THEME["surface"], border=THEME["success"], title_col=THEME["success"], rx=1.2)
ax.text(58, 85.5, "Physical 2x2 Interaction Graph\n8 Directed Arterial Corridors", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

# Dual-Pathway Gating Boxes
draw_box(48.5, 62, 19, 20, "Pathway A: Learned Comm Gate", bg=THEME["card"], border=THEME["border"], title_col=THEME["success"], rx=0.6)
gate_text = (
    "Learned Gating Activation:\n"
    "$g_{ij} = \\sigma(W_g [h_i \\parallel h_j] + b_g) \\in [0, 1]$\n\n"
    "• $g_{ij} > 0.50 \\Rightarrow$ OPEN (Transmit Message)\n"
    "• $g_{ij} \\leq 0.50 \\Rightarrow$ PRUNED (Sparsify Link)\n"
    "Saves ~42% Redundant Bandwidth"
)
ax.text(58, 70.5, gate_text, color=THEME["text"], fontsize=8, ha="center", va="center")

draw_box(48.5, 39, 19, 20, "Pathway B: Counterfactual Causal", bg=THEME["card"], border=THEME["border"], title_col=THEME["peach"], rx=0.6)
causal_text = (
    "Causal Influence Sensitivity:\n"
    "$\\mathcal{C}_{j \\to i} = \\mathcal{D}_{\\mathrm{KL}}(\\pi_i(m_j) \\parallel \\pi_i(\\bar{m}_j))$\n\n"
    "• Evaluates policy shift under masked $\\bar{m}_j$\n"
    "• Causal Modulation: $\\phi(C_{ij}) = 1 + \\tanh(C_{ij})$\n"
    "Amplifies High-Impact Bottlenecks"
)
ax.text(58, 47.5, causal_text, color=THEME["text"], fontsize=8, ha="center", va="center")

draw_box(48.5, 21, 19, 15, "Graph Attention Aggregation", bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], rx=0.6)
gat_text = (
    "Contextualized Embedding Formulation:\n"
    "$z_i = \\sigma\\left(\\sum_{j \\in \\mathcal{N}_i} g_{ij} \\cdot \\alpha_{ij} \\cdot \\phi(C_{ij}) \\cdot W_v h_j\\right)$\n"
    "Outputs Spatial Embedding $z_i(t) \\in \\mathbb{R}^{64}$"
)
ax.text(58, 27, gat_text, color=THEME["text"], fontsize=8, ha="center", va="center")

# Arrow from Temporal to Spatial
draw_arrow((43, 54), (47, 54), "h_i(t) [64-D]", col=THEME["success"])

# ==========================================
# 4. SECTION: CTDE DECISION LAYER (X=73 to 96)
# ==========================================
draw_box(73, 52, 23, 38, "4. CTDE DECISION LAYER", bg=THEME["surface"], border=THEME["primary"], title_col=THEME["primary"], rx=1.2)
ax.text(84.5, 85.5, "Centralized Training with\nDecentralized Execution (MAPPO)", color=THEME["teal"], fontsize=8.5, fontweight="bold", ha="center")

# Decentralized Actor
draw_box(74.5, 54.5, 20, 28, "Decentralized Actor Policy", bg=THEME["card"], border=THEME["border"], title_col=THEME["primary"], rx=0.6)
actor_text = (
    "Actor Input: $[z_i(t) \\parallel e_i] \\in \\mathbb{R}^{68}$\n"
    "(Context $64\\text{-D}$ + Agent Identity $4\\text{-D}$)\n\n"
    "Shared MLP ($68 \\to 64 \\to 64 \\to 4$) + Softmax\n"
    "Discrete Action Distribution $\\pi_i(a_i \\mid z_i, e_i)$:\n"
    "• Action 0: KEEP (Hold Phase)\n"
    "• Action 1: SWITCH (Initiate Transition)\n"
    "• Action 2: EXTEND (+5s Green Target)\n"
    "• Action 3: REDUCE (-5s Green Target)"
)
ax.text(84.5, 67, actor_text, color=THEME["text"], fontsize=8, ha="center", va="center")

# Arrow from Spatial to Actor
draw_arrow((69, 68), (73, 68), "z_i [64-D]", col=THEME["primary"])

# ==========================================
# 5. SECTION: RUNTIME SAFETY SHIELD (X=73 to 96, Bottom)
# ==========================================
draw_box(73, 4, 23, 44, "5. RUNTIME SAFETY SHIELD", bg=THEME["surface"], border=THEME["success"], title_col=THEME["success"], rx=1.2)
ax.text(84.5, 43.5, "Phase 9 Deterministic Invariant Verifier\nGuarantees 0 Green Clashes & Physical Bounds", color=THEME["peach"], fontsize=8.5, fontweight="bold", ha="center")

shield_box = draw_box(74.5, 6.5, 20, 34, "4 Physical Safety Invariants", bg=THEME["card"], border=THEME["border"], title_col=THEME["warning"], rx=0.6)
shield_rules_text = (
    "1. Minimum Green Clamp:\n"
    "   If $t_{\\text{green}} < 5\\text{s} \\Rightarrow a^{\\text{exec}} = 0$ (KEEP)\n\n"
    "2. Maximum Green Enforcement:\n"
    "   If $t_{\\text{green}} \\geq 60\\text{s} \\Rightarrow a^{\\text{exec}} = 1$ (SWITCH)\n\n"
    "3. Clearance Lockout Protection:\n"
    "   During Yellow (3s) & All-Red (1s) $\\Rightarrow a=0$\n\n"
    "4. Minimum Red Enforcement:\n"
    "   Conflicting movement red $t_{\\text{red}} \\geq 5\\text{s}$\n\n"
    "Verified Execution Outcome:\n"
    "Safe Executed Actions $a_i^{\\text{exec}} \\to$ SUMO TraCI\n"
    "Result: 0 Conflicts / 110,000s Tested"
)
ax.text(84.5, 21.5, shield_rules_text, color=THEME["text"], fontsize=7.8, ha="center", va="center")

# Arrow from Actor to Safety Shield
draw_arrow((84.5, 54.5), (84.5, 48), "Proposed a_i", col=THEME["danger"], lw=2.2)

# Centralized Critic Box (Training Sidecar)
draw_box(47, 4, 22, 11, "Centralized Critic (Training Only)", bg=THEME["surface"], border=THEME["mauve"], title_col=THEME["mauve"], rx=0.8)
critic_text = (
    "Global State: $S = [z_A, z_B, z_C, z_D] \\in \\mathbb{R}^{256}$\n"
    "Centralized Value Function: $V(S) \\in \\mathbb{R}$\n"
    "Used during PPO Advantage Estimation only (CTDE)"
)
ax.text(58, 9, critic_text, color=THEME["subtext"], fontsize=7.8, ha="center", va="center")
draw_arrow((58, 21), (58, 15), "Global S [256-D]", col=THEME["mauve"], style="->,head_length=5,head_width=3")

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/10_project_architecture_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plots/10_project_architecture_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved project architecture diagram to report_images/10_project_architecture_diagram.png and plots/")
