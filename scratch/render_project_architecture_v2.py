import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Professional Publication Theme (Clean Light Background for Crystal-Clear Report Printing)
# Off-white background with crisp navy/slate cards, vibrant accent badges, and high-contrast text
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
    # Shadow effect
    shadow = patches.FancyBboxPatch((x+0.3, y-0.4), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor="#e2e8f0", edgecolor="none", zorder=1)
    ax.add_patch(shadow)
    # Main card
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=bg_col, edgecolor=border_col, lw=lw, zorder=2)
    ax.add_patch(card)
    # Header bar inside card
    if title:
        header_bar = patches.FancyBboxPatch((x, y + h - 4.5), w, 4.5, boxstyle=f"round,pad=0.2,rounding_size={rx}", facecolor=header_col, edgecolor="none", zorder=3)
        ax.add_patch(header_bar)
        ax.text(x + w/2, y + h - 2.2, title, color="#ffffff", fontsize=13, fontweight="bold", ha="center", va="center", zorder=4)
        if subtitle:
            ax.text(x + w/2, y + h - 6.2, subtitle, color=THEME["sub_text"], fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)
    return card

# Helper for clean labeled connection arrows
def draw_arrow(p1, p2, label=None, sublabel=None, col=THEME["blue"], lw=3.0, rad=0.0):
    arrow = patches.FancyArrowPatch(p1, p2, arrowstyle="->,head_length=8,head_width=5", connectionstyle=f"arc3,rad={rad}", color=col, lw=lw, zorder=5)
    ax.add_patch(arrow)
    if label:
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2 + 1.8
        ax.text(mid_x, mid_y, label, color=col, fontsize=10.5, fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor=col, lw=1.2))
        if sublabel:
            ax.text(mid_x, mid_y - 2.0, sublabel, color=THEME["sub_text"], fontsize=8.5, ha="center", va="center", zorder=6)

# =========================================================================
# TOP MASTER HEADER
# =========================================================================
ax.text(50, 97.2, "Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning", 
        color=THEME["header_text"], fontsize=20, fontweight="bold", ha="center", va="center")
ax.text(50, 94.4, "End-to-End Neural Policy Architecture: Temporal Transformer + Dual-Pathway GAT + CTDE + Safety Shield", 
        color=THEME["sub_text"], fontsize=12.5, fontweight="bold", ha="center", va="center")
ax.plot([3, 97], [92.5, 92.5], color=THEME["card_border"], lw=2.0)

# =========================================================================
# STAGE 1: OBSERVATION SPACE (Left Column: X=3 to 22, Y=18 to 90)
# =========================================================================
draw_card(3, 18, 19, 72, "1. OBSERVATION SPACE", "Per-Intersection State: 43-D Vector", header_col=THEME["blue"], bg_col="#ffffff")

# Section 1A: 8 Lanes Features
y_cursor = 81.0
ax.text(4.5, y_cursor, "A. 8 Incoming Traffic Lanes", color=THEME["blue"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
lane_box = patches.FancyBboxPatch((4.5, y_cursor - 24.0), 16.0, 24.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["blue_bg"], edgecolor=THEME["blue"], lw=1.2, zorder=3)
ax.add_patch(lane_box)

lane_items = [
    "• 4 Approaches (N, S, E, W)",
    "• 2 Lanes per approach = 8 Lanes",
    "",
    "5 Continuous Metrics / Lane:",
    "  1. Queue Length (0 - 30 veh)",
    "  2. Waiting Time (0 - 150 s)",
    "  3. Average Speed (0 - 14 m/s)",
    "  4. Lane Occupancy (0.0 - 1.0)",
    "  5. Vehicle Count (0 - 35 veh)",
    "",
    "Total Lane Features: 40-D"
]
for item in lane_items:
    weight = "bold" if ("5 Continuous" in item or "Total" in item or "• 4" in item) else "normal"
    col = THEME["blue"] if "Total" in item else THEME["body_text"]
    ax.text(5.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.2

# Section 1B: Intersection Signal State
y_cursor -= 4.0
ax.text(4.5, y_cursor, "B. Signal Timing State", color=THEME["blue"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
sig_box = patches.FancyBboxPatch((4.5, y_cursor - 16.0), 16.0, 16.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["blue_bg"], edgecolor=THEME["blue"], lw=1.2, zorder=3)
ax.add_patch(sig_box)

sig_items = [
    "Current Phase (One-Hot, 2-D):",
    "  • [1, 0]: Phase 0 (NS Green)",
    "  • [0, 1]: Phase 2 (EW Green)",
    "",
    "Phase Duration Target (1-D):",
    "  • Elapsed Green Time (0 - 60s)",
    "",
    "Total Observation: 43-D"
]
for item in sig_items:
    weight = "bold" if ("Total" in item or "Current" in item or "Phase Duration" in item) else "normal"
    col = THEME["blue"] if "Total" in item else THEME["body_text"]
    ax.text(5.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.0

# Bottom Badge
badge_obs = patches.FancyBboxPatch((4.5, 20.0), 16.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#dbeafe", edgecolor=THEME["blue"], lw=1.5, zorder=3)
ax.add_patch(badge_obs)
ax.text(12.5, 22.5, "Input: o_i(t) in R^43", color=THEME["blue"], fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

# =========================================================================
# STAGE 2: TEMPORAL TRANSFORMER (Column 2: X=27 to 46, Y=18 to 90)
# =========================================================================
draw_card(27, 18, 19, 72, "2. TEMPORAL TRANSFORMER", "Sliding History Window (K=4)", header_col=THEME["purple"], bg_col="#ffffff")

# Arrow from Obs to Temporal
draw_arrow((22, 54), (27, 54), "43-D Stream", "Raw Features", col=THEME["blue"])

y_cursor = 81.0
ax.text(28.5, y_cursor, "A. Sliding History Buffer", color=THEME["purple"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
buf_box = patches.FancyBboxPatch((28.5, y_cursor - 18.0), 16.0, 18.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["purple_bg"], edgecolor=THEME["purple"], lw=1.2, zorder=3)
ax.add_patch(buf_box)

buf_items = [
    "Sliding Window Size: K = 4",
    "Tensor Shape: [Batch, 4, 43]",
    "",
    "• Frame t-3 : Lag 15s history",
    "• Frame t-2 : Lag 10s history",
    "• Frame t-1 : Lag 5s history",
    "• Frame t   : Current observation",
    "",
    "Causal Mask: Zero Future Leakage"
]
for item in buf_items:
    weight = "bold" if ("Sliding" in item or "Tensor" in item or "Causal" in item) else "normal"
    col = THEME["purple"] if "Causal" in item else THEME["body_text"]
    ax.text(29.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.0

# Transformer Architecture
y_cursor -= 4.0
ax.text(28.5, y_cursor, "B. Multi-Head Attention Block", color=THEME["purple"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
trans_box = patches.FancyBboxPatch((28.5, y_cursor - 22.0), 16.0, 22.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["purple_bg"], edgecolor=THEME["purple"], lw=1.2, zorder=3)
ax.add_patch(trans_box)

trans_items = [
    "1. Linear Projection: 43 -> 64-D",
    "2. Learned Positional Encoding",
    "3. Multi-Head Self-Attention:",
    "   • Heads = 4, d_k = 16",
    "   • Causal Attention Mask",
    "4. Feed-Forward MLP Layer:",
    "   • Linear(64 -> 128 -> 64)",
    "5. LayerNorm + Residual Add",
    "",
    "Output: h_i(t) in R^64 (Temporal)"
]
for item in trans_items:
    weight = "bold" if ("Output" in item or "Multi-Head" in item or "Feed-Forward" in item) else "normal"
    col = THEME["purple"] if "Output" in item else THEME["body_text"]
    ax.text(29.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.2

# Bottom Badge
badge_trans = patches.FancyBboxPatch((28.5, 20.0), 16.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#ede9fe", edgecolor=THEME["purple"], lw=1.5, zorder=3)
ax.add_patch(badge_trans)
ax.text(36.5, 22.5, "Node Embedding: h_i in R^64", color=THEME["purple"], fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

# =========================================================================
# STAGE 3: SPATIAL GAT & DUAL-PATHWAY (Column 3: X=51 to 72, Y=18 to 90)
# =========================================================================
draw_card(51, 18, 21, 72, "3. SPATIAL GAT & DUAL-PATHWAY", "8 Directed Corridors (A, B, C, D)", header_col=THEME["teal"], bg_col="#ffffff")

# Arrow from Temporal to Spatial
draw_arrow((46, 54), (51, 54), "h_i(t) [64-D]", "Node Features", col=THEME["purple"])

y_cursor = 81.0
ax.text(52.5, y_cursor, "Pathway 1: Communication Gate (g_ij)", color=THEME["teal"], fontsize=10.5, fontweight="bold")
y_cursor -= 3.0
p1_box = patches.FancyBboxPatch((52.5, y_cursor - 15.0), 18.0, 15.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["teal_bg"], edgecolor=THEME["teal"], lw=1.2, zorder=3)
ax.add_patch(p1_box)

p1_items = [
    "Learned Gating Activation:",
    "  g_ij = sigmoid(W_g [h_i || h_j] + b)",
    "",
    "• g_ij > 0.50 : OPEN (Transmit)",
    "• g_ij <= 0.50 : PRUNED (Quiet)",
    "",
    "Bandwidth Reduction: 42% Messages Pruned"
]
for item in p1_items:
    weight = "bold" if ("Bandwidth" in item or "Learned" in item or "OPEN" in item) else "normal"
    col = THEME["teal"] if "Bandwidth" in item else THEME["body_text"]
    ax.text(53.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.1

y_cursor -= 4.0
ax.text(52.5, y_cursor, "Pathway 2: Causal Sensitivity (C_ij)", color=THEME["amber"], fontsize=10.5, fontweight="bold")
y_cursor -= 3.0
p2_box = patches.FancyBboxPatch((52.5, y_cursor - 15.0), 18.0, 15.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["amber_bg"], edgecolor=THEME["amber"], lw=1.2, zorder=3)
ax.add_patch(p2_box)

p2_items = [
    "Counterfactual Causal KL-Score:",
    "  C_j->i = D_KL( pi_i(m_j) || pi_i(m_bar_j) )",
    "",
    "• Evaluates policy shift under masked m_bar_j",
    "• Amplification: phi(C_ij) = 1 + tanh(C_ij)",
    "",
    "Correlates with Queue Pressure (R^2 = 0.96)"
]
for item in p2_items:
    weight = "bold" if ("Correlates" in item or "Counterfactual" in item or "Amplification" in item) else "normal"
    col = THEME["amber"] if "Correlates" in item else THEME["body_text"]
    ax.text(53.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.1

y_cursor -= 4.0
ax.text(52.5, y_cursor, "Spatial Graph Attention Aggregation", color=THEME["teal"], fontsize=10.5, fontweight="bold")
y_cursor -= 3.0
gat_box = patches.FancyBboxPatch((52.5, y_cursor - 12.0), 18.0, 12.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["teal_bg"], edgecolor=THEME["teal"], lw=1.2, zorder=3)
ax.add_patch(gat_box)

gat_items = [
    "Context Embedding Formulation:",
    "  z_i = sum_j ( g_ij * alpha_ij * phi(C_ij) * W_v h_j )",
    "",
    "Output: z_i in R^64 (Spatial + Temporal)"
]
for item in gat_items:
    weight = "bold" if ("Output" in item or "Context" in item) else "normal"
    col = THEME["teal"] if "Output" in item else THEME["body_text"]
    ax.text(53.5, y_cursor - 1.5, item, color=col, fontsize=9.5, fontweight=weight, zorder=4)
    y_cursor -= 2.6

# Bottom Badge
badge_gat = patches.FancyBboxPatch((52.5, 20.0), 18.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#ccfbf1", edgecolor=THEME["teal"], lw=1.5, zorder=3)
ax.add_patch(badge_gat)
ax.text(61.5, 22.5, "Context: z_i in R^64", color=THEME["teal"], fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

# =========================================================================
# STAGE 4: ACTOR-CRITIC DECISION LAYER (Column 4: X=77 to 97, Y=53 to 90)
# =========================================================================
draw_card(77, 53, 20, 37, "4. DECISION LAYER (CTDE)", "Decentralized Actor + Centralized Critic", header_col=THEME["green"], bg_col="#ffffff")

# Arrow from Spatial to Decision
draw_arrow((72, 71.5), (77, 71.5), "z_i [64-D]", "Context Vector", col=THEME["teal"])

y_cursor = 81.5
ax.text(78.5, y_cursor, "Shared Decentralized Actor MLP", color=THEME["green"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
actor_box = patches.FancyBboxPatch((78.5, y_cursor - 21.0), 17.0, 21.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["green_bg"], edgecolor=THEME["green"], lw=1.2, zorder=3)
ax.add_patch(actor_box)

actor_items = [
    "Actor Input: [z_i || e_i] in R^68",
    "  • Context Vector z_i: 64-D",
    "  • One-Hot Node ID e_i: 4-D",
    "",
    "Network: Linear(68 -> 64 -> 64 -> 4)",
    "Categorical Action Distribution:",
    "  • Action 0 : KEEP (Hold Phase)",
    "  • Action 1 : SWITCH (Change Phase)",
    "  • Action 2 : EXTEND (+5s Green Target)",
    "  • Action 3 : REDUCE (-5s Green Target)"
]
for item in actor_items:
    weight = "bold" if ("Actor Input" in item or "Categorical" in item or "Action" in item) else "normal"
    col = THEME["green"] if "Action" in item else THEME["body_text"]
    ax.text(79.5, y_cursor - 1.5, item, color=col, fontsize=9.2, fontweight=weight, zorder=4)
    y_cursor -= 2.0

# Centralized Critic Sidecar
critic_box = patches.FancyBboxPatch((78.5, 54.5), 17.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#f1f5f9", edgecolor=THEME["card_border"], lw=1.0, zorder=3)
ax.add_patch(critic_box)
ax.text(87.0, 57.0, "Centralized Critic (Training): V(S) in R", color=THEME["sub_text"], fontsize=9.0, fontweight="bold", ha="center", va="center", zorder=4)

# =========================================================================
# STAGE 5: RUNTIME SAFETY SHIELD (Column 4 Bottom: X=77 to 97, Y=5 to 47)
# =========================================================================
draw_card(77, 5, 20, 42, "5. RUNTIME SAFETY SHIELD", "Deterministic Safety Invariant Layer", header_col=THEME["red"], bg_col="#ffffff")

# Downward arrow from Actor to Safety Shield
draw_arrow((87, 53), (87, 47), "Proposed a_i", "Policy Action", col=THEME["red"], lw=3.0)

y_cursor = 40.0
ax.text(78.5, y_cursor, "4 Mandatory Physical Invariants", color=THEME["red"], fontsize=11, fontweight="bold")
y_cursor -= 3.0
shield_box = patches.FancyBboxPatch((78.5, y_cursor - 24.0), 17.0, 24.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor=THEME["red_bg"], edgecolor=THEME["red"], lw=1.2, zorder=3)
ax.add_patch(shield_box)

shield_items = [
    "1. MIN_GREEN Clamp (< 5s):",
    "   Premature switch clamped to KEEP (0)",
    "",
    "2. MAX_GREEN Enforcement (>= 60s):",
    "   Prolonged green forced to SWITCH (1)",
    "",
    "3. CLEARANCE Lockout (Yellow 3s, Red 1s):",
    "   Phase transitions locked to KEEP (0)",
    "",
    "4. MIN_RED Guarantee (>= 5s):",
    "   Protects opposing cross-traffic",
    "",
    "Outcome: 0 Green Clashes / 110,000s Tested"
]
for item in shield_items:
    weight = "bold" if ("1." in item or "2." in item or "3." in item or "4." in item or "Outcome" in item) else "normal"
    col = THEME["red"] if "Outcome" in item else THEME["body_text"]
    ax.text(79.5, y_cursor - 1.5, item, color=col, fontsize=9.0, fontweight=weight, zorder=4)
    y_cursor -= 2.1

# Bottom Verified Badge
badge_safe = patches.FancyBboxPatch((78.5, 6.5), 17.0, 5.0, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#fee2e2", edgecolor=THEME["red"], lw=1.5, zorder=3)
ax.add_patch(badge_safe)
ax.text(87.0, 9.0, "Safe Action: a_i^exec -> TraCI", color=THEME["red"], fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)

# Global Bottom Note
ax.text(50, 1.8, "Figure 10: Complete Neural Architecture & Runtime Decision Pipeline — Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning",
        color=THEME["sub_text"], fontsize=11, fontweight="bold", ha="center", va="center")

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/10_project_architecture_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plots/10_project_architecture_diagram.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved redesigned project architecture diagram to report_images/10_project_architecture_diagram.png and plots/")
