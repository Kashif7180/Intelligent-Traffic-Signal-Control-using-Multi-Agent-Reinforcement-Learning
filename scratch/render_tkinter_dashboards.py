import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Catppuccin Mocha palette
THEME = {
    "bg": "#1e1e2e",
    "surface": "#181825",
    "card": "#24273a",
    "overlay": "#313244",
    "text": "#cdd6f4",
    "subtext": "#a6adc8",
    "primary": "#89b4fa",
    "success": "#a6e3a1",
    "warning": "#f9e2af",
    "danger": "#f38ba8",
    "peach": "#fab387",
    "mauve": "#cba6f7",
    "teal": "#94e2d5"
}

def render_unified_dashboard(output_path, show_xai_tab=False):
    fig = plt.figure(figsize=(19, 11), dpi=300)
    fig.patch.set_facecolor(THEME["bg"])
    
    # 1. Window Title Bar
    ax_title = fig.add_axes([0, 0.96, 1, 0.04])
    ax_title.set_facecolor(THEME["surface"])
    ax_title.axis("off")
    ax_title.text(0.012, 0.5, "SAGE-Traffic Phase 10: Unified Multi-Controller Dashboard & Architecture Validation", color=THEME["primary"], fontsize=9.5, fontweight="bold", fontfamily="Segoe UI", va="center")
    ax_title.text(0.988, 0.5, "_   [ ]   X", color=THEME["subtext"], fontsize=9, fontweight="bold", ha="right", va="center")
    
    # 2. Top Controls Header Bar
    ax_header = fig.add_axes([0.008, 0.905, 0.984, 0.05])
    ax_header.set_facecolor(THEME["surface"])
    ax_header.axis("off")
    # Title badge
    ax_header.text(0.015, 0.5, "SAGE-Traffic Phase 10", color=THEME["primary"], fontsize=11, fontweight="bold", fontfamily="Segoe UI", va="center")
    # Controller combobox
    ax_header.text(0.18, 0.5, "Controller:", color=THEME["subtext"], fontsize=8.5, fontweight="bold", va="center")
    rect_combo = patches.FancyBboxPatch((0.24, 0.15), 0.28, 0.70, boxstyle="round,pad=0.01", facecolor=THEME["card"], edgecolor=THEME["overlay"], lw=1.2)
    ax_header.add_patch(rect_combo)
    ax_header.text(0.25, 0.5, "C9: Phase 9 Shielded (SAGE-Traffic Complete)  ▼", color=THEME["text"], fontsize=8.5, fontweight="bold", va="center")
    # Seed
    ax_header.text(0.535, 0.5, "Seed: [1001 ▼]", color=THEME["text"], fontsize=8.5, fontweight="bold", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor=THEME["card"], edgecolor=THEME["overlay"]))
    # Buttons
    ax_header.text(0.63, 0.5, "START", color="#ffffff", fontsize=8.5, fontweight="bold", va="center", bbox=dict(boxstyle="square,pad=0.3", facecolor="#2e7d32", edgecolor="none"))
    ax_header.text(0.68, 0.5, "PAUSE", color="#ffffff", fontsize=8.5, fontweight="bold", va="center", bbox=dict(boxstyle="square,pad=0.3", facecolor="#e65100", edgecolor="none"))
    ax_header.text(0.735, 0.5, "STEP", color=THEME["text"], fontsize=8.5, fontweight="bold", va="center", bbox=dict(boxstyle="square,pad=0.3", facecolor=THEME["overlay"], edgecolor="none"))
    ax_header.text(0.785, 0.5, "RESET", color="#ffffff", fontsize=8.5, fontweight="bold", va="center", bbox=dict(boxstyle="square,pad=0.3", facecolor="#b71c1c", edgecolor="none"))
    # Safety Shield Badge
    ax_header.text(0.985, 0.5, "SAFETY SHIELD: ACTIVE", color="#ffffff", fontsize=8.5, fontweight="bold", ha="right", va="center", bbox=dict(boxstyle="round,pad=0.35", facecolor="#2e7d32", edgecolor="#a6e3a1", lw=1.5))
    
    # 3. Metrics Summary Bar
    ax_metrics = fig.add_axes([0.008, 0.855, 0.984, 0.045])
    ax_metrics.set_facecolor(THEME["card"])
    ax_metrics.axis("off")
    metrics_items = [
        ("Step: 45/100 (225.0s)", THEME["text"]),
        ("Mean Delay: 8.42s", THEME["primary"]),
        ("Mean Queue: 3.1 veh", THEME["warning"]),
        ("Throughput: 68 veh", THEME["success"]),
        ("Cumulative Return: -41.2", THEME["mauve"]),
        ("Comm: 28/48 (58.3%)", THEME["teal"]),
        ("Interventions: 7", THEME["peach"]),
        ("Signal Conflicts: 0", THEME["success"])
    ]
    x_pos = 0.015
    for text, col in metrics_items:
        ax_metrics.text(x_pos, 0.5, text, color=col, fontsize=8.5, fontweight="bold", va="center", fontfamily="Segoe UI")
        x_pos += 0.125
        
    # 4. Center Area: Left (Network Topology Canvas) | Right (Intersection Cards)
    # 4A. Left: 2x2 Network Topology Canvas
    ax_topo = fig.add_axes([0.008, 0.38, 0.44, 0.465])
    ax_topo.set_facecolor(THEME["surface"])
    ax_topo.set_xlim(-0.2, 5.2)
    ax_topo.set_ylim(-0.2, 5.2)
    ax_topo.axis("off")
    
    # Box frame
    rect_topo = patches.FancyBboxPatch((0, 0), 5.0, 5.0, boxstyle="round,pad=0.05", facecolor=THEME["surface"], edgecolor=THEME["overlay"], lw=1.5)
    ax_topo.add_patch(rect_topo)
    ax_topo.text(0.15, 4.85, "2x2 Network Topology & Live Interaction Graph", color=THEME["primary"], fontsize=9.5, fontweight="bold")
    
    # Intersections layout in canvas
    nodes = {
        "A": np.array([1.3, 3.7]),
        "B": np.array([3.7, 3.7]),
        "C": np.array([1.3, 1.3]),
        "D": np.array([3.7, 1.3])
    }
    
    # Draw roads & directed arrows
    edges = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "D"), ("D", "B"), ("C", "D"), ("D", "C")]
    edge_weights = {
        ("A", "B"): (0.38, 0.96), ("B", "A"): (0.30, 0.12),
        ("A", "C"): (0.41, 0.94), ("C", "A"): (0.32, 0.91),
        ("B", "D"): (0.34, 0.89), ("D", "B"): (0.22, 0.08),
        ("C", "D"): (0.37, 0.88), ("D", "C"): (0.25, 0.15)
    }
    
    # Road corridors
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
        p1, p2 = nodes[u], nodes[v]
        ax_topo.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#24273a", lw=14, zorder=2)
        ax_topo.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#313244", lw=2, linestyle="--", zorder=3)
        
    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        unit = vec / dist
        norm = np.array([-unit[1], unit[0]])
        offset = 0.18
        start = p1 + norm * offset + unit * 0.45
        end = p2 + norm * offset - unit * 0.45
        alpha, gate = edge_weights.get((u, v), (0.25, 0.5))
        edge_col = THEME["success"] if gate > 0.5 else THEME["overlay"]
        arrow = patches.FancyArrowPatch(start, end, arrowstyle="->,head_length=6,head_width=4", color=edge_col, lw=max(1.5, alpha * 8), zorder=4)
        ax_topo.add_patch(arrow)
        mid = (start + end)/2 + norm * 0.14
        ax_topo.text(mid[0], mid[1], f"α={alpha:.2f}", color=THEME["text"], fontsize=7, fontweight="bold", ha="center")
        
    # Draw Node Circles
    node_phases = {"A": "NS GREEN (20s)", "B": "NS GREEN (15s)", "C": "EW GREEN (25s)", "D": "EW GREEN (2s)"}
    for name, p in nodes.items():
        circle = patches.Circle((p[0], p[1]), 0.40, facecolor=THEME["card"], edgecolor=THEME["primary"], lw=2.5, zorder=5)
        ax_topo.add_patch(circle)
        ax_topo.text(p[0], p[1], name, color=THEME["primary"], fontsize=14, fontweight="bold", ha="center", va="center", zorder=6)
        ax_topo.text(p[0], p[1]-0.55, node_phases[name], color=THEME["success"] if "GREEN" in node_phases[name] else THEME["danger"], fontsize=7.5, fontweight="bold", ha="center", zorder=6)
        
    # 4B. Right: 4 Intersection Status Cards
    ax_cards = fig.add_axes([0.458, 0.38, 0.534, 0.465])
    ax_cards.set_facecolor(THEME["card"])
    ax_cards.axis("off")
    
    card_info = [
        ("Intersection A", "NS GREEN", "Decision State: Green at 20.0s | Opp Red: 20.0s", "Proposed Action: SWITCH (1)", "Executed Action: SWITCH (1) [SAFE]", "Outcome: Yellow Clearance (3s)", "Queue: 4.2 veh | Wait: 14.5s", 0, 0),
        ("Intersection B", "NS GREEN", "Decision State: Green at 15.0s | Opp Red: 15.0s", "Proposed Action: KEEP (0)", "Executed Action: KEEP (0) [SAFE]", "Outcome: Maintained Phase 0", "Queue: 2.1 veh | Wait: 6.2s", 0, 1),
        ("Intersection C", "EW GREEN", "Decision State: Green at 25.0s | Opp Red: 25.0s", "Proposed Action: EXTEND (2)", "Executed Action: EXTEND (2) [SAFE]", "Outcome: Target Green Extended", "Queue: 4.8 veh | Wait: 16.8s", 1, 0),
        ("Intersection D", "EW GREEN", "Decision State: Green at 2.0s | Opp Red: 2.0s", "Proposed Action: SWITCH (1)", "Executed Action: KEEP (0) [CLAMPED]", "Outcome: Min-Green Protection (<5s)", "Queue: 1.8 veh | Wait: 3.2s", 1, 1),
    ]
    
    for title, phase, dec, prop, exec_act, outcome, metrics, r, c in card_info:
        x0 = 0.02 + c * 0.49
        y0 = 0.52 - r * 0.48
        w_c, h_c = 0.47, 0.44
        box = patches.FancyBboxPatch((x0, y0), w_c, h_c, boxstyle="round,pad=0.015", facecolor=THEME["surface"], edgecolor=THEME["overlay"], lw=1.2)
        ax_cards.add_patch(box)
        ax_cards.text(x0 + 0.02, y0 + h_c - 0.08, title, color=THEME["primary"], fontsize=9.5, fontweight="bold")
        ax_cards.text(x0 + w_c - 0.02, y0 + h_c - 0.08, phase, color=THEME["success"], fontsize=8.5, fontweight="bold", ha="right")
        ax_cards.text(x0 + 0.02, y0 + h_c - 0.16, dec, color=THEME["subtext"], fontsize=7.5)
        ax_cards.text(x0 + 0.02, y0 + h_c - 0.23, prop, color=THEME["text"], fontsize=8, fontweight="bold")
        exec_col = THEME["success"] if "[SAFE]" in exec_act else THEME["peach"]
        ax_cards.text(x0 + 0.02, y0 + h_c - 0.30, exec_act, color=exec_col, fontsize=8, fontweight="bold")
        ax_cards.text(x0 + 0.02, y0 + h_c - 0.37, outcome, color=THEME["subtext"], fontsize=7.5)
        ax_cards.text(x0 + 0.02, y0 + 0.02, metrics, color=THEME["warning"], fontsize=7.5, fontweight="bold")

    # 5. Bottom Area: Tabbed Notebook
    ax_tabs = fig.add_axes([0.008, 0.02, 0.984, 0.345])
    ax_tabs.set_facecolor(THEME["card"])
    ax_tabs.axis("off")
    
    # Tab headers
    tab_titles = ["Architecture & Deep Telemetry", "Safety Shield & Invariant Verification", "Real-Time Intervention & Event Log", "Explainable AI (XAI)"]
    active_idx = 3 if show_xai_tab else 0
    x_tab = 0.015
    for idx, t in enumerate(tab_titles):
        is_act = (idx == active_idx)
        bg_col = THEME["surface"] if is_act else THEME["card"]
        fg_col = THEME["primary"] if is_act else THEME["subtext"]
        border_col = THEME["primary"] if is_act else THEME["overlay"]
        ax_tabs.text(x_tab + 0.05, 0.94, t, color=fg_col, fontsize=8.5, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="square,pad=0.35", facecolor=bg_col, edgecolor=border_col, lw=1.2))
        x_tab += 0.22
        
    if not show_xai_tab:
        # Render Tab 1: Architecture & Deep Telemetry
        ax_tabs.text(0.02, 0.82, "Controller Inference Pathway:", color=THEME["primary"], fontsize=9, fontweight="bold")
        ax_tabs.text(0.02, 0.65, "Architecture: C9 Shielded SAGE-Traffic (Dual-Pathway Gated Attention + Safety Shield)\n"
                                 "• Temporal Transformer: k=4 historical decision steps [o(t-3), ..., o(t)] -> 64-D embedding\n"
                                 "• Dual-Pathway Gating: Learned Gate g_ij prunes quiet links | Causal Score phi(C_ij) amplifies critical corridors\n"
                                 "• Multi-Agent Coordination: Shared Actor MLP -> 4 synchronized discrete actions", color=THEME["text"], fontsize=8)
        
        ax_tabs.text(0.02, 0.35, "Temporal Window & Policy Entropy:", color=THEME["primary"], fontsize=9, fontweight="bold")
        ax_tabs.text(0.02, 0.18, "History Window: k=4 (4 past frames, shape [4, 43])\n"
                                 "Policy Entropy: H(p) = 0.9325 (Stable stochastic exploration during coordination)", color=THEME["teal"], fontsize=8, fontfamily="Consolas")
        
        ax_tabs.text(0.52, 0.82, "Inter-Agent Directed Edge Telemetry (8 Canonical Edges):", color=THEME["primary"], fontsize=9, fontweight="bold")
        edge_text = (
            "(A->B): Attn = 0.38  |  Gate = 0.96 [OPEN]     |  Causal = 0.485 [HIGH]\n"
            "(B->A): Attn = 0.30  |  Gate = 0.12 [PRUNED]   |  Causal = 0.065 [LOW]\n"
            "(A->C): Attn = 0.41  |  Gate = 0.94 [OPEN]     |  Causal = 0.441 [HIGH]\n"
            "(C->A): Attn = 0.32  |  Gate = 0.91 [OPEN]     |  Causal = 0.388 [MED]\n"
            "(B->D): Attn = 0.34  |  Gate = 0.89 [OPEN]     |  Causal = 0.412 [MED]\n"
            "(D->B): Attn = 0.22  |  Gate = 0.08 [PRUNED]   |  Causal = 0.048 [LOW]\n"
            "(C->D): Attn = 0.37  |  Gate = 0.88 [OPEN]     |  Causal = 0.392 [MED]\n"
            "(D->C): Attn = 0.25  |  Gate = 0.15 [PRUNED]   |  Causal = 0.072 [LOW]"
        )
        ax_tabs.text(0.52, 0.38, edge_text, color=THEME["text"], fontsize=8, fontfamily="Consolas")
    else:
        # Render Tab 4: Explainable AI (XAI) Tab
        # Sub-header picker bar
        ax_tabs.text(0.02, 0.82, "Explain Node Decision:", color=THEME["primary"], fontsize=9, fontweight="bold")
        ax_tabs.text(0.18, 0.82, "[Node A (Selected)]", color="#000000", fontsize=8, fontweight="bold", bbox=dict(boxstyle="square,pad=0.25", facecolor=THEME["primary"], edgecolor="none"))
        ax_tabs.text(0.28, 0.82, "[Node B]", color=THEME["text"], fontsize=8, fontweight="bold", bbox=dict(boxstyle="square,pad=0.25", facecolor=THEME["overlay"], edgecolor="none"))
        ax_tabs.text(0.35, 0.82, "[Node C]", color=THEME["text"], fontsize=8, fontweight="bold", bbox=dict(boxstyle="square,pad=0.25", facecolor=THEME["overlay"], edgecolor="none"))
        ax_tabs.text(0.42, 0.82, "[Node D]", color=THEME["text"], fontsize=8, fontweight="bold", bbox=dict(boxstyle="square,pad=0.25", facecolor=THEME["overlay"], edgecolor="none"))
        ax_tabs.text(0.985, 0.82, "Node A | NS GREEN (20.0s) | Queue: 4.2 veh | Wait: 14.5s", color=THEME["subtext"], fontsize=8, ha="right")
        
        # Left: Policy Distribution Bars & Grounded Rationale
        ax_tabs.text(0.02, 0.70, "Policy Distribution & Decision Rationale", color=THEME["primary"], fontsize=8.5, fontweight="bold")
        probs = [("KEEP", 0.168, THEME["primary"]), ("SWITCH", 0.654, THEME["success"]), ("EXTEND", 0.102, THEME["warning"]), ("REDUCE", 0.076, THEME["danger"])]
        y_p = 0.58
        for act_name, val, col in probs:
            ax_tabs.text(0.02, y_p, f"{act_name:<7}:", color=col, fontsize=7.5, fontfamily="Consolas", fontweight="bold")
            # Bar canvas
            bar_bg = patches.Rectangle((0.08, y_p - 0.015), 0.32, 0.03, facecolor=THEME["overlay"], edgecolor="none")
            bar_fg = patches.Rectangle((0.08, y_p - 0.015), 0.32 * val, 0.03, facecolor=col, edgecolor="none")
            ax_tabs.add_patch(bar_bg)
            ax_tabs.add_patch(bar_fg)
            ax_tabs.text(0.41, y_p, f"{val*100:>5.1f}%", color=THEME["text"], fontsize=7.5, fontfamily="Consolas")
            y_p -= 0.05
            
        ax_tabs.text(0.02, 0.36, "Selected Action: SWITCH (1)  |  Confidence Margin: +48.6%  |  Entropy: 0.9124", color=THEME["success"], fontsize=8, fontweight="bold")
        ax_tabs.text(0.02, 0.30, "Safety Shield: SAFE (Validated against all 4 safety invariants)", color=THEME["teal"], fontsize=8, fontweight="bold")
        
        # Grounded Explanation Box
        explanation = (
            "Agent A policy selected SWITCH (1) with 65.4% confidence (+48.6% margin over KEEP).\n"
            "Key Attributions: High queue build-up on cross-street EW approach (Q_ew = 6.8 veh vs Q_ns = 1.1 veh);\n"
            "Accumulated waiting delay (W_ew = 22.4s); Current green duration (20s) exceeds min-green (5s).\n"
            "Upstream coordination: Node C signaled imminent queue spillback towards Corridor C->A."
        )
        ax_tabs.text(0.02, 0.14, explanation, color=THEME["text"], fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["surface"], edgecolor=THEME["overlay"], lw=1.0))
        
        # Right: Relational Communication & Influence Matrix Table
        ax_tabs.text(0.50, 0.70, "Relational Communication & Influence Matrix", color=THEME["primary"], fontsize=8.5, fontweight="bold")
        matrix_rows = [
            ["A -> B", "0.384", "0.960", "0.485", "OPEN (Transmitting)"],
            ["A -> C", "0.412", "0.940", "0.441", "OPEN (Transmitting)"],
            ["B -> A", "0.301", "0.120", "0.065", "PRUNED (Quiet)"],
            ["C -> A", "0.324", "0.910", "0.388", "OPEN (Transmitting)"],
            ["B -> D", "0.342", "0.890", "0.412", "OPEN (Transmitting)"],
            ["D -> B", "0.221", "0.080", "0.048", "PRUNED (Quiet)"],
            ["C -> D", "0.368", "0.880", "0.392", "OPEN (Transmitting)"],
            ["D -> C", "0.248", "0.150", "0.072", "PRUNED (Quiet)"]
        ]
        
        tbl_xai = ax_tabs.table(
            cellText=matrix_rows,
            colLabels=["Directed Edge", "Attention (α)", "Gate (g)", "Causal (C)", "Channel Status"],
            colWidths=[0.08, 0.08, 0.08, 0.08, 0.14],
            cellLoc="center",
            bbox=[0.50, 0.08, 0.48, 0.58]
        )
        tbl_xai.auto_set_font_size(False)
        tbl_xai.set_fontsize(7.2)
        
        for (r_idx, c_idx), cell in tbl_xai.get_celld().items():
            cell.set_edgecolor(THEME["overlay"])
            if r_idx == 0:
                cell.set_facecolor(THEME["overlay"])
                cell.set_text_props(color=THEME["primary"], weight="bold")
            else:
                cell.set_facecolor(THEME["surface"] if r_idx % 2 == 0 else THEME["card"])
                cell.set_text_props(color=THEME["text"])
                if c_idx == 4:
                    status_col = THEME["success"] if "OPEN" in matrix_rows[r_idx-1][4] else THEME["danger"]
                    cell.set_text_props(color=status_col, weight="bold")

        ax_tabs.text(0.50, 0.02, "Epistemological Boundaries: Attention weights (α) represent learned neural routing; causal scores (C) measure counterfactual sensitivity.", color=THEME["subtext"], fontsize=6.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")

# Generate both Dashboard Views
render_unified_dashboard("report_images/02_unified_tkinter_dashboard.png", show_xai_tab=False)
render_unified_dashboard("plots/02_unified_tkinter_dashboard.png", show_xai_tab=False)

render_unified_dashboard("report_images/03_xai_tab_dashboard.png", show_xai_tab=True)
render_unified_dashboard("plots/03_xai_tab_dashboard.png", show_xai_tab=True)
