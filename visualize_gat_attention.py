"""
Phase 6 Graph Attention Visualization & Interpretability Script.

Extracts real learned attention weights from a trained Phase 6 model (MAPPO + Temporal Transformer + GAT)
during a real evaluation scenario (Seed 1001, Decision Step 50 / 250s SUMO simulation time).

Outputs:
1. experiments/gat_attention_weights.csv: Raw per-head and aggregated attention weights.
2. experiments/gat_attention_visualization.png: High-resolution physical graph diagram
   annotated with real learned attention weights.
"""

import os
import sys
import csv
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environment.traffic_env import MultiAgentTrafficEnv
from models.gat_network import MAPPOGATController


def extract_and_visualize(seed: int = 1001, target_step: int = 50):
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_path = "experiments/gat_attention_weights.csv"
    plot_path = "experiments/gat_attention_visualization.png"

    print("=" * 80)
    print("PHASE 6: GRAPH ATTENTION WEIGHT EXTRACTION & VISUALIZATION")
    print("=" * 80)
    print(f"Evaluation Seed:         {seed}")
    print(f"Target Decision Step:    {target_step} (Simulation time: {target_step * 5}s)")
    print(f"Checkpoint:              models/checkpoints/mappo_transformer_gat_actor.pt")
    print("-" * 80)

    # Initialize environment and controller
    env = MultiAgentTrafficEnv(config_path=config_path)
    ctrl = MAPPOGATController(config_path=config_path)
    ctrl.load_checkpoints("models/checkpoints")

    obs, info = env.reset(seed=seed)
    ctrl.reset_history(obs)

    extracted_weights = None
    step_metrics = {}

    for step in range(1, target_step + 1):
        # Deterministic action selection executes GAT forward pass
        actions, _ = ctrl.get_actions(deterministic=True)
        next_obs, rewards, terminations, truncations, next_info = env.step(actions)
        ctrl.update_history(next_obs)
        obs = next_obs

        if step == target_step:
            extracted_weights = ctrl.get_attention_weights()
            step_metrics = {
                a: {
                    "queue": next_info[a]["metrics"]["queue_length"],
                    "wait": next_info[a]["metrics"]["waiting_time"],
                    "phase": f"Phase {next_info[a]['metrics']['current_phase']}"
                }
                for a in env.agents
            }

        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()

    if extracted_weights is None:
        raise RuntimeError("Failed to extract attention weights at target step.")

    sim_time = target_step * 5

    # 1. Log Raw Attention Values to CSV
    print(f"\n[1/2] Saving raw attention weights to CSV -> {csv_path}")
    fieldnames = [
        "seed",
        "decision_step",
        "simulation_time_s",
        "source",
        "target",
        "head",
        "attention_weight",
        "aggregated_attention_weight",
        "is_self_loop"
    ]

    num_heads = ctrl.gat_num_heads
    agg_map = {(e["source"], e["target"]): e["weight"] for e in extracted_weights["aggregated"]}

    rows = []
    for k in range(num_heads):
        for e in extracted_weights["per_head"][k]:
            src = e["source"]
            tgt = e["target"]
            rows.append({
                "seed": seed,
                "decision_step": target_step,
                "simulation_time_s": sim_time,
                "source": src,
                "target": tgt,
                "head": k,
                "attention_weight": f"{e['weight']:.6f}",
                "aggregated_attention_weight": f"{agg_map[(src, tgt)]:.6f}",
                "is_self_loop": e["is_self_loop"]
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"      Saved {len(rows)} attention records.")

    # 2. Render Graph Attention Visualization
    print(f"\n[2/2] Generating graph visualization -> {plot_path}")
    render_graph(extracted_weights, step_metrics, seed, target_step, sim_time, plot_path)
    print("      Visualization rendered successfully.")

    # Print summary of learned attention
    print("\n" + "=" * 60)
    print(f"LEARNED ATTENTION WEIGHTS (Seed {seed}, Step {target_step}, t={sim_time}s)")
    print("=" * 60)
    print(f"{'Source':<8} -> {'Target':<8} | {'Type':<12} | {'Aggregated Weight':<18}")
    print("-" * 60)
    for e in extracted_weights["aggregated"]:
        t_str = "Self-Loop" if e["is_self_loop"] else "Neighbor Edge"
        print(f"{e['source']:<8} -> {e['target']:<8} | {t_str:<12} | {e['weight']:<18.4f}")
    print("=" * 60)


def render_graph(extracted_weights, step_metrics, seed, step, sim_time, output_path):
    """
    Renders a physical road network diagram with real learned attention weights.
    Physical coordinates:
      A: (1.5, 3.5)   B: (4.5, 3.5)
      C: (1.5, 1.5)   D: (4.5, 1.5)
    """
    fig, (ax_graph, ax_mat) = plt.subplots(1, 2, figsize=(16, 8), dpi=150, gridspec_kw={'width_ratios': [1.3, 1.0]})
    fig.patch.set_facecolor("#1e1e2e")

    # Node physical coordinates in plot space
    pos = {
        "A": np.array([1.5, 3.5]),
        "B": np.array([4.5, 3.5]),
        "C": np.array([1.5, 1.5]),
        "D": np.array([4.5, 1.5])
    }

    ax_graph.set_facecolor("#181825")
    ax_graph.set_xlim(0.2, 5.8)
    ax_graph.set_ylim(0.2, 4.8)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")

    # Physical road connections: pairs of (src, tgt)
    # Note: message goes from src -> tgt (arrow points towards tgt)
    directed_edges = [
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "D"), ("D", "B"),
        ("C", "D"), ("D", "C")
    ]

    agg_map = {(e["source"], e["target"]): e["weight"] for e in extracted_weights["aggregated"]}

    # Draw physical road corridors (underlay)
    road_pairs = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    for u, v in road_pairs:
        p1, p2 = pos[u], pos[v]
        ax_graph.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#313244", lw=18, zorder=1, solid_capstyle="round")
        ax_graph.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#45475a", lw=2, linestyle="--", zorder=2)

    # Draw directed attention edges with offset to separate bidirectional flows
    for src, tgt in directed_edges:
        p_src = pos[src]
        p_tgt = pos[tgt]
        w = agg_map.get((src, tgt), 0.0)

        # Vector from src to tgt
        vec = p_tgt - p_src
        dist = np.linalg.norm(vec)
        unit = vec / dist

        # Normal vector pointing to the right side of the road
        norm = np.array([-unit[1], unit[0]])
        offset = 0.14  # lane separation offset

        start = p_src + norm * offset + unit * 0.42
        end = p_tgt + norm * offset - unit * 0.42

        # Color and line width scaled by attention weight
        lw = max(1.5, w * 12.0)
        # Colormap from cyan to yellow/red
        cmap = plt.cm.viridis
        edge_color = cmap(min(1.0, w * 2.2))

        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle="->,head_length=8,head_width=5",
            connectionstyle="arc3,rad=0.0",
            color=edge_color,
            lw=lw,
            zorder=4
        )
        ax_graph.add_patch(arrow)

        # Edge label at midpoint
        mid = (start + end) / 2.0 + norm * 0.12
        ax_graph.text(
            mid[0], mid[1], f"{w:.2f}",
            color="#cdd6f4", fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#11111b", edgecolor=edge_color, lw=1.0, alpha=0.9),
            zorder=5
        )

    # Draw Nodes (Intersections)
    for node, p in pos.items():
        self_w = agg_map.get((node, node), 0.0)
        q = step_metrics.get(node, {}).get("queue", 0.0)
        ph = step_metrics.get(node, {}).get("phase", "N/A")

        circle = plt.Circle(p, 0.42, facecolor="#1e1e2e", edgecolor="#89b4fa", lw=3, zorder=6)
        ax_graph.add_patch(circle)

        ax_graph.text(p[0], p[1] + 0.08, f"Int {node}", color="#ffffff", fontsize=12, fontweight="heavy", ha="center", va="center", zorder=7)
        ax_graph.text(p[0], p[1] - 0.16, f"Self: {self_w:.2f}", color="#fab387", fontsize=9, fontweight="bold", ha="center", va="center", zorder=7)

        # Info badge outside node
        offset_y = 0.65 if "A" in node or "B" in node else -0.65
        badge_text = f"Q: {q:.1f} veh | {ph}"
        ax_graph.text(
            p[0], p[1] + offset_y, badge_text,
            color="#a6adc8", fontsize=8.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#181825", edgecolor="#45475a", lw=0.8),
            zorder=6
        )

    ax_graph.set_title(
        f"GAT Physical Road Attention Network\nSeed {seed} | Decision Step {step} (t={sim_time}s)",
        color="#cdd6f4", fontsize=13, fontweight="bold", pad=15
    )

    # 2. Right Plot: Attention Weight Matrix Heatmap
    ax_mat.set_facecolor("#181825")
    agents = ["A", "B", "C", "D"]
    mat = np.zeros((4, 4))
    mask_diag = np.zeros((4, 4), dtype=bool)

    for i, tgt in enumerate(agents):
        for j, src in enumerate(agents):
            if (src, tgt) in agg_map:
                mat[i, j] = agg_map[(src, tgt)]
            else:
                mat[i, j] = 0.0
                mask_diag[i, j] = True

    im = ax_mat.imshow(mat, cmap="viridis", vmin=0.0, vmax=0.7)
    ax_mat.set_xticks(range(4))
    ax_mat.set_yticks(range(4))
    ax_mat.set_xticklabels([f"Src {a}" for a in agents], fontsize=10, fontweight="bold", color="#cdd6f4")
    ax_mat.set_yticklabels([f"Tgt {a}" for a in agents], fontsize=10, fontweight="bold", color="#cdd6f4")
    ax_mat.set_title("Aggregated Attention Matrix $\\bar{\\alpha}_{ij}$ (Mean over 4 Heads)", color="#cdd6f4", fontsize=11, fontweight="bold", pad=12)

    # Annotate matrix cells
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            if mask_diag[i, j]:
                txt = "0 (N/A)"
                color = "#585b70"
            else:
                txt = f"{val:.3f}"
                color = "#ffffff" if val > 0.35 else "#cdd6f4"
            ax_mat.text(j, i, txt, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#cdd6f4")
    cbar.set_label("Learned Attention Coefficient $\\alpha_{ij}$", color="#cdd6f4", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    extract_and_visualize(seed=1001, target_step=50)
