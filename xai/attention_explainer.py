"""
GAT Attention Explainer Module (XAI)
====================================
Analyzes Graph Attention Network (GAT) weights across the canonical directed edges,
identifying dominant incoming attention channels while enforcing epistemological boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES, get_adjacent_neighbors

# Explicit Epistemological Safeguard Warning
ATTENTION_EPISTEMOLOGICAL_WARNING = (
    "CAUTION: Learned Graph Attention weights (alpha_ij) represent feature aggregation routing "
    "within the neural network. They do NOT constitute empirical proof of physical causality, "
    "platoon arrival, or ground-truth traffic volume transfer."
)


@dataclass
class AttentionExplanation:
    """Structured explanation of spatial attention coefficients."""
    edge_weights: Dict[Tuple[str, str], float]
    formatted_edge_weights: Dict[str, float]
    strongest_incoming_per_agent: Dict[str, Tuple[str, float]]
    strongest_overall_edge: Tuple[Tuple[str, str], float]
    epistemological_disclaimer: str
    grounded_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_weights": {f"{u}->{v}": round(w, 4) for (u, v), w in self.edge_weights.items()},
            "strongest_incoming": {
                node: {"source": src, "weight": round(w, 4)}
                for node, (src, w) in self.strongest_incoming_per_agent.items()
            },
            "strongest_overall_edge": {
                "edge": f"{self.strongest_overall_edge[0][0]}->{self.strongest_overall_edge[0][1]}",
                "weight": round(self.strongest_overall_edge[1], 4)
            },
            "disclaimer": self.epistemological_disclaimer,
            "summary": self.grounded_summary
        }


class AttentionExplainer:
    """Interprets GAT spatial attention coefficients over canonical physical corridors."""

    def __init__(self):
        self.edges = ALLOWED_DIRECTED_EDGES
        self.agents = PHYSICAL_INTERSECTIONS

    def explain(
        self,
        attention_weights: Dict[Tuple[str, str], float],
        self_loop_weights: Optional[Dict[str, float]] = None
    ) -> AttentionExplanation:
        """
        Analyze attention weights for the 8 directed corridors.

        Args:
            attention_weights: Dict mapping (src, tgt) -> float attention weight alpha_ij
            self_loop_weights: Optional dict mapping node -> float self-loop weight
        """
        # Ensure all 8 canonical edges are present
        clean_weights = {}
        for edge in self.edges:
            clean_weights[edge] = float(attention_weights.get(edge, 1.0 / len(self.edges)))

        formatted = {f"{u}->{v}": w for (u, v), w in clean_weights.items()}

        # 1. Identify strongest incoming source for each target node
        strongest_incoming = {}
        for tgt in self.agents:
            neighbors = get_adjacent_neighbors(tgt)
            incoming = [(src, clean_weights.get((src, tgt), 0.0)) for src in neighbors]
            if incoming:
                incoming.sort(key=lambda x: -x[1])
                strongest_incoming[tgt] = incoming[0]
            else:
                strongest_incoming[tgt] = (tgt, 0.0)

        # 2. Overall strongest edge
        sorted_all = sorted(clean_weights.items(), key=lambda x: -x[1])
        strongest_overall = sorted_all[0] if sorted_all else ((( "A", "B"), 0.0))

        # 3. Grounded summary narrative
        u_top, v_top = strongest_overall[0]
        w_top = strongest_overall[1]

        summary_parts = [
            f"Dominant network attention link is ({u_top}->{v_top}) with weight alpha={w_top:.3f}."
        ]
        for tgt, (src, w) in strongest_incoming.items():
            summary_parts.append(f"Node {tgt} attends most strongly to upstream Node {src} (alpha={w:.3f}).")

        summary = " ".join(summary_parts)

        return AttentionExplanation(
            edge_weights=clean_weights,
            formatted_edge_weights=formatted,
            strongest_incoming_per_agent=strongest_incoming,
            strongest_overall_edge=strongest_overall,
            epistemological_disclaimer=ATTENTION_EPISTEMOLOGICAL_WARNING,
            grounded_summary=summary
        )

