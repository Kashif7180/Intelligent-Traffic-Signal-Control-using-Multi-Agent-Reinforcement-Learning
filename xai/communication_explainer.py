"""
Communication Explainer Module (XAI)
====================================
Interprets dynamic communication gate activations, message transmission decisions,
and bandwidth conservation across the 8 physical communication channels.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from network.topology import ALLOWED_DIRECTED_EDGES


@dataclass
class CommunicationExplanation:
    """Structured record of communication gating behavior and bandwidth status."""
    gate_values: Dict[Tuple[str, str], float]
    formatted_gate_values: Dict[str, float]
    active_edges: List[Tuple[str, str]]
    suppressed_edges: List[Tuple[str, str]]
    threshold: float
    active_count: int
    total_possible: int
    bandwidth_savings_pct: float
    communication_cost: float
    grounded_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_values": {f"{u}->{v}": round(g, 6) for (u, v), g in self.gate_values.items()},
            "active_edges": [f"{u}->{v}" for (u, v) in self.active_edges],
            "suppressed_edges": [f"{u}->{v}" for (u, v) in self.suppressed_edges],
            "threshold": self.threshold,
            "active_count": self.active_count,
            "total_possible": self.total_possible,
            "bandwidth_savings_pct": round(self.bandwidth_savings_pct, 2),
            "communication_cost": round(self.communication_cost, 4),
            "summary": self.grounded_summary
        }


class CommunicationExplainer:
    """Explains neural communication gate activations and channel filtering."""

    def __init__(self, threshold: float = 0.5, cost_per_msg: float = 0.01):
        self.threshold = threshold
        self.cost_per_msg = cost_per_msg
        self.edges = ALLOWED_DIRECTED_EDGES

    def explain(
        self,
        gate_values: Dict[Tuple[str, str], float]
    ) -> CommunicationExplanation:
        """
        Analyze continuous communication gate values against the hard decision threshold.

        Args:
            gate_values: Mapping of (src, tgt) -> float continuous sigmoid activation g_ij
        """
        clean_gates = {}
        for edge in self.edges:
            clean_gates[edge] = float(gate_values.get(edge, 0.0))

        formatted = {f"{u}->{v}": g for (u, v), g in clean_gates.items()}

        active = [e for e, g in clean_gates.items() if g >= self.threshold]
        suppressed = [e for e, g in clean_gates.items() if g < self.threshold]

        active_cnt = len(active)
        total_pos = len(self.edges)  # 8
        savings_pct = (1.0 - (active_cnt / max(1, total_pos))) * 100.0
        comm_cost = active_cnt * self.cost_per_msg

        # Synthesize narrative distinguishing hard suppression from continuous gating
        mean_g = float(np.mean(list(clean_gates.values()))) if clean_gates else 0.0
        if active_cnt == 0:
            summary = (
                f"Full communication suppression: 0 of 8 directed links open (evaluated against threshold tau={self.threshold}; "
                f"{savings_pct:.0f}% bandwidth savings). Continuous gate activations average g={mean_g:.5f} < {self.threshold}. "
                f"In deterministic evaluation, channels below {self.threshold} are fully pruned (effective gate 0.0); "
                f"in soft differentiable mode, messages are continuously attenuated by g_ij. "
                f"The controller relies primarily on local historical temporal features without inter-agent transmission."
            )
        elif active_cnt == total_pos:
            summary = (
                f"Dense communication active: All 8 directed channels open (g >= {self.threshold}). "
                f"Full inter-agent message passing enabled across all adjacent intersections (mean g={mean_g:.3f})."
            )
        else:
            active_str = ", ".join([f"({u}->{v})" for (u, v) in active])
            summary = (
                f"Selective communication active: {active_cnt}/{total_pos} links open under deterministic threshold tau={self.threshold} "
                f"({savings_pct:.1f}% bandwidth savings, mean g={mean_g:.3f}). Active channels: {active_str}. "
                f"Sub-threshold channels are hard-suppressed in evaluation mode and continuously attenuated in soft mode."
            )

        return CommunicationExplanation(
            gate_values=clean_gates,
            formatted_gate_values=formatted,
            active_edges=active,
            suppressed_edges=suppressed,
            threshold=self.threshold,
            active_count=active_cnt,
            total_possible=total_pos,
            bandwidth_savings_pct=savings_pct,
            communication_cost=comm_cost,
            grounded_summary=summary
        )
