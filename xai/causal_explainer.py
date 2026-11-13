"""
Causal Influence Explainer Module (XAI)
=======================================
Interprets Phase 8 and Phase 8R predictive causal influence scores (C_ij),
analyzing directional policy sensitivity while enforcing strict epistemological bounds.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES, get_adjacent_neighbors

# Mandatory Epistemological Safeguard Warning
CAUSAL_EPISTEMOLOGICAL_WARNING = (
    "CAUTION: Causal influence scores (C_ij) quantify observational predictive influence "
    "(Granger-style linear predictive variance reduction on lagged traffic states). "
    "They do NOT evaluate counterfactual model interventions, nor do they establish "
    "physical real-world causation or deterministic vehicle actuation."
)


@dataclass
class CausalExplanation:
    """Structured record of observational predictive causal influence metrics."""
    causal_scores: Dict[Tuple[str, str], float]
    formatted_causal_scores: Dict[str, float]
    strongest_influencer_per_agent: Dict[str, Tuple[str, float]]
    directional_asymmetries: Dict[str, float]
    epistemological_disclaimer: str
    grounded_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "causal_scores": {f"{u}->{v}": round(c, 4) for (u, v), c in self.causal_scores.items()},
            "strongest_influencer": {
                node: {"source": src, "score": round(score, 4)}
                for node, (src, score) in self.strongest_influencer_per_agent.items()
            },
            "directional_asymmetries": {k: round(v, 4) for k, v in self.directional_asymmetries.items()},
            "disclaimer": self.epistemological_disclaimer,
            "summary": self.grounded_summary
        }


class CausalExplainer:
    """Explains observational predictive causal influence estimations across adjacent intersections."""

    def __init__(self):
        self.edges = ALLOWED_DIRECTED_EDGES
        self.agents = PHYSICAL_INTERSECTIONS

    def explain(
        self,
        causal_scores: Dict[Tuple[str, str], float]
    ) -> CausalExplanation:
        """
        Analyze pairwise observational predictive influence scores.

        Args:
            causal_scores: Mapping (src, tgt) -> float C_ij in [0, 1]
        """
        clean_scores = {}
        for edge in self.edges:
            clean_scores[edge] = float(causal_scores.get(edge, 0.0))

        formatted = {f"{u}->{v}": c for (u, v), c in clean_scores.items()}

        # 1. Strongest upstream driver for each target intersection
        strongest = {}
        for tgt in self.agents:
            neighbors = get_adjacent_neighbors(tgt)
            incoming = [(src, clean_scores.get((src, tgt), 0.0)) for src in neighbors]
            if incoming:
                incoming.sort(key=lambda x: -x[1])
                strongest[tgt] = incoming[0]
            else:
                strongest[tgt] = (tgt, 0.0)

        # 2. Directional Asymmetry between adjacent pairs: Delta C = C(u->v) - C(v->u)
        asymmetries = {}
        visited_pairs = set()
        for (u, v) in self.edges:
            pair_key = tuple(sorted([u, v]))
            if pair_key not in visited_pairs:
                visited_pairs.add(pair_key)
                c_fwd = clean_scores.get((u, v), 0.0)
                c_rev = clean_scores.get((v, u), 0.0)
                delta = c_fwd - c_rev
                asymmetries[f"{u}<->{v}"] = delta

        # 3. Grounded narrative summary using predictive influence terminology
        max_edge = max(clean_scores.items(), key=lambda x: x[1]) if clean_scores else ((("A", "B"), 0.0))
        mean_score = float(np.mean(list(clean_scores.values()))) if clean_scores else 0.0

        summary_parts = [
            f"Network-wide mean predictive influence score is C={mean_score:.3f}. "
            f"Highest predictive correlation observed along corridor ({max_edge[0][0]}->{max_edge[0][1]}) "
            f"with score C={max_edge[1]:.3f}."
        ]
        for tgt, (src, s) in strongest.items():
            summary_parts.append(f"Node {tgt}'s predictive context receives highest variance reduction from Node {src} (C={s:.3f}).")

        summary = " ".join(summary_parts)

        return CausalExplanation(
            causal_scores=clean_scores,
            formatted_causal_scores=formatted,
            strongest_influencer_per_agent=strongest,
            directional_asymmetries=asymmetries,
            epistemological_disclaimer=CAUSAL_EPISTEMOLOGICAL_WARNING,
            grounded_summary=summary
        )

