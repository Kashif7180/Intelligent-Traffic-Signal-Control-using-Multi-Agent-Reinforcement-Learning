"""
Dependency-neutral physical network topology definitions for SAGE-Traffic.

Defines the canonical physical adjacency, bidirectional internal corridors,
lane mappings, and helper functions for the 2x2 grid traffic network.

Strict Architectural Rule:
- Zero external dependencies (Python standard library only).
- No imports from PyTorch, NumPy, TraCI, gymnasium, or any models/environment module.
- All neural tensor conversions are performed in consumer modules.
"""

from typing import List, Tuple, Dict, Optional

# Canonical intersection list in physical network order
PHYSICAL_INTERSECTIONS: List[str] = ["A", "B", "C", "D"]

# Exactly 8 allowed physical directed inter-agent communication channels
# Corridors: A <-> B (horizontal), A <-> C (vertical), B <-> D (vertical), C <-> D (horizontal)
# Note: Diagonals A <-> D and B <-> C do not exist physically in the 2x2 grid.
ALLOWED_DIRECTED_EDGES: List[Tuple[str, str]] = [
    ("A", "B"), ("B", "A"),
    ("A", "C"), ("C", "A"),
    ("B", "D"), ("D", "B"),
    ("C", "D"), ("D", "C")
]

# Physical corridor incoming lane mapping at target from source
CORRIDOR_INCOMING_LANES: Dict[Tuple[str, str], List[str]] = {
    ("A", "B"): ["A_to_B_0", "A_to_B_1"],
    ("B", "A"): ["B_to_A_0", "B_to_A_1"],
    ("A", "C"): ["A_to_C_0", "A_to_C_1"],
    ("C", "A"): ["C_to_A_0", "C_to_A_1"],
    ("B", "D"): ["B_to_D_0", "B_to_D_1"],
    ("D", "B"): ["D_to_B_0", "D_to_B_1"],
    ("C", "D"): ["C_to_D_0", "C_to_D_1"],
    ("D", "C"): ["D_to_C_0", "D_to_C_1"]
}

# Physical corridor metrics
CORRIDOR_LENGTH_METERS: float = 300.0
BOUNDARY_LENGTH_METERS: float = 200.0


def build_physical_adjacency_list(
    agents: Optional[List[str]] = None,
    add_self_loops: bool = True
) -> List[List[float]]:
    """
    Build 2D adjacency matrix as a pure Python nested list of floats (0.0 or 1.0).
    Row = target intersection i, Column = source intersection j (flow j -> i).

    Args:
        agents: List of agent IDs. Defaults to PHYSICAL_INTERSECTIONS (["A", "B", "C", "D"]).
        add_self_loops: Whether diagonal elements (self-loops) are set to 1.0.

    Returns:
        List[List[float]] of shape (N, N) where N = len(agents).
    """
    if agents is None:
        agents = PHYSICAL_INTERSECTIONS

    idx_map = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    adj = [[0.0 for _ in range(n)] for _ in range(n)]

    if add_self_loops:
        for i in range(n):
            adj[i][i] = 1.0

    for u, v in ALLOWED_DIRECTED_EDGES:
        if u in idx_map and v in idx_map:
            # Flow from u (source) to v (target): row = target v, col = source u
            adj[idx_map[v]][idx_map[u]] = 1.0

    return adj


def is_valid_physical_edge(src: str, tgt: str) -> bool:
    """Return True if (src, tgt) corresponds to a physical corridor in the 2x2 grid."""
    return (src, tgt) in ALLOWED_DIRECTED_EDGES


def get_adjacent_neighbors(node: str) -> List[str]:
    """Return list of physically connected neighbor intersections for a given intersection."""
    return [tgt for src, tgt in ALLOWED_DIRECTED_EDGES if src == node]

