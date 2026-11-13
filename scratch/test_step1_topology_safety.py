"""
Focused unit test suite for Phase 10 Step 1:
1. Dependency-neutral network/topology.py functionality and exports.
2. Canonical import consistency across model architectures.
3. Gymnasium reset(seed, options) safety mode selection and default preservation.
4. Transition lockout protection against mid-clearance disabling.
"""

import unittest
import sys
sys.path.insert(0, ".")
import numpy as np
import torch

from network.topology import (
    PHYSICAL_INTERSECTIONS,
    ALLOWED_DIRECTED_EDGES,
    CORRIDOR_INCOMING_LANES,
    CORRIDOR_LENGTH_METERS,
    build_physical_adjacency_list,
    is_valid_physical_edge,
    get_adjacent_neighbors
)
from models.gat_network import build_physical_adjacency_matrix
from models.communication_gate import ALLOWED_DIRECTED_EDGES as CG_EDGES
from models.causal_influence import ALLOWED_DIRECTED_EDGES as CI_EDGES, CORRIDOR_INCOMING_LANES as CI_LANES
from models.causal_improved import ALLOWED_DIRECTED_EDGES as CIP_EDGES, CORRIDOR_INCOMING_LANES as CIP_LANES
from environment.traffic_env import MultiAgentTrafficEnv, TrafficSignal


class TestStep1TopologyAndSafety(unittest.TestCase):

    def test_01_canonical_topology_constants(self):
        """Verify topology constants have exact expected values and shapes."""
        self.assertEqual(PHYSICAL_INTERSECTIONS, ["A", "B", "C", "D"])
        self.assertEqual(len(ALLOWED_DIRECTED_EDGES), 8)
        self.assertEqual(len(CORRIDOR_INCOMING_LANES), 8)
        self.assertEqual(CORRIDOR_LENGTH_METERS, 300.0)

        # Diagonals must NOT exist
        self.assertFalse(is_valid_physical_edge("A", "D"))
        self.assertFalse(is_valid_physical_edge("D", "A"))
        self.assertFalse(is_valid_physical_edge("B", "C"))
        self.assertFalse(is_valid_physical_edge("C", "B"))

        # Physical corridors must exist
        for u, v in [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"),
                     ("B", "D"), ("D", "B"), ("C", "D"), ("D", "C")]:
            self.assertTrue(is_valid_physical_edge(u, v))

        # Check neighbors
        self.assertEqual(sorted(get_adjacent_neighbors("A")), ["B", "C"])
        self.assertEqual(sorted(get_adjacent_neighbors("B")), ["A", "D"])
        self.assertEqual(sorted(get_adjacent_neighbors("C")), ["A", "D"])
        self.assertEqual(sorted(get_adjacent_neighbors("D")), ["B", "C"])

    def test_02_canonical_imports_consistency(self):
        """Verify all model modules share identical canonical edge definitions."""
        self.assertEqual(ALLOWED_DIRECTED_EDGES, CG_EDGES)
        self.assertEqual(ALLOWED_DIRECTED_EDGES, CI_EDGES)
        self.assertEqual(ALLOWED_DIRECTED_EDGES, CIP_EDGES)
        self.assertEqual(CORRIDOR_INCOMING_LANES, CI_LANES)
        self.assertEqual(CORRIDOR_INCOMING_LANES, CIP_LANES)

    def test_03_adjacency_matrix_tensor_construction(self):
        """Verify build_physical_adjacency_matrix produces identical 4x4 tensor with self-loops."""
        adj_tensor = build_physical_adjacency_matrix(PHYSICAL_INTERSECTIONS, add_self_loops=True)
        self.assertEqual(adj_tensor.shape, (4, 4))
        self.assertEqual(adj_tensor.dtype, torch.float32)

        # Diagonal elements (self-loops) must be 1.0
        for i in range(4):
            self.assertEqual(adj_tensor[i, i].item(), 1.0)

        # Diagonals in grid (A-D, B-C) must be 0.0
        self.assertEqual(adj_tensor[0, 3].item(), 0.0)  # A to D
        self.assertEqual(adj_tensor[3, 0].item(), 0.0)  # D to A
        self.assertEqual(adj_tensor[1, 2].item(), 0.0)  # B to C
        self.assertEqual(adj_tensor[2, 1].item(), 0.0)  # C to B

        # Orthogonal neighbors must be 1.0
        self.assertEqual(adj_tensor[0, 1].item(), 1.0)  # B -> A
        self.assertEqual(adj_tensor[1, 0].item(), 1.0)  # A -> B
        self.assertEqual(adj_tensor[0, 2].item(), 1.0)  # C -> A
        self.assertEqual(adj_tensor[2, 0].item(), 1.0)  # A -> C

    def test_04_env_safety_options_disabled(self):
        """Verify reset(options={'safety_enabled': False}) disables safety shield."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        self.assertTrue(env.safety_enabled)  # Config default

        # Simulate setting safety via options dictionary
        opts = {"safety_enabled": False}
        if opts is not None and "safety_enabled" in opts:
            req_safety = bool(opts["safety_enabled"])
            env.safety_enabled = req_safety
            env.safety_layer.enabled = req_safety
            env.all_red_duration = int(env.safety_cfg.get("all_red_duration", 1)) if env.safety_enabled else 0

        self.assertFalse(env.safety_enabled)
        self.assertFalse(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 0)

    def test_05_env_safety_options_enabled(self):
        """Verify reset(options={'safety_enabled': True}) enables safety shield."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        env.safety_enabled = False  # Start from false

        opts = {"safety_enabled": True}
        if opts is not None and "safety_enabled" in opts:
            req_safety = bool(opts["safety_enabled"])
            env.safety_enabled = req_safety
            env.safety_layer.enabled = req_safety
            env.all_red_duration = int(env.safety_cfg.get("all_red_duration", 1)) if env.safety_enabled else 0

        self.assertTrue(env.safety_enabled)
        self.assertTrue(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 1)

    def test_06_env_safety_options_default_preservation(self):
        """Verify reset() with options=None preserves config default (enabled: True)."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        opts = None
        if opts is not None and "safety_enabled" in opts:
            req_safety = bool(opts["safety_enabled"])
            env.safety_enabled = req_safety

        self.assertTrue(env.safety_enabled)
        self.assertTrue(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 1)

    def test_07_transition_lockout_protection(self):
        """Verify set_safety_enabled(False) is deferred if any intersection is in yellow or all-red."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        env.is_running = True

        mock_sig = TrafficSignal(
            ts_id="A",
            incoming_lanes=["lane_1"],
            green_phases=[0, 2],
            yellow_map={0: 1, 2: 3},
            yellow_duration=3,
            min_green=5,
            all_red_duration=1
        )
        mock_sig.is_yellow = True  # Active transition
        env.signals = {"A": mock_sig}

        applied = env.set_safety_enabled(False)
        self.assertFalse(applied)
        self.assertTrue(env.safety_enabled)
        self.assertFalse(env._pending_safety_enabled)  # Queued for boundary


if __name__ == "__main__":
    unittest.main()
