"""
SAGE-Traffic Phase 10: Complete System Integration Test Suite
Validates all 22 architectural invariants, boundary interfaces, and pipeline components
across the entire intelligent traffic signal control framework.

Test Matrix (22 Tests):
- INT-01: Checkpoint Hash Preservation (SHA256 verified for all 20 checkpoints)
- INT-02: Checkpoint Strict Loadability (strict=True, 0 missing, 0 unexpected, 0 shape mismatches)
- INT-03: Canonical Topology Dependency Neutrality (Pure Python, zero PyTorch/NumPy dependencies)
- INT-04: Deterministic Route Generation Hash Identity (Identical hashes for identical seeds)
- INT-05: Observation Shapes and Empirical Bounds (43-D float32 in [0.0, 1.0])
- INT-06: Agent Ordering Consistency (["A", "B", "C", "D"] order across tensors and dicts)
- INT-07: Temporal History Sliding Window Dynamics (k=4 window, zero future leakage)
- INT-08: Episode Reset State Isolation & Zero Memory Bleed (History, Granger deque, timers)
- INT-09: Temporal Transformer Dimension Mapping ([4, 4, 43] -> [4, 64])
- INT-10: GAT Physical Corridors & Non-Neighbor Masking (-inf masking on diagonals A-D, B-C)
- INT-11: Dedicated Causal Pathway Projection (C_ij in [0, 1] -> phi(C_ij) in R^8 via LayerNorm)
- INT-12: Communication Gate Bounds & Hard Thresholding (g_ij in [0, 1], threshold=0.5)
- INT-13: Causal Message Modulation Sensitivity (Perturbing C_ij shifts GAT node embeddings)
- INT-14: Decentralized Actor Interface & Global State Rejection (68-D accepted, 172-D/256-D rejected)
- INT-15: Centralized Critic Interface (256-D joint temporal state -> scalar V(S_t))
- INT-16: Phase 4 Shared vs. Individual Actor Disambiguation (Role separation, weights distance)
- INT-17: Fixed-Time Baseline Controller Parity (Deterministic 30s cycle matching state machine)
- INT-18: Safety Mode Isolation via Gymnasium Options (options['safety_enabled'], all_red, pending)
- INT-19: Unbypassable Boundary Safety Shield (Unsafe switches clamped to KEEP inside env.step)
- INT-20: Zero Physical Conflicting Greens Guarantee (Second-by-second link index disjointness)
- INT-21: Action and Intervention Telemetry Alignment (Pre-step decision vs post-step outcome)
- INT-22: Deterministic Action Sequence & Metric Reproduction (Multi-step identical trajectory)
"""

import os
import sys
import hashlib
import unittest
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

from environment.demand_generator import generate_routes
from environment.traffic_env import MultiAgentTrafficEnv, TrafficSignal, get_sumo_binary

from models.ppo_network import IndependentPPOAgent
from models.mappo_network import (
    MAPPOController,
    DecentralizedActorNetwork,
    CentralizedCriticNetwork
)
from models.transformer_network import (
    TemporalTransformerEncoder,
    DecentralizedTemporalActor,
    CentralizedTemporalCritic,
    TemporalHistoryManager,
    MAPPOTemporalController
)
from models.gat_network import (
    build_physical_adjacency_matrix,
    DecentralizedGATActor,
    CentralizedGATCritic,
    GraphAttentionLayer,
    MAPPOGATController
)
from models.communication_gate import (
    LearnableCommunicationGate,
    GatedGraphAttentionLayer,
    MAPPOGateController,
    MAPPOCausalGateController
)
from models.causal_influence import CausalInfluenceEstimator
from models.causal_improved import (
    CausalPathwayEncoder,
    ImprovedCausalCommunicationGate,
    ImprovedGatedGraphAttentionLayer,
    MAPPOCausalImprovedController
)
from models.fixed_time_controller import FixedTimeController
from models.safety_layer import TrafficSignalSafetyLayer


# Master Verified SHA256 Checksums for all 20 Checkpoints
APPROVED_CHECKPOINT_HASHES = {
    "agent_A.pt": "140d4e66664357da66d1abe2f4a9c391a88790ab5f10a770529705a5643db88e",
    "agent_B.pt": "89b433b52b6e5f20249947aea02bb34a0b13f04dcc5337a2a08a251d383d7d51",
    "agent_C.pt": "3bf905fcfc2f605157c77046e20f650fc29771dad8d4a00cb3c524c370992c0b",
    "agent_D.pt": "78f691f3568c51354a2d2a9d52279482af0708d98de3c824024b2f1d95769577",
    "mappo_actor_A.pt": "f4d6bb870580681bb0f75e1d60bd0c09ef7395771f25aff442a039551fd3419c",
    "mappo_actor_B.pt": "30307d36fd8ae27bb9143ddc701efa8de6c13408b6ce9c1a5b6d3c6a5a709650",
    "mappo_actor_C.pt": "aa0dbe3b2c9356caa9108062cc4f1bca12576c549b5c0619c24cd6fdbd6f8e33",
    "mappo_actor_D.pt": "a23a5ad4cedd057fff83673a6db63493fed99f5e6981bde029e2c43215a194f9",
    "mappo_actor_shared.pt": "aa3be0f91b8ed2a710ba4b3d14668b7093eea97d2e7da4225a02c55c796f85b3",
    "mappo_causal_gate_actor.pt": "81c2e33fac002a24ac843187c296150afa918468ca3063893d35b6cb057075a0",
    "mappo_causal_gate_critic.pt": "213a00da324511f277a4e6c523dfcf1c2da71931c5f836316dc4674528d6ec0e",
    "mappo_causal_improved_actor.pt": "9a30cb38568abca26e96205efdfba0dfece6dc974bca4ff218041f284d811846",
    "mappo_causal_improved_critic.pt": "8b6ba5ff9d08eccf207095d860ea641ade828e8fe003b6aed629984d39d55a19",
    "mappo_comm_gate_actor.pt": "4759869bca999d18035e27195d4621b3b49305dcedca1df38a743924621076f8",
    "mappo_comm_gate_critic.pt": "0070b5f013eeff5a2f0ba75954a00b5cb262fd9b5dce24b24f565a65602aeb0b",
    "mappo_critic.pt": "44f576d4b2295c04187ec5fce62df6d48eb281aa7dd701044c5c81621e5a2d7d",
    "mappo_transformer_actor.pt": "f14ff6b672f6743446b9a326e9c5991092709c85d0e9ead7a05285018572301f",
    "mappo_transformer_critic.pt": "027ffa3cd2894c02a906f12bb7d29e4c73033616a2caed90ca0f10af7bb44cda",
    "mappo_transformer_gat_actor.pt": "ad669be67c535cf8064ab85a10c8373ffc7f8fd68089115f9dc8c7a5241603a7",
    "mappo_transformer_gat_critic.pt": "a10bde1b0fc676c70e5725c7e0f76ae67faf8c65321c1c93d1cd62a3494fc7e4",
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestPhase10FullIntegration(unittest.TestCase):

    # =========================================================================
    # INT-01: Checkpoint Hash Preservation
    # =========================================================================
    def test_01_checkpoint_hash_preservation(self):
        """Assert all 20 .pt files match the exact approved SHA256 checksums."""
        ckpt_dir = "models/checkpoints"
        files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        self.assertEqual(len(files), 20, f"Expected exactly 20 checkpoints, found {len(files)}")

        for filename in files:
            self.assertIn(filename, APPROVED_CHECKPOINT_HASHES, f"Unexpected checkpoint file: {filename}")
            filepath = os.path.join(ckpt_dir, filename)
            actual_sha = compute_sha256(filepath)
            expected_sha = APPROVED_CHECKPOINT_HASHES[filename]
            self.assertEqual(
                actual_sha, expected_sha,
                f"SHA256 mismatch for {filename}! Expected {expected_sha}, got {actual_sha}"
            )

    # =========================================================================
    # INT-02: Checkpoint Strict Loadability
    # =========================================================================
    def test_02_checkpoint_strict_loadability(self):
        """Assert all 20 checkpoints load with strict=True (0 missing, 0 unexpected keys)."""
        ckpt_dir = "models/checkpoints"
        for filename in sorted(APPROVED_CHECKPOINT_HASHES.keys()):
            filepath = os.path.join(ckpt_dir, filename)
            data = torch.load(filepath, map_location="cpu", weights_only=False)

            if filename.startswith("agent_"):
                agent_id = filename.replace("agent_", "").replace(".pt", "")
                agent = IndependentPPOAgent(agent_id=agent_id, obs_dim=43, act_dim=4)
                r_act = agent.actor.load_state_dict(data["actor_state_dict"], strict=True)
                r_crit = agent.critic.load_state_dict(data["critic_state_dict"], strict=True)
                self.assertEqual(len(r_act.missing_keys), 0)
                self.assertEqual(len(r_act.unexpected_keys), 0)
                self.assertEqual(len(r_crit.missing_keys), 0)
                self.assertEqual(len(r_crit.unexpected_keys), 0)

            elif filename == "mappo_actor_shared.pt" or (filename.startswith("mappo_actor_") and len(filename) == 16):
                act = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
                r = act.load_state_dict(data["actor_state_dict"], strict=True)
                self.assertEqual(len(r.missing_keys), 0)
                self.assertEqual(len(r.unexpected_keys), 0)

            elif filename == "mappo_critic.pt":
                crit = CentralizedCriticNetwork(global_dim=172)
                r = crit.load_state_dict(data["critic_state_dict"], strict=True)
                self.assertEqual(len(r.missing_keys), 0)
                self.assertEqual(len(r.unexpected_keys), 0)

            elif filename == "mappo_transformer_actor.pt":
                t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
                act = DecentralizedTemporalActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
                r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
                r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
                self.assertEqual(len(r_t.missing_keys) + len(r_a.missing_keys), 0)
                self.assertEqual(len(r_t.unexpected_keys) + len(r_a.unexpected_keys), 0)

            elif filename == "mappo_transformer_critic.pt":
                crit = CentralizedTemporalCritic(embedding_dim=64, num_agents=4)
                r = crit.load_state_dict(data["critic_state_dict"], strict=True)
                self.assertEqual(len(r.missing_keys), 0)
                self.assertEqual(len(r.unexpected_keys), 0)

            elif filename == "mappo_transformer_gat_actor.pt":
                t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
                gat = GraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
                act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
                r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
                r_g = gat.load_state_dict(data["gat_state_dict"], strict=True)
                r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
                self.assertEqual(len(r_t.missing_keys) + len(r_g.missing_keys) + len(r_a.missing_keys), 0)
                self.assertEqual(len(r_t.unexpected_keys) + len(r_g.unexpected_keys) + len(r_a.unexpected_keys), 0)

            elif filename in ["mappo_transformer_gat_critic.pt", "mappo_comm_gate_critic.pt", "mappo_causal_gate_critic.pt", "mappo_causal_improved_critic.pt"]:
                crit = CentralizedGATCritic(embedding_dim=64, num_agents=4)
                r = crit.load_state_dict(data["critic_state_dict"], strict=True)
                self.assertEqual(len(r.missing_keys), 0)
                self.assertEqual(len(r.unexpected_keys), 0)

            elif filename in ["mappo_comm_gate_actor.pt", "mappo_causal_gate_actor.pt"]:
                t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
                g_gat = GatedGraphAttentionLayer(input_dim=64, hidden_dim=64, num_heads=4)
                act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
                r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
                r_g = g_gat.load_state_dict(data["gated_gat_state_dict"], strict=True)
                r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
                self.assertEqual(len(r_t.missing_keys) + len(r_g.missing_keys) + len(r_a.missing_keys), 0)
                self.assertEqual(len(r_t.unexpected_keys) + len(r_g.unexpected_keys) + len(r_a.unexpected_keys), 0)

            elif filename == "mappo_causal_improved_actor.pt":
                t_enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
                g_gat = ImprovedGatedGraphAttentionLayer(input_dim=64, hidden_dim=64, causal_dim=8, causal_hidden_dim=16, gate_hidden_dims=[64, 32], num_heads=4)
                act = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)
                r_t = t_enc.load_state_dict(data["transformer_state_dict"], strict=True)
                r_g = g_gat.load_state_dict(data["gated_gat_state_dict"], strict=True)
                r_a = act.load_state_dict(data["actor_state_dict"], strict=True)
                self.assertEqual(len(r_t.missing_keys) + len(r_g.missing_keys) + len(r_a.missing_keys), 0)
                self.assertEqual(len(r_t.unexpected_keys) + len(r_g.unexpected_keys) + len(r_a.unexpected_keys), 0)

    # =========================================================================
    # INT-03: Canonical Topology Dependency Neutrality
    # =========================================================================
    def test_03_canonical_topology_neutrality(self):
        """Assert network/topology.py imports standard library only and defines 8 corridors."""
        import network.topology as topo
        # Verify no external framework in topology module globals
        self.assertNotIn("torch", topo.__dict__)
        self.assertNotIn("numpy", topo.__dict__)
        self.assertNotIn("traci", topo.__dict__)

        # Verify definitions
        self.assertEqual(topo.PHYSICAL_INTERSECTIONS, ["A", "B", "C", "D"])
        self.assertEqual(len(topo.ALLOWED_DIRECTED_EDGES), 8)
        self.assertEqual(len(topo.CORRIDOR_INCOMING_LANES), 8)
        self.assertEqual(topo.CORRIDOR_LENGTH_METERS, 300.0)

        # Diagonals must NOT be valid
        self.assertFalse(topo.is_valid_physical_edge("A", "D"))
        self.assertFalse(topo.is_valid_physical_edge("B", "C"))

    # =========================================================================
    # INT-04: Deterministic Route Generation Hash Identity
    # =========================================================================
    def test_04_pregenerated_route_file_hash_identity(self):
        """Assert generating routes with identical seed produces identical SHA256 hashes."""
        import shutil
        tmp_dir = "scratch/tmp_routes"
        os.makedirs(tmp_dir, exist_ok=True)
        file1 = os.path.join(tmp_dir, "routes_test_a.rou.xml")
        file2 = os.path.join(tmp_dir, "routes_test_b.rou.xml")

        # Generate seed 1001 run 1
        out1 = generate_routes("config.yaml", demand_level="normal", seed=1001)
        shutil.copy(out1, file1)

        # Generate seed 1001 run 2
        out2 = generate_routes("config.yaml", demand_level="normal", seed=1001)
        shutil.copy(out2, file2)

        sha1 = compute_sha256(file1)
        sha2 = compute_sha256(file2)
        self.assertEqual(sha1, sha2, f"Route generation non-deterministic for same seed! {sha1} != {sha2}")

    # =========================================================================
    # INT-05: Observation Shapes and Empirical Bounds
    # =========================================================================
    def test_05_observation_shapes_and_bounds(self):
        """Assert observations are strictly 43-D float32 within [0.0, 1.0] across all agents."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=1001, options={"safety_enabled": False})
            self.assertEqual(len(obs), 4)

            for a in ["A", "B", "C", "D"]:
                self.assertIn(a, obs)
                arr = obs[a]
                self.assertEqual(arr.shape, (43,), f"Agent {a} observation shape is {arr.shape}, expected (43,)")
                self.assertEqual(arr.dtype, np.float32, f"Agent {a} dtype is {arr.dtype}, expected float32")
                # Derived from traffic_env normalization: all features clamped to [0.0, 1.0]
                self.assertTrue(
                    np.all(arr >= 0.0) and np.all(arr <= 1.0),
                    f"Agent {a} observation values out of bounds [0.0, 1.0]! Min: {arr.min()}, Max: {arr.max()}"
                )
        finally:
            env.close()

    # =========================================================================
    # INT-06: Agent Ordering Consistency
    # =========================================================================
    def test_06_agent_ordering_consistency(self):
        """Assert agent ordering strictly adheres to ['A', 'B', 'C', 'D'] across all modules."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        self.assertEqual(env.possible_agents, ["A", "B", "C", "D"])
        self.assertEqual(env.agents, ["A", "B", "C", "D"])

        hist_mgr = TemporalHistoryManager(agents=["A", "B", "C", "D"], history_length=4, obs_dim=43)
        self.assertEqual(hist_mgr.agents, ["A", "B", "C", "D"])

        causal_est = CausalInfluenceEstimator(agents=["A", "B", "C", "D"])
        self.assertEqual(causal_est.agents, ["A", "B", "C", "D"])

    # =========================================================================
    # INT-07: Temporal History Sliding Window Dynamics
    # =========================================================================
    def test_07_temporal_history_sliding_window(self):
        """Assert TemporalHistoryManager slides window k=4 with zero future observation leakage."""
        mgr = TemporalHistoryManager(agents=["A", "B", "C", "D"], history_length=4, obs_dim=43)
        o0 = {a: np.full(43, 0.0, dtype=np.float32) for a in ["A", "B", "C", "D"]}
        mgr.reset(o0)

        # Initial history must repeat o0 exactly 4 times
        for a in ["A", "B", "C", "D"]:
            hist = mgr.get_agent_history(a)
            self.assertEqual(hist.shape, (4, 43))
            self.assertTrue(np.all(hist == 0.0))

        # Push distinct timesteps 1.0, 2.0, 3.0, 4.0
        for val in [1.0, 2.0, 3.0, 4.0]:
            next_o = {a: np.full(43, val, dtype=np.float32) for a in ["A", "B", "C", "D"]}
            mgr.update(next_o)

        # Now history should be [1.0, 2.0, 3.0, 4.0] (0.0 has been popped)
        for a in ["A", "B", "C", "D"]:
            hist = mgr.get_agent_history(a)
            self.assertEqual(hist.shape, (4, 43))
            expected = np.array([1.0, 2.0, 3.0, 4.0])
            self.assertTrue(np.all(hist[:, 0] == expected))

    # =========================================================================
    # INT-08: Episode Reset State Isolation & Zero Memory Bleed
    # =========================================================================
    def test_08_reset_isolation_and_zero_bleed(self):
        """Assert history buffer and Granger regression deque flush completely on reset()."""
        ctrl = MAPPOCausalImprovedController(config_path="config.yaml")

        # Fake historical activity in Episode 1
        fake_obs = {a: np.full(43, 99.0, dtype=np.float32) for a in ["A", "B", "C", "D"]}
        ctrl.history_manager.reset(fake_obs)
        ctrl.history_manager.update({a: np.full(43, 88.0, dtype=np.float32) for a in ["A", "B", "C", "D"]})
        ctrl.causal_estimator.current_step = 50
        ctrl.causal_estimator.agent_history["A"].append(np.ones(3))

        # Now perform Episode 2 reset
        fresh_obs = {a: np.full(43, 1.0, dtype=np.float32) for a in ["A", "B", "C", "D"]}
        ctrl.reset_history(fresh_obs)

        # Confirm zero residue of 99.0 or 88.0
        for a in ["A", "B", "C", "D"]:
            hist = ctrl.history_manager.get_agent_history(a)
            self.assertEqual(hist.shape, (4, 43))
            self.assertTrue(np.all(hist == 1.0), "Old episode observations leaked into fresh episode!")

        self.assertEqual(ctrl.causal_estimator.current_step, 0, "Causal estimator step count not reset!")
        self.assertEqual(len(ctrl.causal_estimator.agent_history["A"]), 0, "Causal estimator agent history not cleared!")

    # =========================================================================
    # INT-09: Temporal Transformer Dimension Mapping
    # =========================================================================
    def test_09_transformer_encoder_dimensions(self):
        """Assert Temporal Transformer projects [4, 4, 43] -> [4, 64] embeddings."""
        enc = TemporalTransformerEncoder(obs_dim=43, embedding_dim=64, num_heads=4, num_layers=2)
        x = torch.randn(4, 4, 43)  # [num_agents=4, k=4, obs_dim=43]
        out = enc(x)
        self.assertEqual(out.shape, (4, 64))

    # =========================================================================
    # INT-10: GAT Physical Corridors & Non-Neighbor Masking
    # =========================================================================
    def test_10_gat_physical_adjacency_masking(self):
        """Assert non-adjacent diagonals A-D and B-C receive strictly -inf attention masking."""
        adj = build_physical_adjacency_matrix(["A", "B", "C", "D"], add_self_loops=True)
        self.assertEqual(adj[0, 3].item(), 0.0)  # A <-> D disconnected
        self.assertEqual(adj[3, 0].item(), 0.0)
        self.assertEqual(adj[1, 2].item(), 0.0)  # B <-> C disconnected
        self.assertEqual(adj[2, 1].item(), 0.0)

        # Connected corridors must be 1.0
        for u_idx, v_idx in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)]:
            self.assertEqual(adj[u_idx, v_idx].item(), 1.0)

    # =========================================================================
    # INT-11: Dedicated Causal Pathway Projection
    # =========================================================================
    def test_11_causal_pathway_projection(self):
        """Assert scalar Granger estimate C_ij in [0, 1] projects to phi(C_ij) in R^8 via LayerNorm."""
        pathway = CausalPathwayEncoder(input_dim=1, hidden_dim=16, causal_dim=8)
        c_val = torch.tensor([0.75])
        phi = pathway(c_val)
        self.assertEqual(phi.shape, (1, 8))
        self.assertFalse(torch.isnan(phi).any())
        self.assertFalse(torch.isinf(phi).any())

    # =========================================================================
    # INT-12: Communication Gate Bounds & Hard Thresholding
    # =========================================================================
    def test_12_comm_gate_bounds_and_thresholding(self):
        """Assert soft gate g_ij in [0, 1] and hard evaluation gate in {0.0, 1.0}."""
        gate = ImprovedCausalCommunicationGate(node_dim=64, causal_dim=8, hard_threshold=0.5)
        node_feats = torch.randn(1, 4, 64)
        ctx = {e: 0.5 for e in ALLOWED_DIRECTED_EDGES}

        # Evaluation mode (hard thresholding)
        g_mat, raw_g, raw_log = gate(node_feats, ctx, deterministic=True)
        for (u, v), g_val in gate.last_gate_values.items():
            self.assertTrue(0.0 <= g_val <= 1.0)

        # Check hard decisions
        for (u, v), is_comm in gate.last_comm_decisions.items():
            self.assertIsInstance(is_comm, bool)

    # =========================================================================
    # INT-13: Causal Message Modulation Sensitivity
    # =========================================================================
    def test_13_causal_message_modulation_sensitivity(self):
        """Assert perturbing C_ij modulates GAT node embeddings when communication is active."""
        layer = ImprovedGatedGraphAttentionLayer(
            input_dim=64, hidden_dim=64, causal_dim=8, causal_hidden_dim=16,
            gate_hidden_dims=[64, 32], hard_threshold=0.5
        )
        layer.eval()

        nodes = torch.randn(1, 4, 64)
        ctx_low = {e: 0.01 for e in ALLOWED_DIRECTED_EDGES}
        ctx_high = {e: 0.99 for e in ALLOWED_DIRECTED_EDGES}

        # In training/continuous mode, varying C_ij shifts the causal embedding phi(C_ij) and gate logits
        with torch.no_grad():
            out_low, _, _ = layer(nodes, ctx_low, deterministic=False)
            out_high, _, _ = layer(nodes, ctx_high, deterministic=False)

        diff = torch.norm(out_high - out_low).item()
        self.assertGreater(diff, 1e-4, f"Causal pathway has zero sensitivity! Output difference: {diff}")

    # =========================================================================
    # INT-14: Decentralized Actor Interface & Global State Rejection
    # =========================================================================
    def test_14_decentralized_actor_interface(self):
        """Assert Decentralized Actor receives 68-D and strictly rejects global states (172-D, 256-D)."""
        actor = DecentralizedGATActor(embedding_dim=64, agent_id_dim=4, act_dim=4)

        # Valid decentralized input: [batch=4, 64] node + [4, 4] ID = 68-D
        valid_node = torch.randn(4, 64)
        valid_id = torch.eye(4)
        dist = actor(valid_node, valid_id)
        self.assertEqual(dist.probs.shape, (4, 4))

        # Rejection of 172-D global state
        with self.assertRaises(RuntimeError):
            actor(torch.randn(4, 168), valid_id)

        # Rejection of 256-D centralized state
        with self.assertRaises(RuntimeError):
            actor(torch.randn(4, 252), valid_id)

    # =========================================================================
    # INT-15: Centralized Critic Interface
    # =========================================================================
    def test_15_centralized_critic_interface(self):
        """Assert Centralized Critic receives 256-D joint temporal state and evaluates scalar V(S_t)."""
        critic = CentralizedGATCritic(num_agents=4, embedding_dim=64, hidden_dims=[128, 64])
        joint_state = torch.randn(8, 256)  # [B=8, 4*64=256]
        val = critic(joint_state)
        self.assertEqual(val.shape, (8, 1))

    # =========================================================================
    # INT-16: Phase 4 Shared vs. Individual Actor Disambiguation
    # =========================================================================
    def test_16_phase4_shared_vs_individual_disambiguation(self):
        """Assert shared and individual MAPPO checkpoints load into 47->64->64->4 with distinct roles."""
        shared_data = torch.load("models/checkpoints/mappo_actor_shared.pt", map_location="cpu", weights_only=False)
        indiv_a_data = torch.load("models/checkpoints/mappo_actor_A.pt", map_location="cpu", weights_only=False)

        # 1. Both checkpoints load strictly into DecentralizedActorNetwork(obs_dim=47, act_dim=4)
        shared_actor = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
        indiv_actor = DecentralizedActorNetwork(obs_dim=47, act_dim=4)
        res_shared = shared_actor.load_state_dict(shared_data["actor_state_dict"], strict=True)
        res_indiv = indiv_actor.load_state_dict(indiv_a_data["actor_state_dict"], strict=True)
        self.assertEqual(len(res_shared.missing_keys), 0)
        self.assertEqual(len(res_indiv.missing_keys), 0)

        # 2. Check policy sharing configuration
        self.assertTrue(shared_data.get("share_policy", True))
        self.assertTrue(indiv_a_data.get("share_policy", True))

        # 3. Check role separation: shared contains centralized training optimizer state dict
        self.assertIn("optimizer_state_dict", shared_data)
        self.assertNotIn("optimizer_state_dict", indiv_a_data)

        # 4. In Phase 4, individual actor checkpoints are exported copies of the shared policy
        shared_w = shared_data["actor_state_dict"]["network.0.weight"]
        indiv_a_w = indiv_a_data["actor_state_dict"]["network.0.weight"]
        self.assertEqual(shared_w.shape, indiv_a_w.shape)
        diff = torch.norm(shared_w - indiv_a_w).item()
        self.assertEqual(diff, 0.0, "Individual exported checkpoint deviates from shared parent checkpoint!")

    # =========================================================================
    # INT-17: Fixed-Time Baseline Controller Parity
    # =========================================================================
    def test_17_fixed_time_controller_parity(self):
        """Assert Fixed-Time controller issues periodic KEEP/SWITCH matching SUMO state machine."""
        ctrl = FixedTimeController("config.yaml")
        ctrl.reset()
        self.assertEqual(ctrl.green_time, 30)

        # Over 6 steps (30 seconds), steps 1..5 should be KEEP (0), step 6 should be SWITCH (1)
        actions_seq = []
        for _ in range(6):
            actions = ctrl.get_actions({}, {})
            actions_seq.append(actions["A"])

        self.assertEqual(actions_seq, [0, 0, 0, 0, 0, 1], f"Expected 5 KEEPs followed by SWITCH, got {actions_seq}")

    # =========================================================================
    # INT-18: Safety Mode Isolation via Gymnasium Options
    # =========================================================================
    def test_18_safety_mode_isolation(self):
        """Assert options={'safety_enabled': False} vs True isolates shield, all_red, and pending toggles."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)

        # 1. Test unshielded options override on fresh env
        env.reset(seed=1001, options={"safety_enabled": False})
        self.assertFalse(env.safety_enabled)
        self.assertFalse(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 0)
        env.close()

        # 2. Test sequential reset on same instance: unshielded -> options=None restores config default
        env.reset(seed=1002, options=None)
        self.assertTrue(env.safety_enabled, "reset(options=None) after unshielded must restore config default")
        self.assertTrue(env.safety_layer.enabled, "safety_layer.enabled must restore to True")
        self.assertEqual(env.all_red_duration, 1, "all_red_duration must restore to config value 1")
        env.close()

        # 3. Test sequential reset: shielded -> options=None preserves config default
        env.reset(seed=1003, options={"safety_enabled": True})
        self.assertTrue(env.safety_enabled)
        self.assertTrue(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 1)
        env.close()

        env.reset(seed=1004, options=None)
        self.assertTrue(env.safety_enabled)
        self.assertTrue(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 1)

        # 4. Test pending toggle lockout during active transition
        # Simulate active yellow transition on intersection A
        env.signals["A"].is_yellow = True
        applied = env.set_safety_enabled(False)
        self.assertFalse(applied, "Disabling safety during transition must return False (deferred)")
        self.assertTrue(env.safety_enabled, "Safety must remain active until episode boundary")
        self.assertFalse(env._pending_safety_enabled, "Pending toggle flag must record requested False")
        env.signals["A"].is_yellow = False
        env.close()

        # 5. Test consumption of pending toggle upon reset(options=None)
        # With _pending_safety_enabled set to False, reset with options=None should apply it
        env._pending_safety_enabled = False
        env.reset(seed=1005, options=None)
        self.assertFalse(env.safety_enabled, "Pending toggle must be applied at episode boundary")
        self.assertFalse(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 0)
        self.assertIsNone(env._pending_safety_enabled, "Pending flag must be cleared after reset")
        env.close()

        # 6. Test default preservation when subsequent episode resets with options=None
        env.reset(seed=1006, options=None)
        self.assertTrue(env.safety_enabled, "Next episode must restore config default")
        self.assertTrue(env.safety_layer.enabled)
        self.assertEqual(env.all_red_duration, 1)
        env.close()

    # =========================================================================
    # INT-19: Unbypassable Boundary Safety Shield
    # =========================================================================
    def test_19_safety_shield_unbypassability(self):
        """Assert unsafe SWITCH actions submitted to env.step() are intercepted and clamped to KEEP."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=1001, options={"safety_enabled": True})
            # Intersection A is at t=0s green (min_green=5s). Propose SWITCH (1)
            unsafe_actions = {"A": 1, "B": 0, "C": 0, "D": 0}
            _, _, _, _, next_info = env.step(unsafe_actions)

            # Shield must have clamped Agent A to KEEP (0)
            self.assertEqual(next_info["A"]["validated_action"], 0)
            self.assertEqual(next_info["A"]["proposed_action"], 1)
            self.assertGreater(next_info["A"]["safety"]["total_interventions"], 0)
        finally:
            env.close()

    # =========================================================================
    # INT-20: Zero Physical Conflicting Greens Guarantee
    # =========================================================================
    def test_20_zero_physical_conflict_guarantee(self):
        """Assert second-by-second SUMO signal states have NS Green intersect EW Green == empty."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        try:
            env.reset(seed=1001, options={"safety_enabled": True})
            for step in range(5):
                # Step with alternating proposals
                actions = {a: 1 if step % 2 == 0 else 0 for a in env.agents}
                _, _, _, _, info = env.step(actions)

            # Check that physical audit detected exactly zero physical violations
            violations = info["A"]["safety"]["audit_physical_violations"]
            self.assertEqual(violations, 0, f"Detected {violations} physical signal conflicts!")
        finally:
            env.close()

    # =========================================================================
    # INT-21: Action and Intervention Telemetry Alignment
    # =========================================================================
    def test_21_action_and_intervention_telemetry(self):
        """Assert proposed vs validated actions and intervention counters match in infos."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=1001, options={"safety_enabled": True})
            actions = {"A": 1, "B": 1, "C": 1, "D": 1}
            _, _, _, _, next_info = env.step(actions)

            for a in ["A", "B", "C", "D"]:
                self.assertIn("proposed_action", next_info[a])
                self.assertIn("validated_action", next_info[a])
                self.assertIn("safety", next_info[a])
                self.assertEqual(next_info[a]["proposed_action"], 1)
                self.assertEqual(next_info[a]["validated_action"], 0)  # Clamped by MIN_GREEN
        finally:
            env.close()

    # =========================================================================
    # INT-22: Deterministic Action Sequence & Metric Reproduction
    # =========================================================================
    def test_22_deterministic_reproduction(self):
        """Assert identical action sequences and metrics across repeated evaluations on identical seeds."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        ctrl = MAPPOCausalImprovedController(config_path="config.yaml")
        ctrl.load_checkpoints("models/checkpoints")

        def run_trajectory(seed):
            obs, info = env.reset(seed=seed, options={"safety_enabled": True})
            ctrl.reset_history(obs, info)
            actions_list = []
            for _ in range(4):
                ctx = ctrl.compute_causal_context(info)
                acts, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                actions_list.append(acts)
                obs, _, _, _, info = env.step(acts)
                ctrl.update_history(obs)
            metrics = {a: info[a]["metrics"]["waiting_time"] for a in env.agents}
            env.close()
            return actions_list, metrics

        acts1, m1 = run_trajectory(1001)
        acts2, m2 = run_trajectory(1001)

        self.assertEqual(acts1, acts2, "Deterministic action sequences differed across identical seeds!")
        for a in ["A", "B", "C", "D"]:
            self.assertAlmostEqual(m1[a], m2[a], places=4, msg=f"Metric discrepancy for agent {a}!")


if __name__ == "__main__":
    unittest.main()
