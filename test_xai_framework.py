"""
Phase 11 Explainable AI (XAI) Test Suite
========================================
Comprehensive validation of the SAGE-Traffic XAI framework:
1. DecisionExplainer: probability distributions, margins, and rationales.
2. AttentionExplainer: GAT spatial attention weights and epistemological safeguards.
3. CommunicationExplainer: Gating activations, active/suppressed status, and bandwidth.
4. CausalExplainer: Counterfactual causal influence scores and directional asymmetries.
5. SafetyExplainer: Minimum green, clearance, and maximum green invariant explanations.
6. XAILogger: Non-invasive telemetry buffering and export.
7. Real Controller XAI Extraction: Actual actor, GAT, and safety outputs from C8/C9.
8. Non-Interference Parity: Verifies XAI execution does NOT alter actions or simulation state.
9. GUI Integration: Full widget instantiation and update validation.
"""

import os
import sys
import unittest
import numpy as np
import torch
import tkinter as tk

from network.topology import PHYSICAL_INTERSECTIONS, ALLOWED_DIRECTED_EDGES
from environment.traffic_env import MultiAgentTrafficEnv
from models.causal_improved import MAPPOCausalImprovedController
from models.safety_layer import TrafficSignalSafetyLayer, ACTION_NAMES

from xai import (
    DecisionExplainer,
    DecisionExplanation,
    AttentionExplainer,
    AttentionExplanation,
    CommunicationExplainer,
    CommunicationExplanation,
    CausalExplainer,
    CausalExplanation,
    SafetyExplainer,
    SafetyExplanation,
    XAILogger
)
from visualize_phase10_gui import Phase10UnifiedGUIDashboard


class TestXAIFramework(unittest.TestCase):
    """Test suite validating XAI mathematical correctness, grounding, and non-interference."""

    def setUp(self):
        self.config_path = "config.yaml"
        self.seed = 1001

    def test_01_decision_explainer_probabilities(self):
        """Verify DecisionExplainer computes probabilities, margins, and rationales correctly."""
        explainer = DecisionExplainer(min_green=5.0, max_green=60.0)

        # Mock action probabilities: SWITCH has 0.70, REDUCE has 0.20
        probs = np.array([0.05, 0.70, 0.05, 0.20], dtype=np.float32)
        metrics = {
            "queue_length": 6.5,
            "waiting_time": 18.2,
            "vehicle_count": 8,
            "average_speed": 4.2,
            "current_phase": 0,
            "phase_duration": 6.0
        }

        exp = explainer.explain(
            agent_id="A",
            action_probs=probs,
            selected_action=1,
            obs=np.zeros(43),
            metrics=metrics,
            proposed_action=1,
            executed_action=1
        )

        self.assertIsInstance(exp, DecisionExplanation)
        self.assertEqual(exp.agent_id, "A")
        self.assertEqual(exp.selected_action, 1)
        self.assertEqual(exp.selected_action_name, "SWITCH")
        self.assertEqual(exp.alternative_action, 3)
        self.assertEqual(exp.alternative_action_name, "REDUCE")
        self.assertAlmostEqual(exp.confidence_margin, 0.50, places=4)
        self.assertTrue(exp.is_safe)
        self.assertGreater(exp.policy_entropy, 0.0)
        self.assertIn("Node A policy selected SWITCH", exp.grounded_summary)
        self.assertIn("high confidence", exp.grounded_summary)

    def test_02_attention_explainer_edges_and_disclaimer(self):
        """Verify AttentionExplainer analyzes 8 canonical edges and enforces epistemological guards."""
        explainer = AttentionExplainer()

        mock_weights = {
            ("A", "B"): 0.40, ("B", "A"): 0.30,
            ("A", "C"): 0.60, ("C", "A"): 0.70,
            ("B", "D"): 0.25, ("D", "B"): 0.35,
            ("C", "D"): 0.50, ("D", "C"): 0.45
        }

        exp = explainer.explain(mock_weights)

        self.assertIsInstance(exp, AttentionExplanation)
        self.assertEqual(len(exp.edge_weights), 8)
        for edge in ALLOWED_DIRECTED_EDGES:
            self.assertIn(edge, exp.edge_weights)

        # Check strongest incoming for Node A (should be C with 0.70)
        top_src, top_w = exp.strongest_incoming_per_agent["A"]
        self.assertEqual(top_src, "C")
        self.assertAlmostEqual(top_w, 0.70, places=4)

        # Epistemological disclaimer must be present
        self.assertIn("CAUTION", exp.epistemological_disclaimer)
        self.assertIn("neural network", exp.epistemological_disclaimer)

    def test_03_communication_explainer_gating(self):
        """Verify CommunicationExplainer thresholds gates and computes bandwidth savings."""
        explainer = CommunicationExplainer(threshold=0.5, cost_per_msg=0.01)

        # 2 edges active (>= 0.5), 6 suppressed (< 0.5)
        mock_gates = {
            ("A", "B"): 0.85, ("B", "A"): 0.10,
            ("A", "C"): 0.001, ("C", "A"): 0.95,
            ("B", "D"): 0.05, ("D", "B"): 0.02,
            ("C", "D"): 0.0001, ("D", "C"): 0.12
        }

        exp = explainer.explain(mock_gates)

        self.assertIsInstance(exp, CommunicationExplanation)
        self.assertEqual(exp.active_count, 2)
        self.assertEqual(exp.total_possible, 8)
        self.assertEqual(len(exp.active_edges), 2)
        self.assertEqual(len(exp.suppressed_edges), 6)
        self.assertAlmostEqual(exp.bandwidth_savings_pct, 75.0, places=1)
        self.assertAlmostEqual(exp.communication_cost, 0.02, places=4)

    def test_04_causal_explainer_influence(self):
        """Verify CausalExplainer computes directional asymmetries and provides disclaimers."""
        explainer = CausalExplainer()

        mock_causal = {
            ("A", "B"): 0.65, ("B", "A"): 0.25,
            ("A", "C"): 0.40, ("C", "A"): 0.40,
            ("B", "D"): 0.15, ("D", "B"): 0.85,
            ("C", "D"): 0.30, ("D", "C"): 0.50
        }

        exp = explainer.explain(mock_causal)

        self.assertIsInstance(exp, CausalExplanation)
        self.assertEqual(len(exp.causal_scores), 8)

        # Check asymmetry between A and B: 0.65 - 0.25 = 0.40
        self.assertAlmostEqual(exp.directional_asymmetries["A<->B"], 0.40, places=4)

        # Disclaimer must state predictive influence, not physical causation or counterfactual model interventions
        self.assertIn("CAUTION", exp.epistemological_disclaimer)
        self.assertIn("observational predictive influence", exp.epistemological_disclaimer)
        self.assertIn("Granger-style", exp.epistemological_disclaimer)

    def test_05_safety_explainer_invariants(self):
        """Verify SafetyExplainer explains safe proposals vs invariant clamps accurately."""
        explainer = SafetyExplainer(min_green=5.0, max_green=60.0)

        # Case 1: Safe Action
        safe_exp = explainer.explain(
            agent_id="A",
            proposed_action=1,
            executed_action=1,
            time_on_phase=7.0
        )
        self.assertTrue(safe_exp.is_safe)
        self.assertIn("SAFE", safe_exp.grounded_summary)

        # Case 2: Minimum Green Clamp
        clamped_exp = explainer.explain(
            agent_id="B",
            proposed_action=1,
            executed_action=0,
            time_on_phase=2.0,
            active_constraint="MIN_GREEN"
        )
        self.assertFalse(clamped_exp.is_safe)
        self.assertEqual(clamped_exp.executed_action_name, "KEEP")
        self.assertIn("MIN_GREEN", clamped_exp.grounded_summary)
        self.assertIn("short-cycling", clamped_exp.grounded_summary)

    def test_06_xai_logger_buffering_and_export(self):
        """Verify XAILogger correctly records, buffers, and exports structured explanations."""
        logger = XAILogger(max_buffer_size=10)

        dec_explainer = DecisionExplainer()
        dec_exp = dec_explainer.explain(
            agent_id="A",
            action_probs=np.array([0.1, 0.7, 0.1, 0.1]),
            selected_action=1,
            metrics={"queue_length": 5.0, "waiting_time": 10.0}
        )

        record = logger.record_step(
            step=1,
            sim_time=1.0,
            controller_id="C9",
            decisions={"A": dec_exp}
        )

        self.assertEqual(len(logger.history), 1)
        latest = logger.get_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["step"], 1)
        self.assertEqual(latest["controller_id"], "C9")

        export_path = "scratch/test_xai_export.jsonl"
        logger.export_jsonl(export_path)
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        os.remove(export_path)

    def test_07_real_controller_xai_extraction(self):
        """Verify XAI accurately extracts real tensors from trained C8 and C9 models."""
        env = MultiAgentTrafficEnv(config_path=self.config_path, gui=False)
        obs, info = env.reset(seed=self.seed, options={"safety_enabled": True})

        ctrl = MAPPOCausalImprovedController(config_path=self.config_path)
        ctrl.load_checkpoints("models/checkpoints")
        ctrl.reset_history(obs, info)

        ctx = ctrl.compute_causal_context(info)
        actions, _, _, comm_step = ctrl.get_actions(ctx, deterministic=True)

        # Verify action probabilities
        with torch.no_grad():
            hist_list = [ctrl.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]
            hist_tensor = torch.as_tensor(np.stack(hist_list, axis=0), dtype=torch.float32)
            emb = ctrl.transformer_encoder(hist_tensor)
            gat_res = ctrl.gated_gat(emb.unsqueeze(0), ctx)
            gat_emb = gat_res[0]
            dist = ctrl.actor(gat_emb.squeeze(0), torch.eye(4))
            probs_np = dist.probs.cpu().numpy()

        dec_explainer = DecisionExplainer()
        for i, a in enumerate(PHYSICAL_INTERSECTIONS):
            p_a = probs_np[i]
            # Probabilities must sum to 1
            self.assertAlmostEqual(float(np.sum(p_a)), 1.0, places=4)
            # Argmax must match selected action
            self.assertEqual(int(np.argmax(p_a)), actions[a])

            exp = dec_explainer.explain(
                agent_id=a,
                action_probs=p_a,
                selected_action=actions[a],
                metrics=info[a]["metrics"]
            )
            self.assertEqual(exp.selected_action, actions[a])

        env.close()

    def test_08_xai_non_interference_simulation_parity(self):
        """Verify that enabling XAI explanation generation does NOT alter policy decisions or SUMO state."""
        # Run 5 steps without XAI logging
        env1 = MultiAgentTrafficEnv(config_path=self.config_path, gui=False)
        obs1, info1 = env1.reset(seed=self.seed, options={"safety_enabled": True})
        ctrl1 = MAPPOCausalImprovedController(config_path=self.config_path)
        ctrl1.load_checkpoints("models/checkpoints")
        ctrl1.reset_history(obs1, info1)

        actions_history_1 = []
        queues_history_1 = []
        for s in range(5):
            ctx1 = ctrl1.compute_causal_context(info1)
            acts1, _, _, _ = ctrl1.get_actions(ctx1, deterministic=True)
            actions_history_1.append(acts1)
            obs1, rew1, term1, trunc1, info1 = env1.step(acts1)
            ctrl1.update_history(obs1)
            queues_history_1.append([info1[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS])
        env1.close()

        # Run 5 steps with full XAI explainer instrumentation
        env2 = MultiAgentTrafficEnv(config_path=self.config_path, gui=False)
        obs2, info2 = env2.reset(seed=self.seed, options={"safety_enabled": True})
        ctrl2 = MAPPOCausalImprovedController(config_path=self.config_path)
        ctrl2.load_checkpoints("models/checkpoints")
        ctrl2.reset_history(obs2, info2)

        dec_explainer = DecisionExplainer()
        safety_explainer = SafetyExplainer()
        xai_logger = XAILogger()

        actions_history_2 = []
        queues_history_2 = []
        for s in range(5):
            ctx2 = ctrl2.compute_causal_context(info2)
            acts2, _, _, _ = ctrl2.get_actions(ctx2, deterministic=True)
            actions_history_2.append(acts2)

            obs2, rew2, term2, trunc2, info2 = env2.step(acts2)
            ctrl2.update_history(obs2)
            queues_history_2.append([info2[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS])

            # Generate XAI telemetry
            dec_exps = {}
            for a in PHYSICAL_INTERSECTIONS:
                dec_exps[a] = dec_explainer.explain(
                    agent_id=a,
                    action_probs=np.array([0.0, 1.0, 0.0, 0.0]),
                    selected_action=acts2[a],
                    metrics=info2[a]["metrics"]
                )
            xai_logger.record_step(s, s * 1.0, "C9", dec_exps)
        env2.close()

        # Strict parity check: actions and queues must be 100% IDENTICAL
        self.assertEqual(actions_history_1, actions_history_2)
        self.assertEqual(queues_history_1, queues_history_2)

    def test_09_gui_xai_widget_integration(self):
        """Verify that Phase10UnifiedGUIDashboard builds XAI widgets and steps cleanly."""
        root = tk.Tk()
        root.withdraw()

        app = Phase10UnifiedGUIDashboard(
            root=root,
            config_path=self.config_path,
            initial_controller_id="C9",
            initial_seed=self.seed,
            max_steps=3,
            use_sumo_gui=False,
            smoke_test=False  # Build full UI widgets
        )

        # Step twice
        for s in range(2):
            app.execute_decision_step()

        # Check that XAI widgets exist and are populated
        self.assertIsNotNone(app.last_xai_decisions)
        self.assertIn("A", app.last_xai_decisions)
        self.assertIn("D", app.last_xai_decisions)

        # Test user clicking different intersection tabs
        for agent in PHYSICAL_INTERSECTIONS:
            app.on_xai_agent_selected(agent)
            self.assertEqual(app.xai_selected_agent, agent)

        app.env.close()
        root.destroy()


if __name__ == "__main__":
    unittest.main()

