"""
SAGE-Traffic Phase 13 Step 1: Comprehensive End-to-End System Validation Suite
=============================================================================
Final system-wide validation covering:
1. Controller Architecture & Model Verification (C1-C9 + A1, A2)
2. SUMO Environment TraCI Lifecycle & Simulation Progression
3. Phase 9 Safety Shield Invariants & Edge Case Enforcement
4. Unified GUI Dashboard & XAI Explainability Framework
5. Repeated Execution Stability, Reset Isolation & Resource Teardown
6. Checkpoint & Historical Artifact Bitwise Integrity Verification
"""

import os
import sys
import gc
import csv
import time
import hashlib
import unittest
import subprocess
import tracemalloc
import numpy as np
import torch

# Workspace imports
from network.topology import (
    PHYSICAL_INTERSECTIONS,
    ALLOWED_DIRECTED_EDGES,
    get_adjacent_neighbors,
    is_valid_physical_edge
)
from environment.traffic_env import MultiAgentTrafficEnv, get_sumo_binary, TrafficSignal
from models.safety_layer import TrafficSignalSafetyLayer, ACTION_NAMES
from benchmark_phase10_master import load_all_controllers

# XAI imports
from xai import (
    DecisionExplainer,
    AttentionExplainer,
    CommunicationExplainer,
    CausalExplainer,
    SafetyExplainer,
    XAILogger
)
from visualize_phase10_gui import Phase10UnifiedGUIDashboard
import tkinter as tk


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


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class Test01_ArchitectureAndModels(unittest.TestCase):
    """Section 1: Controller Architecture & Model Loading Verification."""

    @classmethod
    def setUpClass(cls):
        cls.controllers = {c["id"]: c for c in load_all_controllers("config.yaml")}

    def test_01_all_11_controllers_loaded(self):
        """Verify all 11 evaluation configurations instantiate correctly."""
        expected_ids = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "A1", "A2"]
        self.assertEqual(len(self.controllers), 11)
        for cid in expected_ids:
            self.assertIn(cid, self.controllers, f"Missing controller {cid}")

    def test_02_controller_inference_synchronization(self):
        """Verify that all 11 controllers output 4 synchronized discrete actions in [0, 1, 2, 3]."""
        mock_obs = {a: np.random.uniform(0.0, 1.0, 43).astype(np.float32) for a in PHYSICAL_INTERSECTIONS}
        mock_info = {
            a: {
                "metrics": {
                    "queue_length": 4.0,
                    "waiting_time": 10.0,
                    "vehicle_count": 5,
                    "average_speed": 8.0,
                    "average_occupancy": 0.2
                }
            }
            for a in PHYSICAL_INTERSECTIONS
        }

        eye_4 = torch.eye(4)

        for cid, cfg in self.controllers.items():
            ctype = cfg["type"]
            cobj = cfg["obj"]

            # Reset history if stateful
            if ctype in ["transformer", "gat", "comm_gate"]:
                cobj.reset_history(mock_obs)
            elif ctype in ["causal_gate", "causal_improved"]:
                cobj.reset_history(mock_obs, mock_info)
            elif ctype == "fixed_time":
                cobj.reset()

            # Execute forward inference
            if ctype == "fixed_time":
                actions = cobj.get_actions(mock_obs, mock_info)
            elif ctype == "ippo":
                actions = {
                    a: int(torch.argmax(cobj[a].actor(torch.as_tensor(mock_obs[a]).unsqueeze(0)).probs, dim=-1).item())
                    for a in PHYSICAL_INTERSECTIONS
                }
            elif ctype == "mappo_shared":
                actions, _ = cobj.get_actions(mock_obs, deterministic=True)
            elif ctype == "mappo_individual":
                actions = {}
                for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                    x = np.concatenate([mock_obs[a], eye_4[i].numpy()], axis=-1)
                    dist = cobj[a](torch.as_tensor(x).unsqueeze(0))
                    actions[a] = int(torch.argmax(dist.probs, dim=-1).item())
            elif ctype in ["transformer", "gat"]:
                actions, _ = cobj.get_actions(deterministic=True)
            elif ctype == "comm_gate":
                ctx = cobj.compute_heuristic_context(mock_info)
                actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)
            elif ctype in ["causal_gate", "causal_improved"]:
                ctx = cobj.compute_causal_context(mock_info)
                actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)

            # Verification assertions
            self.assertEqual(
                sorted(list(actions.keys())),
                ["A", "B", "C", "D"],
                f"{cid} did not produce outputs for all 4 intersections"
            )
            for a, act in actions.items():
                self.assertIn(act, [0, 1, 2, 3], f"{cid} produced out-of-domain action {act} for agent {a}")

    def test_03_policy_entropy_non_negative(self):
        """Verify that policy entropy computation is well-defined (H(p) >= 0.0) for all RL controllers."""
        mock_obs = {a: np.full(43, 0.5, dtype=np.float32) for a in PHYSICAL_INTERSECTIONS}
        eye_4 = torch.eye(4)

        for cid in ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]:
            cfg = self.controllers[cid]
            ctype = cfg["type"]
            cobj = cfg["obj"]

            if ctype == "ippo":
                entropy = sum(float(cobj[a].actor(torch.as_tensor(mock_obs[a]).unsqueeze(0)).entropy().item()) for a in PHYSICAL_INTERSECTIONS) / 4.0
            elif ctype == "mappo_shared":
                entropy = sum(float(cobj.shared_actor(torch.as_tensor(np.concatenate([mock_obs[a], eye_4[i].numpy()])).unsqueeze(0)).entropy().item()) for i, a in enumerate(PHYSICAL_INTERSECTIONS)) / 4.0
            elif ctype in ["transformer", "gat", "comm_gate", "causal_gate", "causal_improved"]:
                cobj.reset_history(mock_obs)
                hist = torch.as_tensor(np.stack([cobj.history_manager.get_agent_history(a) for a in PHYSICAL_INTERSECTIONS]), dtype=torch.float32)
                emb = cobj.transformer_encoder(hist)
                if hasattr(cobj, "gat"):
                    emb = cobj.gat(emb.unsqueeze(0)).squeeze(0)
                elif hasattr(cobj, "gated_gat"):
                    ctx = {e: 0.5 for e in ALLOWED_DIRECTED_EDGES}
                    emb_res = cobj.gated_gat(emb.unsqueeze(0), ctx)
                    emb = emb_res[0].squeeze(0) if isinstance(emb_res, tuple) else emb_res.squeeze(0)
                dist = cobj.actor(emb, eye_4)
                entropy = float(dist.entropy().mean().item())

            self.assertGreaterEqual(entropy, 0.0, f"Entropy for {cid} is negative: {entropy}")


class Test02_SumoEnvironmentLifecycle(unittest.TestCase):
    """Section 2: SUMO Environment Lifecycle, TraCI Connection & Step Progression."""

    def test_01_traci_startup_step_and_shutdown(self):
        """Verify complete TraCI lifecycle: startup, stepping, time progression, metrics, shutdown."""
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=1001, options={"safety_enabled": True})
            self.assertTrue(env.is_running, "Environment is_running flag should be True after reset()")
            self.assertEqual(len(obs), 4)
            self.assertEqual(len(info), 4)

            # Check initial metrics
            for a in PHYSICAL_INTERSECTIONS:
                self.assertEqual(obs[a].shape, (43,))
                self.assertEqual(obs[a].dtype, np.float32)
                self.assertIn("metrics", info[a])
                self.assertIn("safety", info[a])

            # Run 5 simulation steps
            initial_sim_time = info["A"]["sim_time"]
            step_actions = {"A": 0, "B": 0, "C": 0, "D": 0}

            for s in range(5):
                next_obs, rewards, term, trunc, next_info = env.step(step_actions)
                self.assertEqual(len(rewards), 4)
                for a in PHYSICAL_INTERSECTIONS:
                    self.assertIsInstance(rewards[a], float)
                    self.assertIn("queue_length", next_info[a]["metrics"])
                    self.assertIn("waiting_time", next_info[a]["metrics"])

            final_sim_time = next_info["A"]["sim_time"]
            # 5 steps * 5s step_duration = 25s elapsed
            self.assertEqual(final_sim_time - initial_sim_time, 25.0, "Simulation time did not advance by 25.0s")

            # Check for zero teleportations
            import traci
            ending_teleports = traci.simulation.getEndingTeleportNumber()
            starting_teleports = traci.simulation.getStartingTeleportNumber()
            self.assertEqual(ending_teleports, 0, f"Unexpected ending teleportations: {ending_teleports}")
            self.assertEqual(starting_teleports, 0, f"Unexpected starting teleportations: {starting_teleports}")

        finally:
            env.close()
            self.assertFalse(env.is_running, "Environment is_running flag should be False after close()")


class Test03_SafetyShieldValidation(unittest.TestCase):
    """Section 3: Phase 9 Safety Shield Invariants & Edge Case Validation."""

    def setUp(self):
        self.safety_layer = TrafficSignalSafetyLayer(
            enabled=True,
            mode="shield",
            min_green=5,
            max_green=60,
            min_red=5,
            yellow_duration=3,
            all_red_duration=1,
            intersections=["A", "B", "C", "D"],
            log_interventions=False
        )

    def test_01_minimum_green_clamp(self):
        """Assert switch proposal during green < 5s is clamped to KEEP with MIN_GREEN constraint."""
        class MockSignal:
            time_on_phase = 2  # 2s < 5s
            is_yellow = False
            is_all_red = False
            current_phase = 0
            target_phase = 0
            target_green_duration = 30
            delta_green = 5

        signals = {"A": MockSignal()}
        # Propose SWITCH (1)
        res = self.safety_layer.validate_actions({"A": 1}, signals, sim_time=2.0, current_step=0)
        self.assertEqual(res["A"], 0, "Premature SWITCH was not clamped to KEEP")
        self.assertEqual(self.safety_layer.total_interventions, 1)
        self.assertEqual(self.safety_layer.interventions_by_constraint["MIN_GREEN"], 1)

    def test_02_maximum_green_enforcement(self):
        """Assert KEEP or EXTEND proposal when green >= 60s is clamped to SWITCH with MAX_GREEN constraint."""
        class MockSignal:
            time_on_phase = 60  # >= 60s
            is_yellow = False
            is_all_red = False
            current_phase = 0
            target_phase = 0
            target_green_duration = 60
            delta_green = 5

        signals = {"B": MockSignal()}
        # Propose KEEP (0)
        res = self.safety_layer.validate_actions({"B": 0}, signals, sim_time=60.0, current_step=12)
        self.assertEqual(res["B"], 1, "Max green violation was not forced to SWITCH")
        self.assertEqual(self.safety_layer.total_interventions, 1)
        self.assertEqual(self.safety_layer.interventions_by_constraint["MAX_GREEN"], 1)

    def test_03_active_clearance_lockout(self):
        """Assert any non-zero action during yellow or all-red clearance is clamped to KEEP."""
        class MockYellowSignal:
            time_on_phase = 0
            is_yellow = True
            is_all_red = False
            current_phase = 0
            target_phase = 2

        class MockAllRedSignal:
            time_on_phase = 0
            is_yellow = False
            is_all_red = True
            current_phase = 0
            target_phase = 2

        signals = {"C": MockYellowSignal(), "D": MockAllRedSignal()}
        res = self.safety_layer.validate_actions({"C": 1, "D": 2}, signals, sim_time=32.0, current_step=6)
        self.assertEqual(res["C"], 0, "Yellow clearance lockout failed")
        self.assertEqual(res["D"], 0, "All-red clearance lockout failed")
        self.assertEqual(self.safety_layer.interventions_by_constraint["CLEARANCE_ACTIVE"], 2)

    def test_04_invalid_action_clamp(self):
        """Assert out-of-domain actions (-1, 4, 99) are clamped to KEEP with INVALID_ACTION constraint."""
        class MockSignal:
            time_on_phase = 10
            is_yellow = False
            is_all_red = False
            current_phase = 0
            target_phase = 0
            target_green_duration = 30
            delta_green = 5

        signals = {"A": MockSignal()}
        res = self.safety_layer.validate_actions({"A": 99}, signals, sim_time=10.0, current_step=2)
        self.assertEqual(res["A"], 0, "Invalid action 99 was not clamped to KEEP")
        self.assertEqual(self.safety_layer.interventions_by_constraint["INVALID_ACTION"], 1)

    def test_05_safety_layer_reset_clears_all_counters(self):
        """Assert safety_layer.reset() flushes all intervention counters and buffers."""
        self.safety_layer.total_proposals = 100
        self.safety_layer.total_interventions = 42
        self.safety_layer.interventions_by_agent["A"] = 15
        self.safety_layer.interventions_by_constraint["MIN_GREEN"] = 20
        self.safety_layer.recent_interventions.append({"test": 1})

        self.safety_layer.reset()

        self.assertEqual(self.safety_layer.total_proposals, 0)
        self.assertEqual(self.safety_layer.total_interventions, 0)
        self.assertEqual(self.safety_layer.interventions_by_agent["A"], 0)
        self.assertEqual(self.safety_layer.interventions_by_constraint["MIN_GREEN"], 0)
        self.assertEqual(len(self.safety_layer.recent_interventions), 0)


class Test04_GuiAndXaiValidation(unittest.TestCase):
    """Section 4: GUI Dashboard & XAI Framework End-to-End Validation."""

    def test_01_gui_smoke_across_diverse_controllers(self):
        """Verify GUI dashboard initializes, steps, and generates XAI explanations across C1, C3, C5, C8, C9."""
        root = tk.Tk()
        root.withdraw()

        test_controllers = ["C1", "C3", "C5", "C8", "C9"]

        try:
            for cid in test_controllers:
                app = Phase10UnifiedGUIDashboard(
                    root=root,
                    config_path="config.yaml",
                    initial_controller_id=cid,
                    initial_seed=1001,
                    max_steps=3,
                    use_sumo_gui=False,
                    smoke_test=True
                )

                # Step 3 times
                for s in range(3):
                    done = app.execute_decision_step()

                # Verify telemetry and XAI explanations
                self.assertIsNotNone(app.last_xai_decisions, f"Missing XAI decisions for {cid}")
                self.assertIn("A", app.last_xai_decisions)
                self.assertIsNotNone(app.last_xai_safeties, f"Missing XAI safeties for {cid}")
                self.assertIn("A", app.last_xai_safeties)
                self.assertIsNotNone(app.last_xai_attention, f"Missing XAI attention for {cid}")
                self.assertIsNotNone(app.last_xai_comm, f"Missing XAI comm for {cid}")
                self.assertIsNotNone(app.last_xai_causal, f"Missing XAI causal for {cid}")

                # Verify explainer epistemological disclaimers are present
                attn_exp = app.last_xai_attention
                self.assertIn("epistemological_disclaimer", attn_exp.__dict__)

                # Clean teardown for next iteration
                app.env.close()

        finally:
            root.destroy()

    def test_02_xai_agent_selection_and_logger(self):
        """Verify XAI logger records steps and agent selection functions cleanly."""
        logger = XAILogger(max_buffer_size=100)
        explainer = DecisionExplainer()
        exp = explainer.explain(
            agent_id="B",
            action_probs=np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float32),
            selected_action=1,
            metrics={"queue_length": 5.0, "waiting_time": 12.0, "vehicle_count": 6}
        )

        logger.record_step(step=1, sim_time=5.0, controller_id="C9", decisions={"B": exp})
        self.assertEqual(len(logger.history), 1)
        entry = logger.get_latest()
        self.assertIsNotNone(entry)
        self.assertEqual(entry["controller_id"], "C9")


class Test05_StabilityAndResets(unittest.TestCase):
    """Section 5: Repeated Execution Stability, Sequential Controller Switching & Process Teardown."""

    def test_01_sequential_episode_resets_and_memory(self):
        """Execute 5 sequential episode resets on a single environment instance with controller switching."""
        tracemalloc.start()
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)

        switching_schedule = [
            ("C1", 1001),
            ("C3", 1002),
            ("C5", 1003),
            ("C8", 1004),
            ("C9", 1005)
        ]

        try:
            for cid, seed in switching_schedule:
                safety = (cid == "C9")
                obs, info = env.reset(seed=seed, options={"safety_enabled": safety})
                self.assertEqual(env.safety_enabled, safety)
                self.assertTrue(env.is_running)

                # Step 3 times
                for s in range(3):
                    actions = {a: 0 for a in PHYSICAL_INTERSECTIONS}
                    obs, rew, term, trunc, info = env.step(actions)

            env.close()
            gc.collect()

            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            current_mb = current_bytes / (1024 * 1024)
            peak_mb = peak_bytes / (1024 * 1024)
            # Memory delta over 5 resets should remain modest (< 100 MB)
            self.assertLess(current_mb, 100.0, f"Excessive memory footprint: {current_mb:.2f} MB")

        finally:
            env.close()
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    def test_02_zero_orphan_sumo_processes(self):
        """Verify that closing environment leaves zero orphan sumo.exe or sumo-gui.exe processes."""
        import traci
        env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)
        env.reset(seed=1001, options={"safety_enabled": False})
        self.assertTrue(env.is_running)
        self.assertTrue(traci.isLoaded(), "TraCI must be loaded during simulation run")

        env.close()
        self.assertFalse(env.is_running, "Environment is_running must be False after close()")
        self.assertFalse(traci.isLoaded(), "TraCI must be unloaded after env.close()")

        # Verify no orphan sumo process remains running via powershell
        time.sleep(0.5)
        res = subprocess.run(
            ["powershell", "-Command", "Get-Process sumo,sumo-gui -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.stdout.strip(), "", "Found lingering orphan SUMO processes!")


class Test06_ArtifactIntegrity(unittest.TestCase):
    """Section 6: Checkpoint Hashes, Benchmark CSV Row Counts & Git Anchor Integrity."""

    def test_01_all_20_checkpoint_sha256_hashes(self):
        """Verify all 20 .pt model checkpoints match approved SHA256 checksums."""
        ckpt_dir = "models/checkpoints"
        files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        self.assertEqual(len(files), 20, f"Expected 20 checkpoints, found {len(files)}")

        for f in files:
            self.assertIn(f, APPROVED_CHECKPOINT_HASHES, f"Unexpected checkpoint {f}")
            actual_sha = compute_file_sha256(os.path.join(ckpt_dir, f))
            expected_sha = APPROVED_CHECKPOINT_HASHES[f]
            self.assertEqual(actual_sha, expected_sha, f"SHA256 mismatch for checkpoint {f}")

    def test_02_historical_benchmark_csv_integrity(self):
        """Verify existence and row counts of historical benchmark CSV logs."""
        csv_specs = {
            "logs/phase10_master_benchmark.csv": 110,
            "logs/phase12_demand_benchmark.csv": 75,
            "logs/phase12_dynamic_benchmark.csv": 50,
            "logs/phase12_noise_benchmark.csv": 75
        }

        for path, expected_rows in csv_specs.items():
            self.assertTrue(os.path.exists(path), f"Benchmark log missing: {path}")
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                self.assertEqual(
                    len(rows), expected_rows,
                    f"Row count mismatch in {path}: expected {expected_rows}, got {len(rows)}"
                )

    def test_03_git_anchor_commit_verification(self):
        """Verify that git HEAD commit matches the master anchor 8be2d17."""
        git_head_file = ".git/HEAD"
        self.assertTrue(os.path.exists(git_head_file), "Missing .git/HEAD file")

        with open(git_head_file, "r") as f:
            ref = f.read().strip()

        if ref.startswith("ref: "):
            ref_path = os.path.join(".git", ref[5:].replace("/", os.sep))
            self.assertTrue(os.path.exists(ref_path), f"Missing ref file: {ref_path}")
            with open(ref_path, "r") as f:
                commit_hash = f.read().strip()
        else:
            commit_hash = ref

        self.assertTrue(
            commit_hash.startswith("8be2d17"),
            f"Git HEAD {commit_hash} does not match anchor 8be2d17"
        )


if __name__ == "__main__":
    unittest.main()
