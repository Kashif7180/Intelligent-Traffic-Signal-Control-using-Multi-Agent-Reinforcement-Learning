"""
SAGE-Traffic Phase 13 Step 2: Stress Testing & Edge-Case Validation Suite
========================================================================
Comprehensive stress evaluation verifying system resilience across:
1. Empty Traffic & Boundary Zero-State (0 vehicles, 0 queues, division-by-zero guards)
2. Extreme Hyper-Saturation Demand (p=0.50, ~2000 vehicles, high occupancy, low speed)
3. Asymmetric Bottleneck Demand (Node A jammed at p=0.65 vs Nodes B, C, D at p=0.04)
4. Sudden Demand Shock Surge (p=0.08 -> p=0.55 at t=150s)
5. Near-Gridlock Accumulation & Dissipation Recovery (p=0.50 -> p=0.00 at t=250s)
6. Phase 9 Safety Shield Under Extreme Stress (Clamps, Lockouts, Forensic Zero-Conflict Audit)
7. Recovery, Controller Switching, and Resource Teardown Stability
"""

import os
import sys
import gc
import time
import unittest
import tracemalloc
import subprocess
import numpy as np
import torch

from network.topology import (
    PHYSICAL_INTERSECTIONS,
    ALLOWED_DIRECTED_EDGES,
    get_adjacent_neighbors,
    is_valid_physical_edge
)
from environment.traffic_env import MultiAgentTrafficEnv, TrafficSignal
from models.safety_layer import TrafficSignalSafetyLayer, ACTION_NAMES
from benchmark_phase10_master import load_all_controllers

STRESS_ROUTE_DIR = "network/route_cache_phase13_stress"


class Test01_EmptyTrafficStress(unittest.TestCase):
    """Section 1: Empty Traffic & Boundary Zero-State Invariants."""

    @classmethod
    def setUpClass(cls):
        cls.empty_route_path = os.path.join(STRESS_ROUTE_DIR, "routes_empty.rou.xml")
        assert os.path.exists(cls.empty_route_path), "Missing empty route file!"
        cls.controllers = {c["id"]: c for c in load_all_controllers("config.yaml")}

    def test_01_empty_network_rollout_invariants(self):
        """Assert zero vehicles produces 0.0 queues, valid obs in [0, 1], zero NaN/Inf, and valid rewards."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=1001, options={"route_file": self.empty_route_path, "safety_enabled": True})
            self.assertTrue(env.is_running)

            # Step 10 times in empty network
            for step in range(10):
                actions = {a: 0 for a in PHYSICAL_INTERSECTIONS}
                obs, rewards, terminations, truncations, info = env.step(actions)

                for a in PHYSICAL_INTERSECTIONS:
                    # 1. Observation bounds and validity
                    self.assertEqual(obs[a].shape, (43,))
                    self.assertFalse(np.isnan(obs[a]).any(), f"NaN detected in obs for agent {a} at step {step}")
                    self.assertFalse(np.isinf(obs[a]).any(), f"Inf detected in obs for agent {a} at step {step}")
                    self.assertTrue(np.all(obs[a] >= 0.0) and np.all(obs[a] <= 1.0), f"Obs out of [0, 1] bounds for agent {a}")

                    # 2. Metric zero-invariants
                    m = info[a]["metrics"]
                    self.assertEqual(m["queue_length"], 0, f"Queue non-zero in empty network for agent {a}")
                    self.assertEqual(m["waiting_time"], 0.0, f"Waiting time non-zero in empty network for agent {a}")
                    self.assertEqual(m["vehicle_count"], 0, f"Vehicle count non-zero in empty network for agent {a}")
                    self.assertEqual(m["average_speed"], 0.0, f"Average speed non-zero in empty network for agent {a}")
                    self.assertEqual(m["average_occupancy"], 0.0, f"Occupancy non-zero in empty network for agent {a}")

                    # 3. Reward check: with queue=0, wait=0, switch=0, R = 0.0
                    self.assertEqual(rewards[a], 0.0, f"Reward non-zero for KEEP in empty network: {rewards[a]}")

            # Check zero teleportations
            import traci
            self.assertEqual(traci.simulation.getEndingTeleportNumber(), 0)
        finally:
            env.close()

    def test_02_all_controllers_on_empty_network(self):
        """Assert all 11 controllers execute inference on empty network without NaN, LinAlgError or crash."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        eye_4 = torch.eye(4)

        try:
            obs, info = env.reset(seed=1001, options={"route_file": self.empty_route_path, "safety_enabled": True})

            for cid, cfg in self.controllers.items():
                ctype = cfg["type"]
                cobj = cfg["obj"]

                if ctype in ["transformer", "gat", "comm_gate"]:
                    cobj.reset_history(obs)
                elif ctype in ["causal_gate", "causal_improved"]:
                    cobj.reset_history(obs, info)
                elif ctype == "fixed_time":
                    cobj.reset()

                # Execute inference
                if ctype == "fixed_time":
                    actions = cobj.get_actions(obs, info)
                elif ctype == "ippo":
                    actions = {a: int(torch.argmax(cobj[a].actor(torch.as_tensor(obs[a]).unsqueeze(0)).probs, dim=-1).item()) for a in PHYSICAL_INTERSECTIONS}
                elif ctype == "mappo_shared":
                    actions, _ = cobj.get_actions(obs, deterministic=True)
                elif ctype == "mappo_individual":
                    actions = {}
                    for i, a in enumerate(PHYSICAL_INTERSECTIONS):
                        x = np.concatenate([obs[a], eye_4[i].numpy()], axis=-1)
                        dist = cobj[a](torch.as_tensor(x).unsqueeze(0))
                        actions[a] = int(torch.argmax(dist.probs, dim=-1).item())
                elif ctype in ["transformer", "gat"]:
                    actions, _ = cobj.get_actions(deterministic=True)
                elif ctype == "comm_gate":
                    ctx = cobj.compute_heuristic_context(info)
                    actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)
                elif ctype in ["causal_gate", "causal_improved"]:
                    ctx = cobj.compute_causal_context(info)
                    actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)

                self.assertEqual(sorted(list(actions.keys())), ["A", "B", "C", "D"])
                for a in PHYSICAL_INTERSECTIONS:
                    self.assertIn(actions[a], [0, 1, 2, 3], f"{cid} produced invalid action {actions[a]}")

        finally:
            env.close()

    def test_03_repeated_empty_resets(self):
        """Assert 5 consecutive resets with empty route file execute cleanly."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            for r in range(5):
                obs, info = env.reset(seed=1000 + r, options={"route_file": self.empty_route_path})
                self.assertTrue(env.is_running)
                for a in PHYSICAL_INTERSECTIONS:
                    self.assertEqual(info[a]["metrics"]["vehicle_count"], 0)
        finally:
            env.close()


class Test02_ExtremeTrafficStress(unittest.TestCase):
    """Section 2: Extreme Hyper-Saturation Demand (p=0.50, ~2000 vehicles)."""

    @classmethod
    def setUpClass(cls):
        cls.ext_route_path = os.path.join(STRESS_ROUTE_DIR, "routes_extreme_seed_4001.rou.xml")
        assert os.path.exists(cls.ext_route_path), "Missing extreme route file!"

    def test_01_hyper_saturation_step_progression_and_anti_deadlock(self):
        """Assert hyper-saturation accumulates heavy queues, simulation progresses second-by-second without deadlock."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=4001, options={"route_file": self.ext_route_path, "safety_enabled": True})
            ctrl = load_all_controllers("config.yaml")[-3]["obj"]  # C9: Phase 9 Shielded
            ctrl.reset_history(obs, info)

            initial_time = info["A"]["sim_time"]

            # Run 20 decision steps (100 simulation seconds) under hyper-saturated demand
            queues_over_time = []
            occupancies_over_time = []

            for step in range(20):
                ctx = ctrl.compute_causal_context(info)
                actions, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                obs, rewards, term, trunc, info = env.step(actions)
                ctrl.update_history(obs)

                # Total network queue across all 4 intersections
                total_q = sum(info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS)
                mean_occ = np.mean([info[a]["metrics"]["average_occupancy"] for a in PHYSICAL_INTERSECTIONS])
                queues_over_time.append(total_q)
                occupancies_over_time.append(mean_occ)

                # Observation clamping assertion under extreme traffic
                for a in PHYSICAL_INTERSECTIONS:
                    self.assertFalse(np.isnan(obs[a]).any())
                    self.assertFalse(np.isinf(obs[a]).any())
                    self.assertTrue(np.all(obs[a] >= 0.0) and np.all(obs[a] <= 1.0))

            final_time = info["A"]["sim_time"]
            # 20 steps * 5s = 100s simulation progression
            self.assertEqual(final_time - initial_time, 100.0, "Simulation deadlocked or time did not advance by 100s!")

            # Under hyper-saturation, total queue should significantly accumulate
            self.assertGreater(queues_over_time[-1], 10, f"Expected heavy queuing under hyper-saturation, got {queues_over_time[-1]}")
            self.assertGreater(occupancies_over_time[-1], 0.15, f"Expected elevated occupancy, got {occupancies_over_time[-1]}")

        finally:
            env.close()

    def test_02_controller_robustness_under_extreme_traffic(self):
        """Assert controllers C1, C3, C5, C8, C9 execute 10 steps on extreme traffic without NaN or crash."""
        all_ctrls = {c["id"]: c for c in load_all_controllers("config.yaml")}
        tested_ctrls = ["C1", "C3", "C5", "C8", "C9"]

        for cid in tested_ctrls:
            env = MultiAgentTrafficEnv("config.yaml", gui=False)
            cfg = all_ctrls[cid]
            ctype = cfg["type"]
            cobj = cfg["obj"]

            try:
                obs, info = env.reset(seed=4001, options={"route_file": self.ext_route_path, "safety_enabled": (cid == "C9")})
                if ctype in ["transformer", "gat", "comm_gate"]:
                    cobj.reset_history(obs)
                elif ctype in ["causal_gate", "causal_improved"]:
                    cobj.reset_history(obs, info)
                elif ctype == "fixed_time":
                    cobj.reset()

                for s in range(5):
                    if ctype == "fixed_time":
                        actions = cobj.get_actions(obs, info)
                    elif ctype == "mappo_shared":
                        actions, _ = cobj.get_actions(obs, deterministic=True)
                    elif ctype in ["transformer", "gat"]:
                        actions, _ = cobj.get_actions(deterministic=True)
                    elif ctype in ["causal_gate", "causal_improved"]:
                        ctx = cobj.compute_causal_context(info)
                        actions, _, _, _ = cobj.get_actions(ctx, deterministic=True)

                    obs, rew, term, trunc, info = env.step(actions)
                    if ctype in ["transformer", "gat", "causal_improved"]:
                        cobj.update_history(obs)

                self.assertTrue(env.is_running)
            finally:
                env.close()


class Test03_AsymmetricStress(unittest.TestCase):
    """Section 3: Asymmetric Bottleneck Demand (Node A saturated at p=0.65 vs B,C,D at p=0.04)."""

    @classmethod
    def setUpClass(cls):
        cls.asym_route_path = os.path.join(STRESS_ROUTE_DIR, "routes_asymmetric_seed_4001.rou.xml")
        assert os.path.exists(cls.asym_route_path), "Missing asymmetric route file!"

    def test_01_asymmetric_bottleneck_detection_and_safety(self):
        """Assert severe localized congestion at Node A while Nodes B,C,D remain light, with zero safety conflicts."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=4001, options={"route_file": self.asym_route_path, "safety_enabled": True})
            ctrl = load_all_controllers("config.yaml")[-3]["obj"]  # C9
            ctrl.reset_history(obs, info)

            for step in range(20):
                ctx = ctrl.compute_causal_context(info)
                actions, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                obs, rewards, term, trunc, info = env.step(actions)
                ctrl.update_history(obs)

            # Node A queue vs other nodes
            q_A = info["A"]["metrics"]["queue_length"]
            q_other = max(info[a]["metrics"]["queue_length"] for a in ["B", "C", "D"])

            # Node A should experience significantly more congestion than non-bottleneck nodes
            self.assertGreater(q_A, q_other, f"Asymmetric load failed: Node A ({q_A}) not greater than other nodes ({q_other})")

            # Check zero physical conflicts
            conflicts = info["A"]["safety"]["audit_physical_violations"]
            self.assertEqual(conflicts, 0, f"Detected {conflicts} physical signal conflicts during asymmetric stress!")

        finally:
            env.close()


class Test04_DynamicSurgeStress(unittest.TestCase):
    """Section 4: Sudden Demand Shock Surge (p=0.08 -> p=0.55 at t=150s)."""

    @classmethod
    def setUpClass(cls):
        cls.surge_route_path = os.path.join(STRESS_ROUTE_DIR, "routes_surge_seed_4001.rou.xml")
        assert os.path.exists(cls.surge_route_path), "Missing surge route file!"

    def test_01_demand_shock_surge_transition(self):
        """Assert system adapts to sudden demand shock at t=150s with zero crashes, valid state, and rising queues."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=4001, options={"route_file": self.surge_route_path, "safety_enabled": True})
            ctrl = load_all_controllers("config.yaml")[-3]["obj"]  # C9
            ctrl.reset_history(obs, info)

            pre_surge_queues = []
            post_surge_queues = []

            # 40 steps = 200 simulation seconds (covers t=0 to t=200s, crossing t=150s at step 30)
            for step in range(40):
                ctx = ctrl.compute_causal_context(info)
                actions, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                obs, rew, term, trunc, info = env.step(actions)
                ctrl.update_history(obs)

                sim_time = info["A"]["sim_time"]
                tot_q = sum(info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS)

                if sim_time < 150.0:
                    pre_surge_queues.append(tot_q)
                else:
                    post_surge_queues.append(tot_q)

            # Average queue post-surge must be strictly higher than pre-surge
            avg_pre = np.mean(pre_surge_queues)
            avg_post = np.mean(post_surge_queues)
            self.assertGreater(avg_post, avg_pre, f"Post-surge queue ({avg_post:.2f}) not greater than pre-surge ({avg_pre:.2f})")

        finally:
            env.close()


class Test05_NearGridlockDissipation(unittest.TestCase):
    """Section 5: Near-Gridlock Accumulation & Dissipation Recovery (p=0.50 -> p=0.00 at t=250s)."""

    @classmethod
    def setUpClass(cls):
        cls.diss_route_path = os.path.join(STRESS_ROUTE_DIR, "routes_dissipation_seed_4001.rou.xml")
        assert os.path.exists(cls.diss_route_path), "Missing dissipation route file!"

    def test_01_gridlock_dissipation_and_recovery(self):
        """Assert queues accumulate until t=250s then steadily dissipate after demand cutoff without gridlock lockup."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            obs, info = env.reset(seed=4001, options={"route_file": self.diss_route_path, "safety_enabled": True})
            ctrl = load_all_controllers("config.yaml")[-3]["obj"]  # C9
            ctrl.reset_history(obs, info)

            # Step through 70 steps (350 simulation seconds, crossing cutoff at step 50 = 250s)
            peak_q = 0
            for step in range(50):
                ctx = ctrl.compute_causal_context(info)
                actions, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                obs, rew, term, trunc, info = env.step(actions)
                ctrl.update_history(obs)
                tot_q = sum(info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS)
                if tot_q > peak_q:
                    peak_q = tot_q

            # Verify peak queue accumulated during congestion phase
            self.assertGreater(peak_q, 8, f"Expected congestion buildup, got peak {peak_q}")

            # Now step through dissipation phase (step 50 to 80 = 250s to 400s)
            for step in range(30):
                ctx = ctrl.compute_causal_context(info)
                actions, _, _, _ = ctrl.get_actions(ctx, deterministic=True)
                obs, rew, term, trunc, info = env.step(actions)
                ctrl.update_history(obs)

            final_q = sum(info[a]["metrics"]["queue_length"] for a in PHYSICAL_INTERSECTIONS)
            # After 150 seconds of zero demand, network must dissipate substantially (final_q < peak_q)
            self.assertLess(final_q, peak_q, f"Network failed to dissipate: final_q={final_q} >= peak_q={peak_q}")

        finally:
            env.close()


class Test06_SafetyShieldUnderStress(unittest.TestCase):
    """Section 6: Phase 9 Safety Shield Under Extreme Stress & Forensic Conflict Audit."""

    def test_01_rapid_switch_requests_under_heavy_queues(self):
        """Assert policy proposing SWITCH (1) every single step is safely clamped during min_green and clearance."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        ext_path = os.path.join(STRESS_ROUTE_DIR, "routes_extreme_seed_4001.rou.xml")

        try:
            obs, info = env.reset(seed=4001, options={"route_file": ext_path, "safety_enabled": True})

            # Propose aggressive continuous SWITCH (1) on all 4 intersections for 15 steps
            for step in range(15):
                aggressive_actions = {"A": 1, "B": 1, "C": 1, "D": 1}
                obs, rew, term, trunc, info = env.step(aggressive_actions)

            # Check that interventions were triggered
            interventions = info["A"]["safety"]["total_interventions"]
            self.assertGreater(interventions, 0, "Safety layer failed to record interventions during continuous switch attack!")

            # Verify zero physical conflicts
            violations = info["A"]["safety"]["audit_physical_violations"]
            self.assertEqual(violations, 0, f"Safety shield allowed {violations} physical conflicts during switch attack!")

        finally:
            env.close()

    def test_02_invalid_action_injection_under_stress(self):
        """Assert injected negative and out-of-bounds actions are safely neutralized during heavy traffic."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        ext_path = os.path.join(STRESS_ROUTE_DIR, "routes_extreme_seed_4001.rou.xml")

        try:
            obs, info = env.reset(seed=4001, options={"route_file": ext_path, "safety_enabled": True})

            # Propose invalid actions [-5, 99, 4, -1]
            invalid_actions = {"A": -5, "B": 99, "C": 4, "D": -1}
            obs, rew, term, trunc, info = env.step(invalid_actions)

            for a in PHYSICAL_INTERSECTIONS:
                self.assertEqual(info[a]["validated_action"], 0, f"Invalid action for {a} was not neutralized to KEEP (0)")

            violations = info["A"]["safety"]["audit_physical_violations"]
            self.assertEqual(violations, 0)

        finally:
            env.close()

    def test_03_safety_toggle_lockout_during_active_clearance(self):
        """Assert attempting to disable safety while intersection is in clearance returns False and defers."""
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        try:
            env.reset(seed=1001, options={"safety_enabled": True})
            # Manually trigger clearance on intersection A
            env.signals["A"].is_yellow = True
            applied = env.set_safety_enabled(False)
            self.assertFalse(applied, "Disabling safety during active clearance must be rejected (deferred)")
            self.assertTrue(env.safety_enabled, "Safety shield must remain active")
            env.signals["A"].is_yellow = False
        finally:
            env.close()


class Test07_ResourceStabilityUnderStress(unittest.TestCase):
    """Section 7: Recovery, Repeated Stress Execution, Memory Stability & Teardown."""

    def test_01_sequential_stress_episodes_and_memory(self):
        """Execute 5 consecutive stress episodes across different stress route files with controller switching."""
        tracemalloc.start()
        env = MultiAgentTrafficEnv("config.yaml", gui=False)

        schedule = [
            ("C1", os.path.join(STRESS_ROUTE_DIR, "routes_empty.rou.xml"), 4001),
            ("C3", os.path.join(STRESS_ROUTE_DIR, "routes_extreme_seed_4001.rou.xml"), 4002),
            ("C5", os.path.join(STRESS_ROUTE_DIR, "routes_asymmetric_seed_4001.rou.xml"), 4003),
            ("C8", os.path.join(STRESS_ROUTE_DIR, "routes_surge_seed_4001.rou.xml"), 4004),
            ("C9", os.path.join(STRESS_ROUTE_DIR, "routes_dissipation_seed_4001.rou.xml"), 4005)
        ]

        try:
            for cid, r_path, seed in schedule:
                safety = (cid == "C9")
                obs, info = env.reset(seed=seed, options={"route_file": r_path, "safety_enabled": safety})
                self.assertTrue(env.is_running)

                for s in range(4):
                    actions = {a: 0 for a in PHYSICAL_INTERSECTIONS}
                    obs, rew, term, trunc, info = env.step(actions)

            env.close()
            gc.collect()

            cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cur_mb = cur_bytes / (1024 * 1024)
            # Memory delta must remain modest (< 100 MB)
            self.assertLess(cur_mb, 100.0, f"Excessive memory growth: {cur_mb:.2f} MB")

        finally:
            env.close()
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    def test_02_zero_orphan_sumo_processes_after_stress(self):
        """Assert no orphan sumo or sumo-gui processes linger after stress runs."""
        import traci
        env = MultiAgentTrafficEnv("config.yaml", gui=False)
        ext_path = os.path.join(STRESS_ROUTE_DIR, "routes_extreme_seed_4001.rou.xml")
        env.reset(seed=4001, options={"route_file": ext_path, "safety_enabled": True})
        self.assertTrue(env.is_running)
        self.assertTrue(traci.isLoaded())

        env.close()
        self.assertFalse(env.is_running)
        self.assertFalse(traci.isLoaded())

        time.sleep(0.5)
        res = subprocess.run(
            ["powershell", "-Command", "Get-Process sumo,sumo-gui -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.stdout.strip(), "", "Lingering orphan SUMO process found after stress test!")


if __name__ == "__main__":
    unittest.main()

