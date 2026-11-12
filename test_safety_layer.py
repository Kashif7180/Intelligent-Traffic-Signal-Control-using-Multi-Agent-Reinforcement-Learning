"""
SAGE-Traffic Phase 9: Comprehensive Safety Layer Verification Suite
Tests 15 strict safety invariants, boundary conditions, transitions, and reproduction contracts.

Test Suite Coverage:
1.  Min green enforcement (< 5s rejected)
2.  Min green boundary condition (t=4s rejected, t=5s allowed)
3.  Max green enforcement (>= 60s forces switch)
4.  Max green boundary condition (t=59s allowed, t=60s forces switch)
5.  Min red enforcement (< 5s opposing red blocks switch)
6.  Min red boundary condition (t_red < min_red blocked, t_red >= min_red allowed)
7.  Yellow clearance non-preemptible (switch during yellow clamped to KEEP)
8.  All-red clearance non-preemptible (switch during all-red clamped to KEEP)
9.  No direct green-to-green transitions (strictly passes through yellow and all-red)
10. Safe in-bounds actions pass unmodified
11. Consecutive actions handling (rapid toggles, repeated switches without timer corruption)
12. Independent validation across all 4 intersections (A, B, C, D)
13. Final environment boundary cannot be bypassed
14. Reset and termination behavior (clean timer and statistics reset)
15. Safety disabled mode exact reproduction contract (100% pass-through when disabled)
"""

import os
import sys
import unittest
import yaml
from unittest.mock import MagicMock, patch

from models.safety_layer import (
    TrafficSignalSafetyLayer,
    NS_LINK_INDICES,
    EW_LINK_INDICES,
    VALID_SUMO_STATES
)
from environment.traffic_env import TrafficSignal, MultiAgentTrafficEnv


class MockSignal:
    """Mock TrafficSignal state container for precise unit testing."""
    def __init__(
        self,
        ts_id="A",
        current_phase=0,
        time_on_phase=0,
        is_yellow=False,
        is_all_red=False,
        red_time_ns=0,
        red_time_ew=5,
        target_green_duration=30,
        delta_green=5,
        min_green=5,
        max_green=60,
        min_red=5
    ):
        self.ts_id = ts_id
        self.current_phase = current_phase
        self.target_phase = current_phase
        self.time_on_phase = time_on_phase
        self.is_yellow = is_yellow
        self.is_all_red = is_all_red
        self.red_time_ns = red_time_ns
        self.red_time_ew = red_time_ew
        self.target_green_duration = target_green_duration
        self.delta_green = delta_green
        self.min_green = min_green
        self.max_green = max_green
        self.min_red = min_red
        self.last_applied_action = None

    def apply_action(self, action):
        self.last_applied_action = action

    def update_second(self):
        pass

    def compute_observation(self, features, normalization):
        import numpy as np
        return np.zeros(20, dtype=np.float32)

    def compute_reward(self, reward_cfg):
        return 0.0

    def get_metrics(self):
        return {"queue_length": 0.0, "waiting_time": 0.0}


class TestSafetyLayer(unittest.TestCase):

    def setUp(self):
        self.safety = TrafficSignalSafetyLayer(
            enabled=True,
            mode="shield",
            min_green=5,
            max_green=60,
            min_red=5,
            yellow_duration=3,
            all_red_duration=1,
            log_interventions=False,
            intersections=["A", "B", "C", "D"]
        )

    # -------------------------------------------------------------------------
    # Test 1: Min Green Enforcement
    # -------------------------------------------------------------------------
    def test_01_min_green_enforcement(self):
        """Reject SWITCH action when time_on_phase < min_green (e.g. 2s < 5s)."""
        signals = {"A": MockSignal(ts_id="A", time_on_phase=2, red_time_ew=10)}
        proposed = {"A": 1}  # SWITCH
        safe = self.safety.validate_actions(proposed, signals)
        self.assertEqual(safe["A"], 0, "SWITCH before min_green must be clamped to 0 (KEEP)!")
        self.assertEqual(self.safety.interventions_by_constraint["MIN_GREEN"], 1)

    # -------------------------------------------------------------------------
    # Test 2: Min Green Boundary Condition
    # -------------------------------------------------------------------------
    def test_02_min_green_boundary(self):
        """At t=4s (boundary - 1), switch rejected. At t=5s (boundary), switch allowed."""
        # Boundary - 1 (t=4s)
        signals_4 = {"A": MockSignal(ts_id="A", time_on_phase=4, red_time_ew=10)}
        safe_4 = self.safety.validate_actions({"A": 1}, signals_4)
        self.assertEqual(safe_4["A"], 0, "At t=4s (< 5s), switch must be rejected!")

        # Boundary exact (t=5s)
        signals_5 = {"A": MockSignal(ts_id="A", time_on_phase=5, red_time_ew=10)}
        safe_5 = self.safety.validate_actions({"A": 1}, signals_5)
        self.assertEqual(safe_5["A"], 1, "At t=5s (>= min_green), switch must be allowed!")

    # -------------------------------------------------------------------------
    # Test 3: Max Green Enforcement
    # -------------------------------------------------------------------------
    def test_03_max_green_enforcement(self):
        """When green duration reaches or exceeds max_green, KEEP must be overridden to SWITCH."""
        signals = {"A": MockSignal(ts_id="A", time_on_phase=65, red_time_ew=65)}
        proposed = {"A": 0}  # KEEP
        safe = self.safety.validate_actions(proposed, signals)
        self.assertEqual(safe["A"], 1, "At green >= max_green, KEEP must be overridden to 1 (SWITCH)!")
        self.assertEqual(self.safety.interventions_by_constraint["MAX_GREEN"], 1)

    # -------------------------------------------------------------------------
    # Test 4: Max Green Boundary Condition
    # -------------------------------------------------------------------------
    def test_04_max_green_boundary(self):
        """At t=59s, KEEP is allowed. At t=60s, KEEP is overridden to SWITCH."""
        # Boundary - 1 (t=59s)
        signals_59 = {"A": MockSignal(ts_id="A", time_on_phase=59, red_time_ew=59)}
        safe_59 = self.safety.validate_actions({"A": 0}, signals_59)
        self.assertEqual(safe_59["A"], 0, "At t=59s (< max_green), KEEP must be allowed!")

        # Boundary exact (t=60s)
        signals_60 = {"A": MockSignal(ts_id="A", time_on_phase=60, red_time_ew=60)}
        safe_60 = self.safety.validate_actions({"A": 0}, signals_60)
        self.assertEqual(safe_60["A"], 1, "At t=60s (>= max_green), KEEP must be overridden to SWITCH!")

    # -------------------------------------------------------------------------
    # Test 5: Min Red Enforcement
    # -------------------------------------------------------------------------
    def test_05_min_red_enforcement(self):
        """Switch blocked if opposing movement has not accumulated required min_red time."""
        # If current phase is 0 (NS), target is EW. If red_time_ew = 0 and clearance = 3+1=4 < 5
        signals = {"A": MockSignal(ts_id="A", current_phase=0, time_on_phase=10, red_time_ew=0)}
        proposed = {"A": 1}  # SWITCH
        safe = self.safety.validate_actions(proposed, signals)
        self.assertEqual(safe["A"], 0, "Switch must be blocked if opposing red time < min_red!")
        self.assertEqual(self.safety.interventions_by_constraint["MIN_RED"], 1)

    # -------------------------------------------------------------------------
    # Test 6: Min Red Boundary Condition
    # -------------------------------------------------------------------------
    def test_06_min_red_boundary(self):
        """Opposing red projected duration: t_red + clearance(4s) < 5s blocked, >= 5s allowed."""
        # Projected red = 0 + 4 = 4s (< 5s) -> BLOCKED
        signals_0 = {"A": MockSignal(ts_id="A", current_phase=0, time_on_phase=10, red_time_ew=0)}
        safe_0 = self.safety.validate_actions({"A": 1}, signals_0)
        self.assertEqual(safe_0["A"], 0)

        # Projected red = 1 + 4 = 5s (>= 5s) -> ALLOWED
        signals_1 = {"A": MockSignal(ts_id="A", current_phase=0, time_on_phase=10, red_time_ew=1)}
        safe_1 = self.safety.validate_actions({"A": 1}, signals_1)
        self.assertEqual(safe_1["A"], 1)

    # -------------------------------------------------------------------------
    # Test 7: Yellow Clearance Non-Preemptible
    # -------------------------------------------------------------------------
    def test_07_yellow_clearance_non_preemptible(self):
        """During yellow clearance, all disruptive actions (SWITCH, EXTEND, REDUCE) clamped to KEEP."""
        signals = {"A": MockSignal(ts_id="A", is_yellow=True, time_on_phase=0)}
        for act in [1, 2, 3]:
            safe = self.safety.validate_actions({"A": act}, signals)
            self.assertEqual(safe["A"], 0, f"Action {act} during yellow clearance must be clamped to 0 (KEEP)!")
        self.assertEqual(self.safety.interventions_by_constraint["CLEARANCE_ACTIVE"], 3)

    # -------------------------------------------------------------------------
    # Test 8: All-Red Clearance Non-Preemptible
    # -------------------------------------------------------------------------
    def test_08_all_red_clearance_non_preemptible(self):
        """During all-red clearance, all disruptive actions clamped to KEEP."""
        signals = {"A": MockSignal(ts_id="A", is_all_red=True, time_on_phase=0)}
        for act in [1, 2, 3]:
            safe = self.safety.validate_actions({"A": act}, signals)
            self.assertEqual(safe["A"], 0, f"Action {act} during all-red clearance must be clamped to 0 (KEEP)!")

    # -------------------------------------------------------------------------
    # Test 9: No Direct Green-to-Green Transitions
    # -------------------------------------------------------------------------
    def test_09_no_direct_green_transitions(self):
        """Verify TrafficSignal transition sequences strictly: Green -> Yellow -> All-Red -> Green."""
        sig = TrafficSignal(
            ts_id="A",
            incoming_lanes=[],
            green_phases=[0, 2],
            yellow_map={0: 1, 2: 3},
            yellow_duration=3,
            min_green=5,
            max_green=60,
            all_red_duration=1,
            min_red=5
        )
        sig.current_phase = 0
        sig.time_on_phase = 10
        sig.red_time_ew = 10

        # Apply SWITCH (1)
        sig.apply_action(1)
        self.assertTrue(sig.is_yellow, "Switch must first enter yellow clearance!")
        self.assertEqual(sig.yellow_timer, 3)
        self.assertFalse(sig.is_all_red)

        # 3 seconds of yellow
        sig.update_second()  # 1s
        self.assertTrue(sig.is_yellow)
        sig.update_second()  # 2s
        self.assertTrue(sig.is_yellow)
        sig.update_second()  # 3s: yellow expires -> must enter ALL-RED
        self.assertFalse(sig.is_yellow)
        self.assertTrue(sig.is_all_red, "After yellow, signal must enter all-red clearance!")
        self.assertEqual(sig.all_red_timer, 1)

        # 1 second of all-red -> must activate target green (Phase 2)
        sig.update_second()
        self.assertFalse(sig.is_all_red)
        self.assertEqual(sig.current_phase, 2, "Target green phase (2) must be activated only after all-red!")
        self.assertEqual(sig.time_on_phase, 0)

    # -------------------------------------------------------------------------
    # Test 10: Safe In-Bounds Actions Pass Unmodified
    # -------------------------------------------------------------------------
    def test_10_safe_in_bounds_actions_pass(self):
        """Nominal safe actions (KEEP at t=20s, EXTEND at t=20s) must pass 100% unaltered."""
        signals = {"A": MockSignal(ts_id="A", time_on_phase=20, red_time_ew=20)}
        safe_keep = self.safety.validate_actions({"A": 0}, signals)
        self.assertEqual(safe_keep["A"], 0)

        safe_extend = self.safety.validate_actions({"A": 2}, signals)
        self.assertEqual(safe_extend["A"], 2)

        safe_switch = self.safety.validate_actions({"A": 1}, signals)
        self.assertEqual(safe_switch["A"], 1)

        self.assertEqual(self.safety.total_interventions, 0, "No interventions should occur for safe actions!")

    # -------------------------------------------------------------------------
    # Test 11: Consecutive Actions Handling
    # -------------------------------------------------------------------------
    def test_11_consecutive_actions_handling(self):
        """Repeated switch actions on consecutive steps must not cause duplicate timers or erratic jumps."""
        sig = TrafficSignal(
            ts_id="A",
            incoming_lanes=[],
            green_phases=[0, 2],
            yellow_map={0: 1, 2: 3},
            yellow_duration=3,
            min_green=5,
            max_green=60,
            all_red_duration=1,
            min_red=5
        )
        sig.current_phase = 0
        sig.time_on_phase = 10
        sig.red_time_ew = 10

        # Step 1: initiate switch
        sig.apply_action(1)
        self.assertTrue(sig.is_yellow)
        self.assertEqual(sig.yellow_timer, 3)

        # Step 2: attempt another switch immediately while transitioning
        sig.apply_action(1)
        self.assertTrue(sig.is_yellow)
        self.assertEqual(sig.yellow_timer, 3, "Timer must NOT be reset or duplicated!")

        # Step 3: advance through yellow
        sig.update_second()
        self.assertEqual(sig.yellow_timer, 2)
        sig.apply_action(1)  # redundant switch
        self.assertEqual(sig.yellow_timer, 2, "Timer must continue counting down smoothly!")

    # -------------------------------------------------------------------------
    # Test 12: Independent Validation Across All 4 Intersections
    # -------------------------------------------------------------------------
    def test_12_independent_validation_all_intersections(self):
        """Verify independent validation for A, B, C, D simultaneously."""
        signals = {
            "A": MockSignal("A", time_on_phase=2, red_time_ew=10),   # Under min_green -> switch blocked
            "B": MockSignal("B", time_on_phase=20, red_time_ew=20),  # Safe nominal -> switch allowed
            "C": MockSignal("C", time_on_phase=60, red_time_ew=60),  # Max green -> keep forced to switch
            "D": MockSignal("D", is_yellow=True),                    # Clearance active -> extend clamped to keep
        }
        proposed = {"A": 1, "B": 1, "C": 0, "D": 2}
        safe = self.safety.validate_actions(proposed, signals)

        self.assertEqual(safe["A"], 0, "Agent A: switch must be clamped to 0 (MIN_GREEN)")
        self.assertEqual(safe["B"], 1, "Agent B: switch must be permitted (Nominal safe)")
        self.assertEqual(safe["C"], 1, "Agent C: keep must be overridden to 1 (MAX_GREEN)")
        self.assertEqual(safe["D"], 0, "Agent D: extend must be clamped to 0 (CLEARANCE_ACTIVE)")

    # -------------------------------------------------------------------------
    # Test 13: Final Environment Boundary Cannot Be Bypassed
    # -------------------------------------------------------------------------
    def test_13_boundary_bypass_prevention(self):
        """In MultiAgentTrafficEnv, unsafe actions submitted to step() are intercepted before signal application."""
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        env = MultiAgentTrafficEnv(config_path="config.yaml")
        # Populate mock signals to test step boundary without live sumo
        env.signals = {
            a: MockSignal(a, time_on_phase=2, red_time_ew=10) for a in ["A", "B", "C", "D"]
        }
        env.is_running = True
        env.current_step = 0
        env.safety_enabled = True

        # Try to force unsafe switch (action=1) on all 4 agents
        unsafe_actions = {a: 1 for a in ["A", "B", "C", "D"]}

        # Mock the TraCI calls during step
        with patch("traci.simulation.getTime", return_value=0.0), \
             patch("traci.simulationStep"), \
             patch("traci.simulation.getMinExpectedNumber", return_value=10), \
             patch("traci.simulation.getArrivedNumber", return_value=0):
            obs, rewards, terms, truncs, infos = env.step(unsafe_actions)

        # Verify that all 4 interventions took place and actions were shielded to KEEP (0)
        for a in ["A", "B", "C", "D"]:
            self.assertEqual(infos[a]["validated_action"], 0, f"Action for {a} was not shielded!")
            self.assertEqual(infos[a]["proposed_action"], 1)

    # -------------------------------------------------------------------------
    # Test 14: Reset and Termination Behavior
    # -------------------------------------------------------------------------
    def test_14_reset_and_termination_behavior(self):
        """Reset clears all statistics, intervention history, and pending boundary toggles."""
        # Cause an intervention
        signals = {"A": MockSignal(ts_id="A", time_on_phase=2)}
        self.safety.validate_actions({"A": 1}, signals)
        self.assertEqual(self.safety.total_interventions, 1)

        # Reset
        self.safety.reset()
        self.assertEqual(self.safety.total_interventions, 0)
        self.assertEqual(self.safety.total_proposals, 0)
        self.assertEqual(len(self.safety.recent_interventions), 0)

    # -------------------------------------------------------------------------
    # Test 15: Safety Disabled Mode Exact Reproduction Contract
    # -------------------------------------------------------------------------
    def test_15_safety_disabled_mode_reproduction(self):
        """When safety.enabled == False, actions pass 100% unaltered without any interventions."""
        safety_off = TrafficSignalSafetyLayer(enabled=False)
        signals = {"A": MockSignal(ts_id="A", time_on_phase=2)}  # Would normally violate min_green
        proposed = {"A": 1}  # SWITCH

        result = safety_off.validate_actions(proposed, signals)
        self.assertEqual(result["A"], 1, "When disabled, action must pass completely unaltered!")
        self.assertEqual(safety_off.total_interventions, 0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSafetyLayer)
    print("=" * 80)
    print("RUNNING SAGE-TRAFFIC PHASE 9 SAFETY LAYER TEST SUITE (15 INVARIANTS)")
    print("=" * 80)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 80)
    if result.wasSuccessful():
        print("ALL 15 SAFETY INVARIANT TESTS PASSED SUCCESSFULLY (100% PASS RATE)")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
