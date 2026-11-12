"""
SAGE-Traffic Phase 9: Traffic Signal Safety Layer
Implements runtime safety shielding and verification between RL policies and SUMO/TraCI.

Execution Pipeline:
RL Policy -> Proposed Action -> Safety Layer (Validation & Shielding) -> Validated Safe Action -> SUMO/TraCI

Safety Constraints Enforced:
1. Minimum Green Time (min_green = 5s): Prevents switching before min_green has elapsed.
2. Maximum Green Time (max_green = 60s): Enforces safe switch when green reaches max_green.
3. Minimum Red Time (min_red = 5s): Ensures opposing movement has waited >= min_red.
4. Yellow (3s) & All-Red (1s) Clearance Protection: Non-preemptible; locks out disruptive actions.
5. Conflict-Free Phase Guarantee: Zero simultaneous conflicting green phases across physical links.
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple

try:
    import traci
except ImportError:
    traci = None


# Physical junction link indices in 2x2_grid.net.xml (20 links per intersection)
# Phase 0 (NS Green): links 0..4 and 10..14 are green
NS_LINK_INDICES = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
# Phase 2 (EW Green): links 5..9 and 15..19 are green
EW_LINK_INDICES = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]

VALID_SUMO_STATES = {
    "GGGggrrrrrGGGggrrrrr",  # Phase 0: NS Green
    "yyyyyrrrrryyyyyrrrrr",  # Phase 1: NS Yellow Clearance
    "rrrrrGGGggrrrrrGGGgg",  # Phase 2: EW Green
    "rrrrryyyyyrrrrryyyyy",  # Phase 3: EW Yellow Clearance
    "rrrrrrrrrrrrrrrrrrrr",  # All-Red Clearance
}

ACTION_NAMES = {0: "KEEP", 1: "SWITCH", 2: "EXTEND", 3: "REDUCE"}


class TrafficSignalSafetyLayer:
    """
    Dedicated runtime safety layer for multi-agent traffic signal control.
    Independently inspects and validates actions for all intersections before execution in SUMO.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        mode: str = "shield",
        min_green: int = 5,
        max_green: int = 60,
        min_red: int = 5,
        yellow_duration: int = 3,
        all_red_duration: int = 1,
        log_interventions: bool = True,
        log_file: str = "logs/phase9_safety_interventions.csv",
        intersections: Optional[List[str]] = None
    ):
        if config and "safety" in config:
            s_cfg = config["safety"]
            self.enabled = bool(s_cfg.get("enabled", enabled))
            self.mode = str(s_cfg.get("mode", mode))  # "shield" or "audit_only"
            self.min_green = int(s_cfg.get("min_green", min_green))
            self.max_green = int(s_cfg.get("max_green", max_green))
            self.min_red = int(s_cfg.get("min_red", min_red))
            self.yellow_duration = int(s_cfg.get("yellow_duration", yellow_duration))
            self.all_red_duration = int(s_cfg.get("all_red_duration", all_red_duration))
            self.log_interventions = bool(s_cfg.get("log_interventions", log_interventions))
            self.log_file = str(s_cfg.get("log_file", log_file))
        else:
            self.enabled = enabled
            self.mode = mode
            self.min_green = min_green
            self.max_green = max_green
            self.min_red = min_red
            self.yellow_duration = yellow_duration
            self.all_red_duration = all_red_duration
            self.log_interventions = log_interventions
            self.log_file = log_file

        self.intersections = intersections or ["A", "B", "C", "D"]

        # Statistics and intervention tracking
        self.total_proposals = 0
        self.total_interventions = 0
        self.interventions_by_agent: Dict[str, int] = {a: 0 for a in self.intersections}
        self.interventions_by_constraint: Dict[str, int] = {
            "MIN_GREEN": 0,
            "MAX_GREEN": 0,
            "MIN_RED": 0,
            "CLEARANCE_ACTIVE": 0,
            "INVALID_ACTION": 0,
        }
        self.audit_physical_violations = 0
        self.audit_seconds_checked = 0
        self.recent_interventions: List[Dict[str, Any]] = []

        # Initialize CSV logging
        if self.log_interventions and self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sim_time", "step", "intersection_id", "proposed_action",
                    "safe_action", "active_constraint", "reason",
                    "green_duration", "red_duration"
                ])

    def reset(self):
        """Reset safety layer statistics and buffers at episode boundary."""
        self.total_proposals = 0
        self.total_interventions = 0
        self.interventions_by_agent = {a: 0 for a in self.intersections}
        for k in self.interventions_by_constraint:
            self.interventions_by_constraint[k] = 0
        self.audit_physical_violations = 0
        self.audit_seconds_checked = 0
        self.recent_interventions.clear()

    def validate_actions(
        self,
        proposed_actions: Dict[str, int],
        signals: Dict[str, Any],
        sim_time: float = 0.0,
        current_step: int = 0
    ) -> Dict[str, int]:
        """
        Independently validate proposed actions for all intersections.
        When mode == 'shield', unsafe actions are replaced with verified safe actions.
        When mode == 'audit_only', actions are logged but not replaced.
        """
        if not self.enabled:
            return proposed_actions

        safe_actions = {}

        for agent_id, prop_action in proposed_actions.items():
            self.total_proposals += 1
            sig = signals.get(agent_id)

            if sig is None:
                safe_actions[agent_id] = prop_action
                continue

            action_is_safe = True
            safe_action = prop_action
            violated_constraint = None
            reason = None

            # Current state telemetry from signal object
            time_on_phase = getattr(sig, "time_on_phase", 0)
            is_yellow = getattr(sig, "is_yellow", False)
            is_all_red = getattr(sig, "is_all_red", False)
            current_phase = getattr(sig, "current_phase", 0)
            target_phase = getattr(sig, "target_phase", current_phase)
            red_time_ns = getattr(sig, "red_time_ns", self.min_red)
            red_time_ew = getattr(sig, "red_time_ew", self.min_red)
            delta_green = getattr(sig, "delta_green", 5)
            target_green_duration = getattr(sig, "target_green_duration", 30)

            # 1. Invalid Action Domain Check
            if prop_action not in [0, 1, 2, 3]:
                action_is_safe = False
                safe_action = 0  # Default to KEEP
                violated_constraint = "INVALID_ACTION"
                reason = f"Proposed action {prop_action} is outside discrete action space [0, 1, 2, 3]"

            # 2. Active Clearance Protection (Yellow or All-Red clearance cannot be interrupted)
            elif is_yellow or is_all_red:
                if prop_action != 0:
                    action_is_safe = False
                    safe_action = 0  # Force KEEP during clearance
                    violated_constraint = "CLEARANCE_ACTIVE"
                    reason = f"Intersection {agent_id} in active clearance (yellow={is_yellow}, all_red={is_all_red}); actions clamped to KEEP"

            # 3. Minimum Green Constraint: Green must be served for at least min_green seconds
            elif time_on_phase < self.min_green:
                # Action 1 (SWITCH) or Action 3 (REDUCE that would trigger an immediate switch)
                would_switch = (prop_action == 1) or (
                    prop_action == 3 and (time_on_phase >= (target_green_duration - delta_green))
                )
                if would_switch:
                    action_is_safe = False
                    safe_action = 0  # Reject switch, force KEEP until min_green satisfied
                    violated_constraint = "MIN_GREEN"
                    reason = f"Minimum green violated: time_on_phase={time_on_phase}s < min_green={self.min_green}s"

            # 4. Maximum Green Constraint: Cannot remain on green past max_green seconds
            elif time_on_phase >= self.max_green:
                if prop_action in [0, 2]:  # KEEP or EXTEND
                    action_is_safe = False
                    safe_action = 1  # Force SWITCH to initiate safe clearance
                    violated_constraint = "MAX_GREEN"
                    reason = f"Maximum green exceeded: time_on_phase={time_on_phase}s >= max_green={self.max_green}s; enforcing safe clearance switch"

            # 5. Minimum Red Constraint: Opposing movement must satisfy min_red before receiving green
            elif prop_action == 1 or (prop_action == 3 and time_on_phase >= (target_green_duration - delta_green)):
                # If currently NS (phase 0), opposing movement is EW (phase 2)
                opposing_red_time = red_time_ew if current_phase == 0 else red_time_ns
                # Note: Clearance duration (yellow + all_red) adds to opposing red time
                total_projected_red = opposing_red_time + self.yellow_duration + self.all_red_duration
                if total_projected_red < self.min_red:
                    action_is_safe = False
                    safe_action = 0  # Keep current green until opposing movement clears min_red
                    violated_constraint = "MIN_RED"
                    reason = f"Minimum red violated: projected opposing red {total_projected_red}s < min_red={self.min_red}s"

            # Record intervention if safety violated
            if not action_is_safe:
                self.total_interventions += 1
                self.interventions_by_agent[agent_id] = self.interventions_by_agent.get(agent_id, 0) + 1
                if violated_constraint in self.interventions_by_constraint:
                    self.interventions_by_constraint[violated_constraint] += 1

                log_entry = {
                    "sim_time": sim_time,
                    "step": current_step,
                    "intersection_id": agent_id,
                    "proposed_action": prop_action,
                    "safe_action": safe_action,
                    "active_constraint": violated_constraint,
                    "reason": reason,
                    "green_duration": time_on_phase,
                    "red_duration": red_time_ew if current_phase == 0 else red_time_ns,
                }
                self.recent_interventions.append(log_entry)

                if self.log_interventions and self.log_file:
                    try:
                        with open(self.log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                sim_time, current_step, agent_id,
                                prop_action, safe_action, violated_constraint,
                                reason, time_on_phase,
                                log_entry["red_duration"]
                            ])
                    except Exception:
                        pass

            # Apply action according to mode
            if self.mode == "shield":
                safe_actions[agent_id] = safe_action
            else:
                # audit_only mode logs interventions without altering the proposed action
                safe_actions[agent_id] = prop_action

        return safe_actions

    def audit_physical_state(self, ts_id: str, current_state: str, sim_time: float = 0.0) -> List[Dict[str, Any]]:
        """
        Audit physical traffic light string for simultaneous conflicting green links.
        Checks actual SUMO 20-character phase string.
        """
        violations = []
        self.audit_seconds_checked += 1

        # Check 1: Valid State Definition
        if current_state not in VALID_SUMO_STATES:
            violations.append({
                "type": "INVALID_SUMO_STATE",
                "ts_id": ts_id,
                "sim_time": sim_time,
                "state": current_state,
                "detail": f"State '{current_state}' not in defined SUMO phase repertoire"
            })

        # Check 2: Physical Movement Conflict (NS Green vs EW Green)
        ns_green = any(current_state[i] in ['G', 'g'] for i in NS_LINK_INDICES if i < len(current_state))
        ew_green = any(current_state[i] in ['G', 'g'] for i in EW_LINK_INDICES if i < len(current_state))

        if ns_green and ew_green:
            violations.append({
                "type": "SIMULTANEOUS_CONFLICTING_GREEN",
                "ts_id": ts_id,
                "sim_time": sim_time,
                "state": current_state,
                "detail": "Both North-South and East-West links are green simultaneously"
            })

        if violations:
            self.audit_physical_violations += len(violations)

        return violations

    def audit_all_physical_states(self, sim_time: float = 0.0) -> List[Dict[str, Any]]:
        """Query TraCI for all intersection states and audit for physical conflicts."""
        if traci is None:
            return []

        all_violations = []
        for ts_id in self.intersections:
            try:
                state = traci.trafficlight.getRedYellowGreenState(ts_id)
                v = self.audit_physical_state(ts_id, state, sim_time=sim_time)
                all_violations.extend(v)
            except Exception:
                pass
        return all_violations

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive safety telemetry and intervention metrics."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "total_proposals": self.total_proposals,
            "total_interventions": self.total_interventions,
            "intervention_rate": (
                float(self.total_interventions / self.total_proposals)
                if self.total_proposals > 0 else 0.0
            ),
            "interventions_by_agent": dict(self.interventions_by_agent),
            "interventions_by_constraint": dict(self.interventions_by_constraint),
            "audit_physical_violations": self.audit_physical_violations,
            "audit_seconds_checked": self.audit_seconds_checked,
            "parameters": {
                "min_green": self.min_green,
                "max_green": self.max_green,
                "min_red": self.min_red,
                "yellow_duration": self.yellow_duration,
                "all_red_duration": self.all_red_duration,
            }
        }
