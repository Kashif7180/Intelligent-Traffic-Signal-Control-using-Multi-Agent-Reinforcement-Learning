"""
Safety Shield Explainer Module (XAI)
====================================
Explains runtime safety layer interventions, invariant boundary clamping,
and physical clearance guarantees.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from models.safety_layer import ACTION_NAMES


@dataclass
class SafetyExplanation:
    """Structured explanation of safety layer validation and interventions."""
    agent_id: str
    proposed_action: int
    proposed_action_name: str
    executed_action: int
    executed_action_name: str
    is_safe: bool
    active_constraint: Optional[str]
    reason: Optional[str]
    time_on_phase: float
    min_green_threshold: float
    grounded_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "proposed_action": self.proposed_action,
            "proposed_action_name": self.proposed_action_name,
            "executed_action": self.executed_action,
            "executed_action_name": self.executed_action_name,
            "is_safe": self.is_safe,
            "active_constraint": self.active_constraint,
            "reason": self.reason,
            "time_on_phase": round(self.time_on_phase, 1),
            "min_green_threshold": self.min_green_threshold,
            "summary": self.grounded_summary
        }


class SafetyExplainer:
    """Explains runtime safety shield validations and invariant clamping."""

    def __init__(self, min_green: float = 5.0, max_green: float = 60.0):
        self.min_green = min_green
        self.max_green = max_green

    def explain(
        self,
        agent_id: str,
        proposed_action: int,
        executed_action: int,
        time_on_phase: float,
        active_constraint: Optional[str] = None,
        reason: Optional[str] = None
    ) -> SafetyExplanation:
        """
        Explain the verification outcome between proposed and executed actions.
        """
        prop_name = ACTION_NAMES.get(proposed_action, str(proposed_action))
        exec_name = ACTION_NAMES.get(executed_action, str(executed_action))
        is_safe = (proposed_action == executed_action) and (active_constraint is None)

        if is_safe:
            summary = (
                f"Action SAFE: Node {agent_id} proposed {prop_name} ({proposed_action}), "
                f"which complies with all physical invariants (green timer {time_on_phase:.1f}s >= min {self.min_green}s). "
                f"Executed without shield intervention."
            )
        else:
            c_name = active_constraint or "SAFETY_INVARIANT"
            if c_name == "MIN_GREEN":
                summary = (
                    f"SHIELD INTERVENTION (MIN_GREEN): Node {agent_id} policy attempted {prop_name} ({proposed_action}) "
                    f"at {time_on_phase:.1f}s green time. Invariant requires at least {self.min_green}s of continuous green. "
                    f"Shield clamped action to {exec_name} ({executed_action}) to prevent hazardous short-cycling."
                )
            elif c_name == "CLEARANCE_ACTIVE":
                summary = (
                    f"SHIELD INTERVENTION (CLEARANCE_ACTIVE): Node {agent_id} is actively executing a clearance interval "
                    f"(yellow change or all-red buffer). Actions are clamped to {exec_name} ({executed_action}) until clearance completes."
                )
            elif c_name == "MAX_GREEN":
                summary = (
                    f"SHIELD INTERVENTION (MAX_GREEN): Node {agent_id} green time ({time_on_phase:.1f}s) reached maximum boundary ({self.max_green}s). "
                    f"Enforced {exec_name} ({executed_action}) to prevent phase starvation on opposing approaches."
                )
            elif c_name == "MIN_RED":
                summary = (
                    f"SHIELD INTERVENTION (MIN_RED): Opposing approach has not yet satisfied minimum clearance time. "
                    f"Switch proposal deferred by clamping to {exec_name} ({executed_action})."
                )
            else:
                summary = (
                    f"SHIELD INTERVENTION ({c_name}): Node {agent_id} proposed action {prop_name} ({proposed_action}) was overridden "
                    f"to {exec_name} ({executed_action}). Detail: {reason or 'Safety boundary violation.'}"
                )

        return SafetyExplanation(
            agent_id=agent_id,
            proposed_action=proposed_action,
            proposed_action_name=prop_name,
            executed_action=executed_action,
            executed_action_name=exec_name,
            is_safe=is_safe,
            active_constraint=active_constraint,
            reason=reason,
            time_on_phase=time_on_phase,
            min_green_threshold=self.min_green,
            grounded_summary=summary
        )

