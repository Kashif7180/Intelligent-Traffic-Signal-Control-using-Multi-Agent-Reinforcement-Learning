"""
Decision Explainer Module (XAI)
===============================
Extracts, structures, and interprets action distributions, confidence margins,
and local traffic context to explain individual agent signal control decisions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from models.safety_layer import ACTION_NAMES


@dataclass
class DecisionExplanation:
    """Structured decision explanation record for a single intersection."""
    agent_id: str
    selected_action: int
    selected_action_name: str
    action_probabilities: Dict[int, float]
    action_names_probs: Dict[str, float]
    alternative_action: int
    alternative_action_name: str
    confidence_margin: float
    policy_entropy: float
    current_phase_name: str
    phase_duration: float
    queue_length: float
    waiting_time: float
    vehicle_count: int
    average_speed: float
    grounded_summary: str
    is_safe: bool = True
    executed_action: int = 0
    executed_action_name: str = "KEEP"
    override_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "selected_action": self.selected_action,
            "selected_action_name": self.selected_action_name,
            "action_probabilities": self.action_probabilities,
            "alternative_action": self.alternative_action,
            "alternative_action_name": self.alternative_action_name,
            "confidence_margin": round(self.confidence_margin, 4),
            "policy_entropy": round(self.policy_entropy, 4),
            "current_phase_name": self.current_phase_name,
            "phase_duration": round(self.phase_duration, 1),
            "queue_length": round(self.queue_length, 2),
            "waiting_time": round(self.waiting_time, 2),
            "vehicle_count": self.vehicle_count,
            "average_speed": round(self.average_speed, 2),
            "grounded_summary": self.grounded_summary,
            "is_safe": self.is_safe,
            "executed_action": self.executed_action,
            "executed_action_name": self.executed_action_name,
            "override_reason": self.override_reason
        }


class DecisionExplainer:
    """Explains decentralized actor decisions using actual policy distributions and local observations."""

    def __init__(self, min_green: float = 5.0, max_green: float = 60.0):
        self.min_green = min_green
        self.max_green = max_green

    def explain(
        self,
        agent_id: str,
        action_probs: np.ndarray,
        selected_action: int,
        obs: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, Any]] = None,
        proposed_action: Optional[int] = None,
        executed_action: Optional[int] = None,
        override_reason: Optional[str] = None
    ) -> DecisionExplanation:
        """
        Generate a data-grounded decision explanation for an agent.
        
        Args:
            agent_id: Intersection identifier ("A", "B", "C", "D")
            action_probs: 1D array of shape [4] representing softmax probabilities
            selected_action: Action chosen by policy (argmax or sample)
            obs: 43-D or 47-D raw observation vector
            metrics: Intersection metrics from traffic environment
            proposed_action: Action proposed to safety shield
            executed_action: Action returned by safety shield
            override_reason: Invariant violation reason if clamped
        """
        probs = np.asarray(action_probs, dtype=np.float32).flatten()
        if len(probs) != 4:
            probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        # Probs mapping
        prob_dict = {i: float(probs[i]) for i in range(4)}
        name_prob_dict = {ACTION_NAMES.get(i, f"ACT_{i}"): float(probs[i]) for i in range(4)}

        # Find best alternative action
        sorted_indices = np.argsort(-probs)
        best_act = int(sorted_indices[0])
        alt_act = int(sorted_indices[1]) if len(sorted_indices) > 1 else best_act

        margin = float(probs[selected_action] - probs[alt_act]) if selected_action != alt_act else float(probs[selected_action] - probs[sorted_indices[1]])
        
        # Policy entropy: H(p) = -sum p log p
        clipped_probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(clipped_probs * np.log(clipped_probs)))

        # Extract traffic context metrics
        metrics = metrics or {}
        queue_len = float(metrics.get("queue_length", 0.0))
        wait_time = float(metrics.get("waiting_time", 0.0))
        veh_count = int(metrics.get("vehicle_count", 0))
        avg_speed = float(metrics.get("average_speed", 0.0))
        curr_phase = int(metrics.get("current_phase", 0))
        phase_duration = float(metrics.get("phase_duration", 0.0))

        phase_name = "NS GREEN" if curr_phase == 0 else ("EW GREEN" if curr_phase == 2 else f"PHASE_{curr_phase}")

        prop_act = proposed_action if proposed_action is not None else selected_action
        exec_act = executed_action if executed_action is not None else selected_action
        is_safe = (prop_act == exec_act) and (override_reason is None)

        # Construct grounded explanation narrative
        summary = self._synthesize_grounded_rationale(
            agent_id=agent_id,
            selected_action=selected_action,
            selected_prob=probs[selected_action],
            alt_action=alt_act,
            alt_prob=probs[alt_act],
            margin=margin,
            phase_name=phase_name,
            phase_duration=phase_duration,
            queue_len=queue_len,
            wait_time=wait_time,
            is_safe=is_safe,
            override_reason=override_reason
        )

        return DecisionExplanation(
            agent_id=agent_id,
            selected_action=selected_action,
            selected_action_name=ACTION_NAMES.get(selected_action, "UNKNOWN"),
            action_probabilities=prob_dict,
            action_names_probs=name_prob_dict,
            alternative_action=alt_act,
            alternative_action_name=ACTION_NAMES.get(alt_act, "UNKNOWN"),
            confidence_margin=margin,
            policy_entropy=entropy,
            current_phase_name=phase_name,
            phase_duration=phase_duration,
            queue_length=queue_len,
            waiting_time=wait_time,
            vehicle_count=veh_count,
            average_speed=avg_speed,
            grounded_summary=summary,
            is_safe=is_safe,
            executed_action=exec_act,
            executed_action_name=ACTION_NAMES.get(exec_act, "UNKNOWN"),
            override_reason=override_reason
        )

    def _synthesize_grounded_rationale(
        self,
        agent_id: str,
        selected_action: int,
        selected_prob: float,
        alt_action: int,
        alt_prob: float,
        margin: float,
        phase_name: str,
        phase_duration: float,
        queue_len: float,
        wait_time: float,
        is_safe: bool,
        override_reason: Optional[str]
    ) -> str:
        act_name = ACTION_NAMES.get(selected_action, "UNKNOWN")
        alt_name = ACTION_NAMES.get(alt_action, "UNKNOWN")

        confidence_level = "high" if selected_prob >= 0.70 else ("moderate" if selected_prob >= 0.40 else "low/ambiguous")

        text = (
            f"Node {agent_id} policy selected {act_name} with {confidence_level} confidence "
            f"({selected_prob*100:.1f}%, margin +{margin*100:.1f}% over {alt_name} at {alt_prob*100:.1f}%). "
            f"Current state: {phase_name} has elapsed {phase_duration:.1f}s with {queue_len:.1f} queued vehicles "
            f"and {wait_time:.1f}s total delay. "
        )

        # Context-dependent behavioral interpretation (Contextual heuristic, not proven neural feature attribution)
        if selected_action == 1:  # SWITCH
            if phase_duration < self.min_green:
                text += f"Contextual heuristic: Policy requested phase termination before min_green ({self.min_green}s) elapsed. "
            else:
                text += f"Contextual heuristic: Green phase criteria satisfied (duration {phase_duration:.1f}s >= {self.min_green}s); policy switched during observed opposing queue. "
        elif selected_action == 0:  # KEEP
            text += f"Contextual heuristic: Maintaining active green phase consistent with observed traffic movement. "
        elif selected_action == 2:  # EXTEND
            text += f"Contextual heuristic: Extending green phase consistent with observed lane demand. "
        elif selected_action == 3:  # REDUCE
            text += f"Contextual heuristic: Reducing green allocation prior to phase change. "

        text += "(Note: Traffic observations describe operational context, not verified neural feature attributions.)"

        if not is_safe:
            text += f" [SAFETY SHIELD OVERRIDE: {override_reason}]"

        return text
