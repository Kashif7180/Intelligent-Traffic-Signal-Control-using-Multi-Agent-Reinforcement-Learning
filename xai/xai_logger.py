"""
XAI Logger Module
=================
Lightweight, non-invasive logging engine for multi-agent XAI telemetry and explanations.
"""

import json
import csv
import os
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .decision_explainer import DecisionExplanation
from .attention_explainer import AttentionExplanation
from .communication_explainer import CommunicationExplanation
from .causal_explainer import CausalExplanation
from .safety_explainer import SafetyExplanation


class XAILogger:
    """Stores and exports structured XAI records across simulation timesteps."""

    def __init__(self, max_buffer_size: int = 500, log_file: Optional[str] = None):
        self.max_buffer_size = max_buffer_size
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []

    def record_step(
        self,
        step: int,
        sim_time: float,
        controller_id: str,
        decisions: Dict[str, DecisionExplanation],
        safety: Optional[Dict[str, SafetyExplanation]] = None,
        attention: Optional[AttentionExplanation] = None,
        communication: Optional[CommunicationExplanation] = None,
        causal: Optional[CausalExplanation] = None
    ) -> Dict[str, Any]:
        """
        Record a comprehensive explanation snapshot for a single simulation step.
        """
        record = {
            "step": step,
            "sim_time": round(sim_time, 2),
            "controller_id": controller_id,
            "decisions": {agent: d.to_dict() for agent, d in decisions.items()},
            "safety": {agent: s.to_dict() for agent, s in safety.items()} if safety else {},
            "attention": attention.to_dict() if attention else None,
            "communication": communication.to_dict() if communication else None,
            "causal": causal.to_dict() if causal else None
        }

        self.history.append(record)
        if len(self.history) > self.max_buffer_size:
            self.history.pop(0)

        return record

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent XAI explanation record."""
        return self.history[-1] if self.history else None

    def get_agent_decision(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest decision explanation for a specific agent."""
        latest = self.get_latest()
        if latest and "decisions" in latest:
            return latest["decisions"].get(agent_id)
        return None

    def export_jsonl(self, filepath: str):
        """Export the buffered history to JSON Lines format."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in self.history:
                f.write(json.dumps(item) + "\n")

    def clear(self):
        """Reset the telemetry buffer."""
        self.history.clear()

