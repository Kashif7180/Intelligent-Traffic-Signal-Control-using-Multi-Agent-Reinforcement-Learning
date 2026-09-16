"""
Deterministic Fixed-Time Baseline Controller for SAGE-Traffic.
Follows a standard pre-timed schedule (e.g., 30s NS green, 3s yellow, 30s EW green, 3s yellow)
without learning or RL components.
"""

import yaml
from typing import Dict, Any, Optional


class FixedTimeController:
    """
    Deterministic Fixed-Time traffic signal controller.
    Maintains green phases for a fixed duration (e.g. 30s), then issues a switch action.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.agents = config["agent"]["agents"]
        self.step_duration = int(config["simulation"]["step_duration"])
        self.green_time = int(config.get("fixed_time", {}).get("green_time", 30))
        self.yellow_time = int(config.get("fixed_time", {}).get("yellow_time", 3))

        # Internal state tracking per intersection
        self.time_on_phase: Dict[str, int] = {agent: 0 for agent in self.agents}

    def reset(self):
        """Reset internal timers at the beginning of an episode."""
        self.time_on_phase = {agent: 0 for agent in self.agents}

    def get_actions(self, observations: Dict[str, Any], infos: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Determine deterministic action per intersection.
        Actions:
        0 = keep phase (continue current green)
        1 = switch phase (trigger yellow clearance and switch to opposite phase)
        """
        actions = {}
        for agent in self.agents:
            # Advance internal timer
            self.time_on_phase[agent] += self.step_duration

            if self.time_on_phase[agent] >= self.green_time:
                # Green interval elapsed -> issue SWITCH action
                actions[agent] = 1
                self.time_on_phase[agent] = 0
            else:
                # Continue current green phase -> issue KEEP action
                actions[agent] = 0

        return actions
