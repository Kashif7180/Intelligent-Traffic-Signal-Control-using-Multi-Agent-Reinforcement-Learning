"""
TraCI-based Multi-Agent Traffic Signal Control Environment for SAGE-Traffic.
Conforms to Gymnasium-style multi-agent interface with 4 agents (A, B, C, D),
configurable normalized observation vectors, 4 discrete actions (keep/switch/extend/reduce),
and multi-objective delay + queue reward functions.
"""

import os
import sys
import yaml
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import sumolib
import traci
from gymnasium import spaces

from environment.demand_generator import generate_routes
from models.safety_layer import TrafficSignalSafetyLayer


def get_sumo_binary(gui: bool = False) -> str:
    """Resolve sumo or sumo-gui binary path across system and standard install directories."""
    bin_name = "sumo-gui" if gui else "sumo"

    # 1. Check explicit standard Windows installation directories
    candidate_dirs = [
        os.environ.get("SUMO_HOME"),
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"D:\Program Files (x86)\Eclipse\Sumo",
        r"D:\Program Files\Eclipse\Sumo",
    ]
    for c_dir in candidate_dirs:
        if c_dir and os.path.exists(c_dir):
            exe_path = os.path.join(c_dir, "bin", f"{bin_name}.exe")
            if os.path.exists(exe_path):
                return exe_path

    # 2. Check via sumolib / PATH
    try:
        return sumolib.checkBinary(bin_name)
    except Exception:
        pass

    # 3. Fallback to pip package location
    try:
        import sumo
        sumo_dir = os.path.dirname(sumo.__file__)
        exe_path = os.path.join(sumo_dir, "bin", f"{bin_name}.exe")
        if os.path.exists(exe_path):
            return exe_path
    except Exception:
        pass

    raise FileNotFoundError(f"Could not locate binary: {bin_name}")



class TrafficSignal:
    """
    Controller for a single signalized intersection.
    Manages green phases, minimum/maximum green bounds, yellow clearance transitions,
    and 4 discrete actions: [0: keep, 1: switch, 2: extend, 3: reduce].
    """

    def __init__(
        self,
        ts_id: str,
        incoming_lanes: list,
        green_phases: list,
        yellow_map: dict,
        yellow_duration: int,
        min_green: int,
        max_green: int = 60,
        delta_green: int = 5,
        default_target_green: int = 30,
        all_red_duration: int = 0,
        min_red: int = 5
    ):
        self.ts_id = ts_id
        self.incoming_lanes = incoming_lanes
        self.green_phases = green_phases  # e.g. [0, 2]
        self.yellow_map = yellow_map      # e.g. {0: 1, 2: 3}
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        self.min_green = min_green
        self.max_green = max_green
        self.min_red = min_red
        self.delta_green = delta_green
        self.default_target_green = default_target_green

        self.current_phase = green_phases[0]
        self.target_phase = green_phases[0]
        self.is_yellow = False
        self.yellow_timer = 0
        self.is_all_red = False
        self.all_red_timer = 0
        self.time_on_phase = 0
        self.target_green_duration = default_target_green
        self.phase_switched_last_step = False

        # Independent movement red duration tracking (seconds)
        self.red_time_ns = 0
        self.red_time_ew = self.min_red

    def initialize_phase(self):
        """Set initial phase and reset timers in TraCI."""
        self.current_phase = self.green_phases[0]
        self.target_phase = self.green_phases[0]
        self.is_yellow = False
        self.yellow_timer = 0
        self.is_all_red = False
        self.all_red_timer = 0
        self.time_on_phase = 0
        self.target_green_duration = self.default_target_green
        self.phase_switched_last_step = False
        self.red_time_ns = 0
        self.red_time_ew = self.min_red
        if traci is not None and traci.isLoaded():
            traci.trafficlight.setPhase(self.ts_id, self.current_phase)

    def _get_alternative_green_phase(self) -> int:
        """Find the other green phase in a 2-phase signal plan."""
        for p in self.green_phases:
            if p != self.current_phase:
                return p
        return self.green_phases[0]

    def _activate_target_green(self):
        """Transition from clearance directly into target green phase."""
        self.current_phase = self.target_phase
        self.time_on_phase = 0
        self.target_green_duration = self.default_target_green
        if traci is not None and traci.isLoaded():
            traci.trafficlight.setPhase(self.ts_id, self.current_phase)

    def _initiate_switch(self):
        """Trigger clearance transition towards the alternative green phase."""
        # Non-preemptible clearance: do not restart timers if already transitioning
        if self.is_yellow or self.is_all_red:
            return

        if self.time_on_phase < self.min_green:
            # Cannot switch before minimum green duration has elapsed
            return

        target_green = self._get_alternative_green_phase()
        self.target_phase = target_green
        yellow_phase = self.yellow_map.get(self.current_phase)

        if yellow_phase is not None and self.yellow_duration > 0:
            self.is_yellow = True
            self.yellow_timer = self.yellow_duration
            self.is_all_red = False
            self.all_red_timer = 0
            if traci is not None and traci.isLoaded():
                traci.trafficlight.setPhase(self.ts_id, yellow_phase)
        elif self.all_red_duration > 0:
            self.is_yellow = False
            self.yellow_timer = 0
            self.is_all_red = True
            self.all_red_timer = self.all_red_duration
            if traci is not None and traci.isLoaded():
                traci.trafficlight.setRedYellowGreenState(self.ts_id, "r" * 20)
        else:
            self._activate_target_green()

        self.phase_switched_last_step = True

    def apply_action(self, action: int):
        """
        Apply one of the 4 discrete actions:
        - 0: KEEP (maintain current phase)
        - 1: SWITCH (switch to opposite green phase via yellow/all-red clearance)
        - 2: EXTEND (increase green allocation by delta_green)
        - 3: REDUCE (decrease green allocation by delta_green; switch if floor met)
        """
        self.phase_switched_last_step = False

        if action == 0:
            # KEEP: Continue current green phase
            pass
        elif action == 1:
            # SWITCH: Force phase change
            self._initiate_switch()
        elif action == 2:
            # EXTEND: Add delta_green to target duration
            self.target_green_duration = min(self.max_green, self.target_green_duration + self.delta_green)
        elif action == 3:
            # REDUCE: Subtract delta_green from target duration
            self.target_green_duration = max(self.min_green, self.target_green_duration - self.delta_green)
            if self.time_on_phase >= self.target_green_duration:
                self._initiate_switch()
        else:
            raise ValueError(f"Unknown action {action} for agent {self.ts_id}. Expected 0, 1, 2, or 3.")

    def update_second(self):
        """Advance internal timers and clearance state by 1 simulation second."""
        # 1. Update independent movement red times
        if self.current_phase == 0:  # Phase 0: NS Green
            if self.is_yellow:
                # NS yellow, EW red
                self.red_time_ns = 0
                self.red_time_ew += 1
            elif self.is_all_red:
                # NS red, EW red
                self.red_time_ns += 1
                self.red_time_ew += 1
            else:
                # NS green, EW red
                self.red_time_ns = 0
                self.red_time_ew += 1
        elif self.current_phase == 2:  # Phase 2: EW Green
            if self.is_yellow:
                # EW yellow, NS red
                self.red_time_ew = 0
                self.red_time_ns += 1
            elif self.is_all_red:
                # EW red, NS red
                self.red_time_ns += 1
                self.red_time_ew += 1
            else:
                # EW green, NS red
                self.red_time_ew = 0
                self.red_time_ns += 1

        # 2. Advance clearance progression
        if self.is_yellow:
            self.yellow_timer -= 1
            if self.yellow_timer <= 0:
                self.is_yellow = False
                if self.all_red_duration > 0:
                    self.is_all_red = True
                    self.all_red_timer = self.all_red_duration
                    if traci is not None and traci.isLoaded():
                        traci.trafficlight.setRedYellowGreenState(self.ts_id, "r" * 20)
                else:
                    self._activate_target_green()
        elif self.is_all_red:
            self.all_red_timer -= 1
            # Check if all-red duration has elapsed and minimum red requirement satisfied
            target_red_time = self.red_time_ew if self.target_phase == 2 else self.red_time_ns
            if self.all_red_timer <= 0 and target_red_time >= self.min_red:
                self.is_all_red = False
                self._activate_target_green()
            # If all_red_timer <= 0 but min_red not yet reached, remain safely in all-red
        else:
            self.time_on_phase += 1
            # Automatic safety switch if max_green ceiling reached
            if self.time_on_phase >= self.max_green:
                self._initiate_switch()


    def get_metrics(self) -> Dict[str, Any]:
        """Query TraCI for raw traffic metrics on all incoming lanes."""
        total_halting = 0
        total_waiting_time = 0.0
        total_veh_count = 0
        speeds = []
        occupancies = []
        lane_details = {}

        for lane_id in self.incoming_lanes:
            halting = traci.lane.getLastStepHaltingNumber(lane_id)
            waiting = traci.lane.getWaitingTime(lane_id)
            veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            occupancy = traci.lane.getLastStepOccupancy(lane_id)

            total_halting += halting
            total_waiting_time += waiting
            total_veh_count += veh_count
            occupancies.append(occupancy)
            if veh_count > 0:
                speeds.append(mean_speed)

            lane_details[lane_id] = {
                "queue_length": halting,
                "waiting_time": waiting,
                "vehicle_count": veh_count,
                "mean_speed": mean_speed,
                "occupancy": occupancy
            }

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        avg_occupancy = float(np.mean(occupancies)) if occupancies else 0.0

        return {
            "queue_length": total_halting,
            "waiting_time": total_waiting_time,
            "vehicle_count": total_veh_count,
            "average_speed": avg_speed,
            "average_occupancy": avg_occupancy,
            "current_phase": self.current_phase,
            "phase_duration": self.time_on_phase,
            "target_green_duration": self.target_green_duration,
            "is_yellow": self.is_yellow,
            "switched": self.phase_switched_last_step,
            "lane_details": lane_details
        }

    def compute_observation(
        self,
        features: List[str],
        norm_cfg: Dict[str, float]
    ) -> np.ndarray:
        """
        Build normalized observation vector based on configured features.
        
        Features supported:
        - vehicle_count: per-lane normalized by max_vehicle_count
        - queue_length: per-lane normalized by max_queue_length
        - average_speed: per-lane normalized by max_speed
        - lane_occupancy: per-lane (0.0 to 1.0)
        - waiting_time: per-lane normalized by max_waiting_time
        - current_phase: one-hot or normalized phase index
        - phase_duration: normalized by max_phase_duration
        """
        obs_components = []
        metrics = self.get_metrics()
        lane_data = metrics["lane_details"]

        # 1. Per-lane features for each incoming lane
        for lane_id in self.incoming_lanes:
            ld = lane_data.get(lane_id, {})
            if "vehicle_count" in features:
                v_norm = min(1.0, ld.get("vehicle_count", 0) / norm_cfg["max_vehicle_count"])
                obs_components.append(v_norm)

            if "queue_length" in features:
                q_norm = min(1.0, ld.get("queue_length", 0) / norm_cfg["max_queue_length"])
                obs_components.append(q_norm)

            if "average_speed" in features:
                s_norm = min(1.0, ld.get("mean_speed", 0.0) / norm_cfg["max_speed"])
                obs_components.append(s_norm)

            if "lane_occupancy" in features:
                occ = min(1.0, max(0.0, ld.get("occupancy", 0.0)))
                obs_components.append(occ)

            if "waiting_time" in features:
                w_norm = min(1.0, ld.get("waiting_time", 0.0) / norm_cfg["max_waiting_time"])
                obs_components.append(w_norm)

        # 2. Intersection-level features
        if "current_phase" in features:
            # One-hot representation for green phases [Phase 0, Phase 2]
            for gp in self.green_phases:
                obs_components.append(1.0 if self.current_phase == gp else 0.0)

        if "phase_duration" in features:
            dur_norm = min(1.0, self.time_on_phase / norm_cfg["max_phase_duration"])
            obs_components.append(dur_norm)

        return np.array(obs_components, dtype=np.float32)

    def compute_reward(self, reward_cfg: Dict[str, float]) -> float:
        """
        Compute per-agent reward function:
        R_i = - (w_queue * Q_i + w_delay * W_i + w_switch * S_i)
        All coefficients loaded from config.yaml.
        """
        metrics = self.get_metrics()
        q = metrics["queue_length"]
        w = metrics["waiting_time"]
        s = 1.0 if metrics["switched"] else 0.0

        w_queue = float(reward_cfg.get("w_queue", 0.1))
        w_delay = float(reward_cfg.get("w_delay", 0.05))
        w_switch = float(reward_cfg.get("w_switch", 0.02))

        reward = - (w_queue * q + w_delay * w + w_switch * s)
        return float(reward)


class MultiAgentTrafficEnv:
    """
    Gymnasium-style Multi-Agent Traffic Signal Control Environment.
    Wraps SUMO TraCI with discrete action spaces and normalized observation vectors.
    """

    def __init__(self, config_path: str = "config.yaml", gui: Optional[bool] = None):
        self.config_path = os.path.abspath(config_path)
        self.base_dir = os.path.dirname(self.config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Configuration sections
        self.sim_cfg = self.config["simulation"]
        self.net_cfg = self.config["network"]
        self.tl_cfg = self.config["traffic_lights"]
        self.agent_cfg = self.config["agent"]
        self.obs_cfg = self.config["observation"]
        self.reward_cfg = self.config["reward"]
        self.demand_cfg = self.config["demand"]

        self.net_file = os.path.join(self.base_dir, self.net_cfg["net_file"])
        self.rou_file = os.path.join(self.base_dir, self.net_cfg["rou_file"])
        self.sumocfg_file = os.path.join(self.base_dir, self.net_cfg["sumocfg_file"])

        # Agents
        self.possible_agents: List[str] = self.agent_cfg["agents"]
        self.agents: List[str] = list(self.possible_agents)
        self.intersections: List[str] = self.agents  # Backward compatibility alias

        # Simulation intervals
        self.step_duration = int(self.sim_cfg["step_duration"])
        self.yellow_duration = int(self.sim_cfg["yellow_duration"])
        self.sim_max_steps = int(self.sim_cfg["sim_max_steps"])
        self.gui = bool(gui) if gui is not None else bool(self.sim_cfg.get("gui", False))
        self.waiting_time_memory = int(self.sim_cfg.get("waiting_time_memory", 1000))

        # Signal parameters
        self.green_phases = self.tl_cfg["green_phases"]
        self.yellow_map = {int(k): int(v) for k, v in self.tl_cfg["yellow_map"].items()}
        self.min_green = int(self.agent_cfg["min_green"])
        self.max_green = int(self.agent_cfg["max_green"])
        self.delta_green = int(self.agent_cfg["delta_green"])
        self.default_green_target = int(self.agent_cfg.get("default_green_target", 30))

        # Observation configuration
        self.features = self.obs_cfg["features"]
        self.normalization = {k: float(v) for k, v in self.obs_cfg["normalization"].items()}

        # Phase 9: Safety Layer Configuration
        self.safety_cfg = self.config.get("safety", {})
        self.safety_enabled = bool(self.safety_cfg.get("enabled", True))
        self.safety_mode = str(self.safety_cfg.get("mode", "shield"))
        self.safety_min_green = int(self.safety_cfg.get("min_green", self.min_green))
        self.safety_max_green = int(self.safety_cfg.get("max_green", self.max_green))
        self.safety_min_red = int(self.safety_cfg.get("min_red", 5))
        self.all_red_duration = int(self.safety_cfg.get("all_red_duration", 1)) if self.safety_enabled else 0
        self._pending_safety_enabled: Optional[bool] = None

        self.safety_layer = TrafficSignalSafetyLayer(
            config=self.config,
            enabled=self.safety_enabled,
            mode=self.safety_mode,
            min_green=self.safety_min_green,
            max_green=self.safety_max_green,
            min_red=self.safety_min_red,
            yellow_duration=self.yellow_duration,
            all_red_duration=self.all_red_duration,
            intersections=self.agents
        )

        # Action and Observation Spaces per agent

        self.action_spaces: Dict[str, spaces.Discrete] = {
            agent_id: spaces.Discrete(4) for agent_id in self.agents
        }

        obs_dim = self._calculate_observation_dimension()
        self.observation_spaces: Dict[str, spaces.Box] = {
            agent_id: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent_id in self.agents
        }

        self.signals: Dict[str, TrafficSignal] = {}
        self.is_running = False
        self.current_step = 0

    def _calculate_observation_dimension(self) -> int:
        """Calculate observation vector length based on incoming lanes and feature list."""
        # Each intersection in the 2x2 grid has 4 incoming edges * 2 lanes = 8 lanes
        num_lanes = 8
        lane_feature_keys = {"vehicle_count", "queue_length", "average_speed", "lane_occupancy", "waiting_time"}
        active_lane_features = [f for f in self.features if f in lane_feature_keys]
        dim = len(active_lane_features) * num_lanes

        if "current_phase" in self.features:
            dim += len(self.green_phases)  # one-hot for green phases
        if "phase_duration" in self.features:
            dim += 1  # normalized duration scalar

        return dim

    def _discover_incoming_lanes(self) -> Dict[str, list]:
        """Read network file to identify incoming lanes for each intersection."""
        net = sumolib.net.readNet(self.net_file)
        incoming_lanes = {}
        for ts_id in self.possible_agents:
            node = net.getNode(ts_id)
            lanes = []
            for edge in node.getIncoming():
                for lane in edge.getLanes():
                    lanes.append(lane.getID())
            incoming_lanes[ts_id] = lanes
        return incoming_lanes

    def action_space(self, agent_id: str) -> spaces.Discrete:
        """Helper returning action space for an agent."""
        return self.action_spaces[agent_id]

    def observation_space(self, agent_id: str) -> spaces.Box:
        """Helper returning observation space for an agent."""
        return self.observation_spaces[agent_id]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        demand_level: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset simulation and return (observations, infos) keyed by agent ID.
        """
        self.close()

        if seed is None:
            seed = int(self.sim_cfg.get("seed", 42))

        if demand_level is None:
            demand_level = self.demand_cfg.get("active_demand", "normal")

        # Generate routes if needed
        generate_routes(self.config_path, demand_level=demand_level, seed=seed)

        sumo_bin = get_sumo_binary(self.gui)
        sumo_cmd = [
            sumo_bin,
            "-c", self.sumocfg_file,
            "--route-files", self.rou_file,
            "--no-step-log", "true",
            "--waiting-time-memory", str(self.waiting_time_memory),
            "--time-to-teleport", "-1",
            "--seed", str(seed)
        ]

        if self.gui:
            sumo_cmd.extend(["--start", "--quit-on-end"])

        # Check for pending safety toggle deferred to episode boundary
        if self._pending_safety_enabled is not None:
            self.safety_enabled = self._pending_safety_enabled
            self.safety_layer.enabled = self._pending_safety_enabled
            self.all_red_duration = int(self.safety_cfg.get("all_red_duration", 1)) if self.safety_enabled else 0
            self._pending_safety_enabled = None

        self.safety_layer.reset()

        traci.start(sumo_cmd)
        self.is_running = True
        self.current_step = 0
        self.agents = list(self.possible_agents)

        # Initialize signals
        lane_map = self._discover_incoming_lanes()
        self.signals = {}
        for ts_id in self.agents:
            sig = TrafficSignal(
                ts_id=ts_id,
                incoming_lanes=lane_map[ts_id],
                green_phases=self.green_phases,
                yellow_map=self.yellow_map,
                yellow_duration=self.yellow_duration,
                min_green=self.min_green,
                max_green=self.max_green,
                delta_green=self.delta_green,
                default_target_green=self.default_green_target,
                all_red_duration=self.all_red_duration if self.safety_enabled else 0,
                min_red=self.safety_min_red
            )
            sig.initialize_phase()
            self.signals[ts_id] = sig

        observations = {
            agent_id: self.signals[agent_id].compute_observation(self.features, self.normalization)
            for agent_id in self.agents
        }

        infos = {
            agent_id: {
                "metrics": self.signals[agent_id].get_metrics(),
                "sim_time": traci.simulation.getTime() if (traci is not None and traci.isLoaded()) else 0.0,
                "safety": self.safety_layer.get_statistics()
            }
            for agent_id in self.agents
        }

        return observations, infos

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """
        Apply discrete actions per agent, advance simulation, and return:
        (observations, rewards, terminations, truncations, infos) all keyed by agent ID.
        Actions strictly pass through the runtime safety layer before environment application.
        """
        if not self.is_running:
            raise RuntimeError("Simulation is not running. Call reset() before step().")

        # Pass through unbypassable safety layer
        current_sim_time = traci.simulation.getTime() if (traci is not None and traci.isLoaded()) else float(self.current_step * self.step_duration)
        if self.safety_enabled:
            validated_actions = self.safety_layer.validate_actions(
                proposed_actions=actions,
                signals=self.signals,
                sim_time=current_sim_time,
                current_step=self.current_step
            )
        else:
            validated_actions = actions

        # Apply validated discrete action (0: keep, 1: switch, 2: extend, 3: reduce)
        for agent_id, action in validated_actions.items():
            if agent_id in self.signals:
                self.signals[agent_id].apply_action(action)

        # Advance simulation second-by-second for step_duration
        for _ in range(self.step_duration):
            for sig in self.signals.values():
                sig.update_second()
            if traci is not None and traci.isLoaded():
                traci.simulationStep()
                if self.safety_enabled:
                    sec_time = traci.simulation.getTime()
                    self.safety_layer.audit_all_physical_states(sim_time=sec_time)

        self.current_step += 1

        # Check termination & truncation
        sim_ended = traci.simulation.getMinExpectedNumber() <= 0 if (traci is not None and traci.isLoaded()) else False
        truncated_flag = self.current_step >= self.sim_max_steps

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        arrived_this_step = traci.simulation.getArrivedNumber() if (traci is not None and traci.isLoaded()) else 0

        for agent_id in self.agents:
            sig = self.signals[agent_id]
            observations[agent_id] = sig.compute_observation(self.features, self.normalization)
            rewards[agent_id] = sig.compute_reward(self.reward_cfg)
            terminations[agent_id] = sim_ended
            truncations[agent_id] = truncated_flag
            infos[agent_id] = {
                "metrics": sig.get_metrics(),
                "current_step": self.current_step,
                "sim_time": traci.simulation.getTime() if (traci is not None and traci.isLoaded()) else float(self.current_step * self.step_duration),
                "active_vehicles": traci.vehicle.getIDCount() if (traci is not None and traci.isLoaded()) else 0,
                "arrived_vehicles": arrived_this_step,
                "safety": self.safety_layer.get_statistics(),
                "validated_action": validated_actions.get(agent_id, 0),
                "proposed_action": actions.get(agent_id, 0)
            }

        return observations, rewards, terminations, truncations, infos

    def set_safety_enabled(self, enabled: bool) -> bool:
        """
        Safely toggle runtime safety shielding.
        Requirement: Safety cannot be disabled during an active yellow/all-red transition.
        If any intersection is transitioning, the change is deferred to next episode reset.
        Returns True if immediately applied, False if deferred to episode boundary.
        """
        if not enabled and self.is_running:
            any_transition = any(
                (sig.is_yellow or getattr(sig, "is_all_red", False))
                for sig in self.signals.values()
            )
            if any_transition:
                self._pending_safety_enabled = enabled
                return False

        self.safety_enabled = enabled
        self.safety_layer.enabled = enabled
        self.all_red_duration = int(self.safety_cfg.get("all_red_duration", 1)) if enabled else 0
        for sig in self.signals.values():
            sig.all_red_duration = self.all_red_duration if enabled else 0
        return True


    def get_intersection_metrics(self, intersection_id: str) -> Dict[str, Any]:
        """Get raw metrics for a specific intersection."""
        if intersection_id not in self.signals:
            raise KeyError(f"Intersection {intersection_id} not found.")
        return self.signals[intersection_id].get_metrics()

    def get_all_intersection_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get raw metrics for all intersections."""
        return {agent_id: self.get_intersection_metrics(agent_id) for agent_id in self.agents}

    def close(self):
        """Safely terminate TraCI simulation instance."""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_running = False

    def __del__(self):
        self.close()


# Alias TrafficEnv to MultiAgentTrafficEnv for direct backward compatibility
TrafficEnv = MultiAgentTrafficEnv
