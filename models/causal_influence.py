"""
Phase 8: Causal Influence Estimation Module for SAGE-Traffic.

Implements directional causal influence estimation:
    C_ij in [0, 1]
for each allowed directed communication edge j -> i.

Terminology Rule:
Everywhere in code, variables, logs, comments, and reports, this is referred to as:
    "causal influence estimation"
We do NOT claim true causality, discovered causality, or ground-truth causal effect.
The output is an APPROXIMATION of causal influence based on observable traffic history.

Features used:
1. Upstream congestion / corridor queue (c_{j -> i})
2. Source queue (q_j)
3. Source demand / vehicle count (d_j)
4. Source waiting time (w_j)
5. Downstream/target queue (q_i)
6. Downstream/target waiting time (w_i)

Methods supported:
- "granger_style": Linear predictive variance reduction via ridge regression on lagged history.
- "fixed_structural": Fixed structural causal influence approximation based on smoothed traffic variables.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any


# Exactly 8 allowed physical directed inter-agent communication channels
ALLOWED_DIRECTED_EDGES: List[Tuple[str, str]] = [
    ("A", "B"), ("B", "A"),
    ("A", "C"), ("C", "A"),
    ("B", "D"), ("D", "B"),
    ("C", "D"), ("D", "C")
]

# Physical corridor incoming lane mapping at target from source
CORRIDOR_INCOMING_LANES: Dict[Tuple[str, str], List[str]] = {
    ("A", "B"): ["A_to_B_0", "A_to_B_1"],
    ("B", "A"): ["B_to_A_0", "B_to_A_1"],
    ("A", "C"): ["A_to_C_0", "A_to_C_1"],
    ("C", "A"): ["C_to_A_0", "C_to_A_1"],
    ("B", "D"): ["B_to_D_0", "B_to_D_1"],
    ("D", "B"): ["D_to_B_0", "D_to_B_1"],
    ("C", "D"): ["C_to_D_0", "C_to_D_1"],
    ("D", "C"): ["D_to_C_0", "D_to_C_1"]
}


class CausalInfluenceEstimator:
    """
    Causal Influence Estimator.
    
    Maintains temporal traffic history and estimates directional causal influence C_ij in [0, 1]
    for each allowed edge j -> i without future information leakage.
    """

    def __init__(
        self,
        method: str = "granger_style",
        history_length: int = 5,
        window_size: int = 15,
        score_min: float = 0.0,
        score_max: float = 1.0,
        warmup_value: float = 0.5,
        ridge_lambda: float = 1e-3,
        max_queue: float = 40.0,
        max_wait: float = 120.0,
        max_veh: float = 40.0,
        agents: List[str] = ["A", "B", "C", "D"]
    ):
        self.method = method
        self.history_length = int(history_length)
        self.window_size = int(window_size)
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.warmup_value = float(warmup_value)
        self.ridge_lambda = float(ridge_lambda)

        self.max_queue = float(max_queue)
        self.max_wait = float(max_wait)
        self.max_veh = float(max_veh)
        self.agents = agents
        self.allowed_edges = ALLOWED_DIRECTED_EDGES

        # Temporal traffic history buffers per agent and per edge
        # Stores list of normalized feature dictionaries per timestep:
        # agent_history[agent] = list of {"queue": float, "wait": float, "demand": float}
        self.agent_history: Dict[str, List[Dict[str, float]]] = {a: [] for a in agents}
        # edge_history[(src, tgt)] = list of {"corridor_queue": float}
        self.edge_history: Dict[Tuple[str, str], List[float]] = {e: [] for e in self.allowed_edges}

        # Cached last estimates for inspection/logging
        self.last_scores: Dict[Tuple[str, str], float] = {e: self.warmup_value for e in self.allowed_edges}
        self.current_step: int = 0

    def reset(self):
        """Reset temporal history at the start of an episode."""
        self.agent_history = {a: [] for a in self.agents}
        self.edge_history = {e: [] for e in self.allowed_edges}
        self.last_scores = {e: self.warmup_value for e in self.allowed_edges}
        self.current_step = 0

    def update_traffic_state(self, info_dict: Dict[str, Any]):
        """
        Record observable traffic variables for step t into temporal history.
        
        Args:
            info_dict: Standard environment info dict keyed by agent ID, containing 'metrics'.
        """
        self.current_step += 1

        # 1. Update per-agent traffic variables
        for agent in self.agents:
            metrics = info_dict.get(agent, {}).get("metrics", {})
            raw_q = float(metrics.get("queue_length", 0.0))
            raw_w = float(metrics.get("waiting_time", 0.0))
            raw_v = float(metrics.get("vehicle_count", 0.0))

            norm_q = float(np.clip(raw_q / self.max_queue, 0.0, 1.0))
            norm_w = float(np.clip(raw_w / self.max_wait, 0.0, 1.0))
            norm_d = float(np.clip(raw_v / self.max_veh, 0.0, 1.0))

            self.agent_history[agent].append({
                "queue": norm_q,
                "wait": norm_w,
                "demand": norm_d,
                "raw_queue": raw_q,
                "raw_wait": raw_w,
                "raw_demand": raw_v
            })

            # Trim history to rolling window size
            if len(self.agent_history[agent]) > self.window_size:
                self.agent_history[agent].pop(0)

        # 2. Update connecting corridor queue for each allowed directed edge
        for src, tgt in self.allowed_edges:
            tgt_metrics = info_dict.get(tgt, {}).get("metrics", {})
            lane_details = tgt_metrics.get("lane_details", {})
            incoming_lanes = CORRIDOR_INCOMING_LANES.get((src, tgt), [])

            corridor_q = 0.0
            for lane_id in incoming_lanes:
                if lane_id in lane_details:
                    corridor_q += float(lane_details[lane_id].get("queue_length", 0.0))

            norm_corridor_q = float(np.clip(corridor_q / self.max_queue, 0.0, 1.0))
            self.edge_history[(src, tgt)].append(norm_corridor_q)

            if len(self.edge_history[(src, tgt)]) > self.window_size:
                self.edge_history[(src, tgt)].pop(0)

    def estimate_causal_influence(self) -> Dict[Tuple[str, str], float]:
        """
        Compute directional causal influence estimation score C_ij in [0, 1]
        for all 8 allowed edges j -> i.
        
        Returns:
            Dictionary mapping (src, tgt) -> C_ij score.
        """
        scores = {}

        # Check warm-up: if history is shorter than history_length, return warmup_value
        min_hist_len = min(len(self.agent_history[a]) for a in self.agents) if self.agents else 0
        if min_hist_len < self.history_length:
            for edge in self.allowed_edges:
                scores[edge] = float(self.warmup_value)
            self.last_scores = scores
            return scores

        for src, tgt in self.allowed_edges:
            if self.method == "granger_style":
                score = self._compute_granger_influence(src, tgt)
            elif self.method == "fixed_structural":
                score = self._compute_structural_influence(src, tgt)
            else:
                score = self._compute_granger_influence(src, tgt)

            # Strict bounds clipping to [score_min, score_max]
            clipped_score = float(np.clip(score, self.score_min, self.score_max))
            scores[(src, tgt)] = clipped_score

        self.last_scores = scores
        return scores

    def _compute_granger_influence(self, src: str, tgt: str) -> float:
        """
        Granger-style causal influence estimation approximation.
        
        Evaluates whether source j's lagged traffic features provide predictive variance reduction
        for target i's future traffic state compared to an autoregressive model of target i alone.
        
        Models:
        Target signal: y_i(t) = 0.5 * q_i(t) + 0.5 * w_i(t)
        Source features: x_j(t) = [q_j(t), w_j(t), d_j(t), c_{j -> i}(t)]
        
        Lag order p = 2 (or 1 if short window).
        Restricted (target only): y_i(t) = a0 + sum_{l=1}^p a_l y_i(t-l)
        Unrestricted (target + source): y_i(t) = b0 + sum_{l=1}^p a_l y_i(t-l) + sum_{l=1}^p b_l^T x_j(t-l)
        
        Predictive improvement score:
        Delta_RSS = max(0, RSS_R - RSS_U)
        C_ij = clip(Delta_RSS / (RSS_R + 1e-5), 0.0, 1.0)
        """
        tgt_hist = self.agent_history[tgt]
        src_hist = self.agent_history[src]
        corr_hist = self.edge_history[(src, tgt)]

        N = min(len(tgt_hist), len(src_hist), len(corr_hist))
        if N < self.history_length:
            return float(self.warmup_value)

        # Construct time series vectors
        y_tgt = np.array([0.5 * h["queue"] + 0.5 * h["wait"] for h in tgt_hist[-N:]], dtype=np.float64)
        X_src = np.array([
            [src_hist[k]["queue"], src_hist[k]["wait"], src_hist[k]["demand"], corr_hist[k]]
            for k in range(-N, 0)
        ], dtype=np.float64)

        # Lags p
        p = 2 if N >= 6 else 1
        num_samples = N - p
        if num_samples < 2:
            return float(self.warmup_value)

        # Target vector Y to predict: y(p), y(p+1), ..., y(N-1)
        Y = y_tgt[p:]  # shape [num_samples]

        # Restricted Design Matrix: [1, y_tgt(t-1), ..., y_tgt(t-p)]
        X_R_cols = [np.ones(num_samples, dtype=np.float64)]
        for lag in range(1, p + 1):
            X_R_cols.append(y_tgt[p - lag: N - lag])
        X_R = np.column_stack(X_R_cols)  # [num_samples, 1 + p]

        # Unrestricted Design Matrix: X_R + source lags
        X_U_cols = list(X_R_cols)
        for lag in range(1, p + 1):
            X_U_cols.append(X_src[p - lag: N - lag])  # [num_samples, 4]
        X_U = np.column_stack(X_U_cols)  # [num_samples, 1 + p + 4*p]

        # Ridge regression solve for Restricted Model: (X^T X + lambda I)^-1 X^T Y
        try:
            ridge_R = self.ridge_lambda * np.eye(X_R.shape[1], dtype=np.float64)
            beta_R = np.linalg.solve(X_R.T @ X_R + ridge_R, X_R.T @ Y)
            pred_R = X_R @ beta_R
            rss_R = float(np.sum((Y - pred_R) ** 2))

            ridge_U = self.ridge_lambda * np.eye(X_U.shape[1], dtype=np.float64)
            beta_U = np.linalg.solve(X_U.T @ X_U + ridge_U, X_U.T @ Y)
            pred_U = X_U @ beta_U
            rss_U = float(np.sum((Y - pred_U) ** 2))

            # Predictive variance reduction score
            delta_rss = max(0.0, rss_R - rss_U)
            gain = delta_rss / (rss_R + 1e-5)

            # Factor in instantaneous upstream corridor pressure as directional amplifier:
            recent_corridor = float(corr_hist[-1])
            recent_src_demand = float(src_hist[-1]["demand"])
            directional_factor = 0.5 * gain + 0.25 * recent_corridor + 0.25 * recent_src_demand

            return float(np.clip(directional_factor, 0.0, 1.0))
        except np.linalg.LinAlgError:
            return float(self.warmup_value)

    def _compute_structural_influence(self, src: str, tgt: str) -> float:
        """
        Fixed structural causal influence approximation.
        
        Computes a directional score using weighted lagged traffic pressure:
        C_ij = clip(0.35 * q_j + 0.25 * w_j + 0.20 * d_j + 0.20 * c_{j -> i}, 0.0, 1.0)
        """
        src_hist = self.agent_history[src]
        corr_hist = self.edge_history[(src, tgt)]

        if not src_hist or not corr_hist:
            return float(self.warmup_value)

        # Average over recent 3 steps
        k = min(3, len(src_hist))
        mean_q = float(np.mean([h["queue"] for h in src_hist[-k:]]))
        mean_w = float(np.mean([h["wait"] for h in src_hist[-k:]]))
        mean_d = float(np.mean([h["demand"] for h in src_hist[-k:]]))
        mean_c = float(np.mean(corr_hist[-k:]))

        score = 0.35 * mean_q + 0.25 * mean_w + 0.20 * mean_d + 0.20 * mean_c
        return float(np.clip(score, 0.0, 1.0))

    def get_score_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Return the 8 allowed edge C_ij scores as a [1, 8, 1] or [8] tensor for PyTorch modules.
        """
        score_list = [self.last_scores.get(e, self.warmup_value) for e in self.allowed_edges]
        return torch.tensor(score_list, dtype=torch.float32, device=device).view(1, 8, 1)

    def get_context_dict(self) -> Dict[Tuple[str, str], float]:
        """
        Return copy of current directional C_ij scores mapped by (src, tgt).
        """
        return self.last_scores.copy()
