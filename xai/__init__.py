"""
SAGE-Traffic Explainable AI (XAI) Package
=========================================
Lightweight, data-grounded Explainable AI layer for Multi-Agent Traffic Signal Control.

Modules:
- decision_explainer: Action probabilities, margins, and grounded decision rationales.
- attention_explainer: GAT spatial attention weights analysis and epistemological guards.
- communication_explainer: Gate activations, bandwidth savings, and message suppression.
- causal_explainer: Phase 8R counterfactual causal influence scores and asymmetry.
- safety_explainer: Formal safety invariant clamping and override justifications.
- xai_logger: Non-invasive telemetry logging and structured export.
"""

from .decision_explainer import DecisionExplainer, DecisionExplanation
from .attention_explainer import AttentionExplainer, AttentionExplanation
from .communication_explainer import CommunicationExplainer, CommunicationExplanation
from .causal_explainer import CausalExplainer, CausalExplanation
from .safety_explainer import SafetyExplainer, SafetyExplanation
from .xai_logger import XAILogger

__all__ = [
    "DecisionExplainer",
    "DecisionExplanation",
    "AttentionExplainer",
    "AttentionExplanation",
    "CommunicationExplainer",
    "CommunicationExplanation",
    "CausalExplainer",
    "CausalExplanation",
    "SafetyExplainer",
    "SafetyExplanation",
    "XAILogger"
]

