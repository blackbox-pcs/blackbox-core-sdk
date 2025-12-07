"""
Black Box Precision Core SDK

Unlocking High-Stakes Performance with Explainable AI
"""

__version__ = "1.0.0"
__author__ = "The XAI Lab"

from .core import BlackBoxPrecision, ExplanationType, ExplanationMode
from .explainers import SHAPExplainer, LIMEExplainer
from .utils import (
    validate_explanation,
    aggregate_explanations,
    format_explanation_for_audit,
    compare_explanations,
    extract_key_features,
)

__all__ = [
    "BlackBoxPrecision",
    "SHAPExplainer",
    "LIMEExplainer",
    "ExplanationType",
    "ExplanationMode",
    "validate_explanation",
    "aggregate_explanations",
    "format_explanation_for_audit",
    "compare_explanations",
    "extract_key_features",
]

