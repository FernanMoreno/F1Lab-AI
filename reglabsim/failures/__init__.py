"""Failure classification and mitigation layer."""

from reglabsim.failures.classifier import FailureClassifier
from reglabsim.failures.mitigation import MitigationEngine
from reglabsim.failures.taxonomy import FAILURE_TYPES, SEVERITY_ORDER

__all__ = [
    "FAILURE_TYPES",
    "SEVERITY_ORDER",
    "FailureClassifier",
    "MitigationEngine",
]
