"""Track fidelity metadata layer (PR 8.4.1).

Separates synthetic families, public approximate abstractions, inferred
geometry, calibrated models, and high-fidelity digital recreations.
Provides provenance and anti-overclaiming guardrails for all track-conditioned
regulatory stress-test findings.

This package is pure metadata — it does NOT modify runtime physics,
SafetyOracle thresholds, or LegalVerdict semantics.
"""

from reglabsim.tracks.fidelity import (
    FIDELITY_TIER_CALIBRATED,
    FIDELITY_TIER_HIGH_FIDELITY,
    FIDELITY_TIER_INFERRED,
    FIDELITY_TIER_PUBLIC_APPROX,
    FIDELITY_TIER_SYNTHETIC,
    TRACK_FIDELITY_SCHEMA,
    TRACK_FIDELITY_TIERS,
    TRACK_MODEL_SCHEMA,
    TRACK_SEGMENT_SCHEMA,
    TrackFidelityReport,
    build_track_fidelity_report,
    compact_track_fidelity_summary,
)
from reglabsim.tracks.track_model import (
    TrackModel,
    TrackSegmentModel,
    build_public_approx_track_model,
    build_track_model_from_synthetic_family,
    compute_segment_risk_features,
    track_model_from_dict,
    track_model_to_dict,
    track_segment_to_dict,
)

__all__ = [
    "FIDELITY_TIER_CALIBRATED",
    "FIDELITY_TIER_HIGH_FIDELITY",
    "FIDELITY_TIER_INFERRED",
    "FIDELITY_TIER_PUBLIC_APPROX",
    "FIDELITY_TIER_SYNTHETIC",
    "TRACK_FIDELITY_SCHEMA",
    "TRACK_FIDELITY_TIERS",
    "TRACK_MODEL_SCHEMA",
    "TRACK_SEGMENT_SCHEMA",
    "TrackFidelityReport",
    "TrackModel",
    "TrackSegmentModel",
    "build_public_approx_track_model",
    "build_track_fidelity_report",
    "build_track_model_from_synthetic_family",
    "compact_track_fidelity_summary",
    "compute_segment_risk_features",
    "track_model_from_dict",
    "track_model_to_dict",
    "track_segment_to_dict",
]
