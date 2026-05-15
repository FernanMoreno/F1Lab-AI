"""Track fidelity tiers and audit reporting (PR 8.4.1).

Defines five fidelity tiers (T0-T4), fidelity report generation,
and anti-overclaiming guardrails for track-conditioned findings.

This module is pure metadata — it does NOT affect runtime physics,
SafetyOracle thresholds, LegalVerdict semantics, or exploit_score formulas.
No real track names are hardcoded. No geometry blobs are stored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TRACK_FIDELITY_SCHEMA = "track_fidelity.v0"
TRACK_MODEL_SCHEMA = "track_model.v0"
TRACK_SEGMENT_SCHEMA = "track_segment.v0"

# ---------------------------------------------------------------------------
# Fidelity tiers
# ---------------------------------------------------------------------------

FIDELITY_TIER_SYNTHETIC = "T0_synthetic_family"
FIDELITY_TIER_PUBLIC_APPROX = "T1_public_approximation"
FIDELITY_TIER_INFERRED = "T2_inferred_geometry"
FIDELITY_TIER_CALIBRATED = "T3_calibrated_model"
FIDELITY_TIER_HIGH_FIDELITY = "T4_high_fidelity_digital_recreation"

TRACK_FIDELITY_TIERS: list[str] = [
    FIDELITY_TIER_SYNTHETIC,
    FIDELITY_TIER_PUBLIC_APPROX,
    FIDELITY_TIER_INFERRED,
    FIDELITY_TIER_CALIBRATED,
    FIDELITY_TIER_HIGH_FIDELITY,
]

_FIDELITY_TIER_DESCRIPTIONS: dict[str, str] = {
    FIDELITY_TIER_SYNTHETIC: (
        "Synthetic geometry family. No real-circuit attribution. "
        "Properties are parameterized stress-test abstractions."
    ),
    FIDELITY_TIER_PUBLIC_APPROX: (
        "Approximate circuit abstraction based on public metadata. "
        "May include length, turn count, named segments, rough widths/runoff. "
        "Not a digital twin."
    ),
    FIDELITY_TIER_INFERRED: (
        "Geometry inferred from public maps, imagery, onboard videos, timing sectors, "
        "or other non-proprietary evidence. Includes uncertainty estimates."
    ),
    FIDELITY_TIER_CALIBRATED: (
        "Track abstraction calibrated against repeatable public race/lap behavior. "
        "Still not necessarily laser-scanned."
    ),
    FIDELITY_TIER_HIGH_FIDELITY: (
        "Requires high-resolution geometry, elevation/camber, barriers/runoff/sightlines, "
        "and documented source/provenance. Do not use unless source evidence is available."
    ),
}

# Claim levels per tier
_CLAIM_LEVELS: dict[str, str] = {
    FIDELITY_TIER_SYNTHETIC: "synthetic_stress_test_only",
    FIDELITY_TIER_PUBLIC_APPROX: "track_conditioned_public_approximation",
    FIDELITY_TIER_INFERRED: "track_conditioned_inferred_model",
    FIDELITY_TIER_CALIBRATED: "calibrated_track_conditioned_model",
    FIDELITY_TIER_HIGH_FIDELITY: "high_fidelity_digital_recreation",
}

# Fields relevant to risk analysis
_RISK_RELEVANT_FIELDS = [
    "width_m",
    "barrier_distance_m",
    "runoff_type",
    "runoff_risk",
    "curvature_radius_m",
    "sightline_distance_m",
    "elevation_delta_m",
    "camber_deg",
    "drs_zone",
    "overtaking_zone",
]

# Overclaiming phrases to never use in outputs
OVERCLAIMING_PHRASES = [
    "precise digital twin",
    "laser-scanned",
    "exact recreation",
    "proven real-world exploit",
    "real f1 proof",
    "guaranteed unsafe",
]

# ---------------------------------------------------------------------------
# Fidelity report dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackFidelityReport:
    """Fidelity audit report for one track model."""

    schema_version: str
    track_id: str
    fidelity_tier: str
    data_classification: str
    segment_count: int
    coverage: dict[str, float]
    missing_fields: list[str]
    uncertainty_summary: dict[str, float]
    risk_relevant_fields_available: list[str]
    risk_relevant_fields_missing: list[str]
    claim_level: str
    limitations: list[str]


# ---------------------------------------------------------------------------
# Fidelity report builder
# ---------------------------------------------------------------------------

def build_track_fidelity_report(track: Any) -> dict[str, Any]:
    """Build a compact track fidelity report from a TrackModel.

    Args:
        track: TrackModel instance (from reglabsim.tracks.track_model).

    Returns:
        JSON-serializable dict with schema_version, tier, coverage,
        known_gaps, claim_level, and limitations.
    """
    segments = list(track.segments)
    seg_count = len(segments)

    # Compute coverage for each risk-relevant field
    coverage: dict[str, float] = {}
    missing_fields: list[str] = []
    available_fields: list[str] = []
    uncertainty_summary: dict[str, float] = {}

    for fname in _RISK_RELEVANT_FIELDS:
        if fname in ("drs_zone", "overtaking_zone"):
            # Booleans always have a known value (True/False)
            coverage[fname] = 1.0
            available_fields.append(fname)
        elif seg_count == 0:
            coverage[fname] = 0.0
            missing_fields.append(fname)
        else:
            known = sum(
                1 for s in segments
                if getattr(s, fname, None) is not None
            )
            frac = known / seg_count
            coverage[fname] = round(frac, 4)
            if frac >= 1.0:
                available_fields.append(fname)
            elif frac < 1.0:
                missing_fields.append(fname)

    # Uncertainty: average uncertainty values across segments
    all_uncertainties: dict[str, list[float]] = {}
    for s in segments:
        for k, v in (getattr(s, "uncertainty", {}) or {}).items():
            if isinstance(v, (int, float)):
                all_uncertainties.setdefault(k, []).append(float(v))
    for k, vals in all_uncertainties.items():
        uncertainty_summary[k] = round(sum(vals) / len(vals), 4)

    tier = str(track.fidelity_tier)
    claim_level = _CLAIM_LEVELS.get(tier, "unknown_tier")

    # Build limitations
    limitations = _build_tier_limitations(tier, missing_fields)

    # Known gaps = missing risk-relevant fields
    known_gaps = [f"missing_{f}" for f in missing_fields]

    return {
        "schema_version": TRACK_FIDELITY_SCHEMA,
        "track_id": str(track.track_id),
        "fidelity_tier": tier,
        "data_classification": str(track.data_classification),
        "segment_count": seg_count,
        "coverage": coverage,
        "missing_fields": missing_fields,
        "known_gaps": known_gaps,
        "uncertainty_summary": uncertainty_summary,
        "risk_relevant_fields_available": available_fields,
        "risk_relevant_fields_missing": [
            f for f in missing_fields if f not in ("drs_zone", "overtaking_zone")
        ],
        "claim_level": claim_level,
        "limitations": limitations,
    }


def _build_tier_limitations(tier: str, missing_fields: list[str]) -> list[str]:
    """Return tier-appropriate limitation strings."""
    lims: list[str] = []

    if tier == FIDELITY_TIER_SYNTHETIC:
        lims.append("Track model is not a laser-scanned digital twin.")
        lims.append("Findings are conditioned on available segment abstractions.")
        lims.append("Do not attribute synthetic-family findings to a real circuit.")
    elif tier == FIDELITY_TIER_PUBLIC_APPROX:
        lims.append("Track model is a public approximate abstraction, not a digital twin.")
        lims.append("Segment geometry is approximate and may differ from actual circuit.")
        lims.append("Findings are track-conditioned on available public metadata.")
    elif tier == FIDELITY_TIER_INFERRED:
        lims.append("Track model is inferred from non-proprietary public sources.")
        lims.append("Geometry includes uncertainty estimates.")
        lims.append("Findings are track-conditioned on inferred model accuracy.")
    elif tier == FIDELITY_TIER_CALIBRATED:
        lims.append("Track model is calibrated against public race/lap behavior.")
        lims.append("Calibration does not guarantee geometric precision.")
    elif tier == FIDELITY_TIER_HIGH_FIDELITY:
        lims.append(
            "High-fidelity model; verify source provenance before regulatory claims."
        )

    if missing_fields:
        field_str = ", ".join(f for f in missing_fields
                              if f not in ("drs_zone", "overtaking_zone"))
        if field_str:
            lims.append(f"Missing segment data limits precision: {field_str}.")

    return lims


# ---------------------------------------------------------------------------
# Compact summary helper
# ---------------------------------------------------------------------------

def compact_track_fidelity_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal compact version for tool/trace outputs.

    Does not include full segment arrays or uncertainty detail.
    """
    return {
        "schema_version": report.get("schema_version", TRACK_FIDELITY_SCHEMA),
        "track_id": report.get("track_id"),
        "fidelity_tier": report.get("fidelity_tier"),
        "data_classification": report.get("data_classification"),
        "claim_level": report.get("claim_level"),
        "segment_count": report.get("segment_count", 0),
        "coverage": report.get("coverage", {}),
        "known_gaps": (report.get("known_gaps") or [])[:8],
        "limitations": (report.get("limitations") or [])[:4],
    }
