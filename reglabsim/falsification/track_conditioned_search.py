"""Track-conditioned falsification campaign engine (PR 8.4.2).

Uses track segment abstractions to generate and validate deterministic
falsification candidates conditioned on segment properties (width, barrier
distance, runoff type, etc.).

Architecture:
    TrackModel -> fidelity report -> segment risk scoring
    -> segment-conditioned candidate generation
    -> run_candidate(...) [deterministic runtime]
    -> compact segment findings
    -> track-conditioned campaign report

Invariants:
- Segment risk score alone is NOT evidence.
- Only runtime-validated candidates count as evidence.
- SafetyOracle / LegalVerdict remain source of truth.
- Findings are always conditioned on declared fidelity tier.
- No LLM, no NVIDIA, no external services, no API keys.
- No raw geometry blobs, coordinate arrays, or full bundles in outputs.
- Fully deterministic: same seed + track + config -> same results.
- JSON-serializable outputs.
- No if-track_id style special-casing.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from reglabsim.falsification.search import FalsificationCandidate, run_candidate
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
from reglabsim.tracks.fidelity import (
    build_track_fidelity_report,
    compact_track_fidelity_summary,
)
from reglabsim.tracks.track_model import (
    TrackModel,
    TrackSegmentModel,
    compute_segment_risk_features,
    track_model_from_dict,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACK_CONDITIONED_SEARCH_SCHEMA = "track_conditioned_search.v0"
TRACK_CONDITIONED_SEGMENT_FINDING_SCHEMA = "track_conditioned_segment_finding.v0"
TRACK_CONDITIONED_READINESS_SCHEMA = "track_conditioned_readiness.v0"

_READINESS_LEVELS = ("insufficient", "partial", "ready")

_CRITICAL_FIELDS = ["width_m", "barrier_distance_m", "runoff_type", "segment_type"]

_DEFAULT_LIMITS = {
    "width_m": (9.0, 14.0),
    "barrier_distance_m": (4.0, 16.0),
    "unsafe_closing_speed_threshold_kph": (30.0, 55.0),
    "visibility_m": (500.0, 1200.0),
    "wetness_level": (0.0, 0.35),
    "attacker_risk_level": (0.55, 0.95),
    "defender_risk_level": (0.55, 0.90),
    "attacker_ers_soc": (0.45, 0.95),
    "defender_ers_soc": (0.15, 0.65),
    "gap_s": (0.15, 0.65),
}

_CAMPAIGN_LIMITATIONS = [
    "Track-conditioned findings are limited by declared track fidelity.",
    "Only runtime-validated candidates count as evidence.",
    (
        "This is not proof of a real-world circuit defect unless supported by "
        "a calibrated/high-fidelity track model."
    ),
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VALID_TARGET_LABELS = frozenset({
    "exploit_score_total",
    "legacy_score",
    "unsafe_legal_state_count",
    "max_hazard_score",
})

TRACK_CONDITIONED_GUIDANCE_COMPARISON_SCHEMA = (
    "track_conditioned_guidance_comparison.v0"
)


@dataclass(frozen=True)
class TrackConditionedSearchConfig:
    """Configuration for track-conditioned falsification campaign."""

    seed: int = 42
    max_segments: int = 8
    candidates_per_segment: int = 6
    require_min_readiness: str = "partial"
    include_low_readiness_segments: bool = False
    target_label: str = "exploit_score_total"
    use_surrogate_guidance: bool = False
    surrogate_training_trials: int = 0
    surrogate_model_type: str = "nearest_neighbor"
    surrogate_proposal_multiplier: int = 4
    compare_against_heuristic: bool = True

    def __post_init__(self) -> None:
        if self.max_segments <= 0:
            raise ValueError(f"max_segments must be > 0, got {self.max_segments}")
        if self.max_segments > 20:
            raise ValueError(f"max_segments must be <= 20, got {self.max_segments}")
        if self.candidates_per_segment <= 0:
            raise ValueError(
                f"candidates_per_segment must be > 0, got {self.candidates_per_segment}"
            )
        if self.candidates_per_segment > 25:
            raise ValueError(
                f"candidates_per_segment must be <= 25, got {self.candidates_per_segment}"
            )
        if self.surrogate_training_trials < 0:
            raise ValueError(
                f"surrogate_training_trials must be >= 0, got {self.surrogate_training_trials}"
            )
        if self.surrogate_training_trials > 100:
            raise ValueError(
                f"surrogate_training_trials must be <= 100, got {self.surrogate_training_trials}"
            )
        if self.surrogate_proposal_multiplier <= 0:
            raise ValueError(
                f"surrogate_proposal_multiplier must be > 0, "
                f"got {self.surrogate_proposal_multiplier}"
            )
        if self.surrogate_proposal_multiplier > 10:
            raise ValueError(
                f"surrogate_proposal_multiplier must be <= 10, "
                f"got {self.surrogate_proposal_multiplier}"
            )
        if self.require_min_readiness not in _READINESS_LEVELS:
            raise ValueError(
                f"require_min_readiness must be one of {_READINESS_LEVELS}, "
                f"got {self.require_min_readiness!r}"
            )
        if self.target_label not in _VALID_TARGET_LABELS:
            raise ValueError(
                f"target_label {self.target_label!r} not recognized."
            )
        from reglabsim.falsification.surrogate_models import SUPPORTED_MODEL_TYPES
        if self.surrogate_model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"surrogate_model_type {self.surrogate_model_type!r} not recognized. "
                f"Choose from {SUPPORTED_MODEL_TYPES}."
            )


def validate_track_conditioned_config(config: TrackConditionedSearchConfig) -> None:
    """Validate config — raises ValueError on invalid values.
    All validation is done in __post_init__; this is a public hook.
    """
    pass


# ---------------------------------------------------------------------------
# Readiness assessment
# ---------------------------------------------------------------------------

def assess_track_conditioned_readiness(track: TrackModel) -> dict[str, Any]:
    """Assess whether the TrackModel is ready for segment-conditioned campaigns.

    Returns a readiness report with critical field coverage and limitations.
    Does not require T4 fidelity — T0 synthetic families can be ready
    for synthetic stress tests.
    """
    segments = list(track.segments)
    seg_count = len(segments)

    fidelity_report = build_track_fidelity_report(track)
    coverage = fidelity_report.get("coverage", {})
    known_gaps = fidelity_report.get("known_gaps", [])

    # Critical field availability
    critical_available: list[str] = []
    critical_missing: list[str] = []

    for field in _CRITICAL_FIELDS:
        if field == "segment_type":
            # segment_type is always present as a string (possibly "unknown")
            usable = any(
                getattr(s, "segment_type", "unknown") != "unknown"
                for s in segments
            )
            if usable:
                critical_available.append(field)
            else:
                critical_missing.append(field)
        else:
            cov = coverage.get(field, 0.0)
            if cov > 0.0:
                critical_available.append(field)
            else:
                critical_missing.append(field)

    # Count usable segments: have at least width_m or barrier_distance_m
    usable_count = sum(
        1 for s in segments
        if s.width_m is not None or s.barrier_distance_m is not None
    )

    # Determine readiness
    if seg_count == 0 or usable_count == 0:
        readiness = "insufficient"
    elif (
        "width_m" not in critical_missing
        or "barrier_distance_m" not in critical_missing
    ) and seg_count > 0:
        readiness = "ready"
    else:
        readiness = "partial"

    tier = str(track.fidelity_tier)
    claim_level = fidelity_report.get("claim_level", "unknown")

    lims = [
        f"Readiness is conditioned on {tier} fidelity.",
        "Missing critical fields reduce candidate precision.",
    ]
    if critical_missing:
        lims.append(
            f"Critical fields missing: {', '.join(critical_missing)}."
        )

    return {
        "schema_version": TRACK_CONDITIONED_READINESS_SCHEMA,
        "track_id": str(track.track_id),
        "fidelity_tier": tier,
        "claim_level": claim_level,
        "readiness": readiness,
        "critical_fields_available": critical_available,
        "critical_fields_missing": critical_missing,
        "segment_count": seg_count,
        "usable_segment_count": usable_count,
        "coverage": coverage,
        "known_gaps": known_gaps,
        "limitations": lims,
    }


# ---------------------------------------------------------------------------
# Segment risk scoring
# ---------------------------------------------------------------------------

def score_track_segment_for_falsification(segment: TrackSegmentModel) -> dict[str, Any]:
    """Score one segment for falsification priority.

    Uses only segment properties — no track ID, no real circuit special-casing.
    Missing fields contribute a small uncertainty penalty, not fake precision.
    """
    features = compute_segment_risk_features(segment)

    narrowness = features.get("narrowness", 0.0)
    barrier_pressure = features.get("barrier_pressure", 0.0)
    runoff_surface_risk = features.get("runoff_surface_risk", 0.0)
    curvature_pressure = features.get("curvature_pressure", 0.0)
    sightline_pressure = features.get("sightline_pressure", 0.0)
    drs_or_ot = max(features.get("drs_zone", 0.0), features.get("overtaking_zone", 0.0))
    elev_penalty = features.get("elevation_unknown_penalty", 0.0)
    camber_penalty = features.get("camber_unknown_penalty", 0.0)
    unknown_penalty = 0.5 * (elev_penalty + camber_penalty)

    raw_score = (
        0.25 * narrowness
        + 0.20 * barrier_pressure
        + 0.20 * runoff_surface_risk
        + 0.15 * curvature_pressure
        + 0.10 * sightline_pressure
        + 0.05 * drs_or_ot
        + 0.05 * unknown_penalty
    )
    risk_score = max(0.0, min(1.0, raw_score))

    # Build reason codes
    reason_codes: list[str] = []
    if narrowness >= 0.5:
        reason_codes.append("narrow_segment")
    if barrier_pressure >= 0.5:
        reason_codes.append("barrier_close")
    if runoff_surface_risk >= 0.5:
        reason_codes.append("risky_runoff_surface")
    if curvature_pressure >= 0.5:
        reason_codes.append("tight_curvature")
    if sightline_pressure >= 0.5:
        reason_codes.append("limited_sightline")
    if drs_or_ot >= 1.0:
        reason_codes.append("overtaking_zone_or_drs")
    if not reason_codes:
        reason_codes.append("moderate_risk_profile")

    # Missing fields
    missing_fields: list[str] = []
    for fname, attr in (
        ("curvature_radius_m", segment.curvature_radius_m),
        ("elevation_delta_m", segment.elevation_delta_m),
        ("camber_deg", segment.camber_deg),
        ("sightline_distance_m", segment.sightline_distance_m),
        ("runoff_risk", segment.runoff_risk),
    ):
        if attr is None:
            missing_fields.append(fname)

    return {
        "segment_id": segment.segment_id,
        "segment_type": segment.segment_type,
        "risk_score": round(risk_score, 4),
        "risk_components": {
            "narrowness": round(narrowness, 4),
            "barrier_pressure": round(barrier_pressure, 4),
            "runoff_surface_risk": round(runoff_surface_risk, 4),
            "curvature_pressure": round(curvature_pressure, 4),
            "sightline_pressure": round(sightline_pressure, 4),
        },
        "reason_codes": reason_codes,
        "missing_fields": missing_fields,
    }


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def select_track_segments_for_falsification(
    track: TrackModel,
    max_segments: int = 8,
    include_low_readiness_segments: bool = False,
) -> list[dict[str, Any]]:
    """Select and rank segments by falsification priority.

    Filters out completely unusable segments unless include_low_readiness_segments=True.
    Returns compact segment plans, not full TrackSegmentModel objects.
    """
    scored: list[tuple[float, str, dict[str, Any]]] = []

    for seg in track.segments:
        risk_info = score_track_segment_for_falsification(seg)
        risk_score = risk_info["risk_score"]

        # Check usability: must have at least width or barrier
        has_useful_data = (
            seg.width_m is not None or seg.barrier_distance_m is not None
        )
        if not has_useful_data and not include_low_readiness_segments:
            continue

        plan: dict[str, Any] = {
            "segment_id": seg.segment_id,
            "segment_type": seg.segment_type,
            "risk_score": risk_score,
            "reason_codes": risk_info["reason_codes"],
            "missing_fields": risk_info["missing_fields"],
            "width_m": seg.width_m,
            "barrier_distance_m": seg.barrier_distance_m,
            "runoff_type": seg.runoff_type,
            "runoff_risk": seg.runoff_risk,
            "sightline_distance_m": seg.sightline_distance_m,
            "drs_zone": seg.drs_zone,
            "overtaking_zone": seg.overtaking_zone,
        }
        scored.append((risk_score, seg.segment_id, plan))

    # Sort by risk_score desc, segment_id asc
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [plan for _, _, plan in scored[:max_segments]]


# ---------------------------------------------------------------------------
# Stable segment hash for seeding
# ---------------------------------------------------------------------------

def _stable_segment_hash(segment_id: str) -> int:
    """Deterministic integer hash from segment_id string."""
    digest = hashlib.sha256(segment_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Family selector
# ---------------------------------------------------------------------------

def _select_family_for_segment(segment: TrackSegmentModel) -> str:
    """Select the most appropriate synthetic family for a segment.

    Maps segment properties to the closest synthetic family without
    hardcoding real track names or segment IDs.
    """
    seg_type = str(segment.segment_type or "unknown").lower()
    runoff = str(segment.runoff_type or "unknown").lower()
    width = segment.width_m or 12.0

    # Priority: tight/narrow + risky runoff = most constrained family
    is_narrow = width < 10.5
    is_wall_runoff = runoff in ("wall", "barrier")
    is_grass_runoff = runoff in ("grass", "gravel")
    is_chicane = "chicane" in seg_type
    is_fast = "fast" in seg_type
    is_low_vis = (
        segment.sightline_distance_m is not None
        and segment.sightline_distance_m < 700.0
    )

    if is_chicane and is_narrow:
        return "narrow_street_chicane"
    if is_fast and is_wall_runoff:
        return "fast_corner_wall"
    if is_narrow and is_grass_runoff:
        return "confined_corner_grass"
    if is_low_vis:
        return "high_speed_entry_low_visibility"
    if is_fast:
        return "fast_corner_wall"

    # Default: most general high-risk family
    return "confined_corner_grass"


# ---------------------------------------------------------------------------
# Segment-conditioned parameter generation
# ---------------------------------------------------------------------------

def build_segment_conditioned_parameters(
    *,
    segment: TrackSegmentModel,
    segment_risk: dict[str, Any],
    seed: int,
    count: int,
) -> list[dict[str, float]]:
    """Generate deterministic candidate parameters conditioned on a segment.

    Uses segment geometry where available; falls back to conservative
    defaults for missing fields. Never invents real-world precision.

    Args:
        segment: TrackSegmentModel with available properties.
        segment_risk: Output from score_track_segment_for_falsification.
        seed: Base seed; offset by stable_segment_hash for determinism.
        count: Number of parameter sets to generate.

    Returns:
        List of parameter dicts compatible with FalsificationCandidate.
    """
    seg_hash = _stable_segment_hash(segment.segment_id)
    rng = random.Random(seed + seg_hash)

    risk_score = float(segment_risk.get("risk_score", 0.5))

    # Base values from segment geometry (or defaults if missing)
    base_width = segment.width_m if segment.width_m is not None else 11.5
    base_barrier = segment.barrier_distance_m if segment.barrier_distance_m is not None else 8.0

    # Closing speed: tighter for high-risk segments
    # Lower threshold = more likely to trigger SafetyOracle
    base_closing_speed = 30.0 + (1.0 - risk_score) * 25.0

    # Visibility: use sightline if known, else moderate default
    base_visibility = (
        segment.sightline_distance_m
        if segment.sightline_distance_m is not None
        else 900.0
    )

    result: list[dict[str, float]] = []
    for _idx in range(count):
        # Width: vary around segment value, slightly below for stress
        width = rng.uniform(
            max(_DEFAULT_LIMITS["width_m"][0], base_width * 0.85),
            min(_DEFAULT_LIMITS["width_m"][1], base_width * 1.05),
        )
        # Barrier: vary around segment value, tighter for stress
        barrier = rng.uniform(
            max(_DEFAULT_LIMITS["barrier_distance_m"][0], base_barrier * 0.7),
            min(_DEFAULT_LIMITS["barrier_distance_m"][1], base_barrier * 1.1),
        )
        closing_speed = rng.uniform(
            _DEFAULT_LIMITS["unsafe_closing_speed_threshold_kph"][0],
            min(
                _DEFAULT_LIMITS["unsafe_closing_speed_threshold_kph"][1],
                base_closing_speed + 10.0,
            ),
        )
        visibility = rng.uniform(
            max(_DEFAULT_LIMITS["visibility_m"][0], base_visibility * 0.8),
            min(_DEFAULT_LIMITS["visibility_m"][1], base_visibility * 1.1),
        )
        wetness = rng.uniform(0.0, 0.25)
        attacker_risk = rng.uniform(0.60, 0.95)
        defender_risk = rng.uniform(0.55, 0.85)
        attacker_ers = rng.uniform(0.50, 0.95)
        defender_ers = rng.uniform(0.15, 0.60)
        gap_s = rng.uniform(0.15, 0.55)

        result.append({
            "width_m": round(width, 4),
            "barrier_distance_m": round(barrier, 4),
            "unsafe_closing_speed_threshold_kph": round(closing_speed, 4),
            "visibility_m": round(visibility, 4),
            "wetness_level": round(wetness, 4),
            "attacker_risk_level": round(attacker_risk, 4),
            "defender_risk_level": round(defender_risk, 4),
            "attacker_ers_soc": round(attacker_ers, 4),
            "defender_ers_soc": round(defender_ers, 4),
            "gap_s": round(gap_s, 4),
        })

    return result


# ---------------------------------------------------------------------------
# Build FalsificationCandidate objects
# ---------------------------------------------------------------------------

def build_segment_conditioned_candidates(
    *,
    track: TrackModel,
    segment_plan: dict[str, Any],
    parameters_list: list[dict[str, float]],
    seed: int,
) -> list[FalsificationCandidate]:
    """Build FalsificationCandidate objects for one segment.

    Maps to the closest appropriate synthetic family.
    Candidate IDs include track_id and segment_id for traceability.
    """
    segment_id = str(segment_plan.get("segment_id") or "unknown")

    # Build a minimal segment to select family
    seg = TrackSegmentModel(
        segment_id=segment_id,
        name=segment_plan.get("name"),
        segment_type=str(segment_plan.get("segment_type") or "unknown"),
        width_m=segment_plan.get("width_m"),
        barrier_distance_m=segment_plan.get("barrier_distance_m"),
        runoff_type=segment_plan.get("runoff_type"),
        sightline_distance_m=segment_plan.get("sightline_distance_m"),
    )
    family_id = _select_family_for_segment(seg)
    # Ensure family exists
    if family_id not in SYNTHETIC_FAMILIES:
        family_id = "confined_corner_grass"

    track_id = str(track.track_id)
    candidates: list[FalsificationCandidate] = []
    for idx, params in enumerate(parameters_list):
        cid = f"{track_id}:segment:{segment_id}:seed{seed}:trial{idx:04d}"
        candidates.append(
            FalsificationCandidate(
                candidate_id=cid,
                family_id=family_id,
                seed=seed,
                parameters=params,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Compact result helper
# ---------------------------------------------------------------------------

def compact_track_conditioned_candidate_result(
    result: Any,
    *,
    segment_id: str,
    segment_risk: dict[str, Any],
    track_fidelity: dict[str, Any],
) -> dict[str, Any]:
    """Extract compact, safe metadata from a FalsificationResult.

    Excludes raw bundles, event logs, state snapshots, and full geometry.
    """
    tier = track_fidelity.get("fidelity_tier", "unknown")
    claim_level = track_fidelity.get("claim_level", "unknown")

    exploit_total = 0.0
    es = result.exploit_score
    if isinstance(es, dict):
        exploit_total = float(es.get("total") or 0.0)

    failure_modes: list[str] = []
    primary_fm: str | None = None
    ft = result.failure_taxonomy
    if isinstance(ft, dict):
        primary_fm = ft.get("primary_failure_mode")
        for fm in ft.get("failure_modes") or []:
            if isinstance(fm, dict):
                m = fm.get("mode")
                if isinstance(m, str) and m:
                    failure_modes.append(m)

    return {
        "candidate_id": result.candidate_id,
        "segment_id": segment_id,
        "score": round(result.score, 6),
        "score_legacy": round(result.score, 6),
        "exploit_score_total": round(exploit_total, 6),
        "unsafe_legal_state_count": result.unsafe_legal_state_count,
        "max_hazard_score": result.max_hazard_score,
        "primary_failure_mode": primary_fm,
        "failure_modes": failure_modes,
        "event_refs": list(result.event_refs),
        "track_fidelity_tier": tier,
        "track_fidelity_claim_level": claim_level,
        "limitations": [
            f"Finding conditioned on {tier} track abstraction.",
        ],
    }


# ---------------------------------------------------------------------------
# Aggregate segment findings
# ---------------------------------------------------------------------------

def _aggregate_segment_findings(
    segment_plan: dict[str, Any],
    compact_results: list[dict[str, Any]],
    track_fidelity: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate candidate results into one compact segment finding."""
    segment_id = str(segment_plan.get("segment_id") or "unknown")
    segment_type = str(segment_plan.get("segment_type") or "unknown")
    risk_score = float(segment_plan.get("risk_score") or 0.0)

    tier = track_fidelity.get("fidelity_tier", "unknown")

    total_unsafe = sum(r.get("unsafe_legal_state_count") or 0 for r in compact_results)
    max_hazard = max(
        (float(r.get("max_hazard_score") or 0.0) for r in compact_results),
        default=0.0,
    )

    # Best candidate by exploit_score_total
    best: dict[str, Any] | None = None
    best_exploit = -1.0
    for r in compact_results:
        s = float(r.get("exploit_score_total") or 0.0)
        if s > best_exploit:
            best_exploit = s
            best = r

    best_cid = best["candidate_id"] if best else None
    best_legacy = float(best["score_legacy"]) if best else 0.0
    best_exploit_total = float(best["exploit_score_total"]) if best else 0.0

    # Aggregate failure modes
    primary_modes: list[str] = []
    all_modes: list[str] = []
    seen_modes: set[str] = set()
    for r in compact_results:
        pfm = r.get("primary_failure_mode")
        if pfm and pfm not in primary_modes:
            primary_modes.append(pfm)
        for fm in r.get("failure_modes") or []:
            if isinstance(fm, str) and fm not in seen_modes:
                seen_modes.add(fm)
                all_modes.append(fm)

    # Event refs
    event_refs: list[str] = []
    seen_refs: set[str] = set()
    for r in compact_results:
        for ref in r.get("event_refs") or []:
            if isinstance(ref, str) and ref not in seen_refs:
                seen_refs.add(ref)
                event_refs.append(ref)

    return {
        "schema_version": TRACK_CONDITIONED_SEGMENT_FINDING_SCHEMA,
        "segment_id": segment_id,
        "segment_type": segment_type,
        "segment_risk_score": round(risk_score, 4),
        "candidate_count": len(compact_results),
        "validated_count": len(compact_results),
        "unsafe_legal_state_count": total_unsafe,
        "best_candidate_id": best_cid,
        "best_actual_legacy_score": round(best_legacy, 6),
        "best_actual_exploit_score_total": round(best_exploit_total, 6),
        "max_hazard_score": round(max_hazard, 6),
        "primary_failure_modes": primary_modes[:4],
        "failure_modes": all_modes[:8],
        "event_refs": event_refs[:8],
        "limitations": [
            f"Finding is conditioned on {tier} track abstraction.",
            "Segment risk score alone is not evidence; runtime validates.",
        ],
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_track_conditioned_falsification(
    track: TrackModel | dict[str, Any],
    config: TrackConditionedSearchConfig | None = None,
) -> dict[str, Any]:
    """Run deterministic falsification campaign conditioned on track segments.

    Generates segment-parameterized candidates, validates them through
    the deterministic runtime, and returns compact segment findings.

    Segment risk score alone is not evidence.
    Only runtime-validated candidates count as evidence.

    Args:
        track: TrackModel instance or dict (will be deserialized).
        config: Search configuration; uses defaults if None.

    Returns:
        Compact JSON-serializable campaign report.
    """
    # Deserialize if dict
    if isinstance(track, dict):
        track = track_model_from_dict(track)

    if config is None:
        config = TrackConditionedSearchConfig()

    # Build fidelity report
    fidelity_report = build_track_fidelity_report(track)
    compact_fidelity = compact_track_fidelity_summary(fidelity_report)

    # Readiness assessment
    readiness_report = assess_track_conditioned_readiness(track)
    readiness = readiness_report["readiness"]

    # Check minimum readiness
    readiness_rank = {r: i for i, r in enumerate(_READINESS_LEVELS)}
    min_rank = readiness_rank.get(config.require_min_readiness, 0)
    actual_rank = readiness_rank.get(readiness, 0)

    if actual_rank < min_rank:
        return {
            "schema_version": TRACK_CONDITIONED_SEARCH_SCHEMA,
            "track_id": str(track.track_id),
            "display_name": str(track.display_name),
            "seed": config.seed,
            "config": _config_to_dict(config),
            "track_fidelity": compact_fidelity,
            "readiness": readiness_report,
            "selected_segments": [],
            "segment_findings": [],
            "best_segment_finding": None,
            "summary": {
                "segments_evaluated": 0,
                "candidates_validated": 0,
                "segments_with_unsafe_legal_state": 0,
                "total_unsafe_legal_states": 0,
                "best_actual_exploit_score_total": None,
            },
            "limitations": [
                f"Campaign aborted: track readiness is {readiness!r}, "
                f"require_min_readiness={config.require_min_readiness!r}.",
                *list(_CAMPAIGN_LIMITATIONS),
            ],
        }

    # Select candidate segments
    selected_segments = select_track_segments_for_falsification(
        track,
        max_segments=config.max_segments,
        include_low_readiness_segments=config.include_low_readiness_segments,
    )

    # Optionally build surrogate model for candidate prioritization
    surrogate_model: Any = None
    surrogate_guidance_meta: dict[str, Any] | None = None
    surrogate_is_active = False  # True only when model is fitted + has training rows

    if config.use_surrogate_guidance:
        surrogate_model, surrogate_guidance_meta = _build_surrogate_for_guidance(
            config=config,
            track_id=str(track.track_id),
        )
        surrogate_is_active = (
            surrogate_model is not None
            and (surrogate_guidance_meta or {}).get("status") == "active"
        )

    # Whether to also run heuristic candidates for comparison
    run_heuristic_comparison = (
        config.use_surrogate_guidance
        and config.compare_against_heuristic
        and surrogate_is_active
    )

    # Run candidates for each segment
    segment_findings: list[dict[str, Any]] = []
    all_compact_results: list[dict[str, Any]] = []
    # Separate accumulators for comparison
    heuristic_findings_for_cmp: list[dict[str, Any]] = []
    surrogate_findings_for_cmp: list[dict[str, Any]] = []

    for seg_plan in selected_segments:
        seg_id = str(seg_plan["segment_id"])
        seg_obj = next(
            (s for s in track.segments if s.segment_id == seg_id), None
        )
        if seg_obj is None:
            continue

        seg_risk = score_track_segment_for_falsification(seg_obj)

        # Always generate base heuristic params
        heuristic_params = build_segment_conditioned_parameters(
            segment=seg_obj,
            segment_risk=seg_risk,
            seed=config.seed,
            count=config.candidates_per_segment,
        )

        if surrogate_is_active:
            # Generate larger pool, rank by surrogate, validate top-k
            pool_count = config.candidates_per_segment * config.surrogate_proposal_multiplier
            params_pool = build_segment_conditioned_parameters(
                segment=seg_obj,
                segment_risk=seg_risk,
                seed=config.seed,
                count=pool_count,
            )
            params_list, surrogate_preds = _surrogate_rank_params(
                surrogate_model=surrogate_model,
                params_pool=params_pool,
                top_k=config.candidates_per_segment,
                family_id=_select_family_for_segment(seg_obj),
                config=config,
            )
        else:
            params_list = heuristic_params
            surrogate_preds = []

        candidates = build_segment_conditioned_candidates(
            track=track,
            segment_plan=seg_plan,
            parameters_list=params_list,
            seed=config.seed,
        )

        seg_compact_results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            candidate_result = run_candidate(candidate, include_bundle=False)
            compact = compact_track_conditioned_candidate_result(
                candidate_result,
                segment_id=seg_id,
                segment_risk=seg_risk,
                track_fidelity=compact_fidelity,
            )
            # Attach surrogate prediction metadata only when predictions are real
            if surrogate_preds and idx < len(surrogate_preds):
                sp = surrogate_preds[idx]
                compact["predicted_score"] = sp.get("prediction")
                compact["prediction_confidence"] = sp.get("confidence")
                actual = float(compact.get("exploit_score_total") or 0.0)
                pred = float(sp.get("prediction") or 0.0)
                compact["absolute_prediction_error"] = round(abs(actual - pred), 6)
                compact["prediction_available"] = True
            else:
                compact["prediction_available"] = False
            seg_compact_results.append(compact)
            all_compact_results.append(compact)

        finding = _aggregate_segment_findings(
            seg_plan, seg_compact_results, compact_fidelity
        )
        # Add surrogate guidance summary to finding (only when active)
        if surrogate_is_active:
            pred_errors = [
                float(r.get("absolute_prediction_error") or 0.0)
                for r in seg_compact_results
                if r.get("prediction_available")
            ]
            pred_best = max(
                (float(r.get("predicted_score") or 0.0) for r in seg_compact_results
                 if r.get("prediction_available")),
                default=None,
            )
            actual_best = float(finding.get("best_actual_exploit_score_total") or 0.0)
            finding = dict(finding)
            if pred_errors:
                finding["mean_absolute_prediction_error"] = round(
                    sum(pred_errors) / len(pred_errors), 6
                )
            if pred_best is not None:
                finding["predicted_best_score"] = round(pred_best, 6)
            finding["actual_best_score"] = round(actual_best, 6)
        segment_findings.append(finding)
        surrogate_findings_for_cmp.append(finding)

        # Run heuristic candidates for comparison
        if run_heuristic_comparison and heuristic_params is not params_list:
            h_candidates = build_segment_conditioned_candidates(
                track=track,
                segment_plan=dict(seg_plan, **{"segment_id": seg_id + ":h"}),
                parameters_list=heuristic_params,
                seed=config.seed + 1,
            )
            h_results: list[dict[str, Any]] = []
            for h_cand in h_candidates:
                h_res = run_candidate(h_cand, include_bundle=False)
                h_compact = compact_track_conditioned_candidate_result(
                    h_res,
                    segment_id=seg_id,
                    segment_risk=seg_risk,
                    track_fidelity=compact_fidelity,
                )
                h_results.append(h_compact)
            h_finding = _aggregate_segment_findings(seg_plan, h_results, compact_fidelity)
            heuristic_findings_for_cmp.append(h_finding)
        elif not run_heuristic_comparison:
            heuristic_findings_for_cmp.append(finding)

    # Rank findings
    segment_findings.sort(
        key=lambda f: (
            -(f.get("unsafe_legal_state_count") or 0),
            -(f.get("best_actual_exploit_score_total") or 0.0),
            -(f.get("max_hazard_score") or 0.0),
            -(f.get("segment_risk_score") or 0.0),
            f.get("segment_id", ""),
        )
    )

    # Best finding
    best_finding = segment_findings[0] if segment_findings else None

    # Summary
    segments_with_unsafe = sum(
        1 for f in segment_findings
        if (f.get("unsafe_legal_state_count") or 0) > 0
    )
    total_unsafe = sum(
        (f.get("unsafe_legal_state_count") or 0)
        for f in segment_findings
    )
    best_exploit: float | None = None
    if segment_findings:
        vals = [
            f.get("best_actual_exploit_score_total") or 0.0
            for f in segment_findings
        ]
        best_exploit = max(vals) if vals else None

    result: dict[str, Any] = {
        "schema_version": TRACK_CONDITIONED_SEARCH_SCHEMA,
        "track_id": str(track.track_id),
        "display_name": str(track.display_name),
        "seed": config.seed,
        "config": _config_to_dict(config),
        "track_fidelity": compact_fidelity,
        "readiness": readiness_report,
        "selected_segments": selected_segments,
        "segment_findings": segment_findings,
        "best_segment_finding": best_finding,
        "summary": {
            "segments_evaluated": len(segment_findings),
            "candidates_validated": len(all_compact_results),
            "segments_with_unsafe_legal_state": segments_with_unsafe,
            "total_unsafe_legal_states": total_unsafe,
            "best_actual_exploit_score_total": (
                round(best_exploit, 6) if best_exploit is not None else None
            ),
        },
        "limitations": list(_CAMPAIGN_LIMITATIONS),
    }

    if surrogate_guidance_meta is not None:
        result["surrogate_guidance"] = surrogate_guidance_meta

    # guidance_comparison
    if config.use_surrogate_guidance and config.compare_against_heuristic:
        if surrogate_is_active and heuristic_findings_for_cmp != surrogate_findings_for_cmp:
            result["guidance_comparison"] = compare_track_conditioned_guidance_modes(
                heuristic_findings=heuristic_findings_for_cmp,
                surrogate_findings=surrogate_findings_for_cmp,
            )
        else:
            result["guidance_comparison"] = {
                "schema_version": TRACK_CONDITIONED_GUIDANCE_COMPARISON_SCHEMA,
                "verdict": "not_run_insufficient_training_data",
                "limitations": [
                    "Comparison was not run because surrogate training data was unavailable."
                ],
            }

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_to_dict(config: TrackConditionedSearchConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "max_segments": config.max_segments,
        "candidates_per_segment": config.candidates_per_segment,
        "require_min_readiness": config.require_min_readiness,
        "include_low_readiness_segments": config.include_low_readiness_segments,
        "target_label": config.target_label,
        "use_surrogate_guidance": config.use_surrogate_guidance,
        "surrogate_training_trials": config.surrogate_training_trials,
        "surrogate_model_type": config.surrogate_model_type,
        "surrogate_proposal_multiplier": config.surrogate_proposal_multiplier,
        "compare_against_heuristic": config.compare_against_heuristic,
    }


# ---------------------------------------------------------------------------
# Surrogate guidance helpers
# ---------------------------------------------------------------------------

def _build_surrogate_for_guidance(
    config: TrackConditionedSearchConfig,
    track_id: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Build and train a surrogate model for candidate prioritization.

    Returns (model_or_None, guidance_meta_dict).

    If surrogate_training_trials <= 0 or no training rows are available:
        Returns (None, meta with status="fallback_to_heuristic_insufficient_training_data").
        The caller must use heuristic ordering when model is None.

    If training succeeds with > 0 rows:
        Returns (fitted_model, meta with status="active").
    """
    from reglabsim.falsification.surrogate import (
        build_surrogate_dataset_from_search_result,
    )
    from reglabsim.falsification.surrogate_models import create_surrogate_model
    from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

    model_type = config.surrogate_model_type

    # --- Explicit fallback when no training data requested ---
    if config.surrogate_training_trials <= 0:
        return None, {
            "enabled": True,
            "status": "fallback_to_heuristic_insufficient_training_data",
            "model_type": model_type,
            "training_trials": config.surrogate_training_trials,
            "training_rows": 0,
            "target_label": config.target_label,
            "used_for": "not_used_insufficient_training_data",
            "limitations": [
                "Surrogate guidance was requested but not used because no training data "
                "was available.",
                "Heuristic ordering was used instead.",
                "Only runtime-validated candidates count as evidence.",
            ],
        }

    # --- Attempt to build training dataset ---
    family_id = next(iter(SYNTHETIC_FAMILIES.keys()))
    try:
        from reglabsim.falsification.search import run_falsification_search
        sr = run_falsification_search(
            family_id=family_id,
            seed=config.seed,
            max_trials=config.surrogate_training_trials,
        )
        dataset = build_surrogate_dataset_from_search_result(sr)
    except Exception:
        dataset = {
            "schema_version": "surrogate_exploit_dataset.v0",
            "family_id": family_id, "seed": config.seed,
            "row_count": 0, "feature_names": [],
            "label_names": [], "rows": [], "limitations": [],
        }

    train_count = int(dataset.get("row_count") or 0)

    # --- If search produced no rows, fallback to heuristic ---
    if train_count == 0:
        return None, {
            "enabled": True,
            "status": "fallback_to_heuristic_insufficient_training_data",
            "model_type": model_type,
            "training_trials": config.surrogate_training_trials,
            "training_rows": 0,
            "target_label": config.target_label,
            "used_for": "not_used_insufficient_training_data",
            "limitations": [
                "Surrogate search returned 0 training rows.",
                "Heuristic ordering was used instead.",
                "Only runtime-validated candidates count as evidence.",
            ],
        }

    # --- Fit surrogate ---
    limitations = [
        "Surrogate predictions are not evidence.",
        "Only runtime-validated candidates count as findings.",
    ]
    try:
        model = create_surrogate_model(
            model_type=model_type,
            target_label=config.target_label,
            random_state=config.seed,
        )
        model.fit(dataset)
    except (RuntimeError, ValueError) as exc:
        from reglabsim.falsification.surrogate import DeterministicNearestNeighborSurrogate
        model = DeterministicNearestNeighborSurrogate(
            target_label=config.target_label
        ).fit(dataset)
        model_type = "nearest_neighbor"
        limitations.append(f"Fell back to nearest_neighbor: {exc}")

    guidance_meta: dict[str, Any] = {
        "enabled": True,
        "status": "active",
        "model_type": model_type,
        "training_trials": config.surrogate_training_trials,
        "training_rows": train_count,
        "target_label": config.target_label,
        "used_for": "candidate_prioritization_only",
        "validated_top_k_per_segment": config.candidates_per_segment,
        "proposal_multiplier": config.surrogate_proposal_multiplier,
        "limitations": limitations,
    }
    return model, guidance_meta


def _surrogate_rank_params(
    *,
    surrogate_model: Any,
    params_pool: list[dict[str, float]],
    top_k: int,
    family_id: str,
    config: TrackConditionedSearchConfig,
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    """Rank params_pool by surrogate prediction; return top_k + their predictions."""
    from reglabsim.falsification.surrogate import extract_candidate_features

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, params in enumerate(params_pool):
        feats = extract_candidate_features(family_id=family_id, parameters=params)
        pred = surrogate_model.predict_one(feats)
        score = float(pred.get("prediction") or 0.0)
        scored.append((score, i, pred))

    scored.sort(key=lambda t: (-t[0], t[1]))
    top_items = scored[:top_k]

    top_params = [params_pool[i] for _, i, _ in top_items]
    top_preds = [pred for _, _, pred in top_items]
    return top_params, top_preds


def compare_track_conditioned_guidance_modes(
    *,
    heuristic_findings: list[dict[str, Any]],
    surrogate_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare heuristic vs surrogate-guided segment findings compactly.

    Reports verdict honestly — surrogate may be worse. Does not
    treat surrogate prediction as evidence.
    """
    def _summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
        total_unsafe = sum(f.get("unsafe_legal_state_count") or 0 for f in findings)
        best_exploit = max(
            (float(f.get("best_actual_exploit_score_total") or 0.0) for f in findings),
            default=0.0,
        )
        mae_vals = [
            float(f["mean_absolute_prediction_error"])
            for f in findings
            if "mean_absolute_prediction_error" in f
        ]
        mae = sum(mae_vals) / len(mae_vals) if mae_vals else None
        return {
            "segments_evaluated": len(findings),
            "total_unsafe_legal_states": total_unsafe,
            "best_actual_exploit_score_total": round(best_exploit, 6),
            **({"mean_absolute_prediction_error": round(mae, 6)} if mae is not None else {}),
        }

    h = _summary(heuristic_findings)
    s = _summary(surrogate_findings)

    h_unsafe = int(h["total_unsafe_legal_states"])
    s_unsafe = int(s["total_unsafe_legal_states"])
    h_exploit = float(h["best_actual_exploit_score_total"])
    s_exploit = float(s["best_actual_exploit_score_total"])

    delta_unsafe = s_unsafe - h_unsafe
    delta_exploit = round(s_exploit - h_exploit, 6)

    if delta_unsafe > 0 or delta_exploit > 0.01:
        verdict = "surrogate_better"
    elif delta_unsafe < 0 or delta_exploit < -0.01:
        verdict = "heuristic_better"
    elif delta_unsafe == 0 and abs(delta_exploit) <= 0.01:
        verdict = "same"
    else:
        verdict = "mixed"

    return {
        "schema_version": TRACK_CONDITIONED_GUIDANCE_COMPARISON_SCHEMA,
        "heuristic": h,
        "surrogate": s,
        "delta": {
            "unsafe_legal_states": delta_unsafe,
            "best_actual_exploit_score_total": delta_exploit,
        },
        "verdict": verdict,
        "limitations": [
            "Comparison is deterministic for this seed and synthetic configuration.",
            "No guarantee of general superiority.",
            "Surrogate guidance is prediction-only; runtime decides evidence.",
        ],
    }
