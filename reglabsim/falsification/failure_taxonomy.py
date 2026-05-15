"""Deterministic failure taxonomy for falsification search (PR 8.2).

Classifies each unsafe/legal candidate into interpretable regulatory-technical
failure types. Evidence-derived from metrics, events, verdicts, candidate
parameters, patch reruns, and exploit_score reason codes.

Invariants:
- Fully deterministic: same inputs -> same output.
- No LLM, no NVIDIA, no external services.
- Does not modify scoring semantics or ranking.
- Labels are diagnostic categories, not calibrated causal proof.
- JSON-serializable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_TAXONOMY_SCHEMA = "failure_taxonomy.v1"

# Mode ID constants
UNSAFE_CLOSING_SPEED = "unsafe_closing_speed"
GREY_AREA_ACTIVE_AERO = "grey_area_active_aero"
PACK_COMPRESSION_EXPLOIT = "pack_compression_exploit"
LOW_VISIBILITY_ATTACK = "low_visibility_attack"
CONFINED_CORNER_ATTACK = "confined_corner_attack"
REACTION_MARGIN_FAILURE = "reaction_margin_failure"
ENERGY_DELTA_EXPLOIT = "energy_delta_exploit"
REJOIN_SURFACE_EXPLOIT = "rejoin_surface_exploit"
PATCH_RESISTANT_EXPLOIT = "patch_resistant_exploit"
HIGH_HAZARD_LEGAL_STATE = "high_hazard_legal_state"
SPIRIT_OF_REGULATION_EXPLOIT = "spirit_of_regulation_exploit"
TECHNICAL_DIRECTIVE_BOUNDARY = "technical_directive_boundary"
UNKNOWN_FAILURE_MODE = "unknown_failure_mode"

FAILURE_MODE_IDS: list[str] = [
    UNSAFE_CLOSING_SPEED,
    GREY_AREA_ACTIVE_AERO,
    PACK_COMPRESSION_EXPLOIT,
    LOW_VISIBILITY_ATTACK,
    CONFINED_CORNER_ATTACK,
    REACTION_MARGIN_FAILURE,
    ENERGY_DELTA_EXPLOIT,
    REJOIN_SURFACE_EXPLOIT,
    PATCH_RESISTANT_EXPLOIT,
    HIGH_HAZARD_LEGAL_STATE,
    SPIRIT_OF_REGULATION_EXPLOIT,
    TECHNICAL_DIRECTIVE_BOUNDARY,
    UNKNOWN_FAILURE_MODE,
]

_CONFIDENCE_RANK: dict[str, int] = {"high": 2, "medium": 1, "low": 0}

_STANDARD_LIMITATIONS = [
    "Failure taxonomy is deterministic and evidence-derived.",
    "Labels are diagnostic categories, not calibrated causal proof.",
]

_RISKY_RUNOFF_SURFACES: frozenset[str] = frozenset(
    {"grass", "gravel", "wall", "barrier", "concrete"}
)

_ACTIVE_AERO_TERMS: frozenset[str] = frozenset(
    {"active_aero_attack_window", "active_aero", "drs", "ers_attack"}
)

_LEGAL_FAMILY_STATUSES: frozenset[str] = frozenset(
    {"LEGAL", "GREY_AREA", "SPIRIT_VIOLATION", "NEEDS_STEWARD_REVIEW",
     "NEEDS_TECHNICAL_DIRECTIVE"}
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureModeEvidence:
    """Evidence record for one detected failure mode."""

    mode: str
    confidence: str  # "low" | "medium" | "high"
    score: float
    reason_codes: list[str]
    event_refs: list[str] = field(default_factory=list)
    supporting_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureTaxonomyResult:
    """Full failure taxonomy result for one candidate."""

    schema_version: str
    primary_failure_mode: str | None
    failure_modes: list[FailureModeEvidence]
    event_refs: list[str]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def failure_taxonomy_to_dict(result: FailureTaxonomyResult) -> dict[str, Any]:
    """Serialize a FailureTaxonomyResult to a JSON-compatible dict."""
    return {
        "schema_version": result.schema_version,
        "primary_failure_mode": result.primary_failure_mode,
        "failure_modes": [
            {
                "mode": fm.mode,
                "confidence": fm.confidence,
                "score": round(fm.score, 6),
                "reason_codes": list(fm.reason_codes),
                "event_refs": list(fm.event_refs),
                "supporting_fields": dict(fm.supporting_fields),
            }
            for fm in result.failure_modes
        ],
        "event_refs": list(result.event_refs),
        "limitations": list(result.limitations),
    }


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def extract_event_details(event: dict[str, Any]) -> dict[str, Any]:
    """Extract details from event, supporting multiple shapes."""
    details = event.get("details")
    if isinstance(details, dict):
        return details
    payload = event.get("payload")
    if isinstance(payload, dict):
        details = payload.get("details")
        if isinstance(details, dict):
            return details
    # Fall back to flat event fields (excluding meta keys)
    _meta = {"event_type", "event_ref", "payload", "details", "timestamp"}
    return {k: v for k, v in event.items() if k not in _meta}


def extract_failure_event_refs(
    metrics: dict[str, Any] | None,
    unsafe_events: list[dict[str, Any]] | None,
) -> list[str]:
    """Extract unique event refs from metrics and unsafe events."""
    seen: set[str] = set()
    result: list[str] = []
    for ref in (metrics or {}).get("unsafe_legal_event_refs") or []:
        if isinstance(ref, str) and ref not in seen:
            seen.add(ref)
            result.append(ref)
    for ev in (unsafe_events or []):
        for key in ("event_ref", "ref"):
            ref = ev.get(key)
            if isinstance(ref, str) and ref not in seen:
                seen.add(ref)
                result.append(ref)
    return result


def normalize_reason_values(values: Any) -> list[str]:
    """Normalize amplifiers/causes/flags to a flat list of strings."""
    if not values:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        result = []
        for v in values:
            if isinstance(v, str) and v:
                result.append(v)
        return result
    return []


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def detect_unsafe_closing_speed(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
    exploit_score: dict[str, Any] | None,
) -> FailureModeEvidence | None:
    """Detect unsafe closing speed failure mode."""
    max_delta_speed = float(metrics.get("max_delta_speed_kph") or 0.0)
    max_closing_speed = float(metrics.get("max_closing_speed_kph") or 0.0)

    # Scan unsafe events for additional speed evidence
    for ev in (unsafe_events or []):
        details = extract_event_details(ev)
        ev_delta = float(details.get("delta_speed_kph") or 0.0)
        ev_closing = float(details.get("closing_speed_kph") or 0.0)
        if ev_delta > max_delta_speed:
            max_delta_speed = ev_delta
        if ev_closing > max_closing_speed:
            max_closing_speed = ev_closing

    # Check reason codes from exploit_score
    reason_code_hit = False
    es_reason_codes: list[str] = []
    if isinstance(exploit_score, dict):
        es_reason_codes = list(exploit_score.get("reason_codes") or [])
        if UNSAFE_CLOSING_SPEED in es_reason_codes:
            reason_code_hit = True

    # Rule: max_delta_speed >= 60 OR max_closing_speed >= 25
    triggered = max_delta_speed >= 60.0 or max_closing_speed >= 25.0 or reason_code_hit
    if not triggered:
        return None

    reason_codes: list[str] = []
    if max_delta_speed >= 60.0:
        reason_codes.append("high_delta_speed")
    if max_closing_speed >= 25.0:
        reason_codes.append("high_closing_speed")
    if reason_code_hit:
        reason_codes.append("unsafe_closing_speed_reason_present")
    if not reason_codes:
        reason_codes.append("unsafe_closing_speed_reason_present")

    # Confidence
    if max_delta_speed >= 80.0:
        confidence = "high"
    elif max_delta_speed >= 60.0 or max_closing_speed >= 25.0:
        confidence = "medium"
    else:
        confidence = "low"

    # Score
    score = _clamp01(max(max_delta_speed / 100.0, max_closing_speed / 40.0))
    if score == 0.0 and reason_code_hit:
        score = 0.30

    event_refs = extract_failure_event_refs(metrics, unsafe_events)

    return FailureModeEvidence(
        mode=UNSAFE_CLOSING_SPEED,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={
            "max_delta_speed_kph": max_delta_speed,
            "max_closing_speed_kph": max_closing_speed,
        },
    )


def detect_grey_area_active_aero(
    unsafe_events: list[dict[str, Any]] | None,
    exploit_score: dict[str, Any] | None,
) -> FailureModeEvidence | None:
    """Detect grey-area active aero exploit failure mode."""
    active_aero_found = False
    grey_area_active = False
    event_refs: list[str] = []
    reason_codes: list[str] = []

    for ev in (unsafe_events or []):
        legal_status = str(ev.get("legal_status") or "").upper()
        reg_causes = normalize_reason_values(ev.get("regulatory_causes"))
        grey_flags = normalize_reason_values(ev.get("grey_area_flags"))

        # Check for active aero terms
        has_active_aero = any(t in reg_causes for t in _ACTIVE_AERO_TERMS)
        has_active_aero_flag = any(t in grey_flags for t in _ACTIVE_AERO_TERMS)

        if has_active_aero or has_active_aero_flag:
            active_aero_found = True
            if legal_status in ("GREY_AREA",):
                grey_area_active = True

        # Collect event refs
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    # Check exploit_score reason codes
    reason_code_hit = False
    if isinstance(exploit_score, dict):
        es_codes = list(exploit_score.get("reason_codes") or [])
        if "active_aero_attack_window" in es_codes:
            reason_code_hit = True
            active_aero_found = True

    if not active_aero_found:
        return None

    if grey_area_active:
        confidence = "high"
        score = 0.75
        reason_codes.append("grey_area_active_aero_cause")
    else:
        confidence = "medium"
        score = 0.55
        reason_codes.append("active_aero_cause_present")

    if reason_code_hit:
        reason_codes.append("active_aero_attack_window_reason_code")

    return FailureModeEvidence(
        mode=GREY_AREA_ACTIVE_AERO,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={"grey_area_active": grey_area_active},
    )


def detect_pack_compression_exploit(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
) -> FailureModeEvidence | None:
    """Detect pack compression exploit failure mode."""
    pack_found = False
    event_refs: list[str] = []

    for ev in (unsafe_events or []):
        amplifiers = normalize_reason_values(ev.get("amplifiers"))
        details = extract_event_details(ev)
        detail_amplifiers = normalize_reason_values(details.get("amplifiers"))
        safety_verdict_amplifiers = normalize_reason_values(
            details.get("safety_verdict_amplifiers")
        )
        all_amps = amplifiers + detail_amplifiers + safety_verdict_amplifiers

        if "pack_compression" in all_amps:
            pack_found = True
            ref = ev.get("event_ref") or ev.get("ref")
            if isinstance(ref, str) and ref and ref not in event_refs:
                event_refs.append(ref)

    if not pack_found:
        return None

    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    if unsafe_count > 0:
        confidence = "high"
        score = 0.70
        reason_codes = ["pack_compression_amplifier", "unsafe_legal_state_present"]
    else:
        confidence = "medium"
        score = 0.50
        reason_codes = ["pack_compression_amplifier"]

    return FailureModeEvidence(
        mode=PACK_COMPRESSION_EXPLOIT,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={"pack_compression_found": True},
    )


def detect_low_visibility_attack(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
    candidate_parameters: dict[str, float] | None,
) -> FailureModeEvidence | None:
    """Detect low visibility attack failure mode."""
    params = candidate_parameters or {}
    visibility_m = params.get("visibility_m")
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)

    # Also scan event amplifiers/details for visibility terms
    visibility_in_events = False
    event_refs: list[str] = []
    for ev in (unsafe_events or []):
        amplifiers = normalize_reason_values(ev.get("amplifiers"))
        details = extract_event_details(ev)
        detail_text = str(details).lower()
        amp_text = " ".join(amplifiers).lower()
        if "visibility" in detail_text or "visibility" in amp_text or "perception" in amp_text:
            visibility_in_events = True
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    # Rule: visibility_m <= 700 AND unsafe event exists
    if visibility_m is not None:
        if visibility_m <= 700.0 and unsafe_count > 0:
            if visibility_m <= 500.0:
                confidence = "high"
            else:
                confidence = "medium"
            score = _clamp01(1.0 - visibility_m / 700.0)
            return FailureModeEvidence(
                mode=LOW_VISIBILITY_ATTACK,
                confidence=confidence,
                score=score,
                reason_codes=["low_visibility_param", "unsafe_event_present"],
                event_refs=event_refs,
                supporting_fields={"visibility_m": visibility_m},
            )
    elif visibility_in_events:
        # Only amplifier present
        return FailureModeEvidence(
            mode=LOW_VISIBILITY_ATTACK,
            confidence="medium" if len(event_refs) > 0 else "low",
            score=0.40,
            reason_codes=["low_visibility_amplifier"],
            event_refs=event_refs,
            supporting_fields={},
        )

    return None


def detect_confined_corner_attack(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
    candidate_parameters: dict[str, float] | None,
) -> FailureModeEvidence | None:
    """Detect confined corner attack failure mode."""
    params = candidate_parameters or {}
    width_m = params.get("width_m")
    runoff_type = params.get("runoff_type")
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)

    # Scan events
    event_refs: list[str] = []
    corner_slice = False
    risky_runoff_in_events = False

    for ev in (unsafe_events or []):
        slice_hint = str(ev.get("slice_hint") or "").lower()
        amplifiers = normalize_reason_values(ev.get("amplifiers"))
        amp_text = " ".join(amplifiers).lower()
        details = extract_event_details(ev)
        ev_runoff = str(details.get("runoff_type") or "").lower()

        if "corner" in slice_hint:
            corner_slice = True
        if "confined" in amp_text or "escape_margin" in amp_text:
            corner_slice = True
        if ev_runoff in _RISKY_RUNOFF_SURFACES:
            risky_runoff_in_events = True

        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    # Check runoff from params
    if isinstance(runoff_type, str) and runoff_type.lower() in _RISKY_RUNOFF_SURFACES:
        risky_runoff_in_events = True

    # Rule: width_m <= 12.5 AND unsafe event exists
    if width_m is None or unsafe_count == 0:
        return None

    if width_m <= 12.5:
        if risky_runoff_in_events or corner_slice:
            confidence = "high"
            score = 0.86
        else:
            confidence = "medium"
            score = 0.65
        reason_codes = ["narrow_track_width"]
        if corner_slice:
            reason_codes.append("confined_corner_slice_hint")
        if risky_runoff_in_events:
            reason_codes.append("risky_runoff_surface")
        return FailureModeEvidence(
            mode=CONFINED_CORNER_ATTACK,
            confidence=confidence,
            score=score,
            reason_codes=reason_codes,
            event_refs=event_refs,
            supporting_fields={"width_m": width_m, "runoff_type": runoff_type},
        )

    return None


def detect_reaction_margin_failure(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
) -> FailureModeEvidence | None:
    """Detect reaction margin failure mode."""
    min_margin = metrics.get("min_reaction_margin_s")

    # Also scan event details
    event_refs: list[str] = []
    for ev in (unsafe_events or []):
        details = extract_event_details(ev)
        ev_margin = details.get("reaction_margin_s")
        ev_ttc = details.get("time_to_collision_s")
        for val in (ev_margin, ev_ttc):
            if val is not None:
                try:
                    fval = float(val)
                    if min_margin is None or fval < min_margin:
                        min_margin = fval
                except (TypeError, ValueError):
                    pass
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    if min_margin is None:
        return None

    margin = float(min_margin)
    if margin > 0.75:
        return None

    if margin <= 0.5:
        confidence = "high"
    else:
        confidence = "medium"

    score = _clamp01(1.0 - (margin / 0.75))
    reason_codes = ["low_reaction_margin"]
    if margin <= 0.5:
        reason_codes.append("critical_reaction_margin")

    return FailureModeEvidence(
        mode=REACTION_MARGIN_FAILURE,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={"min_reaction_margin_s": margin},
    )


def detect_energy_delta_exploit(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
    candidate_parameters: dict[str, float] | None,
    exploit_score: dict[str, Any] | None,
) -> FailureModeEvidence | None:
    """Detect energy delta (ERS SOC) exploit failure mode."""
    params = candidate_parameters or {}
    attacker_ers = params.get("attacker_ers_soc")
    defender_ers = params.get("defender_ers_soc")
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)

    if attacker_ers is None or defender_ers is None:
        return None

    delta = float(attacker_ers) - float(defender_ers)

    # Rule: delta >= 0.35 AND unsafe event exists
    if delta < 0.35 or unsafe_count == 0:
        return None

    # Also check exploit_score for ers_delta_advantage
    reason_code_hit = False
    if isinstance(exploit_score, dict):
        es_codes = list(exploit_score.get("reason_codes") or [])
        if "ers_delta_advantage" in es_codes:
            reason_code_hit = True

    if delta >= 0.55:
        confidence = "high"
    else:
        confidence = "medium"

    score = _clamp01((delta - 0.35) / 0.20 * 0.5 + 0.5)

    reason_codes = ["ers_soc_delta_advantage"]
    if reason_code_hit:
        reason_codes.append("ers_delta_reason_code")

    event_refs = extract_failure_event_refs(metrics, unsafe_events)

    return FailureModeEvidence(
        mode=ENERGY_DELTA_EXPLOIT,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={
            "attacker_ers_soc": float(attacker_ers),
            "defender_ers_soc": float(defender_ers),
            "ers_delta": round(delta, 4),
        },
    )


def detect_rejoin_surface_exploit(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
    candidate_parameters: dict[str, float] | None,
) -> FailureModeEvidence | None:
    """Detect unsafe rejoin surface exploit failure mode."""
    params = candidate_parameters or {}
    runoff_type = params.get("runoff_type")
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)

    explicit_unsafe_rejoin = False
    risky_runoff = False
    event_refs: list[str] = []

    for ev in (unsafe_events or []):
        amplifiers = normalize_reason_values(ev.get("amplifiers"))
        if "unsafe_rejoin_surface" in amplifiers:
            explicit_unsafe_rejoin = True
        details = extract_event_details(ev)
        ev_runoff = str(details.get("runoff_type") or "").lower()
        if ev_runoff in _RISKY_RUNOFF_SURFACES:
            risky_runoff = True
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    # Check param runoff_type
    if isinstance(runoff_type, str) and runoff_type.lower() in _RISKY_RUNOFF_SURFACES:
        risky_runoff = True

    triggered = explicit_unsafe_rejoin or (risky_runoff and unsafe_count > 0)
    if not triggered:
        return None

    if explicit_unsafe_rejoin:
        confidence = "high"
        score = 0.80
        reason_codes = ["unsafe_rejoin_surface_amplifier"]
    else:
        confidence = "medium"
        score = 0.60
        reason_codes = ["risky_runoff_surface_param"]

    return FailureModeEvidence(
        mode=REJOIN_SURFACE_EXPLOIT,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=event_refs,
        supporting_fields={
            "runoff_type": runoff_type,
            "explicit_unsafe_rejoin": explicit_unsafe_rejoin,
        },
    )


def detect_patch_resistant_exploit(
    patch_reruns: list[dict[str, Any]] | None,
    exploit_score: dict[str, Any] | None,
) -> FailureModeEvidence | None:
    """Detect patch-resistant exploit failure mode."""
    if not patch_reruns:
        return None

    unchanged_or_worse = False
    improved_with_refs = False

    for pr in patch_reruns:
        verdict = str(pr.get("verdict") or "").upper()
        mitigation_success = bool(pr.get("mitigation_success"))
        remaining_refs = list(pr.get("unsafe_legal_event_refs") or [])

        if verdict in ("UNCHANGED", "WORSE"):
            unchanged_or_worse = True
        elif mitigation_success is False and remaining_refs:
            improved_with_refs = True

    # Check exploit_score patch_resistance component
    score_hit = False
    if isinstance(exploit_score, dict):
        components = exploit_score.get("components") or {}
        if isinstance(components, dict):
            pr_val = float(components.get("patch_resistance") or 0.0)
            if pr_val > 0.5:
                score_hit = True

    triggered = unchanged_or_worse or (improved_with_refs and score_hit)
    if not triggered and not score_hit:
        return None

    if unchanged_or_worse:
        confidence = "high"
        score = 0.85
        reason_codes = ["patch_rerun_unchanged_or_worse"]
    elif improved_with_refs:
        confidence = "medium"
        score = 0.60
        reason_codes = ["patch_improved_but_refs_remaining"]
    else:
        confidence = "medium"
        score = 0.55
        reason_codes = ["high_patch_resistance_score"]

    return FailureModeEvidence(
        mode=PATCH_RESISTANT_EXPLOIT,
        confidence=confidence,
        score=score,
        reason_codes=reason_codes,
        event_refs=[],
        supporting_fields={"patch_rerun_count": len(patch_reruns)},
    )


def detect_high_hazard_legal_state(
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
) -> FailureModeEvidence | None:
    """Detect high-hazard legal state failure mode."""
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    max_hazard = float(metrics.get("max_hazard_score") or 0.0)

    if unsafe_count == 0 or max_hazard < 0.65:
        return None

    if max_hazard >= 0.85:
        confidence = "high"
    else:
        confidence = "medium"

    score = _clamp01(max_hazard)
    event_refs = extract_failure_event_refs(metrics, unsafe_events)

    return FailureModeEvidence(
        mode=HIGH_HAZARD_LEGAL_STATE,
        confidence=confidence,
        score=score,
        reason_codes=["high_max_hazard_score", "unsafe_legal_state_present"],
        event_refs=event_refs,
        supporting_fields={
            "max_hazard_score": max_hazard,
            "unsafe_legal_state_count": unsafe_count,
        },
    )


# ---------------------------------------------------------------------------
# Spirit-of-regulation and technical-directive boundary
# (derived from legal_verdicts)
# ---------------------------------------------------------------------------


def detect_spirit_of_regulation_exploit(
    unsafe_events: list[dict[str, Any]] | None,
    legal_verdicts: list[dict[str, Any]] | None,
) -> FailureModeEvidence | None:
    """Detect spirit-of-regulation exploit failure mode."""
    spirit_found = False
    event_refs: list[str] = []

    for ev in (unsafe_events or []):
        legal_status = str(ev.get("legal_status") or "").upper()
        if legal_status == "SPIRIT_VIOLATION":
            spirit_found = True
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    for vd in (legal_verdicts or []):
        status = str(vd.get("status") or vd.get("legal_status") or "").upper()
        if status == "SPIRIT_VIOLATION":
            spirit_found = True

    if not spirit_found:
        return None

    return FailureModeEvidence(
        mode=SPIRIT_OF_REGULATION_EXPLOIT,
        confidence="high",
        score=0.70,
        reason_codes=["spirit_violation_status"],
        event_refs=event_refs,
        supporting_fields={},
    )


def detect_technical_directive_boundary(
    unsafe_events: list[dict[str, Any]] | None,
    legal_verdicts: list[dict[str, Any]] | None,
) -> FailureModeEvidence | None:
    """Detect technical directive boundary failure mode."""
    td_found = False
    event_refs: list[str] = []

    for ev in (unsafe_events or []):
        legal_status = str(ev.get("legal_status") or "").upper()
        if legal_status == "NEEDS_TECHNICAL_DIRECTIVE":
            td_found = True
        ref = ev.get("event_ref") or ev.get("ref")
        if isinstance(ref, str) and ref and ref not in event_refs:
            event_refs.append(ref)

    for vd in (legal_verdicts or []):
        status = str(vd.get("status") or vd.get("legal_status") or "").upper()
        if status == "NEEDS_TECHNICAL_DIRECTIVE":
            td_found = True

    if not td_found:
        return None

    return FailureModeEvidence(
        mode=TECHNICAL_DIRECTIVE_BOUNDARY,
        confidence="medium",
        score=0.65,
        reason_codes=["needs_technical_directive_status"],
        event_refs=event_refs,
        supporting_fields={},
    )


# ---------------------------------------------------------------------------
# Deduplication and ranking
# ---------------------------------------------------------------------------


def _dedup_and_merge(
    modes: list[FailureModeEvidence],
) -> list[FailureModeEvidence]:
    """Deduplicate by mode ID, keeping highest score and merging reason_codes/event_refs."""
    best: dict[str, FailureModeEvidence] = {}
    for fm in modes:
        if fm.mode not in best:
            best[fm.mode] = fm
        else:
            existing = best[fm.mode]
            # Keep highest score; if tied, keep highest confidence
            if fm.score > existing.score or (
                fm.score == existing.score
                and _CONFIDENCE_RANK.get(fm.confidence, 0)
                > _CONFIDENCE_RANK.get(existing.confidence, 0)
            ):
                merged_codes = list(
                    dict.fromkeys(fm.reason_codes + existing.reason_codes)
                )
                merged_refs = list(
                    dict.fromkeys(fm.event_refs + existing.event_refs)
                )
                best[fm.mode] = FailureModeEvidence(
                    mode=fm.mode,
                    confidence=fm.confidence,
                    score=fm.score,
                    reason_codes=merged_codes,
                    event_refs=merged_refs,
                    supporting_fields={**existing.supporting_fields, **fm.supporting_fields},
                )
            else:
                # Merge codes and refs into existing
                merged_codes = list(
                    dict.fromkeys(existing.reason_codes + fm.reason_codes)
                )
                merged_refs = list(
                    dict.fromkeys(existing.event_refs + fm.event_refs)
                )
                best[fm.mode] = FailureModeEvidence(
                    mode=existing.mode,
                    confidence=existing.confidence,
                    score=existing.score,
                    reason_codes=merged_codes,
                    event_refs=merged_refs,
                    supporting_fields={**existing.supporting_fields, **fm.supporting_fields},
                )

    return list(best.values())


def _sort_failure_modes(modes: list[FailureModeEvidence]) -> list[FailureModeEvidence]:
    """Sort: score desc, confidence rank desc, mode asc."""
    return sorted(
        modes,
        key=lambda m: (
            -m.score,
            -_CONFIDENCE_RANK.get(m.confidence, 0),
            m.mode,
        ),
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_failure_taxonomy(
    *,
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None = None,
    legal_verdicts: list[dict[str, Any]] | None = None,
    candidate_parameters: dict[str, float] | None = None,
    patch_reruns: list[dict[str, Any]] | None = None,
    exploit_score: dict[str, Any] | None = None,
    track_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic failure taxonomy for a candidate.

    Runs all detectors, deduplicates results, sorts by score, and returns
    a JSON-compatible dict. Never modifies scoring or ranking.

    Args:
        metrics: Evidence bundle metrics dict.
        unsafe_events: List of unsafe_legal_state event dicts.
        legal_verdicts: List of legal verdict dicts.
        candidate_parameters: Candidate parameter dict.
        patch_reruns: List of patch rerun dicts.
        exploit_score: Multi-objective exploit score dict (PR 8.1).

    Returns:
        JSON-serializable dict with schema_version, primary_failure_mode,
        failure_modes, event_refs, and limitations.
    """
    detected: list[FailureModeEvidence] = []

    # Run all detectors
    r = detect_unsafe_closing_speed(metrics, unsafe_events, exploit_score)
    if r is not None:
        detected.append(r)

    r = detect_grey_area_active_aero(unsafe_events, exploit_score)
    if r is not None:
        detected.append(r)

    r = detect_pack_compression_exploit(metrics, unsafe_events)
    if r is not None:
        detected.append(r)

    r = detect_low_visibility_attack(metrics, unsafe_events, candidate_parameters)
    if r is not None:
        detected.append(r)

    r = detect_confined_corner_attack(metrics, unsafe_events, candidate_parameters)
    if r is not None:
        detected.append(r)

    r = detect_reaction_margin_failure(metrics, unsafe_events)
    if r is not None:
        detected.append(r)

    r = detect_energy_delta_exploit(metrics, unsafe_events, candidate_parameters, exploit_score)
    if r is not None:
        detected.append(r)

    r = detect_rejoin_surface_exploit(metrics, unsafe_events, candidate_parameters)
    if r is not None:
        detected.append(r)

    r = detect_patch_resistant_exploit(patch_reruns, exploit_score)
    if r is not None:
        detected.append(r)

    r = detect_high_hazard_legal_state(metrics, unsafe_events)
    if r is not None:
        detected.append(r)

    r = detect_spirit_of_regulation_exploit(unsafe_events, legal_verdicts)
    if r is not None:
        detected.append(r)

    r = detect_technical_directive_boundary(unsafe_events, legal_verdicts)
    if r is not None:
        detected.append(r)

    # Deduplicate and sort
    deduped = _dedup_and_merge(detected)
    sorted_modes = _sort_failure_modes(deduped)

    # Aggregate event refs
    all_event_refs = extract_failure_event_refs(metrics, unsafe_events)

    # Primary failure mode
    primary = sorted_modes[0].mode if sorted_modes else None

    # Limitations
    limitations = list(_STANDARD_LIMITATIONS)
    if not sorted_modes:
        limitations.append(
            "No failure mode was detected from available evidence."
        )
    # Optionally annotate with track fidelity context (PR 8.4.1).
    # Does NOT change mode detection, thresholds, or scoring.
    if isinstance(track_fidelity, dict):
        tier = str(track_fidelity.get("fidelity_tier") or "")
        if tier:
            limitations.append(
                f"Failure label is conditioned on {tier} geometry."
            )
        known_gaps = track_fidelity.get("known_gaps") or []
        if known_gaps:
            limitations.append(
                f"Track known gaps may affect precision: "
                f"{', '.join(str(g) for g in known_gaps[:4])}."
            )

    result = FailureTaxonomyResult(
        schema_version=FAILURE_TAXONOMY_SCHEMA,
        primary_failure_mode=primary,
        failure_modes=sorted_modes,
        event_refs=all_event_refs,
        limitations=limitations,
    )
    return failure_taxonomy_to_dict(result)
