"""Multi-objective exploit scoring for falsification search (PR 8.1).

Upgrades the single legacy score into a structured five-component score:
    safety_risk, legal_exploit, competitive_advantage (proxy),
    patch_resistance, novelty.

Key invariants:
- Fully deterministic: same inputs → same output.
- Evidence-based: reads from bundle metrics / event dicts only.
- No LLM, no NVIDIA, no external services.
- JSON-serializable output.
- Legacy score_candidate_metrics() is preserved unchanged in search.py.
- This module computes an ADDITIONAL score; it does NOT replace the legacy.

Competitive advantage component is a proxy, not calibrated truth.
See limitations in each build_exploit_score() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPLOIT_SCORE_SCHEMA = "exploit_score.v1"

_SAFETY_RISK_WEIGHT_DEFAULT = 8.0
_LEGAL_EXPLOIT_WEIGHT_DEFAULT = 4.0
_COMPETITIVE_ADVANTAGE_WEIGHT_DEFAULT = 3.0
_PATCH_RESISTANCE_WEIGHT_DEFAULT = 2.0
_NOVELTY_WEIGHT_DEFAULT = 1.0

_LEGAL_FAMILY: frozenset[str] = frozenset({
    "LEGAL",
    "GREY_AREA",
    "SPIRIT_VIOLATION",
    "NEEDS_STEWARD_REVIEW",
    "NEEDS_TECHNICAL_DIRECTIVE",
})
_ILLEGAL_FAMILY: frozenset[str] = frozenset({"ILLEGAL"})

_STANDARD_LIMITATIONS = [
    "Competitive advantage is a proxy, not calibrated.",
    "Patch resistance is zero when no patch rerun evidence is available.",
    "Score summarizes deterministic evidence; it is not an oracle.",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExploitScoreWeights:
    """Weights applied to each score component."""

    safety_risk: float = _SAFETY_RISK_WEIGHT_DEFAULT
    legal_exploit: float = _LEGAL_EXPLOIT_WEIGHT_DEFAULT
    competitive_advantage: float = _COMPETITIVE_ADVANTAGE_WEIGHT_DEFAULT
    patch_resistance: float = _PATCH_RESISTANCE_WEIGHT_DEFAULT
    novelty: float = _NOVELTY_WEIGHT_DEFAULT


@dataclass(frozen=True)
class ExploitScoreComponents:
    """Raw (0.0-1.0) values for each score component."""

    safety_risk: float
    legal_exploit: float
    competitive_advantage: float
    patch_resistance: float
    novelty: float


@dataclass(frozen=True)
class ExploitScore:
    """Structured multi-objective exploit score."""

    schema_version: str
    total: float
    components: ExploitScoreComponents
    weights: ExploitScoreWeights
    reason_codes: list[str]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 2.1 Safety risk component
# ---------------------------------------------------------------------------


def compute_safety_risk_component(
    metrics: dict[str, Any],
) -> tuple[float, list[str]]:
    """Compute safety risk component (0.0-1.0) from evidence bundle metrics.

    Higher = more unsafe-legal evidence in this run.
    """
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    max_hazard = float(metrics.get("max_hazard_score") or 0.0)
    mean_hazard = float(metrics.get("mean_hazard_score") or 0.0)
    status_counts: dict[str, Any] = metrics.get("safety_verdict_status_counts") or {}

    component = (
        min(unsafe_count, 5) / 5.0 * 0.40
        + max_hazard * 0.35
        + mean_hazard * 0.15
        + (0.10 if status_counts.get("UNSAFE_LEGAL", 0) else 0.0)
    )

    reason_codes: list[str] = []
    if unsafe_count > 0:
        reason_codes.append("unsafe_legal_state_present")
    if unsafe_count >= 3:
        reason_codes.append("multiple_unsafe_legal_states")
    if max_hazard >= 0.60:
        reason_codes.append("high_max_hazard_score")
    if status_counts.get("UNSAFE_LEGAL", 0):
        reason_codes.append("unsafe_legal_verdict_present")

    return _clamp(component), reason_codes


# ---------------------------------------------------------------------------
# 2.2 Legal exploit component
# ---------------------------------------------------------------------------


def compute_legal_exploit_component(
    legal_verdicts: list[dict[str, Any]] | None,
    unsafe_events: list[dict[str, Any]] | None,
    metrics: dict[str, Any],
) -> tuple[float, list[str]]:
    """Compute legal exploit component (0.0-1.0).

    Rewards unsafe behaviour that remains legally allowed / grey-area.
    Reduces score when illegal status dominates.
    """
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    if unsafe_count == 0:
        return 0.0, []

    reason_codes: list[str] = []
    events = unsafe_events or []

    # Collect legal statuses from unsafe events
    statuses: list[str] = []
    grey_area_flag_count = 0
    regulatory_cause_present = False

    for ev in events:
        ls = str(ev.get("legal_status") or "").upper()
        if ls:
            statuses.append(ls)
        if ev.get("grey_area_flags"):
            grey_area_flag_count += 1
        if ev.get("regulatory_causes"):
            regulatory_cause_present = True

    # Also check legal_verdicts for statuses
    for vd in (legal_verdicts or []):
        ls = str(vd.get("status") or vd.get("legal_status") or "").upper()
        if ls:
            statuses.append(ls)

    if not statuses:
        # No status info — give modest credit for having an unsafe event
        return 0.20, ["unsafe_event_no_legal_status"]

    # Score by best (highest) legal status in LEGAL_FAMILY
    _status_scores: dict[str, float] = {
        "LEGAL": 0.35,
        "GREY_AREA": 0.45,
        "SPIRIT_VIOLATION": 0.35,
        "NEEDS_STEWARD_REVIEW": 0.25,
        "NEEDS_TECHNICAL_DIRECTIVE": 0.25,
    }

    legal_hits = [s for s in statuses if s in _LEGAL_FAMILY]
    illegal_hits = [s for s in statuses if s in _ILLEGAL_FAMILY]

    base_score = 0.0
    for s in legal_hits:
        val = _status_scores.get(s, 0.0)
        if val > base_score:
            base_score = val

    # Reason codes for statuses present
    if "LEGAL" in legal_hits:
        reason_codes.append("unsafe_but_legal")
    if "GREY_AREA" in legal_hits:
        reason_codes.append("unsafe_grey_area")
    if "SPIRIT_VIOLATION" in legal_hits:
        reason_codes.append("unsafe_spirit_violation")
    if "NEEDS_STEWARD_REVIEW" in legal_hits or "NEEDS_TECHNICAL_DIRECTIVE" in legal_hits:
        reason_codes.append("steward_review_boundary")
    if illegal_hits:
        reason_codes.append("illegal_status_reduces_exploit_value")

    # Bonus for grey_area_flags / regulatory_causes
    bonus = 0.0
    if grey_area_flag_count:
        bonus += 0.10
    if regulatory_cause_present:
        bonus += 0.10
        reason_codes.append("regulatory_cause_present")

    component = base_score + min(bonus, 0.20)

    # If illegal dominates: reduce score by 50%
    if illegal_hits and not legal_hits:
        component *= 0.5

    return _clamp(component), reason_codes


# ---------------------------------------------------------------------------
# 2.3 Competitive advantage component (proxy)
# ---------------------------------------------------------------------------


def compute_competitive_advantage_component(
    candidate_parameters: dict[str, float] | None,
    metrics: dict[str, Any],
    unsafe_events: list[dict[str, Any]] | None,
) -> tuple[float, list[str]]:
    """Compute competitive advantage proxy component (0.0-1.0).

    This is a heuristic proxy using available deterministic evidence.
    It is NOT calibrated. Always includes limitation reason code.
    """
    params = candidate_parameters or {}
    events = unsafe_events or []

    attacker_risk = float(params.get("attacker_risk_level", 0.0))
    attacker_ers = float(params.get("attacker_ers_soc", 0.0))
    defender_ers = float(params.get("defender_ers_soc", 0.0))
    gap_s = float(params.get("gap_s", 1.0))

    ers_delta = max(attacker_ers - defender_ers, 0.0)
    gap_pressure = max(0.0, 1.0 - min(gap_s, 1.0))

    # Normalize delta speed from metrics
    max_delta_speed = float(metrics.get("max_delta_speed_kph") or 0.0)
    # Assume 80 kph as rough normalisation reference for F1 overtake window
    normalized_delta_speed = _clamp(max_delta_speed / 80.0)

    # Regulatory cause bonus from events
    regulatory_cause_found = any(ev.get("regulatory_causes") for ev in events)
    regulatory_cause_bonus = 0.15 if regulatory_cause_found else 0.0

    component = (
        0.30 * attacker_risk
        + 0.25 * ers_delta
        + 0.20 * gap_pressure
        + 0.15 * normalized_delta_speed
        + 0.10 * regulatory_cause_bonus
    )

    reason_codes: list[str] = []
    if attacker_risk >= 0.70:
        reason_codes.append("high_attacker_risk")
    if ers_delta >= 0.20:
        reason_codes.append("ers_delta_advantage")
    if gap_pressure >= 0.50:
        reason_codes.append("close_gap_pressure")
    if normalized_delta_speed >= 0.30:
        reason_codes.append("high_delta_speed_advantage")
    if regulatory_cause_found:
        reason_codes.append("active_aero_attack_window")

    # Always mark as proxy
    reason_codes.append("competitive_advantage_is_proxy_not_calibrated")

    return _clamp(component), reason_codes


# ---------------------------------------------------------------------------
# 2.4 Patch resistance component
# ---------------------------------------------------------------------------


def compute_patch_resistance_component(
    patch_reruns: list[dict[str, Any]] | None,
    metrics: dict[str, Any],
) -> tuple[float, list[str]]:
    """Compute patch resistance component (0.0-1.0).

    Rewards exploits that survive simple regulatory intervention proxies.
    Returns 0.0 with reason no_patch_evidence when no patch reruns provided.
    """
    if not patch_reruns:
        return 0.0, ["no_patch_evidence"]

    reason_codes: list[str] = []
    mitigated_count = 0
    survived_count = 0
    improved_hazard_count = 0

    for pr in patch_reruns:
        verdict = str(pr.get("verdict") or "").upper()
        mitigation_success = bool(pr.get("mitigation_success"))
        hazard_reduced = bool(pr.get("hazard_reduced"))

        if mitigation_success or verdict == "MITIGATED":
            mitigated_count += 1
        elif verdict in ("WORSE", "UNCHANGED"):
            survived_count += 1
        elif hazard_reduced:
            improved_hazard_count += 1
        else:
            survived_count += 1

    total = len(patch_reruns)

    if total == 0:
        return 0.0, ["no_patch_evidence"]

    if mitigated_count == total:
        reason_codes.append("patch_mitigates_exploit")
        return 0.0, reason_codes

    if survived_count >= total:
        reason_codes.append("exploit_survives_patch")
        return 0.8, reason_codes

    if survived_count > 0:
        ratio = survived_count / total
        reason_codes.append("exploit_survives_patch")
        if improved_hazard_count > 0:
            reason_codes.append("patch_only_reduces_hazard")
        return _clamp(0.4 + 0.4 * ratio), reason_codes

    if improved_hazard_count > 0:
        reason_codes.append("patch_only_reduces_hazard")
        return 0.4, reason_codes

    reason_codes.append("exploit_survives_patch")
    return 0.5, reason_codes


# ---------------------------------------------------------------------------
# 2.5 Novelty component
# ---------------------------------------------------------------------------


def compute_novelty_component(
    candidate_id: str | None,
    family_id: str | None,
    event_refs: list[str] | None,
    prior_findings: list[dict[str, Any]] | None = None,
) -> tuple[float, list[str]]:
    """Compute novelty component (0.0-1.0).

    Simple overlap-based novelty. Does not build a database.
    prior_findings passed as argument for determinism.
    """
    if not prior_findings:
        return 0.5, ["no_prior_findings"]

    reason_codes: list[str] = []

    # Check duplicate candidate_id
    if candidate_id:
        for pf in prior_findings:
            if pf.get("candidate_id") == candidate_id:
                reason_codes.append("duplicate_candidate")
                return 0.0, reason_codes

    # Check event_ref overlap
    current_refs: set[str] = set(event_refs or [])
    prior_refs: set[str] = set()
    for pf in prior_findings:
        for ref in pf.get("event_refs") or []:
            prior_refs.add(str(ref))

    if current_refs and prior_refs:
        overlap = len(current_refs & prior_refs)
        total = len(current_refs)
        overlap_ratio = overlap / total if total > 0 else 0.0

        if overlap_ratio >= 0.80:
            reason_codes.append("event_ref_overlap")
            return 0.1, reason_codes

    # Check family novelty
    prior_families: set[str] = {str(pf.get("family_id") or "") for pf in prior_findings}
    if family_id and family_id not in prior_families:
        reason_codes.append("new_family")
        return 0.8, reason_codes

    # Same family, new candidate
    reason_codes.append("new_candidate")
    return 0.5, reason_codes


# ---------------------------------------------------------------------------
# Task 3 — Aggregate score
# ---------------------------------------------------------------------------


def build_exploit_score(
    *,
    metrics: dict[str, Any],
    candidate_parameters: dict[str, float] | None = None,
    legal_verdicts: list[dict[str, Any]] | None = None,
    unsafe_events: list[dict[str, Any]] | None = None,
    patch_reruns: list[dict[str, Any]] | None = None,
    candidate_id: str | None = None,
    family_id: str | None = None,
    event_refs: list[str] | None = None,
    prior_findings: list[dict[str, Any]] | None = None,
    weights: ExploitScoreWeights | None = None,
) -> dict[str, Any]:
    """Build the structured multi-objective exploit score.

    Returns a JSON-serializable dict with schema_version, total,
    components, weighted_components, weights, reason_codes, limitations.

    Legacy score_candidate_metrics() is NOT replaced. This score is
    additional and must not silently change ranking.
    """
    w = weights or ExploitScoreWeights()

    sr_val, sr_codes = compute_safety_risk_component(metrics)
    le_val, le_codes = compute_legal_exploit_component(
        legal_verdicts, unsafe_events, metrics
    )
    ca_val, ca_codes = compute_competitive_advantage_component(
        candidate_parameters, metrics, unsafe_events
    )
    pr_val, pr_codes = compute_patch_resistance_component(patch_reruns, metrics)
    nv_val, nv_codes = compute_novelty_component(
        candidate_id, family_id, event_refs, prior_findings
    )

    weighted: dict[str, float] = {
        "safety_risk": round(sr_val * w.safety_risk, 6),
        "legal_exploit": round(le_val * w.legal_exploit, 6),
        "competitive_advantage": round(ca_val * w.competitive_advantage, 6),
        "patch_resistance": round(pr_val * w.patch_resistance, 6),
        "novelty": round(nv_val * w.novelty, 6),
    }

    total = round(sum(weighted.values()), 4)

    all_reason_codes: list[str] = []
    seen_codes: set[str] = set()
    for code in sr_codes + le_codes + ca_codes + pr_codes + nv_codes:
        if code not in seen_codes:
            seen_codes.add(code)
            all_reason_codes.append(code)

    limitations = list(_STANDARD_LIMITATIONS)

    return {
        "schema_version": EXPLOIT_SCORE_SCHEMA,
        "total": total,
        "components": {
            "safety_risk": round(sr_val, 6),
            "legal_exploit": round(le_val, 6),
            "competitive_advantage": round(ca_val, 6),
            "patch_resistance": round(pr_val, 6),
            "novelty": round(nv_val, 6),
        },
        "weighted_components": weighted,
        "weights": {
            "safety_risk": w.safety_risk,
            "legal_exploit": w.legal_exploit,
            "competitive_advantage": w.competitive_advantage,
            "patch_resistance": w.patch_resistance,
            "novelty": w.novelty,
        },
        "reason_codes": all_reason_codes,
        "limitations": limitations,
    }
