"""Deterministic falsification search engine.

Searches over synthetic family parameters and action parameters to find
candidate worlds and actions that maximize unsafe legal evidence.

No LLM, no NVIDIA, no LangChain. Pure deterministic runtime.
Candidate generation is reproducible: same seed → same candidates.

This is the search core. LangChain tool wrappers come in PR 7.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from reglabsim.falsification.failure_taxonomy import build_failure_taxonomy
from reglabsim.falsification.scoring import build_exploit_score
from reglabsim.logging.audit_report import build_audit_report
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.synthetic.families import (
    SYNTHETIC_FAMILIES,
    SyntheticFamilySpec,
    build_synthetic_actions_for_battle,
    build_synthetic_cars_for_battle,
    build_synthetic_family_run_output,
    build_synthetic_track,
    build_synthetic_track_state,
    build_synthetic_weather,
)
from reglabsim.tracks.fidelity import (
    build_track_fidelity_report,
    compact_track_fidelity_summary,
)
from reglabsim.tracks.track_model import build_track_model_from_synthetic_family

_BASE_REGULATION: dict[str, Any] = {
    "power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchParameterRange:
    """Closed interval + discretisation for one search axis."""

    name: str
    min_value: float
    max_value: float
    steps: int


@dataclass(frozen=True)
class FalsificationCandidate:
    """One deterministic parameter combination to evaluate."""

    candidate_id: str
    family_id: str
    seed: int
    parameters: dict[str, float]


@dataclass
class FalsificationResult:
    """Outcome of running one FalsificationCandidate."""

    candidate_id: str
    family_id: str
    seed: int
    parameters: dict[str, float]
    unsafe_legal_state_count: int
    max_hazard_score: float | None
    mean_hazard_score: float | None
    score: float
    event_refs: list[str] = field(default_factory=list)
    bundle: dict[str, Any] | None = None
    exploit_score: dict[str, Any] | None = None
    failure_taxonomy: dict[str, Any] | None = None
    track_fidelity: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Task 3 — Default search space
# ---------------------------------------------------------------------------


def default_search_space() -> dict[str, SearchParameterRange]:
    """Return the default parameter search space for falsification."""
    return {
        "width_m": SearchParameterRange("width_m", 9.0, 14.0, 3),
        "barrier_distance_m": SearchParameterRange("barrier_distance_m", 4.0, 16.0, 3),
        "unsafe_closing_speed_threshold_kph": SearchParameterRange(
            "unsafe_closing_speed_threshold_kph", 30.0, 55.0, 3
        ),
        "visibility_m": SearchParameterRange("visibility_m", 500.0, 1200.0, 3),
        "wetness_level": SearchParameterRange("wetness_level", 0.0, 0.35, 3),
        "attacker_risk_level": SearchParameterRange("attacker_risk_level", 0.55, 0.95, 3),
        "defender_risk_level": SearchParameterRange("defender_risk_level", 0.55, 0.90, 3),
        "attacker_ers_soc": SearchParameterRange("attacker_ers_soc", 0.45, 0.95, 3),
        "defender_ers_soc": SearchParameterRange("defender_ers_soc", 0.15, 0.65, 3),
        "gap_s": SearchParameterRange("gap_s", 0.15, 0.65, 3),
    }


# ---------------------------------------------------------------------------
# Task 4 — Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    family_id: str,
    seed: int = 42,
    max_trials: int = 25,
    search_space: dict[str, SearchParameterRange] | None = None,
) -> list[FalsificationCandidate]:
    """Generate deterministic parameter candidates for one synthetic family.

    Same seed → same candidates. Uses seeded pseudo-random sampling
    with a bounded uniform draw over each parameter's range.
    """
    if family_id not in SYNTHETIC_FAMILIES:
        known = list(SYNTHETIC_FAMILIES)
        raise ValueError(f"Unknown family_id: {family_id!r}. Choose from {known}")
    space = search_space or default_search_space()
    param_names = sorted(space.keys())
    rng = random.Random(seed)

    candidates: list[FalsificationCandidate] = []
    for idx in range(max_trials):
        params: dict[str, float] = {}
        for name in param_names:
            prange = space[name]
            if prange.steps <= 1:
                params[name] = prange.min_value
            else:
                raw = rng.random()
                params[name] = round(
                    prange.min_value + raw * (prange.max_value - prange.min_value), 4
                )
        candidate_id = f"{family_id}:seed{seed}:trial{idx:04d}"
        candidates.append(
            FalsificationCandidate(
                candidate_id=candidate_id,
                family_id=family_id,
                seed=seed,
                parameters=params,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Task 5 — Apply candidate parameters and run microkernel
# ---------------------------------------------------------------------------


def _apply_candidate_to_spec(
    base_spec: SyntheticFamilySpec,
    params: dict[str, float],
) -> SyntheticFamilySpec:
    """Return a new SyntheticFamilySpec with candidate parameters applied."""
    width_m = float(params.get("width_m", base_spec.width_m))
    barrier_distance_m = float(params.get("barrier_distance_m", base_spec.barrier_distance_m))
    _threshold_key = "unsafe_closing_speed_threshold_kph"
    unsafe_closing_speed = float(
        params.get(_threshold_key, base_spec.unsafe_closing_speed_threshold_kph)
    )
    visibility_m = float(params.get("visibility_m", base_spec.visibility_m))
    wetness_level = float(params.get("wetness_level", base_spec.wetness_level))

    from dataclasses import replace as _replace
    return _replace(
        base_spec,
        width_m=width_m,
        barrier_distance_m=barrier_distance_m,
        unsafe_closing_speed_threshold_kph=unsafe_closing_speed,
        visibility_m=visibility_m,
        wetness_level=wetness_level,
    )


def _apply_action_overrides(
    actions: dict[str, Any],
    params: dict[str, float],
) -> dict[str, Any]:
    """Apply candidate action parameters; returns new dict without mutating input."""
    attacker_id = "car_02"
    defender_id = "car_01"
    result: dict[str, Any] = {}
    for car_id, action in actions.items():
        from dataclasses import replace as _replace
        if car_id == attacker_id:
            overrides: dict[str, Any] = {}
            if "attacker_risk_level" in params:
                overrides["risk_level"] = float(params["attacker_risk_level"])
            result[car_id] = _replace(action, **overrides) if overrides else action
        elif car_id == defender_id:
            overrides = {}
            if "defender_risk_level" in params:
                overrides["risk_level"] = float(params["defender_risk_level"])
            result[car_id] = _replace(action, **overrides) if overrides else action
        else:
            result[car_id] = action
    return result


def _apply_car_overrides(
    cars: list[Any],
    params: dict[str, float],
) -> list[Any]:
    """Apply candidate car state parameters; returns new list without mutating input."""
    from dataclasses import replace as _replace
    result = []
    for car in cars:
        overrides: dict[str, Any] = {}
        if car.car_id == "car_02" and "attacker_ers_soc" in params:
            overrides["ers_soc"] = float(params["attacker_ers_soc"])
            if "gap_s" in params:
                overrides["gap_ahead_s"] = float(params["gap_s"])
                overrides["cumulative_time_s"] = 90.0 + float(params["gap_s"])
        elif car.car_id == "car_01" and "defender_ers_soc" in params:
            overrides["ers_soc"] = float(params["defender_ers_soc"])
        result.append(_replace(car, **overrides) if overrides else car)
    return result


def run_candidate(
    candidate: FalsificationCandidate,
    include_bundle: bool = False,
) -> FalsificationResult:
    """Run one FalsificationCandidate through the deterministic microkernel.

    Builds evidence bundle via ReplayEngine and scores the result.
    Does not use any LLM or external service.
    """
    base_spec = SYNTHETIC_FAMILIES[candidate.family_id]
    spec = _apply_candidate_to_spec(base_spec, candidate.parameters)

    track = build_synthetic_track(spec)
    weather = build_synthetic_weather(spec)
    track_state = build_synthetic_track_state(spec)
    cars = build_synthetic_cars_for_battle(spec)
    actions = build_synthetic_actions_for_battle(spec)

    # Apply action/car overrides from candidate parameters
    cars = _apply_car_overrides(cars, candidate.parameters)
    actions = _apply_action_overrides(actions, candidate.parameters)

    kernel = RaceMicrokernel(regulation=_BASE_REGULATION, seed=candidate.seed)
    _, events, _ = kernel.resolve_lap(
        lap=1,
        total_laps=5,
        cars=cars,
        actions=actions,
        track=track,
        weather=weather,
        track_state=track_state,
        safety_car_active=False,
    )

    event_dicts = [e.to_dict() for e in events]
    unsafe_events = [e for e in event_dicts if e.get("event_type") == "unsafe_legal_state"]

    run_output = build_synthetic_family_run_output(
        {
            "family_id": candidate.family_id,
            "events": event_dicts,
            "unsafe_legal_events": unsafe_events,
            "track": track,
            "weather": weather,
            "track_state": track_state,
        }
    )
    run_output["manifest"]["run_id"] = candidate.candidate_id
    run_output["manifest"]["seed"] = candidate.seed

    bundle = ReplayEngine().build_evidence_bundle(run_output)
    metrics = bundle.get("metrics", {})

    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    max_hazard = metrics.get("max_hazard_score")
    mean_hazard = metrics.get("mean_hazard_score")
    score = score_candidate_metrics(metrics)

    event_refs: list[str] = list(metrics.get("unsafe_legal_event_refs") or [])

    # Build multi-objective exploit score (PR 8.1).
    # Legacy `score` field is preserved and unchanged.
    bundle_unsafe_events: list[dict[str, Any]] = [
        e for e in (bundle.get("events") or [])
        if e.get("event_type") == "unsafe_legal_state"
    ]
    legal_verdicts: list[dict[str, Any]] = list(bundle.get("legal_verdicts") or [])
    exploit_score_dict = build_exploit_score(
        metrics=metrics,
        candidate_parameters=dict(candidate.parameters),
        legal_verdicts=legal_verdicts if legal_verdicts else None,
        unsafe_events=bundle_unsafe_events if bundle_unsafe_events else None,
        patch_reruns=list(bundle.get("patch_reruns") or []) or None,
        candidate_id=candidate.candidate_id,
        family_id=candidate.family_id,
        event_refs=event_refs if event_refs else None,
        prior_findings=None,
    )

    # Build deterministic failure taxonomy (PR 8.2).
    # Does NOT affect scoring or ranking.
    failure_taxonomy_dict = build_failure_taxonomy(
        metrics=metrics,
        unsafe_events=bundle_unsafe_events if bundle_unsafe_events else None,
        legal_verdicts=legal_verdicts if legal_verdicts else None,
        candidate_parameters=dict(candidate.parameters),
        patch_reruns=list(bundle.get("patch_reruns") or []) or None,
        exploit_score=exploit_score_dict,
    )

    # Compact track fidelity metadata (PR 8.4.1).
    # Pure metadata — does NOT affect score, ranking, safety, or legal decisions.
    track_fidelity_dict = _build_compact_track_fidelity(candidate.family_id)

    return FalsificationResult(
        candidate_id=candidate.candidate_id,
        family_id=candidate.family_id,
        seed=candidate.seed,
        parameters=dict(candidate.parameters),
        unsafe_legal_state_count=unsafe_count,
        max_hazard_score=float(max_hazard) if isinstance(max_hazard, (int, float)) else None,
        mean_hazard_score=float(mean_hazard) if isinstance(mean_hazard, (int, float)) else None,
        score=score,
        event_refs=event_refs,
        bundle=bundle if include_bundle else None,
        exploit_score=exploit_score_dict,
        failure_taxonomy=failure_taxonomy_dict,
        track_fidelity=track_fidelity_dict,
    )


# ---------------------------------------------------------------------------
# Task 6 — Score function
# ---------------------------------------------------------------------------


def score_candidate_metrics(metrics: dict[str, Any]) -> float:
    """Compute exploit score from evidence bundle metrics.

    Higher score = stronger unsafe legal evidence.
    Not calibrated truth — a deterministic stress-test ranking proxy.
    """
    unsafe_count = int(metrics.get("unsafe_legal_state_count") or 0)
    max_hazard = float(metrics.get("max_hazard_score") or 0.0)
    mean_hazard = float(metrics.get("mean_hazard_score") or 0.0)
    status_counts = metrics.get("safety_verdict_status_counts") or {}
    unsafe_verdict_bonus = 3.0 if status_counts.get("UNSAFE_LEGAL", 0) else 0.0
    return (
        10.0 * unsafe_count
        + 2.0 * max_hazard
        + 1.0 * mean_hazard
        + unsafe_verdict_bonus
    )


# ---------------------------------------------------------------------------
# Track fidelity helper
# ---------------------------------------------------------------------------


def _build_compact_track_fidelity(family_id: str) -> dict[str, Any]:
    """Return compact track fidelity metadata for a synthetic family.

    Always T0_synthetic_family. Does not affect scoring or ranking.
    """
    spec = SYNTHETIC_FAMILIES.get(family_id)
    spec_dict: dict[str, Any] = {}
    if spec is not None:
        spec_dict = {
            "family_id": spec.family_id,
            "track_id": spec.track_id,
            "segment_id": spec.segment_id,
            "segment_type": spec.segment_type,
            "width_m": spec.width_m,
            "barrier_distance_m": spec.barrier_distance_m,
            "runoff_type": spec.runoff_type,
            "visibility_m": spec.visibility_m,
            "description": spec.description,
        }
    track_model = build_track_model_from_synthetic_family(family_id, spec_dict)
    report = build_track_fidelity_report(track_model)
    return compact_track_fidelity_summary(report)


# ---------------------------------------------------------------------------
# Task 7 — Search entrypoint
# ---------------------------------------------------------------------------


def run_falsification_search(
    family_id: str,
    seed: int = 42,
    max_trials: int = 25,
    include_bundles: bool = False,
) -> dict[str, Any]:
    """Run deterministic falsification search over one synthetic family.

    Returns results ranked by exploit score (descending).
    No LLM, no NVIDIA, no external services.
    """
    if family_id not in SYNTHETIC_FAMILIES:
        raise ValueError(f"Unknown family_id: {family_id!r}")

    space = default_search_space()
    candidates = generate_candidates(
        family_id=family_id,
        seed=seed,
        max_trials=max_trials,
        search_space=space,
    )

    results: list[FalsificationResult] = []
    for candidate in candidates:
        result = run_candidate(candidate, include_bundle=include_bundles)
        results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)

    best = results[0] if results else None
    best_dict: dict[str, Any] | None = None
    if best is not None:
        best_dict = {
            "candidate_id": best.candidate_id,
            "family_id": best.family_id,
            "seed": best.seed,
            "parameters": best.parameters,
            "unsafe_legal_state_count": best.unsafe_legal_state_count,
            "max_hazard_score": best.max_hazard_score,
            "mean_hazard_score": best.mean_hazard_score,
            "score": best.score,
            "score_legacy": best.score,
            "exploit_score": best.exploit_score,
            "failure_taxonomy": best.failure_taxonomy,
            "failure_modes": [
                m["mode"] for m in (best.failure_taxonomy or {}).get("failure_modes", [])
            ],
            "primary_failure_mode": (best.failure_taxonomy or {}).get("primary_failure_mode"),
            "event_refs": best.event_refs,
            "track_fidelity": best.track_fidelity,
        }

    # Compact track fidelity for search-level metadata
    search_track_fidelity = _build_compact_track_fidelity(family_id)

    return {
        "schema_version": "falsification_search.v0",
        "family_id": family_id,
        "seed": seed,
        "max_trials": max_trials,
        "track_fidelity": search_track_fidelity,
        "search_space": {
            name: {
                "min_value": r.min_value,
                "max_value": r.max_value,
                "steps": r.steps,
            }
            for name, r in space.items()
        },
        "best_candidate": best_dict,
        "results": [
            {
                "candidate_id": r.candidate_id,
                "family_id": r.family_id,
                "seed": r.seed,
                "parameters": r.parameters,
                "unsafe_legal_state_count": r.unsafe_legal_state_count,
                "max_hazard_score": r.max_hazard_score,
                "mean_hazard_score": r.mean_hazard_score,
                "score": r.score,
                "score_legacy": r.score,
                "exploit_score": r.exploit_score,
                "primary_failure_mode": (
                    r.failure_taxonomy.get("primary_failure_mode")
                    if r.failure_taxonomy else None
                ),
                "failure_modes": [
                    m["mode"] for m in (r.failure_taxonomy or {}).get("failure_modes", [])
                ],
                "event_refs": r.event_refs,
                **({"bundle": r.bundle} if include_bundles else {}),
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Task 8 — Audit report for best candidate
# ---------------------------------------------------------------------------


def build_best_candidate_audit_report(search_result: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic audit report from the best candidate in a search result.

    Re-runs the best candidate deterministically to obtain a full bundle.
    Returns empty report if no best candidate found.
    """
    best = search_result.get("best_candidate")
    if best is None:
        return build_audit_report({})

    candidate = FalsificationCandidate(
        candidate_id=str(best.get("candidate_id", "best")),
        family_id=str(best.get("family_id", "")),
        seed=int(best.get("seed", 42)),
        parameters=dict(best.get("parameters", {})),
    )

    if candidate.family_id not in SYNTHETIC_FAMILIES:
        return build_audit_report({})

    result = run_candidate(candidate, include_bundle=True)
    bundle = result.bundle or {}
    return build_audit_report(bundle)
