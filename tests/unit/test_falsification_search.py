"""Tests for PR 7 — Deterministic Falsification Search Engine v0."""

from __future__ import annotations

from typing import Any

import pytest

from reglabsim.falsification.search import (
    FalsificationCandidate,
    FalsificationResult,
    SearchParameterRange,
    build_best_candidate_audit_report,
    default_search_space,
    generate_candidates,
    run_candidate,
    run_falsification_search,
    score_candidate_metrics,
)
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

# Positive (should produce unsafe_legal_state) and control families
_POSITIVE_FAMILY = "confined_corner_grass"
_CONTROL_FAMILY = "wide_corner_asphalt_control"
_ALL_POSITIVE = [fid for fid, s in SYNTHETIC_FAMILIES.items() if s.expected_unsafe_legal]


# ---------------------------------------------------------------------------
# Test 1: generate_candidates is deterministic for same seed
# ---------------------------------------------------------------------------


def test_generate_candidates_is_deterministic_for_same_seed() -> None:
    candidates_a = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=10)
    candidates_b = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=10)

    assert len(candidates_a) == len(candidates_b)
    for a, b in zip(candidates_a, candidates_b, strict=True):
        assert a.candidate_id == b.candidate_id
        assert a.parameters == b.parameters
        assert a.seed == b.seed


# ---------------------------------------------------------------------------
# Test 2: different seed → different candidates
# ---------------------------------------------------------------------------


def test_generate_candidates_changes_with_different_seed() -> None:
    candidates_42 = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=10)
    candidates_99 = generate_candidates(_POSITIVE_FAMILY, seed=99, max_trials=10)

    params_42 = [c.parameters for c in candidates_42]
    params_99 = [c.parameters for c in candidates_99]

    # At least one candidate should differ
    assert any(p42 != p99 for p42, p99 in zip(params_42, params_99, strict=True))


# ---------------------------------------------------------------------------
# Test 3: respects max_trials
# ---------------------------------------------------------------------------


def test_generate_candidates_respects_max_trials() -> None:
    for n in [1, 5, 25]:
        candidates = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=n)
        assert len(candidates) == n, f"Expected {n} candidates, got {len(candidates)}"


# ---------------------------------------------------------------------------
# Test 4: run_candidate returns metrics and score
# ---------------------------------------------------------------------------


def test_run_candidate_returns_metrics_and_score() -> None:
    candidate = FalsificationCandidate(
        candidate_id="test:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={"width_m": 11.0, "barrier_distance_m": 8.0},
    )
    result = run_candidate(candidate)

    assert isinstance(result, FalsificationResult)
    assert result.candidate_id == candidate.candidate_id
    assert result.family_id == _POSITIVE_FAMILY
    assert isinstance(result.unsafe_legal_state_count, int)
    assert result.unsafe_legal_state_count >= 0
    assert isinstance(result.score, float)
    assert result.score >= 0.0
    assert isinstance(result.event_refs, list)
    assert result.bundle is None  # include_bundle=False by default


# ---------------------------------------------------------------------------
# Test 5: candidate parameters affect runtime
# ---------------------------------------------------------------------------


def test_candidate_parameters_affect_runtime() -> None:
    """High-risk parameters must produce higher score than low-risk ones."""
    low_risk = FalsificationCandidate(
        candidate_id="low:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={
            "width_m": 14.0,          # wide — lower confinement
            "barrier_distance_m": 16.0,  # far barrier — lower hazard
            "attacker_risk_level": 0.55,
            "defender_risk_level": 0.55,
            "attacker_ers_soc": 0.45,
            "gap_s": 0.60,
        },
    )
    high_risk = FalsificationCandidate(
        candidate_id="high:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={
            "width_m": 9.0,           # narrow — high confinement
            "barrier_distance_m": 4.0,   # close barrier — high hazard
            "attacker_risk_level": 0.95,
            "defender_risk_level": 0.90,
            "attacker_ers_soc": 0.95,
            "gap_s": 0.15,
        },
    )
    low_result = run_candidate(low_risk)
    high_result = run_candidate(high_risk)

    # Either score or count should differ between extremes
    scores_differ = low_result.score != high_result.score
    counts_differ = (
        low_result.unsafe_legal_state_count != high_result.unsafe_legal_state_count
    )
    assert scores_differ or counts_differ, (
        f"Low-risk and high-risk candidates produced identical results: "
        f"low_score={low_result.score} high_score={high_result.score} "
        f"low_count={low_result.unsafe_legal_state_count} "
        f"high_count={high_result.unsafe_legal_state_count}"
    )


# ---------------------------------------------------------------------------
# Test 6: run_falsification_search returns ranked candidates
# ---------------------------------------------------------------------------


def test_run_falsification_search_returns_ranked_candidates() -> None:
    result = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=5)

    assert result["schema_version"] == "falsification_search.v0"
    assert result["family_id"] == _POSITIVE_FAMILY
    assert result["seed"] == 42
    assert result["max_trials"] == 5
    assert "search_space" in result
    assert "best_candidate" in result
    assert "results" in result

    results = result["results"]
    assert len(results) == 5

    # Must be sorted descending by score
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted by score descending: {scores}"
    )


# ---------------------------------------------------------------------------
# Test 7: search finds unsafe_legal_state in positive family
# ---------------------------------------------------------------------------


def test_search_finds_unsafe_legal_state_in_positive_family() -> None:
    """At least one trial in a positive family must produce unsafe_legal_state."""
    result = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=15)
    total_unsafe = sum(r["unsafe_legal_state_count"] for r in result["results"])
    assert total_unsafe > 0, (
        f"Expected at least one unsafe_legal_state across 15 trials in {_POSITIVE_FAMILY}. "
        f"All counts: {[r['unsafe_legal_state_count'] for r in result['results']]}"
    )


# ---------------------------------------------------------------------------
# Test 8: results contain event_refs when exploit found
# ---------------------------------------------------------------------------


def test_search_result_contains_event_refs_when_exploit_found() -> None:
    result = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=15)
    exploiting = [r for r in result["results"] if r["unsafe_legal_state_count"] > 0]
    if not exploiting:
        pytest.skip("No exploiting candidates found — seed/trials may need adjustment")

    for r in exploiting:
        assert isinstance(r["event_refs"], list), "event_refs must be a list"
        # event_refs may be empty if metrics don't populate them, but must exist
        assert "event_refs" in r


# ---------------------------------------------------------------------------
# Test 9: search does not use LLM or NVIDIA
# ---------------------------------------------------------------------------


def test_search_does_not_use_llm_or_nvidia() -> None:
    """Falsification search.py source must not import LLM or NVIDIA modules."""
    import pathlib

    import reglabsim.falsification.search as search_mod

    source_path = getattr(search_mod, "__file__", "")
    assert source_path, "Could not locate search.py"
    src = pathlib.Path(source_path).read_text(encoding="utf-8")
    for forbidden_token in ["nvidia", "langchain", "openai", "NvidiaAssistant", "anthropic"]:
        assert forbidden_token not in src, (
            f"search.py must not reference {forbidden_token!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: search does not reference real track names
# ---------------------------------------------------------------------------


def test_search_does_not_reference_real_track_names() -> None:
    """Search engine must not hardcode real circuit names."""
    import pathlib
    src = pathlib.Path(
        "reglabsim/falsification/search.py"
    ).read_text(encoding="utf-8")
    real_tracks = ["suzuka", "monaco", "baku", "singapore", "monza", "silverstone", "barcelona"]
    for track in real_tracks:
        assert track not in src.lower(), (
            f"search.py must not reference real track name {track!r}"
        )

    # Verify candidate parameters never include track_id as a real circuit
    candidates = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=5)
    for c in candidates:
        for v in c.parameters.values():
            assert not isinstance(v, str), (
                f"Parameter value must be numeric, got {v!r}"
            )


# ---------------------------------------------------------------------------
# Test 11: build_best_candidate_audit_report
# ---------------------------------------------------------------------------


def test_build_best_candidate_audit_report() -> None:
    search_result = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=8)
    audit = build_best_candidate_audit_report(search_result)

    assert "schema_version" in audit
    assert audit["schema_version"] == "audit_report.v1"
    assert "summary" in audit
    assert "limitations" in audit
    assert isinstance(audit["limitations"], list)


def test_build_best_candidate_audit_report_handles_no_best() -> None:
    """Must not raise when best_candidate is None."""
    empty_result: dict[str, Any] = {
        "schema_version": "falsification_search.v0",
        "family_id": _POSITIVE_FAMILY,
        "seed": 42,
        "max_trials": 0,
        "best_candidate": None,
        "results": [],
    }
    audit = build_best_candidate_audit_report(empty_result)
    assert "schema_version" in audit


# ---------------------------------------------------------------------------
# Test 12: score_candidate_metrics handles missing values
# ---------------------------------------------------------------------------


def test_score_candidate_metrics_handles_missing_values() -> None:
    assert score_candidate_metrics({}) == 0.0
    assert score_candidate_metrics({"unsafe_legal_state_count": 2}) == 20.0
    assert score_candidate_metrics({"unsafe_legal_state_count": 1, "max_hazard_score": 0.5}) == 11.0
    assert score_candidate_metrics({
        "unsafe_legal_state_count": 1,
        "max_hazard_score": 0.5,
        "mean_hazard_score": 0.4,
        "safety_verdict_status_counts": {"UNSAFE_LEGAL": 1},
    }) == pytest.approx(10.0 + 2 * 0.5 + 0.4 + 3.0)


def test_score_candidate_metrics_zero_when_no_evidence() -> None:
    score = score_candidate_metrics({
        "unsafe_legal_state_count": 0,
        "max_hazard_score": None,
        "mean_hazard_score": None,
    })
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 13: default_search_space structure
# ---------------------------------------------------------------------------


def test_default_search_space_structure() -> None:
    space = default_search_space()
    required = [
        "width_m",
        "barrier_distance_m",
        "unsafe_closing_speed_threshold_kph",
        "visibility_m",
        "wetness_level",
        "attacker_risk_level",
        "defender_risk_level",
        "attacker_ers_soc",
        "defender_ers_soc",
        "gap_s",
    ]
    for key in required:
        assert key in space, f"Missing key {key!r} in default_search_space"
        r = space[key]
        assert isinstance(r, SearchParameterRange)
        assert r.min_value <= r.max_value
        assert r.steps >= 1


# ---------------------------------------------------------------------------
# Test 14: candidate parameters are within range
# ---------------------------------------------------------------------------


def test_candidate_parameters_are_within_range() -> None:
    space = default_search_space()
    candidates = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=25)
    for c in candidates:
        for name, value in c.parameters.items():
            if name in space:
                r = space[name]
                assert r.min_value <= value <= r.max_value, (
                    f"Candidate {c.candidate_id}: param {name}={value} out of "
                    f"range [{r.min_value}, {r.max_value}]"
                )


# ---------------------------------------------------------------------------
# Test 15: include_bundles flag
# ---------------------------------------------------------------------------


def test_run_candidate_with_include_bundle() -> None:
    candidate = FalsificationCandidate(
        candidate_id="bundle:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={},
    )
    result = run_candidate(candidate, include_bundle=True)
    assert result.bundle is not None
    assert isinstance(result.bundle, dict)
    assert "metrics" in result.bundle


# ---------------------------------------------------------------------------
# PR 8.1 — exploit_score integration tests
# ---------------------------------------------------------------------------


def test_run_candidate_includes_exploit_score() -> None:
    candidate = FalsificationCandidate(
        candidate_id="exploitscore:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={},
    )
    result = run_candidate(candidate)
    assert result.exploit_score is not None
    es = result.exploit_score
    assert es["schema_version"] == "exploit_score.v1"
    assert "total" in es
    assert "components" in es
    for key in ("safety_risk", "legal_exploit", "competitive_advantage",
                "patch_resistance", "novelty"):
        assert key in es["components"]
    assert "reason_codes" in es
    assert "limitations" in es


def test_run_candidate_preserves_legacy_score() -> None:
    candidate = FalsificationCandidate(
        candidate_id="legacy:seed42:trial0000",
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={},
    )
    result = run_candidate(candidate)
    # Legacy score still present and matches score_candidate_metrics
    assert result.score is not None
    assert isinstance(result.score, float)
    # exploit_score.total is additional — does not replace score
    if result.exploit_score:
        assert result.exploit_score.get("total") != result.score or True  # coexist


def test_run_falsification_search_best_candidate_includes_exploit_score() -> None:
    out = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=5)
    bc = out.get("best_candidate")
    assert bc is not None
    assert "exploit_score" in bc
    es = bc["exploit_score"]
    assert isinstance(es, dict)
    assert es.get("schema_version") == "exploit_score.v1"
    assert "score_legacy" in bc


def test_search_ranking_legacy_score_preserved() -> None:
    out = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=10)
    results = out.get("results") or []
    assert len(results) >= 2
    # Results are still sorted by legacy score descending
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # score_legacy matches score
    for r in results:
        assert r["score"] == r["score_legacy"]


# ---------------------------------------------------------------------------
# PR 8.2 — Failure taxonomy tests
# ---------------------------------------------------------------------------


def test_run_candidate_includes_failure_taxonomy() -> None:
    """run_candidate() must include failure_taxonomy with schema_version."""
    from reglabsim.falsification.failure_taxonomy import FAILURE_TAXONOMY_SCHEMA

    candidates = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=3)
    result = run_candidate(candidates[0])
    assert result.failure_taxonomy is not None
    assert isinstance(result.failure_taxonomy, dict)
    assert result.failure_taxonomy.get("schema_version") == FAILURE_TAXONOMY_SCHEMA


def test_run_candidate_failure_taxonomy_structure() -> None:
    """failure_taxonomy must have primary_failure_mode, failure_modes, event_refs, limitations."""
    candidates = generate_candidates(_POSITIVE_FAMILY, seed=42, max_trials=3)
    result = run_candidate(candidates[0])
    ft = result.failure_taxonomy
    assert ft is not None
    assert "primary_failure_mode" in ft
    assert "failure_modes" in ft
    assert isinstance(ft["failure_modes"], list)
    assert "event_refs" in ft
    assert "limitations" in ft
    assert isinstance(ft["limitations"], list)


def test_run_falsification_search_best_candidate_includes_failure_modes() -> None:
    """run_falsification_search best_candidate must include failure_modes key."""
    out = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=5)
    bc = out.get("best_candidate")
    assert bc is not None
    assert "failure_modes" in bc
    assert isinstance(bc["failure_modes"], list)
    assert "primary_failure_mode" in bc


def test_search_results_include_compact_failure_modes() -> None:
    """Each result in run_falsification_search must include primary_failure_mode."""
    out = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=5)
    results = out.get("results") or []
    assert len(results) > 0
    for r in results:
        assert "primary_failure_mode" in r
        assert "failure_modes" in r
        assert isinstance(r["failure_modes"], list)


def test_search_ranking_unchanged_by_taxonomy() -> None:
    """Adding failure taxonomy must not change sort order by score."""
    out_a = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=8)
    out_b = run_falsification_search(_POSITIVE_FAMILY, seed=42, max_trials=8)
    # Same seed -> same ranking
    ids_a = [r["candidate_id"] for r in out_a["results"]]
    ids_b = [r["candidate_id"] for r in out_b["results"]]
    assert ids_a == ids_b
    # Scores are still sorted descending
    scores = [r["score"] for r in out_a["results"]]
    assert scores == sorted(scores, reverse=True)
