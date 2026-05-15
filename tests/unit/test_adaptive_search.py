"""Tests for PR 8 — Adaptive Mutation Loop (adaptive_search.py)."""

from __future__ import annotations

import json

import pytest

from reglabsim.falsification.adaptive_search import (
    ADAPTIVE_SEARCH_SCHEMA,
    AdaptiveRoundSummary,
    MutationConfig,
    clamp,
    compute_improvement_trace,
    generate_adaptive_round_candidates,
    mutate_candidate_parameters,
    mutate_parameter_value,
    parameter_range_width,
    rank_falsification_results,
    run_adaptive_falsification_search,
)
from reglabsim.falsification.search import (
    FalsificationCandidate,
    FalsificationResult,
    SearchParameterRange,
    default_search_space,
)

_POSITIVE_FAMILY = "confined_corner_grass"
_CONTROL_FAMILY = "wide_corner_asphalt_control"

_FORBIDDEN_OUTPUT_KEYS = [
    "event_log",
    "state_snapshots",
    "raw_event",
    "full_bundle",
    "NVIDIA_API_KEY",
    "api_key",
    "password",
    "token",
]

_REAL_TRACK_NAMES = [
    "suzuka", "monaco", "baku", "singapore", "monza", "silverstone", "barcelona",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    candidate_id: str,
    score: float,
    unsafe_count: int = 0,
    max_hazard: float | None = None,
    mean_hazard: float | None = None,
) -> FalsificationResult:
    return FalsificationResult(
        candidate_id=candidate_id,
        family_id=_POSITIVE_FAMILY,
        seed=42,
        parameters={"width_m": 10.0},
        unsafe_legal_state_count=unsafe_count,
        max_hazard_score=max_hazard,
        mean_hazard_score=mean_hazard,
        score=score,
        event_refs=[],
    )


# ---------------------------------------------------------------------------
# MutationConfig validation
# ---------------------------------------------------------------------------


class TestMutationConfigDefaults:
    def test_mutation_config_defaults_are_safe(self) -> None:
        cfg = MutationConfig()
        assert cfg.rounds <= 3
        assert cfg.candidates_per_round <= 10
        assert cfg.elite_count <= cfg.candidates_per_round
        assert cfg.exploit_bias == 0.70
        assert cfg.mutation_scale == 0.35
        assert cfg.min_mutation_scale == 0.08
        assert cfg.seed == 42

    def test_mutation_config_rounds_default(self) -> None:
        assert MutationConfig().rounds == 3

    def test_mutation_config_candidates_per_round_default(self) -> None:
        assert MutationConfig().candidates_per_round == 10

    def test_mutation_config_elite_count_default(self) -> None:
        assert MutationConfig().elite_count == 3


class TestMutationConfigValidation:
    def test_rejects_rounds_zero(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            MutationConfig(rounds=0)

    def test_rejects_candidates_per_round_zero(self) -> None:
        with pytest.raises(ValueError, match="candidates_per_round"):
            MutationConfig(candidates_per_round=0)

    def test_rejects_elite_count_zero(self) -> None:
        with pytest.raises(ValueError, match="elite_count"):
            MutationConfig(elite_count=0)

    def test_rejects_elite_count_greater_than_candidates_per_round(self) -> None:
        with pytest.raises(ValueError, match="elite_count"):
            MutationConfig(elite_count=11, candidates_per_round=10)

    def test_rejects_exploit_bias_above_one(self) -> None:
        with pytest.raises(ValueError, match="exploit_bias"):
            MutationConfig(exploit_bias=1.01)

    def test_rejects_exploit_bias_below_zero(self) -> None:
        with pytest.raises(ValueError, match="exploit_bias"):
            MutationConfig(exploit_bias=-0.01)

    def test_rejects_mutation_scale_zero(self) -> None:
        with pytest.raises(ValueError, match="mutation_scale"):
            MutationConfig(mutation_scale=0.0)

    def test_rejects_min_mutation_scale_zero(self) -> None:
        with pytest.raises(ValueError, match="min_mutation_scale"):
            MutationConfig(min_mutation_scale=0.0)

    def test_rejects_min_mutation_scale_greater_than_mutation_scale(self) -> None:
        with pytest.raises(ValueError, match="min_mutation_scale"):
            MutationConfig(mutation_scale=0.1, min_mutation_scale=0.5)

    def test_accepts_exploit_bias_boundary_zero(self) -> None:
        cfg = MutationConfig(exploit_bias=0.0)
        assert cfg.exploit_bias == 0.0

    def test_accepts_exploit_bias_boundary_one(self) -> None:
        cfg = MutationConfig(exploit_bias=1.0)
        assert cfg.exploit_bias == 1.0

    def test_accepts_equal_elite_and_candidates(self) -> None:
        cfg = MutationConfig(elite_count=5, candidates_per_round=5)
        assert cfg.elite_count == cfg.candidates_per_round


# ---------------------------------------------------------------------------
# Mutation utilities
# ---------------------------------------------------------------------------


class TestClamp:
    def test_clamps_below_min(self) -> None:
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_clamps_above_max(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_passes_through_in_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamps_at_exact_min(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_clamps_at_exact_max(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestParameterRangeWidth:
    def test_width_calculation(self) -> None:
        r = SearchParameterRange("x", 5.0, 15.0, 3)
        assert parameter_range_width(r) == pytest.approx(10.0)

    def test_zero_width(self) -> None:
        r = SearchParameterRange("x", 5.0, 5.0, 1)
        assert parameter_range_width(r) == 0.0


class TestMutateParameterValue:
    def test_stays_within_range(self) -> None:
        import random
        rng = random.Random(42)
        param_range = SearchParameterRange("width_m", 9.0, 14.0, 3)
        for _ in range(50):
            value = 9.0  # near lower edge
            mutated = mutate_parameter_value(value, param_range, rng, 0.35)
            assert param_range.min_value <= mutated <= param_range.max_value

    def test_clamps_when_near_upper_edge(self) -> None:
        import random
        rng = random.Random(99)
        param_range = SearchParameterRange("width_m", 9.0, 14.0, 3)
        for _ in range(50):
            value = 14.0  # upper edge
            mutated = mutate_parameter_value(value, param_range, rng, 0.35)
            assert mutated <= param_range.max_value

    def test_rounds_to_4_decimal_places(self) -> None:
        import random
        rng = random.Random(7)
        param_range = SearchParameterRange("x", 0.0, 1.0, 3)
        value = mutate_parameter_value(0.5, param_range, rng, 0.1)
        assert value == round(value, 4)


class TestMutateCandidateParameters:
    def test_preserves_known_keys_only(self) -> None:
        space = default_search_space()
        parent = FalsificationCandidate(
            candidate_id="test:parent",
            family_id=_POSITIVE_FAMILY,
            seed=42,
            parameters={name: r.min_value for name, r in space.items()},
        )
        import random
        rng = random.Random(42)
        result = mutate_candidate_parameters(parent, space, rng, 0.35)
        assert set(result.keys()) == set(space.keys())

    def test_no_track_id_in_output(self) -> None:
        space = default_search_space()
        parent = FalsificationCandidate(
            candidate_id="test:parent",
            family_id=_POSITIVE_FAMILY,
            seed=42,
            parameters={name: r.min_value for name, r in space.items()},
        )
        import random
        rng = random.Random(42)
        result = mutate_candidate_parameters(parent, space, rng, 0.35)
        assert "track_id" not in result
        for v in result.values():
            assert isinstance(v, float), f"Expected float, got {type(v)}"

    def test_all_values_in_range(self) -> None:
        space = default_search_space()
        parent = FalsificationCandidate(
            candidate_id="test:parent",
            family_id=_POSITIVE_FAMILY,
            seed=42,
            parameters={name: r.min_value for name, r in space.items()},
        )
        import random
        rng = random.Random(42)
        for _ in range(10):
            result = mutate_candidate_parameters(parent, space, rng, 0.35)
            for name, val in result.items():
                r = space[name]
                assert r.min_value <= val <= r.max_value, (
                    f"param {name}={val} out of [{r.min_value}, {r.max_value}]"
                )


# ---------------------------------------------------------------------------
# generate_adaptive_round_candidates
# ---------------------------------------------------------------------------


class TestGenerateAdaptiveRoundCandidates:
    def test_is_deterministic_for_same_seed(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=5)
        prev: list[FalsificationResult] = []
        cands_a = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        cands_b = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        assert len(cands_a) == len(cands_b)
        for a, b in zip(cands_a, cands_b, strict=True):
            assert a.candidate_id == b.candidate_id
            assert a.parameters == b.parameters

    def test_changes_with_different_seed(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=8)
        prev: list[FalsificationResult] = []
        cands_42 = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        cands_99 = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 99, prev, config
        )
        params_42 = [c.parameters for c in cands_42]
        params_99 = [c.parameters for c in cands_99]
        assert any(p42 != p99 for p42, p99 in zip(params_42, params_99, strict=True))

    def test_round_zero_uses_broad_sampling(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=7)
        prev: list[FalsificationResult] = []
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        assert len(cands) == 7
        # All should have adaptive ID format
        for c in cands:
            assert "adaptive_seed42" in c.candidate_id
            assert ":round00:" in c.candidate_id

    def test_round_zero_has_no_parent_concept(self) -> None:
        """Round 0 is broad — no previous results."""
        config = MutationConfig(rounds=3, candidates_per_round=5)
        prev: list[FalsificationResult] = []
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        assert len(cands) == config.candidates_per_round

    def test_later_round_uses_elite_parents(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=5, elite_count=2)
        # Create previous results with distinct scores
        prev = [
            _make_result(f"parent:{i:04d}", score=float(i * 5)) for i in range(5)
        ]
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 1, 42, prev, config
        )
        assert len(cands) == config.candidates_per_round
        # All IDs should use round01
        for c in cands:
            assert ":round01:" in c.candidate_id

    def test_candidate_count_equals_candidates_per_round(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=12)
        prev: list[FalsificationResult] = []
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 0, 42, prev, config
        )
        assert len(cands) == 12

    def test_candidate_id_format(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=3)
        prev: list[FalsificationResult] = []
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 2, 42, prev, config
        )
        for i, c in enumerate(cands):
            expected_id = (
                f"{_POSITIVE_FAMILY}:adaptive_seed42:round02:trial{i:04d}"
            )
            assert c.candidate_id == expected_id

    def test_parameters_all_within_range(self) -> None:
        space = default_search_space()
        config = MutationConfig(rounds=3, candidates_per_round=8)
        prev = [_make_result(f"p:{i}", score=float(i)) for i in range(5)]
        cands = generate_adaptive_round_candidates(
            _POSITIVE_FAMILY, 1, 42, prev, config, search_space=space
        )
        for c in cands:
            for name, val in c.parameters.items():
                if name in space:
                    r = space[name]
                    assert r.min_value <= val <= r.max_value, (
                        f"{name}={val} out of [{r.min_value}, {r.max_value}]"
                    )


# ---------------------------------------------------------------------------
# rank_falsification_results
# ---------------------------------------------------------------------------


class TestRankFalsificationResults:
    def test_orders_by_score_descending(self) -> None:
        results = [
            _make_result("a", score=5.0),
            _make_result("b", score=15.0),
            _make_result("c", score=10.0),
        ]
        ranked = rank_falsification_results(results)
        scores = [r.score for r in ranked]
        assert scores == [15.0, 10.0, 5.0]

    def test_tiebreak_by_unsafe_count(self) -> None:
        results = [
            _make_result("a", score=10.0, unsafe_count=1),
            _make_result("b", score=10.0, unsafe_count=2),
        ]
        ranked = rank_falsification_results(results)
        assert ranked[0].candidate_id == "b"

    def test_tiebreak_by_max_hazard(self) -> None:
        results = [
            _make_result("a", score=10.0, unsafe_count=1, max_hazard=0.5),
            _make_result("b", score=10.0, unsafe_count=1, max_hazard=0.9),
        ]
        ranked = rank_falsification_results(results)
        assert ranked[0].candidate_id == "b"

    def test_tiebreak_by_candidate_id_ascending(self) -> None:
        results = [
            _make_result("z_cand", score=10.0),
            _make_result("a_cand", score=10.0),
        ]
        ranked = rank_falsification_results(results)
        assert ranked[0].candidate_id == "a_cand"

    def test_handles_none_hazard(self) -> None:
        results = [
            _make_result("a", score=5.0, max_hazard=None),
            _make_result("b", score=10.0, max_hazard=None),
        ]
        ranked = rank_falsification_results(results)
        assert ranked[0].score == 10.0

    def test_empty_list_returns_empty(self) -> None:
        assert rank_falsification_results([]) == []


# ---------------------------------------------------------------------------
# compute_improvement_trace
# ---------------------------------------------------------------------------


class TestComputeImprovementTrace:
    def _make_round(
        self, idx: int, score: float | None
    ) -> AdaptiveRoundSummary:
        return AdaptiveRoundSummary(
            round_index=idx,
            parent_candidate_ids=[],
            evaluated_count=5,
            best_candidate_id=f"cand:{idx}",
            best_score=score,
            best_unsafe_legal_state_count=0,
            best_event_refs=[],
            improvement_over_previous_best=None,
        )

    def test_round_zero_delta_is_none(self) -> None:
        rounds = [self._make_round(0, 15.5)]
        trace = compute_improvement_trace(rounds)
        assert trace[0]["delta"] is None
        assert trace[0]["best_score"] == 15.5

    def test_improvement_recorded_when_better(self) -> None:
        rounds = [
            self._make_round(0, 15.5),
            self._make_round(1, 16.0),
        ]
        trace = compute_improvement_trace(rounds)
        assert trace[1]["delta"] == pytest.approx(0.5)

    def test_no_improvement_recorded_honestly(self) -> None:
        rounds = [
            self._make_round(0, 16.0),
            self._make_round(1, 15.0),
        ]
        trace = compute_improvement_trace(rounds)
        assert trace[1]["delta"] == pytest.approx(-1.0)

    def test_zero_improvement_recorded(self) -> None:
        rounds = [
            self._make_round(0, 10.0),
            self._make_round(1, 10.0),
        ]
        trace = compute_improvement_trace(rounds)
        assert trace[1]["delta"] == pytest.approx(0.0)

    def test_none_score_gives_none_delta(self) -> None:
        rounds = [
            self._make_round(0, None),
            self._make_round(1, None),
        ]
        trace = compute_improvement_trace(rounds)
        assert trace[0]["delta"] is None
        assert trace[1]["delta"] is None

    def test_trace_length_equals_round_count(self) -> None:
        rounds = [self._make_round(i, float(i * 5)) for i in range(4)]
        trace = compute_improvement_trace(rounds)
        assert len(trace) == 4

    def test_global_best_used_not_previous_round(self) -> None:
        """Delta compares against global best across all previous rounds."""
        rounds = [
            self._make_round(0, 20.0),
            self._make_round(1, 15.0),  # worse than round 0
            self._make_round(2, 18.0),  # better than round 1 but worse than round 0
        ]
        trace = compute_improvement_trace(rounds)
        # round 2 delta should compare vs global best (20.0), not round 1's (15.0)
        assert trace[2]["delta"] == pytest.approx(18.0 - 20.0)


# ---------------------------------------------------------------------------
# run_adaptive_falsification_search
# ---------------------------------------------------------------------------


class TestRunAdaptiveFalsificationSearch:
    def test_returns_correct_schema(self) -> None:
        result = run_adaptive_falsification_search(
            _POSITIVE_FAMILY, seed=42,
            config=MutationConfig(rounds=2, candidates_per_round=5),
        )
        assert result["schema_version"] == ADAPTIVE_SEARCH_SCHEMA

    def test_rounds_present_and_correct_count(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert len(result["rounds"]) == 3

    def test_best_candidate_exists_or_none(self) -> None:
        result = run_adaptive_falsification_search(
            _POSITIVE_FAMILY, seed=42,
            config=MutationConfig(rounds=2, candidates_per_round=5),
        )
        bc = result["best_candidate"]
        assert bc is None or isinstance(bc, dict)
        if bc is not None:
            assert "candidate_id" in bc
            assert "score" in bc

    def test_total_evaluations_correct(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=7)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert result["total_evaluations"] == 3 * 7

    def test_is_deterministic_for_same_seed(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        r1 = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        r2 = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert r1["best_candidate"] == r2["best_candidate"]
        assert r1["total_evaluations"] == r2["total_evaluations"]
        assert len(r1["rounds"]) == len(r2["rounds"])

    def test_results_sorted_by_score_descending(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=8)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_finds_unsafe_legal_in_positive_family(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=10)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        bc = result["best_candidate"]
        assert bc is not None
        assert bc["unsafe_legal_state_count"] >= 1, (
            f"Expected ≥1 unsafe legal state in positive family. "
            f"best_candidate={bc}"
        )

    def test_improvement_trace_present_and_round_aligned(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        trace = result["improvement_trace"]
        assert len(trace) == 3
        for i, entry in enumerate(trace):
            assert entry["round_index"] == i
            assert "best_score" in entry
            assert "delta" in entry

    def test_improvement_trace_round_zero_delta_is_none(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert result["improvement_trace"][0]["delta"] is None

    def test_rejects_unknown_family(self) -> None:
        with pytest.raises(ValueError, match="Unknown family_id"):
            run_adaptive_falsification_search("nonexistent_family", seed=42)

    def test_limitations_present(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert isinstance(result["limitations"], list)
        assert len(result["limitations"]) >= 3
        lims_text = " ".join(result["limitations"]).lower()
        assert "deterministic" in lims_text or "heuristic" in lims_text

    def test_output_is_compact_no_raw_logs(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(
            _POSITIVE_FAMILY, seed=42, config=config, include_bundles=False
        )
        text = json.dumps(result, sort_keys=True)
        for forbidden in _FORBIDDEN_OUTPUT_KEYS:
            assert forbidden not in text, (
                f"Forbidden key {forbidden!r} found in adaptive search output"
            )

    def test_include_bundles_false_excludes_bundles(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(
            _POSITIVE_FAMILY, seed=42, config=config, include_bundles=False
        )
        text = json.dumps(result, sort_keys=True)
        assert "full_bundle" not in text
        # Results should not contain bundle key
        for r in result["results"]:
            assert "bundle" not in r

    def test_does_not_reference_real_track_names(self) -> None:
        import pathlib
        src_path = pathlib.Path(
            "reglabsim/falsification/adaptive_search.py"
        ).read_text(encoding="utf-8")
        for track in _REAL_TRACK_NAMES:
            assert track not in src_path.lower(), (
                f"adaptive_search.py must not reference real track name {track!r}"
            )

    def test_mutation_config_dict_in_output(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5, elite_count=2)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        mc = result["mutation_config"]
        assert mc["rounds"] == 2
        assert mc["candidates_per_round"] == 5
        assert mc["elite_count"] == 2

    def test_search_space_in_output(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert "search_space" in result
        space = result["search_space"]
        assert "width_m" in space

    def test_rounds_have_required_fields(self) -> None:
        config = MutationConfig(rounds=3, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        required = {
            "round_index", "parent_candidate_ids", "evaluated_count",
            "best_candidate_id", "best_score", "best_unsafe_legal_state_count",
            "best_event_refs", "improvement_over_previous_best",
        }
        for rnd in result["rounds"]:
            for field in required:
                assert field in rnd, f"Missing field {field!r} in round {rnd['round_index']}"

    def test_results_contain_no_bundles_by_default(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        for r in result["results"]:
            assert "bundle" not in r
            assert "event_log" not in r

    def test_family_id_in_output(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        assert result["family_id"] == _POSITIVE_FAMILY

    def test_seed_in_output(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=77, config=config)
        assert result["seed"] == 77

    def test_output_is_json_serializable(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        encoded = json.dumps(result, sort_keys=True)
        decoded = json.loads(encoded)
        assert decoded["schema_version"] == ADAPTIVE_SEARCH_SCHEMA

    def test_control_family_has_no_unsafe_legal(self) -> None:
        """Control family should not produce unsafe legal states."""
        config = MutationConfig(rounds=2, candidates_per_round=8)
        result = run_adaptive_falsification_search(_CONTROL_FAMILY, seed=42, config=config)
        total_unsafe = sum(r["unsafe_legal_state_count"] for r in result["results"])
        assert total_unsafe == 0, (
            f"Control family should not produce unsafe legal states, "
            f"got {total_unsafe}"
        )


# ---------------------------------------------------------------------------
# PR 8.1 — exploit_score integration tests for adaptive search
# ---------------------------------------------------------------------------


class TestAdaptiveExploitScore:
    """Tests that exploit_score is propagated through adaptive search output."""

    def test_adaptive_best_candidate_includes_exploit_score(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        bc = result.get("best_candidate")
        assert bc is not None
        assert "exploit_score" in bc
        es = bc["exploit_score"]
        assert isinstance(es, dict)
        assert es.get("schema_version") == "exploit_score.v1"
        assert "total" in es
        assert "components" in es

    def test_adaptive_top_results_include_exploit_score_summary(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        results = result.get("results") or []
        assert len(results) > 0
        for r in results[:3]:
            assert "exploit_score" in r
            assert "exploit_score_total" in r
            assert "exploit_score_components" in r

    def test_adaptive_ranking_remains_legacy_by_default(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=8)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        results = result.get("results") or []
        assert len(results) >= 2
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_adaptive_output_score_legacy_matches_score(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        bc = result.get("best_candidate")
        assert bc is not None
        assert bc["score"] == bc["score_legacy"]

    def test_adaptive_output_no_raw_logs_with_exploit_score(self) -> None:
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized


# ---------------------------------------------------------------------------
# PR 8.2 — Failure taxonomy tests for adaptive search
# ---------------------------------------------------------------------------


class TestAdaptiveSearchFailureTaxonomy:
    """Tests that adaptive search includes failure taxonomy fields."""

    def test_adaptive_best_candidate_includes_failure_taxonomy(self) -> None:
        """best_candidate in adaptive output must include failure_taxonomy dict."""
        from reglabsim.falsification.failure_taxonomy import FAILURE_TAXONOMY_SCHEMA

        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        bc = result.get("best_candidate")
        assert bc is not None
        assert "failure_taxonomy" in bc
        ft = bc["failure_taxonomy"]
        assert isinstance(ft, dict)
        assert ft.get("schema_version") == FAILURE_TAXONOMY_SCHEMA

    def test_adaptive_best_candidate_includes_primary_failure_mode(self) -> None:
        """best_candidate must include primary_failure_mode and failure_modes."""
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        bc = result.get("best_candidate")
        assert bc is not None
        assert "primary_failure_mode" in bc
        assert "failure_modes" in bc
        assert isinstance(bc["failure_modes"], list)

    def test_adaptive_top_results_include_failure_modes_compactly(self) -> None:
        """Each result in the results list must include primary_failure_mode."""
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        results = result.get("results") or []
        assert len(results) > 0
        for r in results:
            assert "primary_failure_mode" in r
            assert "failure_modes" in r
            assert isinstance(r["failure_modes"], list)

    def test_adaptive_ranking_unchanged_by_taxonomy(self) -> None:
        """Failure taxonomy must not change sort order by score."""
        config = MutationConfig(rounds=2, candidates_per_round=8)
        result_a = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        result_b = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        ids_a = [r["candidate_id"] for r in result_a["results"]]
        ids_b = [r["candidate_id"] for r in result_b["results"]]
        assert ids_a == ids_b
        scores = [r["score"] for r in result_a["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_adaptive_output_no_raw_event_details_from_taxonomy(self) -> None:
        """Taxonomy output must not contain raw event log or full bundle keys."""
        config = MutationConfig(rounds=2, candidates_per_round=5)
        result = run_adaptive_falsification_search(_POSITIVE_FAMILY, seed=42, config=config)
        serialized = json.dumps(result)
        for forbidden in ("event_log", "raw_event_log", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized
