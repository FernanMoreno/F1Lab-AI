"""Adaptive mutation loop for deterministic falsification search (PR 8).

Upgrades the one-shot seeded search into a multi-round adaptive mutation
loop: run initial search → identify elite candidates → mutate parameters
around elites → rerun → rank → store rounds in campaign trace.

Key invariants:
- Fully deterministic: same seed + same inputs → same candidates.
- No LLM, no NVIDIA, no external services.
- Mutation proposes candidates; the deterministic runtime evaluates them.
- Safety/legal status determined solely by SafetyOracle/LegalVerdict.
- Output is compact and JSON-serializable; no raw event logs or bundles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from reglabsim.falsification.search import (
    FalsificationCandidate,
    FalsificationResult,
    SearchParameterRange,
    default_search_space,
    run_candidate,
)
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAPTIVE_SEARCH_SCHEMA = "adaptive_falsification_search.v0"
MAX_ADAPTIVE_ROUNDS_DEFAULT = 3
MAX_CANDIDATES_PER_ROUND_DEFAULT = 10
MAX_TOTAL_EVALUATIONS_DEFAULT = 50

_ADAPTIVE_LIMITATIONS = [
    "Adaptive search is deterministic and synthetic-family based.",
    "Mutation strategy is a heuristic search policy, not a learned model.",
    "Runtime evidence remains determined by deterministic simulation/oracles.",
    "Mutation around a high-scoring candidate may overfit the synthetic family.",
    "Search is single-lap / synthetic-family scoped unless integrated with campaign runner.",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationConfig:
    """Configuration for the adaptive mutation loop.

    Attributes:
        rounds: Total number of rounds (including initial round 0).
        candidates_per_round: Candidates evaluated per round.
        elite_count: Top candidates used as mutation parents each round.
        mutation_scale: Fraction of parameter range used as mutation radius.
        min_mutation_scale: Lower bound for mutation scale.
        exploit_bias: Probability of mutating around elites vs broad sampling.
        seed: Deterministic RNG seed.
    """

    rounds: int = 3
    candidates_per_round: int = 10
    elite_count: int = 3
    mutation_scale: float = 0.35
    min_mutation_scale: float = 0.08
    exploit_bias: float = 0.70
    seed: int = 42

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {self.rounds}")
        if self.candidates_per_round <= 0:
            raise ValueError(
                f"candidates_per_round must be > 0, got {self.candidates_per_round}"
            )
        if self.elite_count <= 0:
            raise ValueError(f"elite_count must be > 0, got {self.elite_count}")
        if self.elite_count > self.candidates_per_round:
            raise ValueError(
                f"elite_count ({self.elite_count}) must be <= "
                f"candidates_per_round ({self.candidates_per_round})"
            )
        if not (0.0 <= self.exploit_bias <= 1.0):
            raise ValueError(
                f"exploit_bias must be in [0.0, 1.0], got {self.exploit_bias}"
            )
        if self.mutation_scale <= 0:
            raise ValueError(
                f"mutation_scale must be > 0, got {self.mutation_scale}"
            )
        if self.min_mutation_scale <= 0:
            raise ValueError(
                f"min_mutation_scale must be > 0, got {self.min_mutation_scale}"
            )
        if self.min_mutation_scale > self.mutation_scale:
            raise ValueError(
                f"min_mutation_scale ({self.min_mutation_scale}) must be <= "
                f"mutation_scale ({self.mutation_scale})"
            )


@dataclass(frozen=True)
class AdaptiveRoundSummary:
    """Summary of one adaptive search round.

    Attributes:
        round_index: Zero-based round index.
        parent_candidate_ids: IDs of elite candidates used as mutation parents.
        evaluated_count: Number of candidates evaluated this round.
        best_candidate_id: ID of the best candidate this round, or None.
        best_score: Score of the best candidate this round, or None.
        best_unsafe_legal_state_count: Unsafe legal state count for best candidate.
        best_event_refs: Event refs from best candidate this round.
        improvement_over_previous_best: Score delta vs previous global best, or None.
    """

    round_index: int
    parent_candidate_ids: list[str]
    evaluated_count: int
    best_candidate_id: str | None
    best_score: float | None
    best_unsafe_legal_state_count: int
    best_event_refs: list[str]
    improvement_over_previous_best: float | None


@dataclass(frozen=True)
class AdaptiveSearchResult:
    """Result of the full adaptive falsification search.

    Attributes:
        schema_version: Schema identifier.
        family_id: Synthetic family searched.
        seed: RNG seed used.
        mutation_config: Config dict (serialized MutationConfig).
        search_space: Parameter ranges dict.
        rounds: Per-round summaries.
        best_candidate: Best candidate found across all rounds, or None.
        results: All results sorted by score descending.
        total_evaluations: Total candidates evaluated.
        improvement_trace: Per-round improvement trace.
        limitations: Disclaimers.
    """

    schema_version: str
    family_id: str
    seed: int
    mutation_config: dict[str, Any]
    search_space: dict[str, Any]
    rounds: list[AdaptiveRoundSummary]
    best_candidate: dict[str, Any] | None
    results: list[dict[str, Any]]
    total_evaluations: int
    improvement_trace: list[dict[str, Any]]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Parameter mutation utilities
# ---------------------------------------------------------------------------


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value to [min_value, max_value]."""
    return max(min_value, min(max_value, value))


def parameter_range_width(param_range: SearchParameterRange) -> float:
    """Return the width of a parameter range."""
    return param_range.max_value - param_range.min_value


def mutate_parameter_value(
    value: float,
    param_range: SearchParameterRange,
    rng: random.Random,
    mutation_scale: float,
) -> float:
    """Mutate a single parameter value within its range.

    Adds a uniform perturbation of ±(width * mutation_scale) then clamps.
    """
    width = parameter_range_width(param_range)
    radius = width * mutation_scale
    delta = rng.uniform(-radius, radius)
    mutated = value + delta
    clamped = clamp(mutated, param_range.min_value, param_range.max_value)
    return round(clamped, 4)


def mutate_candidate_parameters(
    parent: FalsificationCandidate,
    search_space: dict[str, SearchParameterRange],
    rng: random.Random,
    mutation_scale: float,
) -> dict[str, float]:
    """Mutate all known numeric parameters around a parent candidate.

    Only keys in search_space are mutated and included.
    All values are clamped to their ranges and rounded to 4 decimal places.
    Never introduces track_id or non-numeric values.
    """
    result: dict[str, float] = {}
    for name in sorted(search_space.keys()):
        param_range = search_space[name]
        parent_value = parent.parameters.get(name, param_range.min_value)
        result[name] = mutate_parameter_value(
            parent_value, param_range, rng, mutation_scale
        )
    return result


def _broad_sample_parameters(
    search_space: dict[str, SearchParameterRange],
    rng: random.Random,
) -> dict[str, float]:
    """Sample parameters uniformly at random within each range."""
    result: dict[str, float] = {}
    for name in sorted(search_space.keys()):
        param_range = search_space[name]
        if param_range.max_value <= param_range.min_value:
            result[name] = param_range.min_value
        else:
            raw = rng.random()
            result[name] = round(
                param_range.min_value
                + raw * parameter_range_width(param_range),
                4,
            )
    return result


# ---------------------------------------------------------------------------
# Elite selection and ranking
# ---------------------------------------------------------------------------


def _result_sort_key(r: FalsificationResult) -> tuple[float, int, float, float, str]:
    return (
        r.score,
        r.unsafe_legal_state_count,
        float(r.max_hazard_score) if r.max_hazard_score is not None else 0.0,
        float(r.mean_hazard_score) if r.mean_hazard_score is not None else 0.0,
        r.candidate_id,  # ascending for tie-break → invert with negative trick below
    )


def rank_falsification_results(
    results: list[FalsificationResult],
) -> list[FalsificationResult]:
    """Sort results by score desc, counts/hazards desc, candidate_id asc."""
    return sorted(
        results,
        key=lambda r: (
            -r.score,
            -r.unsafe_legal_state_count,
            -(float(r.max_hazard_score) if r.max_hazard_score is not None else 0.0),
            -(float(r.mean_hazard_score) if r.mean_hazard_score is not None else 0.0),
            r.candidate_id,
        ),
    )


def _select_elites(
    previous_results: list[FalsificationResult],
    elite_count: int,
) -> list[FalsificationResult]:
    """Return top elite_count results ranked by score."""
    ranked = rank_falsification_results(previous_results)
    return ranked[:elite_count]


# ---------------------------------------------------------------------------
# Adaptive candidate generation
# ---------------------------------------------------------------------------


def generate_adaptive_round_candidates(
    family_id: str,
    round_index: int,
    seed: int,
    previous_results: list[FalsificationResult],
    config: MutationConfig,
    search_space: dict[str, SearchParameterRange] | None = None,
) -> list[FalsificationCandidate]:
    """Generate candidates for one adaptive round.

    Round 0: broad random sampling (no mutation parents).
    Later rounds: mix of elite mutation (exploit_bias) and broad sampling.

    Same seed + same previous_results → same candidates (deterministic).

    Candidate ID format:
        "{family_id}:adaptive_seed{seed}:round{round_index:02d}:trial{trial_index:04d}"
    """
    space = search_space or default_search_space()

    # Per-round RNG — derives from seed and round_index for full determinism.
    round_seed = seed + round_index * 10_000
    rng = random.Random(round_seed)

    candidates: list[FalsificationCandidate] = []

    if round_index == 0 or not previous_results:
        # Round 0: pure broad sampling
        for trial_idx in range(config.candidates_per_round):
            params = _broad_sample_parameters(space, rng)
            cid = (
                f"{family_id}:adaptive_seed{seed}"
                f":round{round_index:02d}:trial{trial_idx:04d}"
            )
            candidates.append(
                FalsificationCandidate(
                    candidate_id=cid,
                    family_id=family_id,
                    seed=seed,
                    parameters=params,
                )
            )
        return candidates

    # Later rounds: select elites, mix mutation and broad sampling
    elites = _select_elites(previous_results, config.elite_count)

    for trial_idx in range(config.candidates_per_round):
        cid = (
            f"{family_id}:adaptive_seed{seed}"
            f":round{round_index:02d}:trial{trial_idx:04d}"
        )
        use_elite = rng.random() < config.exploit_bias
        if use_elite and elites:
            elite = elites[trial_idx % len(elites)]
            # Wrap FalsificationResult into a candidate for mutation
            parent = FalsificationCandidate(
                candidate_id=elite.candidate_id,
                family_id=elite.family_id,
                seed=elite.seed,
                parameters=elite.parameters,
            )
            params = mutate_candidate_parameters(
                parent, space, rng, config.mutation_scale
            )
        else:
            params = _broad_sample_parameters(space, rng)

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
# Improvement trace
# ---------------------------------------------------------------------------


def compute_improvement_trace(
    rounds: list[AdaptiveRoundSummary],
) -> list[dict[str, Any]]:
    """Compute per-round improvement trace over global best.

    Round 0 delta is None.
    Later delta = round's best_score - global best before this round.
    If a round has no best_score, delta is None.
    Honestly records zero or negative improvement.
    """
    trace: list[dict[str, Any]] = []
    global_best: float | None = None

    for rnd in rounds:
        if rnd.round_index == 0:
            delta = None
        else:
            if rnd.best_score is not None and global_best is not None:
                delta = round(rnd.best_score - global_best, 6)
            else:
                delta = None

        trace.append({
            "round_index": rnd.round_index,
            "best_score": rnd.best_score,
            "delta": delta,
        })

        # Update global best after recording this round's trace entry
        if rnd.best_score is not None:
            if global_best is None or rnd.best_score > global_best:
                global_best = rnd.best_score

    return trace


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def run_adaptive_falsification_search(
    family_id: str,
    seed: int = 42,
    config: MutationConfig | None = None,
    search_space: dict[str, SearchParameterRange] | None = None,
    include_bundles: bool = False,
) -> dict[str, Any]:
    """Run multi-round adaptive falsification search over one synthetic family.

    Initial round uses broad sampling. Later rounds mutate around elite
    candidates from all previous rounds. All rounds are deterministic.

    Args:
        family_id: Key from SYNTHETIC_FAMILIES.
        seed: PRNG seed — same seed → same search.
        config: Mutation configuration. Uses defaults if None.
        search_space: Parameter ranges. Uses default_search_space() if None.
        include_bundles: If True, include bundle for best candidate only.

    Returns:
        Compact JSON-serializable dict with schema_version, rounds,
        best_candidate, results, improvement_trace, and limitations.
    """
    if family_id not in SYNTHETIC_FAMILIES:
        known = sorted(SYNTHETIC_FAMILIES)
        raise ValueError(f"Unknown family_id: {family_id!r}. Choose from {known}")

    if config is None:
        config = MutationConfig()

    space = search_space or default_search_space()

    all_results: list[FalsificationResult] = []
    round_summaries: list[AdaptiveRoundSummary] = []
    prev_global_best_score: float | None = None

    for round_idx in range(config.rounds):
        candidates = generate_adaptive_round_candidates(
            family_id=family_id,
            round_index=round_idx,
            seed=seed,
            previous_results=all_results,
            config=config,
            search_space=space,
        )

        # Determine parent IDs used in this round
        if round_idx == 0 or not all_results:
            parent_ids: list[str] = []
        else:
            elites = _select_elites(all_results, config.elite_count)
            parent_ids = [e.candidate_id for e in elites]

        # Run all candidates for this round
        round_results: list[FalsificationResult] = []
        for candidate in candidates:
            result = run_candidate(candidate, include_bundle=False)
            round_results.append(result)

        all_results.extend(round_results)

        # Find best in this round
        ranked_round = rank_falsification_results(round_results)
        round_best = ranked_round[0] if ranked_round else None

        best_cid = round_best.candidate_id if round_best else None
        best_score = round_best.score if round_best else None
        best_unsafe = round_best.unsafe_legal_state_count if round_best else 0
        best_refs = list(round_best.event_refs) if round_best else []

        # Improvement vs global best before this round
        if round_idx == 0:
            improvement = None
        elif best_score is not None and prev_global_best_score is not None:
            improvement = round(best_score - prev_global_best_score, 6)
        else:
            improvement = None

        # Update global best tracking
        if best_score is not None:
            if prev_global_best_score is None or best_score > prev_global_best_score:
                prev_global_best_score = best_score

        round_summaries.append(
            AdaptiveRoundSummary(
                round_index=round_idx,
                parent_candidate_ids=parent_ids,
                evaluated_count=len(round_results),
                best_candidate_id=best_cid,
                best_score=best_score,
                best_unsafe_legal_state_count=best_unsafe,
                best_event_refs=best_refs,
                improvement_over_previous_best=improvement,
            )
        )

    # Rank all results globally
    ranked_all = rank_falsification_results(all_results)
    global_best = ranked_all[0] if ranked_all else None

    # Build best_candidate dict
    best_candidate_dict: dict[str, Any] | None = None
    if global_best is not None:
        best_candidate_dict = {
            "candidate_id": global_best.candidate_id,
            "family_id": global_best.family_id,
            "seed": global_best.seed,
            "score": global_best.score,
            "score_legacy": global_best.score,
            "exploit_score": global_best.exploit_score,
            "failure_taxonomy": global_best.failure_taxonomy,
            "primary_failure_mode": (
                (global_best.failure_taxonomy or {}).get("primary_failure_mode")
            ),
            "failure_modes": [
                m["mode"]
                for m in (global_best.failure_taxonomy or {}).get("failure_modes", [])
            ],
            "unsafe_legal_state_count": global_best.unsafe_legal_state_count,
            "max_hazard_score": global_best.max_hazard_score,
            "mean_hazard_score": global_best.mean_hazard_score,
            "event_refs": list(global_best.event_refs),
            "parameters": dict(global_best.parameters),
        }
        if include_bundles:
            # Re-run best candidate with bundle
            bundle_result = run_candidate(
                FalsificationCandidate(
                    candidate_id=global_best.candidate_id,
                    family_id=global_best.family_id,
                    seed=global_best.seed,
                    parameters=global_best.parameters,
                ),
                include_bundle=True,
            )
            if bundle_result.bundle is not None:
                metrics = dict(bundle_result.bundle.get("metrics") or {})
                best_candidate_dict["bundle_summary"] = {
                    "unsafe_legal_state_count": int(
                        metrics.get("unsafe_legal_state_count") or 0
                    ),
                    "max_hazard_score": metrics.get("max_hazard_score"),
                    "mean_hazard_score": metrics.get("mean_hazard_score"),
                    "event_refs": list(
                        metrics.get("unsafe_legal_event_refs") or []
                    ),
                }

    # Compact results list (no bundles)
    results_list: list[dict[str, Any]] = [
        {
            "candidate_id": r.candidate_id,
            "score": r.score,
            "score_legacy": r.score,
            "exploit_score": r.exploit_score,
            "exploit_score_total": (
                r.exploit_score.get("total") if r.exploit_score else None
            ),
            "exploit_score_components": (
                r.exploit_score.get("components") if r.exploit_score else None
            ),
            "primary_failure_mode": (
                r.failure_taxonomy.get("primary_failure_mode")
                if r.failure_taxonomy else None
            ),
            "failure_modes": [
                m["mode"] for m in (r.failure_taxonomy or {}).get("failure_modes", [])
            ],
            "unsafe_legal_state_count": r.unsafe_legal_state_count,
            "max_hazard_score": r.max_hazard_score,
            "mean_hazard_score": r.mean_hazard_score,
            "event_refs": list(r.event_refs),
        }
        for r in ranked_all
    ]

    # Round summaries as dicts
    rounds_list: list[dict[str, Any]] = [
        {
            "round_index": rnd.round_index,
            "parent_candidate_ids": list(rnd.parent_candidate_ids),
            "evaluated_count": rnd.evaluated_count,
            "best_candidate_id": rnd.best_candidate_id,
            "best_score": rnd.best_score,
            "best_unsafe_legal_state_count": rnd.best_unsafe_legal_state_count,
            "best_event_refs": list(rnd.best_event_refs),
            "improvement_over_previous_best": rnd.improvement_over_previous_best,
        }
        for rnd in round_summaries
    ]

    improvement_trace = compute_improvement_trace(round_summaries)

    search_space_dict = {
        name: {
            "min_value": r.min_value,
            "max_value": r.max_value,
            "steps": r.steps,
        }
        for name, r in space.items()
    }

    mutation_config_dict: dict[str, Any] = {
        "rounds": config.rounds,
        "candidates_per_round": config.candidates_per_round,
        "elite_count": config.elite_count,
        "mutation_scale": config.mutation_scale,
        "min_mutation_scale": config.min_mutation_scale,
        "exploit_bias": config.exploit_bias,
        "seed": config.seed,
    }

    return {
        "schema_version": ADAPTIVE_SEARCH_SCHEMA,
        "family_id": family_id,
        "seed": seed,
        "mutation_config": mutation_config_dict,
        "search_space": search_space_dict,
        "rounds": rounds_list,
        "best_candidate": best_candidate_dict,
        "results": results_list,
        "total_evaluations": len(all_results),
        "improvement_trace": improvement_trace,
        "limitations": list(_ADAPTIVE_LIMITATIONS),
    }
