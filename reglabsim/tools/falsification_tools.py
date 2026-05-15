"""Deterministic tool wrappers over the falsification search engine.

Every public function in this module:

* Is a **pure Python callable** — no LLM, no NVIDIA, no autonomous agent.
* Returns a compact ``{ok, tool, result, error}`` dict (or a
  JSON-serialisable dict that maps to the same shape).
* Handles invalid inputs with ``ok: false`` rather than crashing.
* Caps ``max_trials`` at ``_MAX_TRIALS`` (100).
* Is deterministic: same ``seed`` → same result.

These wrappers are designed to be safe for future LangChain / LangGraph
tool-calling agents, but they **do not** depend on LangChain at runtime.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from reglabsim.falsification.adaptive_search import (
    MutationConfig,
    run_adaptive_falsification_search,
)
from reglabsim.falsification.search import (
    FalsificationCandidate,
    build_best_candidate_audit_report,
    generate_candidates,
    run_candidate,
    run_falsification_search,
)
from reglabsim.falsification.surrogate import (
    build_surrogate_dataset_from_search_result,
    suggest_candidates_with_surrogate,
    summarize_surrogate_dataset,
    train_surrogate_model,
)
from reglabsim.falsification.surrogate_guided_search import (
    SurrogateGuidedSearchConfig,
    run_surrogate_guided_search,
)
from reglabsim.falsification.surrogate_models import (
    compare_surrogate_models,
    list_surrogate_model_backends,
)
from reglabsim.falsification.track_conditioned_search import (
    TrackConditionedSearchConfig,
    run_track_conditioned_falsification,
)
from reglabsim.logging.audit_report import render_audit_report_markdown
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
from reglabsim.tracks.fidelity import (
    build_track_fidelity_report,
    compact_track_fidelity_summary,
)
from reglabsim.tracks.track_model import (
    build_public_approx_track_model,
    build_track_model_from_synthetic_family,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TRIALS = 100
"""Maximum number of trials a tool will accept (hard cap)."""

_MAX_ADAPTIVE_ROUNDS = 5
"""Hard cap on adaptive rounds per tool call."""

_MAX_ADAPTIVE_CANDIDATES_PER_ROUND = 25
"""Hard cap on candidates per round per tool call."""

_MAX_ADAPTIVE_TOTAL_EVALUATIONS = 100
"""Hard cap on total evaluations per adaptive tool call."""

_ADAPTIVE_TOP_RESULTS_LIMIT = 5
"""Max top_results returned by adaptive search tool."""

_TOP_RESULTS_LIMIT = 5
"""Default number of top results to include in search output."""

_MARKDOWN_EXCERPT_MAX_CHARS = 4000
"""Maximum length of the Markdown excerpt returned by the audit tool."""

_ToolFn = TypeVar("_ToolFn", bound=Callable[..., Any])


class _LangChainToolDecorator(Protocol):
    """Typed view of LangChain's ``tool`` decorator factory."""

    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[_ToolFn], _ToolFn]:
        ...

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tool_ok(tool: str, result: dict[str, Any] | list[Any] | str | None) -> dict[str, Any]:
    """Build a success response envelope."""
    return {"ok": True, "tool": tool, "result": result, "error": None}


def _tool_error(tool: str, exc: Exception) -> dict[str, Any]:
    """Build a safe error envelope — no stack traces leaked."""
    return {
        "ok": False,
        "tool": tool,
        "result": None,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _safe_tool_call(
    tool_name: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Invoke *fn* and wrap the result / exception in the standard envelope."""
    try:
        outcome = fn(*args, **kwargs)
    except Exception as exc:
        return _tool_error(tool_name, exc)
    return _tool_ok(tool_name, outcome)


def _validate_max_trials(tool: str, max_trials: int) -> int:
    """Reject invalid limits and cap to _MAX_TRIALS."""
    if max_trials <= 0:
        raise ValueError(
            f"max_trials must be > 0, got {max_trials}"
        )
    return min(max_trials, _MAX_TRIALS)


def _validate_family_id(tool: str, family_id: str) -> str:
    """Reject unknown synthetic family IDs."""
    if family_id not in SYNTHETIC_FAMILIES:
        known = sorted(SYNTHETIC_FAMILIES)
        raise ValueError(
            f"Unknown family_id: {family_id!r}. Choose from {known}"
        )
    return family_id


# ---------------------------------------------------------------------------
# Tool: list_synthetic_families
# ---------------------------------------------------------------------------


def list_synthetic_families_tool() -> dict[str, Any]:
    """Return compact specs for every registered synthetic family.

    Returns:
        ``{ok, tool, result, error}`` where ``result.families`` is a list
        of compact family summaries (not raw dataclass reprs).

    Example::

        >>> out = list_synthetic_families_tool()
        >>> out["ok"]
        True
        >>> len(out["result"]["families"])
        6
    """
    tool_name = "list_synthetic_families"

    def _list() -> dict[str, Any]:
        families: list[dict[str, Any]] = []
        for spec in SYNTHETIC_FAMILIES.values():
            families.append(
                {
                    "family_id": spec.family_id,
                    "description": spec.description,
                    "segment_type": spec.segment_type,
                    "width_m": spec.width_m,
                    "runoff_type": spec.runoff_type,
                    "barrier_distance_m": spec.barrier_distance_m,
                    "side_by_side_risk": spec.side_by_side_risk,
                    "expected_unsafe_legal": spec.expected_unsafe_legal,
                }
            )
        return {"families": families}

    return _safe_tool_call(tool_name, _list)


# ---------------------------------------------------------------------------
# Tool: generate_falsification_candidates
# ---------------------------------------------------------------------------


def generate_falsification_candidates_tool(
    family_id: str,
    seed: int = 42,
    max_trials: int = 10,
) -> dict[str, Any]:
    """Generate deterministic parameter candidates for a synthetic family.

    Args:
        family_id: Key from ``SYNTHETIC_FAMILIES``.
        seed: PRNG seed — same seed → same candidates.
        max_trials: Number of candidates to generate (capped at 100).

    Returns:
        ``{ok, tool, result, error}`` where ``result`` contains
        ``family_id``, ``seed``, ``max_trials``, ``candidate_count``,
        and ``candidates``.
    """
    tool_name = "generate_falsification_candidates"

    def _generate() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped = _validate_max_trials(tool_name, max_trials)
        candidates = generate_candidates(family_id, seed=seed, max_trials=capped)
        return {
            "family_id": family_id,
            "seed": seed,
            "max_trials": capped,
            "candidate_count": len(candidates),
            "candidates": [
                {"candidate_id": c.candidate_id, "parameters": dict(c.parameters)}
                for c in candidates
            ],
        }

    return _safe_tool_call(tool_name, _generate)


# ---------------------------------------------------------------------------
# Tool: run_falsification_candidate
# ---------------------------------------------------------------------------


def run_falsification_candidate_tool(
    family_id: str,
    parameters: dict[str, float],
    seed: int = 42,
    candidate_id: str | None = None,
    include_bundle: bool = False,
) -> dict[str, Any]:
    """Run a single falsification candidate through the deterministic microkernel.

    Args:
        family_id: Synthetic family key.
        parameters: Candidate parameter overrides (e.g. width_m, gap_s).
        seed: PRNG seed.
        candidate_id: Optional label; auto-generated if omitted.
        include_bundle: If ``True``, include a compact bundle summary
            (not the full raw event log).

    Returns:
        ``{ok, tool, result, error}`` with a compact result payload.
    """
    tool_name = "run_falsification_candidate"

    def _run() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        cid = candidate_id or f"{family_id}:adhoc:seed{seed}"

        candidate = FalsificationCandidate(
            candidate_id=cid,
            family_id=family_id,
            seed=seed,
            parameters=dict(parameters),
        )
        outcome = run_candidate(candidate, include_bundle=include_bundle)

        result: dict[str, Any] = {
            "candidate_id": outcome.candidate_id,
            "family_id": outcome.family_id,
            "seed": outcome.seed,
            "parameters": dict(outcome.parameters),
            "unsafe_legal_state_count": outcome.unsafe_legal_state_count,
            "max_hazard_score": outcome.max_hazard_score,
            "mean_hazard_score": outcome.mean_hazard_score,
            "score": outcome.score,
            "score_legacy": outcome.score,
            "event_refs": list(outcome.event_refs),
        }
        if outcome.exploit_score is not None:
            es = outcome.exploit_score
            result["exploit_score"] = {
                "schema_version": es.get("schema_version"),
                "total": es.get("total"),
                "components": es.get("components"),
                "reason_codes": es.get("reason_codes"),
                "limitations": es.get("limitations"),
            }

        # Compact failure taxonomy
        ft = outcome.failure_taxonomy
        if ft is not None:
            result["primary_failure_mode"] = ft.get("primary_failure_mode")
            result["failure_modes"] = [m["mode"] for m in (ft.get("failure_modes") or [])]
            result["failure_taxonomy"] = {
                "schema_version": ft.get("schema_version"),
                "primary_failure_mode": ft.get("primary_failure_mode"),
                "failure_modes": ft.get("failure_modes"),
                "limitations": ft.get("limitations"),
            }

        if include_bundle and outcome.bundle is not None:
            bundle = outcome.bundle
            metrics: dict[str, Any] = dict(bundle.get("metrics") or {})
            result["bundle_summary"] = {
                "run_id": str(bundle.get("run_id") or ""),
                "world_id": str(bundle.get("world_id") or ""),
                "metrics": {
                    "unsafe_legal_state_count": int(
                        metrics.get("unsafe_legal_state_count") or 0
                    ),
                    "max_hazard_score": metrics.get("max_hazard_score"),
                    "mean_hazard_score": metrics.get("mean_hazard_score"),
                    "unsafe_legal_event_refs": list(
                        metrics.get("unsafe_legal_event_refs") or []
                    ),
                },
                "unsafe_legal_state_count": outcome.unsafe_legal_state_count,
                "event_refs": list(outcome.event_refs),
            }

        return result

    return _safe_tool_call(tool_name, _run)


# ---------------------------------------------------------------------------
# Tool: run_falsification_search
# ---------------------------------------------------------------------------


def run_falsification_search_tool(
    family_id: str,
    seed: int = 42,
    max_trials: int = 25,
) -> dict[str, Any]:
    """Run a deterministic falsification search over one synthetic family.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed.
        max_trials: Number of candidates (capped at 100).

    Returns:
        ``{ok, tool, result, error}`` with compact search output.
        ``top_results`` is limited to ``_TOP_RESULTS_LIMIT`` (5).
    """
    tool_name = "run_falsification_search"

    def _search() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped = _validate_max_trials(tool_name, max_trials)
        search_out = run_falsification_search(
            family_id=family_id,
            seed=seed,
            max_trials=capped,
            include_bundles=False,
        )
        all_results: list[dict[str, Any]] = search_out.get("results") or []
        top = all_results[:_TOP_RESULTS_LIMIT]
        return {
            "schema_version": "falsification_search.v0",
            "family_id": family_id,
            "seed": seed,
            "max_trials": capped,
            "best_candidate": _compact_candidate(search_out.get("best_candidate")),
            "top_results": [_compact_candidate(r) for r in top],
            "result_count": len(all_results),
        }

    return _safe_tool_call(tool_name, _search)


def _compact_candidate(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip non-essential keys from a candidate result dict."""
    if raw is None:
        return None
    keys = (
        "candidate_id",
        "family_id",
        "seed",
        "parameters",
        "unsafe_legal_state_count",
        "max_hazard_score",
        "mean_hazard_score",
        "score",
        "score_legacy",
        "event_refs",
    )
    out = {k: raw[k] for k in keys if k in raw}
    # Include compact exploit_score summary
    es = raw.get("exploit_score")
    if isinstance(es, dict):
        out["exploit_score_total"] = es.get("total")
        out["exploit_score_components"] = es.get("components")
        out["exploit_score_reason_codes"] = (es.get("reason_codes") or [])[:8]
    # Compact taxonomy
    ft = raw.get("failure_taxonomy")
    if isinstance(ft, dict):
        out["primary_failure_mode"] = ft.get("primary_failure_mode")
        out["failure_modes"] = [m["mode"] for m in (ft.get("failure_modes") or [])]
    else:
        # Try flat fields (always include to maintain consistent key presence)
        if "primary_failure_mode" in raw:
            out["primary_failure_mode"] = raw["primary_failure_mode"]
        if "failure_modes" in raw:
            out["failure_modes"] = raw["failure_modes"]
    # Compact track fidelity (PR 8.4.1) — pure metadata, no effect on ranking
    tf = raw.get("track_fidelity")
    if isinstance(tf, dict):
        out["track_fidelity"] = compact_track_fidelity_summary(tf)
    return out


# ---------------------------------------------------------------------------
# Tool: build_best_candidate_audit_report
# ---------------------------------------------------------------------------


def build_best_candidate_audit_report_tool(
    family_id: str,
    seed: int = 42,
    max_trials: int = 25,
) -> dict[str, Any]:
    """Run a search and build an audit report for the best candidate.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed.
        max_trials: Number of candidates (capped at 100).

    Returns:
        ``{ok, tool, result, error}`` containing the audit report dict
        and a bounded Markdown excerpt.
    """
    tool_name = "build_best_candidate_audit_report"

    def _audit() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped = _validate_max_trials(tool_name, max_trials)
        search_out = run_falsification_search(
            family_id=family_id,
            seed=seed,
            max_trials=capped,
            include_bundles=False,
        )
        audit_raw = build_best_candidate_audit_report(search_out)
        best = search_out.get("best_candidate") or {}
        markdown = render_audit_report_markdown(audit_raw)
        # Compose a compact summary block
        summary = dict(audit_raw.get("summary") or {})
        unsafe_count = int(summary.get("unsafe_legal_state_count") or 0)
        return {
            "family_id": family_id,
            "seed": seed,
            "best_candidate_id": str(best.get("candidate_id", "")),
            "audit_report": {
                "schema_version": audit_raw.get("schema_version", ""),
                "summary": summary,
                "unsafe_legal_events_count": unsafe_count,
                "limitations": list(audit_raw.get("limitations") or []),
            },
            "markdown_excerpt": markdown[:_MARKDOWN_EXCERPT_MAX_CHARS],
        }

    return _safe_tool_call(tool_name, _audit)


# ---------------------------------------------------------------------------
# Tool: run_adaptive_falsification_search
# ---------------------------------------------------------------------------


def run_adaptive_falsification_search_tool(
    family_id: str,
    seed: int = 42,
    rounds: int = 3,
    candidates_per_round: int = 10,
    elite_count: int = 3,
) -> dict[str, Any]:
    """Run a multi-round adaptive falsification search over one synthetic family.

    Performs an initial broad search then mutates around elite candidates
    for subsequent rounds. Fully deterministic — same inputs, same results.

    Hard caps: rounds <= 5, candidates_per_round <= 25, total <= 100.
    Invalid or excessive values return ok=False.

    Args:
        family_id: Key from SYNTHETIC_FAMILIES.
        seed: PRNG seed for deterministic replay.
        rounds: Number of adaptive rounds (1-5).
        candidates_per_round: Candidates per round (1-25).
        elite_count: Top candidates used as mutation parents.

    Returns:
        ``{ok, tool, result, error}`` with compact adaptive search output.
        ``top_results`` is limited to 5 entries.
    """
    tool_name = "run_adaptive_falsification_search"

    def _adaptive() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)

        if rounds <= 0:
            raise ValueError(f"rounds must be > 0, got {rounds}")
        if rounds > _MAX_ADAPTIVE_ROUNDS:
            raise ValueError(
                f"rounds exceeds cap of {_MAX_ADAPTIVE_ROUNDS}, got {rounds}"
            )
        if candidates_per_round <= 0:
            raise ValueError(
                f"candidates_per_round must be > 0, got {candidates_per_round}"
            )
        if candidates_per_round > _MAX_ADAPTIVE_CANDIDATES_PER_ROUND:
            raise ValueError(
                f"candidates_per_round exceeds cap of "
                f"{_MAX_ADAPTIVE_CANDIDATES_PER_ROUND}, got {candidates_per_round}"
            )
        if elite_count <= 0:
            raise ValueError(f"elite_count must be > 0, got {elite_count}")
        if elite_count > candidates_per_round:
            raise ValueError(
                f"elite_count ({elite_count}) must be <= "
                f"candidates_per_round ({candidates_per_round})"
            )
        total = rounds * candidates_per_round
        if total > _MAX_ADAPTIVE_TOTAL_EVALUATIONS:
            raise ValueError(
                f"Total evaluations ({total}) exceeds cap of "
                f"{_MAX_ADAPTIVE_TOTAL_EVALUATIONS}"
            )

        config = MutationConfig(
            rounds=rounds,
            candidates_per_round=candidates_per_round,
            elite_count=elite_count,
            seed=seed,
        )
        raw = run_adaptive_falsification_search(
            family_id=family_id,
            seed=seed,
            config=config,
            include_bundles=False,
        )

        # Compact the output for the tool envelope
        all_results: list[dict[str, Any]] = raw.get("results") or []
        top_results = [
            _compact_candidate(r) for r in all_results[:_ADAPTIVE_TOP_RESULTS_LIMIT]
        ]

        # Compact best_candidate (include exploit_score summary)
        raw_best = raw.get("best_candidate")
        compact_best = _compact_candidate(raw_best) if raw_best else None
        if compact_best and isinstance(raw_best, dict):
            es = raw_best.get("exploit_score")
            if isinstance(es, dict):
                compact_best["exploit_score"] = {
                    "schema_version": es.get("schema_version"),
                    "total": es.get("total"),
                    "components": es.get("components"),
                    "reason_codes": (es.get("reason_codes") or [])[:8],
                    "limitations": es.get("limitations"),
                }
            ft = raw_best.get("failure_taxonomy")
            if ft and compact_best:
                compact_best["primary_failure_mode"] = ft.get("primary_failure_mode")
                compact_best["failure_modes"] = [
                    m["mode"] for m in (ft.get("failure_modes") or [])
                ]
                compact_best["failure_taxonomy"] = {
                    "schema_version": ft.get("schema_version"),
                    "primary_failure_mode": ft.get("primary_failure_mode"),
                    "failure_modes": ft.get("failure_modes"),
                    "limitations": ft.get("limitations"),
                }

        return {
            "schema_version": raw.get("schema_version"),
            "family_id": raw.get("family_id"),
            "seed": raw.get("seed"),
            "mutation_config": raw.get("mutation_config"),
            "rounds": raw.get("rounds"),
            "best_candidate": compact_best,
            "top_results": top_results,
            "total_evaluations": raw.get("total_evaluations"),
            "improvement_trace": raw.get("improvement_trace"),
            "limitations": raw.get("limitations"),
        }

    return _safe_tool_call(tool_name, _adaptive)


# ---------------------------------------------------------------------------
# Constants for surrogate tools
# ---------------------------------------------------------------------------

_MAX_SURROGATE_TRIALS = 100
_MAX_SURROGATE_ROWS = 100
_MAX_SURROGATE_CANDIDATE_COUNT = 50
_MAX_SURROGATE_POOL_SIZE = 500
_MAX_SURROGATE_VALIDATE = 20


# ---------------------------------------------------------------------------
# Tool: build_surrogate_dataset
# ---------------------------------------------------------------------------


def build_surrogate_dataset_tool(
    family_id: str,
    seed: int = 42,
    max_trials: int = 25,
    adaptive: bool = True,
) -> dict[str, Any]:
    """Build a surrogate exploit dataset from a deterministic falsification search.

    Runs a search (adaptive or regular), extracts compact numeric features
    and labels for each candidate, and returns a JSON-serializable dataset.
    No raw event logs, no full bundles, no secrets.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed — same seed → same dataset.
        max_trials: Number of candidates to evaluate (capped at 100).
        adaptive: If True, use adaptive search; otherwise regular search.

    Returns:
        ``{ok, tool, result, error}`` with dataset and summary.
    """
    tool_name = "build_surrogate_dataset"

    def _build() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped = min(_validate_max_trials(tool_name, max_trials), _MAX_SURROGATE_TRIALS)

        if adaptive:
            rounds = max(1, min(3, capped // 5))
            per_round = max(1, min(capped // rounds, _MAX_ADAPTIVE_CANDIDATES_PER_ROUND))
            elite = max(1, min(3, per_round))
            config = MutationConfig(
                rounds=rounds,
                candidates_per_round=per_round,
                elite_count=elite,
                seed=seed,
            )
            search_out = run_adaptive_falsification_search(
                family_id=family_id,
                seed=seed,
                config=config,
                include_bundles=False,
            )
        else:
            search_out = run_falsification_search(
                family_id=family_id,
                seed=seed,
                max_trials=capped,
                include_bundles=False,
            )

        dataset = build_surrogate_dataset_from_search_result(search_out)

        # Cap rows for tool output
        rows = dataset.get("rows") or []
        if len(rows) > _MAX_SURROGATE_ROWS:
            dataset = dict(dataset)
            dataset["rows"] = rows[:_MAX_SURROGATE_ROWS]
            dataset["row_count"] = _MAX_SURROGATE_ROWS

        summary = summarize_surrogate_dataset(dataset)

        return {
            "dataset": dataset,
            "summary": summary,
            "warning": (
                "Suggestions from this dataset require deterministic runtime validation "
                "before claiming exploit evidence."
            ),
        }

    return _safe_tool_call(tool_name, _build)


# ---------------------------------------------------------------------------
# Tool: suggest_surrogate_candidates
# ---------------------------------------------------------------------------


def suggest_surrogate_candidates_tool(
    family_id: str,
    seed: int = 42,
    training_trials: int = 30,
    candidate_count: int = 10,
    proposal_pool_size: int = 100,
) -> dict[str, Any]:
    """Build dataset, train nearest-neighbor surrogate, suggest candidates.

    Generates candidates ranked by predicted exploit score.
    Does NOT run the simulator on suggestions — predictions only.
    Suggestions must be validated by deterministic runtime before
    treating them as exploit evidence.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed for determinism.
        training_trials: Candidates to use for surrogate training (capped at 100).
        candidate_count: Number of suggestions to return (capped at 50).
        proposal_pool_size: Size of proposal pool to score (capped at 500).

    Returns:
        ``{ok, tool, result, error}`` with ranked suggestions.
    """
    tool_name = "suggest_surrogate_candidates"

    def _suggest() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped_trials = min(
            _validate_max_trials(tool_name, training_trials), _MAX_SURROGATE_TRIALS
        )
        capped_count = min(max(1, candidate_count), _MAX_SURROGATE_CANDIDATE_COUNT)
        capped_pool = min(max(1, proposal_pool_size), _MAX_SURROGATE_POOL_SIZE)

        # Build dataset via adaptive search
        rounds = max(1, min(3, capped_trials // 5))
        per_round = max(1, min(capped_trials // rounds, _MAX_ADAPTIVE_CANDIDATES_PER_ROUND))
        elite = max(1, min(3, per_round))
        config = MutationConfig(
            rounds=rounds,
            candidates_per_round=per_round,
            elite_count=elite,
            seed=seed,
        )
        search_out = run_adaptive_falsification_search(
            family_id=family_id,
            seed=seed,
            config=config,
            include_bundles=False,
        )
        dataset = build_surrogate_dataset_from_search_result(search_out)
        model = train_surrogate_model(dataset, target_label="exploit_score_total")

        suggestions = suggest_candidates_with_surrogate(
            model=model,
            family_id=family_id,
            seed=seed,
            candidate_count=capped_count,
            proposal_pool_size=capped_pool,
        )
        summary = summarize_surrogate_dataset(dataset)

        return {
            "suggestions": suggestions,
            "dataset_summary": summary,
            "warning": (
                "These are surrogate predictions, not validated exploit evidence. "
                "Run deterministic validation before claiming any exploit."
            ),
        }

    return _safe_tool_call(tool_name, _suggest)


# ---------------------------------------------------------------------------
# Tool: describe_track_fidelity
# ---------------------------------------------------------------------------


def describe_track_fidelity_tool(
    family_id: str | None = None,
    track_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact track fidelity report for a synthetic family or track.

    For synthetic families: builds a T0 fidelity model and report.
    For other track IDs with optional metadata: builds a T1 public
    approximate model and report.
    No geometry blobs, no coordinate arrays in output.

    Args:
        family_id: Key from SYNTHETIC_FAMILIES (T0 model).
        track_id: Arbitrary track identifier for public approximate model.
        metadata: Optional dict with length_m, segments, known_gaps, etc.

    Returns:
        ``{ok, tool, result, error}`` with compact fidelity report.
    """
    tool_name = "describe_track_fidelity"

    def _describe() -> dict[str, Any]:
        if family_id is not None:
            _validate_family_id(tool_name, family_id)
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
            return {
                "fidelity_report": report,
                "track_model_summary": {
                    "track_id": track_model.track_id,
                    "display_name": track_model.display_name,
                    "fidelity_tier": track_model.fidelity_tier,
                    "data_classification": track_model.data_classification,
                    "segment_count": track_model.segment_count,
                    "known_gaps": list(track_model.known_gaps),
                    "limitations": list(track_model.limitations),
                },
            }

        if track_id is not None:
            meta = dict(metadata or {})
            track_model = build_public_approx_track_model(track_id, meta)
            report = build_track_fidelity_report(track_model)
            return {
                "fidelity_report": report,
                "track_model_summary": {
                    "track_id": track_model.track_id,
                    "display_name": track_model.display_name,
                    "fidelity_tier": track_model.fidelity_tier,
                    "data_classification": track_model.data_classification,
                    "segment_count": track_model.segment_count,
                    "known_gaps": list(track_model.known_gaps),
                    "limitations": list(track_model.limitations),
                },
            }

        raise ValueError(
            "Must provide either family_id (for synthetic T0 model) "
            "or track_id (for public approximate T1 model)."
        )

    return _safe_tool_call(tool_name, _describe)


# ---------------------------------------------------------------------------
# Constants for surrogate-guided search tool
# ---------------------------------------------------------------------------

_MAX_GUIDED_ROUNDS = 5
_MAX_GUIDED_INITIAL_TRIALS = 100
_MAX_GUIDED_SUGGESTIONS_PER_ROUND = 50
_MAX_GUIDED_VALIDATION_PER_ROUND = 25
_MAX_GUIDED_PROPOSAL_POOL_SIZE = 500
_MAX_GUIDED_TOP_RESULTS = 10

_VALID_GUIDED_TARGET_LABELS = frozenset({
    "exploit_score_total",
    "legacy_score",
    "unsafe_legal_state_count",
    "max_hazard_score",
})


# ---------------------------------------------------------------------------
# Tool: run_surrogate_guided_search
# ---------------------------------------------------------------------------


def run_surrogate_guided_search_tool(
    family_id: str,
    seed: int = 42,
    rounds: int = 2,
    initial_trials: int = 20,
    suggestions_per_round: int = 10,
    validation_per_round: int = 5,
    proposal_pool_size: int = 100,
    target_label: str = "exploit_score_total",
) -> dict[str, Any]:
    """Run the surrogate-guided active-learning falsification loop.

    Runs an initial deterministic search, builds a surrogate dataset,
    trains a nearest-neighbor surrogate, suggests candidates, validates
    them with the deterministic runtime, and repeats for N rounds.

    Surrogate predictions guide search; only runtime-validated candidates
    count as exploit evidence.

    Hard caps: rounds <= 5, initial_trials <= 100,
    suggestions_per_round <= 50, validation_per_round <= 25,
    proposal_pool_size <= 500.

    Args:
        family_id: Synthetic family key.
        seed: PRNG seed for determinism.
        rounds: Number of surrogate-guided rounds (1-5).
        initial_trials: Candidates for initial baseline search (1-100).
        suggestions_per_round: Surrogate suggestions per round (1-50).
        validation_per_round: Suggestions validated per round (1-25).
        proposal_pool_size: Proposal pool size for surrogate (1-500).
        target_label: Label to optimize (exploit_score_total, legacy_score,
            unsafe_legal_state_count, max_hazard_score).

    Returns:
        ``{ok, tool, result, error}`` with compact search output.
    """
    tool_name = "run_surrogate_guided_search"

    def _run() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)

        if target_label not in _VALID_GUIDED_TARGET_LABELS:
            raise ValueError(
                f"target_label must be one of {sorted(_VALID_GUIDED_TARGET_LABELS)}, "
                f"got {target_label!r}"
            )

        capped_rounds = min(max(1, rounds), _MAX_GUIDED_ROUNDS)
        capped_trials = min(max(1, initial_trials), _MAX_GUIDED_INITIAL_TRIALS)
        capped_suggestions = min(max(1, suggestions_per_round), _MAX_GUIDED_SUGGESTIONS_PER_ROUND)
        capped_validation = min(max(1, validation_per_round), _MAX_GUIDED_VALIDATION_PER_ROUND)
        capped_pool = min(
            max(capped_suggestions, proposal_pool_size), _MAX_GUIDED_PROPOSAL_POOL_SIZE
        )

        if capped_validation > capped_suggestions:
            capped_validation = capped_suggestions

        config = SurrogateGuidedSearchConfig(
            rounds=capped_rounds,
            initial_trials=capped_trials,
            suggestions_per_round=capped_suggestions,
            validation_per_round=capped_validation,
            proposal_pool_size=capped_pool,
            seed=seed,
            target_label=target_label,
        )

        raw = run_surrogate_guided_search(
            family_id=family_id,
            seed=seed,
            config=config,
        )

        # Compact for tool output: limit validated_results
        compact = dict(raw)
        all_results = raw.get("validated_results") or []
        compact["validated_results"] = all_results[:_MAX_GUIDED_TOP_RESULTS]
        compact["total_validated_count"] = len(all_results)

        return compact

    return _safe_tool_call(tool_name, _run)


# ---------------------------------------------------------------------------
# Constants for track-conditioned falsification tool
# ---------------------------------------------------------------------------

_MAX_TC_SEGMENTS = 20
_MAX_SURROGATE_TRAINING_TRIALS_TC = 100


# ---------------------------------------------------------------------------
# Tool: list_surrogate_model_backends
# ---------------------------------------------------------------------------


def list_surrogate_model_backends_tool() -> dict[str, Any]:
    """Return the surrogate model registry with availability status.

    Returns:
        ``{ok, tool, result, error}`` with compact registry.
    """
    tool_name = "list_surrogate_model_backends"
    return _safe_tool_call(tool_name, list_surrogate_model_backends)


# ---------------------------------------------------------------------------
# Tool: compare_surrogate_models
# ---------------------------------------------------------------------------


def compare_surrogate_models_tool(
    family_id: str = "confined_corner_grass",
    seed: int = 42,
    max_trials: int = 30,
    target_label: str = "exploit_score_total",
) -> dict[str, Any]:
    """Build a dataset and compare surrogate model backends.

    Runs a quick falsification search, builds a surrogate dataset,
    evaluates available model backends, and returns compact comparison.
    Unavailable sklearn models appear with available=False.

    Args:
        family_id: Synthetic family key for training data.
        seed: PRNG seed for determinism.
        max_trials: Candidates for training search (capped at 100).
        target_label: Label to predict and evaluate.

    Returns:
        ``{ok, tool, result, error}`` with compact comparison.
    """
    tool_name = "compare_surrogate_models"

    def _compare() -> dict[str, Any]:
        _validate_family_id(tool_name, family_id)
        capped = min(_validate_max_trials(tool_name, max_trials), _MAX_TRIALS)

        from reglabsim.falsification.search import run_falsification_search
        from reglabsim.falsification.surrogate import (
            build_surrogate_dataset_from_search_result,
        )
        sr = run_falsification_search(
            family_id=family_id, seed=seed, max_trials=capped
        )
        dataset = build_surrogate_dataset_from_search_result(sr)
        comparison = compare_surrogate_models(
            dataset=dataset, target_label=target_label, seed=seed
        )
        return comparison

    return _safe_tool_call(tool_name, _compare)
_MAX_TC_CANDIDATES_PER_SEGMENT = 25


# ---------------------------------------------------------------------------
# Tool: run_track_conditioned_falsification
# ---------------------------------------------------------------------------


def run_track_conditioned_falsification_tool(
    family_id: str | None = None,
    track_id: str | None = None,
    seed: int = 42,
    max_segments: int = 5,
    candidates_per_segment: int = 4,
    include_low_readiness_segments: bool = False,
    use_surrogate_guidance: bool = False,
    surrogate_model_type: str = "nearest_neighbor",
    surrogate_training_trials: int = 0,
    compare_against_heuristic: bool = False,
) -> dict[str, Any]:
    """Run a track-conditioned falsification campaign over segment abstractions.

    Generates candidates from segment geometry, validates through the
    deterministic runtime, and returns compact segment findings.

    Segment risk score alone is not evidence — only runtime-validated
    candidates count as findings. Outputs are conditioned on declared
    fidelity tier.

    Args:
        family_id: Key from SYNTHETIC_FAMILIES (builds T0 track model).
        track_id: Arbitrary track ID for a public approximate model.
        seed: PRNG seed for determinism.
        max_segments: Max segments to evaluate (capped at 20).
        candidates_per_segment: Candidates per segment (capped at 25).
        include_low_readiness_segments: Include sparse segments if True.
        use_surrogate_guidance: Use surrogate model for candidate ranking.
        surrogate_model_type: Surrogate backend (nearest_neighbor or sklearn).
        surrogate_training_trials: Trials for surrogate training data (0-100).
        compare_against_heuristic: Include heuristic vs surrogate comparison.

    Returns:
        ``{ok, tool, result, error}`` with compact track-conditioned report.
    """
    tool_name = "run_track_conditioned_falsification"

    def _run() -> dict[str, Any]:
        if family_id is not None and track_id is not None:
            raise ValueError("Provide either family_id or track_id, not both.")
        if family_id is None and track_id is None:
            raise ValueError(
                "Must provide either family_id (synthetic T0 model) "
                "or track_id (public approximate model)."
            )

        capped_segments = min(max(1, max_segments), _MAX_TC_SEGMENTS)
        capped_candidates = min(max(1, candidates_per_segment), _MAX_TC_CANDIDATES_PER_SEGMENT)
        capped_training = min(max(0, surrogate_training_trials), _MAX_SURROGATE_TRAINING_TRIALS_TC)

        if family_id is not None:
            _validate_family_id(tool_name, family_id)
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
        else:
            assert track_id is not None
            track_model = build_public_approx_track_model(track_id, {})
            if not track_model.segments:
                raise ValueError(
                    f"No segment data available for track_id={track_id!r}. "
                    "Provide metadata with segments to run track-conditioned campaign."
                )

        config = TrackConditionedSearchConfig(
            seed=seed,
            max_segments=capped_segments,
            candidates_per_segment=capped_candidates,
            include_low_readiness_segments=include_low_readiness_segments,
            use_surrogate_guidance=use_surrogate_guidance,
            surrogate_model_type=surrogate_model_type,
            surrogate_training_trials=capped_training,
            compare_against_heuristic=compare_against_heuristic,
        )

        result = run_track_conditioned_falsification(track_model, config=config)
        return result

    return _safe_tool_call(tool_name, _run)


# ---------------------------------------------------------------------------
# Optional: LangChain @tool adapters
# ---------------------------------------------------------------------------


def as_langchain_tools() -> list[Any]:
    """Return a list of LangChain ``@tool``-decorated wrappers.

    These adapters are **optional**: LangChain is imported lazily.
    Uses ``parse_docstring=True`` so that the LangChain agent can
    extract argument descriptions from the ``Args:`` sections in
    each tool's docstring — following the official LangChain pattern.

    Raises:
        RuntimeError: If ``langchain-core`` is not installed.

    Example::

        >>> tools = as_langchain_tools()
        >>> [t.name for t in tools]
        ['list_synthetic_families', 'generate_falsification_candidates', ...]
    """
    try:
        from langchain.tools import tool as langchain_tool
    except ImportError:
        try:
            from langchain_core.tools import tool as langchain_tool
        except ImportError as exc:
            raise RuntimeError(
                "LangChain tool adapters require langchain-core. "
                "Install the 'agents' extra: pip install f1lab-ai[agents]"
            ) from exc

    tool = cast(_LangChainToolDecorator, langchain_tool)

    @tool(parse_docstring=True)
    def list_synthetic_families() -> str:
        """List all synthetic circuit/segment families.

        Returns compact specs for every registered synthetic family,
        including segment type, width, runoff type, and expected risk.
        """
        return json.dumps(
            list_synthetic_families_tool(), sort_keys=True, ensure_ascii=True
        )

    @tool(parse_docstring=True)
    def generate_falsification_candidates(
        family_id: str, seed: int = 42, max_trials: int = 10
    ) -> str:
        """Generate deterministic falsification candidates for a synthetic family.

        Same seed always produces the same candidates. Candidates are
        parameter combinations drawn from the default search space.

        Args:
            family_id: Key from the synthetic families registry
                (e.g. 'confined_corner_grass').
            seed: PRNG seed — same seed produces identical candidates.
            max_trials: Number of candidates to generate (capped at 100).
        """
        return json.dumps(
            generate_falsification_candidates_tool(
                family_id=family_id, seed=seed, max_trials=max_trials
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def run_falsification_candidate(
        family_id: str,
        parameters: dict[str, float],
        seed: int = 42,
        candidate_id: str | None = None,
        include_bundle: bool = False,
    ) -> str:
        """Run one falsification candidate through the deterministic microkernel.

        Args:
            family_id: Synthetic family key (e.g. 'fast_corner_wall').
            parameters: Candidate parameter overrides (e.g. width_m, gap_s).
            seed: PRNG seed for deterministic replay.
            candidate_id: Optional label; auto-generated if omitted.
            include_bundle: If True, include a compact bundle summary
                (not the full raw event log).
        """
        return json.dumps(
            run_falsification_candidate_tool(
                family_id=family_id,
                parameters=parameters,
                seed=seed,
                candidate_id=candidate_id,
                include_bundle=include_bundle,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def run_falsification_search(
        family_id: str, seed: int = 42, max_trials: int = 25
    ) -> str:
        """Run a deterministic falsification search over one synthetic family.

        Generates candidates, runs each through the microkernel, and
        returns results ranked by exploit score (descending).

        Args:
            family_id: Synthetic family key (e.g. 'narrow_street_chicane').
            seed: PRNG seed for deterministic replay.
            max_trials: Number of candidates to evaluate (capped at 100).
        """
        return json.dumps(
            run_falsification_search_tool(
                family_id=family_id, seed=seed, max_trials=max_trials
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def build_best_candidate_audit_report(
        family_id: str, seed: int = 42, max_trials: int = 25
    ) -> str:
        """Build an audit report for the best falsification candidate.

        Runs a search, identifies the highest-scoring candidate, and
        produces a deterministic audit report with a Markdown excerpt.

        Args:
            family_id: Synthetic family key (e.g. 'high_speed_entry_low_visibility').
            seed: PRNG seed for deterministic replay.
            max_trials: Number of candidates to evaluate (capped at 100).
        """
        return json.dumps(
            build_best_candidate_audit_report_tool(
                family_id=family_id, seed=seed, max_trials=max_trials
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def run_adaptive_falsification_search(
        family_id: str,
        seed: int = 42,
        rounds: int = 3,
        candidates_per_round: int = 10,
        elite_count: int = 3,
    ) -> str:
        """Run a multi-round adaptive falsification search over one synthetic family.

        Initial round uses broad sampling. Later rounds mutate parameters around
        the top elite candidates. Proposes candidates only — the deterministic
        runtime evaluates them. Do not claim improvement unless tool output shows
        a higher score or stronger evidence.

        Adaptive search is the source of candidate proposals only.
        SafetyOracle and deterministic tools remain the source of truth.

        Args:
            family_id: Synthetic family key (e.g. 'confined_corner_grass').
            seed: PRNG seed for deterministic replay.
            rounds: Number of adaptive rounds (1-5).
            candidates_per_round: Candidates per round (1-25).
            elite_count: Top candidates used as mutation parents per round.
        """
        return json.dumps(
            run_adaptive_falsification_search_tool(
                family_id=family_id,
                seed=seed,
                rounds=rounds,
                candidates_per_round=candidates_per_round,
                elite_count=elite_count,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def build_surrogate_dataset(
        family_id: str,
        seed: int = 42,
        max_trials: int = 25,
        adaptive: bool = True,
    ) -> str:
        """Build a surrogate exploit dataset from deterministic falsification search.

        Output contains compact numeric features and labels per candidate.
        No raw event logs or full bundles.

        Args:
            family_id: Synthetic family key.
            seed: PRNG seed for determinism.
            max_trials: Candidates to evaluate (capped at 100).
            adaptive: If True, use adaptive search; otherwise regular search.
        """
        return json.dumps(
            build_surrogate_dataset_tool(
                family_id=family_id,
                seed=seed,
                max_trials=max_trials,
                adaptive=adaptive,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def suggest_surrogate_candidates(
        family_id: str,
        seed: int = 42,
        training_trials: int = 30,
        candidate_count: int = 10,
        proposal_pool_size: int = 100,
    ) -> str:
        """Train a surrogate model and suggest high-scoring candidate parameters.

        Suggestions are predictions only — they must be validated by the
        deterministic runtime before treating them as exploit evidence.

        Args:
            family_id: Synthetic family key.
            seed: PRNG seed for determinism.
            training_trials: Candidates for surrogate training (capped at 100).
            candidate_count: Number of suggestions to return (capped at 50).
            proposal_pool_size: Proposal pool size (capped at 500).
        """
        return json.dumps(
            suggest_surrogate_candidates_tool(
                family_id=family_id,
                seed=seed,
                training_trials=training_trials,
                candidate_count=candidate_count,
                proposal_pool_size=proposal_pool_size,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def run_surrogate_guided_search(
        family_id: str,
        seed: int = 42,
        rounds: int = 2,
        initial_trials: int = 20,
        suggestions_per_round: int = 10,
        validation_per_round: int = 5,
        proposal_pool_size: int = 100,
        target_label: str = "exploit_score_total",
    ) -> str:
        """Run surrogate-guided active-learning falsification loop.

        Builds a surrogate dataset from an initial search, trains a
        nearest-neighbor surrogate, suggests candidates, validates them
        with the deterministic runtime, and repeats for N rounds.
        Only runtime-validated candidates count as evidence.

        Args:
            family_id: Synthetic family key.
            seed: PRNG seed for determinism.
            rounds: Number of surrogate-guided rounds (1-5).
            initial_trials: Candidates for initial baseline search (1-100).
            suggestions_per_round: Surrogate suggestions per round (1-50).
            validation_per_round: Suggestions validated per round (1-25).
            proposal_pool_size: Proposal pool size for surrogate (1-500).
            target_label: Label to optimize (exploit_score_total, legacy_score,
                unsafe_legal_state_count, max_hazard_score).
        """
        return json.dumps(
            run_surrogate_guided_search_tool(
                family_id=family_id,
                seed=seed,
                rounds=rounds,
                initial_trials=initial_trials,
                suggestions_per_round=suggestions_per_round,
                validation_per_round=validation_per_round,
                proposal_pool_size=proposal_pool_size,
                target_label=target_label,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def run_track_conditioned_falsification(
        family_id: str | None = None,
        track_id: str | None = None,
        seed: int = 42,
        max_segments: int = 5,
        candidates_per_segment: int = 4,
        include_low_readiness_segments: bool = False,
    ) -> str:
        """Run track-conditioned falsification over segment abstractions.

        Generates candidates from segment geometry (width, barrier, runoff),
        validates through the deterministic runtime, and returns compact
        segment findings conditioned on declared fidelity tier.

        Segment risk score alone is not evidence.
        Only runtime-validated candidates count as findings.

        Args:
            family_id: Key from synthetic families registry (T0 model).
            track_id: Arbitrary track identifier for public approximate model.
            seed: PRNG seed for determinism.
            max_segments: Max segments to evaluate (capped at 20).
            candidates_per_segment: Candidates per segment (capped at 25).
            include_low_readiness_segments: Include sparse segments if True.
        """
        return json.dumps(
            run_track_conditioned_falsification_tool(
                family_id=family_id,
                track_id=track_id,
                seed=seed,
                max_segments=max_segments,
                candidates_per_segment=candidates_per_segment,
                include_low_readiness_segments=include_low_readiness_segments,
            ),
            sort_keys=True,
            ensure_ascii=True,
        )

    @tool(parse_docstring=True)
    def list_surrogate_model_backends() -> str:
        """List all available surrogate model backends with availability status.

        Returns registry showing which backends are available and their requirements.
        nearest_neighbor always available; sklearn backends require scikit-learn.
        """
        return json.dumps(
            list_surrogate_model_backends_tool(), sort_keys=True, ensure_ascii=True
        )

    @tool(parse_docstring=True)
    def compare_surrogate_models(
        family_id: str = "confined_corner_grass",
        seed: int = 42,
        max_trials: int = 30,
        target_label: str = "exploit_score_total",
    ) -> str:
        """Compare surrogate model backends on a falsification dataset.

        Builds a small dataset, evaluates available models, and returns
        compact comparison with best_available_model_type.

        Args:
            family_id: Synthetic family for training data.
            seed: PRNG seed for determinism.
            max_trials: Training search trials (capped at 100).
            target_label: Label to predict and evaluate.
        """
        return json.dumps(
            compare_surrogate_models_tool(
                family_id=family_id, seed=seed,
                max_trials=max_trials, target_label=target_label,
            ),
            sort_keys=True, ensure_ascii=True,
        )

    return [
        list_synthetic_families,
        generate_falsification_candidates,
        run_falsification_candidate,
        run_falsification_search,
        build_best_candidate_audit_report,
        run_adaptive_falsification_search,
        build_surrogate_dataset,
        suggest_surrogate_candidates,
        run_surrogate_guided_search,
        run_track_conditioned_falsification,
        list_surrogate_model_backends,
        compare_surrogate_models,
    ]
