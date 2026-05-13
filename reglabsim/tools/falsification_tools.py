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

from reglabsim.falsification.search import (
    FalsificationCandidate,
    build_best_candidate_audit_report,
    generate_candidates,
    run_candidate,
    run_falsification_search,
)
from reglabsim.logging.audit_report import render_audit_report_markdown
from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TRIALS = 100
"""Maximum number of trials a tool will accept (hard cap)."""

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
            "event_refs": list(outcome.event_refs),
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
        "event_refs",
    )
    return {k: raw[k] for k in keys if k in raw}


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

    return [
        list_synthetic_families,
        generate_falsification_candidates,
        run_falsification_candidate,
        run_falsification_search,
        build_best_candidate_audit_report,
    ]
