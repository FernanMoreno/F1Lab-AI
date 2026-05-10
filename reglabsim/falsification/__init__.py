"""Deterministic falsification search engine for unsafe legal state discovery."""

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

__all__ = [
    "FalsificationCandidate",
    "FalsificationResult",
    "SearchParameterRange",
    "build_best_candidate_audit_report",
    "default_search_space",
    "generate_candidates",
    "run_candidate",
    "run_falsification_search",
    "score_candidate_metrics",
]
