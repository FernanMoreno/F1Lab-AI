"""Deterministic tool wrappers for the falsification search engine.

These tools expose safe, deterministic, JSON-serializable callables that
future LangChain / LangGraph agents can use.  No LLM, no NVIDIA, no
autonomous agent — pure Python wrappers over existing falsification code.
"""

from reglabsim.tools.falsification_tools import (
    as_langchain_tools,
    build_best_candidate_audit_report_tool,
    generate_falsification_candidates_tool,
    list_synthetic_families_tool,
    run_adaptive_falsification_search_tool,
    run_falsification_candidate_tool,
    run_falsification_search_tool,
)

__all__ = [
    "as_langchain_tools",
    "build_best_candidate_audit_report_tool",
    "generate_falsification_candidates_tool",
    "list_synthetic_families_tool",
    "run_adaptive_falsification_search_tool",
    "run_falsification_candidate_tool",
    "run_falsification_search_tool",
]
