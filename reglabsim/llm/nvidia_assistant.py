"""NVIDIA/LangChain assistant for F1Lab-AI synthetic family analysis.

The LLM is strictly advisory — it never affects simulation decisions,
oracle thresholds, or metric values.  It only produces human-readable
review notes in cautious language.

Usage requires the `nvidia` optional extra:
    pip install 'f1lab-ai[nvidia]'

Required env vars:
    NVIDIA_API_KEY              — NVIDIA API key
    F1LAB_NVIDIA_MODEL          — optional model override
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

_SYSTEM_INSTRUCTIONS = (
    "You are an F1 regulation stress-test analysis assistant. "
    "Your role is to summarize deterministic simulation outputs in clear, cautious language. "
    "Rules you must follow:\n"
    "- Do not make calibrated safety or regulatory claims.\n"
    "- Do not assert that any action is proven safe or dangerous.\n"
    "- Always state that findings are from a deterministic simulation, not real telemetry.\n"
    "- Identify at least one limitation for every analysis you produce.\n"
    "- Use hedging language: 'suggests', 'may indicate', 'in this scenario', etc.\n"
    "- Never use: 'proven safe', 'guaranteed', or 'calibrated recommendation'."
)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def build_nvidia_llm(model: str | None = None) -> Any:
    """Build and return a ChatNVIDIA LLM instance.

    Raises:
        RuntimeError: if `langchain-nvidia-ai-endpoints` is not installed.
        RuntimeError: if NVIDIA_API_KEY is not set.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA assistant requires optional dependency "
            "`langchain-nvidia-ai-endpoints`. "
            "Install with: pip install 'f1lab-ai[nvidia]'"
        ) from exc

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is required for the NVIDIA assistant. "
            "Export it before calling build_nvidia_llm()."
        )

    resolved_model = model or os.getenv("F1LAB_NVIDIA_MODEL", _DEFAULT_MODEL)
    return ChatNVIDIA(model=resolved_model)


# ---------------------------------------------------------------------------
# Prompt builders (pure — no I/O, no LLM calls)
# ---------------------------------------------------------------------------


def build_audit_summary_prompt(audit_report: dict[str, Any]) -> str:
    """Build a compact audit-summary prompt from an audit report dict.

    Includes only high-level metrics — never dumps raw event payloads.
    """
    run = audit_report.get("run") or {}
    summary = audit_report.get("summary") or {}
    limitations = audit_report.get("limitations") or []

    unsafe_count = int(summary.get("unsafe_legal_state_count") or 0)
    max_hazard = summary.get("max_hazard_score")
    mean_hazard = summary.get("mean_hazard_score")
    segments = ", ".join(sorted(summary.get("unsafe_legal_segments") or [])) or "none"
    status_counts = dict(summary.get("safety_verdict_status_counts") or {})

    lim_text = "\n".join(f"- {lim}" for lim in limitations) if limitations else "- None stated."

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        "## F1Lab-AI Audit Report — Summary for Review\n\n"
        f"Run ID: {run.get('run_id', 'unknown')}\n"
        f"Track: {run.get('track', 'unknown')}\n"
        f"Seed: {run.get('seed', 'unknown')}\n\n"
        "### Key Metrics\n"
        f"- Unsafe legal state count: {unsafe_count}\n"
        f"- Max hazard score: {max_hazard}\n"
        f"- Mean hazard score: {mean_hazard}\n"
        f"- Affected segments: {segments}\n"
        f"- Safety verdict status counts: {status_counts}\n\n"
        "### Simulation Limitations\n"
        f"{lim_text}\n\n"
        "Task: Write a concise (3-5 sentence) review of these results. "
        "Use cautious language. Identify at least one additional limitation not listed above. "
        "Do not claim the simulation proves or disproves any regulatory position."
    )


def build_synthetic_family_review_prompt(families: list[dict[str, Any]]) -> str:
    """Build a prompt asking the LLM to review or propose synthetic family candidates.

    Includes compact family specs — never raw simulation event logs.
    """
    family_lines: list[str] = []
    for f in families:
        fid = f.get("family_id", "unknown")
        seg_type = f.get("segment_type", "unknown")
        width = f.get("width_m", "?")
        runoff = f.get("runoff_type", "unknown")
        barrier = f.get("barrier_distance_m", "?")
        expected = f.get("expected_unsafe_legal", "unknown")
        family_lines.append(
            f"  - {fid}: type={seg_type}, width={width}m, "
            f"runoff={runoff}, barrier={barrier}m, expected_unsafe_legal={expected}"
        )

    families_text = "\n".join(family_lines) if family_lines else "  (none provided)"

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        "## Synthetic Family Review\n\n"
        "The following synthetic circuit families were used in a stress-test run:\n\n"
        f"{families_text}\n\n"
        "Task:\n"
        "1. Briefly describe what physical condition each family is meant to stress-test "
        "(1-2 sentences each).\n"
        "2. Identify any gaps in coverage — segment geometries or conditions NOT represented.\n"
        "3. Suggest at most 2 additional generic (non-real-track) families that could "
        "exercise the safety pipeline differently.\n"
        "Limit your response to 200 words. "
        "Do not reference real track names (Suzuka, Baku, Monaco, Singapore, etc.)."
    )


def build_counterfactual_review_prompt(counterfactual: dict[str, Any]) -> str:
    """Build a prompt for reviewing a single counterfactual patch result.

    Includes compact delta metrics — never raw event payloads.
    """
    patch_id = counterfactual.get("patch_id", "unknown")
    patch_type = counterfactual.get("patch_type", "unknown")
    verdict = counterfactual.get("verdict", "unknown")
    mitigation = counterfactual.get("mitigation_success", False)
    hazard_reduced = counterfactual.get("hazard_reduced", False)

    baseline_m = counterfactual.get("baseline_metrics") or {}
    patched_m = counterfactual.get("patched_metrics") or {}
    delta_m = counterfactual.get("delta_metrics") or {}

    baseline_count = int(baseline_m.get("unsafe_legal_state_count") or 0)
    patched_count = int(patched_m.get("unsafe_legal_state_count") or 0)
    _raw_delta = delta_m.get("unsafe_legal_state_count_delta") or (patched_count - baseline_count)
    count_delta = int(_raw_delta)

    repro = counterfactual.get("reproducibility") or {}
    same_seed = repro.get("same_seed", False)

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        "## Counterfactual Patch Review\n\n"
        f"Patch ID: {patch_id}\n"
        f"Patch type: {patch_type}\n"
        f"Verdict: {verdict}\n"
        f"Mitigation success: {mitigation}\n"
        f"Hazard reduced: {hazard_reduced}\n\n"
        "### Delta Metrics\n"
        f"- Baseline unsafe_legal_state_count: {baseline_count}\n"
        f"- Patched unsafe_legal_state_count: {patched_count}\n"
        f"- Delta: {count_delta:+d}\n"
        f"- Same seed used: {same_seed}\n\n"
        "Task: Write a concise (2-4 sentence) review of this counterfactual result. "
        "Explain what the verdict means in plain language. "
        "State that this is a deterministic stress-test scenario, not a real race. "
        "Identify at least one reason why the result may not generalise. "
        "Do not use phrases like 'proven safe', 'guaranteed', or 'calibrated recommendation'."
    )


# ---------------------------------------------------------------------------
# Assistant functions (injectable LLM)
# ---------------------------------------------------------------------------


def _extract_response_content(response: Any) -> str:
    """Extract string content from an LLM response robustly."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)


def summarize_audit_report_with_llm(
    audit_report: dict[str, Any],
    llm: Any | None = None,
) -> str:
    """Summarize an audit report using an LLM.

    Args:
        audit_report: Audit report dict from ``build_audit_report()``.
        llm: Injectable LLM (must have an ``invoke(prompt: str)`` method).
             If None, builds a ChatNVIDIA instance via ``build_nvidia_llm()``.

    Returns:
        LLM-generated summary string.
    """
    if llm is None:
        llm = build_nvidia_llm()
    prompt = build_audit_summary_prompt(audit_report)
    response = llm.invoke(prompt)
    return _extract_response_content(response)


def review_synthetic_families_with_llm(
    families: list[dict[str, Any]],
    llm: Any | None = None,
) -> str:
    """Review synthetic family specs using an LLM.

    Args:
        families: List of family spec dicts (from SyntheticFamilySpec or similar).
        llm: Injectable LLM.  If None, builds ChatNVIDIA.

    Returns:
        LLM-generated review string.
    """
    if llm is None:
        llm = build_nvidia_llm()
    prompt = build_synthetic_family_review_prompt(families)
    response = llm.invoke(prompt)
    return _extract_response_content(response)


def review_counterfactual_with_llm(
    counterfactual: dict[str, Any],
    llm: Any | None = None,
) -> str:
    """Review a counterfactual patch result using an LLM.

    Args:
        counterfactual: Counterfactual dict from audit report or evidence bundle.
        llm: Injectable LLM.  If None, builds ChatNVIDIA.

    Returns:
        LLM-generated review string.
    """
    if llm is None:
        llm = build_nvidia_llm()
    prompt = build_counterfactual_review_prompt(counterfactual)
    response = llm.invoke(prompt)
    return _extract_response_content(response)
