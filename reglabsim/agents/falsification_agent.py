"""NVIDIA/LangGraph DeepAgent falsification research agent (PR 7.2 → PR 7.3).

This module implements the first agentic layer for regulatory falsification
research. The agent uses deterministic falsification tools to search for
legal/grey-area unsafe scenarios.

PR 7.3 upgrades the campaign trace from a compact one-shot trace into a
structured, auditable campaign trace that records what the agent tried,
which tools it called, what evidence it found, what failed, and what
hypotheses it recommends next.

Key invariants:
- The agent chooses experiments; deterministic tools execute them.
- SafetyOracle / LegalVerdict / EvidenceBundle remain the source of truth.
- The agent must NOT invent evidence or decide safety/legal status.
- No NVIDIA API key is needed for unit tests (use deterministic runner).
- Campaign trace stores compact summaries, not full evidence bundles.
- Campaign trace never invents evidence; it only summarizes tool outputs.
- No raw event_log, full bundle, API keys, stack traces, or LLM CoT in trace.

Licensing note: deepagents and langgraph are optional dependencies.
Import them lazily inside functions — never at module top level.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from reglabsim.agents.campaign_trace import (
    AGENT_TRACE_SCHEMA,
    CAMPAIGN_TRACE_SCHEMA,
    MAX_EVENT_REFS_PER_STEP,
    CampaignTrace,
    CampaignTraceBuilder,
    build_next_hypotheses_from_trace,
    campaign_trace_to_dict,
    compact_text,
    extract_candidate_ids,
    extract_event_refs,
    extract_score,
    summarize_tool_input,
    summarize_tool_output,
)
from reglabsim.tools.falsification_tools import (
    build_best_candidate_audit_report_tool,
    list_synthetic_families_tool,
    run_falsification_search_tool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = AGENT_TRACE_SCHEMA
"""Schema version embedded in every agent output."""

_PREFERRED_POSITIVE_FAMILY = "confined_corner_grass"
"""Default family for the deterministic runner when no objective guides selection."""

_CAMPAIGN_TRACE_LIMITATIONS = [
    "Campaign trace stores compact summaries, not full evidence bundles.",
    "Tool outputs are deterministic evidence sources; the agent summary is interpretive.",
    "This is a regulatory falsification stress-test, not a calibrated real-world F1 prediction.",
    "Search is currently single-lap / synthetic-family based unless otherwise configured.",
]
"""Limitations included in every campaign trace."""


# ---------------------------------------------------------------------------
# Agent config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FalsificationAgentConfig:
    """Configuration for the falsification research agent.

    Attributes:
        model_name: NVIDIA model name override. Falls back to env vars
            NVIDIA_MODEL_NAME → F1LAB_NVIDIA_MODEL → default.
        max_iterations: Maximum reasoning iterations the agent may perform.
        max_trials_per_search: Hard cap on trials per falsification search.
        max_tool_calls: Maximum tool invocations per agent run.
        top_results_limit: Number of top search results to retain.
        require_evidence_refs: If True, the agent must cite tool-returned
            event_refs — never fabricate them.
        allow_real_llm: Must be True (or an explicit llm injected) to build
            a real NVIDIA deepagent. Default False keeps tests safe.
    """

    model_name: str | None = None
    max_iterations: int = 3
    max_trials_per_search: int = 25
    max_tool_calls: int = 12
    top_results_limit: int = 5
    require_evidence_refs: bool = True
    allow_real_llm: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
        if self.max_trials_per_search <= 0:
            raise ValueError(
                f"max_trials_per_search must be > 0, got {self.max_trials_per_search}"
            )
        if self.max_tool_calls <= 0:
            raise ValueError(f"max_tool_calls must be > 0, got {self.max_tool_calls}")

    def to_compact_dict(self) -> dict[str, Any]:
        """Return a compact, safe dict for trace agent_config."""
        return {
            "max_iterations": self.max_iterations,
            "max_trials_per_search": self.max_trials_per_search,
            "max_tool_calls": self.max_tool_calls,
            "top_results_limit": self.top_results_limit,
            "require_evidence_refs": self.require_evidence_refs,
        }


# ---------------------------------------------------------------------------
# AgentTraceStep (legacy, kept for backward compat)
# ---------------------------------------------------------------------------


@dataclass
class AgentTraceStep:
    """One compact step in the agent's campaign trace (legacy).

    Prefer CampaignTraceStep from campaign_trace module for new code.

    Attributes:
        step_index: Zero-based step number.
        action: Human-readable description of what the agent did.
        tool_name: Name of the tool invoked, or None for non-tool steps.
        tool_ok: Whether the tool returned ok=True, or None for non-tool steps.
        observation_summary: Compact summary of the observation/result.
        selected_family_id: Family ID if this step selected one.
        selected_candidate_id: Candidate ID if this step identified one.
        score: Exploit score if this step produced one.
        event_refs: Evidence event references from tool output (never fabricated).
    """

    step_index: int
    action: str
    tool_name: str | None
    tool_ok: bool | None
    observation_summary: str
    selected_family_id: str | None = None
    selected_candidate_id: str | None = None
    score: float | None = None
    event_refs: list[str] = field(default_factory=list)


def _resolve_model_name(config: FalsificationAgentConfig) -> str:
    """Resolve the NVIDIA model name from config or environment.

    Priority: config.model_name → NVIDIA_MODEL_NAME → F1LAB_NVIDIA_MODEL → default.
    """
    return (
        config.model_name
        or os.getenv("NVIDIA_MODEL_NAME")
        or os.getenv("F1LAB_NVIDIA_MODEL")
        or "nvidia/llama-3.1-nemotron-70b-instruct"
    )


# ---------------------------------------------------------------------------
# System prompt (Task 6 / Task 14)
# ---------------------------------------------------------------------------


def build_falsification_agent_system_prompt(
    config: FalsificationAgentConfig,
) -> str:
    """Build the system prompt for the falsification research agent.

    The prompt explicitly constrains the agent to remain evidence-bound
    and prohibits overclaiming or inventing evidence. Updated for PR 7.3
    to instruct the agent to return structured trace-relevant content.
    """
    return (
        "You are a regulatory falsification research agent.\n"
        "Your job is to find legal or grey-area unsafe scenarios using\n"
        "deterministic falsification tools.\n\n"
        "## Rules — you MUST follow these\n\n"
        "1. Tools are the source of truth. Only tool outputs count as evidence.\n"
        "2. SafetyOracle (the deterministic safety-verdict subsystem) decides\n"
        " safety status. LegalVerdict (the legal-verdict subsystem) decides\n"
        " legal status. You may NOT override, infer, or fabricate those.\n"
        "3. Do not invent event_refs. Only use event references returned by tools.\n"
        "4. Do not claim proof of real F1 behavior — this is a synthetic simulation.\n"
        "5. Do not claim calibrated regulatory recommendation.\n"
        "6. Do not use phrases like 'proven safe', 'guaranteed', or\n"
        " 'calibrated recommendation'.\n"
        "7. Prefer compact structured output.\n"
        f"8. Respect max iterations ({config.max_iterations}) and\n"
        f" max trials ({config.max_trials_per_search}).\n"
        "9. If no exploit is found, say so honestly.\n"
        "10. You may propose hypotheses, but only tool outputs count as evidence.\n"
        "11. Never fabricate unsafe_legal_state events, scores, metrics, or event_refs.\n"
        "12. Do not modify RaceMicrokernel, SafetyOracle, scoring semantics,\n"
        " or metrics.\n"
        "13. Do not reference real track names (Suzuka, Monaco, Singapore, etc.).\n"
        "14. When possible, mention selected family_id, selected candidate_id,\n"
        " score, unsafe_legal_state_count, and event_refs from tool outputs.\n"
        "15. Do not include raw event logs or full bundles in your output.\n"
        "16. Do not expose hidden reasoning or chain-of-thought.\n"
        "17. Provide concise rationale / evidence summary only.\n\n"
        "## Available tools\n\n"
        "- list_synthetic_families: list all synthetic circuit families\n"
        "- generate_falsification_candidates: generate parameter candidates\n"
        "- run_falsification_candidate: run one candidate through the microkernel\n"
        "- run_falsification_search: run a search over one family\n"
        "- build_best_candidate_audit_report: build audit report for best candidate\n\n"
        "## Output format\n\n"
        "When done, provide:\n"
        "- A concise human-readable summary\n"
        "- The best finding (family_id, candidate_id, score, event_refs)\n"
        "- Next hypotheses to test\n"
        "- Limitations of this run\n"
    )


# ---------------------------------------------------------------------------
# DeepAgent builder
# ---------------------------------------------------------------------------


class _DeepAgentError(RuntimeError):
    """Raised when the deepagent builder cannot proceed safely."""


def build_falsification_deepagent(
    llm: Any | None = None,
    config: FalsificationAgentConfig | None = None,
) -> Any:
    """Build a falsification deepagent using deepagents.create_deep_agent.

    Args:
        llm: Injectable LLM instance. If None, a real NVIDIA LLM is built
            only when config.allow_real_llm is True.
        config: Agent configuration. Uses defaults if None.

    Returns:
        A compiled deepagent graph/object.

    Raises:
        _DeepAgentError: If no llm is provided and allow_real_llm is False.
        RuntimeError: If deepagents or langchain-core is not installed.
    """
    if config is None:
        config = FalsificationAgentConfig()

    # Safety gate: no LLM provided, no real LLM allowed
    if llm is None and not config.allow_real_llm:
        raise _DeepAgentError(
            "build_falsification_deepagent requires either an injected llm "
            "or config.allow_real_llm=True. Set allow_real_llm=True for "
            "manual/runtime use, or inject a fake/mock LLM for tests."
        )

    # Lazy import — deepagents is an optional dependency
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is required for build_falsification_deepagent. "
            "Install the 'agents' extra: pip install f1lab-ai[agents]"
        ) from exc

    # Build LLM if not injected
    if llm is None:
        from reglabsim.llm.nvidia_assistant import build_nvidia_llm

        model_name = _resolve_model_name(config)
        llm = build_nvidia_llm(model=model_name)

    # Get LangChain-wrapped tools
    tools = _get_langchain_tools_safe()

    system_prompt = build_falsification_agent_system_prompt(config)

    # Build deepagent — all keyword args per current API:
    # create_deep_agent(model=…, tools=…, *, system_prompt=…, …)
    # model and tools are positional-or-keyword; system_prompt is keyword-only.
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


def _get_langchain_tools_safe() -> list[Any]:
    """Return LangChain tools, raising RuntimeError if langchain is absent."""
    try:
        from reglabsim.tools.falsification_tools import as_langchain_tools

        return as_langchain_tools()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "LangChain tool adapters are required for the deepagent. "
            "Install the 'agents' extra: pip install f1lab-ai[agents]"
        ) from exc


# ---------------------------------------------------------------------------
# DeepAgent runner (Task 13 — basic trace integration)
# ---------------------------------------------------------------------------


def run_falsification_agent(
    objective: str,
    config: FalsificationAgentConfig | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Run the falsification research agent with a real deepagent.

    The deepagent runner produces a basic campaign trace (start, invoke,
    summary). Rich tool-call trace extraction is limited in PR 7.3 and
    will be enhanced in future PRs.

    Args:
        objective: Research objective / question for the agent.
        config: Agent configuration. Uses defaults if None.
        llm: Injectable LLM. If None, a real NVIDIA LLM is built
            only when config.allow_real_llm is True.

    Returns:
        Structured output dict with schema_version, summary, best_finding,
        campaign_trace (full object), next_hypotheses, and limitations.
    """
    if config is None:
        config = FalsificationAgentConfig()

    builder = CampaignTraceBuilder(
        objective=objective,
        mode="deepagent_falsification_research",
        agent_config=config.to_compact_dict(),
        seed=None,
    )

    # Add standard limitations
    for lim in _CAMPAIGN_TRACE_LIMITATIONS:
        builder.add_limitation(lim)

    # Step 0: Start
    builder.add_step(
        phase="start",
        action="agent_start",
        observation_summary=f"Objective: {objective}",
    )

    try:
        agent = build_falsification_deepagent(llm=llm, config=config)
    except (_DeepAgentError, RuntimeError) as exc:
        builder.add_failed_attempt(
            step_index=0,
            tool_name="build_deepagent",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        trace = builder.build()
        return _error_output(type(exc).__name__, str(exc), trace)

    # Build user message with objective and limits
    user_message = (
        f"Research objective: {objective}\n\n"
        f"Constraints: max_iterations={config.max_iterations}, "
        f"max_trials={config.max_trials_per_search}, "
        f"max_tool_calls={config.max_tool_calls}.\n\n"
        "Use the available tools to search for unsafe legal scenarios. "
        "Provide a summary, best finding, next hypotheses, and limitations."
    )

    # Invoke agent
    try:
        raw_output = agent.invoke({"messages": [("user", user_message)]})
        builder.add_step(
            phase="execute",
            action="agent_invoke",
            observation_summary="DeepAgent invocation completed",
        )
    except Exception as exc:
        builder.add_failed_attempt(
            step_index=builder._step_counter,
            tool_name="deepagent_invoke",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        trace = builder.build()
        return _error_output(
            type(exc).__name__,
            f"DeepAgent invocation failed: {exc}",
            trace,
        )

    # Deterministic post-processing: extract what we can from raw output
    summary, best_finding, next_hypotheses_raw, limitations = (
        _post_process_agent_output(raw_output)
    )

    builder.add_step(
        phase="report",
        action="agent_summary",
        observation_summary=compact_text(summary),
    )

    # Extract evidence from best_finding if available
    if best_finding is not None:
        event_refs = list(best_finding.get("event_refs") or [])
        score_val = None
        raw_score = best_finding.get("score")
        if raw_score is not None:
            try:
                score_val = float(raw_score)
            except (TypeError, ValueError):
                pass

        builder.add_finding(
            family_id=best_finding.get("family_id"),
            candidate_id=best_finding.get("candidate_id"),
            score=score_val,
            unsafe_legal_state_count=int(
                best_finding.get("unsafe_legal_state_count") or 0
            ),
            max_hazard_score=best_finding.get("max_hazard_score"),
            mean_hazard_score=best_finding.get("mean_hazard_score"),
            event_refs=event_refs,
            summary=compact_text(
                f"Best finding from deepagent: "
                f"family={best_finding.get('family_id')}, "
                f"candidate={best_finding.get('candidate_id')}"
            ),
        )

    # Add DeepAgent-specific limitation
    builder.add_limitation(
        "DeepAgent internal tool-call trace extraction is limited in PR 7.3."
    )

    # Add limitations from post-processed output
    for lim in limitations:
        builder.add_limitation(lim)

    # Build deterministic hypotheses from trace
    trace = builder.build()
    hypotheses = build_next_hypotheses_from_trace(trace)

    # Convert hypotheses to the format expected in top-level output
    next_hypotheses_output = [
        {
            "hypothesis_id": h.hypothesis_id,
            "basis": h.basis,
            "proposed_action": h.proposed_action,
            "expected_signal": h.expected_signal,
            "priority": h.priority,
        }
        for h in hypotheses
    ]

    # Also include any string hypotheses from the deepagent output
    for h_raw in next_hypotheses_raw:
        if isinstance(h_raw, str) and h_raw:
            next_hypotheses_output.append(
                {
                    "hypothesis_id": f"hyp_llm_{len(next_hypotheses_output):04d}",
                    "basis": h_raw,
                    "proposed_action": "See basis.",
                    "expected_signal": "To be determined by next campaign.",
                    "priority": "low",
                }
            )

    model_name = (
        _resolve_model_name(config)
        if (llm is None or config.allow_real_llm)
        else "injected"
    )

    # Build artifacts (compact references only)
    trace_dict = campaign_trace_to_dict(trace)

    return {
        "schema_version": _SCHEMA_VERSION,
        "ok": True,
        "mode": "deepagent_falsification_research",
        "model_name": model_name,
        "summary": summary,
        "best_finding": best_finding,
        "campaign_trace": trace_dict,
        "campaign_trace_steps": trace_dict.get("steps", []),
        "next_hypotheses": next_hypotheses_output,
        "limitations": list(trace.limitations),
    }


# ---------------------------------------------------------------------------
# Deterministic fallback runner (Task 6 / Task 7 / Task 8 / Task 9)
# ---------------------------------------------------------------------------


def run_falsification_agent_deterministic(
    objective: str,
    config: FalsificationAgentConfig | None = None,
) -> dict[str, Any]:
    """Run a deterministic (non-LLM) falsification campaign for tests and CI.

    This function simulates the research flow by directly calling tools:
    list families → choose a positive family → run search → build audit report.
    It proves the campaign trace schema and tools work together.

    PR 7.3 upgrades the campaign_trace from a list of step dicts to a
    full CampaignTrace object. For backward compatibility, the top-level
    output also includes "campaign_trace_steps" as a flat list.

    This is NOT the real agent. It is a deterministic harness for tests.

    Args:
        objective: Research objective (used in trace, does not affect tool calls).
        config: Agent configuration. Uses defaults if None.

    Returns:
        Structured output dict matching the deepagent runner schema with
        a full campaign_trace object.
    """
    if config is None:
        config = FalsificationAgentConfig()

    builder = CampaignTraceBuilder(
        objective=objective,
        mode="deterministic_falsification_agent",
        agent_config=config.to_compact_dict(),
        seed=42,
    )

    # Add standard limitations
    for lim in _CAMPAIGN_TRACE_LIMITATIONS:
        builder.add_limitation(lim)

    # Step 0: Start
    builder.add_step(
        phase="start",
        action="agent_start",
        observation_summary=f"Objective: {objective}",
    )

    # ---------------------------------------------------------------
    # Step 1: List families
    # ---------------------------------------------------------------
    families_out = list_synthetic_families_tool()

    # Record tool call
    families_tc = builder.add_tool_call(
        tool_name="list_synthetic_families_tool",
        ok=families_out["ok"],
        input_summary=summarize_tool_input({}),
        output_summary=summarize_tool_output(families_out),
    )

    families_obs = _summarize_families(families_out)
    families_event_refs = extract_event_refs(families_out)
    extract_candidate_ids(families_out)

    builder.add_step(
        phase="explore",
        action="list_synthetic_families",
        observation_summary=families_obs,
        tool_call_id=families_tc.call_id,
        tool_name="list_synthetic_families_tool",
        tool_ok=families_out["ok"],
        event_refs=families_event_refs,
    )

    if not families_out["ok"]:
        builder.add_failed_attempt(
            step_index=builder._step_counter - 1,
            tool_name="list_synthetic_families_tool",
            error_type="ToolError",
            error_message="list_synthetic_families failed",
        )
        trace = builder.build()
        return _error_output("ToolError", "list_synthetic_families failed", trace)

    # ---------------------------------------------------------------
    # Step 2: Choose a positive family
    # ---------------------------------------------------------------
    families_list = families_out["result"]["families"]
    positive_families = [
        f for f in families_list if f.get("expected_unsafe_legal") is True
    ]
    if not positive_families:
        trace = builder.build()
        return _error_output(
            "NoFamilies",
            "No positive (expected_unsafe_legal=True) families found",
            trace,
        )

    # Prefer confined_corner_grass if available, else first positive
    chosen_family = _PREFERRED_POSITIVE_FAMILY
    if not any(f["family_id"] == chosen_family for f in positive_families):
        chosen_family = str(positive_families[0].get("family_id", ""))

    builder.add_step(
        phase="explore",
        action="select_family",
        observation_summary=f"Selected family: {chosen_family}",
        selected_family_id=chosen_family,
    )

    # ---------------------------------------------------------------
    # Step 3: Run falsification search
    # ---------------------------------------------------------------
    search_kwargs: dict[str, Any] = {
        "family_id": chosen_family,
        "seed": 42,
        "max_trials": config.max_trials_per_search,
    }
    search_out = run_falsification_search_tool(**search_kwargs)

    search_event_refs = extract_event_refs(search_out)
    search_candidate_ids = extract_candidate_ids(search_out)
    search_score = extract_score(search_out)

    search_tc = builder.add_tool_call(
        tool_name="run_falsification_search_tool",
        ok=search_out["ok"],
        input_summary=summarize_tool_input(search_kwargs),
        output_summary=summarize_tool_output(search_out),
        event_refs=search_event_refs,
        candidate_ids=search_candidate_ids,
        score=search_score,
    )

    best_candidate_id: str | None = None
    best_score: float | None = None
    best_event_refs: list[str] = []
    search_summary = "Search completed"

    if search_out["ok"] and search_out["result"].get("best_candidate"):
        bc = search_out["result"]["best_candidate"]
        best_candidate_id = str(bc.get("candidate_id", ""))
        best_score = float(bc.get("score", 0.0))
        best_event_refs = list(bc.get("event_refs") or [])
        unsafe_count = int(bc.get("unsafe_legal_state_count", 0))
        max_hazard = bc.get("max_hazard_score")
        search_summary = (
            f"Best: {best_candidate_id}, score={best_score:.2f}, "
            f"unsafe_legal={unsafe_count}, max_hazard={max_hazard}"
        )

        if not search_out["ok"]:
            builder.add_failed_attempt(
                step_index=builder._step_counter,
                tool_name="run_falsification_search_tool",
                error_type="ToolError",
                error_message="run_falsification_search failed",
                input_summary=search_kwargs,
            )

    builder.add_step(
        phase="evaluate",
        action="run_falsification_search",
        observation_summary=search_summary,
        tool_call_id=search_tc.call_id,
        tool_name="run_falsification_search_tool",
        tool_ok=search_out["ok"],
        selected_family_id=chosen_family,
        selected_candidate_id=best_candidate_id,
        score=best_score,
        event_refs=search_event_refs or best_event_refs,
    )

    # ---------------------------------------------------------------
    # Step 4: Build audit report for best candidate
    # ---------------------------------------------------------------
    audit_kwargs: dict[str, Any] = {
        "family_id": chosen_family,
        "seed": 42,
        "max_trials": config.max_trials_per_search,
    }
    audit_out = build_best_candidate_audit_report_tool(**audit_kwargs)

    audit_event_refs = extract_event_refs(audit_out)
    audit_candidate_ids = extract_candidate_ids(audit_out)

    audit_tc = builder.add_tool_call(
        tool_name="build_best_candidate_audit_report_tool",
        ok=audit_out["ok"],
        input_summary=summarize_tool_input(audit_kwargs),
        output_summary=summarize_tool_output(audit_out),
        event_refs=audit_event_refs,
        candidate_ids=audit_candidate_ids,
    )

    audit_summary = "Audit report built"
    unsafe_legal_count = 0
    audit_max_hazard: float | None = None
    audit_event_refs_final: list[str] = []
    audit_schema_version = ""
    audit_markdown_chars = 0

    if audit_out["ok"]:
        report = audit_out["result"].get("audit_report", {})
        summary_data = report.get("summary", {})
        unsafe_legal_count = int(summary_data.get("unsafe_legal_state_count", 0))
        audit_max_hazard = summary_data.get("max_hazard_score")
        audit_event_refs_final = list(
            summary_data.get("unsafe_legal_event_refs") or []
        )
        audit_schema_version = report.get("schema_version", "")
        audit_summary = (
            f"Audit: unsafe_legal={unsafe_legal_count}, "
            f"max_hazard={audit_max_hazard}"
        )
        md = audit_out["result"].get("markdown_excerpt", "")
        audit_markdown_chars = len(md) if isinstance(md, str) else 0

        if not audit_out["ok"]:
            builder.add_failed_attempt(
                step_index=builder._step_counter,
                tool_name="build_best_candidate_audit_report_tool",
                error_type="ToolError",
                error_message=(
            f"Audit failed: "
            f"{audit_out.get('error', {}).get('message', 'unknown')}"
        ),
                input_summary=audit_kwargs,
            )
    else:
        audit_summary = (
            f"Audit failed: {audit_out.get('error', {}).get('message', 'unknown')}"
        )

    builder.add_step(
        phase="report",
        action="build_best_candidate_audit_report",
        observation_summary=audit_summary,
        tool_call_id=audit_tc.call_id,
        tool_name="build_best_candidate_audit_report_tool",
        tool_ok=audit_out["ok"],
        selected_family_id=chosen_family,
        selected_candidate_id=best_candidate_id,
    )

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    exploit_found = unsafe_legal_count > 0 or bool(best_event_refs)
    if exploit_found:
        summary = (
            f"Deterministic campaign found unsafe legal evidence in family "
            f"'{chosen_family}'. Best candidate {best_candidate_id} scored "
            f"{best_score:.2f} with {unsafe_legal_count} unsafe legal state(s) "
            f"and max hazard {audit_max_hazard}. This is a deterministic "
            f"stress-test finding, not a calibrated regulatory recommendation."
        )
    else:
        summary = (
            f"Deterministic campaign over family '{chosen_family}' found no "
            f"unsafe legal evidence with the current parameter space. This does "
            f"not prove safety — absence of evidence is not evidence of absence."
        )

    builder.add_step(
        phase="report",
        action="agent_summary",
        observation_summary=compact_text(summary),
    )

    # ---------------------------------------------------------------
    # Best finding (Task 8)
    # ---------------------------------------------------------------
    if exploit_found and best_candidate_id is not None:
        merged_refs = list(
            dict.fromkeys(best_event_refs + audit_event_refs_final)
        )
        builder.add_finding(
            family_id=chosen_family,
            candidate_id=best_candidate_id,
            score=best_score,
            unsafe_legal_state_count=unsafe_legal_count,
            max_hazard_score=audit_max_hazard,
            mean_hazard_score=None,
            event_refs=merged_refs,
            audit_report_ref="audit_report:best_candidate",
            summary=compact_text(
                f"Unsafe legal state found in {chosen_family}: "
                f"candidate={best_candidate_id}, "
                f"unsafe_count={unsafe_legal_count}, "
                f"max_hazard={audit_max_hazard}"
            ),
        )

    # ---------------------------------------------------------------
    # Artifacts (Task 11)
    # ---------------------------------------------------------------
    artifact_audit: dict[str, Any] = {
        "artifact_id": "audit_report:best_candidate",
        "schema_version": audit_schema_version,
        "unsafe_legal_state_count": unsafe_legal_count,
        "event_refs": audit_event_refs_final[:MAX_EVENT_REFS_PER_STEP],
    }
    if audit_markdown_chars > 0:
        artifact_audit["markdown_excerpt_chars"] = audit_markdown_chars

    builder.set_artifact(
        "audit_reports",
        [artifact_audit],
    )

    candidate_refs: list[dict[str, Any]] = []
    if best_candidate_id is not None:
        candidate_refs.append(
            {
                "candidate_id": best_candidate_id,
                "family_id": chosen_family,
                "score": best_score,
            }
        )
    if candidate_refs:
        builder.set_artifact("candidate_refs", candidate_refs)

    # ---------------------------------------------------------------
    # Limitations (Task 12)
    # ---------------------------------------------------------------
    builder.add_limitation(
        f"Only {config.max_trials_per_search} trials were explored per family."
    )
    builder.add_limitation(
        "Only one synthetic family was tested in this campaign."
    )
    builder.add_limitation(
        "The search space is bounded and may miss edge cases."
    )
    builder.add_limitation(
        "Results depend on synthetic parameters, not real telemetry."
    )

    # ---------------------------------------------------------------
    # Build trace and hypotheses (Task 10)
    # ---------------------------------------------------------------
    trace = builder.build()
    hypotheses = build_next_hypotheses_from_trace(trace)

    # Convert hypotheses to output format
    next_hypotheses_output = [
        {
            "hypothesis_id": h.hypothesis_id,
            "basis": h.basis,
            "proposed_action": h.proposed_action,
            "expected_signal": h.expected_signal,
            "priority": h.priority,
        }
        for h in hypotheses
    ]

    # Build top-level best_finding dict
    best_finding: dict[str, Any] | None = None
    if best_candidate_id is not None:
        best_finding = {
            "family_id": chosen_family,
            "candidate_id": best_candidate_id,
            "score": best_score,
            "unsafe_legal_state_count": unsafe_legal_count,
            "max_hazard_score": audit_max_hazard,
            "event_refs": best_event_refs or audit_event_refs_final,
        }

    trace_dict = campaign_trace_to_dict(trace)

    return {
        "schema_version": _SCHEMA_VERSION,
        "ok": True,
        "mode": "deterministic_falsification_agent",
        "model_name": "deterministic_harness",
        "summary": summary,
        "best_finding": best_finding,
        "campaign_trace": trace_dict,
        "campaign_trace_steps": trace_dict.get("steps", []),
        "next_hypotheses": next_hypotheses_output,
        "limitations": list(trace.limitations),
    }


# ---------------------------------------------------------------------------
# Manual integration helper
# ---------------------------------------------------------------------------


def run_nvidia_falsification_agent_manual(
    objective: str,
) -> dict[str, Any]:
    """Run the falsification agent with real NVIDIA LLM (manual use only).

    Requires NVIDIA_API_KEY environment variable.
    Uses NVIDIA_MODEL_NAME or F1LAB_NVIDIA_MODEL for model selection.
    Not intended for unit tests — use run_falsification_agent_deterministic
    or run_falsification_agent with injected llm for testing.

    Args:
        objective: Research objective for the agent.

    Returns:
        Structured output dict.

    Raises:
        RuntimeError: If NVIDIA_API_KEY is not set.
    """
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is required for "
            "run_nvidia_falsification_agent_manual(). "
            "Export it before calling this function."
        )
    config = FalsificationAgentConfig(allow_real_llm=True)
    return run_falsification_agent(objective, config=config, llm=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trace_step(
    step_index: int,
    action: str,
    tool_name: str | None,
    tool_ok: bool | None,
    observation_summary: str,
    selected_family_id: str | None = None,
    selected_candidate_id: str | None = None,
    score: float | None = None,
    event_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact trace step dict from AgentTraceStep dataclass.

    Kept for backward compatibility. New code should use CampaignTraceBuilder.
    """
    step = AgentTraceStep(
        step_index=step_index,
        action=action,
        tool_name=tool_name,
        tool_ok=tool_ok,
        observation_summary=observation_summary,
        selected_family_id=selected_family_id,
        selected_candidate_id=selected_candidate_id,
        score=score,
        event_refs=list(event_refs) if event_refs else [],
    )
    return asdict(step)


def _summarize_families(families_out: dict[str, Any]) -> str:
    """Build a compact summary of the list_synthetic_families tool output."""
    if not families_out["ok"]:
        return f"Error: {families_out.get('error', {}).get('message', 'unknown')}"
    families = families_out["result"]["families"]
    positive = [f for f in families if f.get("expected_unsafe_legal") is True]
    control = [f for f in families if f.get("expected_unsafe_legal") is False]
    return (
        f"Found {len(families)} families: "
        f"{len(positive)} positive (stress), {len(control)} control (baseline)"
    )


def _generate_next_hypotheses(
    family_id: str, exploit_found: bool
) -> list[str]:
    """Generate next hypotheses based on campaign results (legacy string format).

    Prefer build_next_hypotheses_from_trace() for new code.
    """
    hypotheses: list[str] = []
    if exploit_found:
        hypotheses.append(
            f"Increase trial count for family '{family_id}' to explore "
            f"parameter space more thoroughly."
        )
        hypotheses.append(
            "Test other positive families to compare exploit scores "
            "across different segment geometries."
        )
        hypotheses.append(
            "Vary the seed to check if findings are robust across "
            "different random parameter draws."
        )
    else:
        hypotheses.append(
            f"Family '{family_id}' showed no exploits with current parameters. "
            f"Try different seed or wider parameter ranges."
        )
        hypotheses.append(
            "Test other positive families that may have different "
            "risk profiles or segment geometries."
        )
        hypotheses.append(
            "Test control family (wide_corner_asphalt_control) to verify "
            "the pipeline correctly reports no unsafe legal states."
        )
    return hypotheses


def _post_process_agent_output(
    raw_output: Any,
) -> tuple[str, dict[str, Any] | None, list[str], list[str]]:
    """Deterministic post-processing of deepagent output.

    Extracts what it can and wraps it in the expected schema.
    Never crashes on weird LLM output.

    Returns:
        (summary, best_finding, next_hypotheses, limitations)
    """
    summary = ""
    best_finding: dict[str, Any] | None = None
    next_hypotheses: list[str] = []
    limitations: list[str] = [
        "This is a deterministic stress-test, not a calibrated regulatory recommendation.",
        "Agent output is LLM-generated and may contain inaccuracies.",
        "Only tool-returned evidence is reliable.",
    ]

    # Try to extract content from the raw output
    if isinstance(raw_output, dict):
        # LangGraph agent output typically has a 'messages' key
        messages = raw_output.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                summary = str(last_msg.content)
            elif isinstance(last_msg, dict):
                summary = str(last_msg.get("content", ""))
            elif isinstance(last_msg, str):
                summary = last_msg
            else:
                summary = str(last_msg)

        # Try to extract structured data if the agent provided it
        output_data = raw_output.get("output") or raw_output.get("result")
        if isinstance(output_data, dict):
            if "best_finding" in output_data:
                best_finding = output_data["best_finding"]
            if "next_hypotheses" in output_data:
                next_hypotheses = list(output_data["next_hypotheses"])
            if "limitations" in output_data:
                limitations = list(output_data["limitations"])
    elif isinstance(raw_output, str):
        summary = raw_output
    elif hasattr(raw_output, "content"):
        summary = str(raw_output.content)
    else:
        summary = str(raw_output)

    # Fallback summary
    if not summary:
        summary = "Agent completed but produced no readable output."

    return summary, best_finding, next_hypotheses, limitations


def _error_output(
    error_type: str,
    error_message: str,
    campaign_trace: CampaignTrace | dict[str, Any] | list[Any],
) -> dict[str, Any]:
    """Build the standard error output envelope.

    Args:
        error_type: Exception class name.
        error_message: Compact error message.
        campaign_trace: CampaignTrace instance, dict, or legacy list.
    """
    if isinstance(campaign_trace, CampaignTrace):
        trace_dict = campaign_trace_to_dict(campaign_trace)
    elif isinstance(campaign_trace, list):
        # Legacy: bare list of step dicts → wrap in minimal dict
        trace_dict = {
            "schema": CAMPAIGN_TRACE_SCHEMA,
            "steps": campaign_trace,
        }
    else:
        trace_dict = campaign_trace

    return {
        "schema_version": _SCHEMA_VERSION,
        "ok": False,
        "error": {
            "type": error_type,
            "message": error_message,
        },
        "campaign_trace": trace_dict,
        "campaign_trace_steps": trace_dict.get("steps", []),
    }
