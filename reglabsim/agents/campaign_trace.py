"""Agent campaign trace / experiment memory (PR 7.3).

Structured, auditable campaign trace that records what the falsification
agent tried, which tools it called, what evidence it found, what failed,
and what hypotheses it recommends next.

Key invariants:
- Campaign trace records evidence produced by deterministic tools only.
- It must never invent evidence. It must only summarize tool outputs.
- No raw full bundles, raw event logs, full unsafe_legal_state payloads,
  API keys, stack traces, LLM chain-of-thought, or hidden prompts.
- Deterministic: same inputs produce same trace.
- Compact and JSON-serializable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

# ---------------------------------------------------------------------------
# Constants (Task 1)
# ---------------------------------------------------------------------------

CAMPAIGN_TRACE_SCHEMA = "agent_campaign_trace.v0"
"""Schema version embedded in every campaign trace."""

AGENT_TRACE_SCHEMA = "falsification_agent.v0"
"""Schema version for the top-level agent output."""

MAX_TRACE_OBSERVATION_CHARS = 600
"""Maximum character length for observation summaries."""

MAX_RAW_OUTPUT_CHARS = 2000
"""Maximum character length for raw output excerpts."""

MAX_EVENT_REFS_PER_STEP = 10
"""Maximum event references retained per trace step."""

MAX_TOOL_RESULT_KEYS = 30
"""Maximum number of keys retained in a tool result summary."""

# Keys that must never appear in trace summaries — security & compactness.
_FORBIDDEN_INPUT_KEYS = frozenset({
    "api_key", "NVIDIA_API_KEY", "nvidia_api_key", "secret",
    "password", "token", "credential", "bundle", "event_log",
    "raw_event_log", "state_snapshots", "unsafe_legal_states",
})

_FORBIDDEN_OUTPUT_KEYS = frozenset({
    "event_log", "raw_event_log", "bundle", "state_snapshots",
    "unsafe_legal_states", "raw_event", "full_payload",
})


# ---------------------------------------------------------------------------
# Dataclasses (Task 2)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """Compact record of a single tool invocation.

    Attributes:
        call_id: Stable identifier (e.g. "tool_0000").
        step_index: Zero-based step number in the campaign.
        tool_name: Name of the deterministic tool called.
        ok: Whether the tool returned ok=True.
        started_at: Optional ISO-8601 timestamp (disabled by default for determinism).
        finished_at: Optional ISO-8601 timestamp.
        duration_ms: Optional wall-clock duration in milliseconds.
        input_summary: Compact summary of tool inputs (no secrets/raw bundles).
        output_summary: Compact summary of tool outputs (no raw logs/bundles).
        error_type: Exception class name if ok=False, else None.
        error_message: Compact error message if ok=False, else None.
        event_refs: Evidence event references from tool output.
        candidate_ids: Candidate identifiers referenced in the output.
        score: Exploit score if the tool produced one.
    """

    call_id: str
    step_index: int
    tool_name: str
    ok: bool
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    event_refs: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    score: float | None = None


@dataclass
class CampaignTraceStep:
    """One compact step in the agent's campaign trace.

    Attributes:
        step_index: Zero-based step number.
        phase: High-level phase (e.g. "explore", "evaluate", "report").
        action: Human-readable description of what the agent did.
        observation_summary: Compact summary of the observation/result.
        tool_call_id: ID of the associated ToolCallRecord, or None.
        tool_name: Name of the tool invoked, or None for non-tool steps.
        tool_ok: Whether the tool returned ok=True, or None for non-tool steps.
        selected_family_id: Family ID if this step selected one.
        selected_candidate_id: Candidate ID if this step identified one.
        score: Exploit score if this step produced one.
        event_refs: Evidence event references from tool output (never fabricated).
    """

    step_index: int
    phase: str
    action: str
    observation_summary: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_ok: bool | None = None
    selected_family_id: str | None = None
    selected_candidate_id: str | None = None
    score: float | None = None
    event_refs: list[str] = field(default_factory=list)


@dataclass
class CampaignFinding:
    """A significant finding from the falsification campaign.

    Attributes:
        finding_id: Stable identifier (e.g. "finding_0000").
        family_id: Synthetic family that produced the finding.
        candidate_id: Best candidate identifier.
        score: Exploit score of the finding.
        unsafe_legal_state_count: Number of unsafe legal states found.
        max_hazard_score: Maximum hazard score observed.
        mean_hazard_score: Mean hazard score observed.
        event_refs: Evidence event references (from tool outputs only).
        audit_report_ref: Reference to the associated audit report artifact.
        summary: Compact human-readable summary of the finding.
    """

    finding_id: str
    family_id: str | None = None
    candidate_id: str | None = None
    score: float | None = None
    unsafe_legal_state_count: int = 0
    max_hazard_score: float | None = None
    mean_hazard_score: float | None = None
    event_refs: list[str] = field(default_factory=list)
    audit_report_ref: str | None = None
    summary: str = ""


@dataclass
class CampaignHypothesis:
    """A hypothesis for future investigation based on campaign results.

    Attributes:
        hypothesis_id: Stable identifier (e.g. "hyp_0000").
        basis: Evidence-based rationale for the hypothesis.
        proposed_action: What to do next to test this hypothesis.
        expected_signal: What evidence would support or refute the hypothesis.
        priority: "high", "medium", or "low".
    """

    hypothesis_id: str
    basis: str
    proposed_action: str
    expected_signal: str
    priority: str = "medium"


@dataclass
class CampaignTrace:
    """Structured, auditable campaign trace for a falsification experiment.

    Attributes:
        schema_version: Schema identifier (CAMPAIGN_TRACE_SCHEMA).
        campaign_id: Deterministic identifier for this campaign.
        objective: Research objective string.
        mode: Execution mode (e.g. "deterministic_falsification_agent").
        seed: PRNG seed used for deterministic replay.
        agent_config: Compact agent configuration dict.
        steps: Ordered list of campaign trace steps.
        tool_calls: Records of all deterministic tool invocations.
        best_findings: Significant findings extracted from tool evidence.
        failed_attempts: Compact records of tool failures.
        next_hypotheses: Evidence-based hypotheses for future campaigns.
        artifacts: Compact references to audit reports and candidates.
        limitations: Disclaimers and caveats for this campaign.
    """

    schema_version: str
    campaign_id: str
    objective: str
    mode: str
    seed: int | None
    agent_config: dict[str, Any] = field(default_factory=dict)
    steps: list[CampaignTraceStep] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    best_findings: list[CampaignFinding] = field(default_factory=list)
    failed_attempts: list[dict[str, Any]] = field(default_factory=list)
    next_hypotheses: list[CampaignHypothesis] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers (Task 3)
# ---------------------------------------------------------------------------


def dataclass_to_dict(value: Any) -> dict[str, Any] | list[Any] | Any:
    """Recursively convert a dataclass (or nested dataclasses) to a dict.

    Falls back to the original value for non-dataclass types so that
    primitives (str, int, float, None) pass through unchanged.

    Args:
        value: A dataclass instance, list, dict, or primitive.

    Returns:
        A JSON-safe dict/list/primitive representation.
    """
    if hasattr(value, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    return value


def campaign_trace_to_dict(trace: CampaignTrace) -> dict[str, Any]:
    """Convert a CampaignTrace to a JSON-serializable dict.

    Args:
        trace: The CampaignTrace instance to serialize.

    Returns:
        A dict suitable for ``json.dumps(...)``.
    """
    trace_dict = dataclass_to_dict(trace)
    if not isinstance(trace_dict, dict):
        raise TypeError("CampaignTrace serialization must produce a dict")
    return trace_dict


# ---------------------------------------------------------------------------
# Compact summary utilities (Task 4)
# ---------------------------------------------------------------------------


def compact_text(value: Any, max_chars: int = MAX_TRACE_OBSERVATION_CHARS) -> str:
    """Safely convert any value to a compact, truncated string.

    Args:
        value: Any value to stringify.
        max_chars: Maximum character length before truncation.

    Returns:
        A string representation, truncated with "..." suffix if needed.
    """
    if value is None:
        return ""
    text = str(value)
    # Replace newlines for compact single-line representation
    text = text.replace("\n", " ").replace("\r", "")
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def extract_event_refs(payload: Any, limit: int = MAX_EVENT_REFS_PER_STEP) -> list[str]:
    """Extract event references from common tool output locations.

    Searches a nested dict/list payload for event refs in known locations:
    - payload["result"]["event_refs"]
    - payload["result"]["best_candidate"]["event_refs"]
    - payload["result"]["audit_report"]["unsafe_legal_events"]
    - payload["result"]["bundle_summary"]["event_refs"]
    - payload["event_refs"]
    - payload["unsafe_legal_event_refs"]

    Args:
        payload: Tool output dict (may be nested).
        limit: Maximum number of unique refs to return.

    Returns:
        Deduplicated list of event ref strings in stable order.
    """
    if not isinstance(payload, dict):
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(refs: Any) -> None:
        if not isinstance(refs, (list, tuple)):
            return
        for ref in refs:
            if isinstance(ref, str) and ref not in seen:
                seen.add(ref)
                result.append(ref)
                if len(result) >= limit:
                    return

    # Common locations (ordered by specificity)
    sub = payload.get("result")
    if isinstance(sub, dict):
        _add(sub.get("event_refs"))
        bc = sub.get("best_candidate")
        if isinstance(bc, dict):
            _add(bc.get("event_refs"))
        ar = sub.get("audit_report")
        if isinstance(ar, dict):
            _add(ar.get("unsafe_legal_events"))
            _add(ar.get("unsafe_legal_event_refs"))
        bs = sub.get("bundle_summary")
        if isinstance(bs, dict):
            _add(bs.get("event_refs"))
            m = bs.get("metrics")
            if isinstance(m, dict):
                _add(m.get("unsafe_legal_event_refs"))
        # Also try summary.unsafe_legal_event_refs
        sm = sub.get("summary") if isinstance(sub.get("summary"), dict) else None
        if isinstance(sm, dict):
            _add(sm.get("unsafe_legal_event_refs"))

    _add(payload.get("event_refs"))
    _add(payload.get("unsafe_legal_event_refs"))

    return result[:limit]


def extract_candidate_ids(payload: Any, limit: int = 10) -> list[str]:
    """Extract candidate IDs from common tool output locations.

    Searches:
    - payload["result"]["candidate_id"]
    - payload["result"]["best_candidate"]["candidate_id"]
    - payload["result"]["candidates"][...]["candidate_id"]
    - payload["result"]["top_results"][...]["candidate_id"]

    Args:
        payload: Tool output dict.
        limit: Maximum number of candidate IDs to return.

    Returns:
        Deduplicated list of candidate ID strings.
    """
    if not isinstance(payload, dict):
        return []

    seen: set[str] = set()
    result: list[str] = []

    def _add(cid: Any) -> None:
        if isinstance(cid, str) and cid not in seen and cid:
            seen.add(cid)
            result.append(cid)

    sub = payload.get("result")
    if isinstance(sub, dict):
        _add(sub.get("candidate_id"))
        bc = sub.get("best_candidate")
        if isinstance(bc, dict):
            _add(bc.get("candidate_id"))
        for item in sub.get("candidates") or []:
            if isinstance(item, dict):
                _add(item.get("candidate_id"))
        for item in sub.get("top_results") or []:
            if isinstance(item, dict):
                _add(item.get("candidate_id"))
        # Also check best_candidate_id (audit report style)
        _add(sub.get("best_candidate_id"))

    _add(payload.get("candidate_id"))

    return result[:limit]


def extract_score(payload: Any) -> float | None:
    """Extract exploit score from common tool output locations.

    Searches:
    - payload["result"]["score"]
    - payload["result"]["best_candidate"]["score"]
    - payload["score"]

    Args:
        payload: Tool output dict.

    Returns:
        Score as float if found, else None.
    """
    if not isinstance(payload, dict):
        return None

    sub = payload.get("result")
    if isinstance(sub, dict):
        raw = sub.get("score")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        bc = sub.get("best_candidate")
        if isinstance(bc, dict):
            raw = bc.get("score")
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass

    raw = payload.get("score")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass

    return None


def summarize_tool_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, safe summary of tool call inputs.

    Includes only compact, safe values (family_id, seed, max_trials,
    candidate_id, include_bundle, parameters). Never includes API keys,
    raw bundles, or event logs.

    Args:
        kwargs: Tool keyword arguments.

    Returns:
        Compact dict of safe input parameters.
    """
    allowed_keys = {
        "family_id", "seed", "max_trials", "candidate_id",
        "include_bundle", "parameters", "mode", "objective",
    }
    summary: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k.lower() in _FORBIDDEN_INPUT_KEYS:
            continue
        if k not in allowed_keys:
            continue
        if isinstance(v, dict):
            # Parameters dict: keep but cap floats for readability
            summary[k] = {
                pk: (round(pv, 4) if isinstance(pv, float) else pv)
                for pk, pv in v.items()
            }
        elif isinstance(v, float):
            summary[k] = round(v, 4)
        else:
            summary[k] = v
    return summary


def summarize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, safe summary of tool call outputs.

    Includes key metadata (ok, tool, candidate_id, family_id, score,
    event_refs, counts, audit_report summary). Excludes raw event_log,
    full bundles, full unsafe_legal_states, and giant markdown excerpts.

    Args:
        output: Tool output dict with ``{ok, tool, result, error}`` shape.

    Returns:
        Compact dict of safe output metadata.
    """
    summary: dict[str, Any] = {"ok": output.get("ok", False)}

    if not output.get("ok"):
        err = output.get("error") or {}
        if isinstance(err, dict):
            summary["error_type"] = err.get("type")
            summary["error_message"] = compact_text(err.get("message"), MAX_TRACE_OBSERVATION_CHARS)
        return summary

    result = output.get("result")
    if not isinstance(result, dict):
        return summary

    # Compact scalar fields
    for key in ("candidate_id", "family_id", "seed", "max_trials",
                "candidate_count", "result_count", "best_candidate_id"):
        val = result.get(key)
        if val is not None:
            summary[key] = val

    # Numeric fields
    for key in ("unsafe_legal_state_count", "max_hazard_score",
                "mean_hazard_score", "score"):
        val = result.get(key)
        if val is not None:
            try:
                summary[key] = round(float(val), 4)
            except (TypeError, ValueError):
                pass

    # Event refs (compact, deduplicated)
    refs = extract_event_refs(output, limit=MAX_EVENT_REFS_PER_STEP)
    if refs:
        summary["event_refs"] = refs

    # Candidate IDs (compact)
    cids = extract_candidate_ids(output, limit=10)
    if cids:
        summary["candidate_ids"] = cids

    # Best candidate (compact)
    bc = result.get("best_candidate")
    if isinstance(bc, dict):
        summary["best_candidate"] = _compact_nested(bc)

    # Top results (compact, limited)
    top = result.get("top_results")
    if isinstance(top, list):
        summary["top_results_count"] = len(top)
        summary["top_results"] = [_compact_nested(r) for r in top[:3]]

    # Families list count
    families = result.get("families")
    if isinstance(families, list):
        summary["family_count"] = len(families)

    # Audit report (compact reference only)
    ar = result.get("audit_report")
    if isinstance(ar, dict):
        ar_summary: dict[str, Any] = {}
        ar_sm = ar.get("summary")
        if isinstance(ar_sm, dict):
            for k in ("unsafe_legal_state_count", "max_hazard_score",
                      "mean_hazard_score"):
                if k in ar_sm and ar_sm[k] is not None:
                    try:
                        ar_summary[k] = round(float(ar_sm[k]), 4)
                    except (TypeError, ValueError):
                        pass
            ar_refs = ar_sm.get("unsafe_legal_event_refs")
            if isinstance(ar_refs, list):
                ar_summary["unsafe_legal_event_refs_count"] = len(ar_refs)
        ar_summary["schema_version"] = ar.get("schema_version", "")
        ar_summary["limitations_count"] = len(ar.get("limitations") or [])
        summary["audit_report_summary"] = ar_summary

    # Markdown excerpt: cap size
    md = result.get("markdown_excerpt")
    if isinstance(md, str) and md:
        summary["markdown_excerpt_chars"] = len(md)
        # Do NOT include full markdown in trace output summary

    # Limitations
    lims = result.get("limitations")
    if isinstance(lims, list):
        summary["limitations_count"] = len(lims)

    # Enforce key count cap
    if len(summary) > MAX_TOOL_RESULT_KEYS:
        keys = list(summary.keys())
        for k in keys[MAX_TOOL_RESULT_KEYS:]:
            summary.pop(k, None)

    return summary


def _compact_nested(d: dict[str, Any]) -> dict[str, Any]:
    """Compact a nested dict by removing forbidden keys and capping size."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in _FORBIDDEN_OUTPUT_KEYS:
            continue
        if k.lower() in _FORBIDDEN_INPUT_KEYS:
            continue
        if isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, list):
            # Keep list length but cap content
            out[k] = v[:MAX_EVENT_REFS_PER_STEP] if k == "event_refs" else len(v)
        elif isinstance(v, dict):
            out[k] = f"<dict:{len(v)} keys>"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CampaignTraceBuilder (Task 5)
# ---------------------------------------------------------------------------


class CampaignTraceBuilder:
    """Helper class for incrementally building a CampaignTrace.

    Provides a fluent API for adding steps, tool calls, findings,
    failed attempts, and hypotheses. The resulting CampaignTrace is
    deterministic and compact.

    Usage::

        builder = CampaignTraceBuilder(
            objective="Find unsafe legal scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        builder.add_step("explore", "list_families", "Found 6 families")
        builder.add_tool_call("list_synthetic_families_tool", True, {}, {"family_count": 6})
        trace = builder.build()
    """

    def __init__(
        self,
        objective: str,
        mode: str,
        agent_config: dict[str, Any],
        seed: int | None = None,
        campaign_id: str | None = None,
    ) -> None:
        self._objective = objective
        self._mode = mode
        self._agent_config = agent_config
        self._seed = seed
        self._campaign_id = campaign_id or _deterministic_campaign_id(
            objective, agent_config, seed
        )
        self._steps: list[CampaignTraceStep] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._best_findings: list[CampaignFinding] = []
        self._failed_attempts: list[dict[str, Any]] = []
        self._next_hypotheses: list[CampaignHypothesis] = []
        self._artifacts: dict[str, Any] = {}
        self._limitations: list[str] = []
        self._step_counter: int = 0
        self._tool_counter: int = 0
        self._finding_counter: int = 0

    def add_step(
        self,
        phase: str,
        action: str,
        observation_summary: str,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_ok: bool | None = None,
        selected_family_id: str | None = None,
        selected_candidate_id: str | None = None,
        score: float | None = None,
        event_refs: list[str] | None = None,
    ) -> CampaignTraceStep:
        """Add a step to the campaign trace.

        Args:
            phase: High-level phase (e.g. "explore", "evaluate", "report").
            action: Human-readable action description.
            observation_summary: Compact observation text.
            tool_call_id: Optional ID of associated ToolCallRecord.
            tool_name: Optional tool name.
            tool_ok: Optional tool success flag.
            selected_family_id: Optional family ID.
            selected_candidate_id: Optional candidate ID.
            score: Optional exploit score.
            event_refs: Optional evidence event references.

        Returns:
            The created CampaignTraceStep.
        """
        step = CampaignTraceStep(
            step_index=self._step_counter,
            phase=phase,
            action=action,
            observation_summary=compact_text(observation_summary),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_ok=tool_ok,
            selected_family_id=selected_family_id,
            selected_candidate_id=selected_candidate_id,
            score=score,
            event_refs=list(event_refs) if event_refs else [],
        )
        self._steps.append(step)
        self._step_counter += 1
        return step

    def add_tool_call(
        self,
        tool_name: str,
        ok: bool,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        event_refs: list[str] | None = None,
        candidate_ids: list[str] | None = None,
        score: float | None = None,
    ) -> ToolCallRecord:
        """Add a tool call record to the campaign trace.

        Args:
            tool_name: Name of the deterministic tool.
            ok: Whether the tool returned ok=True.
            input_summary: Compact input summary (no secrets).
            output_summary: Compact output summary (no raw logs).
            error_type: Exception class name if failed.
            error_message: Compact error message if failed.
            event_refs: Evidence event references.
            candidate_ids: Candidate identifiers.
            score: Exploit score if produced.

        Returns:
            The created ToolCallRecord.
        """
        call_id = f"tool_{self._tool_counter:04d}"
        record = ToolCallRecord(
            call_id=call_id,
            step_index=self._step_counter,
            tool_name=tool_name,
            ok=ok,
            input_summary=input_summary,
            output_summary=output_summary,
            error_type=error_type,
            error_message=compact_text(error_message) if error_message else None,
            event_refs=list(event_refs) if event_refs else [],
            candidate_ids=list(candidate_ids) if candidate_ids else [],
            score=score,
        )
        self._tool_calls.append(record)
        self._tool_counter += 1
        return record

    def add_finding(
        self,
        finding_id: str | None = None,
        family_id: str | None = None,
        candidate_id: str | None = None,
        score: float | None = None,
        unsafe_legal_state_count: int = 0,
        max_hazard_score: float | None = None,
        mean_hazard_score: float | None = None,
        event_refs: list[str] | None = None,
        audit_report_ref: str | None = None,
        summary: str = "",
    ) -> CampaignFinding:
        """Add a finding to the campaign trace.

        Args:
            finding_id: Optional ID; auto-generated if None.
            family_id: Synthetic family that produced the finding.
            candidate_id: Best candidate identifier.
            score: Exploit score.
            unsafe_legal_state_count: Number of unsafe legal states.
            max_hazard_score: Maximum hazard score.
            mean_hazard_score: Mean hazard score.
            event_refs: Evidence event references.
            audit_report_ref: Reference to audit report artifact.
            summary: Compact human-readable summary.

        Returns:
            The created CampaignFinding.
        """
        fid = finding_id or f"finding_{self._finding_counter:04d}"
        self._finding_counter += 1
        finding = CampaignFinding(
            finding_id=fid,
            family_id=family_id,
            candidate_id=candidate_id,
            score=score,
            unsafe_legal_state_count=unsafe_legal_state_count,
            max_hazard_score=max_hazard_score,
            mean_hazard_score=mean_hazard_score,
            event_refs=list(event_refs) if event_refs else [],
            audit_report_ref=audit_report_ref,
            summary=compact_text(summary),
        )
        self._best_findings.append(finding)
        return finding

    def add_failed_attempt(
        self,
        step_index: int,
        tool_name: str,
        error_type: str,
        error_message: str,
        input_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a failed attempt record.

        Args:
            step_index: Step number where failure occurred.
            tool_name: Tool that failed.
            error_type: Exception class name.
            error_message: Compact error message.
            input_summary: Compact input summary.

        Returns:
            The created failed attempt dict.
        """
        record: dict[str, Any] = {
            "step_index": step_index,
            "tool_name": tool_name,
            "error_type": error_type,
            "error_message": compact_text(error_message),
        }
        if input_summary is not None:
            record["input_summary"] = summarize_tool_input(input_summary)
        self._failed_attempts.append(record)
        return record

    def add_hypothesis(
        self,
        basis: str,
        proposed_action: str,
        expected_signal: str,
        *,
        hypothesis_id: str | None = None,
        priority: str = "medium",
    ) -> CampaignHypothesis:
        """Add a hypothesis for future investigation.

        Args:
            basis: Evidence-based rationale.
            proposed_action: What to do next.
            expected_signal: What evidence would support/refute.
            hypothesis_id: Optional ID; auto-generated if None.
            priority: "high", "medium", or "low".

        Returns:
            The created CampaignHypothesis.
        """
        hid = hypothesis_id or f"hyp_{len(self._next_hypotheses):04d}"
        hyp = CampaignHypothesis(
            hypothesis_id=hid,
            basis=basis,
            proposed_action=proposed_action,
            expected_signal=expected_signal,
            priority=priority,
        )
        self._next_hypotheses.append(hyp)
        return hyp

    def set_artifact(self, key: str, value: Any) -> None:
        """Set an artifact reference in the campaign trace.

        Args:
            key: Artifact key (e.g. "audit_reports", "candidate_refs").
            value: Compact artifact reference dict or list.
        """
        self._artifacts[key] = value

    def add_limitation(self, limitation: str) -> None:
        """Add a limitation/disclaimer to the campaign trace.

        Args:
            limitation: Caveat text.
        """
        self._limitations.append(limitation)

    def build(self) -> CampaignTrace:
        """Build and return the final CampaignTrace.

        Returns:
            Complete CampaignTrace with all accumulated data.
        """
        return CampaignTrace(
            schema_version=CAMPAIGN_TRACE_SCHEMA,
            campaign_id=self._campaign_id,
            objective=self._objective,
            mode=self._mode,
            seed=self._seed,
            agent_config=self._agent_config,
            steps=list(self._steps),
            tool_calls=list(self._tool_calls),
            best_findings=list(self._best_findings),
            failed_attempts=list(self._failed_attempts),
            next_hypotheses=list(self._next_hypotheses),
            artifacts=dict(self._artifacts),
            limitations=list(self._limitations),
        )


def _deterministic_campaign_id(
    objective: str,
    agent_config: dict[str, Any],
    seed: int | None,
) -> str:
    """Generate a deterministic campaign ID from inputs.

    Uses a stable hash of the objective, config, and seed so that
    the same inputs always produce the same campaign_id.

    Args:
        objective: Research objective.
        agent_config: Agent configuration dict.
        seed: PRNG seed.

    Returns:
        Campaign ID string (e.g. "campaign_a1b2c3d4e5f6").
    """
    stable = json.dumps(
        {"objective": objective, "config": agent_config, "seed": seed},
        sort_keys=True,
    )
    digest = sha256(stable.encode()).hexdigest()[:12]
    return f"campaign_{digest}"


# ---------------------------------------------------------------------------
# Hypothesis generation from trace (Task 10)
# ---------------------------------------------------------------------------


def build_next_hypotheses_from_trace(trace: CampaignTrace) -> list[CampaignHypothesis]:
    """Generate deterministic, evidence-based hypotheses from campaign results.

    Does NOT call an LLM. Uses only the findings and data already in the
    trace to propose next steps.

    Args:
        trace: A built CampaignTrace with findings and steps.

    Returns:
        List of CampaignHypothesis instances.
    """
    hypotheses: list[CampaignHypothesis] = []

    if trace.best_findings:
        best = trace.best_findings[0]
        has_unsafe = best.unsafe_legal_state_count > 0

        if has_unsafe:
            # Hypothesis 0: Narrow search around best candidate
            score_str = f"score={best.score:.2f}" if best.score is not None else "no score"
            hazard_str = (
                f"max_hazard_score={best.max_hazard_score:.4f}"
                if best.max_hazard_score is not None
                else "unknown hazard"
            )
            hypotheses.append(
                CampaignHypothesis(
                    hypothesis_id="hyp_0000",
                    basis=(
                        f"Best candidate produced unsafe_legal_state "
                        f"with {hazard_str} and {best.unsafe_legal_state_count} "
                        f"unsafe state(s) ({score_str})."
                    ),
                    proposed_action=(
                        "Run adaptive mutation around this candidate with "
                        "narrower width and lower barrier distance."
                    ),
                    expected_signal=(
                        "Higher exploit_score or additional unsafe_legal_event_refs."
                    ),
                    priority="high",
                )
            )
            # Hypothesis 1: Patch comparison
            hypotheses.append(
                CampaignHypothesis(
                    hypothesis_id="hyp_0001",
                    basis="Unsafe legal event survived deterministic search.",
                    proposed_action=(
                        "Compare regulatory intervention proxies against "
                        "this candidate."
                    ),
                    expected_signal=(
                        "mitigated/improved_hazard/worse patch verdicts."
                    ),
                    priority="medium",
                )
            )
            # Hypothesis 2: More trials
            hypotheses.append(
                CampaignHypothesis(
                    hypothesis_id="hyp_0002",
                    basis=(
                        f"Finding from family '{best.family_id}' with "
                        f"current trial count."
                    ),
                    proposed_action=(
                        f"Run same family '{best.family_id}' with more "
                        f"trials to explore parameter space more thoroughly."
                    ),
                    expected_signal=(
                        "Additional unsafe legal states or higher scores "
                        "in unexplored parameter regions."
                    ),
                    priority="medium",
                )
            )
        else:
            # Finding exists but no unsafe legal states
            hypotheses.append(
                CampaignHypothesis(
                    hypothesis_id="hyp_0000",
                    basis=(
                        f"Best candidate from family '{best.family_id}' "
                        f"showed no unsafe legal states with current parameters."
                    ),
                    proposed_action="Increase max_trials to explore more of the parameter space.",
                    expected_signal="Unsafe legal states may appear in unexplored regions.",
                    priority="medium",
                )
            )
    else:
        # No findings at all
        hypotheses.append(
            CampaignHypothesis(
                hypothesis_id="hyp_0000",
                basis="No unsafe legal states found with current parameter space.",
                proposed_action="Increase max_trials to explore more candidates.",
                expected_signal="Unsafe legal states may appear with more samples.",
                priority="medium",
            )
        )
        hypotheses.append(
        CampaignHypothesis(
            hypothesis_id="hyp_0001",
            basis="Current family may not produce unsafe legal states.",
            proposed_action=(
                "Try other positive synthetic families with "
                "different risk profiles."
            ),
            expected_signal=(
                "Unsafe legal states in families with "
                "different segment geometries."
            ),
            priority="medium",
        )
        )
        hypotheses.append(
            CampaignHypothesis(
                hypothesis_id="hyp_0002",
                basis="Parameter ranges may be too narrow.",
                proposed_action="Widen risk, ERS, and gap parameter ranges.",
                expected_signal="Higher exploit scores in wider parameter regions.",
                priority="low",
            )
        )

    return hypotheses
