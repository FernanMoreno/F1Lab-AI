"""Agent orchestration layer for F1Lab-AI falsification research.

This package provides research agents that use deterministic falsification
tools to search for legal/grey-area unsafe scenarios. Agents are evidence-bound:

- The agent chooses experiments.
- The deterministic tools execute them.
- SafetyOracle / LegalVerdict / EvidenceBundle remain the source of truth.
- The agent must NOT invent evidence or decide safety/legal status.

Requires the optional `agents` extra:
pip install 'f1lab-ai[agents]'
"""

from reglabsim.agents.campaign_trace import (
    AGENT_TRACE_SCHEMA,
    CAMPAIGN_TRACE_SCHEMA,
    CampaignFinding,
    CampaignHypothesis,
    CampaignTrace,
    CampaignTraceBuilder,
    CampaignTraceStep,
    ToolCallRecord,
    build_next_hypotheses_from_trace,
    campaign_trace_to_dict,
    compact_text,
    dataclass_to_dict,
    extract_candidate_ids,
    extract_event_refs,
    extract_score,
    summarize_tool_input,
    summarize_tool_output,
)
from reglabsim.agents.falsification_agent import (
    AgentTraceStep,
    FalsificationAgentConfig,
    build_falsification_agent_system_prompt,
    build_falsification_deepagent,
    run_falsification_agent,
    run_falsification_agent_deterministic,
    run_nvidia_falsification_agent_manual,
)

__all__ = [
    "AGENT_TRACE_SCHEMA",
    # Campaign trace (PR 7.3)
    "CAMPAIGN_TRACE_SCHEMA",
    # Agent (PR 7.2)
    "AgentTraceStep",
    "CampaignFinding",
    "CampaignHypothesis",
    "CampaignTrace",
    "CampaignTraceBuilder",
    "CampaignTraceStep",
    "FalsificationAgentConfig",
    "ToolCallRecord",
    "build_falsification_agent_system_prompt",
    "build_falsification_deepagent",
    "build_next_hypotheses_from_trace",
    "campaign_trace_to_dict",
    "compact_text",
    "dataclass_to_dict",
    "extract_candidate_ids",
    "extract_event_refs",
    "extract_score",
    "run_falsification_agent",
    "run_falsification_agent_deterministic",
    "run_nvidia_falsification_agent_manual",
    "summarize_tool_input",
    "summarize_tool_output",
]
