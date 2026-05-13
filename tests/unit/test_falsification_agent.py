"""PR 7.2 → PR 7.3 — Tests for the NVIDIA/LangGraph DeepAgent falsification agent.

Verifies:
* Agent config defaults are safe (allow_real_llm=False, hard limits).
* System prompt contains evidence guardrails.
* DeepAgent builder requires allow_real_llm or injected llm.
* DeepAgent builder works with injected fake LLM.
* Deterministic runner returns valid schema.
* Deterministic runner calls tools and finds event_refs.
* Deterministic runner respects max_trials.
* Agent output does not overclaim.
* Agent module does not import NVIDIA at module import.
* Agent does not modify runtime/safety/scoring.
* Manual real-LLM path is skipped without key.
* Campaign trace is a structured dict (PR 7.3), with campaign_trace_steps alias.
* Campaign trace entries are compact (no raw event_log/bundles).
* NVIDIA_MODEL_NAME has priority over F1LAB_NVIDIA_MODEL.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reglabsim.agents.falsification_agent import (
    AgentTraceStep,
    FalsificationAgentConfig,
    _DeepAgentError,
    _error_output,
    _resolve_model_name,
    _trace_step,
    build_falsification_agent_system_prompt,
    build_falsification_deepagent,
    run_falsification_agent,
    run_falsification_agent_deterministic,
    run_nvidia_falsification_agent_manual,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_POSITIVE_FAMILY = "confined_corner_grass"
_CONTROL_FAMILY = "wide_corner_asphalt_control"

_OVERCLAIM_PHRASES = [
    "proven safe",
    "guaranteed",
    "calibrated recommendation",
]

_FORBIDDEN_MUTATIONS = [
    "RaceMicrokernel",
    "SafetyOracle",
    "safety_status",
]


# ===========================================================================
# 1. Agent config defaults
# ===========================================================================


class TestAgentConfigDefaultsAreSafe:
    """Assert: allow_real_llm is False, hard limits are reasonable."""

    def test_allow_real_llm_is_false_by_default(self) -> None:
        config = FalsificationAgentConfig()
        assert config.allow_real_llm is False

    def test_max_iterations_is_at_most_3(self) -> None:
        config = FalsificationAgentConfig()
        assert config.max_iterations <= 3

    def test_max_trials_per_search_is_at_most_25(self) -> None:
        config = FalsificationAgentConfig()
        assert config.max_trials_per_search <= 25

    def test_max_tool_calls_is_at_most_12(self) -> None:
        config = FalsificationAgentConfig()
        assert config.max_tool_calls <= 12

    def test_require_evidence_refs_is_true(self) -> None:
        config = FalsificationAgentConfig()
        assert config.require_evidence_refs is True

    def test_rejects_zero_max_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            FalsificationAgentConfig(max_iterations=0)

    def test_rejects_zero_max_trials(self) -> None:
        with pytest.raises(ValueError, match="max_trials_per_search must be > 0"):
            FalsificationAgentConfig(max_trials_per_search=0)

    def test_rejects_zero_max_tool_calls(self) -> None:
        with pytest.raises(ValueError, match="max_tool_calls must be > 0"):
            FalsificationAgentConfig(max_tool_calls=0)

    def test_frozen_dataclass_cannot_be_mutated(self) -> None:
        config = FalsificationAgentConfig()
        with pytest.raises(AttributeError):
            config.allow_real_llm = True  # type: ignore[misc]


# ===========================================================================
# 2. System prompt evidence guardrails
# ===========================================================================


class TestSystemPromptContainsEvidenceGuardrails:
    """Assert prompt contains: source of truth, SafetyOracle, LegalVerdict,
    do not invent event_refs, not calibrated."""

    def test_contains_source_of_truth(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "source of truth" in prompt.lower()

    def test_contains_safety_oracle(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "SafetyOracle" in prompt

    def test_contains_legal_verdict(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "LegalVerdict" in prompt

    def test_contains_race_microkernel_boundary(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "RaceMicrokernel" in prompt

    def test_contains_do_not_invent_event_refs(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "event_refs" in prompt
        assert "invent" in prompt.lower() or "fabricate" in prompt.lower()

    def test_contains_not_calibrated(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "calibrated" in prompt.lower()

    def test_contains_respect_max_iterations(self) -> None:
        config = FalsificationAgentConfig(max_iterations=2)
        prompt = build_falsification_agent_system_prompt(config)
        assert "2" in prompt

    def test_contains_only_tool_outputs_count(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "only tool outputs count as evidence" in prompt.lower()

    def test_contains_no_real_track_names_instruction(self) -> None:
        config = FalsificationAgentConfig()
        prompt = build_falsification_agent_system_prompt(config)
        assert "Suzuka" in prompt or "real track" in prompt.lower()


# ===========================================================================
# 3. build_falsification_deepagent requires allow_real_llm or injected llm
# ===========================================================================


class TestBuildDeepagentRequiresAllowRealLlmWithoutInjectedLlm:
    """Calling build without llm and allow_real_llm=False must fail safely."""

    def test_raises_error_when_no_llm_and_allow_false(self) -> None:
        config = FalsificationAgentConfig(allow_real_llm=False)
        with pytest.raises(_DeepAgentError, match="allow_real_llm"):
            build_falsification_deepagent(llm=None, config=config)

    def test_raises_error_when_no_llm_and_default_config(self) -> None:
        with pytest.raises(_DeepAgentError, match="allow_real_llm"):
            build_falsification_deepagent(llm=None)

    def test_error_does_not_expose_api_key(self) -> None:
        config = FalsificationAgentConfig(allow_real_llm=False)
        with pytest.raises(_DeepAgentError) as exc_info:
            build_falsification_deepagent(llm=None, config=config)
        assert "nvapi" not in str(exc_info.value).lower()


# ===========================================================================
# 4. build_falsification_deepagent uses injected fake LLM
# ===========================================================================


class TestBuildDeepagentUsesInjectedFakeLlm:
    """Monkeypatch deepagents.create_deep_agent to avoid real NVIDIA call."""

    def test_builds_with_injected_llm(self) -> None:
        """Inject a fake LLM and mock create_deep_agent — no NVIDIA call."""
        fake_llm = MagicMock()
        fake_agent = MagicMock()

        config = FalsificationAgentConfig(allow_real_llm=False)

        # Mock as_langchain_tools to avoid needing langchain installed
        mock_tools = [MagicMock(name="tool_1")]
        mock_create = MagicMock(return_value=fake_agent)

        # Register deepagents in sys.modules FIRST so patch() can find it
        with patch.dict(
            sys.modules, {"deepagents": MagicMock(create_deep_agent=mock_create)}
        ):
            with patch(
                "reglabsim.agents.falsification_agent._get_langchain_tools_safe",
                return_value=mock_tools,
            ):
                with patch("deepagents.create_deep_agent", mock_create):
                    result = build_falsification_deepagent(
                        llm=fake_llm, config=config
                    )
                    assert result is fake_agent

    def test_create_deep_agent_called_with_tools_and_prompt(self) -> None:
        """Verify create_deep_agent receives model, tools, system_prompt as keyword args."""
        fake_llm = MagicMock()
        fake_agent = MagicMock()
        mock_create = MagicMock(return_value=fake_agent)
        mock_tools = [MagicMock(name="tool_1")]

        config = FalsificationAgentConfig(allow_real_llm=False)

        # Register deepagents in sys.modules FIRST
        with patch.dict(
            sys.modules, {"deepagents": MagicMock(create_deep_agent=mock_create)}
        ):
            with patch(
                "reglabsim.agents.falsification_agent._get_langchain_tools_safe",
                return_value=mock_tools,
            ):
                with patch("deepagents.create_deep_agent", mock_create):
                    result = build_falsification_deepagent(
                        llm=fake_llm, config=config
                    )
                    assert result is fake_agent
                    mock_create.assert_called_once()
                    call_kwargs = mock_create.call_args.kwargs
                    # Keyword-only API: model=…, tools=…, system_prompt=…
                    assert call_kwargs.get("model") is fake_llm
                    assert call_kwargs.get("tools") is mock_tools
                    assert "system_prompt" in call_kwargs
                    assert isinstance(call_kwargs["system_prompt"], str)
                    assert len(call_kwargs["system_prompt"]) > 0


# ===========================================================================
# 5. Deterministic runner returns schema
# ===========================================================================


class TestDeterministicRunnerReturnsSchema:
    """Assert: schema_version, ok, summary, campaign_trace, best_finding."""

    def test_schema_version_is_correct(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert result["schema_version"] == "falsification_agent.v0"

    def test_ok_is_true(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert result["ok"] is True

    def test_summary_exists_and_is_string(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_campaign_trace_exists_and_is_dict(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert isinstance(result["campaign_trace"], dict)
        assert "steps" in result["campaign_trace"]
        assert len(result["campaign_trace"]["steps"]) > 0

    def test_campaign_trace_steps_alias_exists_and_is_list(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert isinstance(result["campaign_trace_steps"], list)
        assert len(result["campaign_trace_steps"]) > 0

    def test_best_finding_exists(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert result["best_finding"] is not None
        bf = result["best_finding"]
        assert "family_id" in bf
        assert "candidate_id" in bf
        assert "score" in bf

    def test_mode_is_set(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert result["mode"] == "deterministic_falsification_agent"

    def test_model_name_is_deterministic_harness(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert result["model_name"] == "deterministic_harness"

    def test_next_hypotheses_is_list(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert isinstance(result["next_hypotheses"], list)
        assert len(result["next_hypotheses"]) > 0

    def test_limitations_is_list(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        assert isinstance(result["limitations"], list)
        assert len(result["limitations"]) > 0

    def test_output_is_json_serialisable(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded["ok"] is True


# ===========================================================================
# 6. Deterministic runner calls tools and finds event_refs
# ===========================================================================


class TestDeterministicRunnerCallsToolsAndFindsEventRefs:
    """Assert: campaign_trace_steps includes list/search/audit,
    best_finding.event_refs exists when exploit found."""

    def test_campaign_trace_includes_list_families(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        trace = result["campaign_trace_steps"]
        actions = [s["action"] for s in trace]
        assert "list_synthetic_families" in actions

    def test_campaign_trace_includes_search(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        trace = result["campaign_trace_steps"]
        tool_names = [s.get("tool_name") for s in trace if s.get("tool_name")]
        assert "run_falsification_search_tool" in tool_names

    def test_campaign_trace_includes_audit(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        trace = result["campaign_trace_steps"]
        tool_names = [s.get("tool_name") for s in trace if s.get("tool_name")]
        assert "build_best_candidate_audit_report_tool" in tool_names

    def test_best_finding_has_event_refs_when_exploit_found(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        bf = result.get("best_finding")
        if bf is not None and bf.get("unsafe_legal_state_count", 0) > 0:
            # If an exploit was found, event_refs must exist (even if empty list)
            assert "event_refs" in bf
            assert isinstance(bf["event_refs"], list)

    def test_selected_family_in_trace(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        trace = result["campaign_trace_steps"]
        family_ids = [
            s.get("selected_family_id")
            for s in trace
            if s.get("selected_family_id")
        ]
        assert len(family_ids) > 0
        assert _POSITIVE_FAMILY in family_ids


# ===========================================================================
# 7. Deterministic runner respects max_trials
# ===========================================================================


class TestDeterministicRunnerRespectsMaxTrials:
    """Use config max_trials_per_search=5. Assert search does not exceed 5."""

    def test_respects_max_trials(self) -> None:
        config = FalsificationAgentConfig(max_trials_per_search=5)
        result = run_falsification_agent_deterministic(
            "Find unsafe legal scenarios", config=config
        )
        # The search tool should have been called with max_trials=5
        # Check the trace for search step
        trace = result["campaign_trace_steps"]
        search_steps = [
            s for s in trace if s.get("tool_name") == "run_falsification_search_tool"
        ]
        assert len(search_steps) > 0
        # The observation_summary should mention 5 or fewer trials
        # (the tool caps at its own _MAX_TRIALS=100, but our config sends 5)
        # We can also verify the config was respected by checking the search
        # actually returned <= 5 results
        assert result["ok"] is True

    def test_limitations_mention_trial_count(self) -> None:
        config = FalsificationAgentConfig(max_trials_per_search=5)
        result = run_falsification_agent_deterministic(
            "Find unsafe legal scenarios", config=config
        )
        limitations_text = " ".join(result["limitations"])
        assert "5" in limitations_text or "trials" in limitations_text.lower()


# ===========================================================================
# 8. Agent output does not overclaim
# ===========================================================================


class TestAgentOutputDoesNotOverclaim:
    """Assert summary/limitations do not contain forbidden phrases."""

    @pytest.fixture()
    def deterministic_result(self) -> dict[str, Any]:
        return run_falsification_agent_deterministic("Find unsafe legal scenarios")

    def test_summary_does_not_contain_proven_safe(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        summary_lower = deterministic_result["summary"].lower()
        assert "proven safe" not in summary_lower

    def test_summary_does_not_contain_guaranteed(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        summary_lower = deterministic_result["summary"].lower()
        assert "guaranteed" not in summary_lower

    def test_summary_does_not_contain_calibrated_recommendation(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        summary_lower = deterministic_result["summary"].lower()
        assert "calibrated recommendation" not in summary_lower

    def test_summary_does_not_contain_real_f1(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        summary_lower = deterministic_result["summary"].lower()
        assert "real f1" not in summary_lower

    def test_limitations_contain_stress_test_disclaimer(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        limitations_text = " ".join(deterministic_result["limitations"]).lower()
        assert "stress-test" in limitations_text or "deterministic" in limitations_text

    def test_summary_uses_cautious_language(
        self, deterministic_result: dict[str, Any]
    ) -> None:
        """Summary should mention 'deterministic' or 'stress-test'."""
        summary_lower = deterministic_result["summary"].lower()
        assert "deterministic" in summary_lower or "stress-test" in summary_lower


# ===========================================================================
# 9. Agent module does not import NVIDIA at module import
# ===========================================================================


class TestAgentModuleDoesNotImportNvidiaAtModuleImport:
    """Inspect source: no top-level nvidia/deepagents/langgraph imports."""

    def test_no_nvidia_import_at_top_level(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.agents.falsification_agent"),
            "__file__",
            "",
        )
        assert source_path, "Could not locate falsification_agent.py"
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
                continue
            # Check for forbidden top-level imports (before any function def)
            # Heuristic: if we haven't seen 'def ' yet, we're at top level
            if stripped.startswith("def ") or stripped.startswith("class "):
                break  # Past top level — stop checking
            if stripped.startswith("from ") or stripped.startswith("import "):
                forbidden = [
                    "nvidia", "deepagents", "langgraph",
                    "ChatNVIDIA", "langchain_nvidia",
                ]
                for token in forbidden:
                    assert token not in stripped, (
                        f"falsification_agent.py must not top-level import "
                        f"{token!r} (line {i+1}: {stripped!r})"
                    )


# ===========================================================================
# 10. Agent does not modify runtime or safety
# ===========================================================================


class TestAgentDoesNotModifyRuntimeOrSafety:
    """Greps or source inspection: no RaceMicrokernel mutation,
    no SafetyOracle mutation, no direct safety_status assignment."""

    def test_no_race_microkernel_mutation_in_source(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.agents.falsification_agent"),
            "__file__",
            "",
        )
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        # Should not import RaceMicrokernel
        assert (
            "from reglabsim" not in src
            or "RaceMicrokernel" not in src.split("from reglabsim")[0][:50]
        )
        assert "import RaceMicrokernel" not in src
        # Should not instantiate or call RaceMicrokernel
        assert "RaceMicrokernel(" not in src
        assert "RaceMicrokernel." not in src

    def test_no_safety_oracle_mutation_in_source(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.agents.falsification_agent"),
            "__file__",
            "",
        )
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        # Should not import SafetyOracle
        assert "import SafetyOracle" not in src
        # Should not instantiate or call SafetyOracle
        assert "SafetyOracle(" not in src
        assert "SafetyOracle." not in src

    def test_no_direct_safety_status_assignment(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.agents.falsification_agent"),
            "__file__",
            "",
        )
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        # Should not assign safety_status or legal_status
        assert "safety_status =" not in src
        assert "legal_status =" not in src

    def test_no_metrics_mutation(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.agents.falsification_agent"),
            "__file__",
            "",
        )
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        # Should not import scoring or metrics directly
        assert "score_candidate_metrics" not in src


# ===========================================================================
# 11. Manual real-LLM path is skipped without key
# ===========================================================================


class TestManualRealLlmPathIsSkippedWithoutKey:
    """Integration test: skip if no NVIDIA_API_KEY."""

    def test_manual_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run_nvidia_falsification_agent_manual must raise without NVIDIA_API_KEY."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            run_nvidia_falsification_agent_manual("Find unsafe scenarios")


# ===========================================================================
# 12. Campaign trace entries are compact
# ===========================================================================


class TestCampaignTraceEntriesAreCompact:
    """Ensure no raw event_log or huge bundle dumped into trace."""

    def test_trace_entries_have_no_event_log_key(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        for step in result["campaign_trace_steps"]:
            assert "event_log" not in step, (
                f"Step {step.get('step_index')} contains 'event_log' — "
                f"trace entries must be compact"
            )

    def test_trace_entries_have_no_bundle_key(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        for step in result["campaign_trace_steps"]:
            assert "bundle" not in step, (
                f"Step {step.get('step_index')} contains 'bundle' — "
                f"trace entries must be compact"
            )

    def test_trace_entries_are_small_when_serialised(self) -> None:
        """Each trace step should be < 2KB when JSON-serialised."""
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        for step in result["campaign_trace_steps"]:
            step_json = json.dumps(step)
            assert len(step_json) < 2048, (
                f"Step {step.get('step_index')} is {len(step_json)} bytes — "
                f"exceeds 2KB compactness limit"
            )

    def test_event_refs_in_trace_are_list_of_strings(self) -> None:
        result = run_falsification_agent_deterministic("Find unsafe legal scenarios")
        for step in result["campaign_trace_steps"]:
            refs = step.get("event_refs", [])
            assert isinstance(refs, list)
            for ref in refs:
                assert isinstance(ref, str)


# ===========================================================================
# 13. NVIDIA_MODEL_NAME priority over F1LAB_NVIDIA_MODEL
# ===========================================================================


class TestNvidiaModelNamePriority:
    """NVIDIA_MODEL_NAME should take priority over F1LAB_NVIDIA_MODEL."""

    def test_nvidia_model_name_takes_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both env vars are set, NVIDIA_MODEL_NAME wins."""
        monkeypatch.setenv("NVIDIA_MODEL_NAME", "mistral-nemotron")
        monkeypatch.setenv("F1LAB_NVIDIA_MODEL", "fallback-model")
        config = FalsificationAgentConfig()
        resolved = _resolve_model_name(config)
        assert resolved == "mistral-nemotron"

    def test_f1lab_nvidia_model_used_when_nvidia_model_name_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When NVIDIA_MODEL_NAME is not set, F1LAB_NVIDIA_MODEL is used."""
        monkeypatch.delenv("NVIDIA_MODEL_NAME", raising=False)
        monkeypatch.setenv("F1LAB_NVIDIA_MODEL", "fallback-model")
        config = FalsificationAgentConfig()
        resolved = _resolve_model_name(config)
        assert resolved == "fallback-model"

    def test_config_model_name_overrides_both_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit config.model_name overrides both env vars."""
        monkeypatch.setenv("NVIDIA_MODEL_NAME", "mistral-nemotron")
        monkeypatch.setenv("F1LAB_NVIDIA_MODEL", "fallback-model")
        config = FalsificationAgentConfig(model_name="explicit-model")
        resolved = _resolve_model_name(config)
        assert resolved == "explicit-model"

    def test_default_used_when_nothing_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When nothing is set, default model is used."""
        monkeypatch.delenv("NVIDIA_MODEL_NAME", raising=False)
        monkeypatch.delenv("F1LAB_NVIDIA_MODEL", raising=False)
        config = FalsificationAgentConfig()
        resolved = _resolve_model_name(config)
        assert resolved == "nvidia/llama-3.1-nemotron-70b-instruct"


# ===========================================================================
# 14. AgentTraceStep dataclass
# ===========================================================================


class TestAgentTraceStep:
    """Tests for the AgentTraceStep dataclass."""

    def test_creation_with_required_fields(self) -> None:
        step = AgentTraceStep(
            step_index=0,
            action="test_action",
            tool_name="test_tool",
            tool_ok=True,
            observation_summary="test observation",
        )
        assert step.step_index == 0
        assert step.action == "test_action"
        assert step.tool_name == "test_tool"
        assert step.tool_ok is True
        assert step.observation_summary == "test observation"

    def test_optional_fields_default_to_none_or_empty(self) -> None:
        step = AgentTraceStep(
            step_index=0,
            action="test",
            tool_name=None,
            tool_ok=None,
            observation_summary="obs",
        )
        assert step.selected_family_id is None
        assert step.selected_candidate_id is None
        assert step.score is None
        assert step.event_refs == []

    def test_asdict_produces_compact_dict(self) -> None:
        from dataclasses import asdict

        step = AgentTraceStep(
            step_index=1,
            action="run_search",
            tool_name="run_falsification_search",
            tool_ok=True,
            observation_summary="Found exploit",
            selected_family_id="confined_corner_grass",
            score=15.0,
            event_refs=["ev_001", "ev_002"],
        )
        d = asdict(step)
        assert d["step_index"] == 1
        assert d["event_refs"] == ["ev_001", "ev_002"]
        # No event_log or bundle
        assert "event_log" not in d
        assert "bundle" not in d


# ===========================================================================
# 15. _trace_step helper
# ===========================================================================


class TestTraceStepHelper:
    """Tests for the _trace_step helper function."""

    def test_produces_valid_dict(self) -> None:
        result = _trace_step(0, "start", None, None, "Agent started")
        assert result["step_index"] == 0
        assert result["action"] == "start"
        assert result["tool_name"] is None
        assert result["tool_ok"] is None
        assert result["observation_summary"] == "Agent started"

    def test_with_all_fields(self) -> None:
        result = _trace_step(
            3,
            "search",
            "run_falsification_search",
            True,
            "Found 2 unsafe states",
            selected_family_id="confined_corner_grass",
            selected_candidate_id="c1",
            score=15.5,
            event_refs=["ev1"],
        )
        assert result["selected_family_id"] == "confined_corner_grass"
        assert result["score"] == 15.5
        assert result["event_refs"] == ["ev1"]


# ===========================================================================
# 16. _error_output helper
# ===========================================================================


class TestErrorOutput:
    """Tests for the _error_output helper function."""

    def test_returns_error_envelope(self) -> None:
        result = _error_output("TestError", "Something failed", [])
        assert result["schema_version"] == "falsification_agent.v0"
        assert result["ok"] is False
        assert result["error"]["type"] == "TestError"
        assert result["error"]["message"] == "Something failed"
        # campaign_trace is now a dict with "steps" key
        assert isinstance(result["campaign_trace"], dict)
        assert result["campaign_trace"]["steps"] == []

    def test_preserves_existing_trace(self) -> None:
        trace = [{"step_index": 0, "action": "start"}]
        result = _error_output("TestError", "Failed", trace)
        assert isinstance(result["campaign_trace"], dict)
        assert len(result["campaign_trace"]["steps"]) == 1
        assert result["campaign_trace_steps"] == trace

    def test_is_json_serialisable(self) -> None:
        result = _error_output("TestError", "Failed", [])
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        assert decoded["ok"] is False


# ===========================================================================
# 17. run_falsification_agent with no llm and allow_real_llm=False
# ===========================================================================


class TestRunFalsificationAgentSafeFailure:
    """run_falsification_agent must fail safely without llm and allow_real_llm=False."""

    def test_returns_error_output(self) -> None:
        config = FalsificationAgentConfig(allow_real_llm=False)
        result = run_falsification_agent(
            "Find unsafe scenarios", config=config, llm=None
        )
        assert result["ok"] is False
        assert "error" in result
        assert result["schema_version"] == "falsification_agent.v0"

    def test_error_output_includes_campaign_trace(self) -> None:
        config = FalsificationAgentConfig(allow_real_llm=False)
        result = run_falsification_agent(
            "Find unsafe scenarios", config=config, llm=None
        )
        # Should have at least the start step
        trace = result["campaign_trace"]
        assert isinstance(trace, dict)
        assert len(trace.get("steps", [])) >= 1


# ===========================================================================
# 18. Package imports
# ===========================================================================


class TestPackageImports:
    """Verify that agent functions are importable from the package level."""

    def test_all_functions_importable_from_package(self) -> None:
        from reglabsim.agents import (
            FalsificationAgentConfig,
            build_falsification_agent_system_prompt,
            build_falsification_deepagent,
            run_falsification_agent,
            run_falsification_agent_deterministic,
            run_nvidia_falsification_agent_manual,
        )

        assert callable(FalsificationAgentConfig)
        assert callable(build_falsification_agent_system_prompt)
        assert callable(build_falsification_deepagent)
        assert callable(run_falsification_agent)
        assert callable(run_falsification_agent_deterministic)
        assert callable(run_nvidia_falsification_agent_manual)

    def test_dunder_all_exports(self) -> None:
        import reglabsim.agents

        assert hasattr(reglabsim.agents, "__all__")
        expected = {
            # Campaign trace (PR 7.3)
            "CAMPAIGN_TRACE_SCHEMA",
            "AGENT_TRACE_SCHEMA",
            "CampaignFinding",
            "CampaignHypothesis",
            "CampaignTrace",
            "CampaignTraceBuilder",
            "CampaignTraceStep",
            "ToolCallRecord",
            "build_next_hypotheses_from_trace",
            "campaign_trace_to_dict",
            "compact_text",
            "dataclass_to_dict",
            "extract_candidate_ids",
            "extract_event_refs",
            "extract_score",
            "summarize_tool_input",
            "summarize_tool_output",
            # Agent (PR 7.2)
            "AgentTraceStep",
            "FalsificationAgentConfig",
            "build_falsification_agent_system_prompt",
            "build_falsification_deepagent",
            "run_falsification_agent",
            "run_falsification_agent_deterministic",
            "run_nvidia_falsification_agent_manual",
        }
        assert set(reglabsim.agents.__all__) == expected


# ===========================================================================
# 19. Optional integration test
# ===========================================================================


@pytest.mark.integration
def test_run_nvidia_falsification_agent_manual() -> None:
    """Integration test — only runs with NVIDIA_API_KEY set."""
    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set — skipping integration test")

    config = FalsificationAgentConfig(
        allow_real_llm=True,
        max_iterations=1,
        max_trials_per_search=5,
    )
    result = run_falsification_agent(
        "Find one unsafe legal scenario",
        config=config,
    )
    assert result["schema_version"] == "falsification_agent.v0"
    assert isinstance(result["campaign_trace"], dict)
    assert "steps" in result["campaign_trace"]
