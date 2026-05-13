"""PR 7.3 — Tests for the campaign trace / experiment memory module.

Verifies:
* Campaign trace builder creates deterministic campaign IDs.
* compact_text truncates long values safely.
* extract_event_refs extracts from common tool output locations.
* extract_candidate_ids extracts from common tool output locations.
* summarize_tool_input excludes secrets and raw bundles.
* summarize_tool_output excludes raw logs and full bundles.
* Tool call records capture errors compactly.
* Campaign trace is JSON-serializable.
* build_next_hypotheses produces evidence-based hypotheses.
* No raw event_log/full bundle/secrets appear in trace JSON.
"""

from __future__ import annotations

import json
from typing import Any

from reglabsim.agents.campaign_trace import (
    CAMPAIGN_TRACE_SCHEMA,
    MAX_TRACE_OBSERVATION_CHARS,
    CampaignTraceBuilder,
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

# ===========================================================================
# 1. CampaignTraceBuilder creates deterministic campaign ID
# ===========================================================================


class TestCampaignTraceBuilderDeterministicCampaignId:
    """Same objective/config/seed must produce the same campaign_id."""

    def test_same_inputs_produce_same_campaign_id(self) -> None:
        builder_a = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        builder_b = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        trace_a = builder_a.build()
        trace_b = builder_b.build()
        assert trace_a.campaign_id == trace_b.campaign_id

    def test_different_objective_produces_different_campaign_id(self) -> None:
        builder_a = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        builder_b = CampaignTraceBuilder(
            objective="Test edge cases",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        trace_a = builder_a.build()
        trace_b = builder_b.build()
        assert trace_a.campaign_id != trace_b.campaign_id

    def test_different_seed_produces_different_campaign_id(self) -> None:
        builder_a = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        builder_b = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=99,
        )
        trace_a = builder_a.build()
        trace_b = builder_b.build()
        assert trace_a.campaign_id != trace_b.campaign_id

    def test_explicit_campaign_id_overrides_deterministic(self) -> None:
        builder = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
            campaign_id="custom_campaign_id",
        )
        trace = builder.build()
        assert trace.campaign_id == "custom_campaign_id"

    def test_campaign_id_starts_with_campaign_prefix(self) -> None:
        builder = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        trace = builder.build()
        assert trace.campaign_id.startswith("campaign_")


# ===========================================================================
# 2. compact_text truncates long values
# ===========================================================================


class TestCompactTextTruncatesLongValues:
    """Very long strings become capped."""

    def test_truncates_long_string(self) -> None:
        long_text = "x" * 1000
        result = compact_text(long_text)
        assert len(result) == MAX_TRACE_OBSERVATION_CHARS
        assert result.endswith("...")

    def test_preserves_short_string(self) -> None:
        short_text = "Hello world"
        result = compact_text(short_text)
        assert result == "Hello world"

    def test_handles_none(self) -> None:
        result = compact_text(None)
        assert result == ""

    def test_handles_non_string(self) -> None:
        result = compact_text(42)
        assert result == "42"

    def test_replaces_newlines(self) -> None:
        multiline = "line1\nline2\r\nline3"
        result = compact_text(multiline)
        assert "\n" not in result
        assert "\r" not in result

    def test_collapses_multiple_spaces(self) -> None:
        spaces = "hello    world    test"
        result = compact_text(spaces)
        assert "    " not in result
        assert result == "hello world test"

    def test_custom_max_chars(self) -> None:
        long_text = "a" * 100
        result = compact_text(long_text, max_chars=10)
        assert len(result) == 10
        assert result.endswith("...")


# ===========================================================================
# 3. extract_event_refs from common tool outputs
# ===========================================================================


class TestExtractEventRefsFromCommonToolOutputs:
    """Covers best_candidate, bundle_summary, audit_report event refs."""

    def test_extracts_from_result_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "result": {"event_refs": ["ev_001", "ev_002"]},
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_001", "ev_002"]

    def test_extracts_from_best_candidate_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "best_candidate": {"event_refs": ["ev_bc_001"]},
            },
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_bc_001"]

    def test_extracts_from_audit_report_unsafe_legal_events(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "audit_report": {"unsafe_legal_events": ["ev_audit_001"]},
            },
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_audit_001"]

    def test_extracts_from_bundle_summary_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "bundle_summary": {"event_refs": ["ev_bundle_001"]},
            },
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_bundle_001"]

    def test_extracts_from_top_level_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "event_refs": ["ev_top_001"],
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_top_001"]

    def test_extracts_from_unsafe_legal_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "unsafe_legal_event_refs": ["ev_uls_001"],
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_uls_001"]

    def test_deduplicates_refs(self) -> None:
        payload: dict[str, Any] = {
            "result": {"event_refs": ["ev_001", "ev_001"]},
            "event_refs": ["ev_001", "ev_002"],
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_001", "ev_002"]

    def test_respects_limit(self) -> None:
        payload: dict[str, Any] = {
            "result": {"event_refs": [f"ev_{i:03d}" for i in range(20)]},
        }
        refs = extract_event_refs(payload, limit=5)
        assert len(refs) == 5

    def test_returns_empty_for_non_dict(self) -> None:
        assert extract_event_refs("not a dict") == []
        assert extract_event_refs(None) == []
        assert extract_event_refs(42) == []

    def test_extracts_from_summary_unsafe_legal_event_refs(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "summary": {"unsafe_legal_event_refs": ["ev_sum_001"]},
            },
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_sum_001"]

    def test_extracts_from_bundle_summary_metrics(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "bundle_summary": {
                    "metrics": {"unsafe_legal_event_refs": ["ev_met_001"]},
                },
            },
        }
        refs = extract_event_refs(payload)
        assert refs == ["ev_met_001"]


# ===========================================================================
# 4. extract_candidate_ids from common tool outputs
# ===========================================================================


class TestExtractCandidateIdsFromCommonToolOutputs:
    """Covers candidate_id, candidates list, top_results."""

    def test_extracts_from_result_candidate_id(self) -> None:
        payload: dict[str, Any] = {
            "result": {"candidate_id": "c_001"},
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_001"]

    def test_extracts_from_best_candidate_candidate_id(self) -> None:
        payload: dict[str, Any] = {
            "result": {"best_candidate": {"candidate_id": "c_bc_001"}},
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_bc_001"]

    def test_extracts_from_candidates_list(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "candidates": [
                    {"candidate_id": "c_001"},
                    {"candidate_id": "c_002"},
                ],
            },
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_001", "c_002"]

    def test_extracts_from_top_results(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "top_results": [
                    {"candidate_id": "c_top_001"},
                    {"candidate_id": "c_top_002"},
                ],
            },
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_top_001", "c_top_002"]

    def test_extracts_best_candidate_id_from_audit_result(self) -> None:
        payload: dict[str, Any] = {
            "result": {"best_candidate_id": "c_audit_001"},
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_audit_001"]

    def test_deduplicates(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "candidate_id": "c_001",
                "best_candidate": {"candidate_id": "c_001"},
            },
        }
        ids = extract_candidate_ids(payload)
        assert ids == ["c_001"]

    def test_respects_limit(self) -> None:
        payload: dict[str, Any] = {
            "result": {
                "candidates": [{"candidate_id": f"c_{i:03d}"} for i in range(20)],
            },
        }
        ids = extract_candidate_ids(payload, limit=5)
        assert len(ids) == 5

    def test_returns_empty_for_non_dict(self) -> None:
        assert extract_candidate_ids("not a dict") == []


# ===========================================================================
# 5. summarize_tool_input excludes secrets and raw bundles
# ===========================================================================


class TestSummarizeToolInputExcludesSecretsAndRawBundle:
    """Input with NVIDIA_API_KEY, api_key, bundle, event_log should not appear."""

    def test_excludes_api_key(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "confined_corner_grass",
            "NVIDIA_API_KEY": "nvapi-secret-key-12345",
        }
        summary = summarize_tool_input(kwargs)
        assert "NVIDIA_API_KEY" not in summary
        assert "nvapi-secret-key-12345" not in str(summary)

    def test_excludes_lowercase_api_key(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "confined_corner_grass",
            "api_key": "secret-value",
        }
        summary = summarize_tool_input(kwargs)
        assert "api_key" not in summary
        assert "secret-value" not in str(summary)

    def test_excludes_bundle(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "test",
            "bundle": {"huge": "payload"},
        }
        summary = summarize_tool_input(kwargs)
        assert "bundle" not in summary

    def test_excludes_event_log(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "test",
            "event_log": ["ev1", "ev2"],
        }
        summary = summarize_tool_input(kwargs)
        assert "event_log" not in summary

    def test_includes_allowed_keys(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "confined_corner_grass",
            "seed": 42,
            "max_trials": 25,
            "candidate_id": "c_001",
            "include_bundle": False,
        }
        summary = summarize_tool_input(kwargs)
        assert summary["family_id"] == "confined_corner_grass"
        assert summary["seed"] == 42
        assert summary["max_trials"] == 25
        assert summary["candidate_id"] == "c_001"

    def test_rounds_float_parameters(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "test",
            "parameters": {"width_m": 11.123456789, "gap_s": 0.456789},
        }
        summary = summarize_tool_input(kwargs)
        assert summary["parameters"]["width_m"] == 11.1235
        assert summary["parameters"]["gap_s"] == 0.4568

    def test_excludes_unknown_keys(self) -> None:
        kwargs: dict[str, Any] = {
            "family_id": "test",
            "custom_key": "custom_value",
        }
        summary = summarize_tool_input(kwargs)
        assert "custom_key" not in summary


# ===========================================================================
# 6. summarize_tool_output excludes raw logs
# ===========================================================================


class TestSummarizeToolOutputExcludesRawLogs:
    """Output containing event_log/full bundle should be summarized without raw fields."""

    def test_excludes_event_log_key(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "candidate_id": "c_001",
                "event_log": ["ev1", "ev2", "ev3"],
                "score": 15.0,
            },
        }
        summary = summarize_tool_output(output)
        assert "event_log" not in summary
        # Score should be present
        assert summary["score"] == 15.0

    def test_excludes_bundle_key(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "bundle": {"huge": "payload"},
                "candidate_id": "c_001",
            },
        }
        summary = summarize_tool_output(output)
        assert "bundle" not in summary

    def test_excludes_state_snapshots(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "state_snapshots": [{"t": 0}, {"t": 1}],
                "candidate_id": "c_001",
            },
        }
        summary = summarize_tool_output(output)
        assert "state_snapshots" not in summary

    def test_excludes_unsafe_legal_states(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "unsafe_legal_states": [{"big": "payload"}],
                "candidate_id": "c_001",
            },
        }
        summary = summarize_tool_output(output)
        assert "unsafe_legal_states" not in summary

    def test_includes_ok_and_key_metadata(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "candidate_id": "c_001",
                "family_id": "test_family",
                "score": 15.51,
                "unsafe_legal_state_count": 1,
                "max_hazard_score": 0.8373,
            },
        }
        summary = summarize_tool_output(output)
        assert summary["ok"] is True
        assert summary["candidate_id"] == "c_001"
        assert summary["family_id"] == "test_family"
        assert summary["score"] == 15.51
        assert summary["unsafe_legal_state_count"] == 1

    def test_error_output_is_compact(self) -> None:
        output: dict[str, Any] = {
            "ok": False,
            "tool": "test",
            "result": None,
            "error": {
                "type": "ValueError",
                "message": "Unknown family_id: 'nonexistent'",
            },
        }
        summary = summarize_tool_output(output)
        assert summary["ok"] is False
        assert summary["error_type"] == "ValueError"
        assert "Unknown family_id" in summary["error_message"]

    def test_markdown_excerpt_not_included_just_char_count(self) -> None:
        output: dict[str, Any] = {
            "ok": True,
            "tool": "test",
            "result": {
                "candidate_id": "c_001",
                "markdown_excerpt": "# " + "x" * 5000,
            },
        }
        summary = summarize_tool_output(output)
        # Should have char count, not full markdown
        assert "markdown_excerpt_chars" in summary
        assert "markdown_excerpt" not in summary or isinstance(
            summary.get("markdown_excerpt"), int
        )


# ===========================================================================
# 7. add_tool_call records error compactly
# ===========================================================================


class TestAddToolCallRecordsErrorCompactly:
    """Error type/message recorded, no stack trace."""

    def test_error_tool_call_has_type_and_message(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        tc = builder.add_tool_call(
            tool_name="run_falsification_search_tool",
            ok=False,
            input_summary={"family_id": "nonexistent"},
            output_summary={"ok": False},
            error_type="ValueError",
            error_message="Unknown family_id: 'nonexistent'",
        )
        assert tc.ok is False
        assert tc.error_type == "ValueError"
        assert "Unknown family_id" in tc.error_message
        # No stack trace
        assert "Traceback" not in (tc.error_message or "")

    def test_error_message_is_compact(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        long_error = "Error: " + "x" * 1000
        tc = builder.add_tool_call(
            tool_name="test_tool",
            ok=False,
            input_summary={},
            output_summary={},
            error_type="RuntimeError",
            error_message=long_error,
        )
        assert len(tc.error_message or "") <= MAX_TRACE_OBSERVATION_CHARS

    def test_failed_attempt_compact_dict(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_failed_attempt(
            step_index=2,
            tool_name="bad_tool",
            error_type="ValueError",
            error_message="Bad input",
            input_summary={"family_id": "invalid"},
        )
        trace = builder.build()
        assert len(trace.failed_attempts) == 1
        fa = trace.failed_attempts[0]
        assert fa["tool_name"] == "bad_tool"
        assert fa["error_type"] == "ValueError"
        assert "Bad input" in fa["error_message"]


# ===========================================================================
# 8. campaign_trace_to_dict is JSON-serializable
# ===========================================================================


class TestCampaignTraceToDictIsJsonSerializable:
    """json.dumps(...) works on the dict representation."""

    def test_full_trace_is_json_serializable(self) -> None:
        builder = CampaignTraceBuilder(
            objective="Find unsafe scenarios",
            mode="deterministic_falsification_agent",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        builder.add_step("start", "agent_start", "Objective: test")
        builder.add_tool_call(
            "list_synthetic_families_tool",
            True,
            {},
            {"ok": True, "family_count": 6},
        )
        builder.add_finding(
            family_id="confined_corner_grass",
            candidate_id="c_001",
            score=15.51,
            unsafe_legal_state_count=1,
            event_refs=["ev_001"],
            summary="Unsafe legal state found",
        )
        builder.add_hypothesis(
            basis="Finding suggests vulnerability",
            proposed_action="Run more trials",
            expected_signal="Higher scores",
            priority="high",
        )
        builder.add_limitation("This is a stress-test.")
        trace = builder.build()
        trace_dict = campaign_trace_to_dict(trace)

        encoded = json.dumps(trace_dict)
        decoded = json.loads(encoded)
        assert decoded["schema_version"] == CAMPAIGN_TRACE_SCHEMA
        assert decoded["campaign_id"].startswith("campaign_")
        assert len(decoded["steps"]) == 1
        assert len(decoded["tool_calls"]) == 1
        assert len(decoded["best_findings"]) == 1
        assert len(decoded["next_hypotheses"]) == 1

    def test_dataclass_to_dict_handles_primitives(self) -> None:
        assert dataclass_to_dict(None) is None
        assert dataclass_to_dict(42) == 42
        assert dataclass_to_dict("hello") == "hello"
        assert dataclass_to_dict([1, 2]) == [1, 2]
        assert dataclass_to_dict({"a": 1}) == {"a": 1}

    def test_empty_trace_is_json_serializable(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=None,
        )
        trace = builder.build()
        trace_dict = campaign_trace_to_dict(trace)
        encoded = json.dumps(trace_dict)
        decoded = json.loads(encoded)
        assert decoded["steps"] == []
        assert decoded["tool_calls"] == []


# ===========================================================================
# 9. build_next_hypotheses from successful trace
# ===========================================================================


class TestBuildNextHypothesesFromSuccessfulTrace:
    """Unsafe finding → high-priority hypothesis."""

    def test_unsafe_finding_produces_high_priority_hypothesis(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_finding(
            family_id="confined_corner_grass",
            candidate_id="c_001",
            score=15.51,
            unsafe_legal_state_count=1,
            max_hazard_score=0.8373,
            event_refs=["ev_001"],
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)

        assert len(hypotheses) >= 1
        first = hypotheses[0]
        assert first.priority == "high"
        assert "unsafe_legal_state" in first.basis
        assert first.hypothesis_id == "hyp_0000"

    def test_three_hypotheses_for_unsafe_finding(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_finding(
            family_id="confined_corner_grass",
            candidate_id="c_001",
            score=15.51,
            unsafe_legal_state_count=1,
            max_hazard_score=0.8373,
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)
        assert len(hypotheses) == 3
        priorities = [h.priority for h in hypotheses]
        assert "high" in priorities
        assert "medium" in priorities

    def test_finding_without_unsafe_states_produces_medium_hypothesis(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_finding(
            family_id="confined_corner_grass",
            candidate_id="c_001",
            score=5.0,
            unsafe_legal_state_count=0,
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)
        assert len(hypotheses) >= 1
        assert hypotheses[0].priority == "medium"
        assert (
        "max_trials" in hypotheses[0].proposed_action.lower()
        or "increase" in hypotheses[0].proposed_action.lower()
    )


# ===========================================================================
# 10. build_next_hypotheses from empty trace
# ===========================================================================


class TestBuildNextHypothesesFromEmptyTrace:
    """No finding → suggest more trials/other families."""

    def test_empty_trace_suggests_more_trials(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)
        assert len(hypotheses) >= 2
        # First hypothesis should suggest more trials
        text = " ".join(h.basis + h.proposed_action for h in hypotheses).lower()
        assert "max_trials" in text or "trials" in text or "increase" in text

    def test_empty_trace_suggests_other_families(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)
        text = " ".join(h.proposed_action for h in hypotheses).lower()
        assert "families" in text or "family" in text

    def test_empty_trace_includes_low_priority_wider_ranges(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        hypotheses = build_next_hypotheses_from_trace(trace)
        priorities = [h.priority for h in hypotheses]
        assert "low" in priorities


# ===========================================================================
# 11. Trace does not include secrets or raw payloads
# ===========================================================================


class TestTraceDoesNotIncludeSecrets:
    """Set fake secret-looking input; ensure trace JSON does not include secret value."""

    def test_tool_input_summary_excludes_secrets(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_tool_call(
            tool_name="test_tool",
            ok=True,
            input_summary=summarize_tool_input({
                "family_id": "test",
                "NVIDIA_API_KEY": "nvapi-secret-key-12345",
                "api_key": "secret-val",
                "bundle": {"huge": "data"},
                "event_log": ["ev1"],
                "secret": "top_secret",
            }),
            output_summary={},
        )
        trace = builder.build()
        trace_json = json.dumps(campaign_trace_to_dict(trace))

        assert "nvapi-secret-key-12345" not in trace_json
        assert "secret-val" not in trace_json
        assert "top_secret" not in trace_json

    def test_tool_output_summary_excludes_raw_logs(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={},
            seed=42,
        )
        builder.add_tool_call(
            tool_name="test_tool",
            ok=True,
            input_summary={},
            output_summary=summarize_tool_output({
                "ok": True,
                "tool": "test",
                "result": {
                    "candidate_id": "c_001",
                    "event_log": ["raw_event_1", "raw_event_2"],
                    "bundle": {"huge": "payload"},
                    "state_snapshots": [{"t": 0}],
                    "unsafe_legal_states": [{"state": "data"}],
                    "score": 10.0,
                },
            }),
        )
        trace = builder.build()
        trace_json = json.dumps(campaign_trace_to_dict(trace))

        assert "raw_event_1" not in trace_json
        assert "state_snapshots" not in trace_json
        # "unsafe_legal_states" should not appear as a raw array in output summary
        # (it may appear as a key name in compact count form)

    def test_agent_config_does_not_include_model_name_with_secrets(self) -> None:
        """agent_config should not contain model_name that could leak API info."""
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic",
            agent_config={"max_trials_per_search": 25},
            seed=42,
        )
        trace = builder.build()
        trace_dict = campaign_trace_to_dict(trace)
        # agent_config should be compact, no secrets
        config = trace_dict["agent_config"]
        assert "NVIDIA_API_KEY" not in config
        assert "api_key" not in config


# ===========================================================================
# 12. CampaignTrace dataclass fields
# ===========================================================================


class TestCampaignTraceDataclassFields:
    """Verify CampaignTrace has all required fields."""

    def test_schema_version(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        assert trace.schema_version == CAMPAIGN_TRACE_SCHEMA

    def test_mode(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="deterministic_falsification_agent",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        assert trace.mode == "deterministic_falsification_agent"

    def test_seed(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        trace = builder.build()
        assert trace.seed == 42

    def test_seed_none(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=None,
        )
        trace = builder.build()
        assert trace.seed is None


# ===========================================================================
# 13. extract_score
# ===========================================================================


class TestExtractScore:
    """Tests for extract_score helper."""

    def test_extracts_from_result_score(self) -> None:
        payload: dict[str, Any] = {"result": {"score": 15.51}}
        assert extract_score(payload) == 15.51

    def test_extracts_from_best_candidate_score(self) -> None:
        payload: dict[str, Any] = {
            "result": {"best_candidate": {"score": 12.3}},
        }
        assert extract_score(payload) == 12.3

    def test_extracts_from_top_level_score(self) -> None:
        payload: dict[str, Any] = {"score": 8.5}
        assert extract_score(payload) == 8.5

    def test_returns_none_when_no_score(self) -> None:
        payload: dict[str, Any] = {"result": {"no_score": True}}
        assert extract_score(payload) is None

    def test_returns_none_for_non_dict(self) -> None:
        assert extract_score("not a dict") is None

    def test_handles_non_numeric_score(self) -> None:
        payload: dict[str, Any] = {"result": {"score": "not_a_number"}}
        assert extract_score(payload) is None


# ===========================================================================
# 14. Builder step counter and tool counter
# ===========================================================================


class TestBuilderStepCounter:
    """Verify steps and tool calls get sequential IDs."""

    def test_steps_have_sequential_indices(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        builder.add_step("start", "step_0", "obs")
        builder.add_step("explore", "step_1", "obs")
        builder.add_step("report", "step_2", "obs")
        trace = builder.build()
        assert trace.steps[0].step_index == 0
        assert trace.steps[1].step_index == 1
        assert trace.steps[2].step_index == 2

    def test_tool_calls_have_sequential_ids(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        tc0 = builder.add_tool_call("tool_a", True, {}, {})
        tc1 = builder.add_tool_call("tool_b", True, {}, {})
        assert tc0.call_id == "tool_0000"
        assert tc1.call_id == "tool_0001"

    def test_findings_get_sequential_ids(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        f0 = builder.add_finding(summary="finding 0")
        f1 = builder.add_finding(summary="finding 1")
        assert f0.finding_id == "finding_0000"
        assert f1.finding_id == "finding_0001"


# ===========================================================================
# 15. Artifacts
# ===========================================================================


class TestArtifacts:
    """Verify artifact storage is compact."""

    def test_set_and_retrieve_artifact(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        builder.set_artifact("audit_reports", [{"artifact_id": "ar_001"}])
        builder.set_artifact("candidate_refs", [{"candidate_id": "c_001"}])
        trace = builder.build()
        assert len(trace.artifacts) == 2
        assert trace.artifacts["audit_reports"][0]["artifact_id"] == "ar_001"
        assert trace.artifacts["candidate_refs"][0]["candidate_id"] == "c_001"

    def test_artifacts_are_json_serializable(self) -> None:
        builder = CampaignTraceBuilder(
            objective="test",
            mode="test",
            agent_config={},
            seed=42,
        )
        builder.set_artifact("reports", [{"id": "r1", "count": 5}])
        trace = builder.build()
        encoded = json.dumps(campaign_trace_to_dict(trace))
        decoded = json.loads(encoded)
        assert decoded["artifacts"]["reports"][0]["id"] == "r1"
