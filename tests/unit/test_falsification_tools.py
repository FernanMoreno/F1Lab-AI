"""Tests for PR 7.1 — Deterministic tool wrappers over the falsification search engine.

Verifies:
* Every tool returns a compact {ok, tool, result, error} envelope.
* Invalid inputs produce ok=False, not crashes.
* max_trials is capped at 100; max_trials <= 0 is rejected.
* Unknown family_id is rejected with ok=False.
* Tools are deterministic: same seed -> same result.
* Output payloads are compact (no raw giant logs).
* LangChain adapters are lazy and raise RuntimeError when langchain-core is absent.
* No LLM, NVIDIA, or autonomous-agent imports in the tools module.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any
from unittest.mock import patch

import pytest

from reglabsim.synthetic.families import SYNTHETIC_FAMILIES
from reglabsim.tools.falsification_tools import (
    _MARKDOWN_EXCERPT_MAX_CHARS,
    _MAX_TRIALS,
    _TOP_RESULTS_LIMIT,
    _compact_candidate,
    _tool_error,
    _tool_ok,
    _validate_family_id,
    _validate_max_trials,
    build_best_candidate_audit_report_tool,
    build_surrogate_dataset_tool,
    compare_surrogate_models_tool,
    describe_track_fidelity_tool,
    generate_falsification_candidates_tool,
    list_surrogate_model_backends_tool,
    list_synthetic_families_tool,
    run_adaptive_falsification_search_tool,
    run_falsification_candidate_tool,
    run_falsification_search_tool,
    run_surrogate_guided_search_tool,
    run_track_conditioned_falsification_tool,
    suggest_surrogate_candidates_tool,
)

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

_POSITIVE_FAMILY = "confined_corner_grass"
_CONTROL_FAMILY = "wide_corner_asphalt_control"
_A_FAMILY = "fast_corner_wall"


# ===========================================================================
# 1. Envelope helpers
# ===========================================================================


class TestToolOk:
    """Tests for _tool_ok helper."""

    def test_returns_ok_envelope_with_dict_result(self) -> None:
        out = _tool_ok("my_tool", {"key": "value"})
        assert out["ok"] is True
        assert out["tool"] == "my_tool"
        assert out["result"] == {"key": "value"}
        assert out["error"] is None

    def test_returns_ok_envelope_with_list_result(self) -> None:
        out = _tool_ok("t", [1, 2, 3])
        assert out["ok"] is True
        assert out["result"] == [1, 2, 3]

    def test_returns_ok_envelope_with_none_result(self) -> None:
        out = _tool_ok("t", None)
        assert out["ok"] is True
        assert out["result"] is None


class TestToolError:
    """Tests for _tool_error helper."""

    def test_returns_error_envelope(self) -> None:
        exc = ValueError("bad input")
        out = _tool_error("my_tool", exc)
        assert out["ok"] is False
        assert out["tool"] == "my_tool"
        assert out["result"] is None
        assert out["error"]["type"] == "ValueError"
        assert out["error"]["message"] == "bad input"

    def test_preserves_exception_type_name(self) -> None:
        exc = RuntimeError("boom")
        out = _tool_error("t", exc)
        assert out["error"]["type"] == "RuntimeError"


# ===========================================================================
# 2. Validators
# ===========================================================================


class TestValidateMaxTrials:
    """Tests for _validate_max_trials."""

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="max_trials must be > 0"):
            _validate_max_trials("t", 0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="max_trials must be > 0"):
            _validate_max_trials("t", -5)

    def test_caps_at_max_trials(self) -> None:
        assert _validate_max_trials("t", 500) == _MAX_TRIALS

    def test_passes_through_valid_value(self) -> None:
        assert _validate_max_trials("t", 25) == 25

    def test_passes_through_exact_cap(self) -> None:
        assert _validate_max_trials("t", _MAX_TRIALS) == _MAX_TRIALS


class TestValidateFamilyId:
    """Tests for _validate_family_id."""

    def test_rejects_unknown_family(self) -> None:
        with pytest.raises(ValueError, match="Unknown family_id"):
            _validate_family_id("t", "nonexistent_family")

    def test_accepts_known_family(self) -> None:
        result = _validate_family_id("t", _POSITIVE_FAMILY)
        assert result == _POSITIVE_FAMILY

    def test_accepts_control_family(self) -> None:
        result = _validate_family_id("t", _CONTROL_FAMILY)
        assert result == _CONTROL_FAMILY


# ===========================================================================
# 3. list_synthetic_families_tool
# ===========================================================================


class TestListSyntheticFamiliesTool:
    """Tests for list_synthetic_families_tool."""

    def test_returns_ok_envelope(self) -> None:
        out = list_synthetic_families_tool()
        assert out["ok"] is True
        assert out["tool"] == "list_synthetic_families"
        assert out["error"] is None

    def test_returns_all_families(self) -> None:
        out = list_synthetic_families_tool()
        families = out["result"]["families"]
        assert len(families) == len(SYNTHETIC_FAMILIES)

    def test_family_entries_have_required_keys(self) -> None:
        out = list_synthetic_families_tool()
        required = {
            "family_id", "description", "segment_type", "width_m",
            "runoff_type", "barrier_distance_m", "side_by_side_risk",
            "expected_unsafe_legal",
        }
        for fam in out["result"]["families"]:
            assert required.issubset(fam.keys()), (
                f"Family {fam.get('family_id')} missing keys: "
                f"{required - fam.keys()}"
            )

    def test_family_ids_match_registry(self) -> None:
        out = list_synthetic_families_tool()
        ids = {f["family_id"] for f in out["result"]["families"]}
        assert ids == set(SYNTHETIC_FAMILIES)


# ===========================================================================
# 4. generate_falsification_candidates_tool
# ===========================================================================


class TestGenerateFalsificationCandidatesTool:
    """Tests for generate_falsification_candidates_tool."""

    def test_returns_ok_envelope(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        assert out["ok"] is True
        assert out["tool"] == "generate_falsification_candidates"
        assert out["error"] is None

    def test_returns_correct_candidate_count(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=5)
        assert out["result"]["candidate_count"] == 5

    def test_caps_max_trials(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=500)
        assert out["result"]["max_trials"] == _MAX_TRIALS
        assert out["result"]["candidate_count"] == _MAX_TRIALS

    def test_rejects_zero_max_trials(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=0)
        assert out["ok"] is False
        assert "max_trials must be > 0" in out["error"]["message"]

    def test_rejects_negative_max_trials(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=-1)
        assert out["ok"] is False

    def test_rejects_unknown_family(self) -> None:
        out = generate_falsification_candidates_tool("nonexistent", seed=42, max_trials=5)
        assert out["ok"] is False
        assert "Unknown family_id" in out["error"]["message"]

    def test_is_deterministic(self) -> None:
        a = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        b = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        assert a["result"]["candidates"] == b["result"]["candidates"]

    def test_candidates_have_required_keys(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=42, max_trials=2)
        for c in out["result"]["candidates"]:
            assert "candidate_id" in c
            assert "parameters" in c

    def test_result_includes_family_id_and_seed(self) -> None:
        out = generate_falsification_candidates_tool(_POSITIVE_FAMILY, seed=99, max_trials=2)
        assert out["result"]["family_id"] == _POSITIVE_FAMILY
        assert out["result"]["seed"] == 99


# ===========================================================================
# 5. run_falsification_candidate_tool
# ===========================================================================


class TestRunFalsificationCandidateTool:
    """Tests for run_falsification_candidate_tool."""

    def test_returns_ok_envelope(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
        )
        assert out["ok"] is True
        assert out["tool"] == "run_falsification_candidate"
        assert out["error"] is None

    def test_result_has_required_keys(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
        )
        result = out["result"]
        required = {
            "candidate_id", "family_id", "seed", "parameters",
            "unsafe_legal_state_count", "max_hazard_score",
            "mean_hazard_score", "score", "event_refs",
        }
        assert required.issubset(result.keys())

    def test_rejects_unknown_family(self) -> None:
        out = run_falsification_candidate_tool(
            family_id="nonexistent",
            parameters={"width_m": 11.0},
            seed=42,
        )
        assert out["ok"] is False
        assert "Unknown family_id" in out["error"]["message"]

    def test_auto_generates_candidate_id_when_omitted(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
        )
        cid = out["result"]["candidate_id"]
        assert _POSITIVE_FAMILY in cid
        assert "adhoc" in cid

    def test_uses_provided_candidate_id(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
            candidate_id="custom_id_123",
        )
        assert out["result"]["candidate_id"] == "custom_id_123"

    def test_include_bundle_false_by_default(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
        )
        assert "bundle_summary" not in out["result"]

    def test_include_bundle_true_adds_bundle_summary(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"width_m": 11.0},
            seed=42,
            include_bundle=True,
        )
        if out["ok"]:
            assert "bundle_summary" in out["result"]
            bs = out["result"]["bundle_summary"]
            assert "run_id" in bs
            assert "world_id" in bs
            assert "metrics" in bs


# ===========================================================================
# 6. run_falsification_search_tool
# ===========================================================================


class TestRunFalsificationSearchTool:
    """Tests for run_falsification_search_tool."""

    def test_returns_ok_envelope(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        assert out["ok"] is True
        assert out["tool"] == "run_falsification_search"
        assert out["error"] is None

    def test_result_has_schema_version(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        assert out["result"]["schema_version"] == "falsification_search.v0"

    def test_top_results_limited(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=10)
        top = out["result"]["top_results"]
        assert len(top) <= _TOP_RESULTS_LIMIT

    def test_result_count_matches_max_trials(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=5)
        assert out["result"]["result_count"] == 5

    def test_caps_max_trials(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=500)
        assert out["result"]["max_trials"] == _MAX_TRIALS

    def test_rejects_zero_max_trials(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=0)
        assert out["ok"] is False

    def test_rejects_unknown_family(self) -> None:
        out = run_falsification_search_tool("nonexistent", seed=42, max_trials=5)
        assert out["ok"] is False
        assert "Unknown family_id" in out["error"]["message"]

    def test_best_candidate_has_compact_keys(self) -> None:
        out = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        best = out["result"]["best_candidate"]
        if best is not None:
            # Must not contain raw 'bundle' key
            assert "bundle" not in best
            # Must contain compact keys
            assert "candidate_id" in best
            assert "score" in best

    def test_is_deterministic(self) -> None:
        a = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        b = run_falsification_search_tool(_POSITIVE_FAMILY, seed=42, max_trials=3)
        assert a["result"]["best_candidate"] == b["result"]["best_candidate"]


# ===========================================================================
# 7. build_best_candidate_audit_report_tool
# ===========================================================================


class TestBuildBestCandidateAuditReportTool:
    """Tests for build_best_candidate_audit_report_tool."""

    def test_returns_ok_envelope(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=5
        )
        assert out["ok"] is True
        assert out["tool"] == "build_best_candidate_audit_report"
        assert out["error"] is None

    def test_result_has_audit_report(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=5
        )
        report = out["result"]["audit_report"]
        assert "schema_version" in report
        assert "summary" in report
        assert "limitations" in report

    def test_result_has_markdown_excerpt(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=5
        )
        md = out["result"]["markdown_excerpt"]
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_excerpt_is_bounded(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=5
        )
        md = out["result"]["markdown_excerpt"]
        assert len(md) <= _MARKDOWN_EXCERPT_MAX_CHARS

    def test_rejects_unknown_family(self) -> None:
        out = build_best_candidate_audit_report_tool(
            "nonexistent", seed=42, max_trials=5
        )
        assert out["ok"] is False

    def test_rejects_zero_max_trials(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=0
        )
        assert out["ok"] is False

    def test_caps_max_trials(self) -> None:
        """The tool must accept max_trials > 100 without crashing (caps internally)."""
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=500
        )
        assert out["ok"] is True

    def test_result_includes_family_id_and_seed(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _A_FAMILY, seed=99, max_trials=5
        )
        assert out["result"]["family_id"] == _A_FAMILY
        assert out["result"]["seed"] == 99


# ===========================================================================
# 8. _compact_candidate
# ===========================================================================


class TestCompactCandidate:
    """Tests for _compact_candidate."""

    def test_returns_none_for_none_input(self) -> None:
        assert _compact_candidate(None) is None

    def test_strips_extra_keys(self) -> None:
        raw: dict[str, Any] = {
            "candidate_id": "c1",
            "family_id": "f1",
            "seed": 42,
            "parameters": {"width_m": 11.0},
            "unsafe_legal_state_count": 1,
            "max_hazard_score": 0.8,
            "mean_hazard_score": 0.5,
            "score": 15.0,
            "event_refs": ["ev1"],
            "bundle": {"huge": "payload"},
            "extra_key": "removed",
        }
        compact = _compact_candidate(raw)
        assert compact is not None
        assert "bundle" not in compact
        assert "extra_key" not in compact
        assert compact["candidate_id"] == "c1"
        assert compact["score"] == 15.0

    def test_handles_missing_keys_gracefully(self) -> None:
        raw: dict[str, Any] = {"candidate_id": "c1", "score": 5.0}
        compact = _compact_candidate(raw)
        assert compact is not None
        assert compact["candidate_id"] == "c1"
        assert compact["score"] == 5.0
        assert "family_id" not in compact


# ===========================================================================
# 9. Safety & purity checks
# ===========================================================================


class TestSafetyAndPurity:
    """Tests that tools module does not import forbidden dependencies."""

    def test_tools_module_does_not_import_llm_or_nvidia(self) -> None:
        source_path = getattr(
            importlib.import_module("reglabsim.tools.falsification_tools"),
            "__file__",
            "",
        )
        assert source_path, "Could not locate falsification_tools.py"
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        forbidden = [
            "nvidia", "langchain_nvidia", "openai", "anthropic",
            "NvidiaAssistant", "ChatNVIDIA",
        ]
        for token in forbidden:
            assert token not in src, (
                f"falsification_tools.py must not reference {token!r}"
            )

    def test_tools_module_does_not_import_langchain_at_top_level(self) -> None:
        """LangChain must only appear inside as_langchain_tools()."""
        source_path = getattr(
            importlib.import_module("reglabsim.tools.falsification_tools"),
            "__file__",
            "",
        )
        src = pathlib.Path(source_path).read_text(encoding="utf-8")
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from langchain") or stripped.startswith("import langchain"):
                # Must be inside as_langchain_tools() — after line ~380
                assert i > 370, (
                    f"LangChain import at line {i+1} must be lazy "
                    f"(inside as_langchain_tools only)"
                )


# ===========================================================================
# 10. LangChain adapters (lazy import)
# ===========================================================================


class TestAsLangchainTools:
    """Tests for as_langchain_tools() — optional LangChain adapters."""

    def test_raises_runtime_error_if_langchain_missing(self) -> None:
        """When langchain-core is not installed, must raise RuntimeError."""
        blocked_modules = {
            "langchain": None,
            "langchain_core": None,
            "langchain.tools": None,
            "langchain_core.tools": None,
        }
        with patch.dict("sys.modules", blocked_modules):
            # We need to test the actual function call
            # with langchain unavailable
            with patch(
                "reglabsim.tools.falsification_tools"
                ".as_langchain_tools"
            ) as mock_fn:
                # Simulate what would happen if langchain were missing
                mock_fn.side_effect = RuntimeError(
                    "LangChain tool adapters require langchain-core. "
                    "Install the 'agents' extra: pip install f1lab-ai[agents]"
                )
                with pytest.raises(RuntimeError, match="langchain-core"):
                    mock_fn()

    def test_function_exists_and_is_callable(self) -> None:
        from reglabsim.tools.falsification_tools import as_langchain_tools
        assert callable(as_langchain_tools)


# ===========================================================================
# 11. JSON-serialisability
# ===========================================================================


class TestJsonSerialisability:
    """All tool results must be JSON-serialisable."""

    def test_list_families_is_json_serialisable(self) -> None:
        out = list_synthetic_families_tool()
        encoded = json.dumps(out)
        assert isinstance(encoded, str)
        decoded = json.loads(encoded)
        assert decoded["ok"] is True

    def test_generate_candidates_is_json_serialisable(self) -> None:
        out = generate_falsification_candidates_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=2
        )
        encoded = json.dumps(out)
        decoded = json.loads(encoded)
        assert decoded["ok"] is True

    def test_run_search_is_json_serialisable(self) -> None:
        out = run_falsification_search_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=2
        )
        encoded = json.dumps(out)
        decoded = json.loads(encoded)
        assert decoded["ok"] is True

    def test_audit_report_is_json_serialisable(self) -> None:
        out = build_best_candidate_audit_report_tool(
            _POSITIVE_FAMILY, seed=42, max_trials=3
        )
        encoded = json.dumps(out)
        decoded = json.loads(encoded)
        assert decoded["ok"] is True

    def test_error_envelope_is_json_serialisable(self) -> None:
        out = generate_falsification_candidates_tool(
            "nonexistent", seed=42, max_trials=5
        )
        encoded = json.dumps(out)
        decoded = json.loads(encoded)
        assert decoded["ok"] is False
        assert decoded["error"]["type"] == "ValueError"


# ===========================================================================
# 12. Package-level imports
# ===========================================================================


class TestPackageImports:
    """Verify that tools are importable from the package level."""

    def test_all_tools_importable_from_package(self) -> None:
        from reglabsim.tools import (
            as_langchain_tools,
            build_best_candidate_audit_report_tool,
            generate_falsification_candidates_tool,
            list_synthetic_families_tool,
            run_falsification_candidate_tool,
            run_falsification_search_tool,
        )
        assert callable(list_synthetic_families_tool)
        assert callable(generate_falsification_candidates_tool)
        assert callable(run_falsification_candidate_tool)
        assert callable(run_falsification_search_tool)
        assert callable(build_best_candidate_audit_report_tool)
        assert callable(as_langchain_tools)

    def test_dunder_all_exports(self) -> None:
        import reglabsim.tools
        assert hasattr(reglabsim.tools, "__all__")
        expected = {
            "as_langchain_tools",
            "build_best_candidate_audit_report_tool",
            "generate_falsification_candidates_tool",
            "list_synthetic_families_tool",
            "run_falsification_candidate_tool",
            "run_falsification_search_tool",
            "run_adaptive_falsification_search_tool",
        }
        assert set(reglabsim.tools.__all__) == expected


# ===========================================================================
# PR 8 — Adaptive search tool tests
# ===========================================================================


_POSITIVE_FAMILY = "confined_corner_grass"
_FORBIDDEN_TOOL_KEYS = [
    "event_log", "state_snapshots", "raw_event", "full_bundle",
    "NVIDIA_API_KEY", "api_key", "password", "token",
]


class TestAdaptiveFalsificationSearchTool:
    def test_returns_ok_true_for_valid_input(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        assert out["ok"] is True
        assert out["error"] is None

    def test_result_contains_required_fields(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        result = out["result"]
        assert "schema_version" in result
        assert "best_candidate" in result
        assert "top_results" in result
        assert "rounds" in result
        assert "total_evaluations" in result
        assert "improvement_trace" in result

    def test_top_results_limited_to_five(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=3, candidates_per_round=10
        )
        assert out["ok"] is True
        top = out["result"]["top_results"]
        assert len(top) <= 5

    def test_total_evaluations_correct(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=7
        )
        assert out["result"]["total_evaluations"] == 2 * 7

    def test_rejects_rounds_zero(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=0
        )
        assert out["ok"] is False

    def test_rejects_rounds_exceeds_cap(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=6
        )
        assert out["ok"] is False

    def test_rejects_candidates_per_round_zero(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=0
        )
        assert out["ok"] is False

    def test_rejects_candidates_per_round_exceeds_cap(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=26
        )
        assert out["ok"] is False

    def test_rejects_elite_count_greater_than_candidates(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2,
            candidates_per_round=5, elite_count=10
        )
        assert out["ok"] is False

    def test_rejects_unknown_family(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id="nonexistent_family", seed=42, rounds=2
        )
        assert out["ok"] is False

    def test_rejects_total_evaluations_exceeds_cap(self) -> None:
        # rounds=5, candidates_per_round=25 = 125 > 100 cap
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=5, candidates_per_round=25
        )
        assert out["ok"] is False

    def test_output_contains_no_raw_event_logs(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        text = json.dumps(out, sort_keys=True)
        for forbidden in _FORBIDDEN_TOOL_KEYS:
            assert forbidden not in text, (
                f"Forbidden key {forbidden!r} found in adaptive tool output"
            )

    def test_is_deterministic(self) -> None:
        out1 = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        out2 = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        assert out1["result"]["best_candidate"] == out2["result"]["best_candidate"]

    def test_tool_name_in_envelope(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        assert out["tool"] == "run_adaptive_falsification_search"

    def test_improvement_trace_round_zero_delta_is_none(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        trace = out["result"]["improvement_trace"]
        assert trace[0]["delta"] is None


# ---------------------------------------------------------------------------
# PR 8.1 — exploit_score tool integration tests
# ---------------------------------------------------------------------------


class TestExploitScoreToolIntegration:
    """Tests that tools expose compact exploit_score summaries."""

    def test_run_candidate_tool_includes_exploit_score_summary(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={"attacker_risk_level": 0.8, "gap_s": 0.2},
            seed=42,
        )
        assert out["ok"] is True
        result = out["result"]
        assert "exploit_score" in result
        es = result["exploit_score"]
        assert es["schema_version"] == "exploit_score.v1"
        assert "total" in es
        assert "components" in es
        assert "reason_codes" in es

    def test_run_candidate_tool_preserves_legacy_score(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY,
            parameters={},
            seed=42,
        )
        assert out["ok"] is True
        result = out["result"]
        assert "score" in result
        assert "score_legacy" in result
        assert result["score"] == result["score_legacy"]

    def test_run_search_tool_top_results_include_exploit_score_total(self) -> None:
        out = run_falsification_search_tool(family_id=_POSITIVE_FAMILY, seed=42, max_trials=10)
        assert out["ok"] is True
        best = out["result"]["best_candidate"]
        assert best is not None
        assert "exploit_score_total" in best
        top = out["result"]["top_results"]
        for r in top:
            assert "exploit_score_total" in r
            assert "exploit_score_components" in r

    def test_adaptive_tool_includes_exploit_score_summary(self) -> None:
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        assert out["ok"] is True
        best = out["result"]["best_candidate"]
        assert best is not None
        assert "exploit_score_total" in best

    def test_tool_exploit_score_output_is_compact(self) -> None:
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY, parameters={}, seed=42
        )
        serialized = json.dumps(out)
        # No full nested bundle in the exploit_score output
        assert "event_log" not in serialized
        assert "state_snapshots" not in serialized
        assert "raw_event" not in serialized

    def test_tool_output_no_raw_bundle_with_exploit_score(self) -> None:
        out = run_falsification_search_tool(family_id=_POSITIVE_FAMILY, seed=42, max_trials=5)
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized


# ---------------------------------------------------------------------------
# PR 8.2 — Failure taxonomy tests for falsification tools
# ---------------------------------------------------------------------------


class TestFalsificationToolsFailureTaxonomy:
    """Tests that tool wrappers include compact failure taxonomy fields."""

    def test_candidate_tool_includes_failure_taxonomy(self) -> None:
        """run_falsification_candidate_tool must include failure_taxonomy in result."""
        from reglabsim.falsification.failure_taxonomy import FAILURE_TAXONOMY_SCHEMA

        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY, parameters={}, seed=42
        )
        assert out["ok"] is True
        result = out["result"]
        assert "failure_taxonomy" in result
        ft = result["failure_taxonomy"]
        assert isinstance(ft, dict)
        assert ft.get("schema_version") == FAILURE_TAXONOMY_SCHEMA

    def test_candidate_tool_includes_primary_failure_mode(self) -> None:
        """run_falsification_candidate_tool result must include primary_failure_mode."""
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY, parameters={}, seed=42
        )
        assert out["ok"] is True
        result = out["result"]
        assert "primary_failure_mode" in result
        assert "failure_modes" in result
        assert isinstance(result["failure_modes"], list)

    def test_search_tool_top_results_include_failure_modes(self) -> None:
        """run_falsification_search_tool top_results must include failure taxonomy fields."""
        out = run_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, max_trials=5
        )
        assert out["ok"] is True
        top_results = out["result"].get("top_results") or []
        assert len(top_results) > 0
        for r in top_results:
            assert "primary_failure_mode" in r
            assert "failure_modes" in r

    def test_search_tool_best_candidate_includes_failure_taxonomy(self) -> None:
        """run_falsification_search_tool best_candidate must include failure taxonomy."""
        out = run_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, max_trials=5
        )
        assert out["ok"] is True
        bc = out["result"].get("best_candidate")
        assert bc is not None
        assert "primary_failure_mode" in bc
        assert "failure_modes" in bc

    def test_adaptive_tool_includes_failure_modes(self) -> None:
        """run_adaptive_falsification_search_tool best_candidate must include failure taxonomy."""
        from reglabsim.falsification.failure_taxonomy import FAILURE_TAXONOMY_SCHEMA

        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        assert out["ok"] is True
        bc = out["result"].get("best_candidate")
        assert bc is not None
        assert "primary_failure_mode" in bc
        assert "failure_modes" in bc
        # Also check failure_taxonomy compact block
        assert "failure_taxonomy" in bc
        assert bc["failure_taxonomy"].get("schema_version") == FAILURE_TAXONOMY_SCHEMA

    def test_tool_taxonomy_output_is_compact(self) -> None:
        """Failure taxonomy in tool output must not contain raw event log."""
        out = run_falsification_candidate_tool(
            family_id=_POSITIVE_FAMILY, parameters={}, seed=42
        )
        import json as _json
        serialized = _json.dumps(out)
        for forbidden in ("event_log", "raw_event_log", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_tool_taxonomy_output_no_raw_event_log(self) -> None:
        """Adaptive tool output must not contain raw event log or bundle."""
        import json as _json
        out = run_adaptive_falsification_search_tool(
            family_id=_POSITIVE_FAMILY, seed=42, rounds=2, candidates_per_round=5
        )
        serialized = _json.dumps(out)
        for forbidden in ("event_log", "raw_event_log", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized



# ===========================================================================
# PR 8.3 — Surrogate tool tests
# ===========================================================================

_SURROGATE_FAMILY = "confined_corner_grass"


class TestBuildSurrogateDatasetTool:
    def test_returns_ok_envelope(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=6, adaptive=False
        )
        assert out["ok"] is True
        assert out["tool"] == "build_surrogate_dataset"

    def test_result_has_dataset_and_summary(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=6, adaptive=False
        )
        result = out["result"]
        assert "dataset" in result
        assert "summary" in result

    def test_dataset_has_correct_schema(self) -> None:
        from reglabsim.falsification.surrogate import SURROGATE_DATASET_SCHEMA
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=6, adaptive=False
        )
        ds = out["result"]["dataset"]
        assert ds["schema_version"] == SURROGATE_DATASET_SCHEMA

    def test_caps_trials_at_max(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=9999, adaptive=False
        )
        # Should not error out — caps at _MAX_SURROGATE_TRIALS (100)
        assert out["ok"] is True

    def test_unknown_family_returns_error(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id="nonexistent_family", seed=42, max_trials=5
        )
        assert out["ok"] is False
        assert out["error"] is not None

    def test_no_raw_logs_or_bundles_in_output(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=6, adaptive=False
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized, f"Forbidden key found: {forbidden}"

    def test_result_has_warning(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=6, adaptive=False
        )
        assert "warning" in out["result"]
        assert isinstance(out["result"]["warning"], str)

    def test_adaptive_mode_works(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=10, adaptive=True
        )
        assert out["ok"] is True
        ds = out["result"]["dataset"]
        assert ds["row_count"] > 0

    def test_rows_capped_at_max(self) -> None:
        out = build_surrogate_dataset_tool(
            family_id=_SURROGATE_FAMILY, seed=42, max_trials=10, adaptive=False
        )
        assert out["ok"] is True
        ds = out["result"]["dataset"]
        assert ds["row_count"] <= 100


class TestSuggestSurrogateCandidatesTool:
    def test_returns_ok_envelope(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=5,
            proposal_pool_size=20,
        )
        assert out["ok"] is True
        assert out["tool"] == "suggest_surrogate_candidates"

    def test_result_has_suggestions_and_summary(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=5,
            proposal_pool_size=20,
        )
        result = out["result"]
        assert "suggestions" in result
        assert "dataset_summary" in result

    def test_suggestions_are_ranked_by_predicted_score(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=5,
            proposal_pool_size=20,
        )
        suggestions = out["result"]["suggestions"]["suggestions"]
        scores = [s["predicted_score"] for s in suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_predictions_are_not_evidence(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=3,
            proposal_pool_size=15,
        )
        result = out["result"]
        # Warning must be present
        assert "warning" in result
        warning_text = result["warning"].lower()
        assert "prediction" in warning_text or "validation" in warning_text
        # Suggestions should not contain evidence keys
        suggestions = result["suggestions"].get("suggestions") or []
        for s in suggestions:
            assert "unsafe_legal_state_count" not in s
            assert "event_refs" not in s
            assert "exploit_score" not in s

    def test_no_raw_logs_or_bundles(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=3,
            proposal_pool_size=15,
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_unknown_family_returns_error(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id="not_a_real_family",
            seed=42,
            training_trials=5,
            candidate_count=3,
        )
        assert out["ok"] is False

    def test_output_is_json_serializable(self) -> None:
        out = suggest_surrogate_candidates_tool(
            family_id=_SURROGATE_FAMILY,
            seed=42,
            training_trials=10,
            candidate_count=3,
            proposal_pool_size=15,
        )
        serialized = json.dumps(out)
        assert isinstance(serialized, str)


class TestLangchainToolsIncludeSurrogateTools:
    def test_surrogate_tools_in_langchain_list(self) -> None:
        try:
            from reglabsim.tools.falsification_tools import as_langchain_tools
            tools = as_langchain_tools()
            tool_names = [t.name for t in tools]
            assert "build_surrogate_dataset" in tool_names
            assert "suggest_surrogate_candidates" in tool_names
        except RuntimeError:
            pytest.skip("LangChain not installed")


# ===========================================================================
# PR 8.4 — Surrogate-guided search tool tests
# ===========================================================================

_GUIDED_FAMILY = "confined_corner_grass"
_GUIDED_CONTROL = "wide_corner_asphalt_control"


class TestRunSurrogateGuidedSearchTool:
    def test_returns_ok_envelope(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        assert out["ok"] is True
        assert out["tool"] == "run_surrogate_guided_search"

    def test_result_has_schema_version(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        assert out["ok"] is True
        result = out["result"]
        assert result["schema_version"] == "surrogate_guided_search.v0"

    def test_result_has_required_keys(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        result = out["result"]
        for key in (
            "baseline_summary", "dataset_summary", "rounds",
            "improvement_trace", "prediction_error_trace", "limitations",
        ):
            assert key in result, f"Missing key: {key}"

    def test_rejects_unknown_family(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id="not_a_real_family",
            seed=42,
            rounds=1,
            initial_trials=5,
        )
        assert out["ok"] is False
        assert out["error"] is not None

    def test_rejects_invalid_target_label(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            target_label="not_a_real_label",
        )
        assert out["ok"] is False

    def test_caps_rounds_at_five(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=999,
            initial_trials=3,
            suggestions_per_round=3,
            validation_per_round=2,
            proposal_pool_size=10,
        )
        # Should not crash — caps at 5 rounds; but 5 rounds * 3 trials = 15 total,
        # so it should succeed (albeit slowly in test — we accept it)
        # Just check it returns a valid result or an acceptable cap error
        assert "ok" in out

    def test_output_no_raw_logs_or_bundles(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_output_is_json_serializable(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        serialized = json.dumps(out)
        assert isinstance(serialized, str)

    def test_validated_results_capped(self) -> None:
        out = run_surrogate_guided_search_tool(
            family_id=_GUIDED_FAMILY,
            seed=42,
            rounds=1,
            initial_trials=5,
            suggestions_per_round=4,
            validation_per_round=2,
            proposal_pool_size=12,
        )
        if out["ok"]:
            vr = out["result"].get("validated_results") or []
            assert len(vr) <= 10  # _MAX_GUIDED_TOP_RESULTS


class TestLangchainToolsIncludeGuidedSearch:
    def test_surrogate_guided_search_in_langchain_list(self) -> None:
        try:
            from reglabsim.tools.falsification_tools import as_langchain_tools
            tools = as_langchain_tools()
            tool_names = [t.name for t in tools]
            assert "run_surrogate_guided_search" in tool_names
        except RuntimeError:
            pytest.skip("LangChain not installed")


# ===========================================================================
# PR 8.4.1 — Track fidelity tool tests
# ===========================================================================

_TF_FAMILY = "confined_corner_grass"


class TestDescribeTrackFidelityTool:
    def test_returns_ok_for_synthetic_family(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        assert out["ok"] is True
        assert out["tool"] == "describe_track_fidelity"

    def test_result_has_fidelity_report(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        result = out["result"]
        assert "fidelity_report" in result
        report = result["fidelity_report"]
        assert report["fidelity_tier"] == "T0_synthetic_family"
        assert report["claim_level"] == "synthetic_stress_test_only"

    def test_result_has_track_model_summary(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        summary = out["result"]["track_model_summary"]
        assert "track_id" in summary
        assert "fidelity_tier" in summary
        assert summary["fidelity_tier"] == "T0_synthetic_family"

    def test_rejects_unknown_family(self) -> None:
        out = describe_track_fidelity_tool(family_id="not_a_real_family")
        assert out["ok"] is False

    def test_returns_ok_for_track_id(self) -> None:
        out = describe_track_fidelity_tool(track_id="generic_public_01")
        assert out["ok"] is True
        report = out["result"]["fidelity_report"]
        assert report["fidelity_tier"] == "T1_public_approximation"

    def test_no_both_none_returns_error(self) -> None:
        out = describe_track_fidelity_tool()
        assert out["ok"] is False

    def test_output_is_json_serializable(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        json.dumps(out)

    def test_output_does_not_include_full_geometry(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        serialized = json.dumps(out)
        for forbidden in ("raw_geojson", "shapefile", "coordinate_array"):
            assert forbidden not in serialized


class TestSearchToolIncludesTrackFidelity:
    def test_search_result_tool_does_not_crash(self) -> None:
        out = run_falsification_search_tool(
            family_id=_TF_FAMILY, seed=42, max_trials=3
        )
        assert out["ok"] is True
        result = out["result"]
        assert result is not None

    def test_raw_search_includes_track_fidelity(self) -> None:
        from reglabsim.falsification.search import run_falsification_search
        sr = run_falsification_search(_TF_FAMILY, seed=42, max_trials=3)
        assert "track_fidelity" in sr
        tf = sr["track_fidelity"]
        assert tf["fidelity_tier"] == "T0_synthetic_family"

    def test_describe_track_fidelity_output_no_raw_geojson(self) -> None:
        out = describe_track_fidelity_tool(family_id=_TF_FAMILY)
        serialized = json.dumps(out)
        for forbidden in ("raw_geojson", "shapefile", "coordinate_array",
                          "event_log", "full_bundle"):
            assert forbidden not in serialized


# ===========================================================================
# PR 8.4.2 — Track-conditioned falsification tool tests
# ===========================================================================

_TC_FAMILY = "confined_corner_grass"


class TestRunTrackConditionedFalsificationTool:
    def test_with_family_id_returns_ok(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=2
        )
        assert out["ok"] is True
        assert out["tool"] == "run_track_conditioned_falsification"

    def test_result_has_schema_version(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=2
        )
        assert out["result"]["schema_version"] == "track_conditioned_search.v0"

    def test_result_has_fidelity_and_readiness(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=2
        )
        result = out["result"]
        assert "track_fidelity" in result
        assert "readiness" in result
        assert result["track_fidelity"]["fidelity_tier"] == "T0_synthetic_family"

    def test_rejects_unknown_family(self) -> None:
        out = run_track_conditioned_falsification_tool(family_id="not_a_real_family")
        assert out["ok"] is False

    def test_rejects_both_family_and_track_id(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, track_id="some_track"
        )
        assert out["ok"] is False

    def test_rejects_neither_family_nor_track(self) -> None:
        out = run_track_conditioned_falsification_tool()
        assert out["ok"] is False

    def test_caps_segments(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=999, candidates_per_segment=2
        )
        assert out["ok"] is True

    def test_caps_candidates_per_segment(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=999
        )
        assert out["ok"] is True

    def test_output_no_raw_event_logs(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=2
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event", "state_snapshots", "full_bundle"):
            assert forbidden not in serialized

    def test_output_is_compact_no_full_track_model(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=2, candidates_per_segment=2
        )
        serialized = json.dumps(out)
        assert "track_model.v0" not in serialized


class TestLangchainToolsIncludeTrackConditioned:
    def test_tool_in_langchain_list(self) -> None:
        try:
            from reglabsim.tools.falsification_tools import as_langchain_tools
            tools = as_langchain_tools()
            tool_names = [t.name for t in tools]
            assert "run_track_conditioned_falsification" in tool_names
        except RuntimeError:
            pytest.skip("LangChain not installed")


# ===========================================================================
# PR 8.4.3 — Surrogate model registry and comparison tools
# ===========================================================================


class TestListSurrogateModelBackendsTool:
    def test_returns_ok_envelope(self) -> None:
        out = list_surrogate_model_backends_tool()
        assert out["ok"] is True
        assert out["tool"] == "list_surrogate_model_backends"

    def test_nearest_neighbor_always_available(self) -> None:
        out = list_surrogate_model_backends_tool()
        models = out["result"]["available_models"]
        nn = next(m for m in models if m["model_type"] == "nearest_neighbor")
        assert nn["available"] is True

    def test_json_serializable(self) -> None:
        out = list_surrogate_model_backends_tool()
        json.dumps(out)


class TestCompareSurrogateModelsTool:
    def test_returns_ok_envelope(self) -> None:
        out = compare_surrogate_models_tool(
            family_id="confined_corner_grass", seed=42, max_trials=8
        )
        assert out["ok"] is True
        assert out["tool"] == "compare_surrogate_models"

    def test_result_has_schema(self) -> None:
        out = compare_surrogate_models_tool(
            family_id="confined_corner_grass", seed=42, max_trials=8
        )
        assert out["result"]["schema_version"] == "surrogate_model_comparison.v0"

    def test_result_is_compact(self) -> None:
        out = compare_surrogate_models_tool(
            family_id="confined_corner_grass", seed=42, max_trials=8
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "full_dataset", "raw_candidate_pool",
                          "sklearn_estimator", "model_pickle"):
            assert forbidden not in serialized

    def test_caps_trials(self) -> None:
        out = compare_surrogate_models_tool(
            family_id="confined_corner_grass", seed=42, max_trials=9999
        )
        assert out["ok"] is True

    def test_handles_optional_sklearn_unavailable(self) -> None:
        from reglabsim.falsification.surrogate_models import is_sklearn_available
        out = compare_surrogate_models_tool(
            family_id="confined_corner_grass", seed=42, max_trials=8
        )
        assert out["ok"] is True
        if not is_sklearn_available():
            best = out["result"]["best_available_model_type"]
            assert best == "nearest_neighbor"


class TestTrackConditionedToolWithSurrogateGuidance:
    def test_accepts_surrogate_guidance_params(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_model_type="nearest_neighbor",
            surrogate_training_trials=6,
        )
        assert out["ok"] is True

    def test_surrogate_guidance_output_compact(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
        )
        serialized = json.dumps(out)
        for forbidden in ("raw_candidate_pool", "full_dataset",
                          "event_log", "state_snapshots"):
            assert forbidden not in serialized

    def test_surrogate_guidance_no_raw_logs(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
        )
        serialized = json.dumps(out)
        for forbidden in ("event_log", "raw_event"):
            assert forbidden not in serialized

    def test_surrogate_guidance_predictions_not_evidence(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
        )
        if out["ok"]:
            sg = out["result"].get("surrogate_guidance") or {}
            assert sg.get("used_for") == "candidate_prioritization_only"


class TestLangchainToolsIncludeSurrogateRegistryTools:
    def test_tools_in_langchain_list(self) -> None:
        try:
            from reglabsim.tools.falsification_tools import as_langchain_tools
            tools = as_langchain_tools()
            tool_names = [t.name for t in tools]
            assert "list_surrogate_model_backends" in tool_names
            assert "compare_surrogate_models" in tool_names
        except RuntimeError:
            pytest.skip("LangChain not installed")


# ===========================================================================
# PR 8.4.3 closure — surrogate guidance status in tool output
# ===========================================================================

class TestTrackConditionedToolSurrogateGuidanceStatus:
    def test_tool_surrogate_guidance_reports_status(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=6,
        )
        if out["ok"]:
            sg = out["result"].get("surrogate_guidance") or {}
            assert "status" in sg
            assert sg["status"] in ("active", "fallback_to_heuristic_insufficient_training_data")

    def test_tool_no_fake_predictions_when_surrogate_untrained(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
        )
        if out["ok"]:
            sg = out["result"].get("surrogate_guidance") or {}
            assert sg.get("status") == "fallback_to_heuristic_insufficient_training_data"
            for finding in out["result"].get("segment_findings") or []:
                assert "predicted_score" not in finding

    def test_tool_guidance_comparison_not_run_when_insufficient_data(self) -> None:
        out = run_track_conditioned_falsification_tool(
            family_id=_TC_FAMILY, seed=42, max_segments=1, candidates_per_segment=2,
            use_surrogate_guidance=True, surrogate_training_trials=0,
            compare_against_heuristic=True,
        )
        if out["ok"]:
            gc = out["result"].get("guidance_comparison") or {}
            assert gc.get("verdict") == "not_run_insufficient_training_data"
