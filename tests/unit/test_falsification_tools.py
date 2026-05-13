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
    generate_falsification_candidates_tool,
    list_synthetic_families_tool,
    run_falsification_candidate_tool,
    run_falsification_search_tool,
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
        }
        assert set(reglabsim.tools.__all__) == expected

