"""PR 6.1 — Tests for NVIDIA/LangChain assistant.

All tests use a fake LLM — no NVIDIA_API_KEY required.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from reglabsim.llm.nvidia_assistant import (
    _extract_response_content,
    build_audit_summary_prompt,
    build_counterfactual_review_prompt,
    build_nvidia_llm,
    build_synthetic_family_review_prompt,
    review_counterfactual_with_llm,
    review_synthetic_families_with_llm,
    summarize_audit_report_with_llm,
)

# ---------------------------------------------------------------------------
# Fake LLM for injecting into assistant functions
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal injectable LLM stub — no API calls."""

    def __init__(self, content: str = "Fake summary") -> None:
        self._content = content

    def invoke(self, prompt: str) -> Any:
        class _Response:
            def __init__(self, text: str) -> None:
                self.content = text

        return _Response(self._content)


class FakeLLMNoContent:
    """Fake LLM whose response has no .content attribute."""

    def invoke(self, prompt: str) -> Any:
        return "raw string response"


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _sample_audit_report() -> dict[str, Any]:
    return {
        "schema_version": "audit_report.v1",
        "run": {
            "run_id": "test_run_001",
            "world_id": "test_world",
            "seed": 42,
            "track": "generic_circuit_01",
            "regulation_id": "synthetic_regulation_v1",
            "config_hash": "abc123",
        },
        "summary": {
            "unsafe_legal_state_count": 3,
            "has_unsafe_legal_state": True,
            "max_hazard_score": 0.82,
            "mean_hazard_score": 0.74,
            "unsafe_legal_segments": ["tight_corner_01"],
            "safety_verdict_status_counts": {"UNSAFE_LEGAL": 3},
        },
        "unsafe_legal_events": [],
        "counterfactuals": [],
        "limitations": [
            "Deterministic stress-test, not a calibrated regulatory recommendation.",
            "State hash coverage is partial.",
        ],
    }


def _sample_families() -> list[dict[str, Any]]:
    return [
        {
            "family_id": "confined_corner_grass",
            "segment_type": "corner",
            "width_m": 11.5,
            "runoff_type": "grass",
            "barrier_distance_m": 8.0,
            "expected_unsafe_legal": True,
        },
        {
            "family_id": "wide_corner_asphalt_control",
            "segment_type": "corner",
            "width_m": 18.5,
            "runoff_type": "asphalt",
            "barrier_distance_m": 50.0,
            "expected_unsafe_legal": False,
        },
    ]


def _sample_counterfactual() -> dict[str, Any]:
    return {
        "patch_id": "closing_speed_cap_v1",
        "patch_type": "regulation_param",
        "verdict": "mitigated",
        "mitigation_success": True,
        "hazard_reduced": True,
        "baseline_metrics": {
            "unsafe_legal_state_count": 2,
            "max_hazard_score": 0.81,
        },
        "patched_metrics": {
            "unsafe_legal_state_count": 0,
            "max_hazard_score": 0.58,
        },
        "delta_metrics": {
            "unsafe_legal_state_count_delta": -2,
            "verdict": "mitigated",
            "mitigation_success": True,
            "hazard_reduced": True,
        },
        "reproducibility": {
            "same_seed": True,
            "same_world_id": True,
        },
    }


# ---------------------------------------------------------------------------
# Task 5: Prompt builder tests
# ---------------------------------------------------------------------------


def test_audit_summary_prompt_contains_caution_instructions() -> None:
    """Prompt must embed cautious language system instructions."""
    report = _sample_audit_report()
    prompt = build_audit_summary_prompt(report)

    assert "cautious" in prompt.lower() or "caution" in prompt.lower()
    assert "limitation" in prompt.lower()
    assert "deterministic" in prompt.lower()


def test_audit_summary_prompt_contains_unsafe_legal_count() -> None:
    """Prompt must surface the unsafe_legal_state_count metric."""
    report = _sample_audit_report()
    prompt = build_audit_summary_prompt(report)
    assert "3" in prompt  # unsafe_legal_state_count=3


def test_audit_summary_prompt_contains_no_overclaim_phrases() -> None:
    """Prompt must instruct LLM not to overclaim."""
    report = _sample_audit_report()
    prompt = build_audit_summary_prompt(report)
    assert "proven safe" in prompt.lower() or "do not" in prompt.lower()


def test_audit_summary_prompt_includes_limitations() -> None:
    """Prompt must include stated simulation limitations."""
    report = _sample_audit_report()
    prompt = build_audit_summary_prompt(report)
    assert "deterministic" in prompt.lower()
    assert "calibrated" in prompt.lower()


def test_synthetic_family_review_prompt_contains_family_ids() -> None:
    """Prompt must list submitted family IDs."""
    families = _sample_families()
    prompt = build_synthetic_family_review_prompt(families)
    assert "confined_corner_grass" in prompt
    assert "wide_corner_asphalt_control" in prompt


def test_synthetic_family_review_prompt_bans_real_track_names() -> None:
    """Prompt must instruct LLM not to reference real track names."""
    families = _sample_families()
    prompt = build_synthetic_family_review_prompt(families)
    assert "suzuka" in prompt.lower() or "real track" in prompt.lower()


def test_synthetic_family_review_prompt_includes_caution() -> None:
    """Prompt must include cautious instructions."""
    families = _sample_families()
    prompt = build_synthetic_family_review_prompt(families)
    assert "cautious" in prompt.lower() or "do not" in prompt.lower()


def test_counterfactual_review_prompt_contains_verdict() -> None:
    """Prompt must include the patch verdict."""
    cf = _sample_counterfactual()
    prompt = build_counterfactual_review_prompt(cf)
    assert "mitigated" in prompt.lower()


def test_counterfactual_review_prompt_contains_delta_metrics() -> None:
    """Prompt must include before/after unsafe_legal_state_count."""
    cf = _sample_counterfactual()
    prompt = build_counterfactual_review_prompt(cf)
    assert "2" in prompt  # baseline count
    assert "0" in prompt  # patched count


def test_counterfactual_review_prompt_contains_limitations() -> None:
    """Prompt must ask LLM to identify limitations."""
    cf = _sample_counterfactual()
    prompt = build_counterfactual_review_prompt(cf)
    assert "limitation" in prompt.lower() or "generalise" in prompt.lower()


def test_counterfactual_review_prompt_bans_overclaim() -> None:
    """Prompt must forbid overclaim phrases."""
    cf = _sample_counterfactual()
    prompt = build_counterfactual_review_prompt(cf)
    assert "proven safe" in prompt or "do not" in prompt.lower()


# ---------------------------------------------------------------------------
# Task 5: summarize function with fake LLM
# ---------------------------------------------------------------------------


def test_summarize_audit_report_with_fake_llm() -> None:
    """summarize_audit_report_with_llm returns LLM content string."""
    llm = FakeLLM("This run shows 3 unsafe legal states under stress conditions.")
    report = _sample_audit_report()
    result = summarize_audit_report_with_llm(report, llm=llm)
    assert isinstance(result, str)
    assert "3 unsafe legal states" in result


def test_review_synthetic_families_with_fake_llm() -> None:
    """review_synthetic_families_with_llm returns string from fake LLM."""
    llm = FakeLLM("Families cover narrow corner and wide control scenarios.")
    families = _sample_families()
    result = review_synthetic_families_with_llm(families, llm=llm)
    assert isinstance(result, str)
    assert len(result) > 0


def test_review_counterfactual_with_fake_llm() -> None:
    """review_counterfactual_with_llm returns string from fake LLM."""
    llm = FakeLLM("The patch successfully mitigated all unsafe legal states.")
    cf = _sample_counterfactual()
    result = review_counterfactual_with_llm(cf, llm=llm)
    assert isinstance(result, str)
    assert "mitigated" in result.lower()


def test_extract_response_content_with_content_attr() -> None:
    """_extract_response_content prefers .content attribute."""

    class _Resp:
        content = "hello content"

    assert _extract_response_content(_Resp()) == "hello content"


def test_extract_response_content_fallback_to_str() -> None:
    """_extract_response_content falls back to str() when .content absent."""
    result = _extract_response_content("plain string")
    assert result == "plain string"


def test_summarize_with_fake_llm_no_content_attr() -> None:
    """summarize works when LLM response has no .content — falls back to str()."""
    llm = FakeLLMNoContent()
    report = _sample_audit_report()
    result = summarize_audit_report_with_llm(report, llm=llm)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Task 5: error path tests — no real API key required
# ---------------------------------------------------------------------------


def test_build_nvidia_llm_raises_on_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError with helpful message when package is not installed."""
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="langchain-nvidia-ai-endpoints"):
        build_nvidia_llm()


def test_build_nvidia_llm_raises_on_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError with clear message when NVIDIA_API_KEY is absent."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    # If the package is not installed, we get a different error — skip in that case.
    # We only care about the key-missing error path.
    try:
        import langchain_nvidia_ai_endpoints  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("langchain-nvidia-ai-endpoints not installed; skipping key-missing test")

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        build_nvidia_llm()


def test_build_nvidia_llm_error_message_does_not_expose_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error message must not include the actual API key value."""
    fake_key = "nvapi-SUPER-SECRET-12345"
    monkeypatch.setenv("NVIDIA_API_KEY", fake_key)
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError) as exc_info:
        build_nvidia_llm()
    assert fake_key not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration: prompt + fake LLM round-trip for all three functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_id",
    ["confined_corner_grass", "fast_corner_wall", "narrow_street_chicane"],
)
def test_synthetic_family_prompt_round_trip(family_id: str) -> None:
    """Prompt build + fake LLM invoke round-trip for each positive family."""
    import dataclasses

    from reglabsim.synthetic.families import SYNTHETIC_FAMILIES

    spec = SYNTHETIC_FAMILIES[family_id]
    family_dict = dataclasses.asdict(spec)
    llm = FakeLLM(f"Analysis of {family_id}: suggests elevated hazard in confined geometry.")
    result = review_synthetic_families_with_llm([family_dict], llm=llm)
    assert family_id in result


def test_full_pipeline_audit_to_llm_summary() -> None:
    """End-to-end: run synthetic family → build evidence bundle → audit report → LLM summary."""
    from reglabsim.logging.audit_report import build_audit_report
    from reglabsim.logging.replay import ReplayEngine
    from reglabsim.synthetic.families import (
        build_synthetic_family_run_output,
        run_synthetic_family_microkernel,
    )

    result = run_synthetic_family_microkernel("confined_corner_grass", seed=42)
    run_output = build_synthetic_family_run_output(result)
    bundle = ReplayEngine().build_evidence_bundle(run_output)
    report = build_audit_report(bundle)

    llm = FakeLLM("In this deterministic stress-test, 1 unsafe legal state was detected.")
    summary = summarize_audit_report_with_llm(report, llm=llm)

    assert isinstance(summary, str)
    assert "unsafe legal state" in summary.lower()
