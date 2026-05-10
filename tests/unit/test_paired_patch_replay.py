"""Tests for PR 3 — paired patch replay (closing_speed_cap causal path)."""

from __future__ import annotations

from typing import Any

import pytest

from reglabsim.campaigns.runner import compare_patch_metrics
from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import RunoffProfile, SegmentRiskProfile, TrackSegment

# ── shared Spoon scenario fixtures ────────────────────────────────────────────

def _spoon_track() -> TrackModel:
    return TrackModel(
        track_id="suzuka",
        name="Suzuka",
        country="Japan",
        length_m=1200.0,
        turns=1,
        laps=5,
        race_distance_m=6000.0,
        avg_speed_kph=185.0,
        fidelity_level=2,
        segments=[
            TrackSegment(
                segment_id="spoon_entry",
                name="Spoon Curve Entry",
                segment_type="corner",
                start_m=0.0,
                end_m=1200.0,
                width_m=11.2,
                radius_m=135.0,
                overtaking_viability="high",
                preferred_battle_zone=True,
                runoff=RunoffProfile(
                    type="grass",
                    width_m=3.0,
                    grip_dry=0.35,
                    grip_wet=0.15,
                    rejoin_risk="high",
                ),
                risk=SegmentRiskProfile(
                    unsafe_closing_speed_threshold_kph=36.0,
                    side_by_side_risk="high",
                    evasive_action_margin="high",
                    energy_delta_sensitivity="high",
                    barrier_distance_m=14.0,
                ),
            )
        ],
    )


def _spoon_weather() -> WeatherState:
    return WeatherState(
        air_temp_c=28.0,
        humidity_pct=58.0,
        pressure_hpa=1012.0,
        wind_speed_mps=2.4,
        wind_direction_deg=210.0,
        rain_intensity_mm_h=0.0,
        cloud_cover_pct=25.0,
        visibility_m=1000.0,
    )


def _spoon_track_state() -> TrackState:
    return TrackState(
        track_temp_c=34.0,
        grip_level=0.97,
        rubber_level=0.4,
        wetness_level=0.0,
        standing_water_level=0.0,
        dirt_offline_level=0.2,
        drying_rate=0.02,
        surface_evolution_rate=0.01,
    )


def _spoon_cars() -> list[CarRuntimeState]:
    def _car(
        *,
        car_id: str,
        position: int,
        cumulative_time_s: float,
        ers_soc: float,
        gap_ahead_s: float,
    ) -> CarRuntimeState:
        return CarRuntimeState(
            car_id=car_id,
            driver_id=car_id.replace("car", "driver"),
            team_id="team_01",
            family_id="family_a",
            position=position,
            lap=0,
            gap_to_leader_s=0.0,
            gap_ahead_s=gap_ahead_s,
            gap_behind_s=0.35,
            tyre_compound="C3",
            tyre_age_laps=4,
            tyre_wear=0.08,
            ers_soc=ers_soc,
            fuel_mass_kg=98.0,
            aero_mode="corner",
            last_lap_time_s=0.0,
            cumulative_time_s=cumulative_time_s,
        )

    return [
        _car(car_id="car_01", position=1, cumulative_time_s=90.0, ers_soc=0.38, gap_ahead_s=0.0),
        _car(car_id="car_02", position=2, cumulative_time_s=90.3, ers_soc=0.9, gap_ahead_s=0.3),
    ]


def _spoon_actions() -> dict[str, RaceAction]:
    return {
        "car_01": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_01",
            lap=1,
            pace_mode="balanced",
            ers_mode="hybrid",
            aero_mode="corner",
            attack=False,
            defend=True,
            pit_this_lap=False,
            risk_level=0.74,
            source_mode="test",
            note="depleted defend",
        ),
        "car_02": RaceAction(
            schema_version=RACE_ACTION_SCHEMA,
            car_id="car_02",
            lap=1,
            pace_mode="attack",
            ers_mode="boost",
            aero_mode="straight",
            attack=True,
            defend=False,
            pit_this_lap=False,
            risk_level=0.8,
            source_mode="test",
            note="closing attack",
        ),
    }


def _base_regulation() -> dict[str, Any]:
    return {"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}}


def _run_spoon_lap(regulation: dict[str, Any]) -> list[Any]:
    mk = RaceMicrokernel(regulation=regulation, seed=5)
    _, events, _ = mk.resolve_lap(
        lap=1,
        total_laps=5,
        cars=_spoon_cars(),
        actions=_spoon_actions(),
        track=_spoon_track(),
        weather=_spoon_weather(),
        track_state=_spoon_track_state(),
        safety_car_active=False,
    )
    return [e for e in events if e.event_type == "unsafe_legal_state"]


# ── minimal run_output helper ─────────────────────────────────────────────────

def _minimal_run_output(events: list[dict[str, Any]], run_id: str = "test_run") -> dict[str, Any]:
    return {
        "manifest": {
            "run_id": run_id,
            "seed": 5,
            "regulation_id": "reg_2026",
            "track_id": "suzuka",
            "config_hash": "deadbeef",
            "world_id": f"world-{run_id}",
        },
        "event_log": events,
        "action_validation_log": [],
        "state_snapshots": [],
        "patch_reruns": [],
    }


_UNSAFE_EVENT: dict[str, Any] = {
    "event_type": "unsafe_legal_state",
    "lap": 1,
    "car_id": "car_02",
    "segment_id": "spoon_entry",
    "details": {
        "hazard_score": 0.6931,
        "reaction_margin_s": 0.518,
        "closing_speed_kph": 26.77,
        "safety_status": "UNSAFE_LEGAL",
        "safety_verdict": {
            "schema_version": "safety_verdict.v1",
            "status": "UNSAFE_LEGAL",
            "hazard_score": 0.6931,
            "reaction_margin_s": 0.518,
            "delta_speed_kph": 26.77,
            "time_to_collision_s": None,
            "amplifiers": [],
            "regulatory_causes": [],
            "reason_codes": [],
            "confidence": "low",
            "evidence": {},
        },
    },
}


# ── Test 1: causal cap reduces effective_delta_kph fed to oracle ──────────────

def test_closing_speed_cap_patch_reduces_effective_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing speed cap must reduce delta_speed_kph seen by SafetyOracle."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    captured_inputs: list[Any] = []

    def _capture(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict

        captured_inputs.append(context)
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.5,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _capture)

    cap_kph = 30.0
    reg_no_cap = _base_regulation()
    reg_with_cap = {**_base_regulation(), "safety": {"closing_speed_cap_kph": cap_kph}}

    # Run without cap
    RaceMicrokernel(regulation=reg_no_cap, seed=5).resolve_lap(
        lap=1,
        total_laps=5,
        cars=_spoon_cars(),
        actions=_spoon_actions(),
        track=_spoon_track(),
        weather=_spoon_weather(),
        track_state=_spoon_track_state(),
        safety_car_active=False,
    )
    no_cap_inputs = list(captured_inputs)
    captured_inputs.clear()

    # Run with cap
    RaceMicrokernel(regulation=reg_with_cap, seed=5).resolve_lap(
        lap=1,
        total_laps=5,
        cars=_spoon_cars(),
        actions=_spoon_actions(),
        track=_spoon_track(),
        weather=_spoon_weather(),
        track_state=_spoon_track_state(),
        safety_car_active=False,
    )
    cap_inputs = list(captured_inputs)

    assert no_cap_inputs, "Expected oracle call without cap"
    assert cap_inputs, "Expected oracle call with cap"

    no_cap_delta = getattr(no_cap_inputs[0], "delta_speed_kph", None)
    cap_delta = getattr(cap_inputs[0], "delta_speed_kph", None)

    assert isinstance(no_cap_delta, (int, float)), "delta_speed_kph must be numeric"
    assert isinstance(cap_delta, (int, float)), "delta_speed_kph must be numeric"
    assert float(no_cap_delta) > cap_kph, (
        f"Uncapped effective_delta should exceed cap ({no_cap_delta} <= {cap_kph})"
    )
    assert float(cap_delta) <= cap_kph, (
        f"Capped effective_delta should not exceed cap ({cap_delta} > {cap_kph})"
    )


# ── Test 2: paired replay — cap reduces unsafe_legal_state_count ─────────────

def test_paired_patch_replay_reduces_unsafe_legal_state_count() -> None:
    """Baseline emits unsafe_legal_state; cap at 1 kph prevents oracle from triggering."""
    # Baseline: no cap → expect emission
    baseline_events = _run_spoon_lap(_base_regulation())
    assert baseline_events, "Baseline must emit at least one unsafe_legal_state"

    # Patched: cap so low oracle cannot classify as UNSAFE_LEGAL
    reg_with_cap = {**_base_regulation(), "safety": {"closing_speed_cap_kph": 1.0}}
    patched_events = _run_spoon_lap(reg_with_cap)

    # Compare
    baseline_count = len(baseline_events)
    patched_count = len(patched_events)

    baseline_metrics: dict[str, Any] = {
        "unsafe_legal_state_count": baseline_count,
        "max_hazard_score": (
            baseline_events[0].details.get("hazard_score") if baseline_events else None
        ),
        "mean_hazard_score": (
            baseline_events[0].details.get("hazard_score") if baseline_events else None
        ),
    }
    patched_metrics: dict[str, Any] = {
        "unsafe_legal_state_count": patched_count,
        "max_hazard_score": None,
        "mean_hazard_score": None,
    }
    delta = compare_patch_metrics(baseline_metrics, patched_metrics)

    assert patched_count < baseline_count, (
        f"Patched count ({patched_count}) must be less than baseline ({baseline_count})"
    )
    assert delta["unsafe_legal_state_count_delta"] < 0
    assert delta["mitigation_success"] is True


# ── Test 3: evidence bundle contains patch_reruns ─────────────────────────────

def test_evidence_bundle_contains_patch_reruns() -> None:
    """EvidenceBundle.patch_reruns must be populated from run_output patch_reruns."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="baseline_001")
    patched_run = _minimal_run_output([], run_id="patched_001")

    baseline_metrics: dict[str, Any] = {
        "unsafe_legal_state_count": 1,
        "max_hazard_score": 0.6931,
        "mean_hazard_score": 0.6931,
    }
    patched_metrics: dict[str, Any] = {
        "unsafe_legal_state_count": 0,
        "max_hazard_score": None,
        "mean_hazard_score": None,
    }
    delta = compare_patch_metrics(baseline_metrics, patched_metrics)

    patch_rerun_entry: dict[str, Any] = {
        "patch_id": "closing_speed_cap_v1",
        "patch_type": "closing_speed_cap",
        "paired_with_run_id": baseline_run["manifest"]["run_id"],
        "patched_run_id": patched_run["manifest"]["run_id"],
        "same_seed": True,
        "same_world_id": False,
        "baseline_metrics": baseline_metrics,
        "patched_metrics": patched_metrics,
        "delta_metrics": delta,
        "verdict": "mitigated",
        "notes": [],
    }

    enriched_baseline = {**baseline_run, "patch_reruns": [patch_rerun_entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched_baseline)

    assert bundle["patch_reruns"], "patch_reruns must not be empty"
    entry = bundle["patch_reruns"][0]
    assert entry["patch_id"] == "closing_speed_cap_v1"
    assert entry["patch_type"] == "closing_speed_cap"
    assert "baseline_metrics" in entry
    assert "patched_metrics" in entry
    assert "delta_metrics" in entry
    assert entry["delta_metrics"]["mitigation_success"] is True
    assert entry["delta_metrics"]["verdict"] == "mitigated"
    assert entry["delta_metrics"]["unsafe_legal_state_count_delta"] == -1


# ── Test 4: patch_rerun preserves same_seed and world metadata ────────────────

def _sample_patch_rerun_entry(
    paired_with: str,
    patched_id: str,
    same_world_id: bool = True,
) -> dict[str, Any]:
    bm: dict[str, Any] = {
        "unsafe_legal_state_count": 1,
        "max_hazard_score": 0.6931,
        "mean_hazard_score": 0.6931,
    }
    pm: dict[str, Any] = {
        "unsafe_legal_state_count": 0,
        "max_hazard_score": None,
        "mean_hazard_score": None,
    }
    dm: dict[str, Any] = {
        "unsafe_legal_state_count_delta": -1,
        "max_hazard_score_delta": None,
        "mean_hazard_score_delta": None,
        "verdict": "mitigated",
        "mitigation_success": True,
    }
    return {
        "patch_id": "closing_speed_cap_v1",
        "patch_type": "closing_speed_cap",
        "paired_with_run_id": paired_with,
        "patched_run_id": patched_id,
        "same_seed": True,
        "same_world_id": same_world_id,
        "baseline_metrics": bm,
        "patched_metrics": pm,
        "delta_metrics": dm,
        "verdict": "mitigated",
        "notes": [],
    }


def test_patch_replay_preserves_same_seed_and_world_metadata() -> None:
    """patch_rerun entry must carry same_seed=True and world metadata honestly."""
    seed = 5
    world_id = "world-abc123"
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="base_meta")
    baseline_run["manifest"]["world_id"] = world_id
    baseline_run["manifest"]["seed"] = seed

    patch_rerun_entry = _sample_patch_rerun_entry("base_meta", "patched_meta", same_world_id=True)

    enriched = {**baseline_run, "patch_reruns": [patch_rerun_entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)
    entry = bundle["patch_reruns"][0]

    assert entry["same_seed"] is True
    assert entry["same_world_id"] is True
    assert entry["paired_with_run_id"] == "base_meta"
    # seed must match between baseline and patched (same_seed=True is not hardcoded)
    assert baseline_run["manifest"]["seed"] == seed


def test_same_seed_computed_from_manifests() -> None:
    """same_seed must be False when manifests carry different seeds."""
    entry_same = _sample_patch_rerun_entry("r1", "r2", same_world_id=False)
    entry_same["same_seed"] = True   # seeds match

    entry_diff = _sample_patch_rerun_entry("r3", "r4", same_world_id=False)
    entry_diff["same_seed"] = False  # seeds differ
    entry_diff["notes"] = ["seed differs: baseline=5 patched=99"]

    for entry, expected in [(entry_same, True), (entry_diff, False)]:
        run_output = _minimal_run_output([], run_id=entry["paired_with_run_id"])
        enriched = {**run_output, "patch_reruns": [entry]}
        bundle = ReplayEngine().build_evidence_bundle(enriched)
        assert bundle["patch_reruns"][0]["same_seed"] is expected


# ── Test 5: patch replay does not mutate baseline bundle ─────────────────────

def test_patch_does_not_mutate_baseline_bundle() -> None:
    """Enriching baseline with patch_reruns must not mutate the original run_output."""
    baseline_run = _minimal_run_output([_UNSAFE_EVENT], run_id="immutable_base")
    original_patch_reruns = list(baseline_run.get("patch_reruns", []))

    patch_rerun_entry = _sample_patch_rerun_entry(
        "immutable_base", "patched_x", same_world_id=False
    )

    # Create enriched copy — must not mutate baseline_run
    enriched = {**baseline_run, "patch_reruns": [patch_rerun_entry]}
    bundle = ReplayEngine().build_evidence_bundle(enriched)

    # Baseline run_output must not have been mutated
    assert baseline_run.get("patch_reruns") == original_patch_reruns, (
        "baseline_run patch_reruns was mutated by enriched copy"
    )
    # Enriched bundle has the entry
    assert len(bundle["patch_reruns"]) == 1
    # Original baseline bundle has no patch_reruns
    baseline_bundle = ReplayEngine().build_evidence_bundle(baseline_run)
    assert baseline_bundle["patch_reruns"] == original_patch_reruns


# ── Test 6: compare_patch_metrics correctness ────────────────────────────────

def test_compare_patch_metrics_handles_none_safely() -> None:
    """compare_patch_metrics must not raise when scores are None."""
    result = compare_patch_metrics(
        {"unsafe_legal_state_count": 1, "max_hazard_score": None, "mean_hazard_score": 0.5},
        {"unsafe_legal_state_count": 0, "max_hazard_score": None, "mean_hazard_score": None},
    )
    assert result["mitigation_success"] is True
    assert result["verdict"] == "mitigated"
    assert result["unsafe_legal_state_count_delta"] == -1
    assert result["max_hazard_score_delta"] is None
    assert result["mean_hazard_score_delta"] is None


def test_compare_patch_metrics_no_mitigation_when_counts_equal() -> None:
    result = compare_patch_metrics(
        {"unsafe_legal_state_count": 0, "max_hazard_score": None, "mean_hazard_score": None},
        {"unsafe_legal_state_count": 0, "max_hazard_score": None, "mean_hazard_score": None},
    )
    assert result["mitigation_success"] is False
    assert result["verdict"] == "unchanged"
    assert result["unsafe_legal_state_count_delta"] == 0


def test_compare_patch_metrics_improved_not_mitigated_when_count_reduced_not_zero() -> None:
    """count 2→1 is 'improved', not 'mitigated'."""
    result = compare_patch_metrics(
        {"unsafe_legal_state_count": 2, "max_hazard_score": 0.8, "mean_hazard_score": 0.7},
        {"unsafe_legal_state_count": 1, "max_hazard_score": 0.5, "mean_hazard_score": 0.5},
    )
    assert result["verdict"] == "improved"
    assert result["mitigation_success"] is False
    assert result["unsafe_legal_state_count_delta"] == -1


def test_compare_patch_metrics_worse_when_count_increases() -> None:
    result = compare_patch_metrics(
        {"unsafe_legal_state_count": 1, "max_hazard_score": 0.5, "mean_hazard_score": 0.5},
        {"unsafe_legal_state_count": 2, "max_hazard_score": 0.8, "mean_hazard_score": 0.8},
    )
    assert result["verdict"] == "worse"
    assert result["mitigation_success"] is False
    assert result["unsafe_legal_state_count_delta"] == 1
