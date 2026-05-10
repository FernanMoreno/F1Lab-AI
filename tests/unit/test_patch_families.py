"""Tests for PR 7 — More patch families and catalog comparison."""

from __future__ import annotations

from typing import Any

import pytest

from reglabsim.campaigns.runner import CampaignRunner, rank_patch_results
from reglabsim.conditions.scenarios import TrackState, WeatherState
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.schema import RACE_ACTION_SCHEMA, CarRuntimeState, RaceAction
from reglabsim.synthetic.families import (
    SYNTHETIC_FAMILIES,
    build_synthetic_actions_for_battle,
    build_synthetic_cars_for_battle,
    build_synthetic_track,
    build_synthetic_track_state,
    build_synthetic_weather,
)
from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import RunoffProfile, SegmentRiskProfile, TrackSegment

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EXPECTED_PATCH_IDS = [
    "closing_speed_cap_v1",
    "closing_speed_cap_v2",
    "minimum_reaction_margin_v1",
    "confined_corner_attack_restriction_v1",
    "active_aero_delay_high_risk_v1",
    "pack_compression_overtake_limit_v1",
]

_VERSIONED_PATCH_IDS = [
    "closing_speed_cap_v1",
    "closing_speed_cap_v2",
    "minimum_reaction_margin_v1",
    "confined_corner_attack_restriction_v1",
    "active_aero_delay_high_risk_v1",
    "pack_compression_overtake_limit_v1",
]


def _minimal_runner() -> CampaignRunner:
    return CampaignRunner(regulations={}, car_families={})


def _confined_track(track_id: str = "generic_circuit_01") -> TrackModel:
    """Narrow corner, grass runoff — triggers confined_corner_attack_restriction."""
    return TrackModel(
        track_id=track_id,
        name="Test Confined Corner",
        country="Synthetic",
        length_m=1200.0,
        turns=1,
        laps=5,
        race_distance_m=6000.0,
        avg_speed_kph=185.0,
        fidelity_level=1,
        segments=[
            TrackSegment(
                segment_id="tight_corner_test",
                name="Tight Corner",
                segment_type="corner",
                start_m=0.0,
                end_m=1200.0,
                width_m=11.0,
                radius_m=120.0,
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
                    barrier_distance_m=8.0,
                ),
            )
        ],
    )


def _standard_weather() -> WeatherState:
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


def _standard_track_state() -> TrackState:
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


def _battle_cars() -> list[CarRuntimeState]:
    return [
        CarRuntimeState(
            car_id="car_01",
            driver_id="driver_01",
            team_id="team_01",
            family_id="family_a",
            position=1,
            lap=0,
            gap_to_leader_s=0.0,
            gap_ahead_s=0.0,
            gap_behind_s=0.35,
            tyre_compound="C3",
            tyre_age_laps=4,
            tyre_wear=0.08,
            ers_soc=0.38,
            fuel_mass_kg=98.0,
            aero_mode="corner",
            last_lap_time_s=0.0,
            cumulative_time_s=90.0,
        ),
        CarRuntimeState(
            car_id="car_02",
            driver_id="driver_02",
            team_id="team_02",
            family_id="family_b",
            position=2,
            lap=0,
            gap_to_leader_s=0.0,
            gap_ahead_s=0.3,
            gap_behind_s=0.35,
            tyre_compound="C3",
            tyre_age_laps=4,
            tyre_wear=0.08,
            ers_soc=0.9,
            fuel_mass_kg=98.0,
            aero_mode="corner",
            last_lap_time_s=0.0,
            cumulative_time_s=90.3,
        ),
    ]


def _battle_actions() -> dict[str, RaceAction]:
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
            note="defender",
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
            note="attacker",
        ),
    }


def _base_regulation() -> dict[str, Any]:
    return {"power_unit": {"ers_max_energy_mj": 6.0, "ers_deployment_max_kw": 250.0}}


def _run_lap_events(
    regulation: dict[str, Any],
    track: TrackModel | None = None,
    seed: int = 5,
) -> list[Any]:
    mk = RaceMicrokernel(regulation=regulation, seed=seed)
    t = track if track is not None else _confined_track()
    _, events, _ = mk.resolve_lap(
        lap=1,
        total_laps=5,
        cars=_battle_cars(),
        actions=_battle_actions(),
        track=t,
        weather=_standard_weather(),
        track_state=_standard_track_state(),
        safety_car_active=False,
    )
    return [e for e in events if e.event_type == "unsafe_legal_state"]


# ---------------------------------------------------------------------------
# Task 5 — Test 1: catalog resolves expected patch families
# ---------------------------------------------------------------------------


def test_patch_catalog_contains_expected_patch_families() -> None:
    runner = _minimal_runner()
    for patch_id in _EXPECTED_PATCH_IDS:
        patch = runner._resolve_patch_candidate(patch_id)
        assert "name" in patch, f"{patch_id}: missing name"
        assert "patch_type" in patch, f"{patch_id}: missing patch_type"
        assert "regulation_overrides" in patch, f"{patch_id}: missing regulation_overrides"
        assert "expected_tradeoffs" in patch, f"{patch_id}: missing expected_tradeoffs"
        assert patch["name"] == patch_id, (
            f"{patch_id}: name mismatch — got {patch['name']!r}"
        )


# ---------------------------------------------------------------------------
# Task 5 — Test 2: patch_type is generic, not the versioned patch id
# ---------------------------------------------------------------------------


def test_patch_types_are_generic_not_version_ids() -> None:
    runner = _minimal_runner()
    for patch_id in _VERSIONED_PATCH_IDS:
        patch = runner._resolve_patch_candidate(patch_id)
        pt = patch.get("patch_type", "")
        assert pt != patch_id, (
            f"{patch_id}: patch_type should be generic, not the same as name. Got {pt!r}"
        )
        assert pt, f"{patch_id}: patch_type is empty"

    # Specific known cases
    v1 = runner._resolve_patch_candidate("closing_speed_cap_v1")
    v2 = runner._resolve_patch_candidate("closing_speed_cap_v2")
    assert v1["patch_type"] == "closing_speed_cap"
    assert v2["patch_type"] == "closing_speed_cap"
    assert v1["patch_type"] == v2["patch_type"], "v1 and v2 should share the same patch_type"


# ---------------------------------------------------------------------------
# Task 5 — Test 3: patch effects modify SafetyOracleInput before evaluate
# ---------------------------------------------------------------------------


def test_patch_effects_modify_safety_oracle_input_before_evaluate_cap_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """closing_speed_cap_v2 must reduce delta_speed_kph fed to SafetyOracle."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    captured: list[Any] = []

    def _capture(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        captured.append(context)
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.5,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _capture)

    cap_kph = 55.0
    reg_no_cap = _base_regulation()
    reg_v2 = {**_base_regulation(), "safety": {"closing_speed_cap_kph": cap_kph}}

    RaceMicrokernel(regulation=reg_no_cap, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    no_cap_inputs = list(captured)
    captured.clear()

    RaceMicrokernel(regulation=reg_v2, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    cap_inputs = list(captured)

    assert no_cap_inputs, "Expected oracle call without cap"
    assert cap_inputs, "Expected oracle call with cap"

    no_cap_delta = getattr(no_cap_inputs[0], "delta_speed_kph", None)
    cap_delta = getattr(cap_inputs[0], "delta_speed_kph", None)

    assert isinstance(no_cap_delta, (int, float))
    assert isinstance(cap_delta, (int, float))
    assert float(no_cap_delta) > cap_kph, (
        f"Uncapped delta ({no_cap_delta}) should exceed v2 cap ({cap_kph})"
    )
    assert float(cap_delta) <= cap_kph, (
        f"Capped delta ({cap_delta}) must not exceed v2 cap ({cap_kph})"
    )


def test_patch_effects_modify_safety_oracle_input_confined_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """confined_corner_attack_restriction must reduce delta_speed_kph for confined segments."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    captured: list[Any] = []

    def _capture(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        captured.append(context)
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.5,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _capture)

    reg_no_patch = _base_regulation()
    reg_confined = {
        **_base_regulation(),
        "safety": {
            "confined_corner_attack_restriction": True,
            "confined_corner_width_m": 12.5,
            "confined_corner_runoff_types": ["grass", "gravel", "wall", "barrier", "concrete"],
        },
    }

    RaceMicrokernel(regulation=reg_no_patch, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    baseline_inputs = list(captured)
    captured.clear()

    RaceMicrokernel(regulation=reg_confined, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    patched_inputs = list(captured)

    assert baseline_inputs, "Expected oracle call without patch"
    assert patched_inputs, "Expected oracle call with confined patch"

    baseline_delta = float(getattr(baseline_inputs[0], "delta_speed_kph", 0))
    patched_delta = float(getattr(patched_inputs[0], "delta_speed_kph", 0))

    assert patched_delta < baseline_delta, (
        f"Confined patch must reduce delta: baseline={baseline_delta} patched={patched_delta}"
    )


def test_patch_effects_modify_safety_oracle_input_active_aero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """active_aero_delay_high_risk must reduce delta_speed_kph when geometry_risk is high."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    captured: list[Any] = []

    def _capture(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        captured.append(context)
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.5,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _capture)

    reg_no_patch = _base_regulation()
    # Use low threshold so geometry_risk (driven by narrow+high-risk segment) triggers it
    reg_aero = {
        **_base_regulation(),
        "safety": {
            "active_aero_delay_high_risk": True,
            "active_aero_delay_risk_threshold": 0.1,
        },
    }

    RaceMicrokernel(regulation=reg_no_patch, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    baseline_inputs = list(captured)
    captured.clear()

    RaceMicrokernel(regulation=reg_aero, seed=5).resolve_lap(
        lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
        track=_confined_track(), weather=_standard_weather(),
        track_state=_standard_track_state(), safety_car_active=False,
    )
    patched_inputs = list(captured)

    assert baseline_inputs, "Expected oracle call without patch"
    assert patched_inputs, "Expected oracle call with active_aero patch"

    baseline_delta = float(getattr(baseline_inputs[0], "delta_speed_kph", 0))
    patched_delta = float(getattr(patched_inputs[0], "delta_speed_kph", 0))

    assert patched_delta < baseline_delta, (
        f"Active aero patch must reduce delta: baseline={baseline_delta} patched={patched_delta}"
    )


# ---------------------------------------------------------------------------
# Task 5 — Test 4: patch effects do not depend on track_id
# ---------------------------------------------------------------------------


def test_patch_effects_do_not_depend_on_track_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch causal path must work identically across different synthetic track IDs."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    deltas_by_track: dict[str, list[float]] = {}

    def _capture_factory(track_label: str) -> Any:
        def _capture(self: object, context: object) -> object:
            from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
            deltas_by_track.setdefault(track_label, []).append(
                float(getattr(context, "delta_speed_kph", 0))
            )
            return SafetyVerdict(
                schema_version="safety_verdict.v1",
                status=SafetyStatus.UNSAFE_LEGAL,
                hazard_score=0.5,
                delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
                confidence="high",
            )
        return _capture

    reg = {
        **_base_regulation(),
        "safety": {
            "confined_corner_attack_restriction": True,
            "confined_corner_width_m": 12.5,
            "confined_corner_runoff_types": ["grass", "gravel", "wall", "barrier", "concrete"],
        },
    }

    synthetic_ids = ["generic_circuit_01", "generic_circuit_02", "generic_circuit_03"]

    for tid in synthetic_ids:
        monkeypatch.setattr(SafetyOracle, "evaluate", _capture_factory(tid))
        RaceMicrokernel(regulation=reg, seed=5).resolve_lap(
            lap=1, total_laps=5, cars=_battle_cars(), actions=_battle_actions(),
            track=_confined_track(track_id=tid), weather=_standard_weather(),
            track_state=_standard_track_state(), safety_car_active=False,
        )

    # Each track should produce the same patched delta (same segment geometry, same regulation)
    all_deltas = [deltas_by_track.get(tid, [None])[0] for tid in synthetic_ids]
    non_none = [d for d in all_deltas if d is not None]
    if len(non_none) >= 2:
        track_delta_map = dict(zip(synthetic_ids, all_deltas, strict=False))
        assert all(abs(d - non_none[0]) < 0.01 for d in non_none), (
            f"Patch must not vary by track_id. Deltas: {track_delta_map}"
        )


# ---------------------------------------------------------------------------
# Task 5 — Test 5: rank_patch_results produces correct order
# ---------------------------------------------------------------------------


def test_compare_patch_catalog_returns_ranked_results() -> None:
    """rank_patch_results must order: mitigated > improved > improved_hazard > unchanged > worse."""
    inputs: list[dict[str, Any]] = [
        {
            "patch_id": "patch_worse",
            "verdict": "worse",
            "unsafe_legal_state_count_delta": 2,
            "max_hazard_score_delta": 0.1,
            "mean_hazard_score_delta": 0.1,
        },
        {
            "patch_id": "patch_unchanged",
            "verdict": "unchanged",
            "unsafe_legal_state_count_delta": 0,
            "max_hazard_score_delta": None,
            "mean_hazard_score_delta": None,
        },
        {
            "patch_id": "patch_mitigated",
            "verdict": "mitigated",
            "unsafe_legal_state_count_delta": -3,
            "max_hazard_score_delta": None,
            "mean_hazard_score_delta": None,
        },
        {
            "patch_id": "patch_improved_hazard",
            "verdict": "improved_hazard",
            "unsafe_legal_state_count_delta": 0,
            "max_hazard_score_delta": -0.1,
            "mean_hazard_score_delta": -0.05,
        },
        {
            "patch_id": "patch_improved",
            "verdict": "improved",
            "unsafe_legal_state_count_delta": -1,
            "max_hazard_score_delta": -0.05,
            "mean_hazard_score_delta": -0.02,
        },
    ]

    ranked = rank_patch_results(inputs)
    ids = [r["patch_id"] for r in ranked]

    assert ids[0] == "patch_mitigated"
    assert ids[1] == "patch_improved"
    assert ids[2] == "patch_improved_hazard"
    assert ids[3] == "patch_unchanged"
    assert ids[4] == "patch_worse"


def test_rank_patch_results_within_same_verdict_lower_delta_first() -> None:
    """Within same verdict, lower count_delta (more negative) ranks first."""
    inputs: list[dict[str, Any]] = [
        {"patch_id": "p_count_minus1", "verdict": "improved",
         "unsafe_legal_state_count_delta": -1, "max_hazard_score_delta": 0.0,
         "mean_hazard_score_delta": 0.0},
        {"patch_id": "p_count_minus3", "verdict": "improved",
         "unsafe_legal_state_count_delta": -3, "max_hazard_score_delta": 0.0,
         "mean_hazard_score_delta": 0.0},
    ]
    ranked = rank_patch_results(inputs)
    assert ranked[0]["patch_id"] == "p_count_minus3"
    assert ranked[1]["patch_id"] == "p_count_minus1"


# ---------------------------------------------------------------------------
# Task 5 — Test 6: patch rerun event contains patch effect metadata
# ---------------------------------------------------------------------------


def test_patch_rerun_contains_patch_effect_metadata_when_event_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Events emitted under a patch must include regulatory_patch_effects and delta fields."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    # Force oracle to always emit UNSAFE_LEGAL so we get the event with metadata
    def _always_unsafe(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.7,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _always_unsafe)

    # Use a cap that will definitely trigger (cap below raw delta)
    reg = {**_base_regulation(), "safety": {"closing_speed_cap_kph": 55.0}}
    events = _run_lap_events(reg, track=_confined_track(), seed=5)

    assert events, "Expected at least one unsafe_legal_state event under forced oracle"
    event = events[0]
    details = event.to_dict().get("details", {})

    assert "regulatory_patch_effects" in details, "Missing regulatory_patch_effects"
    assert "effective_delta_kph_before_patch" in details, "Missing effective_delta_kph_before_patch"
    assert "effective_delta_kph_after_patch" in details, "Missing effective_delta_kph_after_patch"

    before = details["effective_delta_kph_before_patch"]
    after = details["effective_delta_kph_after_patch"]
    effects = details["regulatory_patch_effects"]

    assert isinstance(effects, list)
    # Cap was applied — before should be > after
    assert before >= after, f"before_patch ({before}) must be >= after_patch ({after})"
    if before > 55.0:
        assert "closing_speed_cap_applied" in effects


def test_patch_rerun_no_patch_effects_when_no_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no safety patches, regulatory_patch_effects must be empty."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    def _always_unsafe(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.7,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _always_unsafe)

    events = _run_lap_events(_base_regulation(), track=_confined_track(), seed=5)

    assert events, "Expected at least one unsafe_legal_state event"
    event = events[0]
    details = event.to_dict().get("details", {})

    assert "regulatory_patch_effects" in details
    assert details["regulatory_patch_effects"] == [], (
        f"No patches active — effects must be empty, got {details['regulatory_patch_effects']}"
    )
    # before == after when no patch changes anything
    before = details.get("effective_delta_kph_before_patch")
    after = details.get("effective_delta_kph_after_patch")
    assert before == after, f"No patch: before ({before}) must equal after ({after})"


# ---------------------------------------------------------------------------
# Task 5 — Test 7: baseline not mutated across multiple patches
# ---------------------------------------------------------------------------


def test_patch_catalog_comparison_does_not_mutate_baseline() -> None:
    """rank_patch_results on different subsets must not mutate the input list."""
    baseline: list[dict[str, Any]] = [
        {"patch_id": "p1", "verdict": "mitigated", "unsafe_legal_state_count_delta": -1,
         "max_hazard_score_delta": None, "mean_hazard_score_delta": None},
        {"patch_id": "p2", "verdict": "worse", "unsafe_legal_state_count_delta": 1,
         "max_hazard_score_delta": 0.1, "mean_hazard_score_delta": 0.1},
    ]
    original_ids = [r["patch_id"] for r in baseline]

    ranked = rank_patch_results(baseline)

    # baseline list itself must not be reordered
    assert [r["patch_id"] for r in baseline] == original_ids, (
        "rank_patch_results must not mutate the input list"
    )
    # ranked must be sorted
    assert ranked[0]["patch_id"] == "p1"
    assert ranked[1]["patch_id"] == "p2"


# ---------------------------------------------------------------------------
# Bonus: synthetic family smoke test — new patches work with existing families
# ---------------------------------------------------------------------------


def test_new_patches_work_with_synthetic_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """All new patches must not crash on synthetic families."""
    from reglabsim.safety.safety_oracle import SafetyOracle

    def _pass_through(self: object, context: object) -> object:
        from reglabsim.runtime.schema import SafetyStatus, SafetyVerdict
        return SafetyVerdict(
            schema_version="safety_verdict.v1",
            status=SafetyStatus.UNSAFE_LEGAL,
            hazard_score=0.5,
            delta_speed_kph=getattr(context, "delta_speed_kph", 0.0),
            confidence="high",
        )

    monkeypatch.setattr(SafetyOracle, "evaluate", _pass_through)

    new_patch_regs = [
        {"safety": {"closing_speed_cap_kph": 55.0}},
        {"safety": {"minimum_reaction_margin_s": 0.75}},
        {
            "safety": {
                "confined_corner_attack_restriction": True,
                "confined_corner_width_m": 12.5,
                "confined_corner_runoff_types": ["grass", "gravel", "wall", "barrier", "concrete"],
            }
        },
        {
            "safety": {
                "active_aero_delay_high_risk": True,
                "active_aero_delay_risk_threshold": 0.1,
            }
        },
        {
            "safety": {
                "pack_compression_overtake_limit": True,
                "pack_compression_threshold": 0.35,
            }
        },
    ]

    positive_families = [
        fid for fid, spec in SYNTHETIC_FAMILIES.items() if spec.expected_unsafe_legal
    ]
    test_family = positive_families[0]
    spec = SYNTHETIC_FAMILIES[test_family]
    track = build_synthetic_track(spec)
    weather = build_synthetic_weather(spec)
    track_state = build_synthetic_track_state(spec)
    cars = build_synthetic_cars_for_battle(spec)
    actions = build_synthetic_actions_for_battle(spec)

    for patch_safety in new_patch_regs:
        reg = {**_base_regulation(), **patch_safety}
        mk = RaceMicrokernel(regulation=reg, seed=5)
        _, events, _ = mk.resolve_lap(
            lap=1, total_laps=5, cars=cars, actions=actions,
            track=track, weather=weather, track_state=track_state, safety_car_active=False,
        )
        # Must not raise — events may or may not be present
        unsafe = [e for e in events if e.event_type == "unsafe_legal_state"]
        for ev in unsafe:
            details = ev.to_dict().get("details", {})
            assert "regulatory_patch_effects" in details
