from __future__ import annotations

import numpy as np

from reglabsim.race.traffic import TrafficModel


def test_traffic_model_probability_rewards_slipstream_and_penalizes_dirty_air() -> None:
    model = TrafficModel()

    clean_air = model.calculate_overtake_probability(
        pace_diff_s_per_lap=-0.2,
        closing_speed_kph=32.0,
        drs_available=False,
        ers_advantage=1.0,
        slipstream_gain_mps=3.5,
        dirty_air_penalty_mps=0.4,
        pack_compression_ratio=0.2,
        local_density=1.1,
    )
    turbulent_air = model.calculate_overtake_probability(
        pace_diff_s_per_lap=-0.2,
        closing_speed_kph=32.0,
        drs_available=False,
        ers_advantage=1.0,
        slipstream_gain_mps=0.8,
        dirty_air_penalty_mps=2.6,
        pack_compression_ratio=0.8,
        local_density=2.4,
    )

    assert clean_air > turbulent_air
    assert 0.0 <= clean_air <= 1.0
    assert 0.0 <= turbulent_air <= 1.0


def test_traffic_model_simulate_overtake_attempt_accepts_deterministic_rng() -> None:
    model = TrafficModel()
    rng = np.random.default_rng(3)

    event = model.simulate_overtake_attempt(
        attacker_config={"car_id": "attacker", "lap_time_s": 80.0, "top_speed_kph": 312},
        defender_config={"car_id": "defender", "lap_time_s": 80.4, "top_speed_kph": 305},
        regulation={},
        drs_available=True,
        rng=rng,
    )

    assert event.attacker == "attacker"
    assert event.defender == "defender"
    assert event.closing_speed_kph == 7
    assert isinstance(event.success, bool)


def test_traffic_model_filters_non_battle_position_shuffles() -> None:
    model = TrafficModel()

    assert not model.is_battle_eligible(
        gap_s=2.3,
        battle_distance_m=120.0,
        closing_speed_kph=6.0,
        attacker_committed=False,
        defender_committed=False,
    )
    assert model.is_battle_eligible(
        gap_s=0.7,
        battle_distance_m=38.0,
        closing_speed_kph=24.0,
        attacker_committed=True,
        defender_committed=False,
    )
