"""Tests for track seed building and public primitive calibration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reglabsim import create_facade
from reglabsim.data import LocalDataLake


def _persist_public_session(tmp_path: Path) -> str:
    lake_root = tmp_path / "lake"
    lake = LocalDataLake(lake_root)
    partition = "year=2024/track=suzuka/session=race"
    lake.persist_frame(
        pd.DataFrame(
            [
                {
                    "driver_number": 1,
                    "lap_number": 3,
                    "lap_duration": 91.2,
                    "duration_sector_1": 31.0,
                    "duration_sector_2": 28.8,
                    "duration_sector_3": 31.4,
                    "st_speed": 305.0,
                    "is_pit_out_lap": False,
                },
                {
                    "driver_number": 1,
                    "lap_number": 4,
                    "lap_duration": 90.9,
                    "duration_sector_1": 30.9,
                    "duration_sector_2": 28.7,
                    "duration_sector_3": 31.3,
                    "st_speed": 307.0,
                    "is_pit_out_lap": False,
                },
                {
                    "driver_number": 2,
                    "lap_number": 3,
                    "lap_duration": 91.6,
                    "duration_sector_1": 31.2,
                    "duration_sector_2": 28.9,
                    "duration_sector_3": 31.5,
                    "st_speed": 304.0,
                    "is_pit_out_lap": False,
                },
                {
                    "driver_number": 2,
                    "lap_number": 4,
                    "lap_duration": 91.1,
                    "duration_sector_1": 31.0,
                    "duration_sector_2": 28.8,
                    "duration_sector_3": 31.3,
                    "st_speed": 306.0,
                    "is_pit_out_lap": False,
                },
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="laps",
        partition=partition,
    )
    lake.persist_frame(
        pd.DataFrame(
            [
                {
                    "air_temperature": 23.0,
                    "track_temperature": 31.0,
                    "wind_speed": 14.0,
                    "rainfall": 0.0,
                    "humidity": 62.0,
                    "pressure": 1011.0,
                },
                {
                    "air_temperature": 24.0,
                    "track_temperature": 32.0,
                    "wind_speed": 16.0,
                    "rainfall": 0.0,
                    "humidity": 58.0,
                    "pressure": 1010.0,
                },
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="weather",
        partition=partition,
    )
    lake.persist_frame(
        pd.DataFrame(
            [
                {"date": "2024-04-07T05:00:00Z", "driver_number": 1, "position": 1},
                {"date": "2024-04-07T05:01:00Z", "driver_number": 1, "position": 2},
                {"date": "2024-04-07T05:02:00Z", "driver_number": 1, "position": 1},
                {"date": "2024-04-07T05:00:00Z", "driver_number": 2, "position": 2},
                {"date": "2024-04-07T05:01:00Z", "driver_number": 2, "position": 1},
                {"date": "2024-04-07T05:02:00Z", "driver_number": 2, "position": 2},
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="position",
        partition=partition,
    )
    lake.persist_frame(
        pd.DataFrame([{"message": "GREEN FLAG"}]),
        layer="silver",
        source="openf1",
        dataset_name="race_control",
        partition=partition,
    )
    return str(lake_root)


def test_facade_build_track_seed_from_planar_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "seed.csv"
    csv_path.write_text(
        "\n".join(
            [
                "x_m,y_m,width_m",
                "0,0,14",
                "120,0,14",
                "220,20,14",
                "260,120,14",
                "240,220,14",
                "120,260,14",
                "20,240,14",
                "0,120,14",
                "0,0,14",
            ]
        ),
        encoding="utf-8",
    )
    facade = create_facade()

    result = facade.build_track_seed(
        track_id="seed_test",
        name="Seed Test",
        country="Nowhere",
        source_kind="csv",
        seed_path=csv_path,
        turns=4,
        laps=20,
        race_distance_m=6200.0,
        output_path=tmp_path / "seed_test.yaml",
    )

    assert Path(result["saved_path"]).exists()
    assert result["summary"]["track_id"] == "seed_test"
    assert result["summary"]["length_m"] > 0
    assert result["summary"]["segments"]


def test_facade_calibrate_public_lap_returns_profile(tmp_path: Path) -> None:
    data_root = _persist_public_session(tmp_path)
    output_dir = tmp_path / "calibration"
    facade = create_facade()

    result = facade.calibrate_public_lap(
        year=2024,
        track_id="suzuka",
        session_type="race",
        data_root=data_root,
        candidate_families=["race_pace_concept", "low_drag_missile"],
        output_dir=output_dir,
        ingest_if_missing=False,
    )

    assert result["primitive"] == "lap"
    assert "straight_speed_factor" in result["calibration_profile"]
    assert result["selected_family"] in {"race_pace_concept", "low_drag_missile"}
    assert Path(result["saved_report_path"]).exists()
    assert Path(result["saved_profile_path"]).exists()


def test_facade_calibrate_public_battle_returns_profile(tmp_path: Path) -> None:
    data_root = _persist_public_session(tmp_path)
    output_dir = tmp_path / "battle_calibration"
    facade = create_facade()

    result = facade.calibrate_public_battle(
        year=2024,
        track_id="suzuka",
        session_type="race",
        data_root=data_root,
        mode="rule_based",
        num_cars=4,
        laps=6,
        output_dir=output_dir,
        ingest_if_missing=False,
    )

    assert result["primitive"] == "battle"
    assert "pace_delta_scale" in result["calibration_profile"]
    assert "closing_speed_scale" in result["calibration_profile"]
    assert Path(result["saved_report_path"]).exists()
    assert Path(result["saved_profile_path"]).exists()
