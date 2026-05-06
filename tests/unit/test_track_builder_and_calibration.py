"""Tests for track seed building and public primitive calibration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reglabsim import create_facade
from reglabsim.data import LocalDataLake
from reglabsim.track import GeospatialTrackBuilder
from reglabsim.track.track_loader import TrackRepository
from reglabsim.validation.primitives import PublicPrimitiveCalibrator


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
    lake.persist_frame(
        pd.DataFrame(
            [
                {"driver_number": 1, "interval": 0.9, "gap_to_leader": 1.1},
                {"driver_number": 2, "interval": 1.4, "gap_to_leader": 2.2},
                {"driver_number": 1, "interval": 0.7, "gap_to_leader": 0.9},
                {"driver_number": 2, "interval": 1.1, "gap_to_leader": 1.7},
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="intervals",
        partition=partition,
    )
    lake.persist_frame(
        pd.DataFrame(
            [
                {"date": "2024-04-07T05:00:00Z", "driver_number": 1, "x": 0.0, "y": 0.0},
                {"date": "2024-04-07T05:00:00Z", "driver_number": 2, "x": 40.0, "y": 0.0},
                {"date": "2024-04-07T05:00:02Z", "driver_number": 1, "x": 10.0, "y": 0.0},
                {"date": "2024-04-07T05:00:02Z", "driver_number": 2, "x": 55.0, "y": 0.0},
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="location",
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


def test_builder_builds_from_overpass_payload(tmp_path: Path) -> None:
    builder = GeospatialTrackBuilder(tmp_path)
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 101,
                "geometry": [
                    {"lat": 34.0, "lon": 136.0},
                    {"lat": 34.0005, "lon": 136.0005},
                    {"lat": 34.001, "lon": 136.001},
                ],
                "tags": {"highway": "raceway", "width": "14"},
            },
            {
                "type": "way",
                "id": 102,
                "geometry": [
                    {"lat": 34.001, "lon": 136.001},
                    {"lat": 34.0012, "lon": 136.0015},
                    {"lat": 34.0002, "lon": 136.0018},
                    {"lat": 34.0, "lon": 136.0},
                ],
                "tags": {"highway": "raceway"},
            },
        ]
    }

    result = builder.build_from_overpass_payload(
        track_id="overpass_test",
        metadata={"name": "Overpass Test", "country": "Nowhere", "avg_speed_kph": 180.0},
        payload=payload,
        turns=3,
        laps=10,
        race_distance_m=3000.0,
        fidelity_level=2,
    )

    assert result["track_id"] == "overpass_test"
    assert result["metadata"]["osm_source"] == "overpass"
    assert result["segments"]


def test_builder_prefers_component_near_expected_length_and_excludes_pit_lane(
    tmp_path: Path,
) -> None:
    builder = GeospatialTrackBuilder(tmp_path)
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 201,
                "geometry": [
                    {"lat": 34.0, "lon": 136.0},
                    {"lat": 34.0, "lon": 136.005},
                    {"lat": 34.005, "lon": 136.005},
                ],
                "tags": {"highway": "raceway", "name": "Main Straight"},
            },
            {
                "type": "way",
                "id": 202,
                "geometry": [
                    {"lat": 34.005, "lon": 136.005},
                    {"lat": 34.005, "lon": 136.0},
                    {"lat": 34.0, "lon": 136.0},
                ],
                "tags": {"highway": "raceway", "name": "Back Section"},
            },
            {
                "type": "way",
                "id": 203,
                "geometry": [
                    {"lat": 34.0, "lon": 136.0},
                    {"lat": 34.0002, "lon": 136.0015},
                ],
                "tags": {"highway": "raceway", "name": "Pit Lane"},
            },
            {
                "type": "way",
                "id": 204,
                "geometry": [
                    {"lat": 34.02, "lon": 136.02},
                    {"lat": 34.0205, "lon": 136.021},
                    {"lat": 34.021, "lon": 136.02},
                ],
                "tags": {"highway": "raceway", "name": "Short Loop"},
            },
        ]
    }

    result = builder.build_from_overpass_payload(
        track_id="component_test",
        metadata={
            "name": "Component Test",
            "country": "Nowhere",
            "avg_speed_kph": 180.0,
            "target_length_m": 1800.0,
        },
        payload=payload,
        turns=4,
        laps=12,
        race_distance_m=21600.0,
        fidelity_level=2,
    )

    assert result["metadata"]["osm_auxiliary_way_count"] == 1
    assert 1500.0 < result["metadata"]["osm_component_length_m"] < 2200.0


def test_builder_can_export_curated_track_model_fallback(tmp_path: Path) -> None:
    builder = GeospatialTrackBuilder(tmp_path)
    track = TrackRepository("configs/tracks").get("baku")

    payload = builder.build_from_track_model(
        track,
        fallback_reason="OSM coverage ratio 0.019 below threshold.",
    )

    assert payload["validation_status"] == "fallback_curated_track_model"
    assert "curated_track_model_fallback" in payload["sources"]
    assert payload["metadata"]["fallback_reason"].startswith("OSM coverage ratio")


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
    assert "close_following_ratio" in result["actual_summary"]
    assert "mean_interval_s" in result["actual_summary"]
    assert Path(result["saved_report_path"]).exists()
    assert Path(result["saved_profile_path"]).exists()


def test_facade_calibrate_public_battle_accepts_mixed_iso_location_dates(
    tmp_path: Path,
) -> None:
    data_root = Path(_persist_public_session(tmp_path))
    location_path = (
        data_root
        / "silver"
        / "openf1"
        / "location"
        / "year=2024"
        / "track=suzuka"
        / "session=race"
        / "data.parquet"
    )
    location = pd.read_parquet(location_path)
    location["date"] = [
        "2024-04-07T05:00:00Z",
        "2024-04-07T05:00:00.000000+00:00",
        "2024-04-07T05:00:02Z",
        "2024-04-07T05:00:02.000000+00:00",
    ]
    location.to_parquet(location_path, index=False)

    facade = create_facade()
    result = facade.calibrate_public_battle(
        year=2024,
        track_id="suzuka",
        session_type="race",
        data_root=str(data_root),
        mode="rule_based",
        num_cars=4,
        laps=6,
        ingest_if_missing=False,
    )

    assert result["primitive"] == "battle"
    assert result["actual_summary"]["closing_speed_proxy_from_location_kph"] >= 0.0


def test_location_density_ignores_unrealistic_nearest_distance_jumps() -> None:
    calibrator = PublicPrimitiveCalibrator()
    location = pd.DataFrame(
        [
            {"date": "2024-04-07T05:00:00Z", "driver_number": 1, "x": 0.0, "y": 0.0},
            {"date": "2024-04-07T05:00:00Z", "driver_number": 2, "x": 180.0, "y": 0.0},
            {"date": "2024-04-07T05:00:00Z", "driver_number": 3, "x": 400.0, "y": 0.0},
            {"date": "2024-04-07T05:00:02Z", "driver_number": 1, "x": 0.0, "y": 0.0},
            {"date": "2024-04-07T05:00:02Z", "driver_number": 2, "x": 5.0, "y": 0.0},
            {"date": "2024-04-07T05:00:02Z", "driver_number": 3, "x": 400.0, "y": 0.0},
        ]
    )

    summary = calibrator._summarize_actual_location_density(location, None)

    assert summary["tight_spatial_ratio"] > 0.0
    assert summary["closing_speed_proxy_kph"] == 0.0
