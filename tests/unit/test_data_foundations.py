"""Tests for data ingestion and condition/track provenance foundations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reglabsim import create_facade
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.data import LocalDataLake, SessionQuery, UnifiedDataSource
from reglabsim.data.openf1_client import OpenF1Client
from reglabsim.data.pipelines import standard_pipeline
from reglabsim.validation.public_session import PublicSessionValidator


class _FakeOpenF1Source:
    def __init__(self) -> None:
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def resolve_session(self, query: SessionQuery) -> dict[str, object]:
        return {
            "session_key": 9158,
            "meeting_key": 1219,
            "session_name": "Race",
            "circuit_short_name": "Suzuka",
        }

    def fetch_session_bundle(self, query: SessionQuery) -> dict[str, pd.DataFrame]:
        return {
            "sessions": pd.DataFrame(
                [
                    {
                        "session_key": 9158,
                        "meeting_key": 1219,
                        "session_name": "Race",
                        "circuit_short_name": "Suzuka",
                    }
                ]
            ),
            "laps": pd.DataFrame([{"driver_number": 1, "lap_number": 1, "lap_duration": 91.2}]),
            "weather": pd.DataFrame([{"air_temperature": 23.0, "track_temperature": 31.0}]),
            "race_control": pd.DataFrame([{"category": "Flag", "flag": "GREEN"}]),
            "stints": pd.DataFrame([{"driver_number": 1, "stint_number": 1, "compound": "SOFT"}]),
            "intervals": pd.DataFrame(
                [{"driver_number": 1, "interval": 0.8, "gap_to_leader": 1.2}]
            ),
            "location": pd.DataFrame(
                [{"driver_number": 1, "date": "2024-04-07T05:00:00Z", "x": 10.0, "y": 20.0}]
            ),
        }


class _FakeOpenMeteoClient:
    def fetch_historical_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2024-04-07T00:00",
                    "air_temperature": 22.0,
                    "humidity": 60.0,
                    "pressure": 1010.0,
                    "rainfall": 0.0,
                    "wind_speed": 14.4,
                    "wind_direction": 95.0,
                    "latitude": latitude,
                    "longitude": longitude,
                    "source": "openmeteo",
                    "dataset_name": "historical_weather",
                },
                {
                    "date": "2024-04-07T01:00",
                    "air_temperature": 24.0,
                    "humidity": 58.0,
                    "pressure": 1011.0,
                    "rainfall": 0.2,
                    "wind_speed": 18.0,
                    "wind_direction": 100.0,
                    "latitude": latitude,
                    "longitude": longitude,
                    "source": "openmeteo",
                    "dataset_name": "historical_weather",
                },
            ]
        )


class _FallbackLocationOpenF1Client(OpenF1Client):
    def __init__(self) -> None:
        super().__init__()
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def resolve_session(self, query: SessionQuery) -> dict[str, object]:
        return {
            "session_key": 9991,
            "meeting_key": 2001,
            "session_name": "Race",
            "circuit_short_name": "Suzuka",
        }

    def fetch_lap_data(self, circuit_id: str, session_type: str, year: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"driver_number": 1, "lap_number": 1, "lap_duration": 91.0},
                {"driver_number": 2, "lap_number": 1, "lap_duration": 91.4},
            ]
        )

    def fetch_weather(self, session_id: str) -> pd.DataFrame:
        return pd.DataFrame([{"air_temperature": 24.0, "track_temperature": 32.0}])

    def fetch_race_control(self, session_id: str) -> pd.DataFrame:
        return pd.DataFrame([{"message": "GREEN FLAG"}])

    def fetch_stints(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_position(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2024-04-07T05:00:00Z", "driver_number": 1, "position": 1},
                {"date": "2024-04-07T05:00:00Z", "driver_number": 2, "position": 2},
            ]
        )

    def fetch_intervals(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        return pd.DataFrame([{"driver_number": 1, "interval": 0.7, "gap_to_leader": 1.1}])

    def fetch_location(self, session_id: str, driver_id: str | None = None) -> pd.DataFrame:
        if driver_id is None:
            return pd.DataFrame()
        number = int(driver_id)
        return pd.DataFrame(
            [
                {
                    "date": "2024-04-07T05:00:00Z",
                    "driver_number": number,
                    "x": float(number) * 10.0,
                    "y": 0.0,
                    "z": 0.0,
                }
            ]
        )


def _campaign_config(tmp_path: Path, source_name: str) -> Path:
    source = Path("configs/campaigns") / source_name
    with open(source, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["output_root"] = str(tmp_path / "runs")
    target = tmp_path / source_name
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return target


def test_campaign_spec_loads_weather_profile_and_inline_overrides(tmp_path: Path) -> None:
    config_path = _campaign_config(tmp_path, "baku_redteam.yaml")

    spec = CampaignSpec.from_yaml(config_path)

    assert spec.weather_profile == "windy_baku"
    assert spec.conditions is not None
    assert spec.conditions.weather.wind_speed_mps == 8.0
    assert spec.conditions.track.track_temp_c == 31.0
    assert spec.conditions.metadata["validation_status"] == "draft_profile"


def test_facade_describe_track_exposes_provenance() -> None:
    facade = create_facade()

    track = facade.describe_track("suzuka")

    assert track["fidelity_level"] == 4
    assert "manual_curation" in track["sources"]
    assert track["validation_status"] == "seeded_manual_review"


def test_unified_data_source_persists_openf1_bundle(tmp_path: Path) -> None:
    data_source = UnifiedDataSource(primary="openf1")
    fake = _FakeOpenF1Source()
    fake.connect()
    data_source.add_source("openf1", fake)
    data_source.connect()

    persisted = data_source.ingest_openf1_session(
        SessionQuery(year=2024, track_id="suzuka", session_type="race"),
        data_root=str(tmp_path / "lake"),
    )

    assert "raw::laps" in persisted
    assert "silver::weather" in persisted
    assert "silver::intervals" in persisted
    assert "silver::location" in persisted
    assert Path(persisted["silver::laps"].data_path).exists()
    assert persisted["silver::laps"].metadata["resolved_session"]["session_key"] == 9158


def test_standard_pipeline_normalizes_openf1_intervals() -> None:
    query = SessionQuery(year=2024, track_id="suzuka", session_type="race")
    frame = pd.DataFrame(
        [
            {"driver_number": 1, "interval": 0.8, "gap_to_leader": 1.2},
            {"driver_number": 2, "interval": "0.9", "gap_to_leader": "+1 LAP"},
        ]
    )

    normalized = standard_pipeline(query=query, dataset_name="intervals").run(frame)

    assert float(normalized.loc[0, "interval"]) == 0.8
    assert pd.isna(normalized.loc[1, "gap_to_leader"])
    assert "gap_to_leader_raw" in normalized.columns
    assert normalized.loc[1, "gap_to_leader_raw"] == "+1 LAP"


def test_local_datalake_persists_mixed_object_columns(tmp_path: Path) -> None:
    lake = LocalDataLake(tmp_path / "lake")
    frame = pd.DataFrame(
        [
            {"gap_to_leader": 1.2, "meta": {"flag": "green"}},
            {"gap_to_leader": "+1 LAP", "meta": {"flag": "blue"}},
        ]
    )

    persisted = lake.persist_frame(
        frame,
        layer="raw",
        source="openf1",
        dataset_name="intervals",
        partition="year=2024/track=suzuka/session=race",
    )
    loaded = lake.load_frame(
        layer="raw",
        source="openf1",
        dataset_name="intervals",
        partition="year=2024/track=suzuka/session=race",
    )

    assert Path(persisted.data_path).exists()
    assert str(loaded.loc[1, "gap_to_leader"]) == "+1 LAP"


def test_openf1_bundle_backfills_location_per_driver() -> None:
    client = _FallbackLocationOpenF1Client()
    client.connect()

    bundle = client.fetch_session_bundle(
        SessionQuery(year=2024, track_id="suzuka", session_type="race")
    )

    assert not bundle["location"].empty
    assert set(bundle["location"]["driver_number"].tolist()) == {1, 2}


def test_build_weather_profile_from_historical_weather(tmp_path: Path) -> None:
    facade = create_facade(config_dir="configs")
    facade._openmeteo_client = _FakeOpenMeteoClient()

    result = facade.build_weather_profile(
        track_id="suzuka",
        start_date="2024-04-07",
        end_date="2024-04-07",
        profile_id="generated_suzuka_profile",
        data_root=str(tmp_path / "lake"),
        save_profile=False,
    )

    assert result["profile"]["name"] == "generated_suzuka_profile"
    assert result["profile"]["weather"]["air_temp_c"] == 23.0
    assert result["profile"]["metadata"]["validation_status"] == "generated_from_historical_weather"


def test_public_session_validator_scores_saved_public_data(tmp_path: Path) -> None:
    lake = LocalDataLake(tmp_path / "lake")
    partition = "year=2024/track=suzuka/session=race"
    lake.persist_frame(
        pd.DataFrame(
            [
                {"lap_duration": 91.0, "is_pit_out_lap": False},
                {"lap_duration": 92.0, "is_pit_out_lap": False},
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
                },
                {
                    "air_temperature": 24.0,
                    "track_temperature": 32.0,
                    "wind_speed": 16.0,
                    "rainfall": 0.0,
                },
            ]
        ),
        layer="silver",
        source="openf1",
        dataset_name="weather",
        partition=partition,
    )
    lake.persist_frame(
        pd.DataFrame([{"message": "GREEN FLAG"}]),
        layer="silver",
        source="openf1",
        dataset_name="race_control",
        partition=partition,
    )

    run_output = {
        "metrics": {"incident_count": 0},
        "physics_resolution_log": [{"lap_time_s": 90.5}, {"lap_time_s": 91.5}],
        "state_snapshots": [
            {
                "weather": {"air_temp_c": 23.0, "wind_speed_mps": 4.0, "rain_intensity_mm_h": 0.0},
                "track_state": {"track_temp_c": 31.0},
            },
            {
                "weather": {"air_temp_c": 24.0, "wind_speed_mps": 4.5, "rain_intensity_mm_h": 0.0},
                "track_state": {"track_temp_c": 32.0},
            },
        ],
    }
    validator = PublicSessionValidator(data_root=str(tmp_path / "lake"))
    report = validator.validate_run_against_session(
        run_output=run_output,
        query=SessionQuery(year=2024, track_id="suzuka", session_type="race"),
    )

    assert report["actual_summary"]["avg_lap_time_s"] == 91.5
    assert "overall_score" in report["scorecard"]
