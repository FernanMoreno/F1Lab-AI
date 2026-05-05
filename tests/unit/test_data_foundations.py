"""Tests for data ingestion and condition/track provenance foundations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from reglabsim import create_facade
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.data import SessionQuery, UnifiedDataSource


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
        }


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
    assert Path(persisted["silver::laps"].data_path).exists()
    assert persisted["silver::laps"].metadata["resolved_session"]["session_key"] == 9158

