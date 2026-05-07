"""Tests for multi-circuit public race validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from reglabsim import create_facade


def test_public_race_pack_runs_for_available_session(tmp_path: Path) -> None:
    config = {
        "name": "smoke_public_race_pack",
        "regulation_id": "regulation_2026_refined",
        "defaults": {
            "year": 2024,
            "session_type": "race",
            "mode": "llm_event_driven",
            "llm_provider": "heuristic",
            "llm_model": "event-driven-fallback",
            "num_cars": 10,
            "laps": 8,
            "seed": 42,
        },
        "thresholds": {
            "mean_overall_score_min": 0.0,
            "min_case_overall_score": 0.0,
            "mean_lap_mape_pct_max": 100.0,
        },
        "sessions": [{"track_id": "suzuka"}],
    }
    config_path = tmp_path / "public_race_smoke.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    facade = create_facade()
    report = facade.validate_public_race_pack(
        config_path=config_path,
        data_root="data",
        output_dir=tmp_path / "validation",
        ingest_if_missing=False,
    )

    assert report["case_count"] == 1
    assert report["cases"][0]["query"]["track_id"] == "suzuka"
    assert report["cases"][0]["public_validation"]["scorecard"]["overall_score"] >= 0.0
    assert report["summary"]["status"] in {"meets_thresholds", "needs_calibration"}
    assert Path(report["saved_report_path"]).exists()
