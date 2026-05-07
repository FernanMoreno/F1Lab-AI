from __future__ import annotations

from dashboards.streamlit_app import (
    CONFIG_DIR,
    RUNS_DIR,
    VALIDATION_REPORT_PATH,
    load_home_metrics,
    load_run_summaries,
    load_track_pack_rows,
    load_validation_overview,
)


def test_track_pack_rows_follow_canonical_pack() -> None:
    rows = load_track_pack_rows(CONFIG_DIR)

    assert len(rows) == 8
    assert [row["track_id"] for row in rows] == [
        "suzuka",
        "baku",
        "monaco",
        "monza",
        "austria",
        "singapore",
        "barcelona",
        "silverstone",
    ]


def test_validation_overview_reads_multi_circuit_report() -> None:
    overview = load_validation_overview(VALIDATION_REPORT_PATH)

    assert overview is not None
    assert overview["overall_status"] == "meets_thresholds"
    assert overview["case_count"] == 5
    assert overview["battle_mean_score"] is not None
    assert overview["lap_mean_score"] is not None
    assert len(overview["case_rows"]) == 5


def test_run_summaries_read_real_run_artifacts() -> None:
    summaries = load_run_summaries(RUNS_DIR, limit=5)

    assert summaries
    assert all("run_id" in summary for summary in summaries)
    assert all("track_id" in summary for summary in summaries)
    assert all("failure_count" in summary for summary in summaries)


def test_home_metrics_use_real_outputs() -> None:
    metrics = load_home_metrics()

    assert metrics["regulation_count"] >= 4
    assert metrics["car_family_count"] >= 1
    assert metrics["track_count"] == 8
    assert metrics["validation_status"] == "meets_thresholds"
