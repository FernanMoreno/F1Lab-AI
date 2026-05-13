"""Streamlit dashboard backed by real validation, runs, and track-pack outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

try:
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency is optional in tests.
    st = None


APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = APP_ROOT / "configs"
OUTPUTS_DIR = APP_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
VALIDATION_REPORT_PATH = (
    OUTPUTS_DIR
    / "validation"
    / "public_primitives_target_pack"
    / "primitive_validation_pack_report.json"
)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else None


def load_track_pack_rows(config_dir: Path = CONFIG_DIR) -> list[dict[str, Any]]:
    payload = _load_yaml(config_dir / "track_pack.yaml") or {}
    rows: list[dict[str, Any]] = []
    for entry in payload.get("tracks", []):
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "track_id": str(entry.get("track_id", "unknown")),
                "build_priority": int(entry.get("build_priority", 0)),
                "builder_hint": str(entry.get("builder_hint", "unknown")),
                "expected_fidelity_level": int(entry.get("expected_fidelity_level", 0)),
                "target_validation_status": str(
                    entry.get("target_validation_status", "unknown")
                ),
                "notes": ", ".join(str(note) for note in entry.get("notes", [])),
            }
        )
    return rows


def load_validation_overview(
    report_path: Path = VALIDATION_REPORT_PATH,
) -> dict[str, Any] | None:
    report = _load_json(report_path)
    if not isinstance(report, dict):
        return None

    summary = report.get("summary", {})
    lap_summary = summary.get("lap", {}) if isinstance(summary, dict) else {}
    battle_summary = summary.get("battle", {}) if isinstance(summary, dict) else {}
    track_coverage = report.get("track_coverage", [])

    case_rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            continue
        lap = case.get("lap", {}) if isinstance(case.get("lap"), dict) else {}
        battle = case.get("battle", {}) if isinstance(case.get("battle"), dict) else {}
        query = lap.get("query") or battle.get("query") or {}
        lap_history = lap.get("candidate_history", [])
        battle_history = battle.get("candidate_history", [])
        case_rows.append(
            {
                "track_id": str(query.get("track_id", "unknown")),
                "year": int(query.get("year", 0)),
                "session_type": str(query.get("session_type", "unknown")),
                "lap_status": str(lap.get("status", "missing")),
                "lap_score": _best_candidate_score(lap_history),
                "battle_status": str(battle.get("status", "missing")),
                "battle_score": _best_candidate_score(battle_history),
            }
        )

    return {
        "pack_name": str(report.get("pack_name", "unknown")),
        "overall_status": str(report.get("overall_status", "unknown")),
        "case_count": int(report.get("case_count", 0)),
        "successful_primitive_runs": int(report.get("successful_primitive_runs", 0)),
        "failure_count": int(report.get("failure_count", 0)),
        "coverage_count": len(track_coverage) if isinstance(track_coverage, list) else 0,
        "lap_mean_score": _as_float(lap_summary.get("mean_score")),
        "lap_max_score": _as_float(lap_summary.get("max_score")),
        "battle_mean_score": _as_float(battle_summary.get("mean_score")),
        "battle_max_score": _as_float(battle_summary.get("max_score")),
        "lap_status_counts": lap_summary.get("status_counts", {}),
        "battle_status_counts": battle_summary.get("status_counts", {}),
        "case_rows": case_rows,
        "threshold_evaluation": report.get("threshold_evaluation", {}),
        "report_path": str(report_path),
    }


def load_run_summaries(
    runs_dir: Path = RUNS_DIR,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    run_dirs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    summaries: list[dict[str, Any]] = []

    for run_dir in run_dirs[:limit]:
        manifest = _load_json(run_dir / "manifest.json")
        metrics = _load_json(run_dir / "metrics.json")
        failures = _load_json(run_dir / "failure_log.json")
        result = _load_json(run_dir / "result.json")
        if not isinstance(manifest, dict):
            continue
        metrics = metrics if isinstance(metrics, dict) else {}
        failures_list = failures if isinstance(failures, list) else []
        result = result if isinstance(result, dict) else {}
        summaries.append(
            {
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "race_name": str(manifest.get("race_name", run_dir.name)),
                "track_id": str(manifest.get("track_id", "unknown")),
                "regulation_id": str(manifest.get("regulation_id", "unknown")),
                "mode": str(manifest.get("mode", "unknown")),
                "seed": manifest.get("seed"),
                "incident_count": int(metrics.get("incident_count", 0)),
                "forcing_off_track_events": int(metrics.get("forcing_off_track_events", 0)),
                "unsafe_defending_events": int(metrics.get("unsafe_defending_events", 0)),
                "retirements": len(result.get("retirements", []))
                if isinstance(result.get("retirements"), list)
                else int(metrics.get("retirements", 0)),
                "failure_count": len(failures_list),
                "summary_path": str(run_dir / "summary.md"),
            }
        )

    return summaries


def load_home_metrics(
    *,
    config_dir: Path = CONFIG_DIR,
    report_path: Path = VALIDATION_REPORT_PATH,
    runs_dir: Path = RUNS_DIR,
) -> dict[str, Any]:
    regulations = list((config_dir / "regulations").glob("*.yaml"))
    car_families = _load_yaml(config_dir / "car_families.yaml") or {}
    track_rows = load_track_pack_rows(config_dir)
    validation = load_validation_overview(report_path)
    run_summaries = load_run_summaries(runs_dir, limit=50)
    return {
        "regulation_count": len(regulations),
        "car_family_count": len(car_families.get("car_families", {})),
        "track_count": len(track_rows),
        "run_count": len(run_summaries),
        "validation_status": validation["overall_status"] if validation else "missing",
        "validation_case_count": validation["case_count"] if validation else 0,
        "validation_battle_mean_score": validation["battle_mean_score"] if validation else None,
        "validation_lap_mean_score": validation["lap_mean_score"] if validation else None,
    }


def _best_candidate_score(history: Any) -> float | None:
    if not isinstance(history, list) or not history:
        return None
    candidate = history[0]
    if not isinstance(candidate, dict):
        return None
    return _as_float(candidate.get("score"))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    """Main dashboard entry point."""
    if st is None:
        raise RuntimeError("Streamlit is not installed in this environment")

    st.set_page_config(
        page_title="F1Lab-AI Dashboard",
        layout="wide",
    )

    st.title("F1Lab-AI Dashboard")
    st.caption("Real outputs from validation, track pack, and race runs.")

    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Validation", "Runs", "Track Pack", "Regulations"],
    )

    if page == "Overview":
        show_overview()
    elif page == "Validation":
        show_validation()
    elif page == "Runs":
        show_runs()
    elif page == "Track Pack":
        show_track_pack()
    elif page == "Regulations":
        show_regulations()


def show_overview() -> None:
    metrics = load_home_metrics()
    validation = load_validation_overview()
    runs = load_run_summaries(limit=8)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regulations", metrics["regulation_count"])
    col2.metric("Car Families", metrics["car_family_count"])
    col3.metric("Target Tracks", metrics["track_count"])
    col4.metric("Recorded Runs", metrics["run_count"])

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Validation Status", metrics["validation_status"])
    col_b.metric("Validation Cases", metrics["validation_case_count"])
    col_c.metric(
        "Battle Mean Score",
        _format_score(metrics["validation_battle_mean_score"]),
    )

    if validation:
        st.subheader("Validation Snapshot")
        st.dataframe(validation["case_rows"], use_container_width=True, hide_index=True)
    else:
        st.warning("Validation report not found.")

    st.subheader("Latest Runs")
    if runs:
        st.dataframe(runs, use_container_width=True, hide_index=True)
    else:
        st.info("No run artifacts found in outputs/runs.")


def show_validation() -> None:
    overview = load_validation_overview()
    if overview is None:
        st.warning(f"Validation report not found at {VALIDATION_REPORT_PATH}.")
        return

    st.header("Public Primitive Validation")
    st.caption(overview["report_path"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pack", overview["pack_name"])
    col2.metric("Status", overview["overall_status"])
    col3.metric("Lap Mean Score", _format_score(overview["lap_mean_score"]))
    col4.metric("Battle Mean Score", _format_score(overview["battle_mean_score"]))

    st.subheader("Per-Circuit Cases")
    st.dataframe(overview["case_rows"], use_container_width=True, hide_index=True)

    st.subheader("Threshold Evaluation")
    st.json(overview["threshold_evaluation"])


def show_runs() -> None:
    st.header("Run Artifacts")
    runs = load_run_summaries(limit=20)
    if not runs:
        st.info("No run artifacts found in outputs/runs.")
        return

    st.dataframe(runs, use_container_width=True, hide_index=True)
    selected_run = st.selectbox("Select run", [run["run_id"] for run in runs])
    selected = next(run for run in runs if run["run_id"] == selected_run)
    st.json(selected)


def show_track_pack() -> None:
    st.header("Canonical Track Pack")
    rows = load_track_pack_rows()
    if not rows:
        st.warning("Track pack config not found.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_regulations() -> None:
    st.header("Regulations")
    regulation_dir = CONFIG_DIR / "regulations"
    regulation_files = sorted(regulation_dir.glob("*.yaml"))
    if not regulation_files:
        st.warning("Regulation configs not found.")
        return

    selected = st.selectbox("Select regulation", [path.stem for path in regulation_files])
    payload = _load_yaml(regulation_dir / f"{selected}.yaml") or {}

    col1, col2 = st.columns(2)
    col1.metric("Version", payload.get("version", "unknown"))
    col2.metric("Status", payload.get("status", "unknown"))
    st.json(payload)


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
