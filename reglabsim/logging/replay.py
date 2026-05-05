"""Replay and persistence helpers for multiagent race runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReplayEngine:
    """Persist and replay run outputs."""

    def save_run(self, run_output: dict[str, Any], output_dir: str | Path) -> Path:
        """Write one run output bundle to disk as JSON files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, value in run_output.items():
            if key == "summary_markdown":
                (out_dir / "summary.md").write_text(value, encoding="utf-8")
                continue
            path = out_dir / f"{key}.json"
            path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
        return out_dir

    def load_run(self, path: str | Path) -> dict[str, Any]:
        """Load a run bundle from disk."""
        directory = Path(path)
        result: dict[str, Any] = {}
        for json_file in directory.glob("*.json"):
            result[json_file.stem] = json.loads(json_file.read_text(encoding="utf-8"))
        summary_file = directory / "summary.md"
        if summary_file.exists():
            result["summary_markdown"] = summary_file.read_text(encoding="utf-8")
        return result

    def replay_audit_exact(self, run_output: dict[str, Any]) -> dict[str, Any]:
        """Return the persisted result exactly as captured."""
        return {
            "mode": "replay_audit_exact",
            "manifest": run_output["manifest"],
            "state_snapshots": run_output["state_snapshots"],
            "event_log": run_output["event_log"],
            "steward_log": run_output["steward_log"],
            "failure_log": run_output.get("failure_log", []),
        }

    def extract_policy_replay_actions(self, run_output: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
        """Build a lookup for policy replay mode."""
        replay_actions: dict[tuple[int, str], dict[str, Any]] = {}
        for entry in run_output.get("action_log", []):
            replay_actions[(int(entry["lap"]), str(entry["car_id"]))] = entry["action"]
        return replay_actions
