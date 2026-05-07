"""Simple campaign reporting helpers."""

from __future__ import annotations

from typing import Any


def markdown_summary(run_output: dict[str, Any]) -> str:
    """Build a compact Markdown summary for one run."""
    manifest = run_output["manifest"]
    metrics = run_output["metrics"]
    failures = run_output.get("failure_log", [])
    return "\n".join(
        [
            f"# {manifest['race_name']}",
            "",
            f"- Track: `{manifest['track_id']}`",
            f"- Regulation: `{manifest['regulation_id']}`",
            f"- Mode: `{manifest['mode']}`",
            f"- Winner: `{run_output['result']['winner']}`",
            f"- Overtakes: `{metrics['total_overtakes']}`",
            f"- Incidents: `{metrics['incident_count']}`",
            f"- Failures: `{len(failures)}`",
        ]
    )


def campaign_summary(
    campaign_name: str, runs: list[dict[str, Any]], ranking: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a compact campaign summary object."""
    return {
        "campaign_name": campaign_name,
        "num_runs": len(runs),
        "tracks": sorted({run["manifest"]["track_id"] for run in runs}),
        "top_failure": ranking[0]["failure_type"] if ranking else None,
        "total_failures": sum(len(run.get("failure_log", [])) for run in runs),
    }
