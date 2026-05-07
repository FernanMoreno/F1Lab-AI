"""Helpers for extracting metric inputs from run outputs."""

from __future__ import annotations

from typing import Any


def extract_event_log(simulation_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Return event log or derive a lightweight overtake log."""
    if "event_log" in simulation_output:
        return list(simulation_output["event_log"])
    overtakes = []
    for overtake in simulation_output.get("overtakes", []):
        overtakes.append(
            {
                "event_type": overtake.get("type", "overtake"),
                "details": overtake,
            }
        )
    return overtakes


def extract_events(simulation_output: dict[str, Any], *event_types: str) -> list[dict[str, Any]]:
    """Return events filtered by event type."""
    target = set(event_types)
    return [
        event for event in extract_event_log(simulation_output) if event.get("event_type") in target
    ]


def positions_history(simulation_output: dict[str, Any]) -> list[list[str]]:
    """Return position snapshots from runtime state history if available."""
    if "positions_history" in simulation_output:
        return list(simulation_output["positions_history"])
    snapshots = simulation_output.get("state_snapshots", [])
    history = []
    for snapshot in snapshots:
        cars = sorted(snapshot.get("cars", []), key=lambda item: item["position"])
        history.append([car["car_id"] for car in cars])
    return history


def weather_series(simulation_output: dict[str, Any], field_name: str) -> list[float]:
    """Return a weather or track-state series from snapshots."""
    values = []
    for snapshot in simulation_output.get("state_snapshots", []):
        if field_name in snapshot.get("weather", {}):
            values.append(float(snapshot["weather"][field_name]))
        elif field_name in snapshot.get("track_state", {}):
            values.append(float(snapshot["track_state"][field_name]))
    return values
