"""Track repository and YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reglabsim.track.geometry import TrackModel
from reglabsim.track.segments import (
    KerbProfile,
    RunoffProfile,
    SegmentRiskProfile,
    TrackLimitProfile,
    TrackSegment,
    TrackSurface,
)


class TrackRepository:
    """Load and cache digital track models from YAML configs."""

    def __init__(self, tracks_dir: str | Path = "configs/tracks"):
        self._tracks_dir = Path(tracks_dir)
        self._cache: dict[str, TrackModel] = {}

    def list_ids(self) -> list[str]:
        """List available track identifiers."""
        if not self._tracks_dir.exists():
            return []
        return sorted(path.stem for path in self._tracks_dir.glob("*.yaml"))

    def get(self, track_id: str) -> TrackModel:
        """Load one track model by identifier."""
        if track_id not in self._cache:
            path = self._tracks_dir / f"{track_id}.yaml"
            if not path.exists():
                raise KeyError(f"Track '{track_id}' not found")
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            self._cache[track_id] = self._parse_track(data)
        return self._cache[track_id]

    def _parse_track(self, data: dict[str, Any]) -> TrackModel:
        segments = [self._parse_segment(segment_data) for segment_data in data.get("segments", [])]
        return TrackModel(
            track_id=data["track_id"],
            name=data["name"],
            country=data["country"],
            length_m=float(data["length_m"]),
            turns=int(data["turns"]),
            laps=int(data["laps"]),
            race_distance_m=float(data["race_distance_m"]),
            avg_speed_kph=float(data["avg_speed_kph"]),
            fidelity_level=int(data.get("fidelity_level", 1)),
            segments=segments,
            sources=list(data.get("sources", ["manual_seed"])),
            validation_status=data.get("validation_status", "draft"),
            fidelity_notes=list(data.get("fidelity_notes", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def _parse_surface(self, data: dict[str, Any] | None, fallback: TrackSurface) -> TrackSurface:
        if not data:
            return fallback
        return TrackSurface(
            type=data.get("type", fallback.type),
            grip_dry=float(data.get("grip_dry", fallback.grip_dry)),
            grip_wet=float(data.get("grip_wet", fallback.grip_wet)),
            roughness=float(data.get("roughness", fallback.roughness)),
            drainage=data.get("drainage", fallback.drainage),
            dirt_level=float(data.get("dirt_level", fallback.dirt_level)),
            marbles_level=float(data.get("marbles_level", fallback.marbles_level)),
        )

    def _parse_kerb(self, data: dict[str, Any] | None) -> KerbProfile | None:
        if not data:
            return None
        return KerbProfile(
            type=data["type"],
            height_mm=float(data.get("height_mm", 0.0)),
            width_m=float(data.get("width_m", 1.0)),
            grip_dry=float(data.get("grip_dry", 0.8)),
            grip_wet=float(data.get("grip_wet", 0.45)),
            destabilization_factor=float(data.get("destabilization_factor", 0.1)),
            bottoming_risk=data.get("bottoming_risk", "low"),
            launch_risk=data.get("launch_risk", "low"),
            track_limits_sensitive=bool(data.get("track_limits_sensitive", False)),
        )

    def _parse_track_limits(self, data: dict[str, Any] | None) -> TrackLimitProfile | None:
        if not data:
            return None
        return TrackLimitProfile(
            rule=data.get("rule", "white_line"),
            allowed_wheels_out=int(data.get("allowed_wheels_out", 0)),
            detection_probability=float(data.get("detection_probability", 1.0)),
            warning_threshold=int(data.get("warning_threshold", 3)),
            penalty_after=int(data.get("penalty_after", 4)),
            time_gain_sensitive=bool(data.get("time_gain_sensitive", False)),
            estimated_gain_if_abused_s=float(data.get("estimated_gain_if_abused_s", 0.0)),
        )

    def _parse_runoff(self, data: dict[str, Any] | None) -> RunoffProfile:
        data = data or {}
        return RunoffProfile(
            type=data.get("type", "asphalt"),
            width_m=float(data.get("width_m", 12.0)),
            grip_dry=float(data.get("grip_dry", 0.8)),
            grip_wet=float(data.get("grip_wet", 0.5)),
            rejoin_risk=data.get("rejoin_risk", "medium"),
            recovery_probability=data.get("recovery_probability", "medium"),
        )

    def _parse_risk(self, data: dict[str, Any] | None) -> SegmentRiskProfile:
        data = data or {}
        return SegmentRiskProfile(
            unsafe_closing_speed_threshold_kph=float(
                data.get("unsafe_closing_speed_threshold_kph", 45.0)
            ),
            side_by_side_risk=data.get("side_by_side_risk", "medium"),
            evasive_action_margin=data.get("evasive_action_margin", "medium"),
            energy_delta_sensitivity=data.get("energy_delta_sensitivity", "medium"),
            active_aero_sensitivity=data.get("active_aero_sensitivity", "medium"),
            visibility_risk=data.get("visibility_risk", "medium"),
            barrier_distance_m=float(data.get("barrier_distance_m", 25.0)),
            impact_severity_multiplier=float(data.get("impact_severity_multiplier", 1.0)),
        )

    def _parse_segment(self, data: dict[str, Any]) -> TrackSegment:
        fallback_main = TrackSurface(type="asphalt", grip_dry=1.0, grip_wet=0.72)
        fallback_line = TrackSurface(type="asphalt", grip_dry=1.03, grip_wet=0.76)
        fallback_offline = TrackSurface(
            type="asphalt",
            grip_dry=0.9,
            grip_wet=0.62,
            dirt_level=0.2,
            marbles_level=0.2,
        )
        surface = data.get("surface", {})
        kerbs = data.get("kerbs", {})

        return TrackSegment(
            segment_id=data["id"],
            name=data.get("name", data["id"]),
            segment_type=data["type"],
            start_m=float(data["start_m"]),
            end_m=float(data["end_m"]),
            width_m=float(data.get("width_m", 12.0)),
            radius_m=float(data["radius_m"]) if data.get("radius_m") is not None else None,
            elevation_delta_m=float(data.get("elevation_delta_m", 0.0)),
            overtaking_viability=data.get("overtaking_viability", "low"),
            preferred_battle_zone=bool(data.get("preferred_battle_zone", False)),
            primary_recharge_zone=bool(data.get("primary_recharge_zone", False)),
            primary_boost_zone=bool(data.get("primary_boost_zone", False)),
            main_surface=self._parse_surface(surface.get("main_track"), fallback_main),
            racing_line_surface=self._parse_surface(surface.get("racing_line"), fallback_line),
            offline_surface=self._parse_surface(surface.get("offline"), fallback_offline),
            inside_kerb=self._parse_kerb(kerbs.get("inside")),
            outside_kerb=self._parse_kerb(kerbs.get("outside")),
            runoff=self._parse_runoff(data.get("runoff", {}).get("outside")),
            track_limits=self._parse_track_limits(data.get("track_limits")),
            risk=self._parse_risk(data.get("risk")),
            metadata=data.get("metadata", {}),
        )
