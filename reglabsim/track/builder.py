"""Geospatial track builder for digital-twin YAML generation."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrackPoint:
    """One geospatial point in a track centerline."""

    latitude: float | None = None
    longitude: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    elevation_m: float | None = None
    width_m: float | None = None


@dataclass(frozen=True)
class BuiltSegment:
    """Intermediate segment representation before YAML export."""

    segment_id: str
    name: str
    segment_type: str
    start_m: float
    end_m: float
    width_m: float
    radius_m: float | None
    elevation_delta_m: float
    overtaking_viability: str
    preferred_battle_zone: bool
    primary_recharge_zone: bool
    primary_boost_zone: bool
    risk: dict[str, Any]
    surface: dict[str, Any]
    runoff: dict[str, Any]
    track_limits: dict[str, Any] | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return YAML-ready mapping."""
        data = asdict(self)
        data["id"] = data.pop("segment_id")
        data["type"] = data.pop("segment_type")
        if self.track_limits is None:
            data.pop("track_limits")
        return data


class GeospatialTrackBuilder:
    """Build track YAML from local geospatial seeds or optional OSM fetches."""

    def __init__(self, tracks_dir: str | Path = "configs/tracks"):
        self._tracks_dir = Path(tracks_dir)

    def build_from_existing_seed(
        self,
        *,
        track_id: str,
        metadata: dict[str, Any],
        centerline: list[TrackPoint],
        turns: int | None = None,
        laps: int | None = None,
        race_distance_m: float | None = None,
        fidelity_level: int = 2,
        sources: list[str] | None = None,
        validation_status: str = "generated_seed",
        fidelity_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a track YAML payload from an explicit centerline."""
        if len(centerline) < 3:
            raise ValueError("Centerline requires at least 3 points")
        cumulative = self._cumulative_distances(centerline)
        total_length_m = cumulative[-1]
        segments = self._segment_centerline(track_id, centerline, cumulative)
        return {
            "track_id": track_id,
            "name": metadata["name"],
            "country": metadata["country"],
            "length_m": round(total_length_m, 3),
            "turns": int(
                turns if turns is not None else sum(seg["type"] != "straight" for seg in segments)
            ),
            "laps": int(laps if laps is not None else metadata.get("laps", 0)),
            "race_distance_m": float(
                race_distance_m
                if race_distance_m is not None
                else metadata.get("race_distance_m", total_length_m)
            ),
            "avg_speed_kph": float(metadata.get("avg_speed_kph", 200.0)),
            "fidelity_level": fidelity_level,
            "sources": list(sources or ["local_centerline_seed"]),
            "validation_status": validation_status,
            "fidelity_notes": list(fidelity_notes or ["Generated from centerline heuristics."]),
            "metadata": {
                **metadata,
                "builder": "geospatial_track_builder.v1",
                "centerline_points": len(centerline),
            },
            "segments": segments,
        }

    def build_from_csv(
        self,
        *,
        track_id: str,
        csv_path: str | Path,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build track YAML from a CSV centerline file."""
        points = self._load_csv_points(csv_path)
        return self.build_from_existing_seed(
            track_id=track_id, metadata=metadata, centerline=points, **kwargs
        )

    def build_from_geojson(
        self,
        *,
        track_id: str,
        geojson_path: str | Path,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build track YAML from a GeoJSON LineString feature."""
        points = self._load_geojson_points(geojson_path)
        return self.build_from_existing_seed(
            track_id=track_id, metadata=metadata, centerline=points, **kwargs
        )

    def build_from_osm(
        self,
        *,
        track_id: str,
        metadata: dict[str, Any],
        latitude: float,
        longitude: float,
        search_radius_m: int = 1200,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a raceway centerline via OSMnx and build a track YAML seed."""
        try:
            import osmnx as ox  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "OSMnx is not installed. Install optional dependency group 'geo' to build from OSM."
            ) from exc

        tags = {"highway": "raceway"}
        north = latitude + search_radius_m / 111_000.0
        south = latitude - search_radius_m / 111_000.0
        east = longitude + search_radius_m / 85_000.0
        west = longitude - search_radius_m / 85_000.0
        features = ox.features_from_bbox((north, south, east, west), tags=tags)
        if features.empty:
            raise ValueError(f"No OSM raceway features found near {track_id}")
        geometry = features.iloc[0].geometry
        coords = list(geometry.coords) if hasattr(geometry, "coords") else []
        if len(coords) < 3:
            raise ValueError(f"OSM geometry for {track_id} does not contain enough points")
        points = [TrackPoint(latitude=lat, longitude=lon) for lon, lat in coords]
        return self.build_from_existing_seed(
            track_id=track_id,
            metadata=metadata,
            centerline=points,
            sources=["osm_raceway", "generated_seed"],
            **kwargs,
        )

    def save_yaml(self, track_payload: dict[str, Any], path: str | Path | None = None) -> Path:
        """Persist one generated track YAML payload."""
        self._tracks_dir.mkdir(parents=True, exist_ok=True)
        target = (
            Path(path)
            if path is not None
            else self._tracks_dir / f"{track_payload['track_id']}.yaml"
        )
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(track_payload, handle, sort_keys=False)
        return target

    def _load_csv_points(self, csv_path: str | Path) -> list[TrackPoint]:
        points: list[TrackPoint] = []
        with open(csv_path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                latitude = self._float_from_keys(row, "latitude", "lat")
                longitude = self._float_from_keys(row, "longitude", "lon", "lng")
                x_m = self._float_from_keys(row, "x_m", "x", "x_centerline")
                y_m = self._float_from_keys(row, "y_m", "y", "y_centerline")
                if latitude is None or longitude is None:
                    if x_m is None or y_m is None:
                        raise ValueError(
                            "CSV centerline rows require either"
                            " latitude/longitude or x_m/y_m columns"
                        )
                points.append(
                    TrackPoint(
                        latitude=latitude,
                        longitude=longitude,
                        x_m=x_m,
                        y_m=y_m,
                        elevation_m=float(row["elevation_m"]) if row.get("elevation_m") else None,
                        width_m=float(row["width_m"]) if row.get("width_m") else None,
                    )
                )
        return points

    def _load_geojson_points(self, geojson_path: str | Path) -> list[TrackPoint]:
        import json

        with open(geojson_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        feature = payload["features"][0] if payload.get("type") == "FeatureCollection" else payload
        geometry = feature["geometry"]
        if geometry["type"] != "LineString":
            raise ValueError("GeoJSON input must contain a LineString")
        points = []
        for coordinate in geometry["coordinates"]:
            lon, lat = coordinate[:2]
            elev = coordinate[2] if len(coordinate) > 2 else None
            points.append(
                TrackPoint(
                    latitude=float(lat),
                    longitude=float(lon),
                    elevation_m=elev,
                )
            )
        return points

    def _cumulative_distances(self, points: list[TrackPoint]) -> list[float]:
        cumulative = [0.0]
        for index in range(1, len(points)):
            cumulative.append(cumulative[-1] + self._distance_m(points[index - 1], points[index]))
        return cumulative

    def _segment_centerline(
        self,
        track_id: str,
        points: list[TrackPoint],
        cumulative: list[float],
    ) -> list[dict[str, Any]]:
        if len(points) < 3:
            return []
        windows: list[tuple[int, int]] = []
        start = 0
        curvature_state = self._classify_curvature(points, 1)
        for index in range(2, len(points) - 1):
            next_state = self._classify_curvature(points, index)
            if next_state != curvature_state and cumulative[index] - cumulative[start] > 120.0:
                windows.append((start, index))
                start = index
                curvature_state = next_state
        windows.append((start, len(points) - 1))

        built_segments = []
        for segment_index, (start_idx, end_idx) in enumerate(windows, start=1):
            segment_points = points[start_idx : end_idx + 1]
            segment_length = cumulative[end_idx] - cumulative[start_idx]
            width = self._segment_width(segment_points)
            elevation_delta = (segment_points[-1].elevation_m or 0.0) - (
                segment_points[0].elevation_m or 0.0
            )
            kind = self._classify_window(points, start_idx, end_idx)
            radius = self._estimate_radius(segment_points) if kind != "straight" else None
            risk = self._default_risk(kind, segment_length, radius)
            built_segments.append(
                BuiltSegment(
                    segment_id=f"{track_id}_{segment_index:02d}",
                    name=f"Segment {segment_index:02d}",
                    segment_type=kind,
                    start_m=round(cumulative[start_idx], 3),
                    end_m=round(cumulative[end_idx], 3),
                    width_m=round(width, 2),
                    radius_m=round(radius, 2) if radius is not None else None,
                    elevation_delta_m=round(elevation_delta, 2),
                    overtaking_viability="high"
                    if kind == "straight" and segment_length > 450
                    else ("medium" if kind == "braking_zone" else "low"),
                    preferred_battle_zone=kind in {"straight", "braking_zone"}
                    and segment_length > 220,
                    primary_recharge_zone=kind in {"medium_corner", "slow_corner", "braking_zone"},
                    primary_boost_zone=kind == "straight" and segment_length > 300,
                    risk=risk,
                    surface={
                        "main_track": {"type": "asphalt", "grip_dry": 1.0, "grip_wet": 0.72},
                        "offline": {
                            "type": "asphalt",
                            "grip_dry": 0.9,
                            "grip_wet": 0.62,
                            "dirt_level": 0.2,
                            "marbles_level": 0.2,
                        },
                    },
                    runoff={
                        "outside": {
                            "type": "asphalt",
                            "width_m": max(8.0, width * 0.8),
                            "grip_dry": 0.82,
                            "grip_wet": 0.54,
                        }
                    },
                    track_limits=self._default_track_limits(kind),
                    metadata={
                        "source_builder": "geospatial_track_builder.v1",
                        "point_span": len(segment_points),
                    },
                ).to_dict()
            )
        return built_segments

    def _classify_window(self, points: list[TrackPoint], start_idx: int, end_idx: int) -> str:
        curvatures = [
            self._turn_angle_deg(points[index - 1], points[index], points[index + 1])
            for index in range(max(1, start_idx + 1), min(end_idx, len(points) - 2) + 1)
        ]
        if not curvatures:
            return "straight"
        avg_curvature = sum(abs(value) for value in curvatures) / len(curvatures)
        if avg_curvature < 3.0:
            return "straight"
        if avg_curvature < 8.0:
            return "fast_corner"
        if avg_curvature < 15.0:
            return "medium_corner"
        return "slow_corner"

    def _classify_curvature(self, points: list[TrackPoint], index: int) -> str:
        angle = abs(self._turn_angle_deg(points[index - 1], points[index], points[index + 1]))
        if angle < 3.0:
            return "straight"
        if angle < 8.0:
            return "fast_corner"
        if angle < 15.0:
            return "medium_corner"
        return "slow_corner"

    def _estimate_radius(self, points: list[TrackPoint]) -> float | None:
        if len(points) < 3:
            return None
        p1, p2, p3 = points[0], points[len(points) // 2], points[-1]
        a = self._distance_m(p1, p2)
        b = self._distance_m(p2, p3)
        c = self._distance_m(p1, p3)
        semi = (a + b + c) / 2.0
        area_term = semi * (semi - a) * (semi - b) * (semi - c)
        if area_term <= 0:
            return None
        area = math.sqrt(area_term)
        return (a * b * c) / max(4.0 * area, 1e-6)

    def _segment_width(self, points: list[TrackPoint]) -> float:
        widths = [point.width_m for point in points if point.width_m is not None]
        return sum(widths) / len(widths) if widths else 14.0

    def _default_risk(self, kind: str, length_m: float, radius_m: float | None) -> dict[str, Any]:
        threshold = 55.0 if kind == "straight" else 42.0
        if radius_m is not None and radius_m < 120:
            threshold -= 7.0
        return {
            "unsafe_closing_speed_threshold_kph": threshold,
            "side_by_side_risk": "medium" if kind in {"straight", "braking_zone"} else "high",
            "evasive_action_margin": "medium" if length_m > 250 else "low",
            "energy_delta_sensitivity": "high" if kind == "straight" else "medium",
            "active_aero_sensitivity": "high" if kind in {"straight", "fast_corner"} else "medium",
            "barrier_distance_m": 25.0 if kind == "straight" else 18.0,
            "impact_severity_multiplier": 1.0 if kind == "straight" else 1.2,
        }

    def _default_track_limits(self, kind: str) -> dict[str, Any] | None:
        if kind == "straight":
            return None
        return {
            "rule": "white_line",
            "allowed_wheels_out": 0,
            "detection_probability": 0.96,
            "warning_threshold": 3,
            "penalty_after": 4,
            "time_gain_sensitive": True,
            "estimated_gain_if_abused_s": 0.03 if kind == "slow_corner" else 0.015,
        }

    def _turn_angle_deg(
        self, prev_point: TrackPoint, point: TrackPoint, next_point: TrackPoint
    ) -> float:
        v1x, v1y = self._planar_xy(prev_point, point)
        v2x, v2y = self._planar_xy(point, next_point)
        dot = v1x * v2x + v1y * v2y
        det = v1x * v2y - v1y * v2x
        return math.degrees(math.atan2(det, dot))

    def _planar_xy(self, start: TrackPoint, end: TrackPoint) -> tuple[float, float]:
        if (
            start.x_m is not None
            and start.y_m is not None
            and end.x_m is not None
            and end.y_m is not None
        ):
            return (end.x_m - start.x_m, end.y_m - start.y_m)
        if (
            start.latitude is None
            or start.longitude is None
            or end.latitude is None
            or end.longitude is None
        ):
            raise ValueError(
                "Track points require either planar x/y or latitude/longitude coordinates"
            )
        lat_scale = 111_320.0
        lon_scale = 111_320.0 * math.cos(math.radians((start.latitude + end.latitude) / 2.0))
        return (
            (end.longitude - start.longitude) * lon_scale,
            (end.latitude - start.latitude) * lat_scale,
        )

    def _distance_m(self, start: TrackPoint, end: TrackPoint) -> float:
        if (
            start.x_m is not None
            and start.y_m is not None
            and end.x_m is not None
            and end.y_m is not None
        ):
            return math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        if (
            start.latitude is None
            or start.longitude is None
            or end.latitude is None
            or end.longitude is None
        ):
            raise ValueError(
                "Track points require either planar x/y or latitude/longitude coordinates"
            )
        radius = 6_371_000.0
        lat1 = math.radians(start.latitude)
        lat2 = math.radians(end.latitude)
        d_lat = lat2 - lat1
        d_lon = math.radians(end.longitude - start.longitude)
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
        return radius * c

    def _float_from_keys(self, row: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = row.get(key)
            if value in (None, ""):
                continue
            return float(value)
        return None
