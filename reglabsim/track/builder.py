"""Geospatial track builder for digital-twin YAML generation."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from reglabsim.data.openmeteo_client import OpenMeteoClient
from reglabsim.track.geometry import TrackModel

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
STREET_HIGHWAY_PATTERN = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
)
STREET_HIGHWAY_PRIORITY = {
    "motorway": 0.0,
    "trunk": 0.05,
    "primary": 0.1,
    "secondary": 0.15,
    "tertiary": 0.2,
    "unclassified": 0.3,
    "residential": 0.35,
    "living_street": 0.4,
    "service": 0.55,
}
MAX_STREET_COMPONENT_WAYS = 120
STREET_NODE_THRESHOLD_M = 20.0
STREET_CONNECT_THRESHOLD_M = 25.0


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


@dataclass(frozen=True)
class OSMWay:
    """One OSM raceway polyline plus tags and precomputed length."""

    way_id: int
    points: list[TrackPoint]
    tags: dict[str, Any]
    length_m: float


class GeospatialTrackBuilder:
    """Build track YAML from local geospatial seeds or optional OSM fetches."""

    def __init__(
        self,
        tracks_dir: str | Path = "configs/tracks",
        weather_client: OpenMeteoClient | None = None,
    ):
        self._tracks_dir = Path(tracks_dir)
        self._weather_client = weather_client

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
        enrich_elevation: bool = False,
    ) -> dict[str, Any]:
        """Build a track YAML payload from an explicit centerline."""
        if len(centerline) < 3:
            raise ValueError("Centerline requires at least 3 points")
        centerline = self._prepare_centerline(centerline, enrich_elevation=enrich_elevation)
        cumulative = self._cumulative_distances(centerline)
        total_length_m = cumulative[-1]
        segments = self._segment_centerline(track_id, centerline, cumulative)
        elevations = [point.elevation_m for point in centerline if point.elevation_m is not None]
        target_length_m = self._expected_length_from_metadata(metadata)
        coverage_ratio = (
            round(total_length_m / target_length_m, 4)
            if target_length_m is not None and target_length_m > 0.0
            else None
        )
        effective_validation_status = validation_status
        effective_notes = list(fidelity_notes or ["Generated from centerline heuristics."])
        if coverage_ratio is not None and coverage_ratio < 0.5:
            effective_validation_status = "generated_seed_low_coverage"
            effective_notes.append(
                f"Centerline coverage ratio {coverage_ratio:.3f} below target length."
            )
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
            "validation_status": effective_validation_status,
            "fidelity_notes": effective_notes,
            "metadata": {
                **metadata,
                "builder": "geospatial_track_builder.v1",
                "centerline_points": len(centerline),
                "elevation_source": (
                    "openmeteo_elevation" if enrich_elevation else metadata.get("elevation_source")
                ),
                "elevation_min_m": min(elevations) if elevations else None,
                "elevation_max_m": max(elevations) if elevations else None,
                "length_coverage_ratio": coverage_ratio,
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
        expected_length_m = self._expected_length_from_metadata(metadata)
        try:
            import osmnx as ox

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
            osm_metadata = {"osm_source": "osmnx", "osm_feature_count": len(features)}
        except ImportError:
            points, osm_metadata = self._fetch_osm_points_overpass(
                latitude=latitude,
                longitude=longitude,
                search_radius_m=search_radius_m,
                expected_length_m=expected_length_m,
            )
        return self.build_from_existing_seed(
            track_id=track_id,
            metadata={**metadata, **osm_metadata},
            centerline=points,
            sources=["osm_raceway", "generated_seed", "openmeteo_elevation"],
            enrich_elevation=True,
            **kwargs,
        )

    def build_from_osm_street(
        self,
        *,
        track_id: str,
        metadata: dict[str, Any],
        latitude: float,
        longitude: float,
        search_radius_m: int = 1600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a street-network centerline via Overpass and build a track YAML seed."""
        expected_length_m = self._expected_length_from_metadata(metadata)
        points, osm_metadata = self._fetch_osm_points_overpass(
            latitude=latitude,
            longitude=longitude,
            search_radius_m=search_radius_m,
            expected_length_m=expected_length_m,
            street_mode=True,
        )
        return self.build_from_existing_seed(
            track_id=track_id,
            metadata={**metadata, **osm_metadata},
            centerline=points,
            sources=["osm_street_network", "generated_seed", "openmeteo_elevation"],
            enrich_elevation=True,
            **kwargs,
        )

    def build_from_overpass_payload(
        self,
        *,
        track_id: str,
        metadata: dict[str, Any],
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a track seed from a raw Overpass JSON payload."""
        points, osm_metadata = self._points_from_overpass_payload(
            payload,
            expected_length_m=self._expected_length_from_metadata(metadata),
        )
        return self.build_from_existing_seed(
            track_id=track_id,
            metadata={**metadata, **osm_metadata},
            centerline=points,
            sources=["osm_overpass_payload", "generated_seed", "openmeteo_elevation"],
            enrich_elevation=True,
            **kwargs,
        )

    def build_from_street_overpass_payload(
        self,
        *,
        track_id: str,
        metadata: dict[str, Any],
        payload: dict[str, Any],
        latitude: float,
        longitude: float,
        search_radius_m: int = 1600,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a street-circuit seed from raw Overpass JSON road payload."""
        points, osm_metadata = self._points_from_street_overpass_payload(
            payload,
            expected_length_m=self._expected_length_from_metadata(metadata),
            center_latitude=latitude,
            center_longitude=longitude,
            search_radius_m=search_radius_m,
        )
        return self.build_from_existing_seed(
            track_id=track_id,
            metadata={**metadata, **osm_metadata},
            centerline=points,
            sources=["osm_street_network", "generated_seed", "openmeteo_elevation"],
            enrich_elevation=True,
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

    def build_from_track_model(
        self,
        track: TrackModel,
        *,
        validation_status: str = "fallback_curated_track_model",
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        """Export an existing curated track model as a builder payload fallback."""
        metadata = dict(track.metadata)
        metadata["builder"] = "geospatial_track_builder.curated_fallback.v1"
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        notes = list(track.fidelity_notes)
        if fallback_reason:
            notes.append(fallback_reason)
        return {
            "track_id": track.track_id,
            "name": track.name,
            "country": track.country,
            "length_m": round(track.length_m, 3),
            "turns": track.turns,
            "laps": track.laps,
            "race_distance_m": round(track.race_distance_m, 3),
            "avg_speed_kph": round(track.avg_speed_kph, 3),
            "fidelity_level": track.fidelity_level,
            "sources": list(dict.fromkeys([*track.sources, "curated_track_model_fallback"])),
            "validation_status": validation_status,
            "fidelity_notes": notes,
            "metadata": metadata,
            "segments": [self._segment_to_payload(segment) for segment in track.segments],
        }

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

    def _prepare_centerline(
        self,
        centerline: list[TrackPoint],
        *,
        enrich_elevation: bool,
    ) -> list[TrackPoint]:
        if not enrich_elevation:
            return centerline
        if any(point.latitude is None or point.longitude is None for point in centerline):
            return centerline
        if all(point.elevation_m is not None for point in centerline):
            return centerline
        coordinates: list[tuple[float, float]] = []
        for point in centerline:
            if point.latitude is None or point.longitude is None:
                return centerline
            coordinates.append((point.latitude, point.longitude))
        elevations = self._elevation_client().fetch_elevation_profile(coordinates=coordinates)
        return [
            TrackPoint(
                latitude=point.latitude,
                longitude=point.longitude,
                x_m=point.x_m,
                y_m=point.y_m,
                elevation_m=elevation if point.elevation_m is None else point.elevation_m,
                width_m=point.width_m,
            )
            for point, elevation in zip(centerline, elevations, strict=True)
        ]

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
                    overtaking_viability=(
                        "high"
                        if kind == "straight" and segment_length > 450
                        else ("medium" if kind == "braking_zone" else "low")
                    ),
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
            if isinstance(value, (int, float, str)):
                return float(value)
        return None

    def _elevation_client(self) -> OpenMeteoClient:
        if self._weather_client is None:
            self._weather_client = OpenMeteoClient()
            self._weather_client.connect()
        return self._weather_client

    def _fetch_osm_points_overpass(
        self,
        *,
        latitude: float,
        longitude: float,
        search_radius_m: int,
        expected_length_m: float | None,
        street_mode: bool = False,
    ) -> tuple[list[TrackPoint], dict[str, Any]]:
        query = (
            "[out:json][timeout:25];("
            + (
                f'way["highway"~"{STREET_HIGHWAY_PATTERN}"](around:{search_radius_m},{latitude},{longitude});'
                if street_mode
                else (
                    f'way["highway"="raceway"](around:{search_radius_m},{latitude},{longitude});'
                    f'relation["highway"="raceway"](around:{search_radius_m},{latitude},{longitude});'
                )
            )
            + ");out geom tags;"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            request = Request(
                OVERPASS_URL,
                data=query.encode("utf-8"),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "f1lab-ai/0.2",
                },
            )
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if street_mode:
                    return self._points_from_street_overpass_payload(
                        payload,
                        expected_length_m=expected_length_m,
                        center_latitude=latitude,
                        center_longitude=longitude,
                        search_radius_m=search_radius_m,
                    )
                return self._points_from_overpass_payload(
                    payload,
                    expected_length_m=expected_length_m,
                )
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise ValueError(
                    f"Overpass fetch failed near lat={latitude} lon={longitude}: {exc}"
                ) from exc
        raise ValueError(f"Overpass fetch failed near lat={latitude} lon={longitude}: {last_error}")

    def _points_from_overpass_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_length_m: float | None,
    ) -> tuple[list[TrackPoint], dict[str, Any]]:
        elements = payload.get("elements", [])
        if not elements:
            raise ValueError("Overpass payload does not contain raceway elements")
        ways = [
            element
            for element in elements
            if element.get("type") == "way" and element.get("geometry")
        ]
        if not ways:
            raise ValueError("Overpass payload does not contain usable way geometries")
        way_entries: list[OSMWay] = []
        widths: list[float] = []
        source_way_ids: list[int] = []
        for element in ways:
            tags = dict(element.get("tags", {}))
            width = self._parse_width_tag(tags.get("width"))
            if width is not None:
                widths.append(width)
            source_way_ids.append(int(element["id"]))
            polyline = [
                TrackPoint(
                    latitude=float(point["lat"]),
                    longitude=float(point["lon"]),
                    width_m=width,
                )
                for point in element.get("geometry", [])
            ]
            if len(polyline) >= 2:
                way_entries.append(
                    OSMWay(
                        way_id=int(element["id"]),
                        points=polyline,
                        tags=tags,
                        length_m=self._polyline_length_m(polyline),
                    )
                )
        components = self._build_osm_components(way_entries)
        selected = self._select_osm_component(components, expected_length_m=expected_length_m)
        mainline = [way for way in selected if not self._is_auxiliary_way(way.tags)] or selected
        merged = self._assemble_osm_component(mainline)
        metadata = {
            "osm_source": "overpass",
            "osm_way_count": len(way_entries),
            "osm_width_samples": len(widths),
            "osm_mean_width_m": round(sum(widths) / len(widths), 3) if widths else None,
            "osm_way_ids": source_way_ids[:20],
            "osm_component_way_count": len(mainline),
            "osm_component_length_m": round(sum(way.length_m for way in mainline), 3),
            "osm_target_length_m": expected_length_m,
            "osm_auxiliary_way_count": len(selected) - len(mainline),
            "osm_auxiliary_way_ids": [
                way.way_id for way in selected if self._is_auxiliary_way(way.tags)
            ][:20],
        }
        return merged, metadata

    def _points_from_street_overpass_payload(
        self,
        payload: dict[str, Any],
        *,
        expected_length_m: float | None,
        center_latitude: float,
        center_longitude: float,
        search_radius_m: int,
    ) -> tuple[list[TrackPoint], dict[str, Any]]:
        elements = payload.get("elements", [])
        if not elements:
            raise ValueError("Overpass payload does not contain street-network elements")
        center = TrackPoint(latitude=center_latitude, longitude=center_longitude)
        way_entries: list[OSMWay] = []
        widths: list[float] = []
        source_way_ids: list[int] = []
        for element in elements:
            if element.get("type") != "way" or not element.get("geometry"):
                continue
            tags = dict(element.get("tags", {}))
            if not self._is_street_candidate(tags):
                continue
            width = self._parse_width_tag(tags.get("width"))
            if width is not None:
                widths.append(width)
            raw_polyline = [
                TrackPoint(
                    latitude=float(point["lat"]),
                    longitude=float(point["lon"]),
                    width_m=width,
                )
                for point in element.get("geometry", [])
            ]
            polyline = self._clip_polyline_to_radius(
                raw_polyline,
                center=center,
                max_radius_m=search_radius_m * 1.2,
            )
            if len(polyline) < 2:
                continue
            source_way_ids.append(int(element["id"]))
            way_entries.append(
                OSMWay(
                    way_id=int(element["id"]),
                    points=polyline,
                    tags=tags,
                    length_m=self._polyline_length_m(polyline),
                )
            )
        if not way_entries:
            raise ValueError("Street-network payload does not contain usable road geometries")

        filtered = self._select_street_way_subset(
            way_entries,
            center=center,
            search_radius_m=search_radius_m,
            expected_length_m=expected_length_m,
        )
        components = self._build_osm_components(
            filtered,
            distance_threshold_m=STREET_CONNECT_THRESHOLD_M,
        )
        selected = self._select_street_component(
            components,
            center=center,
            expected_length_m=expected_length_m,
        )
        cycle = self._find_best_osm_cycle(
            selected,
            expected_length_m=expected_length_m,
            node_threshold_m=STREET_NODE_THRESHOLD_M,
        )
        if cycle:
            merged = self._flatten_ordered_edges(cycle)
            cycle_length_m = sum(way.length_m for _, _, way in cycle)
            cycle_way_count = len(cycle)
        else:
            merged = self._assemble_osm_component(
                selected,
                node_threshold_m=STREET_NODE_THRESHOLD_M,
            )
            cycle_length_m = self._polyline_length_m(merged)
            cycle_way_count = len(selected)
        merged = self._downsample_points(merged)
        metadata = {
            "osm_source": "overpass_street_network",
            "osm_way_count": len(way_entries),
            "osm_street_candidate_count": len(filtered),
            "osm_width_samples": len(widths),
            "osm_mean_width_m": round(sum(widths) / len(widths), 3) if widths else None,
            "osm_way_ids": source_way_ids[:20],
            "osm_component_way_count": len(selected),
            "osm_component_length_m": round(sum(way.length_m for way in selected), 3),
            "osm_cycle_way_count": cycle_way_count,
            "osm_cycle_length_m": round(cycle_length_m, 3),
            "osm_target_length_m": expected_length_m,
            "osm_search_kind": "street_network",
            "osm_auxiliary_way_count": sum(
                1 for way in selected if self._is_auxiliary_way(way.tags)
            ),
        }
        return merged, metadata

    def _build_osm_components(
        self,
        ways: list[OSMWay],
        distance_threshold_m: float = 80.0,
    ) -> list[list[OSMWay]]:
        if not ways:
            raise ValueError("No OSM ways available")
        adjacency: dict[int, set[int]] = {index: set() for index in range(len(ways))}
        endpoints = [(way.points[0], way.points[-1]) for way in ways]
        for left in range(len(ways)):
            for right in range(left + 1, len(ways)):
                distance = min(
                    self._distance_m(point_a, point_b)
                    for point_a in endpoints[left]
                    for point_b in endpoints[right]
                )
                if distance <= distance_threshold_m:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        seen: set[int] = set()
        components: list[list[OSMWay]] = []
        for index in range(len(ways)):
            if index in seen:
                continue
            stack = [index]
            seen.add(index)
            component: list[OSMWay] = []
            while stack:
                current = stack.pop()
                component.append(ways[current])
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    def _select_osm_component(
        self,
        components: list[list[OSMWay]],
        *,
        expected_length_m: float | None,
    ) -> list[OSMWay]:
        scored: list[tuple[float, list[OSMWay]]] = []
        for component in components:
            mainline = [way for way in component if not self._is_auxiliary_way(way.tags)]
            if not mainline:
                mainline = component
            length_m = sum(way.length_m for way in mainline)
            if expected_length_m is None or expected_length_m <= 0.0:
                score = -length_m
            else:
                relative_error = abs(length_m - expected_length_m) / expected_length_m
                score = relative_error - min(len(mainline), 200) / 10_000.0
            scored.append((score, component))
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _select_street_way_subset(
        self,
        ways: list[OSMWay],
        *,
        center: TrackPoint,
        search_radius_m: int,
        expected_length_m: float | None,
    ) -> list[OSMWay]:
        scored: list[tuple[float, OSMWay]] = []
        for way in ways:
            centroid = self._polyline_centroid(way.points)
            center_distance = self._distance_m(center, centroid)
            if center_distance > search_radius_m * 1.3:
                continue
            priority = self._street_way_priority(way.tags)
            score = center_distance + priority * 250.0
            if self._is_auxiliary_way(way.tags):
                score += 75.0
            scored.append((score, way))
        scored.sort(key=lambda item: (item[0], -item[1].length_m))
        if not scored:
            raise ValueError("Street-network builder could not score any usable ways")
        selected: list[OSMWay] = []
        accumulated_length = 0.0
        target_budget = (
            expected_length_m * 1.8
            if expected_length_m is not None and expected_length_m > 0.0
            else 8_000.0
        )
        for _, way in scored[:MAX_STREET_COMPONENT_WAYS]:
            selected.append(way)
            accumulated_length += way.length_m
            if accumulated_length >= target_budget and len(selected) >= 12:
                break
        return selected

    def _select_street_component(
        self,
        components: list[list[OSMWay]],
        *,
        center: TrackPoint,
        expected_length_m: float | None,
    ) -> list[OSMWay]:
        scored: list[tuple[float, list[OSMWay]]] = []
        for component in components:
            length_m = sum(way.length_m for way in component)
            centroid_distances = [
                self._distance_m(center, self._polyline_centroid(way.points)) for way in component
            ]
            mean_center_distance = (
                sum(centroid_distances) / len(centroid_distances) if centroid_distances else 9999.0
            )
            if expected_length_m is None or expected_length_m <= 0.0:
                relative_error = 0.0
            else:
                relative_error = abs(length_m - expected_length_m) / expected_length_m
            score = (
                relative_error
                + (mean_center_distance / 1_000.0)
                - min(len(component), 200) / 20_000.0
            )
            scored.append((score, component))
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _assemble_osm_component(
        self,
        ways: list[OSMWay],
        node_threshold_m: float = 35.0,
    ) -> list[TrackPoint]:
        if not ways:
            raise ValueError("No OSM ways available for assembly")
        edges, adjacency = self._build_osm_edge_graph(
            ways,
            node_threshold_m=node_threshold_m,
        )

        degree_one_nodes = [node for node, edge_ids in adjacency.items() if len(edge_ids) == 1]
        if degree_one_nodes:
            start_node = degree_one_nodes[0]
            start_edge = adjacency[start_node][0]
        else:
            start_edge = max(range(len(edges)), key=lambda index: edges[index][2].length_m)
            start_node = edges[start_edge][0]

        ordered = self._walk_osm_edges(
            edges, adjacency, start_node=start_node, start_edge=start_edge
        )
        used_edge_count = len({way.way_id for _, _, way in ordered})
        if used_edge_count < len(ways):
            remaining = [
                way.points
                for _, _, way in edges
                if way.way_id not in {edge_way.way_id for _, _, edge_way in ordered}
            ]
            stitched = self._append_remaining_polylines(
                self._flatten_ordered_edges(ordered),
                remaining,
            )
        else:
            stitched = self._flatten_ordered_edges(ordered)

        if len(stitched) > 3 and self._distance_m(stitched[0], stitched[-1]) > 25.0:
            stitched.append(stitched[0])
        return self._downsample_points(stitched)

    def _build_osm_edge_graph(
        self,
        ways: list[OSMWay],
        *,
        node_threshold_m: float,
    ) -> tuple[list[tuple[int, int, OSMWay]], dict[int, list[int]]]:
        node_refs: list[TrackPoint] = []

        def assign_node(point: TrackPoint) -> int:
            for node_index, reference in enumerate(node_refs):
                if self._distance_m(point, reference) <= node_threshold_m:
                    return node_index
            node_refs.append(point)
            return len(node_refs) - 1

        edges: list[tuple[int, int, OSMWay]] = []
        adjacency: dict[int, list[int]] = {}
        for edge_index, way in enumerate(ways):
            start_node = assign_node(way.points[0])
            end_node = assign_node(way.points[-1])
            edges.append((start_node, end_node, way))
            adjacency.setdefault(start_node, []).append(edge_index)
            adjacency.setdefault(end_node, []).append(edge_index)
        return edges, adjacency

    def _find_best_osm_cycle(
        self,
        ways: list[OSMWay],
        *,
        expected_length_m: float | None,
        node_threshold_m: float,
    ) -> list[tuple[int, bool, OSMWay]] | None:
        if not ways or expected_length_m is None or expected_length_m <= 0.0:
            return None
        edges, adjacency = self._build_osm_edge_graph(ways, node_threshold_m=node_threshold_m)
        start_edges = sorted(
            range(len(edges)),
            key=lambda index: edges[index][2].length_m,
            reverse=True,
        )[: min(len(edges), 20)]
        best_ordered: list[tuple[int, bool, OSMWay]] | None = None
        best_score = float("inf")
        explored_states = 0
        max_states = 4000
        max_depth = min(max(len(edges) + 6, 8), 60)

        def explore(
            start_node: int,
            current_node: int,
            length_m: float,
            ordered: list[tuple[int, bool, OSMWay]],
            used: set[int],
        ) -> None:
            nonlocal best_ordered, best_score, explored_states
            explored_states += 1
            if explored_states > max_states or len(ordered) > max_depth:
                return
            if (
                current_node == start_node
                and len(ordered) >= 3
                and length_m >= expected_length_m * 0.55
            ):
                score = abs(length_m - expected_length_m) / expected_length_m
                if score < best_score:
                    best_score = score
                    best_ordered = list(ordered)
                return
            if length_m > expected_length_m * 1.35:
                return
            candidates: list[tuple[float, int, bool, int, OSMWay]] = []
            for edge_id in adjacency.get(current_node, []):
                if edge_id in used:
                    continue
                edge_start, edge_end, way = edges[edge_id]
                forward = edge_start == current_node
                next_node = edge_end if forward else edge_start
                projected = length_m + way.length_m
                candidates.append(
                    (
                        abs(expected_length_m - projected),
                        edge_id,
                        forward,
                        next_node,
                        way,
                    )
                )
            candidates.sort(key=lambda item: (item[0], -item[4].length_m))
            for _, edge_id, forward, next_node, way in candidates[:4]:
                ordered.append((edge_id, forward, way))
                used.add(edge_id)
                explore(start_node, next_node, length_m + way.length_m, ordered, used)
                used.remove(edge_id)
                ordered.pop()

        for start_edge in start_edges:
            edge_start, edge_end, way = edges[start_edge]
            for start_node in (edge_start, edge_end):
                forward = edge_start == start_node
                next_node = edge_end if forward else edge_start
                explore(
                    start_node,
                    next_node,
                    way.length_m,
                    [(start_edge, forward, way)],
                    {start_edge},
                )
        return best_ordered

    def _walk_osm_edges(
        self,
        edges: list[tuple[int, int, OSMWay]],
        adjacency: dict[int, list[int]],
        *,
        start_node: int,
        start_edge: int,
    ) -> list[tuple[int, bool, OSMWay]]:
        ordered: list[tuple[int, bool, OSMWay]] = []
        used: set[int] = set()
        current_node = start_node
        current_edge = start_edge

        while current_edge not in used:
            edge_start, edge_end, way = edges[current_edge]
            forward = edge_start == current_node
            ordered.append((current_edge, forward, way))
            used.add(current_edge)
            current_node = edge_end if forward else edge_start
            candidates = [
                edge_id for edge_id in adjacency.get(current_node, []) if edge_id not in used
            ]
            if not candidates:
                break
            current_edge = max(candidates, key=lambda edge_id: edges[edge_id][2].length_m)

        return ordered

    def _flatten_ordered_edges(
        self,
        ordered: list[tuple[int, bool, OSMWay]],
    ) -> list[TrackPoint]:
        stitched: list[TrackPoint] = []
        for _, forward, way in ordered:
            points = way.points if forward else list(reversed(way.points))
            if not stitched:
                stitched.extend(points)
            else:
                stitched.extend(points[1:])
        return stitched

    def _append_remaining_polylines(
        self,
        stitched: list[TrackPoint],
        remaining: list[list[TrackPoint]],
        distance_threshold_m: float = 80.0,
    ) -> list[TrackPoint]:
        pending = [list(polyline) for polyline in remaining if len(polyline) >= 2]
        while pending:
            best_index: int | None = None
            best_distance = float("inf")
            best_mode = ""
            for index, candidate in enumerate(pending):
                modes = {
                    "append_forward": self._distance_m(stitched[-1], candidate[0]),
                    "append_reverse": self._distance_m(stitched[-1], candidate[-1]),
                    "prepend_forward": self._distance_m(candidate[-1], stitched[0]),
                    "prepend_reverse": self._distance_m(candidate[0], stitched[0]),
                }
                mode, distance = min(modes.items(), key=lambda item: item[1])
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
                    best_mode = mode
            if best_index is None or best_distance > distance_threshold_m:
                break
            candidate = pending.pop(best_index)
            if best_mode == "append_forward":
                stitched.extend(candidate[1:])
            elif best_mode == "append_reverse":
                stitched.extend(list(reversed(candidate))[1:])
            elif best_mode == "prepend_forward":
                stitched = candidate[:-1] + stitched
            else:
                stitched = list(reversed(candidate))[:-1] + stitched
        return stitched

    def _downsample_points(
        self, points: list[TrackPoint], target_points: int = 220
    ) -> list[TrackPoint]:
        if len(points) <= target_points:
            return points
        step = max(1, len(points) // target_points)
        reduced = [point for index, point in enumerate(points) if index % step == 0]
        if reduced[-1] != points[-1]:
            reduced.append(points[-1])
        return reduced

    def _parse_width_tag(self, width_tag: Any) -> float | None:
        if width_tag in (None, ""):
            return None
        try:
            return float(str(width_tag).replace("m", "").strip())
        except ValueError:
            return None

    def _polyline_length_m(self, points: list[TrackPoint]) -> float:
        return sum(self._distance_m(left, right) for left, right in pairwise(points))

    def _segment_to_payload(self, segment: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": segment.segment_id,
            "name": segment.name,
            "type": segment.segment_type,
            "start_m": segment.start_m,
            "end_m": segment.end_m,
            "width_m": segment.width_m,
            "radius_m": segment.radius_m,
            "elevation_delta_m": segment.elevation_delta_m,
            "overtaking_viability": segment.overtaking_viability,
            "preferred_battle_zone": segment.preferred_battle_zone,
            "primary_recharge_zone": segment.primary_recharge_zone,
            "primary_boost_zone": segment.primary_boost_zone,
            "surface": {
                "main_track": self._surface_to_payload(segment.main_surface),
                "racing_line": self._surface_to_payload(segment.racing_line_surface),
                "offline": self._surface_to_payload(segment.offline_surface),
            },
            "runoff": {"outside": self._runoff_to_payload(segment.runoff)},
            "risk": self._risk_to_payload(segment.risk),
            "metadata": dict(segment.metadata),
        }
        kerbs: dict[str, Any] = {}
        if segment.inside_kerb is not None:
            kerbs["inside"] = self._kerb_to_payload(segment.inside_kerb)
        if segment.outside_kerb is not None:
            kerbs["outside"] = self._kerb_to_payload(segment.outside_kerb)
        if kerbs:
            payload["kerbs"] = kerbs
        if segment.track_limits is not None:
            payload["track_limits"] = self._track_limits_to_payload(segment.track_limits)
        return payload

    def _surface_to_payload(self, surface: Any) -> dict[str, Any]:
        return {
            "type": surface.type,
            "grip_dry": surface.grip_dry,
            "grip_wet": surface.grip_wet,
            "roughness": surface.roughness,
            "drainage": surface.drainage,
            "dirt_level": surface.dirt_level,
            "marbles_level": surface.marbles_level,
        }

    def _runoff_to_payload(self, runoff: Any) -> dict[str, Any]:
        return {
            "type": runoff.type,
            "width_m": runoff.width_m,
            "grip_dry": runoff.grip_dry,
            "grip_wet": runoff.grip_wet,
            "rejoin_risk": runoff.rejoin_risk,
            "recovery_probability": runoff.recovery_probability,
        }

    def _risk_to_payload(self, risk: Any) -> dict[str, Any]:
        return {
            "unsafe_closing_speed_threshold_kph": risk.unsafe_closing_speed_threshold_kph,
            "side_by_side_risk": risk.side_by_side_risk,
            "evasive_action_margin": risk.evasive_action_margin,
            "energy_delta_sensitivity": risk.energy_delta_sensitivity,
            "active_aero_sensitivity": risk.active_aero_sensitivity,
            "visibility_risk": risk.visibility_risk,
            "barrier_distance_m": risk.barrier_distance_m,
            "impact_severity_multiplier": risk.impact_severity_multiplier,
        }

    def _kerb_to_payload(self, kerb: Any) -> dict[str, Any]:
        return {
            "type": kerb.type,
            "height_mm": kerb.height_mm,
            "width_m": kerb.width_m,
            "grip_dry": kerb.grip_dry,
            "grip_wet": kerb.grip_wet,
            "destabilization_factor": kerb.destabilization_factor,
            "bottoming_risk": kerb.bottoming_risk,
            "launch_risk": kerb.launch_risk,
            "track_limits_sensitive": kerb.track_limits_sensitive,
        }

    def _track_limits_to_payload(self, limits: Any) -> dict[str, Any]:
        return {
            "rule": limits.rule,
            "allowed_wheels_out": limits.allowed_wheels_out,
            "detection_probability": limits.detection_probability,
            "warning_threshold": limits.warning_threshold,
            "penalty_after": limits.penalty_after,
            "time_gain_sensitive": limits.time_gain_sensitive,
            "estimated_gain_if_abused_s": limits.estimated_gain_if_abused_s,
        }

    def _polyline_centroid(self, points: list[TrackPoint]) -> TrackPoint:
        if not points:
            raise ValueError("Polyline requires at least one point")
        if all(point.latitude is not None and point.longitude is not None for point in points):
            return TrackPoint(
                latitude=sum(point.latitude for point in points if point.latitude is not None)
                / len(points),
                longitude=sum(point.longitude for point in points if point.longitude is not None)
                / len(points),
            )
        if all(point.x_m is not None and point.y_m is not None for point in points):
            return TrackPoint(
                x_m=sum(point.x_m for point in points if point.x_m is not None) / len(points),
                y_m=sum(point.y_m for point in points if point.y_m is not None) / len(points),
            )
        return points[len(points) // 2]

    def _clip_polyline_to_radius(
        self,
        points: list[TrackPoint],
        *,
        center: TrackPoint,
        max_radius_m: float,
    ) -> list[TrackPoint]:
        kept = [point for point in points if self._distance_m(center, point) <= max_radius_m]
        if len(kept) >= 2:
            return kept
        return points

    def _street_way_priority(self, tags: dict[str, Any]) -> float:
        highway = str(tags.get("highway", "")).strip().lower()
        return STREET_HIGHWAY_PRIORITY.get(highway, 1.0)

    def _is_street_candidate(self, tags: dict[str, Any]) -> bool:
        highway = str(tags.get("highway", "")).strip().lower()
        if highway not in STREET_HIGHWAY_PRIORITY:
            return False
        access = str(tags.get("access", "")).strip().lower()
        if access == "private":
            return False
        service = str(tags.get("service", "")).strip().lower()
        if service in {"parking_aisle", "driveway", "alley"}:
            return False
        if str(tags.get("area", "")).strip().lower() == "yes":
            return False
        return True

    def _is_auxiliary_way(self, tags: dict[str, Any]) -> bool:
        name = str(tags.get("name", "")).strip().lower()
        highway = str(tags.get("highway", "")).strip().lower()
        service = str(tags.get("service", "")).strip().lower()
        return (
            "pit lane" in name
            or service == "pit_lane"
            or service in {"parking_aisle", "driveway", "alley"}
            or (highway in {"living_street"} and "circuit" not in name)
        )

    def _expected_length_from_metadata(self, metadata: dict[str, Any]) -> float | None:
        candidate = metadata.get("target_length_m")
        if candidate in (None, ""):
            return None
        try:
            if isinstance(candidate, (int, float, str)):
                return float(candidate)
            return None
        except (TypeError, ValueError):
            return None
