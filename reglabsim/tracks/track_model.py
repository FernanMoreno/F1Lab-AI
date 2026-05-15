"""Track segment and model dataclasses for fidelity metadata (PR 8.4.1).

These are fidelity-layer models — they represent what we KNOW about a track
abstraction, including uncertainty and provenance. They do not contain raw
geometry blobs, coordinate arrays, or shapefiles.

No real track names are hardcoded. No track-specific special-casing.
Does NOT modify runtime physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reglabsim.tracks.fidelity import (
    FIDELITY_TIER_PUBLIC_APPROX,
    FIDELITY_TIER_SYNTHETIC,
    TRACK_MODEL_SCHEMA,
    TRACK_SEGMENT_SCHEMA,
)

# ---------------------------------------------------------------------------
# Allowed value sets
# ---------------------------------------------------------------------------

_ALLOWED_SEGMENT_TYPES = frozenset({
    "straight",
    "fast_corner",
    "medium_corner",
    "slow_corner",
    "corner",
    "chicane",
    "hairpin",
    "pit_entry",
    "pit_exit",
    "start_finish",
    "braking_zone",
    "unknown",
})

_ALLOWED_RUNOFF_TYPES = frozenset({
    "asphalt",
    "grass",
    "gravel",
    "wall",
    "barrier",
    "concrete",
    "mixed",
    "unknown",
})

_ALLOWED_DATA_CLASSIFICATIONS = frozenset({
    "synthetic",
    "public",
    "inferred",
    "calibrated",
    "mixed",
    "unknown",
})


# ---------------------------------------------------------------------------
# TrackSegmentModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackSegmentModel:
    """One segment in a fidelity-layer track model.

    All fields except segment_id, name, and segment_type are optional
    to allow partial abstractions. Missing fields indicate knowledge gaps.
    """

    segment_id: str
    name: str | None
    segment_type: str
    length_m: float | None = None
    start_distance_m: float | None = None
    end_distance_m: float | None = None
    curvature_radius_m: float | None = None
    curvature_direction: str | None = None
    width_m: float | None = None
    barrier_distance_m: float | None = None
    runoff_type: str | None = None
    runoff_risk: float | None = None
    surface_type: str | None = None
    elevation_delta_m: float | None = None
    camber_deg: float | None = None
    sightline_distance_m: float | None = None
    drs_zone: bool = False
    overtaking_zone: bool = False
    source_tags: list[str] = field(default_factory=list)
    uncertainty: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrackModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackModel:
    """Fidelity-layer track model.

    Represents what is known about a circuit abstraction, including tier,
    provenance, known gaps, and limitations. Not a physics simulation model.
    """

    schema_version: str
    track_id: str
    display_name: str
    fidelity_tier: str
    provenance: list[str]
    data_classification: str
    length_m: float | None
    segment_count: int
    segments: list[TrackSegmentModel]
    known_gaps: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def track_segment_to_dict(segment: TrackSegmentModel) -> dict[str, Any]:
    """Serialize a TrackSegmentModel to a JSON-compatible dict."""
    return {
        "schema_version": TRACK_SEGMENT_SCHEMA,
        "segment_id": segment.segment_id,
        "name": segment.name,
        "segment_type": segment.segment_type,
        "length_m": segment.length_m,
        "start_distance_m": segment.start_distance_m,
        "end_distance_m": segment.end_distance_m,
        "curvature_radius_m": segment.curvature_radius_m,
        "curvature_direction": segment.curvature_direction,
        "width_m": segment.width_m,
        "barrier_distance_m": segment.barrier_distance_m,
        "runoff_type": segment.runoff_type,
        "runoff_risk": segment.runoff_risk,
        "surface_type": segment.surface_type,
        "elevation_delta_m": segment.elevation_delta_m,
        "camber_deg": segment.camber_deg,
        "sightline_distance_m": segment.sightline_distance_m,
        "drs_zone": segment.drs_zone,
        "overtaking_zone": segment.overtaking_zone,
        "source_tags": list(segment.source_tags),
        "uncertainty": dict(segment.uncertainty),
    }


def track_model_to_dict(track: TrackModel) -> dict[str, Any]:
    """Serialize a TrackModel to a JSON-compatible dict."""
    return {
        "schema_version": track.schema_version,
        "track_id": track.track_id,
        "display_name": track.display_name,
        "fidelity_tier": track.fidelity_tier,
        "provenance": list(track.provenance),
        "data_classification": track.data_classification,
        "length_m": track.length_m,
        "segment_count": track.segment_count,
        "segments": [track_segment_to_dict(s) for s in track.segments],
        "known_gaps": list(track.known_gaps),
        "limitations": list(track.limitations),
    }


def track_model_from_dict(data: dict[str, Any]) -> TrackModel:
    """Deserialize a TrackModel from a dict."""
    raw_segments = data.get("segments") or []
    segments = [_segment_from_dict(s) for s in raw_segments]
    return TrackModel(
        schema_version=str(data.get("schema_version") or TRACK_MODEL_SCHEMA),
        track_id=str(data.get("track_id") or "unknown"),
        display_name=str(data.get("display_name") or ""),
        fidelity_tier=str(data.get("fidelity_tier") or FIDELITY_TIER_SYNTHETIC),
        provenance=list(data.get("provenance") or []),
        data_classification=str(data.get("data_classification") or "synthetic"),
        length_m=_optional_float(data.get("length_m")),
        segment_count=int(data.get("segment_count") or len(segments)),
        segments=segments,
        known_gaps=list(data.get("known_gaps") or []),
        limitations=list(data.get("limitations") or []),
    )


def _segment_from_dict(data: dict[str, Any]) -> TrackSegmentModel:
    return TrackSegmentModel(
        segment_id=str(data.get("segment_id") or "unknown"),
        name=data.get("name"),
        segment_type=str(data.get("segment_type") or "unknown"),
        length_m=_optional_float(data.get("length_m")),
        start_distance_m=_optional_float(data.get("start_distance_m")),
        end_distance_m=_optional_float(data.get("end_distance_m")),
        curvature_radius_m=_optional_float(data.get("curvature_radius_m")),
        curvature_direction=data.get("curvature_direction"),
        width_m=_optional_float(data.get("width_m")),
        barrier_distance_m=_optional_float(data.get("barrier_distance_m")),
        runoff_type=data.get("runoff_type"),
        runoff_risk=_optional_float(data.get("runoff_risk")),
        surface_type=data.get("surface_type"),
        elevation_delta_m=_optional_float(data.get("elevation_delta_m")),
        camber_deg=_optional_float(data.get("camber_deg")),
        sightline_distance_m=_optional_float(data.get("sightline_distance_m")),
        drs_zone=bool(data.get("drs_zone", False)),
        overtaking_zone=bool(data.get("overtaking_zone", False)),
        source_tags=list(data.get("source_tags") or []),
        uncertainty={
            str(k): float(v)
            for k, v in (data.get("uncertainty") or {}).items()
            if isinstance(v, (int, float))
        },
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Segment risk features
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_segment_risk_features(segment: TrackSegmentModel) -> dict[str, float]:
    """Compute compact risk proxy features from segment metadata.

    Uses available fields only. Missing fields get 0.0 with no invented value.
    Boolean fields (drs_zone, overtaking_zone) are treated as 0/1 indicators.
    """
    width_m = segment.width_m
    barrier_distance_m = segment.barrier_distance_m
    curvature_radius_m = segment.curvature_radius_m
    sightline_distance_m = segment.sightline_distance_m
    elevation_delta_m = segment.elevation_delta_m
    camber_deg = segment.camber_deg
    runoff_type = segment.runoff_type

    narrowness = (
        _clamp((12.5 - width_m) / 4.0)
        if width_m is not None
        else 0.0
    )
    barrier_pressure = (
        _clamp((8.0 - barrier_distance_m) / 8.0)
        if barrier_distance_m is not None
        else 0.0
    )

    # Runoff surface risk encoding
    _runoff_risk_map = {
        "wall": 1.0,
        "barrier": 0.9,
        "concrete": 0.7,
        "gravel": 0.5,
        "grass": 0.4,
        "asphalt": 0.1,
        "mixed": 0.5,
        "unknown": 0.3,
    }
    runoff_surface_risk = (
        _runoff_risk_map.get(str(runoff_type).lower(), 0.3)
        if runoff_type is not None
        else 0.0
    )

    curvature_pressure = (
        _clamp(1.0 - curvature_radius_m / 300.0)
        if curvature_radius_m is not None
        else 0.0
    )

    sightline_pressure = (
        _clamp((500.0 - sightline_distance_m) / 500.0)
        if sightline_distance_m is not None
        else 0.0
    )

    # Unknown penalties: signal that data is missing
    elevation_unknown_penalty = 0.1 if elevation_delta_m is None else 0.0
    camber_unknown_penalty = 0.1 if camber_deg is None else 0.0

    return {
        "narrowness": round(narrowness, 4),
        "barrier_pressure": round(barrier_pressure, 4),
        "runoff_surface_risk": round(runoff_surface_risk, 4),
        "curvature_pressure": round(curvature_pressure, 4),
        "sightline_pressure": round(sightline_pressure, 4),
        "elevation_unknown_penalty": round(elevation_unknown_penalty, 4),
        "camber_unknown_penalty": round(camber_unknown_penalty, 4),
        "drs_zone": 1.0 if segment.drs_zone else 0.0,
        "overtaking_zone": 1.0 if segment.overtaking_zone else 0.0,
    }


# ---------------------------------------------------------------------------
# Builder: synthetic family -> TrackModel
# ---------------------------------------------------------------------------

def build_track_model_from_synthetic_family(
    family_id: str,
    family_spec: dict[str, Any] | None = None,
) -> TrackModel:
    """Convert a synthetic family spec into a T0 fidelity TrackModel.

    No real track attribution. Fidelity tier is always T0_synthetic_family.
    """
    spec = family_spec or {}

    # Build one representative segment from known family parameters
    segment = TrackSegmentModel(
        segment_id=str(spec.get("segment_id") or f"{family_id}:seg:000"),
        name=None,  # No real segment names
        segment_type=str(spec.get("segment_type") or "unknown"),
        width_m=_optional_float(spec.get("width_m")),
        barrier_distance_m=_optional_float(spec.get("barrier_distance_m")),
        runoff_type=spec.get("runoff_type"),
        runoff_risk=_optional_float(spec.get("runoff_risk")),
        sightline_distance_m=_optional_float(spec.get("visibility_m")),
        # Not available from synthetic families (knowledge gaps):
        curvature_radius_m=None,
        elevation_delta_m=None,
        camber_deg=None,
        drs_zone=False,
        overtaking_zone=True,
        source_tags=["synthetic_family"],
        uncertainty={},
    )

    gaps = [
        "missing_curvature_radius_m",
        "missing_elevation_delta_m",
        "missing_camber_deg",
    ]
    if segment.sightline_distance_m is None:
        gaps.append("missing_sightline_distance_m")

    lims = [
        "Track model is not a laser-scanned digital twin.",
        "Findings are conditioned on available segment abstractions.",
        "Do not attribute synthetic-family findings to a real circuit.",
    ]

    return TrackModel(
        schema_version=TRACK_MODEL_SCHEMA,
        track_id=str(spec.get("track_id") or family_id),
        display_name=str(spec.get("description") or f"Synthetic family: {family_id}"),
        fidelity_tier=FIDELITY_TIER_SYNTHETIC,
        provenance=[f"synthetic_family:{family_id}"],
        data_classification="synthetic",
        length_m=None,
        segment_count=1,
        segments=[segment],
        known_gaps=gaps,
        limitations=lims,
    )


# ---------------------------------------------------------------------------
# Builder: public approximate track
# ---------------------------------------------------------------------------

def build_public_approx_track_model(
    track_id: str,
    metadata: dict[str, Any],
) -> TrackModel:
    """Build a T1_public_approximation TrackModel from sparse public metadata.

    Uses only what is already in the metadata dict. No scraped/proprietary data.
    Missing fields become known_gaps.
    """
    length_m = _optional_float(metadata.get("length_m"))
    raw_segments = metadata.get("segments") or []
    segments: list[TrackSegmentModel] = []
    for i, seg_data in enumerate(raw_segments):
        seg = TrackSegmentModel(
            segment_id=str(seg_data.get("segment_id") or f"{track_id}:seg:{i:03d}"),
            name=seg_data.get("name"),
            segment_type=str(seg_data.get("segment_type") or "unknown"),
            length_m=_optional_float(seg_data.get("length_m")),
            width_m=_optional_float(seg_data.get("width_m")),
            barrier_distance_m=_optional_float(seg_data.get("barrier_distance_m")),
            runoff_type=seg_data.get("runoff_type"),
            runoff_risk=_optional_float(seg_data.get("runoff_risk")),
            sightline_distance_m=_optional_float(seg_data.get("sightline_distance_m")),
            curvature_radius_m=_optional_float(seg_data.get("curvature_radius_m")),
            elevation_delta_m=_optional_float(seg_data.get("elevation_delta_m")),
            camber_deg=_optional_float(seg_data.get("camber_deg")),
            drs_zone=bool(seg_data.get("drs_zone", False)),
            overtaking_zone=bool(seg_data.get("overtaking_zone", False)),
            source_tags=list(seg_data.get("source_tags") or ["public_metadata"]),
            uncertainty={
                str(k): float(v)
                for k, v in (seg_data.get("uncertainty") or {}).items()
                if isinstance(v, (int, float))
            },
        )
        segments.append(seg)

    # Determine known gaps
    gaps = list(metadata.get("known_gaps") or [])
    _standard_gaps = [
        ("missing_segment_widths", not any(s.width_m is not None for s in segments)),
        ("missing_barrier_distances", not any(s.barrier_distance_m is not None for s in segments)),
        ("missing_sightlines", not any(s.sightline_distance_m is not None for s in segments)),
        ("missing_camber", not any(s.camber_deg is not None for s in segments)),
        ("missing_elevation", not any(s.elevation_delta_m is not None for s in segments)),
        ("missing_curvature", not any(s.curvature_radius_m is not None for s in segments)),
    ]
    for gap_name, condition in _standard_gaps:
        if condition and gap_name not in gaps:
            gaps.append(gap_name)

    lims = [
        "Track model is a public approximate abstraction, not a digital twin.",
        "Segment geometry is approximate and may differ from actual circuit.",
        "Findings are track-conditioned on available public metadata.",
    ]
    user_lims = list(metadata.get("limitations") or [])
    for lim in user_lims:
        if lim not in lims:
            lims.append(lim)

    provenance = list(metadata.get("provenance") or [f"public_metadata:{track_id}"])

    return TrackModel(
        schema_version=TRACK_MODEL_SCHEMA,
        track_id=track_id,
        display_name=str(metadata.get("display_name") or track_id),
        fidelity_tier=FIDELITY_TIER_PUBLIC_APPROX,
        provenance=provenance,
        data_classification="public",
        length_m=length_m,
        segment_count=len(segments),
        segments=segments,
        known_gaps=gaps,
        limitations=lims,
    )
