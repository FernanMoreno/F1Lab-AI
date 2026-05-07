"""Legacy circuit compatibility layer backed by digital track configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Any, ClassVar

from reglabsim.track.geometry import TrackModel as DigitalTrackModel
from reglabsim.track.segments import TrackSegment as DigitalTrackSegment
from reglabsim.track.track_loader import TrackRepository


@dataclass(frozen=True)
class CircuitModel:
    """Legacy circuit representation derived from the digital-twin layer."""

    circuit_id: str
    name: str
    country: str
    length_m: float
    corners: int
    drs_zones: int
    avg_speed_kph: float
    characteristics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_track_model(cls, track: DigitalTrackModel) -> CircuitModel:
        """Create a legacy circuit view from a digital track model."""
        track_family = str(track.metadata.get("track_family", ""))
        boost_zones = sum(1 for segment in track.segments if segment.primary_boost_zone)
        characteristics: dict[str, Any] = {
            "track_family": track_family,
            "validation_status": track.validation_status,
            "fidelity_level": track.fidelity_level,
            "digital_twin_sources": list(track.sources),
            "street_circuit": "street" in track_family,
            "high_speed": track.avg_speed_kph > 220.0,
        }
        characteristics.update(dict(track.metadata))
        return cls(
            circuit_id=track.track_id,
            name=track.name,
            country=track.country,
            length_m=track.length_m,
            corners=track.turns,
            drs_zones=max(1, boost_zones) if track.segments else 1,
            avg_speed_kph=track.avg_speed_kph,
            characteristics=characteristics,
        )

    @property
    def lap_count_estimate(self) -> int:
        """Estimate race lap count from the canonical 305 km reference distance."""
        race_distance_km = 305.0
        if self.length_m <= 0:
            return 0
        return int(race_distance_km / (self.length_m / 1000.0))

    @property
    def is_high_speed(self) -> bool:
        """Check if this behaves like a high-speed circuit."""
        return self.avg_speed_kph > 220.0

    @property
    def is_street_circuit(self) -> bool:
        """Check if the underlying track family marks a street circuit."""
        return bool(self.characteristics.get("street_circuit", False))

    def get_segment_count(self) -> int:
        """Approximate segment count using turn count and derived straights."""
        return self.corners + self._estimate_straight_count()

    def _estimate_straight_count(self) -> int:
        """Estimate number of meaningful straights."""
        return max(2, self.corners // 3)


@dataclass(frozen=True)
class CircuitSegment:
    """Legacy segment view derived from a digital track segment."""

    segment_id: int
    start_m: float
    end_m: float
    segment_type: str = "corner"
    length_m: float = 0.0
    corner_radius_m: float = inf
    max_speed_mps: float = 0.0

    @classmethod
    def from_track_segment(
        cls,
        segment: DigitalTrackSegment,
        *,
        segment_index: int,
    ) -> CircuitSegment:
        """Convert one digital track segment into the legacy shape."""
        estimated_speed = float(segment.metadata.get("estimated_max_speed_mps", 0.0))
        return cls(
            segment_id=segment_index,
            start_m=segment.start_m,
            end_m=segment.end_m,
            segment_type=segment.segment_type,
            length_m=segment.length_m,
            corner_radius_m=float(segment.radius_m) if segment.radius_m is not None else inf,
            max_speed_mps=estimated_speed,
        )

    @property
    def is_straight(self) -> bool:
        """Check if this is a straight segment."""
        return self.segment_type == "straight"

    @property
    def is_corner(self) -> bool:
        """Check if this is a corner-like segment."""
        return self.segment_type not in {"straight"}


class CircuitRepository:
    """Legacy circuit repository redirected to `configs/tracks`."""

    _tracks_dir: ClassVar[Path] = Path("configs/tracks")
    _track_repository: ClassVar[TrackRepository | None] = None
    _registered: ClassVar[dict[str, CircuitModel]] = {}

    @classmethod
    def configure(cls, tracks_dir: str | Path) -> None:
        """Point the compatibility layer at a different track directory."""
        cls._tracks_dir = Path(tracks_dir)
        cls._track_repository = TrackRepository(cls._tracks_dir)

    @classmethod
    def _repo(cls) -> TrackRepository:
        if cls._track_repository is None:
            cls._track_repository = TrackRepository(cls._tracks_dir)
        return cls._track_repository

    @classmethod
    def get_track_model(cls, circuit_id: str) -> DigitalTrackModel:
        """Return the canonical digital track model for a legacy circuit id."""
        return cls._repo().get(circuit_id)

    @classmethod
    def get(cls, circuit_id: str) -> CircuitModel:
        """Get a legacy circuit view by id."""
        if circuit_id in cls._registered:
            return cls._registered[circuit_id]
        return CircuitModel.from_track_model(cls.get_track_model(circuit_id))

    @classmethod
    def list_ids(cls) -> list[str]:
        """List all known circuit ids from curated tracks plus ad-hoc registrations."""
        return sorted(set(cls._repo().list_ids()) | set(cls._registered.keys()))

    @classmethod
    def register(cls, circuit: CircuitModel) -> None:
        """Register an ad-hoc legacy circuit without touching the curated track pack."""
        cls._registered[circuit.circuit_id] = circuit
