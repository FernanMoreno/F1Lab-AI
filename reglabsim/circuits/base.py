"""Circuit base model.

Represents F1 circuits with physical properties and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class CircuitModel:
    """Immutable circuit representation.

    Attributes:
        circuit_id: Unique identifier.
        name: Circuit name.
        country: Country location.
        length_m: Track length in meters.
        corners: Number of corners.
        drs_zones: Number of DRS zones.
        avg_speed_kph: Average speed in km/h.
        characteristics: Dict of circuit characteristics.
    """

    circuit_id: str
    name: str
    country: str
    length_m: float
    corners: int
    drs_zones: int
    avg_speed_kph: float
    characteristics: dict[str, Any] = field(default_factory=dict)

    @property
    def lap_count_estimate(self) -> int:
        """Estimate race lap count (305km race distance)."""
        race_distance_km = 305
        return int(race_distance_km / (self.length_m / 1000))

    @property
    def is_high_speed(self) -> bool:
        """Check if high-speed circuit."""
        return self.avg_speed_kph > 220

    @property
    def is_street_circuit(self) -> bool:
        """Check if street circuit."""
        return bool(self.characteristics.get("street_circuit", False))

    def get_segment_count(self) -> int:
        """Get number of track segments (straights + corners)."""
        return self.corners + self._estimate_straight_count()

    def _estimate_straight_count(self) -> int:
        """Estimate number of straights."""
        # Very simplified - would need real track data
        return max(2, self.corners // 3)


@dataclass
class CircuitSegment:
    """A segment of the circuit (straight or corner).

    Attributes:
        segment_id: Segment identifier.
        start_m: Start distance in meters.
        end_m: End distance in meters.
        segment_type: 'straight', 'corner', 'chicane'.
        length_m: Segment length.
        corner_radius_m: Radius if corner (infinity for straight).
        max_speed_mps: Maximum speed in this segment.
    """

    segment_id: int
    start_m: float
    end_m: float
    segment_type: str = "corner"
    length_m: float = 0.0
    corner_radius_m: float = float("inf")
    max_speed_mps: float = 0.0

    @property
    def is_straight(self) -> bool:
        """Check if segment is a straight."""
        return self.segment_type == "straight"

    @property
    def is_corner(self) -> bool:
        """Check if segment is a corner."""
        return self.segment_type == "corner"


class CircuitRepository:
    """Repository of known circuits."""

    _circuits: ClassVar[dict[str, CircuitModel]] = {
        "monza": CircuitModel(
            circuit_id="monza",
            name="Autodromo Nazionale Monza",
            country="Italy",
            length_m=5793.0,
            corners=11,
            drs_zones=1,
            avg_speed_kph=250.0,
            characteristics={"high_speed": True, "low_downforce": True},
        ),
        "monaco": CircuitModel(
            circuit_id="monaco",
            name="Circuit de Monaco",
            country="Monaco",
            length_m=3371.0,
            corners=19,
            drs_zones=1,
            avg_speed_kph=160.0,
            characteristics={"tight_corners": True, "street_circuit": True},
        ),
        "baku": CircuitModel(
            circuit_id="baku",
            name="Baku City Circuit",
            country="Azerbaijan",
            length_m=6003.0,
            corners=20,
            drs_zones=1,
            avg_speed_kph=200.0,
            characteristics={"straight_heavy": True, "street_circuit": True},
        ),
        "barcelona": CircuitModel(
            circuit_id="barcelona",
            name="Circuit de Barcelona-Catalunya",
            country="Spain",
            length_m=4677.0,
            corners=16,
            drs_zones=1,
            avg_speed_kph=200.0,
            characteristics={"balanced": True, "technical_corners": True},
        ),
    }

    @classmethod
    def get(cls, circuit_id: str) -> CircuitModel:
        """Get circuit by ID.

        Args:
            circuit_id: Circuit identifier.

        Returns:
            CircuitModel.

        Raises:
            KeyError: If circuit not found.
        """
        if circuit_id not in cls._circuits:
            raise KeyError(f"Circuit '{circuit_id}' not found")
        return cls._circuits[circuit_id]

    @classmethod
    def list_ids(cls) -> list[str]:
        """List all available circuit IDs."""
        return list(cls._circuits.keys())

    @classmethod
    def register(cls, circuit: CircuitModel) -> None:
        """Register a new circuit.

        Args:
            circuit: Circuit to register.
        """
        cls._circuits[circuit.circuit_id] = circuit
