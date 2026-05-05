"""Benchmark dataset for validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRace:
    """A benchmark race for validation.

    Attributes:
        race_id: Unique identifier.
        year: Season year.
        circuit: Circuit identifier.
        conditions: Weather conditions.
        results: Race results.
        telemetry: Optional telemetry data.
    """

    race_id: str
    year: int
    circuit: str
    conditions: Dict[str, Any]
    results: List[Dict[str, Any]]
    telemetry: Optional[Dict[str, Any]] = None


class BenchmarkDataset:
    """Dataset of known races for validation.

    Provides ground truth data for backtesting simulation.

    Example:
        >>> dataset = BenchmarkDataset()
        >>> races = dataset.get_races(circuit="monza", year=2023)
    """

    def __init__(self):
        """Initialize dataset."""
        self._races: Dict[str, BenchmarkRace] = {}
        self._load_builtin_races()

    def _load_builtin_races(self) -> None:
        """Load built-in benchmark races."""
        # Built-in synthetic benchmarks for validation
        self._races["monza_2023_synthetic"] = BenchmarkRace(
            race_id="monza_2023_synthetic",
            year=2023,
            circuit="monza",
            conditions={"air_temp_c": 25, "track_temp_c": 35, "grip": 1.0},
            results=[
                {"driver_id": "HAM", "position": 1, "lap_time_s": 81.2},
                {"driver_id": "VER", "position": 2, "lap_time_s": 81.5},
                {"driver_id": "NOR", "position": 3, "lap_time_s": 81.8},
            ],
        )

    def add_race(self, race: BenchmarkRace) -> None:
        """Add a race to the dataset.

        Args:
            race: BenchmarkRace to add.
        """
        self._races[race.race_id] = race

    def get_race(self, race_id: str) -> Optional[BenchmarkRace]:
        """Get race by ID.

        Args:
            race_id: Race identifier.

        Returns:
            BenchmarkRace or None.
        """
        return self._races.get(race_id)

    def get_races(
        self,
        circuit: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[BenchmarkRace]:
        """Get races matching criteria.

        Args:
            circuit: Filter by circuit.
            year: Filter by year.

        Returns:
            List of matching races.
        """
        results = list(self._races.values())

        if circuit:
            results = [r for r in results if r.circuit == circuit]
        if year:
            results = [r for r in results if r.year == year]

        return results

    def list_circuits(self) -> List[str]:
        """List available circuits in dataset."""
        return list(set(r.circuit for r in self._races.values()))

    def list_years(self) -> List[int]:
        """List available years in dataset."""
        return list(set(r.year for r in self._races.values()))