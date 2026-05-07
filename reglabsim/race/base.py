"""Base race simulation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CarState:
    """State of a car during race.

    Attributes:
        car_id: Unique car identifier.
        driver_id: Driver identifier.
        position: Current race position.
        lap: Current lap number.
        lap_time_s: Current lap time.
        gap_to_leader_s: Gap to race leader.
        tyre_compound: Current tyre compound.
        tyre_age_laps: Tyre age in laps.
        pit_stops: Number of pit stops.
        ers_soc: ERS state of charge.
        fuel_mass_kg: Current fuel mass.
        is_in_pit: Whether car is in pit.
    """

    car_id: str
    driver_id: str
    position: int = 1
    lap: int = 0
    lap_time_s: float = 0.0
    gap_to_leader_s: float = 0.0
    tyre_compound: str = "C3"
    tyre_age_laps: int = 0
    pit_stops: int = 0
    ers_soc: float = 0.8
    fuel_mass_kg: float = 100.0
    is_in_pit: bool = False


@dataclass
class RaceState:
    """Complete race state.

    Attributes:
        race_id: Race identifier.
        lap: Current race lap.
        total_laps: Total race laps.
        cars: List of car states.
        is_finished: Whether race is complete.
        winner: Winner car ID if finished.
    """

    race_id: str
    lap: int = 0
    total_laps: int = 53
    cars: list[CarState] = field(default_factory=list)
    is_finished: bool = False
    winner: str | None = None

    def get_positions(self) -> list[CarState]:
        """Get cars sorted by position."""
        return sorted(self.cars, key=lambda c: c.position)

    def get_car(self, car_id: str) -> CarState | None:
        """Get car state by ID."""
        for car in self.cars:
            if car.car_id == car_id:
                return car
        return None

    def advance_lap(self) -> None:
        """Advance race by one lap."""
        self.lap += 1
        if self.lap >= self.total_laps:
            self.is_finished = True
            self.winner = self.cars[0].car_id if self.cars else None


class RaceSimulatorBase(ABC):
    """Abstract base for race simulators."""

    @abstractmethod
    def simulate(
        self,
        race_config: dict[str, Any],
        cars: list[dict[str, Any]],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a race.

        Args:
            race_config: Race configuration.
            cars: List of car configurations.
            seed: Random seed.

        Returns:
            Race results dictionary.
        """
        ...


class RaceSimulator(RaceSimulatorBase):
    """Simple race simulator.

    Simulates full race with positions, overtakes, pit stops.
    """

    def __init__(self) -> None:
        """Initialize race simulator."""
        pass

    def simulate(
        self,
        race_config: dict[str, Any],
        cars: list[dict[str, Any]],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Simulate a race.

        Simplified race simulation with position changes.
        """
        import numpy as np

        rng = np.random.default_rng(seed)

        n_cars = len(cars)
        n_laps = int(race_config.get("laps", 53))

        # Initialize car states
        car_states: list[CarState] = []
        for i, car_config in enumerate(cars):
            car_states.append(
                CarState(
                    car_id=str(car_config.get("car_id", f"car_{i}")),
                    driver_id=str(car_config.get("driver_id", f"driver_{i}")),
                    position=i + 1,
                    lap_time_s=80.0 + rng.uniform(-2, 2),
                    fuel_mass_kg=float(car_config.get("fuel_mass_kg", 100.0)),
                    ers_soc=float(car_config.get("ers_soc", 0.8)),
                )
            )

        # Simulate race laps
        positions_history: list[list[int]] = [list(range(1, n_cars + 1))]
        overtakes: list[dict[str, Any]] = []

        for lap in range(1, n_laps + 1):
            # Update each car's lap time
            for car in car_states:
                car.lap = lap
                # Add variation
                car.lap_time_s += rng.normal(0, 0.5)
                car.fuel_mass_kg = max(0, car.fuel_mass_kg - 0.5)
                car.ers_soc = max(0, min(1, car.ers_soc + rng.uniform(-0.1, 0.05)))

            # Sort by lap time to get positions
            car_states.sort(key=lambda c: c.lap_time_s)

            # Update positions
            for i, car in enumerate(car_states):
                new_pos = i + 1
                if car.position != new_pos:
                    overtakes.append(
                        {
                            "lap": lap,
                            "car_id": car.car_id,
                            "old_position": car.position,
                            "new_position": new_pos,
                        }
                    )
                car.position = new_pos

            positions_history.append([c.position for c in car_states])

        return {
            "race_id": race_config.get("race_id", "race_1"),
            "winner": car_states[0].car_id,
            "final_positions": [c.car_id for c in car_states],
            "positions_history": positions_history,
            "total_overtakes": len(overtakes),
            "overtakes": overtakes,
            "laps": n_laps,
            "cars": [
                {
                    "car_id": c.car_id,
                    "position": c.position,
                    "lap_time_s": c.lap_time_s,
                    "pit_stops": c.pit_stops,
                }
                for c in car_states
            ],
        }
