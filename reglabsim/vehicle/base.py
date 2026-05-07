"""Base vehicle model.

Defines vehicle interface and common attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Current state of a vehicle.

    Attributes:
        speed_mps: Current speed in m/s.
        position_m: Distance traveled.
        throttle: Throttle position (0-1).
        brake: Brake position (0-1).
        ers_soc: ERS state of charge (0-1).
        fuel_mass_kg: Current fuel mass.
        tyre_age_laps: Tyre age in laps.
    """

    speed_mps: float = 0.0
    position_m: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    ers_soc: float = 0.8
    fuel_mass_kg: float = 100.0
    tyre_age_laps: int = 0


class VehicleModel(ABC):
    """Abstract base class for vehicle models.

    Defines interface that all vehicle implementations must provide.
    """

    @property
    @abstractmethod
    def vehicle_id(self) -> str:
        """Get unique vehicle identifier."""
        ...

    @property
    @abstractmethod
    def mass_kg(self) -> float:
        """Get total vehicle mass in kg."""
        ...

    @abstractmethod
    def get_drag_force(self, speed_mps: float, aero_mode: str = "straight") -> float:
        """Calculate aerodynamic drag force.

        Args:
            speed_mps: Speed in m/s.
            aero_mode: Aero mode ('straight', 'corner', 'drs').

        Returns:
            Drag force in Newtons.
        """
        ...

    @abstractmethod
    def get_downforce(self, speed_mps: float, aero_mode: str = "straight") -> float:
        """Calculate aerodynamic downforce.

        Args:
            speed_mps: Speed in m/s.
            aero_mode: Aero mode.

        Returns:
            Downforce in Newtons.
        """
        ...

    @abstractmethod
    def get_power_available(self, throttle: float) -> float:
        """Get available power at given throttle.

        Args:
            throttle: Throttle position (0-1).

        Returns:
            Power in kW.
        """
        ...

    def get_acceleration(
        self,
        state: VehicleState,
        grade: float = 0.0,
    ) -> float:
        """Calculate vehicle acceleration.

        Args:
            state: Current vehicle state.
            grade: Road grade (radians).

        Returns:
            Acceleration in m/s².
        """
        # F = ma, so a = F/m
        # Forces: engine - drag - rolling resistance - climbing
        power_kw = self.get_power_available(state.throttle)
        power_n = power_kw * 1000 / max(state.speed_mps, 0.1)

        drag = self.get_drag_force(state.speed_mps)
        self.get_downforce(state.speed_mps)

        # Simplified rolling resistance
        rolling_res = 0.015 * self.mass_kg * 9.81

        # Grade force
        grade_force = self.mass_kg * 9.81 * grade

        net_force = power_n - drag - rolling_res - grade_force
        return net_force / self.mass_kg
