"""Weather conditions model.

Models air temperature, track temperature, wind, humidity, and rainfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WeatherCondition:
    """Immutable weather condition representation.

    Attributes:
        air_temperature_c: Air temperature in Celsius.
        track_temperature_c: Track surface temperature in Celsius.
        humidity_percent: Relative humidity (0-100).
        wind_speed_mps: Wind speed in meters per second.
        wind_direction_deg: Wind direction in degrees (0-360).
        rainfall_mm_h: Rainfall rate in mm/hour.
        grip_level: Relative grip level (0.0 to 1.0+).
    """

    air_temperature_c: float = 25.0
    track_temperature_c: float = 30.0
    humidity_percent: float = 50.0
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0
    rainfall_mm_h: float = 0.0
    grip_level: float = 1.0

    @property
    def is_wet(self) -> bool:
        """Check if track is wet (rainfall or high wetness)."""
        return self.rainfall_mm_h > 0.0 or self.track_temperature_c < 15.0

    @property
    def is_dry(self) -> bool:
        """Check if track is dry."""
        return not self.is_wet

    @property
    def air_density_kg_m3(self) -> float:
        """Estimate air density in kg/m³.

        Uses simplified ideal gas calculation.
        """
        # Simplified: dry air at sea level ~1.225 kg/m³
        # Adjust for temperature and humidity
        temp_k = self.air_temperature_c + 273.15
        # Very simplified - real calculation would use more factors
        return 1.225 * (273.15 / temp_k)

    def adjusted_grip(self, base_grip: float) -> float:
        """Calculate grip adjusted for weather conditions.

        Args:
            base_grip: Base grip level.

        Returns:
            Adjusted grip level.
        """
        grip = base_grip * self.grip_level

        # Wet conditions reduce grip
        if self.is_wet:
            grip *= 0.7

        # Very hot track can reduce grip
        if self.track_temperature_c > 50:
            grip *= 0.95

        # Very cold track can reduce grip
        if self.track_temperature_c < 15:
            grip *= 0.9

        return grip


@dataclass(frozen=True)
class WeatherScenario:
    """Collection of weather conditions over time for race simulation."""

    initial: WeatherCondition
    evolution: str = "stable"  # 'stable', 'warming', 'cooling', 'deteriorating'
    probability_rain: float = 0.0
    probability_vsc: float = 0.1

    def get_condition_at_lap(self, lap: int, total_laps: int) -> WeatherCondition:
        """Get interpolated weather condition at given lap.

        Args:
            lap: Current lap number.
            total_laps: Total race laps.

        Returns:
            Weather condition at specified lap.
        """
        # For now, return initial condition
        # More sophisticated interpolation can be added later
        return self.initial