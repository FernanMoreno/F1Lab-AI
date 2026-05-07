"""Power unit model.

Models F1 hybrid power unit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerUnitState:
    """Power unit operational state.

    Attributes:
        rpm: Current engine speed.
        power_output_kw: Current power output.
        fuel_flow_kg_h: Current fuel consumption.
        ers_soc: ERS state of charge (0-1).
        ers_deployed_kw: Current ERS deployment.
        mgu_k_power: MGU-K output power.
        mgu_h_power: MGU-H output power.
    """

    rpm: int = 0
    power_output_kw: float = 0.0
    fuel_flow_kg_h: float = 0.0
    ers_soc: float = 0.8
    ers_deployed_kw: float = 0.0
    mgu_k_power: float = 0.0
    mgu_h_power: float = 0.0


class PowerUnitModel:
    """F1 hybrid power unit model.

    Models ICE + MGU-K + MGU-H with energy management.

    Attributes:
        max_power_kw: Maximum ICE power.
        max_rpm: Maximum engine speed.
        fuel_capacity_kg: Fuel tank capacity.
        ers_max_energy_mj: Maximum ERS energy storage.
    """

    def __init__(
        self,
        max_power_kw: float = 750.0,
        max_rpm: int = 15000,
        fuel_capacity_kg: float = 110.0,
        ers_max_energy_mj: float = 4.0,
    ):
        """Initialize power unit.

        Args:
            max_power_kw: Maximum ICE power.
            max_rpm: Maximum RPM.
            fuel_capacity_kg: Fuel tank capacity.
            ers_max_energy_mj: Maximum ERS energy.
        """
        self.max_power_kw = max_power_kw
        self.max_rpm = max_rpm
        self.fuel_capacity_kg = fuel_capacity_kg
        self.ers_max_energy_mj = ers_max_energy_mj

    def get_power(
        self,
        throttle: float,
        rpm: int,
        ers_soc: float,
        ers_deployment_kw: float = 0.0,
    ) -> PowerUnitState:
        """Calculate power output.

        Args:
            throttle: Throttle position (0-1).
            rpm: Engine speed.
            ers_soc: ERS state of charge.
            ers_deployment_kw: Additional power from ERS.

        Returns:
            PowerUnitState with current values.
        """
        # Base ICE power based on throttle and RPM
        ice_power = throttle * self.max_power_kw * (rpm / self.max_rpm)

        # Total power
        total_power = ice_power + ers_deployment_kw

        # Fuel consumption (simplified)
        fuel_flow = throttle * 100.0  # kg/h at full throttle

        return PowerUnitState(
            rpm=rpm,
            power_output_kw=total_power,
            fuel_flow_kg_h=fuel_flow,
            ers_soc=ers_soc,
            ers_deployed_kw=ers_deployment_kw,
        )

    def get_ers_recovery(
        self,
        speed_mps: float,
        throttle: float,
        braking_intensity: float,
    ) -> float:
        """Calculate ERS energy recovery.

        Args:
            speed_mps: Current speed.
            throttle: Throttle position.
            braking_intensity: Braking intensity (0-1).

        Returns:
            ERS energy recovered in MJ per lap.
        """
        # MGU-K recovers during braking
        if braking_intensity > 0:
            recovery = braking_intensity * speed_mps * 0.01
        else:
            recovery = 0.0

        # MGU-H recovers from exhaust
        if throttle < 0.8:
            recovery += 0.02  # Some recovery from decel

        return min(recovery, self.ers_max_energy_mj * 0.1)

    def get_max_ers_deployment(self, ers_soc: float) -> float:
        """Get maximum ERS deployment based on SOC.

        Args:
            ers_soc: ERS state of charge.

        Returns:
            Maximum ERS power in kW.
        """
        # Can always deploy some if SOC > 0
        if ers_soc <= 0:
            return 0.0

        # Simplified - real ERS has complex deployment limits
        return min(120.0, ers_soc * 150.0)
