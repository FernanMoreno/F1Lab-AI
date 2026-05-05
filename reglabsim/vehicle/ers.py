"""ERS (Energy Recovery System) model.

Models hybrid energy recovery and deployment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ERSState:
    """ERS system state.

    Attributes:
        soc: State of charge (0-1).
        energy_mj: Stored energy in MJ.
        deployment_mode: Current deployment mode.
        recovery_mode: Current recovery mode.
    """

    soc: float
    energy_mj: float
    deployment_mode: str
    recovery_mode: str


class ERSModel:
    """ERS energy management model.

    Models MGU-K and MGU-H energy recovery and deployment.

    Attributes:
        max_energy_mj: Maximum energy storage in MJ.
        max_deployment_kw: Maximum deployment power in kW.
        efficiency: ERS efficiency (0-1).
    """

    def __init__(
        self,
        max_energy_mj: float = 4.0,
        max_deployment_kw: float = 120.0,
        efficiency: float = 0.75,
    ):
        """Initialize ERS model.

        Args:
            max_energy_mj: Maximum energy storage.
            max_deployment_kw: Maximum deployment power.
            efficiency: ERS efficiency.
        """
        self.max_energy_mj = max_energy_mj
        self.max_deployment_kw = max_deployment_kw
        self.efficiency = efficiency

    def compute_deployment(
        self,
        current_soc: float,
        requested_kw: float,
        mode: str = "hybrid",
    ) -> tuple[float, ERSState]:
        """Compute ERS deployment.

        Args:
            current_soc: Current state of charge.
            requested_kw: Requested deployment power.
            mode: Deployment mode ('boost', 'hybrid', 'charge').

        Returns:
            Tuple of (actual_deployment_kw, new_state).
        """
        # Determine actual deployment
        if mode == "off":
            actual = 0.0
        elif mode == "charge":
            actual = -self.max_deployment_kw * 0.5  # Negative = charging
        elif mode == "boost":
            # Maximum deployment
            max_from_soc = current_soc * self.max_energy_mj * 1000 / 3.6  # kW*s
            actual = min(requested_kw, self.max_deployment_kw, max_from_soc)
        else:  # hybrid
            actual = min(requested_kw, self.max_deployment_kw * current_soc)

        # Update SOC
        # Energy withdrawn (MJ) = kW * s / 3600
        energy_used_mj = max(0, actual) / 3600.0 if actual > 0 else 0.0
        new_soc = max(0, min(1, current_soc - energy_used_mj / self.max_energy_mj))

        state = ERSState(
            soc=new_soc,
            energy_mj=new_soc * self.max_energy_mj,
            deployment_mode=mode,
            recovery_mode="auto",
        )

        return actual, state

    def compute_recovery(
        self,
        speed_mps: float,
        braking: float,
        throttle: float,
    ) -> float:
        """Compute ERS recovery.

        Args:
            speed_mps: Current speed.
            braking: Braking intensity (0-1).
            throttle: Throttle position (0-1).

        Returns:
            Energy recovered in MJ per second.
        """
        # MGU-K recovers during braking
        mgu_k_recovery = braking * speed_mps * 0.003 * self.efficiency

        # MGU-H recovers from exhaust when throttle < 1.0
        mgu_h_recovery = 0.0
        if throttle < 1.0:
            mgu_h_recovery = 0.002 * (1.0 - throttle) * self.efficiency

        total = mgu_k_recovery + mgu_h_recovery

        return min(total, self.max_energy_mj / 10.0)  # Cap at 10% per second

    def get_deployment_limit(self, soc: float, mode: str) -> float:
        """Get deployment limit for current state.

        Args:
            soc: Current state of charge.
            mode: Deployment mode.

        Returns:
            Maximum deployment power in kW.
        """
        if mode == "boost":
            return self.max_deployment_kw
        elif mode == "hybrid":
            return self.max_deployment_kw * soc
        elif mode == "charge":
            return 0.0
        return self.max_deployment_kw * soc * 0.5