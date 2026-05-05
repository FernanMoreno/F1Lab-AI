"""Aerodynamic model.

Calculates drag and downforce forces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AeroForces:
    """Result of aerodynamic calculations.

    Attributes:
        drag_n: Drag force in Newtons.
        downforce_n: Downforce in Newtons.
        cda_m2: Drag area in m².
        cla_m2: Lift coefficient times area in m².
    """

    drag_n: float
    downforce_n: float
    cda_m2: float
    cla_m2: float


class AeroModel:
    """Aerodynamic model for F1 car.

    Calculates drag and downforce based on speed and aero mode.

    Attributes:
        cda_straight: Drag area in straight mode.
        cda_corner: Drag area in corner mode.
        cla_straight: Downforce area in straight mode.
        cla_corner: Downforce area in corner mode.
    """

    AIR_DENSITY_KG_M3 = 1.225  # Sea level standard

    def __init__(
        self,
        cda_straight: float = 0.9,
        cda_corner: float = 1.15,
        cla_straight: float = 2.3,
        cla_corner: float = 4.0,
    ):
        """Initialize aero model.

        Args:
            cda_straight: Drag area in straight mode (m²).
            cda_corner: Drag area in corner mode (m²).
            cla_straight: Downforce area in straight mode (m²).
            cla_corner: Downforce area in corner mode (m²).
        """
        self.cda_straight = cda_straight
        self.cda_corner = cda_corner
        self.cla_straight = cla_straight
        self.cla_corner = cla_corner

    def calculate_forces(
        self,
        speed_mps: float,
        mode: str = "straight",
    ) -> AeroForces:
        """Calculate aero forces at given speed.

        Args:
            speed_mps: Speed in m/s.
            mode: Aero mode ('straight', 'corner', 'drs').

        Returns:
            AeroForces with drag and downforce.
        """
        # Select appropriate CdA and ClA based on mode
        if mode == "corner":
            cda = self.cda_corner
            cla = self.cla_corner
        elif mode == "drs":
            # DRS reduces drag
            cda = self.cda_straight * 0.85
            cla = self.cla_straight * 0.95
        else:
            cda = self.cda_straight
            cla = self.cla_straight

        # F_drag = 0.5 * rho * CdA * v²
        drag = 0.5 * self.AIR_DENSITY_KG_M3 * cda * speed_mps**2

        # F_downforce = 0.5 * rho * ClA * v²
        downforce = 0.5 * self.AIR_DENSITY_KG_M3 * cla * speed_mps**2

        return AeroForces(
            drag_n=drag,
            downforce_n=downforce,
            cda_m2=cda,
            cla_m2=cla,
        )

    def get_drag(self, speed_mps: float, mode: str = "straight") -> float:
        """Get drag force.

        Args:
            speed_mps: Speed in m/s.
            mode: Aero mode.

        Returns:
            Drag force in Newtons.
        """
        return self.calculate_forces(speed_mps, mode).drag_n

    def get_downforce(self, speed_mps: float, mode: str = "straight") -> float:
        """Get downforce force.

        Args:
            speed_mps: Speed in m/s.
            mode: Aero mode.

        Returns:
            Downforce in Newtons.
        """
        return self.calculate_forces(speed_mps, mode).downforce_n

    def get_lift_drag_ratio(self, speed_mps: float, mode: str = "straight") -> float:
        """Get lift-to-drag ratio (L/D).

        Args:
            speed_mps: Speed in m/s.
            mode: Aero mode.

        Returns:
            L/D ratio.
        """
        forces = self.calculate_forces(speed_mps, mode)
        if forces.drag_n > 0:
            return forces.downforce_n / forces.drag_n
        return 0.0