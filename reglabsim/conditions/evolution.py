"""Track evolution model.

Models how track conditions evolve during a race weekend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackEvolution:
    """Models track evolution over a race weekend.

    Track grip typically increases as more rubber is laid down,
    but can plateau or decrease with excessive rubber.

    Attributes:
        initial_grip: Starting grip level.
        peak_grip: Peak grip level expected.
        peak_laps: Laps at which peak grip occurs.
        final_grip: Final grip level after evolution.
    """

    initial_grip: float = 0.95
    peak_grip: float = 1.05
    peak_laps: int = 30
    final_grip: float = 1.0

    def get_grip_at_lap(self, lap: int) -> float:
        """Get grip level at given lap.

        Args:
            lap: Lap number.

        Returns:
            Grip level.
        """
        if lap <= 0:
            return self.initial_grip
        elif lap < self.peak_laps:
            # Linear increase to peak
            slope = (self.peak_grip - self.initial_grip) / self.peak_laps
            return self.initial_grip + slope * lap
        else:
            # Gradual decrease after peak
            decay = (lap - self.peak_laps) * 0.001
            return max(self.final_grip, self.peak_grip - decay)


@dataclass
class SessionEvolution:
    """Track evolution specific to a session type."""

    fp1: TrackEvolution | None = None
    fp2: TrackEvolution | None = None
    fp3: TrackEvolution | None = None
    quali: TrackEvolution | None = None
    race: TrackEvolution | None = None

    def __post_init__(self) -> None:
        """Initialize default evolution if not provided."""
        if self.fp1 is None:
            self.fp1 = TrackEvolution(initial_grip=0.9, peak_grip=0.98, peak_laps=20)
        if self.fp2 is None:
            self.fp2 = TrackEvolution(initial_grip=0.95, peak_grip=1.02, peak_laps=25)
        if self.fp3 is None:
            self.fp3 = TrackEvolution(initial_grip=0.98, peak_grip=1.03, peak_laps=15)
        if self.quali is None:
            self.quali = TrackEvolution(initial_grip=1.0, peak_grip=1.05, peak_laps=10)
        if self.race is None:
            self.race = TrackEvolution(initial_grip=1.02, peak_grip=1.06, peak_laps=30)

    def get_evolution(
        self,
        session_type: str,
        prev_session_laps: int = 0,
    ) -> TrackEvolution:
        """Get evolution model for session.

        Args:
            session_type: Session identifier ('fp1', 'fp2', etc.).
            prev_session_laps: Laps from previous sessions.

        Returns:
            TrackEvolution for the session.
        """
        assert (
            self.fp1 is not None
            and self.fp2 is not None
            and self.fp3 is not None
            and self.quali is not None
            and self.race is not None
        )
        session_map: dict[str, TrackEvolution] = {
            "fp1": self.fp1,
            "fp2": self.fp2,
            "fp3": self.fp3,
            "quali": self.quali,
            "race": self.race,
        }
        return session_map.get(session_type, TrackEvolution())
