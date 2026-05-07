"""Legacy circuit compatibility module backed by `reglabsim.track`."""

from reglabsim.circuits.base import CircuitModel, CircuitRepository, CircuitSegment
from reglabsim.circuits.track_model import TrackModel, TrackSegment, create_simple_track_model

__all__ = [
    "CircuitModel",
    "CircuitRepository",
    "CircuitSegment",
    "TrackModel",
    "TrackSegment",
    "create_simple_track_model",
]
