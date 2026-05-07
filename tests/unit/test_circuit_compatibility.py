"""Compatibility tests for the legacy `circuits/` facade."""

from __future__ import annotations

from reglabsim.circuits.base import CircuitRepository
from reglabsim.circuits.track_model import create_simple_track_model
from reglabsim.track.pack import TrackPackRepository
from reglabsim.track.track_loader import TrackRepository


def test_track_pack_manifest_matches_curated_tracks() -> None:
    pack_repo = TrackPackRepository("configs/track_pack.yaml")
    track_repo = TrackRepository("configs/tracks")

    pack = pack_repo.load()

    assert pack is not None
    assert pack.track_ids == [
        "suzuka",
        "baku",
        "monaco",
        "monza",
        "austria",
        "singapore",
        "barcelona",
        "silverstone",
    ]
    assert pack_repo.validate_against_repository(track_repo) == []


def test_circuit_repository_reads_from_track_repository() -> None:
    digital_track = TrackRepository("configs/tracks").get("monaco")
    circuit = CircuitRepository.get("monaco")

    assert circuit.circuit_id == digital_track.track_id
    assert circuit.length_m == digital_track.length_m
    assert circuit.corners == digital_track.turns
    assert circuit.is_street_circuit is True
    assert "manual_curation" in circuit.characteristics["digital_twin_sources"]


def test_create_simple_track_model_uses_digital_segments_for_known_tracks() -> None:
    circuit = CircuitRepository.get("baku")
    digital_track = TrackRepository("configs/tracks").get("baku")

    compat_track = create_simple_track_model(circuit)

    assert compat_track.get_total_segments() == len(digital_track.segments)
    assert compat_track.segments[0].segment_type == digital_track.segments[0].segment_type
    assert compat_track.segments[0].start_distance_m == digital_track.segments[0].start_m
    assert compat_track.segments[0].end_distance_m == digital_track.segments[0].end_m
