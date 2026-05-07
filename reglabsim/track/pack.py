"""Track-pack manifests for the digital-twin layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from reglabsim.track.track_loader import TrackRepository


@dataclass(frozen=True)
class TrackPackEntry:
    """One target track inside a curated pack."""

    track_id: str
    build_priority: int
    builder_hint: str
    expected_fidelity_level: int
    target_validation_status: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrackPack:
    """Curated set of tracks used by runtime and validation flows."""

    name: str
    version: str
    description: str
    tracks: tuple[TrackPackEntry, ...] = field(default_factory=tuple)

    @property
    def track_ids(self) -> list[str]:
        """Return ordered track identifiers."""
        return [entry.track_id for entry in self.tracks]


class TrackPackRepository:
    """Load and validate the curated track pack manifest."""

    def __init__(self, pack_path: str | Path = "configs/track_pack.yaml"):
        self._pack_path = Path(pack_path)
        self._cache: TrackPack | None = None

    def load(self) -> TrackPack | None:
        """Load the configured track pack manifest, if present."""
        if self._cache is not None:
            return self._cache
        if not self._pack_path.exists():
            return None
        with open(self._pack_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        tracks = tuple(
            TrackPackEntry(
                track_id=str(item["track_id"]),
                build_priority=int(item.get("build_priority", index)),
                builder_hint=str(item.get("builder_hint", "raceway")),
                expected_fidelity_level=int(item.get("expected_fidelity_level", 1)),
                target_validation_status=str(
                    item.get("target_validation_status", "seeded_manual_review")
                ),
                notes=tuple(str(note) for note in item.get("notes", [])),
            )
            for index, item in enumerate(data.get("tracks", []), start=1)
        )
        self._cache = TrackPack(
            name=str(data.get("name", "target_track_pack")),
            version=str(data.get("version", "0")),
            description=str(data.get("description", "")),
            tracks=tracks,
        )
        return self._cache

    def list_target_ids(self) -> list[str]:
        """Return ordered track ids from the manifest."""
        pack = self.load()
        return pack.track_ids if pack else []

    def validate_against_repository(self, track_repository: TrackRepository) -> list[str]:
        """Return validation issues for the curated pack against loaded track configs."""
        pack = self.load()
        if pack is None:
            return ["Track pack manifest not found."]
        issues: list[str] = []
        seen: set[str] = set()
        for entry in pack.tracks:
            if entry.track_id in seen:
                issues.append(f"Duplicate track id '{entry.track_id}' in track pack manifest.")
                continue
            seen.add(entry.track_id)
            try:
                track = track_repository.get(entry.track_id)
            except KeyError:
                issues.append(f"Track '{entry.track_id}' is listed in the pack but missing.")
                continue
            if track.fidelity_level < entry.expected_fidelity_level:
                issues.append(
                    f"Track '{entry.track_id}' fidelity {track.fidelity_level} is below "
                    f"expected {entry.expected_fidelity_level}."
                )
            if track.validation_status != entry.target_validation_status:
                issues.append(
                    f"Track '{entry.track_id}' validation status '{track.validation_status}' "
                    f"does not match '{entry.target_validation_status}'."
                )
            missing_metadata = [
                field_name
                for field_name in ("latitude", "longitude", "track_family")
                if field_name not in track.metadata
            ]
            if missing_metadata:
                joined = ", ".join(missing_metadata)
                issues.append(f"Track '{entry.track_id}' missing metadata fields: {joined}.")
        return issues

