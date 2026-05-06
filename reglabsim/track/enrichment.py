"""Boundary, kerb, and runoff enrichment for generated track seeds."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class TrackBoundaryProfileEnricher:
    """Apply YAML-defined boundary profiles to generated track payloads."""

    def __init__(
        self,
        config_path: str | Path = "configs/track_boundary_profiles.yaml",
    ) -> None:
        self._config_path = Path(config_path)
        self._config = self._load_config()

    def enrich_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return payload with segment-level boundary profiles merged in."""
        enriched = deepcopy(payload)
        metadata = enriched.setdefault("metadata", {})
        track_family = str(metadata.get("track_family", "")).strip().lower()
        defaults = self._config.get("defaults", {})
        family_profiles = self._config.get("track_families", {})
        family_profile = family_profiles.get(track_family, {})

        for segment in enriched.get("segments", []):
            segment_type = str(segment.get("type", "")).strip().lower()
            merged_profile = self._deep_merge({}, defaults.get("base", {}))
            merged_profile = self._deep_merge(
                merged_profile,
                defaults.get("segment_types", {}).get(segment_type, {}),
            )
            merged_profile = self._deep_merge(merged_profile, family_profile.get("base", {}))
            merged_profile = self._deep_merge(
                merged_profile,
                family_profile.get("segment_types", {}).get(segment_type, {}),
            )
            if merged_profile:
                self._apply_segment_profile(segment, merged_profile)

        metadata["boundary_profile_version"] = self._config.get(
            "schema_version",
            "track_boundary_profiles.v1",
        )
        metadata["boundary_profile_family"] = track_family or "default"
        metadata["boundary_profile_applied"] = True
        sources = list(enriched.get("sources", []))
        if "track_boundary_profiles" not in sources:
            sources.append("track_boundary_profiles")
        enriched["sources"] = sources
        notes = list(enriched.get("fidelity_notes", []))
        profile_note = "Boundary/kerb/runoff profile enrichment applied."
        if profile_note not in notes:
            notes.append(profile_note)
        enriched["fidelity_notes"] = notes
        return enriched

    def _load_config(self) -> dict[str, Any]:
        if not self._config_path.exists():
            return {"schema_version": "track_boundary_profiles.v1", "defaults": {}}
        with open(self._config_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _apply_segment_profile(
        self,
        segment: dict[str, Any],
        profile: dict[str, Any],
    ) -> None:
        for key in ("surface", "runoff", "kerbs", "track_limits", "risk", "metadata"):
            if key in profile:
                segment[key] = self._deep_merge(segment.get(key, {}), profile[key])

    def _deep_merge(self, base: Any, override: Any) -> Any:
        if not isinstance(base, dict) or not isinstance(override, dict):
            return deepcopy(override)
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged


__all__ = ["TrackBoundaryProfileEnricher"]
