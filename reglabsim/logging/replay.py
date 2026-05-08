"""Replay and persistence helpers for multiagent race runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reglabsim.runtime.schema import (
    EVIDENCE_BUNDLE_REQUIRED_KEYS,
    EVENT_ENVELOPE_SCHEMA,
    STATE_HASH_REQUIRED_KEYS,
    WORLD_MANIFEST_REQUIRED_KEYS,
    WORLD_MANIFEST_SCHEMA,
    EvidenceBundle,
)


class ReplayEngine:
    """Persist and replay run outputs."""

    EVIDENCE_BUNDLE_SCHEMA = "evidence_bundle.v1"

    def save_run(self, run_output: dict[str, Any], output_dir: str | Path) -> Path:
        """Write one run output bundle to disk as JSON files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, value in run_output.items():
            if key == "summary_markdown":
                (out_dir / "summary.md").write_text(value, encoding="utf-8")
                continue
            path = out_dir / f"{key}.json"
            path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
        return out_dir

    def load_run(self, path: str | Path) -> dict[str, Any]:
        """Load a run bundle from disk."""
        directory = Path(path)
        result: dict[str, Any] = {}
        for json_file in directory.glob("*.json"):
            result[json_file.stem] = json.loads(json_file.read_text(encoding="utf-8"))
        summary_file = directory / "summary.md"
        if summary_file.exists():
            result["summary_markdown"] = summary_file.read_text(encoding="utf-8")
        return result

    def replay_audit_exact(self, run_output: dict[str, Any]) -> dict[str, Any]:
        """Return the persisted result exactly as captured."""
        return {
            "mode": "replay_audit_exact",
            "manifest": run_output["manifest"],
            "state_snapshots": run_output["state_snapshots"],
            "event_log": run_output["event_log"],
            "steward_log": run_output["steward_log"],
            "failure_log": run_output.get("failure_log", []),
        }

    def extract_policy_replay_actions(
        self, run_output: dict[str, Any]
    ) -> dict[tuple[int, str], dict[str, Any]]:
        """Build a lookup for policy replay mode."""
        replay_actions: dict[tuple[int, str], dict[str, Any]] = {}
        for entry in run_output.get("action_log", []):
            replay_actions[(int(entry["lap"]), str(entry["car_id"]))] = entry["action"]
        return replay_actions

    def build_evidence_bundle(self, run_output: dict[str, Any]) -> dict[str, Any]:
        """Build a normalized evidence bundle without mutating the source run output."""
        manifest = dict(run_output.get("manifest", {}))
        world_manifest = self._build_world_manifest(run_output)
        event_envelopes = self._build_event_envelopes(run_output)
        unsafe_legal_states = self._extract_unsafe_legal_states(run_output)
        legal_verdicts = self._extract_legal_verdicts(run_output)
        patch_reruns = self._extract_patch_reruns(run_output)
        state_hashes = self._build_state_hashes(run_output)
        replay_integrity = self._build_replay_integrity(run_output)
        bundle = EvidenceBundle(
            schema_version=self.EVIDENCE_BUNDLE_SCHEMA,
            run_id=str(manifest.get("run_id", "")),
            slice_id=str(manifest.get("slice_id", "")),
            world_id=str(manifest.get("world_id", "")),
            seed=int(manifest.get("seed", 0)),
            config_hash=str(manifest.get("config_hash", "")),
            regulation_id=str(manifest.get("regulation_id", "")),
            track=str(manifest.get("track_id", "")),
            segment_focus=self._resolve_segment_focus(run_output),
            world_manifest=world_manifest,
            legal_verdicts=legal_verdicts,
            event_envelopes=event_envelopes,
            unsafe_legal_states=unsafe_legal_states,
            patch_reruns=patch_reruns,
            metrics=dict(run_output.get("metrics", {})),
            state_hashes=state_hashes,
            replay_integrity=replay_integrity,
        )
        result = bundle.to_dict()
        result["falsification"] = self._build_falsification_manifest(run_output)
        result["scores"] = {
            "public_anchor_score": manifest.get("public_anchor_score"),
            "baseline_plausibility_score": manifest.get("baseline_plausibility_score"),
            "regulation_breaking_score": manifest.get("regulation_breaking_score"),
        }
        return result

    def export_evidence_bundle(
        self,
        run_output_or_path: dict[str, Any] | str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Export one normalized evidence bundle alongside the existing replay layout."""
        run_output = self._coerce_run_output(run_output_or_path)
        bundle_path = self._resolve_bundle_path(run_output_or_path, run_output, output_path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = self.build_evidence_bundle(run_output)
        bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
        return bundle_path

    def _coerce_run_output(
        self, run_output_or_path: dict[str, Any] | str | Path
    ) -> dict[str, Any]:
        if isinstance(run_output_or_path, dict):
            return run_output_or_path
        return self.load_run(run_output_or_path)

    def _resolve_bundle_path(
        self,
        run_output_or_path: dict[str, Any] | str | Path,
        run_output: dict[str, Any],
        output_path: str | Path | None,
    ) -> Path:
        if output_path is not None:
            target = Path(output_path)
            return target if target.suffix else target / "evidence_bundle.json"
        if isinstance(run_output_or_path, (str, Path)):
            return Path(run_output_or_path) / "evidence_bundle.json"
        run_id = str(run_output.get("manifest", {}).get("run_id", "unknown_run"))
        return Path("outputs") / "evidence_bundles" / run_id / "evidence_bundle.json"

    def _build_world_manifest(self, run_output: dict[str, Any]) -> dict[str, Any]:
        existing = run_output.get("world_manifest")
        if isinstance(existing, dict) and existing:
            base = dict(existing)
        else:
            manifest = dict(run_output.get("manifest", {}))
            spec = run_output.get("spec", {})
            falsification = spec.get("falsification", {}) if isinstance(spec, dict) else {}
            conditions = run_output.get("conditions", {})
            base = {
                "schema_version": WORLD_MANIFEST_SCHEMA,
                "world_id": manifest.get("world_id") or manifest.get("run_id") or "unknown_world",
                "slice_id": manifest.get("slice_id") or falsification.get("slice_id"),
                "regulation_id": manifest.get("regulation_id", "unknown_regulation"),
                "track_id": manifest.get("track_id", "unknown_track"),
                "seed": int(manifest.get("seed", 0)),
                "priors_profile": falsification.get("priors_profile"),
                "car_family_assignments": self._collect_car_family_assignments(run_output),
                "world_parameters": falsification.get("world_parameters", {}),
                "condition_profile": conditions,
                "perception_profile": falsification.get("perception_profile", {}),
                "notes": falsification.get("notes", []),
            }
        manifest = dict(run_output.get("manifest", {}))
        spec = run_output.get("spec", {})
        falsification = spec.get("falsification", {}) if isinstance(spec, dict) else {}
        base.setdefault("world_id", manifest.get("world_id", ""))
        base.setdefault("seed", int(manifest.get("seed", 0)))
        base.setdefault("regulation_id", manifest.get("regulation_id", ""))
        base.setdefault("track_id", manifest.get("track_id", ""))
        base.setdefault("segment_focus", self._resolve_segment_focus(run_output))
        base.setdefault("slice_id", manifest.get("slice_id") or falsification.get("slice_id", ""))
        base.setdefault("config_hash", manifest.get("config_hash", ""))
        if "sim_profile" not in base:
            base["sim_profile"] = getattr(
                type("Spec", (), {"sim_profile": "public_baseline"})(), "sim_profile"
            )
            if isinstance(spec, dict):
                base["sim_profile"] = spec.get("sim_profile", "public_baseline")
        if "patch_id" not in base:
            base["patch_id"] = manifest.get("patch_id")
        return base

    def _build_falsification_manifest(self, run_output: dict[str, Any]) -> dict[str, Any]:
        existing = run_output.get("falsification")
        if isinstance(existing, dict) and existing:
            return existing
        manifest = dict(run_output.get("manifest", {}))
        spec = run_output.get("spec", {})
        falsification = spec.get("falsification", {}) if isinstance(spec, dict) else {}
        return {
            **(falsification if isinstance(falsification, dict) else {}),
            "slice_id": manifest.get("slice_id"),
            "world_id": manifest.get("world_id"),
            "patch_id": manifest.get("patch_id"),
            "scores": {
                "public_anchor_score": manifest.get("public_anchor_score"),
                "baseline_plausibility_score": manifest.get("baseline_plausibility_score"),
                "regulation_breaking_score": manifest.get("regulation_breaking_score"),
            },
        }

    def _collect_car_family_assignments(self, run_output: dict[str, Any]) -> dict[str, str]:
        snapshots = run_output.get("state_snapshots", [])
        if snapshots:
            first_snapshot = snapshots[0]
            cars = first_snapshot.get("cars", []) if isinstance(first_snapshot, dict) else []
            assignments = {
                str(car.get("car_id")): str(car.get("family_id"))
                for car in cars
                if isinstance(car, dict) and car.get("car_id") and car.get("family_id")
            }
            if assignments:
                return assignments
        result = run_output.get("result", {})
        final_positions = result.get("final_positions", []) if isinstance(result, dict) else []
        return {str(car_id): "unknown_family" for car_id in final_positions}

    def _build_event_envelopes(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        existing = run_output.get("event_envelopes")
        if isinstance(existing, list) and existing:
            return existing

        manifest = dict(run_output.get("manifest", {}))
        run_id = str(manifest.get("run_id", "unknown_run"))
        world_id = manifest.get("world_id")
        slice_id = manifest.get("slice_id")
        patch_id = manifest.get("patch_id")
        envelopes: list[dict[str, Any]] = []
        for index, event in enumerate(run_output.get("event_log", [])):
            event_payload = event if isinstance(event, dict) else {"raw_event": event}
            event_type = str(event_payload.get("event_type", "unknown"))
            envelopes.append(
                {
                    "schema_version": EVENT_ENVELOPE_SCHEMA,
                    "event_id": f"{run_id}:event:{index:04d}",
                    "run_id": run_id,
                    "event_type": event_type,
                    "lap": int(event_payload.get("lap", 0)),
                    "segment_id": str(event_payload.get("segment_id", "unknown")),
                    "payload": event_payload,
                    "state_hash_before": None,
                    "state_hash_after": None,
                    "world_id": world_id,
                    "slice_id": slice_id,
                    "patch_id": patch_id,
                }
            )
        return envelopes

    def _build_state_hashes(self, run_output: dict[str, Any]) -> dict[str, Any]:
        """Build deterministic partial state hashes from manifest and event data."""
        manifest = dict(run_output.get("manifest", {}))
        event_log = run_output.get("event_log", [])
        
        initial_state_hash = ""
        final_state_hash = ""
        event_log_hash = ""
        
        # Hash manifest (contains world_id, seed, regulation_id, etc.)
        manifest_str = json.dumps(manifest, sort_keys=True, ensure_ascii=True)
        initial_state_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()[:16]
        
        # Hash final state from last snapshot
        state_snapshots = run_output.get("state_snapshots", [])
        if state_snapshots:
            final_snapshot = state_snapshots[-1]
            final_str = json.dumps(final_snapshot, sort_keys=True, ensure_ascii=True)
            final_state_hash = hashlib.sha256(final_str.encode("utf-8")).hexdigest()[:16]
        else:
            # Fallback: hash manifest + final event
            fallback_str = manifest_str + json.dumps(
                event_log[-1] if event_log else {}, 
                sort_keys=True, 
                ensure_ascii=True
            )
            final_state_hash = hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()[:16]
        
        # Hash event log
        event_str = json.dumps(event_log, sort_keys=True, ensure_ascii=True)
        event_log_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()[:16]
        
        return {
            "initial_state_hash": initial_state_hash,
            "final_state_hash": final_state_hash,
            "event_log_hash": event_log_hash,
        }

    def _build_replay_integrity(self, run_output: dict[str, Any]) -> dict[str, Any]:
        """Build replay integrity declaration - honestly partial for now."""
        return {
            "paired": False,
            "state_hash_coverage": "partial",
            "notes": ["full state snapshot hashing pending"]
        }

    def _extract_unsafe_legal_states(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract unsafe legal state events."""
        events = run_output.get("event_log", [])
        return [
            event for event in events
            if isinstance(event, dict) and event.get("event_type") == "unsafe_legal_state"
        ]

    def _extract_legal_verdicts(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract legal verdicts from validation log."""
        validation_log = run_output.get("action_validation_log", [])
        verdicts = []
        for entry in validation_log:
            if isinstance(entry, dict) and "legal_verdict" in entry:
                verdicts.append(entry["legal_verdict"])
        # Also check steward log for legal assessments
        steward_log = run_output.get("steward_log", [])
        for entry in steward_log:
            if isinstance(entry, dict) and "legal_verdict" in entry:
                verdicts.append(entry["legal_verdict"])
        return verdicts

    def _extract_patch_reruns(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract patch rerun metadata."""
        patch_reruns = run_output.get("patch_reruns", [])
        if isinstance(patch_reruns, list):
            return patch_reruns
        
        # Check counterfactuals if present
        counterfactuals = run_output.get("counterfactuals", [])
        if isinstance(counterfactuals, list):
            patch_reruns.extend(counterfactuals)
        
        return patch_reruns

    def _resolve_segment_focus(self, run_output: dict[str, Any]) -> str:
        """Resolve segment focus from manifest or spec falsification."""
        manifest = dict(run_output.get("manifest", {}))
        spec = run_output.get("spec", {})
        falsification = spec.get("falsification", {}) if isinstance(spec, dict) else {}
        
        segment_focus = manifest.get("segment_focus")
        if segment_focus is not None:
            return str(segment_focus)
        
        segment_focus = falsification.get("segment_focus")
        if segment_focus is not None:
            if isinstance(segment_focus, list) and segment_focus:
                return str(segment_focus[0])
            return str(segment_focus)
        
        return "unknown_segment"
