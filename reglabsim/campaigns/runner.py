"""Campaign runner for deterministic multiagent races."""

from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from reglabsim.campaigns.ranking import rank_failures
from reglabsim.campaigns.report import campaign_summary, markdown_summary
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.failures.classifier import FailureClassifier
from reglabsim.failures.mitigation import MitigationEngine
from reglabsim.failures.taxonomy import summarize_failures
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.action_arbitrator import ActionArbitrator
from reglabsim.runtime.action_validator import ActionValidator
from reglabsim.runtime.agents import (
    DeepAgentDriverAgent,
    DeepAgentTeamAgent,
    EventDrivenDriverAgent,
    EventDrivenTeamAgent,
    PolicyReplayDriverAgent,
    RuleBasedDriverAgent,
    RuleBasedTeamAgent,
)
from reglabsim.runtime.microkernel import RaceMicrokernel
from reglabsim.runtime.observation_builder import ObservationBuilder
from reglabsim.runtime.schema import CampaignReport, CarRuntimeState, RunManifest
from reglabsim.steward.engine import StewardEngine
from reglabsim.track.track_loader import TrackRepository


def compare_patch_metrics(
    baseline_metrics: dict[str, Any],
    patched_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute delta between baseline and patched unsafe-legal metrics.

    verdict values:
      "mitigated"  — baseline_count > 0 and patched_count == 0
      "improved"   — patched_count < baseline_count but patched_count > 0
      "unchanged"  — patched_count == baseline_count
      "worse"      — patched_count > baseline_count

    mitigation_success is True only when verdict == "mitigated" (full elimination).
    """

    def _delta(key: str) -> float | None:
        b = baseline_metrics.get(key)
        p = patched_metrics.get(key)
        if isinstance(b, (int, float)) and isinstance(p, (int, float)):
            return round(float(p) - float(b), 6)
        return None

    baseline_count = int(baseline_metrics.get("unsafe_legal_state_count", 0))
    patched_count = int(patched_metrics.get("unsafe_legal_state_count", 0))

    if baseline_count > 0 and patched_count == 0:
        verdict = "mitigated"
    elif patched_count < baseline_count:
        verdict = "improved"
    elif patched_count == baseline_count:
        verdict = "unchanged"
    else:
        verdict = "worse"

    return {
        "unsafe_legal_state_count_delta": patched_count - baseline_count,
        "max_hazard_score_delta": _delta("max_hazard_score"),
        "mean_hazard_score_delta": _delta("mean_hazard_score"),
        "verdict": verdict,
        "mitigation_success": verdict == "mitigated",
    }


class CampaignRunner:
    """Run single races and multi-run campaigns."""

    def __init__(
        self,
        *,
        regulations: dict[str, dict[str, Any]],
        car_families: dict[str, dict[str, Any]],
        track_repository: TrackRepository | None = None,
    ):
        self._regulations = regulations
        self._car_families = car_families
        self._track_repository = track_repository or TrackRepository()
        self._classifier = FailureClassifier()
        self._replay = ReplayEngine()
        self._mitigations = MitigationEngine()

    def run_race(
        self,
        spec: CampaignSpec,
        *,
        track_id: str | None = None,
        replay_actions: dict[tuple[int, str], dict[str, Any]] | None = None,
        regulation_override: dict[str, Any] | None = None,
        enforcement_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single race run."""
        target_track = track_id or spec.tracks[0]
        track = self._track_repository.get(target_track)
        track = self._apply_segment_focus(track=track, spec=spec)
        regulation = deepcopy(self._regulations[spec.regulation])
        enforcement = deepcopy(spec.enforcement)
        if regulation_override:
            self._merge_nested(regulation, regulation_override)
        if enforcement_override:
            self._merge_nested(enforcement, enforcement_override)

        run_id = str(uuid.uuid4())
        manifest = RunManifest(
            run_id=run_id,
            race_name=spec.campaign_name,
            regulation_id=spec.regulation,
            track_id=target_track,
            mode=spec.mode,
            seed=spec.seed,
            config_hash=spec.config_hash(),
            simulator_version="0.2.0",
            schema_version="runtime.v1",
            track_model_version=spec.track_model_version,
            steward_policy_version=spec.steward_policy_version,
            data_version=spec.data_version,
            llm_provider=spec.llm_provider,
            llm_model=spec.llm_model,
            prompt_template_version=spec.prompt_template_version,
        )

        cars: list[CarRuntimeState] = self._build_grid(spec)
        builder = ObservationBuilder()
        arbitrator = ActionArbitrator()
        validator = ActionValidator()
        steward = StewardEngine(enforcement)
        microkernel = RaceMicrokernel(
            regulation=regulation,
            seed=spec.seed,
            battle_calibration=spec.battle_calibration_profile,
            sim_profile=spec.sim_profile,
        )

        team_agents, driver_agents = self._build_agents(spec, replay_actions)
        if spec.conditions is None:
            raise ValueError("CampaignSpec.conditions must be resolved before run_race")
        weather = spec.conditions.weather
        track_state = spec.conditions.track
        forecast = spec.conditions.forecast
        safety_car_active = False

        observation_log: list[dict[str, Any]] = []
        action_log: list[dict[str, Any]] = []
        validation_log: list[dict[str, Any]] = []
        physics_resolution_log: list[dict[str, Any]] = []
        event_log: list[dict[str, Any]] = []
        steward_log: list[dict[str, Any]] = []
        state_snapshots: list[dict[str, Any]] = [
            microkernel.snapshot(
                lap=0,
                total_laps=spec.laps,
                cars=cars,
                weather=weather,
                track_state=track_state,
                safety_car_active=safety_car_active,
            ).to_dict()
        ]
        prompt_trace_metadata: list[dict[str, Any]] = []
        recent_events: list[dict[str, Any]] = []

        for lap in range(1, spec.laps + 1):
            weather, track_state = microkernel.evolve_conditions(
                weather=weather,
                track_state=track_state,
                lap=lap,
                total_laps=spec.laps,
                cars_on_track=sum(not car.retired for car in cars),
                safety_car_active=safety_car_active,
            )

            team_orders: dict[str, Any] = {}
            actions: dict[str, Any] = {}
            for team_id in sorted({car.team_id for car in cars if not car.retired}):
                team_cars = [
                    car.to_dict() for car in cars if car.team_id == team_id and not car.retired
                ]
                rivals = [
                    car.to_dict() for car in cars if car.team_id != team_id and not car.retired
                ][:4]
                team_obs = builder.build_team_observation(
                    team_id=team_id,
                    cars=team_cars,
                    lap=lap,
                    total_laps=spec.laps,
                    forecast=forecast,
                    track_state=track_state,
                    rivals=rivals,
                    recent_events=recent_events,
                )
                prompt_trace_metadata.append(
                    {
                        "lap": lap,
                        "team_id": team_id,
                        "mode": team_agents[team_id].mode,
                        "llm_provider": getattr(team_agents[team_id], "llm_provider", "heuristic"),
                        "llm_model": getattr(team_agents[team_id], "llm_model", "heuristic"),
                    }
                )
                for team_car in team_cars:
                    team_order = team_agents[team_id].decide(team_obs, team_car["car_id"])
                    team_orders[team_car["car_id"]] = team_order

            for car in cars:
                if car.retired:
                    continue
                rival = self._estimated_rival(cars, car)
                driver_obs = builder.build_driver_observation(
                    car_state=car.to_dict(),
                    lap=lap,
                    total_laps=spec.laps,
                    track=track,
                    weather=weather,
                    track_state=track_state,
                    estimated_rival_state=rival,
                    warnings=car.warnings,
                    memory=[],
                )
                observation_log.append(
                    {"lap": lap, "car_id": car.car_id, "observation": driver_obs.to_dict()}
                )
                prompt_trace_metadata.append(
                    {
                        "lap": lap,
                        "car_id": car.car_id,
                        "mode": driver_agents[car.car_id].mode,
                        "llm_provider": getattr(
                            driver_agents[car.car_id], "llm_provider", "heuristic"
                        ),
                        "llm_model": getattr(
                            driver_agents[car.car_id], "llm_model", "heuristic"
                        ),
                    }
                )
                driver_intent = driver_agents[car.car_id].decide(driver_obs)
                action = arbitrator.arbitrate(team_orders[car.car_id], driver_intent, spec.mode)
                validated_action, validation_entry = validator.validate(
                    action, regulation, spec.laps
                )
                actions[car.car_id] = validated_action
                action_log.append(
                    {"lap": lap, "car_id": car.car_id, "action": validated_action.to_dict()}
                )
                validation_log.append(validation_entry)

            cars, lap_events, lap_physics = microkernel.resolve_lap(
                lap=lap,
                total_laps=spec.laps,
                cars=cars,
                actions=actions,
                track=track,
                weather=weather,
                track_state=track_state,
                safety_car_active=safety_car_active,
            )
            decisions = steward.adjudicate(
                lap=lap,
                events=lap_events,
                cars=cars,
                weather={
                    "visibility_m": weather.visibility_m,
                    "rain_intensity_mm_h": weather.rain_intensity_mm_h,
                },
            )

            physics_resolution_log.extend(lap_physics)
            event_log.extend(event.to_dict() for event in lap_events)
            steward_log.extend(decision.to_dict() for decision in decisions)
            safety_car_active = any(event.event_type == "incident" for event in lap_events)
            recent_events = [event.to_dict() for event in lap_events][-10:]
            state_snapshots.append(
                microkernel.snapshot(
                    lap=lap,
                    total_laps=spec.laps,
                    cars=cars,
                    weather=weather,
                    track_state=track_state,
                    safety_car_active=safety_car_active,
                ).to_dict()
            )

        steward_log.extend(
            decision.to_dict()
            for decision in steward.flush_pending(final_lap=spec.laps, cars=cars)
        )
        cars_sorted = sorted(cars, key=lambda item: item.cumulative_time_s + item.penalties_s)
        result = {
            "winner": cars_sorted[0].car_id if cars_sorted else None,
            "final_positions": [car.car_id for car in cars_sorted],
            "retirements": [car.car_id for car in cars if car.retired],
        }
        metrics = self._compute_metrics(event_log, cars_sorted)
        run_output = {
            "manifest": manifest.to_dict(),
            "track_provenance": {
                "track_id": track.track_id,
                "sources": track.sources,
                "validation_status": track.validation_status,
                "fidelity_notes": track.fidelity_notes,
                "metadata": track.metadata,
            },
            "conditions": {
                "name": spec.conditions.name,
                "weather": vars(weather),
                "track_state": vars(track_state),
                "metadata": spec.conditions.metadata,
            },
            "enforcement": enforcement,
            "observation_log": observation_log,
            "action_log": action_log,
            "action_validation_log": validation_log,
            "physics_resolution_log": physics_resolution_log,
            "event_log": event_log,
            "steward_log": steward_log,
            "failure_log": [],
            "state_snapshots": state_snapshots,
            "metrics": metrics,
            "prompt_trace_metadata": prompt_trace_metadata,
            "result": result,
            "summary_markdown": "",
            "spec": spec.to_dict(),
        }
        failures = self._classifier.classify(run_output)
        run_output["failure_log"] = [failure.to_dict() for failure in failures]
        world_manifest = self._build_world_manifest(
            spec=spec,
            track_id=target_track,
            cars=cars,
            regulation=regulation,
            enforcement=enforcement,
        )
        run_output["world_manifest"] = world_manifest
        scores = self._score_run_output(run_output)
        manifest_payload = run_output["manifest"]
        if not isinstance(manifest_payload, dict):
            raise TypeError("run_output['manifest'] must be a mapping")
        manifest_payload.update(
            {
                "world_id": world_manifest["world_id"],
                "slice_id": self._slice_id(spec, target_track),
                "patch_id": None,
                **scores,
            }
        )
        run_output["event_envelopes"] = self._build_event_envelopes(run_output)
        run_output["summary_markdown"] = markdown_summary(run_output)
        self._persist_run(spec, run_output)
        return run_output

    def run_campaign(self, spec: CampaignSpec) -> CampaignReport:
        """Run a multi-track or repeated campaign."""
        runs = []
        for track_id in spec.tracks:
            for _repetition in range(spec.repetitions):
                runs.append(self.run_race(spec, track_id=track_id))
        ranking = rank_failures(runs)
        summary = campaign_summary(spec.campaign_name, runs, ranking)
        return CampaignReport(
            schema_version="campaign_report.v1",
            campaign_name=spec.campaign_name,
            mode=spec.mode,
            runs=[
                {"manifest": run["manifest"], "metrics": run["metrics"], "result": run["result"]}
                for run in runs
            ],
            ranking=ranking,
            summary=summary,
        )

    def propose_mitigations(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate and evaluate simple mitigations on the same scenario."""
        candidates = self._mitigations.propose_candidates(run_output.get("failure_log", []))
        evaluated: list[dict[str, Any]] = []
        for candidate in candidates:
            evaluated.append(self.evaluate_patch(run_output, candidate))
        evaluated.sort(
            key=lambda item: (
                item["after_priority_score"],
                item["after_failures"],
                item["after_incidents"],
                -item["priority_delta"],
            )
        )
        return evaluated

    def evaluate_patch(
        self,
        run_output: dict[str, Any],
        patch: dict[str, Any] | str,
    ) -> dict[str, Any]:
        """Re-run one saved world with an explicit patch candidate."""
        spec = CampaignSpec.from_dict(run_output["spec"])
        candidate = self._resolve_patch_candidate(patch)
        rerun = self._rerun_with_candidate(run_output=run_output, spec=spec, candidate=candidate)
        before_summary = summarize_failures(run_output.get("failure_log", []))
        after_summary = summarize_failures(rerun.get("failure_log", []))
        return {
            "candidate": candidate,
            "before_failures": len(run_output.get("failure_log", [])),
            "after_failures": len(rerun.get("failure_log", [])),
            "before_priority_score": before_summary["total_priority_score"],
            "after_priority_score": after_summary["total_priority_score"],
            "priority_delta": round(
                before_summary["total_priority_score"] - after_summary["total_priority_score"],
                4,
            ),
            "before_incidents": run_output["metrics"]["incident_count"],
            "after_incidents": rerun["metrics"]["incident_count"],
            "before_summary": before_summary,
            "after_summary": after_summary,
            "rerun_manifest": rerun["manifest"],
            "rerun_output": rerun,
        }

    def _build_grid(self, spec: CampaignSpec) -> list[CarRuntimeState]:
        families = list(self._car_families.keys())
        calibration_profile = spec.battle_calibration_profile or {}
        uses_calibration_grid = any(
            key in calibration_profile
            for key in (
                "grid_gap_scale",
                "grid_gap_growth",
                "grid_sync_cumulative",
            )
        )
        base_gap_s = 0.45 * float(calibration_profile.get("grid_gap_scale", 1.0))
        gap_growth = float(calibration_profile.get("grid_gap_growth", 0.0))
        sync_cumulative = bool(calibration_profile.get("grid_sync_cumulative", False))
        cars = []
        previous_gap_to_leader = 0.0
        for index in range(spec.num_cars):
            family = families[index % len(families)]
            team_id = f"team_{index // 2 + 1:02d}"
            if index == 0:
                gap_ahead_s = 0.0
                gap_to_leader_s = 0.0
            elif uses_calibration_grid:
                gap_ahead_s = round(
                    base_gap_s * (1.0 + gap_growth * max(index - 1, 0)),
                    3,
                )
                gap_to_leader_s = round(previous_gap_to_leader + gap_ahead_s, 3)
            else:
                gap_ahead_s = 0.45
                gap_to_leader_s = round(index * 0.45, 3)
            cars.append(
                CarRuntimeState(
                    car_id=f"car_{index + 1:02d}",
                    driver_id=f"driver_{index + 1:02d}",
                    team_id=team_id,
                    family_id=family,
                    position=index + 1,
                    lap=0,
                    gap_to_leader_s=gap_to_leader_s,
                    gap_ahead_s=gap_ahead_s,
                    gap_behind_s=0.45,
                    tyre_compound="C3",
                    tyre_age_laps=0,
                    tyre_wear=0.0,
                    ers_soc=min(1.0, 0.72 + (index % 4) * 0.04),
                    fuel_mass_kg=105.0,
                    aero_mode="straight",
                    last_lap_time_s=0.0,
                    cumulative_time_s=gap_to_leader_s if sync_cumulative else 0.0,
                    damage=0.0,
                )
            )
            previous_gap_to_leader = gap_to_leader_s
        for index, car in enumerate(cars):
            car.gap_behind_s = (
                round(cars[index + 1].gap_ahead_s, 3) if index < len(cars) - 1 else 999.0
            )
        return cars

    def _build_agents(
        self,
        spec: CampaignSpec,
        replay_actions: dict[tuple[int, str], dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        team_agents: dict[str, Any] = {}
        driver_agents: dict[str, Any] = {}
        use_deep_agents = spec.mode != "rule_based" and spec.llm_provider != "heuristic"
        for team_number in range(1, 12):
            team_id = f"team_{team_number:02d}"
            if spec.mode == "rule_based":
                team_agents[team_id] = RuleBasedTeamAgent()
            elif use_deep_agents:
                team_agents[team_id] = DeepAgentTeamAgent(
                    llm_provider=spec.llm_provider,
                    llm_model=spec.llm_model,
                    prompt_template_version=spec.prompt_template_version,
                )
            else:
                team_agents[team_id] = EventDrivenTeamAgent()
        for car_index in range(1, spec.num_cars + 1):
            car_id = f"car_{car_index:02d}"
            if replay_actions is not None:
                driver_agents[car_id] = PolicyReplayDriverAgent(replay_actions)
            elif spec.mode == "rule_based":
                driver_agents[car_id] = RuleBasedDriverAgent()
            elif use_deep_agents:
                driver_agents[car_id] = DeepAgentDriverAgent(
                    llm_provider=spec.llm_provider,
                    llm_model=spec.llm_model,
                    prompt_template_version=spec.prompt_template_version,
                )
            else:
                driver_agents[car_id] = EventDrivenDriverAgent()
        return team_agents, driver_agents

    def _estimated_rival(self, cars: list[CarRuntimeState], car: CarRuntimeState) -> dict[str, Any]:
        rivals = [
            candidate
            for candidate in cars
            if candidate.car_id != car.car_id and not candidate.retired
        ]
        if not rivals:
            return {}
        rival = min(rivals, key=lambda candidate: abs(candidate.position - car.position))
        return {
            "estimated_rival_soc": round(rival.ers_soc, 1),
            "estimated_tyre_wear": round(rival.tyre_wear, 2),
            "visibility_level": "good",
        }

    def _compute_metrics(
        self, event_log: list[dict[str, Any]], cars: list[CarRuntimeState]
    ) -> dict[str, Any]:
        overtakes = [event for event in event_log if event["event_type"] == "overtake"]
        incidents = [event for event in event_log if event["event_type"] == "incident"]
        near_misses = [event for event in event_log if event["event_type"] == "near_miss"]
        warnings = [event for event in event_log if event["event_type"] == "warning"]
        minor_contacts = [event for event in event_log if event["event_type"] == "minor_contact"]
        major_contacts = [event for event in event_log if event["event_type"] == "major_contact"]
        attack_events = [event for event in event_log if event["event_type"] == "attack_phase"]
        track_limits = [event for event in event_log if event["event_type"] == "track_limit_breach"]
        unsafe_defending = [
            event for event in event_log if event["event_type"] == "unsafe_defending"
        ]
        forcing_off_track = [
            event for event in event_log if event["event_type"] == "forcing_off_track"
        ]
        avg_damage = round(sum(car.damage for car in cars) / max(len(cars), 1), 4)
        physical_contacts = minor_contacts + major_contacts + incidents
        return {
            "total_overtakes": len(overtakes),
            "incident_count": len(incidents),
            "near_miss_count": len(near_misses),
            "warning_count": len(warnings),
            "minor_contact_count": len(minor_contacts),
            "major_contact_count": len(major_contacts),
            "physical_contact_count": len(physical_contacts),
            "attack_events": len(attack_events),
            "track_limit_breaches": len(track_limits),
            "unsafe_defending_events": len(unsafe_defending),
            "forcing_off_track_events": len(forcing_off_track),
            "avg_closing_speed_kph": round(
                sum(
                    event["details"].get("closing_speed_kph", 0.0)
                    for event in overtakes + incidents
                )
                / max(len(overtakes) + len(incidents), 1),
                3,
            ),
            "avg_damage": avg_damage,
            "retirements": sum(car.retired for car in cars),
        }

    def _persist_run(self, spec: CampaignSpec, run_output: dict[str, Any]) -> None:
        out_dir = Path(spec.output_root) / run_output["manifest"]["run_id"]
        self._replay.save_run(run_output, out_dir)

    def _apply_segment_focus(self, *, track: Any, spec: CampaignSpec) -> Any:
        falsification = getattr(spec, "falsification", {})
        if not isinstance(falsification, dict):
            return track
        focus_segments = [str(item) for item in falsification.get("segment_focus", []) if item]
        if not focus_segments:
            return track
        focus_set = set(focus_segments)
        segments = [
            replace(
                segment,
                preferred_battle_zone=segment.segment_id in focus_set,
            )
            for segment in track.segments
        ]
        metadata = dict(track.metadata)
        metadata["slice_segment_focus"] = focus_segments
        return replace(track, segments=segments, metadata=metadata)

    def _rerun_with_candidate(
        self,
        *,
        run_output: dict[str, Any],
        spec: CampaignSpec,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        regulation_override, enforcement_override = self._mitigations.apply_overrides(
            base_regulation=self._regulations[spec.regulation],
            base_enforcement=spec.enforcement,
            candidate=candidate,
        )
        rerun = self.run_race(
            spec,
            track_id=run_output["manifest"]["track_id"],
            regulation_override=regulation_override,
            enforcement_override=enforcement_override,
        )
        rerun["manifest"]["patch_id"] = str(candidate.get("name", "adhoc_patch"))
        return rerun

    def _resolve_patch_candidate(self, patch: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(patch, dict):
            return patch
        catalog = {
            "depleted_car_signal": {
                "name": "depleted_car_signal",
                "description": "Force high-visibility signaling for low-energy cars",
                "failure_targets": ["unsafe_closing_speed", "grey_area_exploit"],
                "regulation_overrides": {
                    "sporting": {
                        "mandatory_depleted_car_signal": True,
                        "depleted_signal_soc_threshold": 0.22,
                    }
                },
                "enforcement_overrides": {
                    "detection_probability": {"unsafe_closing_speed": 0.92},
                },
                "expected_tradeoffs": ["higher detectability", "possible signaling games"],
            },
            "closing_speed_warning": {
                "name": "closing_speed_warning",
                "description": "Emit early closing-speed warning in high-risk battle zones",
                "failure_targets": ["unsafe_closing_speed", "no_escape_zone_failure"],
                "regulation_overrides": {
                    "sporting": {
                        "closing_speed_warning_enabled": True,
                        "closing_speed_warning_threshold_kph": 42.0,
                    }
                },
                "enforcement_overrides": {
                    "detection_probability": {"unsafe_closing_speed": 0.95},
                },
                "expected_tradeoffs": ["lower surprise delta", "more conservative attacks"],
            },
            "boost_derating_fast_corners": {
                "name": "boost_derating_fast_corners",
                "description": "Reduce deploy cap through fast-corner attack windows",
                "failure_targets": ["unsafe_closing_speed", "wind_active_aero_instability"],
                "regulation_overrides": {"power_unit": {"ers_deployment_max_kw": 215.0}},
                "enforcement_overrides": {},
                "expected_tradeoffs": ["less peak attack delta", "lower artificial passing"],
            },
            "sustained_overtake_eligibility": {
                "name": "sustained_overtake_eligibility",
                "description": "Smooth overtake eligibility to reduce abrupt energy spikes",
                "failure_targets": ["battery_dominance", "unsafe_closing_speed"],
                "regulation_overrides": {
                    "overtake_mode": {
                        "activation_gap_s": 1.2,
                        "sustained_eligibility_window_s": 4.0,
                    }
                },
                "enforcement_overrides": {},
                "expected_tradeoffs": ["more stable attack prep", "less burst exploitation"],
            },
            "closing_speed_cap_v1": {
                "name": "closing_speed_cap_v1",
                "patch_type": "closing_speed_cap",
                "description": (
                    "Cap effective delta speed fed to SafetyOracle before evaluation. "
                    "Causally reduces hazard score by limiting amplified closing speed."
                ),
                "failure_targets": ["unsafe_closing_speed", "unsafe_legal_state"],
                "regulation_overrides": {"safety": {"closing_speed_cap_kph": 65.0}},
                "enforcement_overrides": {},
                "expected_tradeoffs": [
                    "lower effective hazard score",
                    "fewer unsafe_legal_state emissions",
                ],
            },
        }
        if patch in catalog:
            return catalog[patch]
        return {
            "name": str(patch),
            "description": f"Ad-hoc patch '{patch}'",
            "failure_targets": ["generic_race_instability"],
            "regulation_overrides": {},
            "enforcement_overrides": {},
            "expected_tradeoffs": [],
        }

    def _build_world_manifest(
        self,
        *,
        spec: CampaignSpec,
        track_id: str,
        cars: list[CarRuntimeState],
        regulation: dict[str, Any],
        enforcement: dict[str, Any],
    ) -> dict[str, Any]:
        car_families = sorted({car.family_id for car in cars})
        world_seed_material = {
            "campaign": spec.campaign_name,
            "track_id": track_id,
            "seed": spec.seed,
            "mode": spec.mode,
            "sim_profile": getattr(spec, "sim_profile", "public_baseline"),
            "car_families": car_families,
            "falsification": getattr(spec, "falsification", {}),
        }
        world_id = hashlib.sha256(
            json.dumps(world_seed_material, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "schema_version": "world_manifest.v1",
            "world_id": world_id,
            "seed": spec.seed,
            "track_id": track_id,
            "regulation_id": spec.regulation,
            "mode": spec.mode,
            "sim_profile": getattr(spec, "sim_profile", "public_baseline"),
            "car_families": car_families,
            "num_cars": spec.num_cars,
            "laps": spec.laps,
            "conditions_profile": spec.weather_profile,
            "falsification": deepcopy(getattr(spec, "falsification", {})),
            "regulation_constants": {
                "ers_deployment_max_kw": regulation.get("power_unit", {}).get(
                    "ers_deployment_max_kw"
                ),
                "ers_max_energy_mj": regulation.get("power_unit", {}).get("ers_max_energy_mj"),
                "active_aero_modes": regulation.get("active_aero", {}).get("modes", []),
                "overtake_activation_gap_s": regulation.get("overtake_mode", {}).get(
                    "activation_gap_s"
                ),
            },
            "enforcement": deepcopy(enforcement),
        }

    def _score_run_output(self, run_output: dict[str, Any]) -> dict[str, float]:
        metrics = run_output.get("metrics", {})
        failures = run_output.get("failure_log", [])
        unsafe_events = sum(
            1
            for event in run_output.get("event_log", [])
            if event.get("event_type")
            in {"unsafe_legal_state", "unsafe_defending", "forcing_off_track"}
        )
        near_miss_count = int(metrics.get("near_miss_count", 0))
        track_limits = int(metrics.get("track_limit_breaches", 0))
        physical_contacts = int(metrics.get("physical_contact_count", 0))
        total_overtakes = int(metrics.get("total_overtakes", 0))
        public_anchor_score = max(
            0.0,
            min(
                1.0,
                0.72
                - abs(total_overtakes - 8) * 0.02
                - max(0, physical_contacts - 3) * 0.04
                - max(0, track_limits - 4) * 0.015,
            ),
        )
        baseline_plausibility_score = max(
            0.0,
            min(
                1.0,
                0.45
                + min(near_miss_count, 6) * 0.04
                + min(total_overtakes, 12) * 0.015
                - max(0, physical_contacts - 5) * 0.03,
            ),
        )
        regulation_breaking_score = max(
            0.0,
            min(
                1.0,
                0.18
                + min(unsafe_events, 8) * 0.08
                + min(len(failures), 6) * 0.06
                + min(near_miss_count, 6) * 0.04,
            ),
        )
        return {
            "public_anchor_score": round(public_anchor_score, 4),
            "baseline_plausibility_score": round(baseline_plausibility_score, 4),
            "regulation_breaking_score": round(regulation_breaking_score, 4),
        }

    def _build_event_envelopes(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        snapshots = run_output.get("state_snapshots", [])
        envelopes: list[dict[str, Any]] = []
        for index, event in enumerate(run_output.get("event_log", []), start=1):
            lap = int(event.get("lap", 0))
            before = snapshots[max(0, lap - 1)] if snapshots else {}
            after = snapshots[min(lap, max(len(snapshots) - 1, 0))] if snapshots else {}
            envelopes.append(
                {
                    "schema_version": "event_envelope.v1",
                    "run_id": run_output["manifest"]["run_id"],
                    "event_id": f"evt_{index:05d}",
                    "lap": lap,
                    "segment_id": event.get("segment_id"),
                    "event_type": event.get("event_type"),
                    "state_hash_before": self._state_hash(before),
                    "state_hash_after": self._state_hash(after),
                    "payload": deepcopy(event),
                }
            )
        return envelopes

    def _slice_id(self, spec: CampaignSpec, track_id: str) -> str:
        falsification = getattr(spec, "falsification", {})
        explicit = falsification.get("slice_id") if isinstance(falsification, dict) else None
        if explicit:
            return str(explicit)
        return f"{spec.campaign_name}:{track_id}"

    @staticmethod
    def _state_hash(snapshot: dict[str, Any]) -> str:
        payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _merge_nested(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_nested(target[key], value)
            else:
                target[key] = value
