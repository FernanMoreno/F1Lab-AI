"""Campaign runner for deterministic multiagent races."""

from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from reglabsim.campaigns.ranking import rank_failures
from reglabsim.campaigns.report import campaign_summary, markdown_summary
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.failures.classifier import FailureClassifier
from reglabsim.failures.mitigation import MitigationEngine
from reglabsim.logging.replay import ReplayEngine
from reglabsim.runtime.action_arbitrator import ActionArbitrator
from reglabsim.runtime.action_validator import ActionValidator
from reglabsim.runtime.agents import (
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

        cars = self._build_grid(spec)
        builder = ObservationBuilder()
        arbitrator = ActionArbitrator()
        validator = ActionValidator()
        steward = StewardEngine(enforcement)
        microkernel = RaceMicrokernel(regulation=regulation, seed=spec.seed)

        team_agents, driver_agents = self._build_agents(spec, replay_actions)
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
            actions = {}
            for team_id in sorted({car.team_id for car in cars if not car.retired}):
                team_cars = [car.to_dict() for car in cars if car.team_id == team_id and not car.retired]
                rivals = [car.to_dict() for car in cars if car.team_id != team_id and not car.retired][:4]
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
                prompt_trace_metadata.append({"lap": lap, "team_id": team_id, "mode": team_agents[team_id].mode})
                for car in team_cars:
                    team_order = team_agents[team_id].decide(team_obs, car["car_id"])
                    team_orders[car["car_id"]] = team_order

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
                observation_log.append({"lap": lap, "car_id": car.car_id, "observation": driver_obs.to_dict()})
                prompt_trace_metadata.append({"lap": lap, "car_id": car.car_id, "mode": driver_agents[car.car_id].mode})
                driver_intent = driver_agents[car.car_id].decide(driver_obs)
                action = arbitrator.arbitrate(team_orders[car.car_id], driver_intent, spec.mode)
                validated_action, validation_entry = validator.validate(action, regulation, spec.laps)
                actions[car.car_id] = validated_action
                action_log.append({"lap": lap, "car_id": car.car_id, "action": validated_action.to_dict()})
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
                weather={"visibility_m": weather.visibility_m, "rain_intensity_mm_h": weather.rain_intensity_mm_h},
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

        cars_sorted = sorted(cars, key=lambda item: item.position)
        result = {
            "winner": cars_sorted[0].car_id if cars_sorted else None,
            "final_positions": [car.car_id for car in cars_sorted],
            "retirements": [car.car_id for car in cars if car.retired],
        }
        metrics = self._compute_metrics(event_log, cars_sorted)
        run_output = {
            "manifest": manifest.to_dict(),
            "conditions": {
                "name": spec.conditions.name,
                "weather": vars(weather),
                "track_state": vars(track_state),
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
        run_output["summary_markdown"] = markdown_summary(run_output)
        self._persist_run(spec, run_output)
        return run_output

    def run_campaign(self, spec: CampaignSpec) -> CampaignReport:
        """Run a multi-track or repeated campaign."""
        runs = []
        for track_id in spec.tracks:
            for repetition in range(spec.repetitions):
                runs.append(self.run_race(spec, track_id=track_id))
        ranking = rank_failures(runs)
        summary = campaign_summary(spec.campaign_name, runs, ranking)
        return CampaignReport(
            schema_version="campaign_report.v1",
            campaign_name=spec.campaign_name,
            mode=spec.mode,
            runs=[{"manifest": run["manifest"], "metrics": run["metrics"], "result": run["result"]} for run in runs],
            ranking=ranking,
            summary=summary,
        )

    def propose_mitigations(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate and evaluate simple mitigations on the same scenario."""
        spec = CampaignSpec.from_dict(run_output["spec"])
        candidates = self._mitigations.propose_candidates(run_output.get("failure_log", []))
        evaluated: list[dict[str, Any]] = []
        for candidate in candidates:
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
            evaluated.append(
                {
                    "candidate": candidate,
                    "before_failures": len(run_output.get("failure_log", [])),
                    "after_failures": len(rerun.get("failure_log", [])),
                    "before_incidents": run_output["metrics"]["incident_count"],
                    "after_incidents": rerun["metrics"]["incident_count"],
                    "rerun_manifest": rerun["manifest"],
                }
            )
        evaluated.sort(key=lambda item: (item["after_failures"], item["after_incidents"]))
        return evaluated

    def _build_grid(self, spec: CampaignSpec) -> list[CarRuntimeState]:
        families = list(self._car_families.keys())
        cars = []
        for index in range(spec.num_cars):
            family = families[index % len(families)]
            team_id = f"team_{index // 2 + 1:02d}"
            cars.append(
                CarRuntimeState(
                    car_id=f"car_{index + 1:02d}",
                    driver_id=f"driver_{index + 1:02d}",
                    team_id=team_id,
                    family_id=family,
                    position=index + 1,
                    lap=0,
                    gap_to_leader_s=index * 0.45,
                    gap_ahead_s=0.0 if index == 0 else 0.45,
                    gap_behind_s=0.45,
                    tyre_compound="C3",
                    tyre_age_laps=0,
                    tyre_wear=0.0,
                    ers_soc=min(1.0, 0.72 + (index % 4) * 0.04),
                    fuel_mass_kg=105.0,
                    aero_mode="straight",
                    last_lap_time_s=0.0,
                    cumulative_time_s=0.0,
                    damage=0.0,
                )
            )
        return cars

    def _build_agents(
        self,
        spec: CampaignSpec,
        replay_actions: dict[tuple[int, str], dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        team_agents: dict[str, Any] = {}
        driver_agents: dict[str, Any] = {}
        for team_number in range(1, 12):
            team_id = f"team_{team_number:02d}"
            if spec.mode == "rule_based":
                team_agents[team_id] = RuleBasedTeamAgent()
            else:
                team_agents[team_id] = EventDrivenTeamAgent()
        for car_index in range(1, spec.num_cars + 1):
            car_id = f"car_{car_index:02d}"
            if replay_actions is not None:
                driver_agents[car_id] = PolicyReplayDriverAgent(replay_actions)
            elif spec.mode == "rule_based":
                driver_agents[car_id] = RuleBasedDriverAgent()
            else:
                driver_agents[car_id] = EventDrivenDriverAgent()
        return team_agents, driver_agents

    def _estimated_rival(self, cars: list[CarRuntimeState], car: CarRuntimeState) -> dict[str, Any]:
        rivals = [candidate for candidate in cars if candidate.car_id != car.car_id and not candidate.retired]
        if not rivals:
            return {}
        rival = min(rivals, key=lambda candidate: abs(candidate.position - car.position))
        return {
            "estimated_rival_soc": round(rival.ers_soc, 1),
            "estimated_tyre_wear": round(rival.tyre_wear, 2),
            "visibility_level": "good",
        }

    def _compute_metrics(self, event_log: list[dict[str, Any]], cars: list[CarRuntimeState]) -> dict[str, Any]:
        overtakes = [event for event in event_log if event["event_type"] == "overtake"]
        incidents = [event for event in event_log if event["event_type"] == "incident"]
        attack_events = [event for event in event_log if event["event_type"] == "attack_phase"]
        track_limits = [event for event in event_log if event["event_type"] == "track_limit_breach"]
        avg_damage = round(sum(car.damage for car in cars) / max(len(cars), 1), 4)
        return {
            "total_overtakes": len(overtakes),
            "incident_count": len(incidents),
            "attack_events": len(attack_events),
            "track_limit_breaches": len(track_limits),
            "avg_closing_speed_kph": round(
                sum(event["details"].get("closing_speed_kph", 0.0) for event in overtakes + incidents)
                / max(len(overtakes) + len(incidents), 1),
                3,
            ),
            "avg_damage": avg_damage,
            "retirements": sum(car.retired for car in cars),
        }

    def _persist_run(self, spec: CampaignSpec, run_output: dict[str, Any]) -> None:
        out_dir = Path(spec.output_root) / run_output["manifest"]["run_id"]
        self._replay.save_run(run_output, out_dir)

    def _merge_nested(self, target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge_nested(target[key], value)
            else:
                target[key] = value
