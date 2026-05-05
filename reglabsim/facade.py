"""Unified simulation facade for experiments, races and campaigns."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from reglabsim.campaigns.runner import CampaignRunner
from reglabsim.campaigns.spec import CampaignSpec
from reglabsim.failures.classifier import FailureClassifier
from reglabsim.logging.replay import ReplayEngine
from reglabsim.regulation.base import Regulation
from reglabsim.track.track_loader import TrackRepository
from reglabsim.vehicle.car_family import CarFamily


class SimulationFacadeImpl:
    """Main facade for deterministic and multiagent simulation flows."""

    def __init__(
        self,
        config_dir: Path | str = "configs",
        regulation_dir: Path | str | None = None,
        car_families_path: Path | str | None = None,
        data_dir: Path | str | None = None,
    ):
        self._config_dir = Path(config_dir)
        self._regulation_dir = Path(regulation_dir) if regulation_dir else self._config_dir / "regulations"
        self._car_families_path = (
            Path(car_families_path) if car_families_path else self._config_dir / "car_families.yaml"
        )
        self._data_dir = Path(data_dir) if data_dir else Path("outputs")
        self._track_repo = TrackRepository(self._config_dir / "tracks")
        self._regulation_registry: dict[str, Regulation] = {}
        self._regulation_payloads: dict[str, dict[str, Any]] = {}
        self._car_family_registry: dict[str, CarFamily] = {}
        self._car_family_payloads: dict[str, dict[str, Any]] = {}
        self._replay = ReplayEngine()
        self._failure_classifier = FailureClassifier()

    # ------------------------------------------------------------------
    # Registry loading
    # ------------------------------------------------------------------

    def _ensure_regulations_loaded(self) -> None:
        if self._regulation_registry:
            return
        if not self._regulation_dir.exists():
            return
        for reg_file in self._regulation_dir.glob("*.yaml"):
            with open(reg_file, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            reg_id = data.get("name", reg_file.stem)
            self._regulation_payloads[reg_id] = data
            self._regulation_registry[reg_id] = Regulation(
                name=data.get("name", reg_id),
                version=data.get("version", "0.0"),
                status=data.get("status", "unknown"),
                power_unit=data.get("power_unit", {}),
                active_aero=data.get("active_aero", {}),
                aero=data.get("aero", {}),
                tyres=data.get("tyres", {}),
                safety=data.get("safety", {}),
                weights=data.get("weights", {}),
                sessions=data.get("sessions", {}),
                assumptions=data.get("assumptions", []),
            )

    def _ensure_car_families_loaded(self) -> None:
        if self._car_family_registry:
            return
        if not self._car_families_path.exists():
            return
        with open(self._car_families_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        for family_id, family_data in data.get("car_families", {}).items():
            self._car_family_payloads[family_id] = family_data
            self._car_family_registry[family_id] = CarFamily(
                family_id=family_id,
                description=family_data.get("description", ""),
                mass_kg=family_data.get("mass_kg", 780.0),
                cda_straight_m2=family_data.get("cda_straight_m2", 0.9),
                cda_corner_m2=family_data.get("cda_corner_m2", 1.2),
                cla_straight_m2=family_data.get("cla_straight_m2", 2.2),
                cla_corner_m2=family_data.get("cla_corner_m2", 3.8),
                power_kw=family_data.get("power_kw", 740.0),
                ers_efficiency=family_data.get("ers_efficiency", 0.75),
                tyre_deg_factor=family_data.get("tyre_deg_factor", 1.0),
                dirty_air_sensitivity=family_data.get("dirty_air_sensitivity", 0.15),
                strength=family_data.get("strength", []),
                weakness=family_data.get("weakness", []),
            )

    def _campaign_runner(self) -> CampaignRunner:
        self._ensure_regulations_loaded()
        self._ensure_car_families_loaded()
        regulation_payloads = {key: deepcopy(value) for key, value in self._regulation_payloads.items()}
        car_family_payloads = {key: deepcopy(value) for key, value in self._car_family_payloads.items()}
        return CampaignRunner(
            regulations=regulation_payloads,
            car_families=car_family_payloads,
            track_repository=self._track_repo,
        )

    # ------------------------------------------------------------------
    # Public registry API
    # ------------------------------------------------------------------

    def list_regulations(self) -> list[str]:
        self._ensure_regulations_loaded()
        return list(self._regulation_registry.keys())

    def load_regulation(self, regulation_id: str) -> Regulation:
        self._ensure_regulations_loaded()
        if regulation_id not in self._regulation_registry:
            raise KeyError(f"Regulation '{regulation_id}' not found")
        return self._regulation_registry[regulation_id]

    def list_car_families(self) -> list[str]:
        self._ensure_car_families_loaded()
        return list(self._car_family_registry.keys())

    def load_car_family(self, family_id: str) -> CarFamily:
        self._ensure_car_families_loaded()
        if family_id not in self._car_family_registry:
            raise KeyError(f"Car family '{family_id}' not found")
        return self._car_family_registry[family_id]

    def list_circuits(self) -> list[str]:
        return self._track_repo.list_ids()

    # ------------------------------------------------------------------
    # Legacy experiment compatibility
    # ------------------------------------------------------------------

    def run_lap_experiment(
        self,
        config_path: str | Path,
        regulation_id: str,
        car_family_id: str,
        circuit_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        spec = CampaignSpec.from_dict(
            {
                "campaign_name": "lap_experiment",
                "regulation": regulation_id,
                "track": circuit_id,
                "num_cars": 1,
                "laps": 1,
                "mode": "rule_based",
                "seed": seed or 42,
            }
        )
        result = self._campaign_runner().run_race(spec, track_id=circuit_id)
        final_car = result["state_snapshots"][-1]["cars"][0]
        return {
            "experiment_name": "lap_experiment",
            "regulation_id": regulation_id,
            "car_family_id": car_family_id,
            "circuit_id": circuit_id,
            "seed": spec.seed,
            "lap_time_s": final_car["last_lap_time_s"],
            "sector_times": [final_car["last_lap_time_s"] / 3.0] * 3,
            "speed_trace": [],
            "energy_used_mj": (1.0 - final_car["ers_soc"]) * 6.0,
            "energy_recovered_mj": max(0.0, final_car["ers_soc"] * 1.5),
            "top_speed_kph": self._track_repo.get(circuit_id).avg_speed_kph + 35.0,
        }

    def run_battle_experiment(self, config_path: str | Path, seed: int | None = None) -> dict[str, Any]:
        raw = self._load_yaml(config_path)
        spec = CampaignSpec.from_dict(
            {
                "campaign_name": raw.get("experiment_name", "battle_experiment"),
                "description": raw.get("description", ""),
                "regulation": raw.get("regulation", "regulation_2026_refined"),
                "track": raw.get("track", "baku"),
                "num_cars": 2,
                "laps": raw.get("simulation", {}).get("laps", 8),
                "mode": "llm_event_driven",
                "seed": seed if seed is not None else raw.get("simulation", {}).get("seed", 42),
                "conditions": raw.get("conditions", {}),
                "objectives": raw.get("metrics", []),
            }
        )
        run = self._campaign_runner().run_race(spec)
        overtakes = [
            event["details"] | {"lap": event["lap"], "type": event["event_type"]}
            for event in run["event_log"]
            if event["event_type"] in {"overtake", "incident"}
        ]
        return {
            "experiment_name": spec.campaign_name,
            "seed": spec.seed,
            "num_overtakes": len([event for event in run["event_log"] if event["event_type"] == "overtake"]),
            "overtakes": overtakes,
            "max_closing_speed_kph": max((event.get("closing_speed_kph", 0.0) for event in overtakes), default=0.0),
            "dangerous_closing_speed_index": sum(
                1 for event in overtakes if event.get("closing_speed_kph", 0.0) > 55.0
            )
            / max(len(overtakes), 1),
            "train_formation_index": max(0.0, 1.0 - len(overtakes) / max(spec.laps, 1)),
            "attacker_win_rate": 1.0 if run["result"]["winner"] == "car_01" else 0.0,
            "_run_output": run,
        }

    def run_race_experiment(self, config_path: str | Path, seed: int | None = None) -> dict[str, Any]:
        raw = self._load_yaml(config_path)
        spec = CampaignSpec.from_dict(raw | {"seed": seed if seed is not None else raw.get("seed", 42)})
        return self._campaign_runner().run_race(spec)

    # ------------------------------------------------------------------
    # New multiagent API
    # ------------------------------------------------------------------

    def run_multiagent_race(
        self,
        config_path: str | Path,
        mode: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        raw = self._load_yaml(config_path)
        if mode is not None:
            raw["mode"] = mode
        if seed is not None:
            raw["seed"] = seed
        spec = CampaignSpec.from_dict(raw)
        return self._campaign_runner().run_race(spec)

    def run_redteam_campaign(self, config_path: str | Path, budget: int | None = None) -> dict[str, Any]:
        raw = self._load_yaml(config_path)
        if budget is not None:
            raw["repetitions"] = budget
        spec = CampaignSpec.from_dict(raw)
        return self._campaign_runner().run_campaign(spec).to_dict()

    def replay_race(
        self,
        run_output_or_path: dict[str, Any] | str | Path,
        mode: str = "replay_audit_exact",
    ) -> dict[str, Any]:
        run_output = (
            self._replay.load_run(run_output_or_path)
            if isinstance(run_output_or_path, (str, Path))
            else run_output_or_path
        )
        if mode == "replay_audit_exact":
            return self._replay.replay_audit_exact(run_output)

        replay_actions = self._replay.extract_policy_replay_actions(run_output)
        spec = CampaignSpec.from_dict(run_output["spec"] | {"mode": "policy_replay"})
        rerun = self._campaign_runner().run_race(
            spec,
            track_id=run_output["manifest"]["track_id"],
            replay_actions=replay_actions,
        )
        return {"mode": "replay_resimulate", "original": run_output["manifest"], "rerun": rerun["manifest"], "result": rerun["result"]}

    def classify_failures(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        return [failure.to_dict() for failure in self._failure_classifier.classify(run_output)]

    def propose_mitigations(self, run_output: dict[str, Any]) -> list[dict[str, Any]]:
        return self._campaign_runner().propose_mitigations(run_output)

    def compare_regulations(
        self,
        regulation_a: str,
        regulation_b: str,
        experiment_config: str | Path,
        n_repetitions: int = 3,
        seed: int | None = None,
    ) -> dict[str, Any]:
        raw = self._load_yaml(experiment_config)
        raw["repetitions"] = max(1, min(n_repetitions, 5))
        if seed is not None:
            raw["seed"] = seed
        spec = CampaignSpec.from_dict(raw)

        metrics_a = []
        metrics_b = []
        runner = self._campaign_runner()
        for index in range(spec.repetitions):
            spec.seed = (seed or spec.seed) + index
            run_a = runner.run_race(CampaignSpec.from_dict(spec.to_dict() | {"regulation": regulation_a}))
            run_b = runner.run_race(CampaignSpec.from_dict(spec.to_dict() | {"regulation": regulation_b}))
            metrics_a.append(run_a["metrics"])
            metrics_b.append(run_b["metrics"])

        def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
            return {
                "avg_overtakes": sum(item["total_overtakes"] for item in items) / len(items),
                "avg_incidents": sum(item["incident_count"] for item in items) / len(items),
                "avg_closing_speed": sum(item["avg_closing_speed_kph"] for item in items) / len(items),
            }

        return {
            "regulation_a": regulation_a,
            "regulation_b": regulation_b,
            "n_repetitions": spec.repetitions,
            "regulation_a_metrics": aggregate(metrics_a),
            "regulation_b_metrics": aggregate(metrics_b),
        }

    def compute_metrics(self, simulation_output: dict[str, Any], metric_names: list[str] | None = None) -> dict[str, Any]:
        metrics = simulation_output.get("metrics")
        if metrics is None and "_run_output" in simulation_output:
            metrics = simulation_output["_run_output"].get("metrics")
        if metrics is None and "overtakes" in simulation_output:
            overtakes = simulation_output.get("overtakes", [])
            metrics = {
                "total_overtakes": len(overtakes),
                "avg_closing_speed_kph": sum(item.get("closing_speed_kph", 0.0) for item in overtakes)
                / max(len(overtakes), 1),
            }
        if metrics is not None:
            return metrics if metric_names is None else {name: metrics.get(name) for name in metric_names}
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)


def create_facade(config_dir: str | Path = "configs", **kwargs: Any) -> SimulationFacadeImpl:
    """Create a simulation facade."""
    return SimulationFacadeImpl(config_dir=config_dir, **kwargs)


__all__ = ["SimulationFacadeImpl", "create_facade"]
