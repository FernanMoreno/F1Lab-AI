"""CLI entrypoint for F1Lab-AI."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from reglabsim.facade import create_facade

app = typer.Typer(add_completion=False, help="F1Lab-AI simulation CLI")


@app.command("ingest-session-data")
def ingest_session_data(
    year: int = typer.Argument(..., help="Season year"),
    track_id: str = typer.Argument(..., help="Track id, e.g. suzuka"),
    session_type: str = typer.Argument(..., help="Session type, e.g. race or quali"),
    drivers: str = typer.Option("", help="Comma-separated driver numbers"),
    data_root: str = typer.Option("data", help="Local data-lake root"),
) -> None:
    facade = create_facade()
    driver_numbers = [int(value) for value in drivers.split(",") if value.strip()]
    result = facade.ingest_public_session_data(
        year=year,
        track_id=track_id,
        session_type=session_type,
        driver_numbers=driver_numbers,
        data_root=data_root,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("ingest-weekend-results")
def ingest_weekend_results(
    season: int = typer.Argument(..., help="Season year"),
    round_num: int = typer.Argument(..., help="Round number"),
    data_root: str = typer.Option("data", help="Local data-lake root"),
) -> None:
    facade = create_facade()
    result = facade.ingest_public_weekend_results(
        season=season,
        round_num=round_num,
        data_root=data_root,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("ingest-historical-weather")
def ingest_historical_weather(
    track_id: str = typer.Argument(..., help="Track id"),
    start_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    end_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    data_root: str = typer.Option("data", help="Local data-lake root"),
) -> None:
    facade = create_facade()
    result = facade.ingest_historical_weather(
        track_id=track_id,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("build-weather-profile")
def build_weather_profile(
    track_id: str = typer.Argument(..., help="Track id"),
    start_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    end_date: str = typer.Argument(..., help="YYYY-MM-DD"),
    profile_id: str | None = typer.Option(None, help="Optional profile id"),
    data_root: str = typer.Option("data", help="Local data-lake root"),
    save_profile: bool = typer.Option(True, help="Persist generated YAML profile"),
) -> None:
    facade = create_facade()
    result = facade.build_weather_profile(
        track_id=track_id,
        start_date=start_date,
        end_date=end_date,
        profile_id=profile_id,
        save_profile=save_profile,
        data_root=data_root,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("describe-track")
def describe_track(track_id: str = typer.Argument(..., help="Track id")) -> None:
    facade = create_facade()
    typer.echo(json.dumps(facade.describe_track(track_id), indent=2))


@app.command("show-condition-profile")
def show_condition_profile(profile_id: str = typer.Argument(..., help="Condition profile id")) -> None:
    facade = create_facade()
    typer.echo(json.dumps(facade.load_condition_profile(profile_id), indent=2))


@app.command("run-multiagent-race")
def run_multiagent_race(
    config: Path = typer.Argument(..., exists=True, readable=True),
    mode: str | None = typer.Option(None, help="Override agent mode"),
    seed: int | None = typer.Option(None, help="Override random seed"),
) -> None:
    facade = create_facade()
    result = facade.run_multiagent_race(config, mode=mode, seed=seed)
    typer.echo(json.dumps({"manifest": result["manifest"], "result": result["result"], "metrics": result["metrics"]}, indent=2))


@app.command("run-redteam-campaign")
def run_redteam_campaign(
    config: Path = typer.Argument(..., exists=True, readable=True),
    budget: int | None = typer.Option(None, help="Override repetitions/budget"),
) -> None:
    facade = create_facade()
    result = facade.run_redteam_campaign(config, budget=budget)
    typer.echo(json.dumps(result, indent=2))


@app.command("replay-race")
def replay_race(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    mode: str = typer.Option("replay_audit_exact", help="Replay mode"),
) -> None:
    facade = create_facade()
    result = facade.replay_race(run_dir, mode=mode)
    typer.echo(json.dumps(result, indent=2))


@app.command("classify-failures")
def classify_failures(run_dir: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    facade = create_facade()
    run_output = facade._replay.load_run(run_dir)
    result = facade.classify_failures(run_output)
    typer.echo(json.dumps(result, indent=2))


@app.command("compare-regulations")
def compare_regulations(
    experiment_config: Path = typer.Argument(..., exists=True, readable=True),
    regulation_a: str = typer.Argument(...),
    regulation_b: str = typer.Argument(...),
    repetitions: int = typer.Option(3, help="Number of repeated comparisons"),
) -> None:
    facade = create_facade()
    result = facade.compare_regulations(regulation_a, regulation_b, experiment_config, n_repetitions=repetitions)
    typer.echo(json.dumps(result, indent=2))


@app.command("validate-public-session")
def validate_public_session(
    config: Path = typer.Argument(..., exists=True, readable=True),
    year: int = typer.Argument(..., help="Season year"),
    track_id: str = typer.Argument(..., help="Track id"),
    session_type: str = typer.Argument(..., help="Session type"),
    data_root: str = typer.Option("data", help="Local data-lake root"),
    drivers: str = typer.Option("", help="Comma-separated driver numbers"),
) -> None:
    facade = create_facade()
    driver_numbers = [int(value) for value in drivers.split(",") if value.strip()]
    result = facade.validate_against_public_session(
        config_path=config,
        year=year,
        track_id=track_id,
        session_type=session_type,
        data_root=data_root,
        driver_numbers=driver_numbers,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("propose-mitigations")
def propose_mitigations(run_dir: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    facade = create_facade()
    run_output = facade.run_multiagent_race(Path(run_dir)) if run_dir.suffix == ".yaml" else facade._replay.load_run(run_dir)
    result = facade.propose_mitigations(run_output)
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
