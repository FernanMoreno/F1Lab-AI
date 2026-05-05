"""CLI entrypoint for F1Lab-AI."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from reglabsim.facade import create_facade

app = typer.Typer(add_completion=False, help="F1Lab-AI simulation CLI")


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
