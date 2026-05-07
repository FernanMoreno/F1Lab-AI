# F1Lab-AI

F1Lab-AI is a Formula 1 regulation stress-testing laboratory. It simulates plausible car families, laps, battles, races, and adversarial scenarios to find where a regulation breaks before those failures appear on track.

It does not model real team internals. Public data, inferred parameters, calibrated parameters, and synthetic car families are kept separate on purpose.

## Verified State

Verified on 2026-05-07 in this repository state:

- Canonical target pack is `8` tracks: `suzuka`, `baku`, `monaco`, `monza`, `austria`, `singapore`, `barcelona`, `silverstone`.
- Primitive validation pack is closed for `5` public race sessions and reports `meets_thresholds` in [primitive_validation_pack_report.json](outputs/validation/public_primitives_target_pack/primitive_validation_pack_report.json).
- Full-race public validation now runs on all `8` target tracks and writes [public_race_validation_pack_report.json](outputs/validation/public_race_target_pack/public_race_validation_pack_report.json).
- Deep Agents runtime is real when a provider is configured. `llm_event_driven` no longer has to be heuristic-only.
- Dashboard reads real outputs from `outputs/validation`, `outputs/runs`, and `configs/track_pack.yaml`.
- RL and optimization layers exist as baseline code paths, not as production-scale training or search systems.

Current public-race validation summary from [public_race_validation_pack_report.json](outputs/validation/public_race_target_pack/public_race_validation_pack_report.json):

- Coverage: `8/8` target tracks
- Status: `needs_calibration`
- Mean overall score: `0.5288`
- Mean lap MAPE: `5.134`
- Strongest tracks: `silverstone 0.8349`, `monza 0.7619`, `suzuka 0.6980`
- Weakest tracks: `singapore 0.3105`, `barcelona 0.3130`, `monaco 0.3764`, `baku 0.4294`

This is the correct reading:

- lap+battle primitive validation is in defendable shape
- full-race public validation exists and is repeatable
- full-race credibility is not closed yet
- track fidelity 4 across the 8-pack is still not fully achieved

## Install

```bash
pip install -e ".[data,dashboard,rl,agents,optimization,dev]"
```

If you want real LLM runtime through Deep Agents, also provide provider credentials. The runtime currently supports `openai`, `azure_openai`, `anthropic`, and `google_genai` through `deepagents`.

## Core Commands

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy reglabsim agents dashboards/streamlit_app.py
```

Run a deterministic multiagent race:

```bash
.\.venv\Scripts\python.exe - <<'PY'
from reglabsim import create_facade
facade = create_facade()
result = facade.run_multiagent_race("configs/campaigns/suzuka_mini_multiagent.yaml")
print(result["manifest"]["run_id"], result["metrics"])
PY
```

Run the primitive validation pack:

```bash
.\.venv\Scripts\python.exe - <<'PY'
from reglabsim.validation.multi_circuit import run_target_pack
report = run_target_pack()
print(report["summary"])
PY
```

Run the full-race public validation pack:

```bash
.\.venv\Scripts\python.exe - <<'PY'
from reglabsim.validation.public_race import run_public_race_target_pack
report = run_public_race_target_pack()
print(report["summary"])
PY
```

Launch the dashboard:

```bash
streamlit run dashboards/streamlit_app.py
```

## Deep Agents Runtime

`reglabsim/runtime/agents.py` now contains two layers:

- deterministic rule-based and event-driven fallbacks
- Deep Agents wrappers for team-wall and driver intents

The runner switches to Deep Agents only when:

- `mode` is not `rule_based`
- `llm_provider` is not `heuristic`
- required provider credentials are present

Otherwise it falls back to the deterministic event-driven path.

Example config:

```yaml
campaign_name: suzuka_mini_deepagents
regulation: regulation_2026_refined
track: suzuka
scale_preset: mini
mode: llm_event_driven
llm_provider: openai
llm_model: openai:gpt-5.4
prompt_template_version: prompt.v1
seed: 42
weather_profile: dry_hot
```

More detail is in [docs/llm_runtime.md](docs/llm_runtime.md).

## Track Pack

The target pack lives in [track_pack.yaml](configs/track_pack.yaml). All eight tracks are available as canonical `reglabsim.track` YAMLs in `configs/tracks/`.

Current truth:

- the pack is canonical
- the pack is usable for validation and dashboard flows
- the pack is not yet uniformly fidelity-4 reviewed
- `validation_status` still reflects seed/manual-review reality, not completed digital-twin signoff

## Validation Layers

There are now two separate public validation layers:

1. Primitive validation in [public_primitives_target_pack.yaml](configs/validation/public_primitives_target_pack.yaml)
2. Full-race validation in [public_race_target_pack.yaml](configs/validation/public_race_target_pack.yaml)

This separation is intentional:

- primitive validation checks lap and battle behaviour directly
- full-race validation checks run-level plausibility against public session summaries

## Optimization and RL

`reglabsim/optimization/` and `reglabsim/rl/` are baseline frameworks, not closed research stacks.

What is real today:

- Monte Carlo, Bayesian, evolutionary, and adversarial search entry points exist
- adversarial search can now consume a real evaluator instead of generating only random placeholder metrics
- the RL environment is now seeded, action-space aware, observation-space aligned, and reward-backed

What is not real yet:

- large Optuna / pymoo / Nevergrad campaign scheduling
- production reward tuning and baseline training runs
- cross-track RL benchmarking

## Data and Outputs

Public session data is stored under `data/raw` and `data/silver`.

Validation and run outputs are written under `outputs/`:

- `outputs/validation/public_primitives_target_pack`
- `outputs/validation/public_race_target_pack`
- `outputs/runs/<run_id>`

The repository intentionally keeps raw public data and generated outputs out of the normal happy path for docs, but the code is built around those artifacts.

## What Is Still Open

The following items are still not honestly "done":

- fidelity 4 real for all 8 target tracks
- strong full-race credibility on street tracks and pack-compression cases
- large-scale optimization campaigns
- trained RL baselines
- a richer high-level agent orchestration graph beyond runtime control

Those are active next-step areas, not hidden gaps.
