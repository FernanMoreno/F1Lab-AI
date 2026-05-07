# LLM Runtime

This repository has two runtime paths for `llm_event_driven` campaigns:

1. Deterministic event-driven fallback
2. Deep Agents-backed runtime decisions

The selection happens in [runner.py](../reglabsim/campaigns/runner.py):

- `rule_based` always uses deterministic agents
- `llm_event_driven` uses Deep Agents only when `llm_provider != heuristic`
- if provider credentials are missing or agent invocation fails, the runtime falls back to the deterministic event-driven path

## Supported Providers

The current runtime wrapper resolves these providers:

- `openai`
- `azure_openai`
- `anthropic`
- `google_genai`

Environment variables expected by the runtime:

- `openai`: `OPENAI_API_KEY`
- `azure_openai`: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`
- `anthropic`: `ANTHROPIC_API_KEY`
- `google_genai`: `GOOGLE_API_KEY`

If `llm_model` is omitted or left as `event-driven-fallback`, the wrapper picks a provider default:

- `openai:gpt-5.4`
- `azure_openai:gpt-5.4`
- `anthropic:claude-sonnet-4-6`
- `google_genai:gemini-3.1-pro-preview`

## Runtime Contract

Deep Agents do not write directly into the microkernel. They produce structured outputs that are converted into existing contracts:

- team wall -> `TeamOrder`
- driver -> `DriverIntent`

The wrappers live in [agents.py](../reglabsim/runtime/agents.py) and use:

- `response_format` for structured output
- `memory=[AGENTS.md]` so repo rules are always in prompt
- short trigger-based invocation to control cost

## Trigger Policy

The runtime does not call the LLM every lap by default.

Team-wall Deep Agent triggers on combinations of:

- recent safety events
- wind warning
- forecast threat
- close midfield fight
- tyre risk

Driver Deep Agent triggers on combinations of:

- warnings
- low visibility
- wetness
- attack window
- immediate defensive pressure

If no trigger fires, the runtime keeps the deterministic event-driven policy.

## Example Campaign

```yaml
campaign_name: suzuka_mini_deepagents
description: Deep Agents runtime smoke case
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

## Validation Notes

The unit tests do not call live providers. They inject fake compiled agents and verify:

- runner class selection
- structured response usage
- prompt payload shape

That keeps the runtime testable without network or secret dependencies while still using the real Deep Agents integration path in production.
