# GPT‑5 Architect Pipeline and `model_settings`

This guide explains how Flujo supports GPT‑5 provider controls via `model_settings` in YAML and showcases a reference Architect pipeline. The CLI’s Architect is now implemented programmatically (state machine) in `flujo/architect/builder.py`; the YAML example remains useful for studying agent settings and prompts.

## Overview

With GPT‑5, you can pass fine‑grained controls (e.g., reasoning effort and text verbosity) directly to agents in your YAML blueprints. Flujo forwards these settings to the underlying `pydantic-ai` Agent during compilation.

Key benefits:
- Single powerful architect agent replaces multiple planning steps.
- In‑memory YAML validation with a self‑correction loop.
- Optional repair agent leverages GPT‑5 reasoning to fix issues.

## Using `model_settings` in YAML

Add `model_settings` under any declarative agent:

```yaml
agents:
  architect_agent:
    model: "openai:gpt-5"
    model_settings:
      reasoning: { effort: "high" }
      text: { verbosity: "low" }
    system_prompt: |
      You are the Flujo AI Architect...
    output_schema:
      type: object
      properties: { yaml_text: { type: string } }
      required: [yaml_text]
```

These settings are passed as‑is to `pydantic_ai.Agent` (via Flujo’s compiler) so providers can interpret them natively.

## GPT‑5 Architect Pipeline (Example)

See `examples/architect_pipeline.yaml` for a full example (reference). Highlights:
- `agents.architect_agent` designs and emits the YAML (`YamlWriter(yaml_text: str)`).
- Validation loop uses `flujo.builtins.validate_yaml` and branches on `flujo.utils.context:predicate_is_valid_report`.
- Valid branch is a passthrough; invalid branch uses `agents.repair_agent` (also with `model_settings`).

If you prefer a declarative state machine that handles interactive clarification (HITL) and validation phases, see `examples/architect_pipeline_state_machine.yaml`. It demonstrates a `transitions:` block with an `on: pause` self‑transition for the `Clarification` state so that, upon resume, the state re‑enters to process new user input.

## Timeouts & Retries

Complex GPT‑5 calls can take longer than typical LLM requests. You can tune timeouts and retries in two places:

- Agent-level (applies to the LLM call itself; enforced by the Agent wrapper):

```yaml
agents:
  architect_agent:
    model: "openai:gpt-5"
    timeout: 180        # seconds
    max_retries: 1
    model_settings:
      reasoning: { effort: "high" }
      text: { verbosity: "low" }
```

- Step-level (alias for plugin/validator phases; normalized to `timeout_s`):

```yaml
steps:
  - name: DesignAndBuildBlueprint
    uses: agents.architect_agent
    config:
      timeout: 180      # used for plugin/validator stages; agent call uses agent.timeout above
      max_retries: 1
```

Notes:
- Agent `timeout` and `max_retries` are forwarded to `make_agent_async` and enforced by the Agent wrapper.
- Step `config.timeout` is normalized to `timeout_s` and used by plugin/validator phases. The agent call timeout is governed by the agent-level `timeout`.

## CLI Support

`flujo create` passes a list of available skills to the architect via initial context. In the programmatic builder, this is set during the `GatheringContext` state. To enable the full conversational state machine for the CLI, set `FLUJO_ARCHITECT_STATE_MACHINE=1`.

## Testing Notes

- The compiler now accepts `model_settings` in the agent schema and forwards them through `make_agent_async`.
- E2E tests can assert that `model_settings` are received by the agent constructor using monkeypatching of `flujo.domain.blueprint.compiler.make_agent_async`.
