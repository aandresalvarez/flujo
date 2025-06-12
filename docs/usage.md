# Usage
Copy `.env.example` to `.env` and add your API keys before running the CLI.

## CLI

```bash
orch solve "Write a summary of this document."
orch show-config
orch bench --prompt "hi" --rounds 3
orch --profile
```

`orch bench` depends on `numpy`. Install with the optional `[bench]` extra:

```bash
pip install pydantic-ai-orchestrator[bench]
```

## API

```python
from pydantic_ai_orchestrator import Orchestrator, init_telemetry

# Initialize telemetry (optional)
init_telemetry()

orch = Orchestrator()
result = orch.run_sync("Write a poem.")
print(result)
```

Call `init_telemetry()` once at startup to configure logging and tracing for your application.

## Environment Variables

- `OPENAI_API_KEY` (optional for OpenAI models)
- `GOOGLE_API_KEY` (optional for Gemini models)
- `ANTHROPIC_API_KEY` (optional for Claude models)
- `LOGFIRE_API_KEY` (optional)
- `REFLECTION_ENABLED` (default: true)
- `REWARD_ENABLED` (default: true) — toggles the reward model scorer on/off
- `MAX_ITERS`, `K_VARIANTS`
- `TELEMETRY_EXPORT_ENABLED` (default: false)
- `OTLP_EXPORT_ENABLED` (default: false)
- `OTLP_ENDPOINT` (optional, e.g. https://otlp.example.com)

## OTLP Exporter (Tracing/Telemetry)

If you want to export traces to an OTLP-compatible backend (such as OpenTelemetry Collector, Honeycomb, or Datadog), set the following environment variables:

- `OTLP_EXPORT_ENABLED=true` — Enable OTLP trace exporting
- `OTLP_ENDPOINT=https://your-otlp-endpoint` — (Optional) Custom OTLP endpoint URL

When enabled, the orchestrator will send traces using the OTLP HTTP exporter. This is useful for distributed tracing and observability in production environments.

## Scoring
- Ratio and weighted scoring supported
- Reward model stub included (extendable)

## Reflection
- Reflection agent can be toggled via `REFLECTION_ENABLED`

## Weighted Scoring

You can provide weights for checklist items to customize the scoring logic. This is useful when some criteria are more important than others.

Provide the weights via the `Task` metadata:

```python
from pydantic_ai_orchestrator import Orchestrator, Task

orch = Orchestrator(...)
task = Task(
    prompt="Generate a Python class.",
    metadata={
        "weights": [
            {"item": "Has a docstring", "weight": 0.7},
            {"item": "Includes type hints", "weight": 0.3},
        ]
    }
)
result = orch.run_sync(task)
```
The `scorer` setting must be set to `"weighted"`.
