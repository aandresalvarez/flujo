# Usage
Copy `.env.example` to `.env` and add your API keys before running the CLI.
Environment variables are loaded automatically from this file.

## CLI

```bash
flujo solve "Write a summary of this document."
flujo show-config
flujo bench --prompt "hi" --rounds 3
flujo explain path/to/pipeline.py
flujo add-eval-case -d my_evals.py -n new_case -i "input"
flujo --profile
```

Use `flujo improve --improvement-model MODEL` to override the model powering the
self-improvement agent when generating suggestions.

`flujo bench` depends on `numpy`. Install with the optional `[bench]` extra:

```bash
pip install flujo[bench]
```

## API

```python
from flujo.recipes import Default
from flujo import (
    Flujo, Task, init_telemetry,
    review_agent, solution_agent, validator_agent,
)

# Initialize telemetry (optional)
init_telemetry()

# Create the default recipe with built-in agents
orch = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
)
result = orch.run_sync(Task(prompt="Write a poem."))
print(result)
```

The `Default` recipe runs a fixed Review → Solution → Validate pipeline. It does
not include a reflection step by default, but you can pass a
`reflection_agent` to enable one. For fully custom workflows or more complex
reflection logic, use the `Step` API with the `Flujo` engine.

Call `init_telemetry()` once at startup to configure logging and tracing for your application.

### Pipeline DSL

You can define custom workflows using the `Step` class and execute them with `Flujo`:

```python
from flujo import Step, Flujo
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

solution_step = Step.solution(StubAgent(["SELECT FROM"]))
validate_step = Step.validate(StubAgent([None]), plugins=[SQLSyntaxValidator()])
pipeline = solution_step >> validate_step
result = Flujo(pipeline).run("SELECT FROM")
```

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

## Scoring Utilities
Functions like `ratio_score` and `weighted_score` are available for custom workflows.
The default orchestrator always returns a score of `1.0`.

## Reflection
Add a reflection step by composing your own pipeline with `Step` and running it with `Flujo`.
