# Usage
Copy `.env.example` to `.env` and add your API keys before running the CLI.
Environment variables are loaded automatically from this file.

## CLI

```bash
orch solve "Write a summary of this document."
orch show-config
orch bench --prompt "hi" --rounds 3
orch explain path/to/pipeline.py
orch add-eval-case -d my_evals.py -n new_case -i "input"
orch --profile
```

Use `orch improve --improvement-model MODEL` to override the model powering the
self-improvement agent when generating suggestions.

`orch bench` depends on `numpy`. Install with the optional `[bench]` extra:

```bash
pip install pydantic-ai-orchestrator[bench]
```

## API

```python
from pydantic_ai_orchestrator import (
    Orchestrator, Task, init_telemetry,
    review_agent, solution_agent, validator_agent,
)

# Initialize telemetry (optional)
init_telemetry()

# Create an orchestrator with default agents
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
)
result = orch.run_sync(Task(prompt="Write a poem."))
print(result)
```

The default `Orchestrator` runs a fixed Review -> Solution -> Validate pipeline.
It does not include a reflection step by default, but you can pass a
`reflection_agent` to enable one. For fully custom workflows or more complex
reflection logic, use the `Step` API with `PipelineRunner`.

Call `init_telemetry()` once at startup to configure logging and tracing for your application.

### Pipeline DSL

You can define custom workflows using the `Step` class and execute them with `PipelineRunner`:

```python
from pydantic_ai_orchestrator import Step, PipelineRunner
from pydantic_ai_orchestrator.plugins.sql_validator import SQLSyntaxValidator
from pydantic_ai_orchestrator.testing.utils import StubAgent

solution_step = Step.solution(StubAgent(["SELECT FROM"]))
validate_step = Step.validate(StubAgent([None]), plugins=[SQLSyntaxValidator()])
pipeline = solution_step >> validate_step
result = PipelineRunner(pipeline).run("SELECT FROM")
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
Add a reflection step by composing your own pipeline with `Step` and running it with `PipelineRunner`.
