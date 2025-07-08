# Usage
Copy `.env.example` to `.env` and add your API keys before running the CLI.
Environment variables are loaded automatically from this file.

## CLI

```bash
flujo solve "Write a summary of this document."
flujo show-config
flujo bench "hi" --rounds 3
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

> **Note:** The class-based `Default` recipe is deprecated. Use the new `make_default_pipeline` factory function for full transparency, composability, and future YAML/AI support.

```python
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.infra.agents import make_review_agent, make_solution_agent, make_validator_agent
from flujo import (
    Flujo, Task, init_telemetry,
)

# Initialize telemetry (optional)
init_telemetry()

# Create the default pipeline using the factory
pipeline = make_default_pipeline(
    review_agent=make_review_agent(),
    solution_agent=make_solution_agent(),
    validator_agent=make_validator_agent(),
)

# Run the pipeline
result = await run_default_pipeline(pipeline, Task(prompt="Write a poem."))
print(result)
```

The `make_default_pipeline` factory creates a Review → Solution → Validate pipeline. It does
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

## Running Custom Pipelines from the CLI: `flujo run`

The `flujo run` command lets you execute any custom pipeline directly from the command line—no need to write a `if __name__ == "__main__":` script. This makes rapid iteration and testing of your workflows much easier.

### Basic Usage

```sh
flujo run my_pipeline.py --input "Hello world" --context-model MyContext
```

- `my_pipeline.py` should define a top-level variable (default: `pipeline`) of type `Pipeline`.
- `--input` provides the initial input to the pipeline.
- `--context-model` (optional) specifies the name of a context model class defined in the file.

### Passing Context Data

You can pass initial context data as a JSON string or from a file (JSON or YAML):

```sh
flujo run my_pipeline.py --input "Prompt" --context-model MyContext --context-data '{"counter": 5}'

flujo run my_pipeline.py --input "Prompt" --context-model MyContext --context-file context.json

flujo run my_pipeline.py --input "Prompt" --context-model MyContext --context-file context.yaml
```

### Customizing the Pipeline Variable Name

If your pipeline variable is not named `pipeline`, use `--pipeline-name`:

```sh
flujo run my_pipeline.py --input "Prompt" --pipeline-name my_custom_pipeline
```

### Output

By default, the CLI prints a summary table and the final context. For machine-readable output, use `--json`:

```sh
flujo run my_pipeline.py --input "Prompt" --context-model MyContext --json
```

### Example Pipeline File

```python
from flujo import step, Pipeline
from flujo.domain.models import PipelineContext
from pydantic import Field

class MyContext(PipelineContext):
    counter: int = Field(default=0)

@step
async def inc(data: str, *, context: MyContext | None = None) -> str:
    if context:
        context.counter += 1
    return data.upper()

pipeline = inc >> inc
```

### Example Context File (YAML)

```yaml
counter: 5
```

### Example Command

```sh
flujo run my_pipeline.py --input "hello" --context-model MyContext --context-file context.yaml
```

### Why Use `flujo run`?

- No boilerplate needed for quick experiments.
- Test and debug pipelines interactively.
- Pass context and input flexibly.
- Integrates with the full DSL and context system.

See also: [Pipeline DSL Guide](pipeline_dsl.md), [Typed Pipeline Context](pipeline_context.md)

### Full CLI Demo Example

Below is a complete example pipeline file you can run directly with the CLI:

```python
from flujo import step, Pipeline
from flujo.domain.models import PipelineContext
from pydantic import Field

class DemoContext(PipelineContext):
    counter: int = Field(default=0)
    log: list[str] = Field(default_factory=list)

@step
async def greet(data: str, *, context: DemoContext | None = None) -> str:
    msg = f"Hello, {data}!"
    if context:
        context.counter += 1
        context.log.append(msg)
    return msg

@step
async def emphasize(data: str, *, context: DemoContext | None = None) -> str:
    msg = data.upper() + "!!!"
    if context:
        context.counter += 1
        context.log.append(msg)
    return msg

@step
async def summarize(data: str, *, context: DemoContext | None = None) -> str:
    summary = f"Summary: {data} (steps: {context.counter if context else 0})"
    if context:
        context.counter += 1
        context.log.append(summary)
    return summary

pipeline = greet >> emphasize >> summarize
```

You can run this file with:

```sh
flujo run examples/10_cli_run_demo.py --input "quickstart" --context-model DemoContext
```

Or with context data:

```sh
flujo run examples/10_cli_run_demo.py --input "with context" --context-model DemoContext --context-data '{"counter": 10}'
```
