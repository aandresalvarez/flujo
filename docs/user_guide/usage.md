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

### Debugging and Observability

Flujo provides rich tracing and debugging capabilities:

```bash
# List all pipeline runs
flujo lens list

# Show detailed information about a specific run
flujo lens show <run_id>

# View hierarchical execution trace
flujo lens trace <run_id>

# List individual spans with filtering
flujo lens spans <run_id> --status completed

# Show aggregated span statistics
flujo lens stats
```

Use `flujo improve --improvement-model MODEL` to override the model powering the
self-improvement agent when generating suggestions.

`flujo bench` depends on `numpy`. Install with the optional `[bench]` extra:

```bash
pip install flujo[bench]
```

## API

Use the `make_default_pipeline` factory function for full transparency, composability, and future YAML/AI support.

```python
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.infra.agents import make_review_agent, make_solution_agent, make_validator_agent
from flujo import (
    Flujo,
    Task,
)
from flujo.infra import init_telemetry

# Initialize telemetry (optional)
init_telemetry()

# Create the default pipeline using the factory
pipeline = make_default_pipeline(
    review_agent=make_review_agent(),
    solution_agent=make_solution_agent(),
    validator_agent=make_validator_agent(),
)

# Run the pipeline with tracing enabled
result = await run_default_pipeline(pipeline, Task(prompt="Write a poem."))
print(result)

# Access trace information
if result.trace_tree:
    print(f"Pipeline trace: {result.trace_tree.name}")
    print(f"Status: {result.trace_tree.status}")
    print(f"Duration: {result.trace_tree.end_time - result.trace_tree.start_time:.3f}s")
```

The `make_default_pipeline` factory creates a Review → Solution → Validate pipeline. It does
not include a reflection step by default, but you can pass a
`reflection_agent` to enable one. For fully custom workflows or more complex
reflection logic, use the `Step` API with the `Flujo` engine.

Call `init_telemetry()` once at startup to configure logging and tracing for your application.

### Pipeline DSL with Tracing

You can define custom workflows using the `Step` class and execute them with `Flujo`:

```python
from flujo import Step, Flujo
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

solution_step = Step.solution(StubAgent(["SELECT FROM"]))
validate_step = Step.validate(StubAgent([None]), plugins=[SQLSyntaxValidator()])
pipeline = solution_step >> validate_step

# Run with tracing enabled
flujo = Flujo(pipeline, enable_tracing=True)
result = flujo.run("SELECT FROM")

# Analyze execution trace
if result.trace_tree:
    print(f"Steps executed: {len(result.step_history)}")
    for step in result.step_history:
        print(f"  {step.name}: {'✅' if step.success else '❌'} ({step.latency_s:.3f}s)")

# When you're done, close the runner (or use it as a context manager) to release resources.
flujo.close()

# Context manager usage guarantees cleanup even if an exception is raised.
with Flujo(pipeline) as runner:
    runner.run("SELECT FROM")

# Note: the sync context manager (`with Flujo(...)`) must not be used inside `async def` functions.
# In async contexts, use `async with` or call `await runner.aclose()` explicitly.
async with Flujo(pipeline) as runner:
    await runner.run_result_async("SELECT FROM")
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

## Cost Tracking and Usage Limits

Flujo provides integrated cost and token usage tracking to help you monitor and control spending on LLM operations.

### Basic Cost Tracking

Once you've configured pricing in your `flujo.toml` file, cost tracking is automatically enabled (see [Configuration Guide](../advanced/configuration.md#cost-tracking-configuration)):

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async

# Create agents
solution_agent = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)
validator_agent = make_agent_async("openai:gpt-4o", "You are a validator.", str)

# Create pipeline
pipeline = Step.solution(solution_agent) >> Step.validate(validator_agent)
runner = Flujo(pipeline)

# Run and access cost information
result = runner.run("Write a short story about a robot.")

# Print cost details for each step
total_cost = 0
total_tokens = 0

for step_result in result.step_history:
    cost = step_result.cost_usd
    tokens = step_result.token_counts
    total_cost += cost
    total_tokens += tokens

    print(f"{step_result.name}:")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Tokens: {tokens}")
    print(f"  Success: {step_result.success}")

print(f"\nTotal cost: ${total_cost:.4f}")
print(f"Total tokens: {total_tokens}")
```

### Setting Usage Limits

Prevent excessive spending by setting cost and token limits:

```python
from flujo import Flujo, Step, UsageLimits

# Define limits
limits = UsageLimits(
    total_cost_usd_limit=0.50,  # Maximum $0.50 total cost
    total_tokens_limit=2000     # Maximum 2,000 tokens
)

# Apply limits to pipeline
runner = Flujo(pipeline, usage_limits=limits)

try:
    result = runner.run("Write a comprehensive analysis.")
    print("Pipeline completed successfully!")
except UsageLimitExceededError as e:
    print(f"Pipeline stopped due to usage limits: {e}")
    # Access partial results
    partial_result = e.partial_result
    print(f"Completed {len(partial_result.step_history)} steps before stopping")
```

### Step-Level Limits

Set limits on individual steps for fine-grained control:

```python
from flujo import Step, UsageLimits

# Set limits for specific steps
solution_limits = UsageLimits(
    total_cost_usd_limit=0.20,  # Maximum $0.20 for solution step
    total_tokens_limit=800       # Maximum 800 tokens for solution step
)

validation_limits = UsageLimits(
    total_cost_usd_limit=0.10,  # Maximum $0.10 for validation step
    total_tokens_limit=400       # Maximum 400 tokens for validation step
)

pipeline = (
    Step.solution(solution_agent, usage_limits=solution_limits)
    >> Step.validate(validator_agent, usage_limits=validation_limits)
)
```

### Parallel Execution with Limits

When using parallel steps, Flujo can cancel sibling branches when limits are exceeded:

```python
from flujo import Step, Pipeline, UsageLimits

# Create parallel branches
expensive_branch = Pipeline.from_step(Step("expensive", costly_agent))
cheap_branch = Pipeline.from_step(Step("cheap", cheap_agent))

parallel = Step.parallel_branch(expensive_branch, cheap_branch)

# If expensive_branch exceeds limits, cheap_branch will be cancelled
limits = UsageLimits(total_cost_usd_limit=0.10)
runner = Flujo(parallel, usage_limits=limits)

try:
    result = runner.run("Process this data.")
except UsageLimitExceededError as e:
    print("One or more branches exceeded limits")
```

### Monitoring Costs in Production

Log cost information for analysis and monitoring:

```python
import logging
from flujo import Flujo, Step

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_costs(result):
    """Log cost information for monitoring."""
    total_cost = sum(step.cost_usd for step in result.step_history)
    total_tokens = sum(step.token_counts for step in result.step_history)

    logger.info(f"Pipeline completed - Cost: ${total_cost:.4f}, Tokens: {total_tokens}")

    # Log per-step details
    for step in result.step_history:
        logger.info(f"  {step.name}: ${step.cost_usd:.4f} ({step.token_counts} tokens)")

# Run pipeline with cost logging
pipeline = Step.solution(my_agent) >> Step.validate(validator_agent)
runner = Flujo(pipeline)

result = runner.run("Your prompt")
log_costs(result)
```

### Cost-Effective Pipeline Design

Design pipelines with cost efficiency in mind:

```python
from flujo import Step, Flujo, UsageLimits

# Use cheaper models for simple tasks
simple_agent = make_agent_async("openai:gpt-3.5-turbo", "Simple task agent.", str)
complex_agent = make_agent_async("openai:gpt-4o", "Complex task agent.", str)

# Design pipeline with cost considerations
pipeline = (
    Step.solution(simple_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.05))
    >> Step.validate(complex_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.15))
)

# Set overall pipeline limits
runner = Flujo(pipeline, usage_limits=UsageLimits(total_cost_usd_limit=0.25))
```

### Troubleshooting Cost Tracking

If cost tracking isn't working as expected:

1. **Check your `flujo.toml` configuration**:
   ```toml
   [cost.providers.openai.gpt-4o]
   prompt_tokens_per_1k = 0.005
   completion_tokens_per_1k = 0.015
   ```

2. **Enable debug logging**:
   ```python
   import logging
   logging.getLogger("flujo.cost").setLevel(logging.DEBUG)
   ```

3. **Verify agent usage information**:
   ```python
   # Check if your agent returns usage information
   result = my_agent.run("test")
   if hasattr(result, 'usage'):
       usage = result.usage()
       print(f"Prompt tokens: {usage.prompt_tokens}")
       print(f"Completion tokens: {usage.completion_tokens}")
   ```

For more detailed configuration information, see the [Configuration Guide](../advanced/configuration.md#cost-tracking-configuration).

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

Project-aware default:

- Inside a Flujo project (created via `flujo init`) you can simply run:
  ```sh
  flujo run --input "Hello world"
  ```
  This implicitly runs the project’s `pipeline.yaml`.

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

See also: [Pipeline DSL Guide](../user_guide/pipeline_dsl.md), [Typed Pipeline Context](../user_guide/pipeline_context.md)

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
