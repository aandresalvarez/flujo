# Pipeline DSL Guide

The Pipeline Domain-Specific Language (DSL) is a powerful way to create custom AI workflows in `flujo`. This guide explains how to use it effectively.

## Overview

The Pipeline DSL lets you:

- Compose complex workflows from simple steps **and from other pipelines**
- Mix and match different agents
- Add custom validation and scoring
- Create reusable pipeline components

## Steps vs. Agents

A **Step** is a declarative node in the pipeline. It holds configuration and a
reference to the **agent** that performs the actual work. During execution the
runner iterates over the steps and invokes their agents in order.

```mermaid
graph LR
    S[Step] -->|uses| A[Agent]
    A --> O[Output]
```

`Step` objects do not execute anything themselves—they simply describe what
should happen. The agent may be an async function, an `AsyncAgentWrapper` created
with `make_agent_async`, or any object implementing `run()`.

## Basic Usage

!!! tip "Recommended Pattern"
    For creating pipeline steps from your own `async` functions, the `@step` decorator is the simplest and most powerful approach. It automatically infers types and reduces boilerplate, making your code cleaner and safer.

### Creating a Pipeline

```python
from flujo import Flujo, step

@step
async def add_one(x: int) -> int:
    return x + 1

@step
async def add_two(x: int) -> int:
    return x + 2

pipeline = add_one >> add_two
runner = Flujo(pipeline)
result = runner.run(1)
```

The `@step` decorator infers the input and output types from the
function's signature so the pipeline is typed as `Step[int, int]`.

### Streaming lifecycle events

Use `run_with_events` when you need lifecycle visibility (e.g., background launches or
streaming chunks) and still want the final `PipelineResult` without a second call:

```python
from flujo.domain.models import BackgroundLaunched, Chunk, PipelineResult

async for event in runner.run_with_events(1):
    if isinstance(event, BackgroundLaunched):
        print(f"Background step launched: {event.step_name} ({event.task_id})")
    elif isinstance(event, Chunk):
        print("stream chunk:", event.data)
    elif isinstance(event, PipelineResult):
        print("done:", event.step_history[-1].output)
```

Prefer `run_async` if you only care about the final result, or `run_outcomes_async`
if you want only `StepOutcome` events without the final `PipelineResult`.

### Runner entrypoints (choose the right one)

- `run_result_async(input)`: return the final `PipelineResult` (async, no events).
- `run_async(input)`: legacy awaitable/async-iterable; yields events + final result.
- `run_outcomes_async(input)`: yield only `StepOutcome` events (Success/Failure/Paused/Chunk).
- `run_stream(input)` / `run_outcomes(input)`: explicit streaming aliases.
- `run_with_events(input)`: yield lifecycle events (e.g., `BackgroundLaunched`, `Chunk`) plus the final `PipelineResult`.

### Pipeline Composition

The `>>` operator chains steps together:

```python
@step
async def multiply(x: int) -> int:
    return x * 2

@step
async def add_three(x: int) -> int:
    return x + 3

pipeline1 = multiply >> add_three
pipeline2 = add_three >> multiply
```

---

### **Chaining Pipelines with Pipelines: Modular Multi-Stage Workflows**

> **New** You can now compose entire pipelines from other pipelines using the `>>` operator. This allows you to break complex workflows into logical, independent pipelines and then chain them together in a clean, readable sequence.

#### **How it Works**

- `Step >> Step` → `Pipeline`
- `Pipeline >> Step` → `Pipeline`
- **`Pipeline >> Pipeline` → `Pipeline`** (new!)

When you chain two pipelines, their steps are concatenated into a single, flat pipeline. The output of the first pipeline becomes the input to the second.

#### **Example: Chaining Pipelines**

```python
from flujo import Pipeline, Step

# Define two independent pipelines
pipeline_a = Step("A1") >> Step("A2")
pipeline_b = Step("B1") >> Step("B2")

# Chain them together
master_pipeline = pipeline_a >> pipeline_b

# master_pipeline.steps == [A1, A2, B1, B2]
```

#### **Real-World Example: Multi-Stage Data Processing**

Suppose you want to process text in two stages: first, resolve concepts; then, generate and validate SQL.

```python
from flujo import Pipeline, Step

# 1. Build each independent pipeline
concept_pipeline = Step("resolve_concepts", agent=concept_agent)
sql_pipeline = (
    Step("generate_sql", agent=sql_gen_agent) >>
    Step("validate_sql", agent=sql_val_agent)
)

# 2. Chain them together using the >> operator
master_pipeline = concept_pipeline >> sql_pipeline

# The resulting pipeline takes text and outputs validated SQL
```

This approach:
- Keeps each stage modular and testable
- Produces a single, flat pipeline for unified context and observability
- Is fully type-safe and backward compatible

> **Tip:** You can chain as many pipelines as you want: `p1 >> p2 >> p3`.

#### **Why This Matters**
- **True Sequencing:** Models a sequence of operations, not just nested sub-pipelines.
- **Unified Context:** All steps share a single context and are visible to the tracer.
- **Simplicity:** No need for special sub-pipeline steps or wrappers.

---

### Creating Steps from Functions

Use the `@step` decorator to wrap your own async functions. The decorator infers
both the input and output types:

```python
@step
async def to_upper(text: str) -> str:
    return text.upper()

upper_step = to_upper
```

The resulting `upper_step` has the type `Step[str, str]` and can be composed
like any other step.

## Step Types

### Review Steps

Review steps create quality checklists:

```python
# Basic review step
review_step = Step.review(review_agent)

# With custom timeout
review_step = Step.review(review_agent, timeout=30)

# With custom retry logic
review_step = Step.review(
    review_agent,
    retries=3,
    backoff_factor=2
)
```

### Solution Steps

Solution steps generate the main output:

```python
# Basic solution step
solution_step = Step.solution(solution_agent)

# With structured output
from pydantic import BaseModel

class CodeSnippet(BaseModel):
    language: str
    code: str
    explanation: str

code_agent = make_agent_async(
    "openai:gpt-4",
    "You are a programming expert.",
    CodeSnippet
)

solution_step = Step.solution(code_agent)

# With tools
from pydantic_ai import Tool

def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

weather_tool = Tool(get_weather)
solution_step = Step.solution(
    solution_agent,
    tools=[weather_tool]
)
```

### Validation Steps

Validation steps verify the solution:

```python
# Basic validation
validate_step = Step.validate_step(validator_agent)

# With strict validation (default) - step fails if validation fails
strict_step = Step.validate_step(validator_agent, validators=[...], strict=True)

# With non-strict validation - step passes but records validation failure in metadata
audit_step = Step.validate_step(validator_agent, validators=[...], strict=False)
```

**Strict vs Non-Strict Validation:**

- **`strict=True` (default)**: If any validation fails, the entire step fails and the pipeline stops or retries.
- **`strict=False`**: The step always reports `success=True`, but validation failures are recorded in `StepResult.metadata_['validation_passed'] = False`. This is useful for creating "warning" or "auditing" steps that don't block the pipeline.

```python
# Example: Audit step that warns but doesn't fail
audit_step = Step.validate_step(
    validator_agent,
    validators=[WordCountValidator()],
    strict=False  # Will pass even if validation fails
)

# Later in your pipeline, you can check the metadata
if result.step_history[-1].metadata_.get('validation_passed') == False:
    print("Warning: Validation failed but pipeline continued")
```

# With custom scoring
from flujo.domain.scoring import weighted_score

weights = {
    "correctness": 0.6,
    "readability": 0.4
}

validate_step = Step.validate_step(
    validator_agent,
    scorer=lambda c: weighted_score(c, weights)
)

# With plugins
from flujo.plugins import SQLSyntaxValidator

validate_step = Step.validate_step(
    validator_agent,
    plugins=[SQLSyntaxValidator()]
)

# With programmatic validators
from flujo.domain.validation import BaseValidator, ValidationResult

class WordCountValidator(BaseValidator):
    async def validate(self, output_to_check: str, *, context=None) -> ValidationResult:
        return ValidationResult(is_valid=len(output_to_check.split()) < 5, validator_name=self.name,
                                feedback="Too many words" if len(output_to_check.split()) >= 5 else None)

validate_step = Step.validate_step(
    validator_agent,
    validators=[WordCountValidator()]
)

See [Hybrid Validation Cookbook](../cookbook/hybrid_validation.md) for a complete example.
```

All step factories also accept a `processors: Optional[AgentProcessors]` parameter
to run pre-processing and post-processing hooks. See [Using Processors](../cookbook/using_processors.md)
for details.
For complex data shaping before calling another step, consider using an [Adapter Step](../cookbook/adapter_step.md).

## Advanced Features

git ### Looping and Iteration

Repeat a sub-pipeline until a condition is met using `Step.loop_until()`.
See [LoopStep documentation](pipeline_looping.md) for full details.

```python
loop_step = Step.loop_until(
    name="refine",
    loop_body_pipeline=Pipeline.from_step(Step.solution(solution_agent)),
    exit_condition_callable=lambda out, ctx: "done" in out,
)

pipeline = Step.review(review_agent) >> loop_step >> Step.validate_step(validator_agent)
```

## Typed Pipeline Context

a `Flujo` runner can share a mutable Pydantic model instance across all steps in a single run. Pass a context model when creating the runner and declare a `context` parameter in your step functions or agents. See [Typed Pipeline Context](pipeline_context.md) for a full explanation.

```python
from flujo.domain.models import PipelineContext

class MyContext(PipelineContext):
    counter: int = 0

@step
async def increment(data: str, *, context: MyContext | None = None) -> str:
    if context:
        context.counter += 1
    return data

pipeline = increment >> increment
runner = Flujo(pipeline, context_model=MyContext)
result = runner.run("hi")
print(result.final_pipeline_context.counter)  # 2
```

Each `run()` call gets a fresh context instance. Access the final state via
`PipelineResult.final_pipeline_context`.

You can also have a step return a partial context object and mark it with
`updates_context=True` to automatically merge those fields into the running
context:

```python
@step(updates_context=True)
async def bootstrap(_: str) -> MyContext:
    return MyContext(counter=42)

pipeline = bootstrap >> increment
runner = Flujo(pipeline, context_model=MyContext)
result = runner.run("hi")
print(result.final_pipeline_context.counter)  # 43
```

When a step marked with `updates_context=True` returns a dictionary or a Pydantic
model, the new data is merged into the current pipeline context. This merge is
validation-safe: Pydantic recursively reconstructs all nested models and the
entire context is revalidated. If the update would result in an invalid context,
the step fails and the previous state is restored, preventing data corruption in
later steps.

## Managed Resources

You can also pass a shared resources container to the runner. Declare a
keyword-only `resources` argument in your agents or plugins to use it. If the
`resources` object implements a sync or async context manager, Flujo will
enter/exit it **per step attempt** (including retries and parallel branches),
so you can bind a database transaction to a single attempt and rollback on
failure.

```python
class MyResources(AppResources):
    db_pool: Any

@step
async def query(data: int, *, resources: MyResources) -> str:
    return resources.db_pool.get_user(data)

runner = Flujo(query, resources=my_resources)
```

Notes:
- Context managers must be re-entrant or hand out per-attempt handles so
  parallel steps do not fight over shared state.
- If your resources are plain containers (no `__enter__/__aenter__`), they are
  still injected as-is without additional lifecycle hooks.

### Conditional Branching

Use `Step.branch_on()` to route to different sub-pipelines at runtime. See [ConditionalStep](pipeline_branching.md) for full details.

```python
def choose_branch(out, ctx):
    return "a" if "important" in out else "b"

branch_step = Step.branch_on(
    name="router",
    condition_callable=choose_branch,
    branches={
        "a": Pipeline.from_step(Step("a_step", agent_a)),
        "b": Pipeline.from_step(Step("b_step", agent_b)),
    },
)

pipeline = Step.solution(solution_agent) >> branch_step >> Step.validate_step(validator_agent)
```

### Custom Step Factories

Create reusable step factories:

```python
def create_code_step(agent, **config):
    """Create a solution step with code validation."""
    step = Step.solution(agent, **config)
    step.add_plugin(SQLSyntaxValidator())
    return step

# Use the factory
pipeline = (
    Step.review(review_agent)
    >> create_code_step(solution_agent)
    >> Step.validate_step(validator_agent)
)
```

## Error Handling

### Retry Logic

```python
# Configure retries at the step level
step = Step.solution(
    solution_agent,
    retries=3,
    backoff_factor=2,
    retry_on_error=True
)

# Configure retries at the pipeline level
runner = Flujo(
    pipeline,
    max_retries=3,
    retry_on_error=True
)
```

## Best Practices

1. **Pipeline Design**
   - Keep pipelines focused and simple
   - Use meaningful step names
   - Document complex pipelines
   - Test thoroughly

2. **Error Handling**
   - Add appropriate retries
   - Log errors properly
   - Monitor performance

3. **Performance**
   - Optimize step order
   - Cache results when possible
   - Monitor resource usage

4. **Maintenance**
   - Create reusable components
   - Version your pipelines
   - Document dependencies
   - Test regularly

## Examples

### Code Generation Pipeline

```python
from flujo import Step, Flujo
from flujo.plugins import (
    SQLSyntaxValidator,
    CodeStyleValidator
)

# Create a code generation pipeline
pipeline = (
    Step.review(review_agent)  # Define requirements
    >> Step.solution(code_agent)  # Generate code
    >> Step.validate_step(
        validator_agent,
        plugins=[
            SQLSyntaxValidator(),
            CodeStyleValidator()
        ]
    )
)

# Run it
runner = Flujo(pipeline)
result = runner.run("Write a SQL query to find active users")
```

### Content Generation Pipeline

```python
# Create a content generation pipeline
pipeline = (
    Step.review(review_agent)  # Define content guidelines
    >> Step.solution(writer_agent)  # Generate content
    >> Step.validate_step(
        validator_agent,
        scorer=lambda c: weighted_score(c, {
            "grammar": 0.3,
            "style": 0.3,
            "tone": 0.4
        })
    )
)

# Run it
runner = Flujo(pipeline)
result = runner.run("Write a blog post about AI")
```

## Troubleshooting

### Common Issues

1. **Pipeline Errors**
   - Check step order
   - Verify agent compatibility
   - Review error messages
   - Check configuration

2. **Performance Issues**
   - Monitor step durations
   - Check resource usage
   - Optimize step order

3. **Quality Issues**
   - Review scoring weights
   - Check validation rules
   - Monitor success rates
   - Adjust agents

### Getting Help

- Check the [Troubleshooting Guide](../advanced/troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md)
- Explore [Advanced Topics](../advanced/extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)
- Future work: a `pipeline.visualize()` helper will output a Mermaid graph so you
  can instantly diagram your pipeline.

# Pipeline DSL

The Pipeline DSL provides a fluent interface for building complex workflows. It supports sequential execution, conditional branching, parallel execution, and looping.

## Steps

Steps are the basic building blocks of pipelines. Each step has a name and an agent that performs the actual work.

### Creating Steps

```python
from flujo import Step, StepConfig, step

# Create a step with an agent
step = Step("my_step", my_agent)

# Create a step with configuration
step = Step("my_step", my_agent, max_retries=3, timeout_s=30.0)

# Use StepConfig for reuse or clarity
cfg = StepConfig(max_retries=2, timeout_s=10, execution_mode="background")
bg_step = Step("bg", my_agent, config=cfg)

# Decorator form (preferred for functions)
@step(name="compute", config=cfg)
async def compute(x: int) -> int:
    return x + 1

# You can override specific fields even when passing config; overrides win
@step(config=cfg, timeout_s=5)
async def compute_fast(x: int) -> int:
    return x + 1
```

### Step Configuration

Steps can be configured with various options:

- `max_retries`: Number of retry attempts (default: 1)
- `timeout_s`: Timeout in seconds (default: None)
- `temperature`: Temperature for LLM agents (default: None)

### Fallback Steps

Use `.fallback(other_step)` to specify an alternate step to run if the primary
step fails after exhausting its retries. The fallback receives the same input as
the original step.

```python
from flujo import Step

primary = Step("generate", primary_agent, max_retries=2)
backup = Step("backup", backup_agent)
primary.fallback(backup)
```

If the fallback succeeds, the overall step is marked successful and
`StepResult.metadata_['fallback_triggered']` is set to `True`.
Metrics like latency, cost, and token counts from the fallback step are merged
into the primary result. Circular fallback references raise
`InfiniteFallbackError`.

## Pipelines

Pipelines are sequences of steps that execute in order.

### Creating Pipelines

```python
from flujo import Pipeline, Step

# Create a pipeline from steps
pipeline = Step("step1", agent1) >> Step("step2", agent2) >> Step("step3", agent3)

# Or create a pipeline directly
pipeline = Pipeline([step1, step2, step3])
```

### Pipeline Composition

Pipelines can be composed using the `>>` operator:

```python
pipeline1 = Step("a", agent_a) >> Step("b", agent_b)
pipeline2 = Step("c", agent_c) >> Step("d", agent_d)
combined = pipeline1 >> pipeline2
```

## Conditional Steps

Conditional steps execute different branches based on a condition.

```python
from flujo import Step

def route_by_type(data, context):
    if "code" in str(data):
        return "code"
    return "text"

conditional = Step.branch_on(
    name="router",
    condition_callable=route_by_type,
    branches={
        "code": Pipeline.from_step(Step("code_gen", code_agent)),
        "text": Pipeline.from_step(Step("text_gen", text_agent)),
    }
)
```

## Parallel Steps

Parallel steps execute multiple branches concurrently and aggregate their outputs.

### Basic Parallel Execution

```python
from flujo import Step

parallel = Step.parallel(
    name="parallel_processing",
    branches={
        "analysis": Pipeline.from_step(Step("analyze", analysis_agent)),
        "summary": Pipeline.from_step(Step("summarize", summary_agent)),
    }
)
```

### Optimized Context Copying

For pipelines with large context objects, you can optimize performance by specifying which context fields each branch needs:

```python
from flujo import Step

# Only copy specific context fields to each branch
parallel = Step.parallel(
    name="parallel_optimized",
    branches={
        "analysis": Pipeline.from_step(Step("analyze", analysis_agent)),
        "summary": Pipeline.from_step(Step("summarize", summary_agent)),
    },
    context_include_keys=["user_id", "document_id"]  # Only copy these fields
)
```

This feature provides significant performance improvements when:
- Your context contains large data structures (documents, images, etc.)
- You have many parallel branches
- Each branch only needs a subset of the context data

### Context Merging and Failure Handling

`Step.parallel` can merge context updates from its branches back into the main
pipeline context. Use the `merge_strategy` parameter to control how merging is
performed and `on_branch_failure` to define failure behavior.

```python
from flujo import Step
from flujo.domain import MergeStrategy, BranchFailureStrategy

parallel = Step.parallel(
    name="parallel_merge",
    branches={"a": Pipeline.from_step(Step("a", a_agent)), "b": Pipeline.from_step(Step("b", b_agent))},
    merge_strategy=MergeStrategy.CONTEXT_UPDATE,
    on_branch_failure=BranchFailureStrategy.IGNORE,
)
```

Available `MergeStrategy` values:

- `CONTEXT_UPDATE` (default) – apply validated updates; conflicts on the same field now raise a configuration error unless resolved via `field_mapping`. Use this to merge structured branch outputs explicitly.
- `ERROR_ON_CONFLICT` – explicitly fail on any conflicting field updates.
- `OVERWRITE` – context from the last declared successful branch overwrites matching fields.
- `NO_MERGE` – skip merging branch contexts (returns per-branch outputs only).
- `KEEP_FIRST` – keep the first occurrence of each key when merging (non-validating).
- `MERGE_SCRATCHPAD` has been removed; scratchpad is framework-reserved and must not be merged from branches. Use typed context fields and `CONTEXT_UPDATE` instead.

`on_branch_failure` accepts `PROPAGATE` (default) or `IGNORE`. When set to
`IGNORE`, the parallel step succeeds as long as one branch succeeds and the
output dictionary includes the failed `StepResult` objects for inspection.

### Consensus Reducers

Parallel steps can reduce branch results into a single consensus output by
providing a `reduce` callable. Built-in reducers include:

- `majority_vote` (most common output)
- `code_consensus` (identical string outputs)
- `judge_selection` (Multi-Signal evaluator selection)

```python
from flujo import Step
from flujo.domain.consensus import majority_vote

parallel = Step.parallel(
    name="panel",
    branches={
        "a": Pipeline.from_step(Step("a", agent_a)),
        "b": Pipeline.from_step(Step("b", agent_b)),
    },
    reduce=majority_vote,
)
```

YAML example:

```yaml
version: "0.1"
name: "panel_pipeline"
steps:
  - kind: parallel
    name: panel
    reduce: "majority_vote"
    branches:
      a:
        - kind: step
          name: a
          agent: flujo.builtins.passthrough
      b:
        - kind: step
          name: b
          agent: flujo.builtins.passthrough
```

### Proactive Governor Cancellation

Parallel steps now support proactive cancellation when usage limits are breached. When any branch exceeds cost or token limits, sibling branches are immediately cancelled to prevent unnecessary resource consumption:

```python
from flujo import Step, UsageLimits

parallel = Step.parallel(
    name="parallel_governed",
    branches={
        "fast_expensive": Pipeline.from_step(Step("expensive", costly_agent)),
        "slow_cheap": Pipeline.from_step(Step("cheap", cheap_agent)),
    }
)

# If fast_expensive breaches the limit, slow_cheap will be cancelled immediately
limits = UsageLimits(total_cost_usd_limit=0.10)
runner = Flujo(parallel, usage_limits=limits)
```

This feature is particularly beneficial when:
- You have branches with varying costs and execution times
- You want to minimize wasted resources when limits are exceeded
- You need predictable execution times under resource constraints

### Dynamic Parallel Router

Use `Step.dynamic_parallel_branch()` when an agent selects which parallel branches to run at runtime. The router agent returns a list of branch names.

```python
router = Step.dynamic_parallel_branch(
    name="router",
    router_agent=my_router_agent,
    branches={"billing": billing_pipe, "support": support_pipe},
)
```

The step behaves like `Step.parallel` and records executed branches in `StepResult.metadata_["executed_branches"]`.

## Loop Steps

Loop steps execute a pipeline repeatedly until a condition is met.

```python
from flujo import Step

def should_continue(output, context):
    return len(str(output)) < 100

loop = Step.loop_until(
    name="refinement_loop",
    loop_body_pipeline=Pipeline.from_step(Step("refine", refine_agent)),
    exit_condition_callable=should_continue,
    max_loops=5
)
```

## Tree Search Steps

`TreeSearchStep` runs a quota-aware A* search with proposer/evaluator agents.
The evaluator should return a `Checklist`, `EvaluationScore`, or `EvaluationReport`
so the heuristic score can be derived from rubric pass rates.

```python
from flujo.domain.dsl.tree_search import TreeSearchStep

tree = TreeSearchStep(
    name="search",
    proposer=proposer_agent,
    evaluator=evaluator_agent,
    branching_factor=3,
    beam_width=3,
    max_depth=5,
    candidate_validator=lambda c: bool(c),
)
```

YAML example:

```yaml
version: "0.1"
name: "tree_search_pipeline"
steps:
  - kind: tree_search
    name: search
    proposer: "skills.search:proposer"
    evaluator: "skills.search:evaluator"
    branching_factor: 3
    beam_width: 3
    max_depth: 5
    max_iterations: 30
    path_max_tokens: 2000
    goal_score_threshold: 0.9
    require_goal: false
```

## Human-in-the-Loop Steps

Human-in-the-loop steps pause execution for human input.

```python
from flujo import Step

hitl = Step.human_in_the_loop(
    name="approval",
    message_for_user="Please review and approve the generated content"
)
```

## Map Steps

Map steps apply a pipeline to each item in an iterable from the context.

```python
from flujo import Step

class Context(BaseModel):
    items: List[str]

map_step = Step.map_over(
    name="process_items",
    pipeline_to_run=Pipeline.from_step(Step("process", process_agent)),
    iterable_input="items"
)
```

## Step Factories

Flujo provides several factory methods for creating specialized steps.

### From Callable

```python
from flujo import Step

async def my_function(data: str, *, context: BaseModel | None = None) -> str:
    return data.upper()

step = Step.from_callable(my_function, name="uppercase")
```

### From Mapper

```python
from flujo import Step

async def double(x: int) -> int:
    return x * 2

step = Step.from_mapper(double, name="double")
```

### Caching Step Results

Use `Step.cached()` to store the result of an expensive step in a cache backend.

```python
from flujo import Step
from flujo.infra.caching import InMemoryCache

expensive = Step("slow", agent)
cached = Step.cached(expensive, cache_backend=InMemoryCache())
```

On a cache hit, `StepResult.metadata_["cache_hit"]` will be `True`. The cache key
includes a stable hash of the step's full definition (agent, config, plugins,
etc.), the step input data, and any
context or resources provided.

## Validation and Error Handling

Steps can include validation plugins and error handlers.

```python
from flujo import Step

step = Step("validated", agent).add_plugin(validator, priority=1)
```

## Context Updates

Steps can update the pipeline context.

```python
from flujo import Step

step = Step("updater", agent, updates_context=True)
```

## Step Metadata

Steps can carry arbitrary metadata.

```python
from flujo import Step

step = Step("metadata", agent, meta={"version": "1.0", "author": "team"})
```

> [!TIP]
> **Rapid Iteration:**
> You can now run any custom pipeline directly from the command line using `flujo run my_pipeline.py --input "your prompt"`. This is the fastest way to test and debug your pipelines—no need for a custom script. See [usage.md](../user_guide/usage.md#running-custom-pipelines-from-the-cli-flujo-run) for details.

> [!TIP]
> **Pipeline Composition:**
> For advanced pipeline composition patterns, including wrapping entire pipelines as steps within other pipelines, see the [Pipeline as a Step](../cookbook/pipeline_as_step.md) cookbook guide.
