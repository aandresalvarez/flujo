# Core Concepts

This guide explains the fundamental concepts that power `flujo`. Understanding these concepts will help you build more effective AI workflows.

## The Default Recipe

The **`Default` recipe** is a high-level helper that manages a **standard,
pre-defined AI workflow**. Think of it as a project manager for a specific team
structure: a Review Agent, a Solution Agent, a Validator Agent, and optionally a
Reflection Agent. It handles the flow of information between these fixed roles,
manages retries based on their internal configuration, and returns the best
`Candidate` from this standard process. For building custom workflows with
different steps or logic, you would use the `Pipeline` DSL and the `Flujo`
engine.

```python
from flujo.recipes import Default
from flujo import review_agent, solution_agent, validator_agent, reflection_agent

# Assemble the default recipe with the built-in agents
recipe = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,  # Optional but recommended
)
```

## Agents

**Agents** are specialized AI models with specific roles. Each agent has:

- A system prompt that defines its role
- An output type (string, Pydantic model, etc.)
- Optional tools for external interactions

### Default Agents

The library provides four default agents:

1. **Review Agent** (`review_agent`)
   - Role: Creates a quality checklist
   - Output: `Checklist` model
   - Purpose: Defines what "good" looks like

2. **Solution Agent** (`solution_agent`)
   - Role: Generates the actual solution
   - Output: String or custom model
   - Purpose: Does the main work

3. **Validator Agent** (`validator_agent`)
   - Role: Evaluates the solution
   - Output: `Checklist` model
   - Purpose: Quality control

4. **Reflection Agent** (`reflection_agent`)
   - Role: Provides improvement suggestions and meta-analysis
   - Output: String
   - Purpose: Self-improvement and iteration guidance

### Creating Custom Agents

```python
from flujo import make_agent_async

custom_agent = make_agent_async(
    "openai:gpt-4",  # Model
    "You are a Python expert.",  # System prompt
    str  # Output type
)
```

## Tasks

A **Task** represents a single request to the orchestrator. It contains:

- The prompt (what you want to achieve)
- Optional metadata for additional context

```python
from flujo import Task

task = Task(
    prompt="Write a function to calculate prime numbers",
    metadata={"language": "python", "complexity": "medium"}
)
```

## Candidates

A **Candidate** is a potential solution produced by the orchestrator. It includes:

- The solution itself
- A quality score (0.0 to 1.0)
- A quality checklist evaluation

```python
result = orch.run_sync(task)
if result:  # result is a Candidate
    print(f"Solution: {result.solution}")
    print(f"Quality Score: {result.score}")
    if result.checklist:
        print("Checklist:")
        for item in result.checklist.items:
            print(f"- {item.description}: {'✅' if item.passed else '❌'}")
```

## The Pipeline DSL

The **Pipeline Domain-Specific Language (DSL)**, using `Step` objects and
executed by `Flujo`, is the primary way to create **flexible and custom
multi-agent workflows**. This gives you full control over the sequence of
operations, the agents used at each stage, and the integration of plugins.

`Flujo` can also maintain a shared, typed context object for each run.
Steps declare a `pipeline_context` parameter to access or modify this object. See
[Typed Pipeline Context](pipeline_context.md) for full documentation.

The built-in [**Default recipe**](#the-default-recipe) uses this DSL under the hood. When you need different logic, you can use the same tools directly through the `Flujo` engine. The DSL also supports advanced constructs like [**LoopStep**](pipeline_looping.md) for iteration and [**ConditionalStep**](pipeline_branching.md) for branching workflows.

```python
from flujo import (
    Step, Flujo, review_agent, solution_agent, validator_agent
)

# Define a pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

# Run it
runner = Flujo(pipeline)
pipeline_result = runner.run("Your prompt here")
for step_res in pipeline_result.step_history:
    print(f"Step: {step_res.name}, Success: {step_res.success}")
```

### Step Types

1. **Review Steps**
   - Create quality checklists
   - Define success criteria

2. **Solution Steps**
   - Generate the main output
   - Can use tools and external services

3. **Validation Steps**
   - Verify the solution
   - Apply custom validation rules using plugins

4. **Custom Steps**
   - Any agent can be used in a step
   - Flexible configuration and tool integration

### Advanced Pipeline Constructs

#### Loop Steps

For iterative refinement:

```python
from flujo import Step, Pipeline

loop_step = Step.loop_until(
    name="refinement_loop",
    loop_body_pipeline=Pipeline.from_step(Step.solution(solution_agent)),
    exit_condition_callable=lambda output, context: "done" in output.lower(),
)
```

#### Conditional Steps

For branching logic:

```python
router_step = Step.branch_on(
    name="content_router",
    condition_callable=lambda output, context: "code" if "function" in output else "text",
    branches={
        "code": Pipeline.from_step(Step.solution(code_agent)),
        "text": Pipeline.from_step(Step.solution(text_agent)),
    },
)
```

## Scoring

The orchestrator uses scoring to evaluate and select the best solution. Scoring strategies include:

- **Ratio-based**: Simple pass/fail ratio from checklist items
- **Weighted**: Different criteria have different importance
- **Model-based**: Using an AI model to evaluate quality

```python
from flujo.domain.scoring import ratio_score, weighted_score

# Simple ratio scoring (default)
score = ratio_score(checklist)

# Weighted scoring with custom weights
weights = {
    "correctness": 0.5,
    "readability": 0.3,
    "efficiency": 0.2
}
score = weighted_score(checklist, weights)
```

## Tools

Tools allow agents to interact with external systems. They can:

- Fetch data from APIs
- Execute code
- Interact with databases
- Call other services

```python
from pydantic_ai import Tool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Implementation here
    return f"Weather in {city}: Sunny"

# Create a tool
weather_tool = Tool(get_weather)

# Give it to an agent
agent = make_agent_async(
    "openai:gpt-4",
    "You are a weather assistant.",
    str,
    tools=[weather_tool]
)
```

## Plugins

Plugins extend pipeline functionality, particularly for validation:

```python
from flujo import ValidationPlugin, PluginOutcome
from flujo.plugins import SQLSyntaxValidator

# Use built-in SQL validator
sql_validator = SQLSyntaxValidator()

# Create custom plugin
class CustomValidator(ValidationPlugin):
    def validate(self, output: Any, context: Any) -> PluginOutcome:
        if self.is_valid(output):
            return PluginOutcome(passed=True)
        return PluginOutcome(passed=False, feedback="Validation failed")

# Use in pipeline
pipeline = Step.validate(validator_agent, plugins=[sql_validator, CustomValidator()])
```

## Self-Improvement & Evaluation

The library includes intelligent evaluation capabilities:

```python
from flujo import evaluate_and_improve, SelfImprovementAgent
from flujo.infra.agents import make_self_improvement_agent

# Create improvement agent
improvement_agent = SelfImprovementAgent(make_self_improvement_agent())

# Generate improvement suggestions
report = await evaluate_and_improve(
    task_fn=your_task_function,
    dataset=your_evaluation_dataset,
    agent=improvement_agent,
    pipeline_definition=your_pipeline
)

# Review suggestions
for suggestion in report.suggestions:
    print(f"Issue: {suggestion.failure_pattern_summary}")
    print(f"Fix: {suggestion.detailed_explanation}")
```

## Telemetry

The orchestrator includes built-in telemetry for:

- Performance monitoring
- Usage tracking
- Error reporting
- Distributed tracing

```python
from flujo import init_telemetry

# Initialize telemetry
init_telemetry()

# Configure via environment variables:
# TELEMETRY_EXPORT_ENABLED=true
# OTLP_EXPORT_ENABLED=true
# OTLP_ENDPOINT=https://your-otlp-endpoint
```

## Configuration

Settings can be controlled via environment variables or the settings object:

```python
from flujo import settings

# Access current settings
print(f"Default solution model: {settings.default_solution_model}")
print(f"Reflection enabled: {settings.reflection_enabled}")

# Environment variables (in .env file):
# DEFAULT_SOLUTION_MODEL=openai:gpt-4o
# REFLECTION_ENABLED=true
# AGENT_TIMEOUT=60
```

## Managed Resources

`Flujo` supports an optional *resources* container that is passed to every step
and plugin during a pipeline run. This is a convenient place to keep shared
objects like database connections or API clients.

Create your own container by inheriting from `AppResources` and pass an
instance to the runner:

```python
class MyResources(AppResources):
    db_pool: Any

resources = MyResources(db_pool=make_pool())
runner = Flujo(pipeline, resources=resources)
```

Any agent or plugin can declare a keyword-only argument named `resources` to
receive this object.

## Lifecycle Hooks

Lifecycle hooks let you run custom code before and after key events such as
`pre_run`, `post_run`, `pre_step`, `post_step`, and `on_step_failure`.

```python
async def log_hook(**kwargs):
    print("event", kwargs.get("event_name"))

runner = Flujo(pipeline, hooks=[log_hook])
```

Raise `PipelineAbortSignal` from a hook to stop execution.

## Best Practices

1. **Agent Design**
   - Give clear, specific system prompts
   - Use appropriate output types
   - Include relevant tools when needed

2. **Pipeline Design**
   - Start simple, add complexity as needed
   - Use validation steps for quality control
   - Consider cost and performance implications

3. **Error Handling**
   - Implement proper retry logic
   - Handle API failures gracefully
   - Log errors for debugging

4. **Performance**
   - Use appropriate models for each step
   - Implement caching where possible
   - Monitor and optimize costs

5. **Quality Control**
   - Use reflection agents for self-improvement
   - Implement custom validation plugins
   - Monitor quality metrics over time

## Usage Governor

`Flujo` can enforce **cost and token limits** during a pipeline run. Provide a
`UsageLimits` object when creating the runner. If the pipeline exceeds either
limit, it stops and raises `UsageLimitExceededError` with the partial result.

```python
from flujo import Flujo, Step, UsageLimits

limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=5000)
runner = Flujo(my_pipeline, usage_limits=limits)
```

## Next Steps

- Try the [Tutorial](tutorial.md) for hands-on examples
- Explore [Use Cases](use_cases.md) for inspiration
- Read the [API Reference](api_reference.md) for details
- Learn about [Custom Components](extending.md) 
