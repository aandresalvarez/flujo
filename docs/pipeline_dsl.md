# Pipeline DSL Guide

The Pipeline Domain-Specific Language (DSL) is a powerful way to create custom AI workflows in `flujo`. This guide explains how to use it effectively.

## Overview

The Pipeline DSL lets you:

- Compose complex workflows from simple steps **and from other pipelines**
- Mix and match different agents
- Add custom validation and scoring
- Create reusable pipeline components

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

# With custom scoring
from flujo import weighted_score

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
from flujo.validation import BaseValidator, ValidationResult

class WordCountValidator(BaseValidator):
    async def validate(self, output_to_check: str, *, context=None) -> ValidationResult:
        return ValidationResult(is_valid=len(output_to_check.split()) < 5, validator_name=self.name,
                                feedback="Too many words" if len(output_to_check.split()) >= 5 else None)

validate_step = Step.validate_step(
    validator_agent,
    validators=[WordCountValidator()]
)

See [Hybrid Validation Cookbook](cookbook/hybrid_validation.md) for a complete example.
```

All step factories also accept a `processors: Optional[AgentProcessors]` parameter
to run pre-processing and post-processing hooks. See [Using Processors](cookbook/using_processors.md)
for details.
For complex data shaping before calling another step, consider using an [Adapter Step](cookbook/adapter_step.md).

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

a `Flujo` runner can share a mutable Pydantic model instance across all steps in a single run. Pass a context model when creating the runner and declare either `context` or `pipeline_context` in your step functions or agents. If both are present, `context` is prioritized for backward compatibility. See [Typed Pipeline Context](pipeline_context.md) for a full explanation.

```python
from pydantic import BaseModel

class MyContext(BaseModel):
    counter: int = 0

@step
async def increment(data: str, *, pipeline_context: MyContext | None = None) -> str:
    if pipeline_context:
        pipeline_context.counter += 1
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

You can also pass a long-lived resources container to the runner. Declare a
keyword-only `resources` argument in your agents or plugins to use it.

```python
class MyResources(AppResources):
    db_pool: Any

@step
async def query(data: int, *, resources: MyResources) -> str:
    return resources.db_pool.get_user(data)

runner = Flujo(query, resources=my_resources)
```

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

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](usage.md)
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](use_cases.md) 