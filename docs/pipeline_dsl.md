# Pipeline DSL Guide

The Pipeline Domain-Specific Language (DSL) is a powerful way to create custom AI workflows in `pydantic-ai-orchestrator`. This guide explains how to use it effectively.

## Overview

The Pipeline DSL lets you:

- Compose complex workflows from simple steps
- Mix and match different agents
- Add custom validation and scoring
- Create reusable pipeline components

## Basic Usage

### Creating a Pipeline

```python
from pydantic_ai_orchestrator import Step, PipelineRunner
from pydantic_ai_orchestrator.infra.agents import (
    review_agent, solution_agent, validator_agent
)

# Create a basic pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

# Run it
runner = PipelineRunner(pipeline)
result = runner.run("Write a function to sort a list")
```

### Pipeline Composition

The `>>` operator chains steps together:

```python
# Create reusable steps
review_step = Step.review(review_agent)
solution_step = Step.solution(solution_agent)
validate_step = Step.validate(validator_agent)

# Compose them in different ways
pipeline1 = review_step >> solution_step >> validate_step
pipeline2 = solution_step >> validate_step  # Skip review
pipeline3 = review_step >> solution_step >> solution_step >> validate_step  # Double solution
```

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
validate_step = Step.validate(validator_agent)

# With custom scoring
from pydantic_ai_orchestrator import weighted_score

weights = {
    "correctness": 0.6,
    "readability": 0.4
}

validate_step = Step.validate(
    validator_agent,
    scorer=lambda c: weighted_score(c, weights)
)

# With plugins
from pydantic_ai_orchestrator.plugins import SQLSyntaxValidator

validate_step = Step.validate(
    validator_agent,
    plugins=[SQLSyntaxValidator()]
)
```

## Advanced Features

### Parallel Execution

Run steps in parallel:

```python
from pydantic_ai_orchestrator import parallel

# Run multiple solution steps in parallel
pipeline = (
    Step.review(review_agent)
    >> parallel(
        Step.solution(solution_agent),
        Step.solution(alternative_agent)
    )
    >> Step.validate(validator_agent)
)
```

### Looping and Iteration

Repeat a sub-pipeline until a condition is met using `Step.loop_until()`.
See [LoopStep documentation](pipeline_looping.md) for full details.

```python
loop_step = Step.loop_until(
    name="refine",
    loop_body_pipeline=Pipeline.from_step(Step.solution(solution_agent)),
    exit_condition_callable=lambda out, ctx: "done" in out,
)

pipeline = Step.review(review_agent) >> loop_step >> Step.validate(validator_agent)
```

## Typed Pipeline Context

`PipelineRunner` can share a mutable Pydantic model instance across all steps in
a single run. Pass a context model when creating the runner and declare
`pipeline_context` in your step functions or agents. See
[Typed Pipeline Context](pipeline_context.md) for a full explanation.

```python
from pydantic import BaseModel

class MyContext(BaseModel):
    counter: int = 0

async def increment(data: str, *, pipeline_context: MyContext | None = None) -> str:
    if pipeline_context:
        pipeline_context.counter += 1
    return data

pipeline = Step("inc", increment) >> Step("read", increment)
runner = PipelineRunner(pipeline, context_model=MyContext)
result = runner.run("hi")
print(result.final_pipeline_context.counter)  # 2
```

Each `run()` call gets a fresh context instance. Access the final state via
`PipelineResult.final_pipeline_context`.

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

pipeline = Step.solution(solution_agent) >> branch_step >> Step.validate(validator_agent)
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
    >> Step.validate(validator_agent)
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
runner = PipelineRunner(
    pipeline,
    max_retries=3,
    retry_on_error=True
)
```

### Error Recovery

```python
from pydantic_ai_orchestrator import fallback

# Add fallback steps
pipeline = (
    Step.solution(solution_agent)
    >> fallback(
        Step.solution(backup_agent),
        on_error=True
    )
    >> Step.validate(validator_agent)
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
   - Include fallback steps
   - Log errors properly
   - Monitor performance

3. **Performance**
   - Use parallel execution
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
from pydantic_ai_orchestrator import Step, PipelineRunner
from pydantic_ai_orchestrator.plugins import (
    SQLSyntaxValidator,
    CodeStyleValidator
)

# Create a code generation pipeline
pipeline = (
    Step.review(review_agent)  # Define requirements
    >> Step.solution(code_agent)  # Generate code
    >> Step.validate(
        validator_agent,
        plugins=[
            SQLSyntaxValidator(),
            CodeStyleValidator()
        ]
    )
)

# Run it
runner = PipelineRunner(pipeline)
result = runner.run("Write a SQL query to find active users")
```

### Content Generation Pipeline

```python
# Create a content generation pipeline
pipeline = (
    Step.review(review_agent)  # Define content guidelines
    >> parallel(
        Step.solution(writer_agent),  # Main writer
        Step.solution(editor_agent)   # Alternative version
    )
    >> Step.validate(
        validator_agent,
        scorer=lambda c: weighted_score(c, {
            "grammar": 0.3,
            "style": 0.3,
            "tone": 0.4
        })
    )
)

# Run it
runner = PipelineRunner(pipeline)
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
   - Use parallel execution

3. **Quality Issues**
   - Review scoring weights
   - Check validation rules
   - Monitor success rates
   - Adjust agents

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/rloop/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](usage.md)
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](use_cases.md) 