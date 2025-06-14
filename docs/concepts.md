# Core Concepts

This guide explains the fundamental concepts that power `pydantic-ai-orchestrator`. Understanding these concepts will help you build more effective AI workflows.

## The Orchestrator

The **`Orchestrator` class** is a central component that manages a **standard,
pre-defined AI workflow**. Think of it as a project manager for a specific team
structure: a Review Agent, a Solution Agent, and a Validator Agent (optionally
followed by a Reflection Agent). It handles the flow of information between
these fixed roles, manages retries based on their internal configuration, and
aims to return the best `Candidate` solution from this standard process.
For building custom workflows with different steps or logic, you would use the
`Pipeline` DSL and `PipelineRunner`.

```python
from pydantic_ai_orchestrator import (
    Orchestrator, review_agent, solution_agent, validator_agent
)

# Assemble the orchestrator with default agents
orch = Orchestrator(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
)
```

## Agents

**Agents** are specialized AI models with specific roles. Each agent has:

- A system prompt that defines its role
- An output type (string, Pydantic model, etc.)
- Optional tools for external interactions

### Default Agents

The library provides three default agents:

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

### Creating Custom Agents

```python
from pydantic_ai_orchestrator import make_agent_async

custom_agent = make_agent_async(
    "openai:gpt-4",  # Model
    "You are a Python expert.",  # System prompt
    str  # Output type
)
```

## Tasks

A **Task** represents a single request to the orchestrator. It contains:

- The prompt (what you want to achieve)
- Optional metadata
- Optional constraints

```python
from pydantic_ai_orchestrator import Task

task = Task(
    prompt="Write a function to calculate prime numbers",
    metadata={"language": "python", "complexity": "medium"}
)
```

## Candidates

A **Candidate** is a potential solution produced by the orchestrator. It includes:

- The solution itself
- A quality checklist
- Metadata about the generation process

```python
result = orch.run_sync(task)
if result:  # result is a Candidate
    print(f"Solution: {result.solution}")
    print(f"Quality Score: {result.score}")
    print("Checklist:")
    for item in result.checklist.items:
        print(f"- {item.description}: {'✅' if item.passed else '❌'}")
```

## The Pipeline DSL

The **Pipeline Domain-Specific Language (DSL)**, using `Step` objects and
executed by `PipelineRunner`, is the primary way to create **flexible and custom
multi-agent workflows**. This gives you full control over the sequence of
operations, the agents used at each stage, and the integration of plugins.

`PipelineRunner` can also maintain a shared, typed context object for each run.
Steps declare a `pipeline_context` parameter to access or modify this object. See
[Typed Pipeline Context](pipeline_context.md) for full documentation.
The built-in [**Orchestrator**](#the-orchestrator) uses this DSL under the hood. When you need different logic, you can use the same tools directly. The DSL also supports advanced constructs like [**LoopStep**](pipeline_looping.md) for iteration and [**ConditionalStep**](pipeline_branching.md) for branching workflows.

```python
from pydantic_ai_orchestrator import (
    Step, PipelineRunner, review_agent, solution_agent, validator_agent
)

# Define a pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

# Run it
runner = PipelineRunner(pipeline)
pipeline_result = runner.run("Your prompt here")
for step_res in pipeline_result.step_history:
    print(step_res.name, step_res.success)
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
   - Apply custom validation rules

## Scoring

The orchestrator uses scoring to evaluate and select the best solution. Scoring can be:

- **Ratio-based**: Simple pass/fail ratio
- **Weighted**: Different criteria have different importance
- **Model-based**: Using an AI model to evaluate quality

```python
from pydantic_ai_orchestrator import ratio_score, weighted_score

# Simple ratio scoring
score = ratio_score(checklist)

# Weighted scoring
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

## Telemetry

The orchestrator includes built-in telemetry for:

- Performance monitoring
- Usage tracking
- Error reporting
- Distributed tracing

```python
from pydantic_ai_orchestrator import init_telemetry

# Initialize telemetry
init_telemetry()

# Enable OTLP export
import os
os.environ["OTLP_EXPORT_ENABLED"] = "true"
os.environ["OTLP_ENDPOINT"] = "https://your-otlp-endpoint"
```

## Best Practices

1. **Agent Design**
   - Give clear, specific system prompts
   - Use appropriate output types
   - Include relevant tools

2. **Pipeline Design**
   - Start simple, add complexity as needed
   - Use validation steps for quality control
   - Consider cost and performance

3. **Error Handling**
   - Implement proper retry logic
   - Handle API failures gracefully
   - Log errors for debugging

4. **Performance**
   - Use appropriate models for each step
   - Implement caching where possible
   - Monitor and optimize costs

## Next Steps

- Try the [Tutorial](tutorial.md) for hands-on examples
- Explore [Use Cases](use_cases.md) for inspiration
- Read the [API Reference](usage.md) for details
- Learn about [Custom Agents](extending.md) 