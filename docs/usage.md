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

# Basic Usage Guide

This guide covers the essential patterns and workflows for using Flujo effectively. You'll learn how to work with different recipes, handle results, and implement common patterns.

## Overview

Flujo provides two main approaches to AI workflow orchestration:

1. **Recipes** - Pre-built workflow patterns for common use cases
2. **Pipeline DSL** - Flexible system for building custom workflows

## Working with Recipes

### Default Recipe

The **Default Recipe** is perfect for straightforward tasks that benefit from quality control.

```python
from flujo.recipes import Default
from flujo import Task, init_telemetry

init_telemetry()

# Create the orchestrator
orch = Default()

# Define tasks
tasks = [
    Task(prompt="Write a Python function to calculate prime numbers"),
    Task(prompt="Create a REST API endpoint for user authentication"),
    Task(prompt="Generate a marketing email for a new product launch")
]

# Process tasks
for task in tasks:
    result = orch.run_sync(task)
    
    if result:
        print(f"✅ Task: {task.prompt[:50]}...")
        print(f"   Score: {result.score:.2f}")
        print(f"   Cost: ${result.metadata.get('cost_usd', 0):.4f}")
        print()
    else:
        print(f"❌ Task failed: {task.prompt[:50]}...")
```

### AgenticLoop Recipe

**AgenticLoop** is ideal for complex, multi-step tasks that require planning and tool usage.

```python
from flujo.recipes import AgenticLoop
from flujo import make_agent_async
from flujo.domain.commands import AgentCommand
from pydantic import TypeAdapter

# Define tool agents
async def web_search(query: str) -> str:
    """Simulate web search functionality."""
    return f"Search results for: {query}"

async def calculator(expression: str) -> str:
    """Simple calculator tool."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

# Create planner agent
PLANNER_PROMPT = """
You are a research assistant with access to web search and calculator tools.

Available tools:
- web_search: Find information on the web
- calculator: Perform mathematical calculations

Use these tools to answer questions thoroughly. When you have a complete answer, use FinishCommand.
"""

planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    TypeAdapter(AgentCommand)
)

# Create AgenticLoop
loop = AgenticLoop(
    planner_agent=planner,
    agent_registry={
        "web_search": web_search,
        "calculator": calculator
    }
)

# Run complex task
result = loop.run("What is the population of Tokyo and how does it compare to New York's population?")

print("Final Answer:", result.final_pipeline_context.command_log[-1].execution_result)
```

## Working with Results

### Handling Default Recipe Results

```python
from flujo.recipes import Default
from flujo import Task

orch = Default()
result = orch.run_sync(Task(prompt="Write a Python decorator"))

if result:
    # Access the solution
    solution = result.solution
    print(f"Solution:\n{solution}")
    
    # Check quality score
    score = result.score
    print(f"Quality Score: {score:.2f}")
    
    # Review the checklist
    if result.checklist:
        print("\nQuality Assessment:")
        for item in result.checklist.items:
            status = "✅" if item.passed else "❌"
            print(f"  {status} {item.description}")
    
    # Check metadata
    metadata = result.metadata
    print(f"\nMetadata:")
    print(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")
    print(f"  Duration: {metadata.get('duration_ms', 0)}ms")
    print(f"  Steps: {metadata.get('steps', 0)}")
else:
    print("Workflow failed")
```

### Handling AgenticLoop Results

```python
from flujo.recipes import AgenticLoop

# ... setup code ...

result = loop.run("Your task here")

# Access the final answer
final_answer = result.final_pipeline_context.command_log[-1].execution_result
print(f"Final Answer: {final_answer}")

# Review the execution history
print("\nExecution History:")
for i, command in enumerate(result.final_pipeline_context.command_log, 1):
    print(f"{i}. {command.command_type}")
    if command.agent_name:
        print(f"   Agent: {command.agent_name}")
    if command.execution_result:
        print(f"   Result: {command.execution_result[:100]}...")
    print()
```

## Error Handling

### Graceful Error Handling

```python
from flujo.recipes import Default
from flujo import Task
from flujo.exceptions import FlujoError

orch = Default()

try:
    result = orch.run_sync(Task(prompt="Your task"))
    
    if result:
        print(f"Success: {result.solution}")
    else:
        print("Workflow completed but no result returned")
        
except FlujoError as e:
    print(f"Flujo error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Retry Logic

```python
import asyncio
from flujo.recipes import Default
from flujo import Task

async def run_with_retry(task, max_retries=3):
    orch = Default()
    
    for attempt in range(max_retries):
        try:
            result = orch.run_sync(task)
            if result:
                return result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("All retry attempts failed")

# Usage
task = Task(prompt="Your task")
result = await run_with_retry(task)
```

## Batch Processing

### Processing Multiple Tasks

```python
from flujo.recipes import Default
from flujo import Task
import asyncio

async def process_batch(tasks, max_concurrent=5):
    orch = Default()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(task):
        async with semaphore:
            return await orch.run_async(task)
    
    # Process tasks concurrently
    results = await asyncio.gather(
        *[process_single(task) for task in tasks],
        return_exceptions=True
    )
    
    return results

# Usage
tasks = [
    Task(prompt="Task 1"),
    Task(prompt="Task 2"),
    Task(prompt="Task 3"),
    # ... more tasks
]

results = await process_batch(tasks)
```

## Configuration Management

### Environment-Based Configuration

```python
import os
from flujo.recipes import Default
from flujo import make_agent_async

# Configure based on environment
env = os.getenv("FLUJO_ENV", "development")

if env == "production":
    # Use more expensive models for better quality
    solution_agent = make_agent_async("openai:gpt-4", "You are a production assistant", str)
else:
    # Use cheaper models for development
    solution_agent = make_agent_async("openai:gpt-4o-mini", "You are a development assistant", str)

orch = Default(solution_agent=solution_agent)
```

### Custom Configuration

```python
from flujo.recipes import Default
from flujo import make_agent_async

# Create custom agents with specific configurations
review_agent = make_agent_async(
    "openai:gpt-4",
    "You are a senior code reviewer. Be thorough and critical.",
    str
)

solution_agent = make_agent_async(
    "openai:gpt-4o",
    "You are a Python expert. Write clean, efficient code.",
    str
)

validator_agent = make_agent_async(
    "openai:gpt-4",
    "You are a QA engineer. Validate code quality and functionality.",
    str
)

# Create custom orchestrator
orch = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    max_iters=5,  # Limit iterations
    k_variants=3  # Generate multiple variants
)
```

## Performance Optimization

### Cost Control

```python
from flujo.recipes import Default
from flujo import Task

orch = Default()

# Monitor costs
task = Task(prompt="Your task")
result = orch.run_sync(task)

if result:
    cost = result.metadata.get('cost_usd', 0)
    print(f"Task cost: ${cost:.4f}")
    
    # Set budget limits
    if cost > 0.10:  # $0.10 limit
        print("⚠️ Task exceeded cost threshold")
```

### Caching

```python
from flujo.recipes import Default
from flujo import Task
import hashlib
import json

class CachedOrchestrator:
    def __init__(self, cache_file="flujo_cache.json"):
        self.orch = Default()
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _get_cache_key(self, task):
        return hashlib.md5(task.prompt.encode()).hexdigest()
    
    def run_sync(self, task):
        cache_key = self._get_cache_key(task)
        
        if cache_key in self.cache:
            print("Using cached result")
            return self.cache[cache_key]
        
        result = self.orch.run_sync(task)
        if result:
            self.cache[cache_key] = result
            self._save_cache()
        
        return result

# Usage
cached_orch = CachedOrchestrator()
result = cached_orch.run_sync(Task(prompt="Your task"))
```

## Best Practices

### 1. Use Appropriate Recipes

- **Default Recipe**: For straightforward tasks with quality requirements
- **AgenticLoop**: For complex, multi-step tasks requiring planning

### 2. Handle Results Properly

- Always check if results exist before accessing them
- Use try-catch blocks for error handling
- Monitor costs and performance metrics

### 3. Optimize for Your Use Case

- Use cheaper models for development
- Implement caching for repeated tasks
- Use batch processing for multiple tasks

### 4. Monitor and Debug

- Enable telemetry for production monitoring
- Use verbose logging for debugging
- Track costs and performance metrics

### 5. Error Recovery

- Implement retry logic for transient failures
- Use fallback strategies when possible
- Log errors for debugging

## Common Patterns

### Code Generation with Review

```python
from flujo.recipes import Default
from flujo import Task

def generate_code_with_review(requirements: str):
    orch = Default()
    
    task = Task(
        prompt=f"Generate Python code for: {requirements}",
        metadata={"type": "code_generation"}
    )
    
    result = orch.run_sync(task)
    
    if result and result.score > 0.8:
        return result.solution
    else:
        # Regenerate if quality is too low
        return generate_code_with_review(requirements)
```

### Content Generation with Validation

```python
from flujo.recipes import Default
from flujo import Task

def generate_content_with_validation(content_type: str, topic: str):
    orch = Default()
    
    task = Task(
        prompt=f"Generate {content_type} about {topic}",
        metadata={"content_type": content_type, "topic": topic}
    )
    
    result = orch.run_sync(task)
    
    if result:
        return {
            "content": result.solution,
            "quality_score": result.score,
            "validation_passed": result.checklist is not None
        }
    return None
```

This guide covers the essential patterns for using Flujo effectively. For more advanced usage, see the [Pipeline DSL Guide](pipeline_dsl.md) and [Cookbook](cookbook/) sections.
