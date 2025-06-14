# Quickstart Guide

Get up and running with `pydantic-ai-orchestrator` in 5 minutes!

## 1. Install the Package

```bash
pip install pydantic-ai-orchestrator
```

## 2. Set Up Your API Keys

Create a `.env` file in your project directory:

```bash
cp .env.example .env
```

Add your API keys to `.env`:
```env
OPENAI_API_KEY=your_key_here
# Optional: Add other provider keys as needed
# ANTHROPIC_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

## 3. Your First Orchestration

Create a new file `hello_orchestrator.py`:

```python
from pydantic_ai_orchestrator import (
    Orchestrator, Task,
    review_agent, solution_agent, validator_agent
)

# Create an orchestrator with default agents
orch = Orchestrator(review_agent, solution_agent, validator_agent)

# Define a simple task
task = Task(prompt="Write a haiku about programming")

# Run the orchestrator
result = orch.run_sync(task)

# Print the result
if result:
    print("\n🎉 Best result:")
    print("-" * 40)
    print(f"Solution:\n{result.solution}")
    print("\nQuality Checklist:")
    for item in result.checklist.items:
        status = "✅" if item.passed else "❌"
        print(f"{status} {item.description}")
```

## 4. Run Your First Orchestration

```bash
python hello_orchestrator.py
```

## 5. Try the CLI

The package includes a command-line interface for quick tasks:

```bash
# Solve a simple task
orch solve "Write a function to calculate fibonacci numbers"

# Show your current configuration
orch show-config

# Run a quick benchmark
orch bench --prompt "Write a hello world program" --rounds 3
```

## 6. Next Steps

Now that you've got the basics working, you can:

1. Read the [Tutorial](tutorial.md) for a deeper dive
2. Explore [Use Cases](use_cases.md) for inspiration
3. Check out the [API Reference](usage.md) for more features
4. Learn about [Custom Agents](extending.md) to build your own workflows

## Common Patterns

### Using Different Models

```python
from pydantic_ai_orchestrator import make_agent_async

# Create a custom agent with a specific model
custom_agent = make_agent_async(
    "openai:gpt-4",  # Model identifier
    "You are a helpful AI assistant.",  # System prompt
    str  # Output type
)

# Use it in your orchestrator
orch = Orchestrator(custom_agent, solution_agent, validator_agent)
```

### Structured Output

```python
from pydantic import BaseModel, Field

class CodeSnippet(BaseModel):
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="The actual code")
    explanation: str = Field(..., description="Brief explanation")

# Create an agent that outputs structured data
code_agent = make_agent_async(
    "openai:gpt-4",
    "You are a programming expert. Write clean, well-documented code.",
    CodeSnippet
)
```

### Custom Pipeline

```python
from pydantic_ai_orchestrator import Step, PipelineRunner

# Create a custom pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)

# Run it
runner = PipelineRunner(pipeline)
result = runner.run("Write a function to sort a list")
```

## Need Help?

- Check the [Troubleshooting Guide](troubleshooting.md)
- Join our [Discord Community](https://discord.gg/your-server)
- Open an [Issue](https://github.com/aandresalvarez/rloop/issues) 