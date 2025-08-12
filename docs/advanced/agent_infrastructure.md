# Agent Infrastructure

This document explains the agent infrastructure in `flujo` and how to use factory functions for creating agents.

## Overview

The agent infrastructure provides a clean, decoupled approach to creating and configuring agents. Factory functions allow you to create agents with specific configurations, providing better separation of concerns and explicit control over agent creation.

## Factory Functions

### `make_review_agent()`

Creates a review agent that generates quality checklists.

```python
from flujo.agents import make_review_agent

# Create with default settings
review_agent = make_review_agent()

# Create with custom model
review_agent = make_review_agent(model="openai:gpt-4o")

# Create with custom prompt
review_agent = make_review_agent(
    model="openai:gpt-4o",
    prompt="You are a specialized code reviewer. Create detailed checklists for Python code quality."
)
```

### `make_solution_agent()`

Creates a solution agent that generates the main output.

```python
from flujo.agents import make_solution_agent

# Create with default settings
solution_agent = make_solution_agent()

# Create with custom model
solution_agent = make_solution_agent(model="openai:gpt-4o-mini")

# Create with custom prompt
solution_agent = make_solution_agent(
    model="openai:gpt-4o-mini",
    prompt="You are a creative writer. Generate engaging and original content."
)
```

### `make_validator_agent()`

Creates a validator agent that evaluates solutions against checklists.

```python
from flujo.agents import make_validator_agent

# Create with default settings
validator_agent = make_validator_agent()

# Create with custom model
validator_agent = make_validator_agent(model="openai:gpt-4o")

# Create with custom prompt
validator_agent = make_validator_agent(
    model="openai:gpt-4o",
    prompt="You are a strict quality analyst. Rigorously evaluate solutions against provided checklists."
)
```

## Prompt Management

All system prompts are centralized in the `flujo.prompts` module. This provides better organization and makes it easier to maintain and customize prompts.

### Accessing Prompts

```python
from flujo.prompts import (
    REVIEW_PROMPT,
    SOLUTION_PROMPT,
    VALIDATION_PROMPT,
    REFLECTION_PROMPT
)

# Use prompts in custom agent creation
custom_review_agent = make_agent_async(
    "openai:gpt-4o",
    REVIEW_PROMPT,
    Checklist
)
```

### Customizing Prompts

You can create custom prompts by extending the base prompts:

```python
from flujo.prompts import REVIEW_PROMPT

# Create a specialized review prompt
CODE_REVIEW_PROMPT = f"""
{REVIEW_PROMPT}

Additionally, focus on:
- Code style and formatting
- Performance considerations
- Security best practices
- Documentation quality
"""

code_review_agent = make_agent_async(
    "openai:gpt-4o",
    CODE_REVIEW_PROMPT,
    Checklist
)
```

## Usage Examples

### Basic Pipeline Creation

```python
from flujo import Step, Flujo
from flujo.agents import make_review_agent, make_solution_agent, make_validator_agent

# Create a pipeline using factory functions
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(make_solution_agent())
    >> Step.validate(make_validator_agent())
)

# Run the pipeline
runner = Flujo(pipeline)
result = runner.run("Write a Python function to calculate fibonacci numbers")
```

### Recipe Creation

```python
from flujo.recipes.factories import make_default_pipeline
from flujo.agents import make_review_agent, make_solution_agent, make_validator_agent

# Create a pipeline using the factory functions
pipeline = make_default_pipeline(
    review_agent=make_review_agent(),
    solution_agent=make_solution_agent(),
    validator_agent=make_validator_agent(),
)
```

### Pipeline Factory Usage

```python
from flujo.recipes.factories import make_default_pipeline
from flujo.agents import make_review_agent, make_solution_agent, make_validator_agent

# Create a pipeline using the factory
pipeline = make_default_pipeline(
    review_agent=make_review_agent(),
    solution_agent=make_solution_agent(),
    validator_agent=make_validator_agent(),
)
```

### Custom Agent Configuration

```python
# Create agents with specific configurations
review_agent = make_review_agent(
    model="openai:gpt-4o",
    prompt="You are a code quality expert. Create detailed checklists for Python code."
)

solution_agent = make_solution_agent(
    model="openai:gpt-4o-mini",  # Faster, cheaper model
    prompt="You are a Python developer. Write clean, efficient code."
)

validator_agent = make_validator_agent(
    model="openai:gpt-4o",
    prompt="You are a senior developer. Rigorously review code quality."
)

# Use in pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(validator_agent)
)
```

## Benefits

### 1. Explicit Dependencies

Factory functions make dependencies explicit and easier to understand:

```python
# Clear what agents are being used
pipeline = (
    Step.review(make_review_agent(model="openai:gpt-4o"))
    >> Step.solution(make_solution_agent(model="openai:gpt-4o-mini"))
    >> Step.validate(make_validator_agent(model="openai:gpt-4o"))
)
```

### 2. Better Testing

Factory functions make it easier to test with different configurations:

```python
# Test with different models
test_pipeline = (
    Step.review(make_review_agent(model="test-model"))
    >> Step.solution(make_solution_agent(model="test-model"))
    >> Step.validate(make_validator_agent(model="test-model"))
)
```

### 3. Improved Maintainability

Centralized prompt management makes it easier to maintain and update system prompts:

```python
# All prompts in one place
from flujo.prompts import *

# Easy to customize
custom_prompt = f"{REVIEW_PROMPT}\n\nAdditional instructions: ..."
```

### 4. Better Composition

Factory functions enable better composition and reuse:

```python
def create_code_review_pipeline():
    """Create a specialized code review pipeline."""
    return (
        Step.review(make_review_agent(prompt=CODE_REVIEW_PROMPT))
        >> Step.solution(make_solution_agent(model="openai:gpt-4o"))
        >> Step.validate(make_validator_agent(prompt=CODE_VALIDATION_PROMPT))
    )
```

## Best Practices

### 1. Use Factory Functions

Always use factory functions to create agents:

```python
# ✅ Good
agent = make_review_agent()

# ✅ Good - with customization
agent = make_review_agent(
    model="openai:gpt-4o",
    prompt="Specialized prompt for code review"
)
```

### 2. Centralize Custom Prompts

Create custom prompts in a dedicated module:

```python
# prompts/custom.py
from flujo.prompts import REVIEW_PROMPT

CODE_REVIEW_PROMPT = f"""
{REVIEW_PROMPT}

Focus on:
- Code quality and style
- Performance considerations
- Security best practices
"""
```

### 3. Use Type Hints

Leverage type hints for better IDE support:

```python
from flujo.agents import make_review_agent
from flujo.models import Checklist

# Type hints help with IDE support
review_agent: AsyncAgentProtocol[Any, Checklist] = make_review_agent()
```

### 4. Configure for Your Use Case

Choose appropriate models and prompts for your specific needs:

```python
# For code generation
code_review_agent = make_review_agent(
    model="openai:gpt-4o",
    prompt="You are a senior Python developer. Create checklists for code quality."
)

# For content creation
content_review_agent = make_review_agent(
    model="openai:gpt-4o",
    prompt="You are a content editor. Create checklists for writing quality."
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're importing factory functions from `flujo.infra.agents`
   - Check that you're using the correct import path

2. **Configuration Issues**
   - Use factory function parameters for customization
   - Check prompt content in `flujo.prompts`

3. **Model Issues**
   - Verify model names are correct
   - Check API key configuration

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review the [API Reference](../api/index.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md) for examples
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)
