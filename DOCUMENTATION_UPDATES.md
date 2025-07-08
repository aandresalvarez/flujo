# Documentation for Agent Infrastructure

This document summarizes the documentation for the agent infrastructure in `flujo`.

## Overview

The agent infrastructure provides a clean, decoupled approach to creating and configuring agents. Factory functions allow you to create agents with specific configurations, providing better separation of concerns and explicit control over agent creation.

## Key Features

### 1. Factory Functions

The library provides factory functions for creating agents:

- `make_review_agent()` - Creates review agents that generate quality checklists
- `make_solution_agent()` - Creates solution agents that generate the main output
- `make_validator_agent()` - Creates validator agents that evaluate solutions

### 2. Centralized Prompt Management

All system prompts are centralized in the `flujo.prompts` module, providing better organization and easier maintenance.

### 3. Explicit Dependencies

Factory functions make dependencies explicit and easier to understand, enabling better testing and composition.

## Documentation Structure

### 1. Core Documentation Files

#### `docs/usage.md`
- Shows how to use factory functions to create agents
- Demonstrates pipeline creation with factory functions
- Includes proper import statements for factory functions

#### `docs/concepts.md`
- Uses factory functions in all examples
- Shows recipe creation with factory functions
- Demonstrates pipeline DSL usage with factory functions

#### `docs/tutorial.md`
- Uses generic agent descriptions
- Shows code examples with factory functions
- Demonstrates model mixing with factory functions

### 2. API Reference

#### `docs/api_reference.md`
- Imports factory functions in examples
- Shows pipeline creation with factory functions
- Demonstrates advanced constructs with factory functions

### 3. Specialized Documentation

#### `docs/tools.md`
- Shows tool integration with factory functions
- Demonstrates pipeline creation with tools

#### `docs/scoring.md`
- Shows scoring examples with factory functions
- Demonstrates step-level and pipeline-level scoring

#### `docs/extending.md`
- Shows extension examples with factory functions
- Demonstrates custom pipeline creation

#### `docs/troubleshooting.md`
- Shows debugging examples with factory functions
- Demonstrates agent testing with factory functions

### 4. Cookbook and Architecture

#### `docs/cookbook/hybrid_validation.md`
- Shows validation examples with factory functions

#### `docs/The_flujo_way.md`
- Shows architectural examples with factory functions
- Demonstrates quality gates with factory functions

### 5. Agent Infrastructure Guide

#### `docs/agent_infrastructure.md`
- Comprehensive guide to the agent infrastructure
- Detailed factory function documentation
- Prompt management guide
- Usage examples
- Benefits explanation
- Best practices
- Troubleshooting section

### 6. Navigation and Structure

#### `docs/index.md`
- Enhanced feature list including agent infrastructure
- Getting started section with links to key documentation
- Link to agent infrastructure guide

#### `mkdocs.yml`
- Agent infrastructure section in navigation menu

## Usage Examples

### Basic Pipeline Creation

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_review_agent, make_solution_agent, make_validator_agent

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
from flujo.recipes import Default
from flujo.infra.agents import make_review_agent, make_solution_agent, make_validator_agent

# Create a recipe with factory functions
recipe = Default(
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
from flujo.infra.agents import make_review_agent
from flujo.domain.models import Checklist

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

## Testing

All documentation has been verified by:
- Running the full test suite to ensure accuracy
- Checking that all examples use factory functions correctly
- Verifying that imports are correctly specified
- Ensuring navigation works properly

## Files Documented

### Core Documentation
- `docs/usage.md`
- `docs/concepts.md`
- `docs/tutorial.md`
- `docs/api_reference.md`
- `docs/index.md`

### Specialized Documentation
- `docs/tools.md`
- `docs/scoring.md`
- `docs/extending.md`
- `docs/troubleshooting.md`

### Cookbook
- `docs/cookbook/hybrid_validation.md`

### Architecture
- `docs/The_flujo_way.md`

### Agent Infrastructure
- `docs/agent_infrastructure.md`

### Configuration
- `mkdocs.yml`

## Summary

The documentation comprehensively covers the agent infrastructure using factory functions. All examples demonstrate the proper use of factory functions for creating agents, providing users with clear guidance on how to use the system effectively. The `agent_infrastructure.md` guide serves as a complete reference for understanding and using the agent system.
