# Configuration Guide

This guide explains all configuration options available in `flujo`.

## Settings Overview

`flujo` uses a `Settings` class (powered by Pydantic-settings) to manage its configuration. Settings are primarily loaded from environment variables, with support for `.env` files for local development. This provides a flexible and robust way to configure your `flujo` applications.

### How Settings are Loaded

1.  **Environment Variables**: `flujo` will automatically read environment variables. For example, `OPENAI_API_KEY`.
2.  **.env files**: For local development, you can create a `.env` file in your project root. Variables defined in this file will be loaded and take precedence over system environment variables.

### `Settings` Class Properties

Below is a comprehensive list of all available settings, their types, default values, and a brief description.

#### API Keys

These settings manage API keys for various language model providers. They support `AliasChoices` for backward compatibility with older environment variable names.

*   `openai_api_key`: `Optional[SecretStr]`
    *   **Environment Variables**: `OPENAI_API_KEY`, `ORCH_OPENAI_API_KEY`, `orch_openai_api_key`
    *   **Description**: API key for OpenAI models.

*   `google_api_key`: `Optional[SecretStr]`
    *   **Environment Variables**: `GOOGLE_API_KEY`, `ORCH_GOOGLE_API_KEY`, `orch_google_api_key`
    *   **Description**: API key for Google models (e.g., Gemini).

*   `anthropic_api_key`: `Optional[SecretStr]`
    *   **Environment Variables**: `ANTHROPIC_API_KEY`, `ORCH_ANTHROPIC_API_KEY`, `orch_anthropic_api_key`
    *   **Description**: API key for Anthropic models.

*   `logfire_api_key`: `Optional[SecretStr]`
    *   **Environment Variables**: `LOGFIRE_API_KEY`, `ORCH_LOGFIRE_API_KEY`, `orch_logfire_api_key`
    *   **Description**: API key for Logfire telemetry integration.

*   `provider_api_keys`: `Dict[str, SecretStr]`
    *   **Description**: Dynamically loaded dictionary for any other `_API_KEY` environment variables not explicitly listed above (e.g., `MYPROVIDER_API_KEY`).

#### Feature Toggles

These boolean settings enable or disable specific `flujo` features.

*   `reflection_enabled`: `bool = True`
    *   **Description**: Enables or disables the reflection agent in multi-agent pipelines.

*   `reward_enabled`: `bool = True`
    *   **Description**: Enables or disables reward model scoring.

*   `telemetry_export_enabled`: `bool = False`
    *   **Description**: Enables or disables the export of telemetry data.

*   `otlp_export_enabled`: `bool = False`
    *   **Description**: Enables or disables OpenTelemetry Protocol (OTLP) export for distributed tracing.

#### Default Models

These settings define the default language models used by various agents within `flujo`.

*   `default_solution_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the Solution agent.

*   `default_review_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the Review agent.

*   `default_validator_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the Validator agent.

*   `default_reflection_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the Reflection agent.

*   `default_self_improvement_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the `SelfImprovementAgent`.

*   `default_repair_model`: `str = "openai:gpt-4o"`
    *   **Description**: Default model for the internal JSON repair agent.

#### Orchestrator Tuning

These settings control the behavior and performance of the `flujo` orchestrator.

*   `max_iters`: `int = 5`
    *   **Description**: Maximum number of iterations for multi-agent loops.

*   `k_variants`: `int = 3`
    *   **Description**: Number of solution variants to generate per iteration.

*   `reflection_limit`: `int = 3`
    *   **Description**: Maximum number of reflection steps allowed.

*   `scorer`: `Literal["ratio", "weighted", "reward"] = "ratio"`
    *   **Description**: The default scoring strategy to use.

*   `t_schedule`: `list[float] = [1.0, 0.8, 0.5, 0.2]`
    *   **Description**: A list of floating-point numbers representing the temperature for each iteration round. The last value is used for any rounds beyond the schedule's length. This setting is validated to ensure it's not empty.

*   `otlp_endpoint`: `Optional[str] = None`
    *   **Description**: The endpoint URL for OpenTelemetry Protocol (OTLP) export.

*   `agent_timeout`: `int = 60`
    *   **Description**: Timeout in seconds for individual agent calls.

### Python Configuration

You can also configure the orchestrator programmatically by importing the `settings` object and modifying its attributes directly. This is useful for dynamic configuration or testing scenarios.

```python
from flujo.infra.settings import settings

# Override a setting programmatically
settings.max_iters = 10
settings.reflection_enabled = False

# Access a setting
print(f"Current solution model: {settings.default_solution_model}")
```

## Model Configuration

### Model Selection

```python
from flujo import make_agent_async

# Use different models for different agents
review_agent = make_agent_async(
    "openai:gpt-4",  # More capable model for review
    "You are a critical reviewer...",
    Checklist
)

solution_agent = make_agent_async(
    "openai:gpt-3.5-turbo",  # Faster model for generation
    "You are a creative writer...",
    str
)
```

### Model Parameters

```python
# Configure model parameters
agent = make_agent_async(
    "openai:gpt-4",
    "You are a helpful assistant...",
    str,
    temperature=0.7,  # Control randomness
    max_tokens=1000,  # Limit response length
    top_p=0.9,       # Nucleus sampling
    frequency_penalty=0.5,  # Reduce repetition
    presence_penalty=0.5    # Encourage diversity
)
```

## Pipeline Configuration

### Step Configuration

```python
from flujo import Step, Flujo

# Configure individual steps
pipeline = (
    Step.review(review_agent, timeout=30)  # 30-second timeout
    >> Step.solution(
        solution_agent,
        retries=3,            # Number of retries
        temperature=0.7,      # Control randomness
    )
    >> Step.validate(validator_agent)
)
```

### Runner Configuration

```python
# Configure the pipeline runner
runner = Flujo(
    pipeline,
    retry_on_error=True
)
```

## Scoring Configuration

### Custom Scoring

```python
from flujo import weighted_score

# Define custom weights
weights = {
    "correctness": 0.4,
    "readability": 0.3,
    "efficiency": 0.2,
    "documentation": 0.1
}

# Use in pipeline
pipeline = (
    Step.review(review_agent)
    >> Step.solution(solution_agent)
    >> Step.validate(
        validator_agent,
        scorer=lambda c: weighted_score(c, weights)
    )
)
```

## Tool Configuration

### Tool Settings

```python
from pydantic_ai import Tool

def my_tool(param: str) -> str:
    """Tool description."""
    return f"Processed: {param}"

# Configure tool
tool = Tool(
    my_tool,
    timeout=10,  # Tool timeout
    retries=2,   # Number of retries
    backoff_factor=1.5,  # Backoff between retries
)
```

## Best Practices

1. **Environment Variables**
   - Use `.env` for development
   - Use secure environment variables in production
   - Never commit API keys to version control

2. **Model Selection**
   - Choose models based on task requirements
   - Consider cost and performance trade-offs
   - Use appropriate model parameters

3. **Pipeline Design**
   - Set appropriate timeouts
   - Configure retries for reliability
   - Use parallel execution when possible

4. **Telemetry**
   - Enable in production
   - Configure appropriate sampling
   - Use secure endpoints

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Verify keys are set correctly
   - Check key permissions
   - Ensure keys are valid

2. **Timeout Issues**
   - Increase timeouts for complex tasks
   - Check network latency
   - Monitor model response times

3. **Memory Issues**
   - Reduce batch sizes
   - Use appropriate model sizes
   - Monitor memory usage

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](usage.md) for examples
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](use_cases.md)
