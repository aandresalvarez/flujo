# Configuration Guide

This guide explains all configuration options available in `flujo`.

## Environment Variables

The orchestrator uses environment variables for configuration. These can be set in your `.env` file or directly in your environment.

### API Keys

```env
# Required for OpenAI models
OPENAI_API_KEY=your_key_here

# Required for Anthropic models
ANTHROPIC_API_KEY=your_key_here

# Required for Google models
GOOGLE_API_KEY=your_key_here

# Optional: For telemetry
LOGFIRE_API_KEY=your_key_here
```

### Core Settings

```env
# Enable/disable reflection (default: true)
REFLECTION_ENABLED=true

# Enable/disable reward model scoring (default: true)
REWARD_ENABLED=true

# Maximum number of iterations (default: 3)
MAX_ITERS=3

# Number of variants to generate (default: 2)
K_VARIANTS=2
```

### Telemetry Settings

```env
# Enable telemetry export (default: false)
TELEMETRY_EXPORT_ENABLED=false

# Enable OTLP export (default: false)
OTLP_EXPORT_ENABLED=false

# OTLP endpoint URL (optional)
OTLP_ENDPOINT=https://your-otlp-endpoint
```

## Python Configuration

You can also configure the orchestrator programmatically:

### Basic Configuration

```python
from flujo.recipes import Default
from flujo import Task

# Create the default recipe with custom settings
orch = Default(
    review_agent,
    solution_agent,
    validator_agent,
    max_iters=5,  # Override MAX_ITERS
    k_variants=3  # Override K_VARIANTS
)
```

### Telemetry Configuration

```python
from flujo import init_telemetry

# Initialize with custom settings
init_telemetry(
    service_name="my-app",
    environment="production",
    version="1.0.0"
)
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
    max_parallel=2,  # Maximum parallel executions
    timeout=60,      # Overall timeout
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