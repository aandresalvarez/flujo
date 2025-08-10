# Cost Tracking Guide

This guide explains how to use Flujo's integrated cost and token usage tracking features to monitor and control spending on LLM operations.

## Quick Start

1. **Configure pricing** in your `flujo.toml`:
   ```toml
   [cost.providers.openai.gpt-4o]
   prompt_tokens_per_1k = 0.005
   completion_tokens_per_1k = 0.015
   ```

2. **Run a pipeline** with automatic cost tracking:
   ```python
   from flujo import Step, Flujo

   pipeline = Step.solution(my_agent)
   runner = Flujo(pipeline)
   result = runner.run("Your prompt")

   # Access cost information
   for step in result.step_history:
       print(f"{step.name}: ${step.cost_usd:.4f} ({step.token_counts} tokens)")
   ```

3. **Set usage limits** to prevent excessive spending:
   ```python
   from flujo import UsageLimits

   limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=5000)
   runner = Flujo(pipeline, usage_limits=limits)
   ```

## Configuration

### Provider Pricing

Configure pricing for your LLM providers in `flujo.toml`:

```toml
[cost]
[cost.providers]

# OpenAI Models (Pricing: https://openai.com/pricing)
[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015

[cost.providers.openai.gpt-4o-mini]
prompt_tokens_per_1k = 0.00015
completion_tokens_per_1k = 0.0006

[cost.providers.openai.gpt-3.5-turbo]
prompt_tokens_per_1k = 0.0005
completion_tokens_per_1k = 0.0015

# Anthropic Models (Pricing: https://www.anthropic.com/pricing)
[cost.providers.anthropic.claude-3-sonnet]
prompt_tokens_per_1k = 0.003
completion_tokens_per_1k = 0.015

[cost.providers.anthropic.claude-3-haiku]
prompt_tokens_per_1k = 0.00025
completion_tokens_per_1k = 0.00125

# Google Models (Pricing: https://ai.google.dev/pricing)
[cost.providers.google.gemini-1.5-pro]
prompt_tokens_per_1k = 0.0035
completion_tokens_per_1k = 0.0105

[cost.providers.google.gemini-1.5-flash]
prompt_tokens_per_1k = 0.000075
completion_tokens_per_1k = 0.0003
```

### Pricing Structure

- **Provider**: `openai`, `anthropic`, `google`, etc.
- **Model**: Specific model name (e.g., `gpt-4o`, `claude-3-sonnet`)
- **Prompt tokens**: Cost per 1,000 input tokens (`prompt_tokens_per_1k`)
- **Completion tokens**: Cost per 1,000 output tokens (`completion_tokens_per_1k`)

### Cost Calculation

Costs are calculated using the formula:
```
cost = (prompt_tokens / 1000) * prompt_tokens_per_1k +
       (completion_tokens / 1000) * completion_tokens_per_1k
```

**Note**: The pricing units use `_per_1k` (per 1,000 tokens) rather than `_per_million_tokens` to align with common provider pricing pages and provide more intuitive configuration values.

## Image Generation Cost Tracking

Flujo supports automatic cost tracking for image generation models like DALL-E 3. The system automatically detects image models and attaches cost calculation post-processors.

### Image Model Configuration

Configure image generation pricing in your `flujo.toml`:

```toml
# OpenAI image generation models
[cost.providers.openai."dall-e-3"]
prompt_tokens_per_1k = 0.0  # No token costs for image generation
completion_tokens_per_1k = 0.0  # No token costs for image generation
price_per_image_standard_1024x1024 = 0.040
price_per_image_hd_1024x1024 = 0.080
price_per_image_standard_1792x1024 = 0.080
price_per_image_hd_1792x1024 = 0.120
price_per_image_standard_1024x1792 = 0.080
price_per_image_hd_1024x1792 = 0.120
```

### Supported Image Models

Flujo automatically detects and configures cost tracking for these image generation models:

- **OpenAI DALL-E**: `dall-e-2`, `dall-e-3`
- **Midjourney**: `midjourney:v6`
- **Stable Diffusion**: `stable-diffusion:xl`
- **Google Imagen**: `imagen-2`

### Image Cost Calculation

Image costs are calculated based on:
- **Quality**: `standard` or `hd`
- **Size**: `1024x1024`, `1792x1024`, `1024x1792`
- **Number of images**: Reported in the usage details

The cost formula is:
```
cost = number_of_images * price_per_image_{quality}_{size}
```

### Using Image Models

Image models work seamlessly with the existing Flujo API:

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async

# Create a DALL-E 3 agent
dalle_agent = make_agent_async(
    model="openai:dall-e-3",
    system_prompt="Generate beautiful images",
    output_type=str,
)

# Create a pipeline
pipeline = Step.solution(dalle_agent)
runner = Flujo(pipeline)

# Run the pipeline
result = runner.run("Generate a landscape")

# Access cost information
for step in result.step_history:
    print(f"{step.name}: ${step.cost_usd:.4f}")
```

### Image Model Features

- **Automatic Detection**: Image models are automatically detected and configured
- **Quality Support**: Different pricing for standard and HD quality
- **Size Support**: Different pricing for various image sizes
- **Usage Limits**: Image costs integrate with existing usage limits
- **Backward Compatibility**: Chat models continue to work normally

## Strict Pricing Mode

For production environments where cost accuracy is critical, Flujo provides a **Strict Pricing Mode** that ensures all cost calculations are based on your explicit configuration.

### Enabling Strict Mode

Add the `strict = true` flag to your `flujo.toml`:

```toml
[cost]
strict = true  # <-- Enable strict pricing mode

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
```

### How Strict Mode Works

When strict mode is enabled:

1. **Explicit Configuration Required**: Every model used in your pipeline must have explicit pricing configured in `flujo.toml`
2. **No Fallback to Hardcoded Defaults**: The system will not use hardcoded default prices, even for common models
3. **Immediate Failure**: If a model is used without explicit pricing, the pipeline will fail immediately with a `PricingNotConfiguredError`

### Example: Strict Mode Success

```toml
# flujo.toml
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015

[cost.providers.openai.gpt-3.5-turbo]
prompt_tokens_per_1k = 0.0005
completion_tokens_per_1k = 0.0015

[cost.providers.openai."dall-e-3"]
prompt_tokens_per_1k = 0.0
completion_tokens_per_1k = 0.0
price_per_image_standard_1024x1024 = 0.040
```

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async

# These agents will work with strict mode
chat_agent = make_agent_async("openai:gpt-4o", "You are helpful", str)
image_agent = make_agent_async("openai:dall-e-3", "Generate images", str)

pipeline = Step.solution(chat_agent) >> Step.validate(image_agent)
runner = Flujo(pipeline)

# This will work because all models have explicit pricing
result = runner.run("Generate a response and an image")
```

### Example: Strict Mode Failure

```toml
# flujo.toml
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015

# Missing pricing for dall-e-3
```

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async
from flujo.exceptions import PricingNotConfiguredError

# This will fail because dall-e-3 has no pricing
image_agent = make_agent_async("openai:dall-e-3", "Generate images", str)
pipeline = Step.solution(image_agent)
runner = Flujo(pipeline)

try:
    result = runner.run("Generate an image")
except PricingNotConfiguredError as e:
    print(f"Pipeline failed: {e}")
    # Output: Pipeline failed: Pricing not configured for provider=openai, model=dall-e-3
```

## Using Cost Tracking in Pipelines

Once configured, cost tracking is automatically enabled for all pipeline steps that use LLM agents.

### Accessing Cost Information

Cost and token information is available in pipeline results:

```python
from flujo import Step, Flujo

# Create a pipeline with cost tracking
pipeline = Step.solution(my_agent) >> Step.validate(validator_agent)
runner = Flujo(pipeline)

# Run the pipeline
result = runner.run("Your prompt")

# Access cost information
for step_result in result.step_history:
    print(f"Step: {step_result.name}")
    print(f"  Cost: ${step_result.cost_usd:.4f}")
    print(f"  Tokens: {step_result.token_counts}")
    print(f"  Success: {step_result.success}")
```

### Setting Usage Limits

You can set cost and token limits to prevent excessive spending:

```python
from flujo import Flujo, Step, UsageLimits

# Define usage limits
limits = UsageLimits(
    total_cost_usd_limit=1.0,    # Maximum $1.00 total cost
    total_tokens_limit=5000       # Maximum 5,000 tokens
)

# Apply limits to pipeline
runner = Flujo(pipeline, usage_limits=limits)

try:
    result = runner.run("Your prompt")
except UsageLimitExceededError as e:
    print(f"Pipeline failed due to usage limits: {e}")
```

### Cost Tracking with Different Model Types

Flujo automatically handles different types of models:

#### Chat Models (Token-based)
```python
# GPT-4, Claude, etc. - cost calculated from tokens
chat_agent = make_agent_async("openai:gpt-4o", "You are helpful", str)
```

#### Image Models (Unit-based)
```python
# DALL-E 3 - cost calculated per image
image_agent = make_agent_async("openai:dall-e-3", "Generate images", str)
```

#### Embedding Models (Token-based)
```python
# Text embeddings - cost calculated from tokens
embedding_agent = make_agent_async("openai:text-embedding-3-large", "Embed text", str)
```

## Advanced Features

### Explicit Cost Reporting

For custom operations, you can implement the `ExplicitCostReporter` protocol:

```python
class CustomImageResult:
    def __init__(self, cost_usd: float, token_counts: int = 0):
        self.cost_usd = cost_usd
        self.token_counts = token_counts

# This object will automatically be recognized for cost tracking
result = CustomImageResult(cost_usd=0.25, token_counts=0)
```

### Cost Tracking in Complex Pipelines

Cost tracking works seamlessly in complex pipeline scenarios:

```python
from flujo import Step, Flujo, UsageLimits

# Create agents for different tasks
chat_agent = make_agent_async("openai:gpt-4o", "You are helpful", str)
image_agent = make_agent_async("openai:dall-e-3", "Generate images", str)
validator_agent = make_agent_async("openai:gpt-4o", "Validate responses", str)

# Create a complex pipeline
pipeline = (
    Step.solution(chat_agent) >>
    Step.validate(validator_agent) >>
    Step.reflect(image_agent)
)

# Set usage limits
limits = UsageLimits(total_cost_usd_limit=2.0, total_tokens_limit=10000)
runner = Flujo(pipeline, usage_limits=limits)

# Run the pipeline
result = runner.run("Complex task with multiple model types")

# Analyze costs
total_cost = sum(step.cost_usd for step in result.step_history)
total_tokens = sum(step.token_counts for step in result.step_history)

print(f"Total cost: ${total_cost:.4f}")
print(f"Total tokens: {total_tokens}")
```

## Troubleshooting

### Common Issues

1. **No Cost Reported**: Check that your model has pricing configured in `flujo.toml`
2. **Incorrect Costs**: Verify pricing values match the current provider rates
3. **Missing Image Costs**: Ensure image models have the correct pricing fields configured
4. **Strict Mode Failures**: Add explicit pricing for all models used in your pipeline

### Debugging Cost Calculation

Enable debug logging to see cost calculation details:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Run your pipeline and check the logs for cost calculation details
```

### Testing Cost Configuration

Use the provided examples to test your cost configuration:

```bash
# Test basic cost tracking
python examples/cost_tracking_demo.py

# Test image cost tracking
python examples/image_cost_tracking_demo.py
```
