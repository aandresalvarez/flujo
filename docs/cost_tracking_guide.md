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
```

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async

# These agents will work with strict mode
agent1 = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)
agent2 = make_agent_async("openai:gpt-3.5-turbo", "You are a validator.", str)

pipeline = Step.solution(agent1) >> Step.validate(agent2)
runner = Flujo(pipeline)

# This will succeed because both models are explicitly configured
result = runner.run("Your prompt")
```

### Example: Strict Mode Failure

```toml
# flujo.toml
[cost]
strict = true

[cost.providers.openai.gpt-4o]
prompt_tokens_per_1k = 0.005
completion_tokens_per_1k = 0.015
# Missing gpt-3.5-turbo configuration
```

```python
from flujo import Step, Flujo
from flujo.exceptions import PricingNotConfiguredError
from flujo.infra.agents import make_agent_async

agent1 = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)
agent2 = make_agent_async("openai:gpt-3.5-turbo", "You are a validator.", str)  # Not configured!

pipeline = Step.solution(agent1) >> Step.validate(agent2)
runner = Flujo(pipeline)

# This will raise PricingNotConfiguredError
try:
    result = runner.run("Your prompt")
except PricingNotConfiguredError as e:
    print(f"Pipeline failed: {e}")
    # Error: "Strict pricing is enabled, but no configuration was found for provider='openai', model='gpt-3.5-turbo' in flujo.toml."
```

### When to Use Strict Mode

**Use strict mode when:**
- You're in a production environment
- Cost accuracy is critical for billing
- You want to prevent silent inaccuracies from hardcoded defaults
- You want to ensure all models are explicitly configured

**Don't use strict mode when:**
- You're in development/testing
- You want to use hardcoded defaults for convenience
- You're experimenting with new models

### Default Behavior (Strict Mode Off)

When `strict = false` (default) or when the flag is not specified:

1. **User Configuration First**: Uses your explicit pricing from `flujo.toml`
2. **Hardcoded Defaults**: Falls back to hardcoded defaults for common models
3. **Warning Logged**: Logs a critical warning when using hardcoded defaults
4. **Graceful Degradation**: Returns `0.0` cost if no pricing is available

## Robust Error Handling and Best Practices

### Provider Inference

Flujo automatically infers the provider from model names for common patterns:

- **OpenAI**: Models starting with `gpt-`, `dall-e`, or `text-` (excluding `text-bison`)
- **Anthropic**: Models starting with `claude-`, `haiku`, `sonnet`, or `opus`
- **Google**: Models starting with `gemini-`, `text-bison`, or `chat-bison`

**Important**: For ambiguous models (e.g., `llama-2-7b`, `mistral-7b`), Flujo will **not** attempt to infer the provider and will return `0.0` cost for safety. Use explicit provider format:

```python
# ✅ Recommended - explicit provider
agent = make_agent_async("groq:llama-2-7b", system_prompt, output_type)

# ❌ Avoid - ambiguous model
agent = make_agent_async("llama-2-7b", system_prompt, output_type)  # Cost will be 0.0
```

### Model ID Requirements

For accurate cost tracking, ensure your agents have a `model_id` attribute:

```python
# ✅ Good - explicit model_id
class MyAgent:
    def __init__(self):
        self.model_id = "openai:gpt-4o"

    async def run(self, data):
        # ... implementation

# ✅ Good - model attribute
class MyAgent:
    def __init__(self):
        self.model = "anthropic:claude-3-sonnet"

    async def run(self, data):
        # ... implementation

# ❌ Avoid - no model identification
class MyAgent:
    async def run(self, data):
        # ... implementation (cost will be 0.0)
```

### Error Handling

Flujo implements robust error handling to prevent pipeline failures:

1. **Unknown Providers/Models**: Returns `0.0` cost with warning logs
2. **Missing Model IDs**: Returns `0.0` cost with detailed error messages
3. **Configuration Errors**: Falls back to `0.0` cost rather than crashing

### Hardcoded Default Pricing

⚠️ **CRITICAL WARNING**: Flujo includes hardcoded default pricing for development/testing only. These prices are **not suitable for production billing** and may be outdated.

**For production use, always configure explicit pricing in `flujo.toml`.**

The system will log critical warnings when using hardcoded defaults:
```
🚨 CRITICAL WARNING: Using stale, hardcoded default price for 'openai:gpt-4o'.
This cost calculation is likely INACCURATE and may be outdated.
These prices are for development/testing only and should NOT be used for production billing.
Configure explicit pricing in flujo.toml for production use.
```

## Usage Examples

### Basic Cost Tracking

```python
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async

# Create agents
solution_agent = make_agent_async("openai:gpt-4o", "You are a helpful assistant.", str)
validator_agent = make_agent_async("openai:gpt-4o", "You are a validator.", str)

# Create pipeline
pipeline = Step.solution(solution_agent) >> Step.validate(validator_agent)
runner = Flujo(pipeline)

# Run pipeline
result = runner.run("Write a short story about a robot.")

# Access cost information
total_cost = 0
total_tokens = 0

for step_result in result.step_history:
    cost = step_result.cost_usd
    tokens = step_result.token_counts
    total_cost += cost
    total_tokens += tokens

    print(f"{step_result.name}:")
    print(f"  Cost: ${cost:.4f}")
    print(f"  Tokens: {tokens}")
    print(f"  Success: {step_result.success}")

print(f"\nTotal cost: ${total_cost:.4f}")
print(f"Total tokens: {total_tokens}")
```

### Setting Usage Limits

Prevent excessive spending by setting cost and token limits:

```python
from flujo import Flujo, Step, UsageLimits

# Define limits
limits = UsageLimits(
    total_cost_usd_limit=0.50,  # Maximum $0.50 total cost
    total_tokens_limit=2000     # Maximum 2,000 tokens
)

# Apply limits to pipeline
runner = Flujo(pipeline, usage_limits=limits)

try:
    result = runner.run("Write a comprehensive analysis.")
    print("Pipeline completed successfully!")
except UsageLimitExceededError as e:
    print(f"Pipeline stopped due to usage limits: {e}")
    # Access partial results
    partial_result = e.partial_result
    print(f"Completed {len(partial_result.step_history)} steps before stopping")
```

### Step-Level Limits

Set limits on individual steps for fine-grained control:

```python
from flujo import Step, Flujo, UsageLimits

# Set limits for specific steps
step_limits = UsageLimits(
    total_cost_usd_limit=0.10,   # Maximum $0.10 for this step
    total_tokens_limit=1000       # Maximum 1,000 tokens for this step
)

pipeline = (
    Step.solution(my_agent, usage_limits=step_limits)
    >> Step.validate(validator_agent)
)
```

### Parallel Execution Limits

When using parallel steps, Flujo can proactively cancel sibling branches when limits are exceeded:

```python
from flujo import Step, Pipeline, UsageLimits

# Create parallel branches
fast_expensive = Pipeline.from_step(Step("expensive", costly_agent))
slow_cheap = Pipeline.from_step(Step("cheap", cheap_agent))

parallel = Step.parallel_branch(fast_expensive, slow_cheap)

# If fast_expensive breaches the limit, slow_cheap will be cancelled immediately
limits = UsageLimits(total_cost_usd_limit=0.10)
runner = Flujo(parallel, usage_limits=limits)
```

### Monitoring and Logging

```python
import logging
from flujo import Flujo, Step

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_costs(result):
    """Log cost information for monitoring."""
    total_cost = sum(step.cost_usd for step in result.step_history)
    total_tokens = sum(step.token_counts for step in result.step_history)

    logger.info(f"Pipeline completed - Cost: ${total_cost:.4f}, Tokens: {total_tokens}")

    # Log per-step details
    for step in result.step_history:
        logger.info(f"  {step.name}: ${step.cost_usd:.4f} ({step.token_counts} tokens)")

# Run pipeline with monitoring
pipeline = Step.solution(my_agent) >> Step.validate(validator_agent)
runner = Flujo(pipeline)

result = runner.run("Your prompt")
log_costs(result)
```

### Cost-Efficient Pipeline Design

```python
from flujo import Step, Flujo, UsageLimits

# Use cheaper models for simple tasks
simple_agent = make_agent_async("openai:gpt-3.5-turbo", "Simple task agent.", str)
complex_agent = make_agent_async("openai:gpt-4o", "Complex task agent.", str)

# Design pipeline with cost considerations
pipeline = (
    Step.solution(simple_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.05))
    >> Step.validate(complex_agent, usage_limits=UsageLimits(total_cost_usd_limit=0.15))
)

# Set overall pipeline limits
runner = Flujo(pipeline, usage_limits=UsageLimits(total_cost_usd_limit=0.25))
```

## Troubleshooting

### Common Issues

1. **No cost calculated**
   - Check that pricing is configured for your model in `flujo.toml`
   - Verify the model name matches exactly
   - Ensure your agent returns usage information

2. **Incorrect costs**
   - Verify pricing values in `flujo.toml`
   - Check that token counts are being extracted correctly
   - Enable debug logging to see calculation details

3. **Missing token counts**
   - Ensure your agent returns usage information via `AgentRunResult.usage()`
   - Check that the agent is properly configured
   - Verify the model supports token counting

4. **PricingNotConfiguredError in strict mode**
   - Add explicit pricing for the model in `flujo.toml`
   - Check that the provider and model names match exactly
   - Consider disabling strict mode for development/testing

### Debugging

Enable debug logging to troubleshoot cost tracking:

```python
import logging
logging.getLogger("flujo.cost").setLevel(logging.DEBUG)

# Run pipeline to see detailed cost calculation logs
result = runner.run("Your prompt")
```

### Verification

Check if your agent returns usage information:

```python
# Test agent usage information
result = my_agent.run("test")
if hasattr(result, 'usage'):
    usage = result.usage()
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
else:
    print("Agent does not return usage information")
```

## Best Practices

### 1. Regular Price Updates

- Monitor provider pricing changes
- Update `flujo.toml` when prices change
- Use provider-specific pricing for accuracy

### 2. Appropriate Limits

- Start with conservative limits
- Monitor actual usage patterns
- Adjust limits based on your budget

### 3. Cost Monitoring

- Log cost information for analysis
- Set up alerts for high-cost runs
- Track cost trends over time

### 4. Model Selection

- Use cheaper models for simple tasks
- Reserve expensive models for complex work
- Consider token efficiency

### 5. Pipeline Design

- Design pipelines with cost efficiency in mind
- Use step-level limits for fine control
- Consider parallel execution for cost optimization

### 6. Production Readiness

- Use strict pricing mode in production
- Configure explicit pricing for all models
- Monitor cost accuracy regularly

## Advanced Features

### Custom Cost Calculators

You can implement custom cost calculation logic:

```python
from flujo.cost import CostCalculator

class CustomCostCalculator(CostCalculator):
    def calculate(self, model_name, prompt_tokens, completion_tokens, provider=None):
        # Implement custom cost calculation logic
        # For example, apply discounts for high-volume usage
        base_cost = super().calculate(model_name, prompt_tokens, completion_tokens, provider)

        if prompt_tokens + completion_tokens > 10000:
            return base_cost * 0.9  # 10% discount for high volume

        return base_cost
```

### Cost Tracking with Custom Models

For custom or proprietary models, you can configure pricing:

```toml
# flujo.toml
[cost.providers.custom.my-model]
prompt_tokens_per_1k = 0.001
completion_tokens_per_1k = 0.002
```

```python
# Use with explicit provider
agent = make_agent_async("custom:my-model", system_prompt, output_type)
```

This comprehensive guide covers all aspects of Flujo's cost tracking system, from basic usage to advanced features like strict pricing mode. The system is designed to be both powerful and safe, providing accurate cost tracking while preventing pipeline failures due to missing pricing information.
