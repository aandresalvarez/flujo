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

*   `state_backend_span_export_enabled`: `Optional[bool] = None`
    *   **Description**: Export OpenTelemetry spans to the configured state backend. `None` enables auto mode (SQLite only).

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

### Safety, Sandboxing, and Memory

These settings control governance (pre-execution allow/deny), sandboxed code execution, and optional vector memory indexing.

#### Governance (allow/deny gating)

- `governance_policy_module`: `Optional[str]`
  - **Environment Variables**: `FLUJO_GOVERNANCE_POLICY_MODULE`, `flujo_governance_policy_module`
  - **Description**: Module path in the form `pkg.mod:Class` to load a custom governance policy.
  - **Behavior**: Policies run before agent execution. The default policy is allow-all.

Example:
```toml
[settings]
governance_policy_module = "my_project.policies:MyPolicy"
```

Built-in environment toggles:

- `FLUJO_GOVERNANCE_MODE=allow_all|deny_all`
- `FLUJO_GOVERNANCE_PII_SCRUB=1`
- `FLUJO_GOVERNANCE_PII_STRONG=1` (requires `pip install "flujo[pii]"`, falls back safely if not installed)
- `FLUJO_GOVERNANCE_TOOL_ALLOWLIST=tool_a,tool_b` (enforced at tool-call-time; also used for pre-execution checks)

#### Sandbox (built-in `flujo.builtins.code_interpreter`)

- `sandbox.mode`: `"null" | "remote" | "docker"`
  - **Environment Variables**: `FLUJO_SANDBOX_MODE`
- `sandbox.api_url`: `Optional[str]` (remote mode)
  - **Environment Variables**: `FLUJO_SANDBOX_API_URL`
- `sandbox.api_key`: `Optional[str]`
  - **Environment Variables**: `FLUJO_SANDBOX_API_KEY`
- `sandbox.timeout_seconds`: `int`
  - **Environment Variables**: `FLUJO_SANDBOX_TIMEOUT_S`
- `sandbox.verify_ssl`: `bool` (remote mode)
  - **Environment Variables**: `FLUJO_SANDBOX_VERIFY_SSL`
- `sandbox.docker_image`: `str` (docker mode)
  - **Environment Variables**: `FLUJO_SANDBOX_DOCKER_IMAGE`
- `sandbox.docker_pull`: `bool` (docker mode)
  - **Environment Variables**: `FLUJO_SANDBOX_DOCKER_PULL`
- `sandbox.docker_mem_limit`: `Optional[str]` (docker mode)
  - **Environment Variables**: `FLUJO_SANDBOX_DOCKER_MEM_LIMIT`
- `sandbox.docker_pids_limit`: `Optional[int]` (docker mode)
  - **Environment Variables**: `FLUJO_SANDBOX_DOCKER_PIDS_LIMIT`
- `sandbox.docker_network_mode`: `Optional[str]` (docker mode)
  - **Environment Variables**: `FLUJO_SANDBOX_DOCKER_NETWORK_MODE`

Example:
```toml
[settings.sandbox]
mode = "remote"
api_url = "https://your-sandbox.example"
timeout_seconds = 60
```

Notes:
- Docker mode is intended for local development and currently focuses on `language="python"`.
- The built-in tool is registered as `flujo.builtins.code_interpreter` and returns structured `stdout`/`stderr`/`exit_code`.

#### Memory (RAG indexing + retrieval)

- `memory_indexing_enabled`: `bool`
  - **Environment Variables**: `FLUJO_MEMORY_INDEXING_ENABLED`, `flujo_memory_indexing_enabled`
- `memory_embedding_model`: `Optional[str]` (e.g., `openai:text-embedding-3-small`)
  - **Environment Variables**: `FLUJO_MEMORY_EMBEDDING_MODEL`, `flujo_memory_embedding_model`

Example:
```toml
[settings]
memory_indexing_enabled = true
memory_embedding_model = "openai:text-embedding-3-small"
```

When enabled, successful step outputs are indexed asynchronously into a vector store chosen from your configured `state_uri`:
- SQLite state → SQLite-backed vector store (cosine search)
- Postgres state → pgvector-backed store

#### Shadow Evaluations (experimental)

Shadow evaluations (LLM-as-judge scoring) are implemented in the codebase (including DB persistence and `flujo lens evals`) and default to disabled. They can be enabled via environment variables:

- `FLUJO_SHADOW_EVAL_ENABLED=1`
- `FLUJO_SHADOW_EVAL_SAMPLE_RATE=0.1` (fraction of runs to sample; cached per `run_id`)
- `FLUJO_SHADOW_EVAL_TIMEOUT_S=30`
- `FLUJO_SHADOW_EVAL_JUDGE_MODEL=openai:gpt-4o-mini`
- `FLUJO_SHADOW_EVAL_SINK=telemetry|database`
- `FLUJO_SHADOW_EVAL_EVALUATE_ON_FAILURE=1` (optional; only score failed steps)
- `FLUJO_SHADOW_EVAL_RUN_LEVEL=1` (optional; also score the overall run as pseudo-step `__run__`)

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
from flujo.infra.agents import make_agent_async

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

## Cost Tracking Configuration

Flujo provides integrated cost and token usage tracking for LLM steps. This feature allows you to monitor and control spending across your AI pipelines.

### Configuring Provider Pricing

To enable cost tracking, you need to configure pricing for your LLM providers in your `flujo.toml` file. The cost tracking system automatically calculates costs based on token usage and configured pricing.

#### Basic Cost Configuration

Add a `[cost]` section to your `flujo.toml`:

```toml
# flujo.toml
[cost]
[cost.providers]
[cost.providers.openai]
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

[cost.providers.anthropic]
# Anthropic Models (Pricing: https://www.anthropic.com/pricing)
[cost.providers.anthropic.claude-3-sonnet]
prompt_tokens_per_1k = 0.003
completion_tokens_per_1k = 0.015

[cost.providers.anthropic.claude-3-haiku]
prompt_tokens_per_1k = 0.00025
completion_tokens_per_1k = 0.00125

[cost.providers.google]
# Google Models (Pricing: https://ai.google.dev/pricing)
[cost.providers.google.gemini-1.5-pro]
prompt_tokens_per_1k = 0.0035
completion_tokens_per_1k = 0.0105

[cost.providers.google.gemini-1.5-flash]
prompt_tokens_per_1k = 0.000075
completion_tokens_per_1k = 0.0003
```

#### Pricing Structure

The pricing configuration follows this hierarchy:
- `[cost.providers.{provider_name}]` - Provider section (e.g., `openai`, `anthropic`, `google`)
- `[cost.providers.{provider_name}.{model_name}]` - Model-specific pricing
- `prompt_tokens_per_1k` - Cost per 1,000 prompt tokens (input)
- `completion_tokens_per_1k` - Cost per 1,000 completion tokens (output)

**Note**: The configuration uses `[cost.providers]` rather than `[providers]` at the top level to group all cost-related configuration under one section. This provides better organization and avoids potential naming conflicts.

#### Supported Providers

Flujo supports cost tracking for these providers:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.
- **Anthropic**: `claude-3-sonnet`, `claude-3-haiku`, etc.
- **Google**: `gemini-1.5-pro`, `gemini-1.5-flash`, etc.

### Using Cost Tracking in Pipelines

Once configured, cost tracking is automatically enabled for all pipeline steps that use LLM agents.

#### Accessing Cost Information

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

#### Setting Usage Limits

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
    print(f"Pipeline stopped due to usage limits: {e}")
    # Access partial results
    partial_result = e.partial_result
```

#### Step-Level Limits

You can also set limits on individual steps:

```python
from flujo import Step, UsageLimits

# Set limits for a specific step
step_limits = UsageLimits(
    total_cost_usd_limit=0.10,   # Maximum $0.10 for this step
    total_tokens_limit=1000       # Maximum 1,000 tokens for this step
)

pipeline = (
    Step.solution(my_agent, usage_limits=step_limits)
    >> Step.validate(validator_agent)
)
```

### Cost Tracking Features

#### Automatic Token Counting

Flujo automatically extracts token usage from LLM responses:
- **Prompt tokens**: Input tokens sent to the model
- **Completion tokens**: Output tokens generated by the model
- **Total tokens**: Sum of prompt and completion tokens

#### Cost Calculation

Costs are calculated using the formula:
```
cost = (prompt_tokens / 1000) * prompt_tokens_per_1k +
       (completion_tokens / 1000) * completion_tokens_per_1k
```

**Pricing Units**: The configuration uses `_per_1k` (per 1,000 tokens) rather than `_per_million_tokens` to align with common provider pricing pages and provide more intuitive configuration values. For example, GPT-4o costs $0.005 per 1K prompt tokens, which is configured as `prompt_tokens_per_1k = 0.005`.

#### Fallback Pricing

If a model is not explicitly configured in `flujo.toml`, Flujo will check against a list of hardcoded default prices for popular models. A critical warning will be logged if a default is used. If no default exists, the cost will be 0.0.

⚠️ **CRITICAL WARNING**: Hardcoded defaults are for development/testing only and may be outdated. Always configure explicit pricing in `flujo.toml` for production use.

#### Parallel Execution Limits

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

### Best Practices

#### 1. Regular Price Updates

Keep your pricing configuration up to date:
- Monitor provider pricing changes
- Update `flujo.toml` when prices change
- Use provider-specific pricing for accuracy

#### 2. Appropriate Limits

Set reasonable usage limits:
- Start with conservative limits
- Monitor actual usage patterns
- Adjust limits based on your budget

#### 3. Cost Monitoring

Monitor costs in production:
- Log cost information for analysis
- Set up alerts for high-cost runs
- Track cost trends over time

#### 4. Model Selection

Choose cost-effective models:
- Use cheaper models for simple tasks
- Reserve expensive models for complex work
- Consider token efficiency

### Troubleshooting Cost Tracking

#### Common Issues

1. **No cost calculated**: Check that pricing is configured for your model
2. **Incorrect costs**: Verify pricing values in `flujo.toml`
3. **Missing token counts**: Ensure your agent returns usage information

#### Debugging

Enable debug logging to troubleshoot cost tracking:

```python
import logging
logging.getLogger("flujo.cost").setLevel(logging.DEBUG)
```

This will show detailed information about cost calculations and token extraction.

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
from flujo.domain.scoring import weighted_score

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

5. **Cost Management**
   - Configure accurate pricing
   - Set appropriate usage limits
   - Monitor costs regularly

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

4. **Cost Tracking Issues**
   - Verify pricing configuration
   - Check model name matching
   - Ensure usage information is available

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md) for examples
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)

## Robust Path Handling for SQLite State Backends

Flujo supports robust, standards-compliant path handling for all SQLite state backends, ensuring correct behavior in both CLI and programmatic usage.

### How `state_uri` is Resolved
- **Absolute paths** (e.g., `sqlite:////abs/path/to/flujo_ops.db`) are used as-is.
- **Relative paths** (e.g., `sqlite:///./flujo_ops.db` or `sqlite:///flujo_ops.db`) are always resolved relative to the current working directory of the process (not the config file location), following [RFC 3986](https://datatracker.ietf.org/doc/html/rfc3986#section-3).
- The path normalization logic ensures that URIs like `sqlite:///./foo.db` are interpreted as `./foo.db` (relative), while `sqlite:////foo.db` is `/foo.db` (absolute).

### How the CLI and Scripts Find the Config File
- The CLI and all Flujo scripts search for `flujo.toml` in the current directory and parent directories, unless the `FLUJO_CONFIG_PATH` environment variable is set.
- The `state_uri` in the config is then resolved as described above.
- This guarantees that both CLI and scripts use the same database file, regardless of where they are run from, as long as the working directory and config are consistent.

### Best Practices for Multi-Directory Projects and CI
- Always use **relative paths** in `state_uri` for portable, environment-agnostic workflows.
- In CI or multi-directory setups, set the working directory to the location where you want the database file to be created/accessed, or use an absolute path if you need a fixed location.
- Use the `FLUJO_CONFIG_PATH` environment variable to explicitly specify the config file if running from outside the project root.

### Example URIs and Their Effects
- `sqlite:///./flujo_ops.db` → `./flujo_ops.db` (relative to CWD)
- `sqlite:///flujo_ops.db` → `flujo_ops.db` (relative to CWD)
- `sqlite:////tmp/flujo_ops.db` → `/tmp/flujo_ops.db` (absolute)
- `sqlite:///../data/ops.db` → `../data/ops.db` (relative to CWD)

### Design Principles
- **Separation of concerns:** Path normalization and config loading are handled in dedicated, testable functions.
- **No hardcoded paths:** All logic is parameterized and standards-compliant; no magic values are used.
- **Single responsibility:** Each function does one thing—parsing, normalization, or backend instantiation.

This approach guarantees robust, predictable, and portable state management for all Flujo workflows.
