# Troubleshooting Guide

This guide helps you resolve common issues when using `flujo`.

## Installation Issues

### 1. Package Installation Fails

**Symptoms:**
- `pip install` fails with dependency errors
- Version conflicts
- Missing system dependencies

**Solutions:**
1. Ensure Python 3.13+ is installed:
   ```bash
   python --version
   ```

2. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

4. Install with verbose output:
   ```bash
   pip install -v flujo
   ```

5. Check system dependencies:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev
   ```

### 2. Development Installation Issues

**Symptoms:**
- `make pip-dev` fails
- Editable install doesn't work
- Missing development dependencies

**Solutions:**
1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Check Makefile:
   ```bash
   make -n pip-dev  # Show commands without executing
   ```

3. Install build tools:
   ```bash
   pip install build wheel
   ```

## Configuration Issues

### 1. API Key Problems

**Symptoms:**
- Authentication errors
- Rate limit errors
- Model not found errors

**Solutions:**
1. Verify API keys in `.env`:
   ```bash
   cat .env | grep API_KEY
   ```

2. Check environment variables:
   ```python
   import os
   print(os.getenv("OPENAI_API_KEY"))
   ```

3. Test API key directly:
   ```python
   from openai import OpenAI
   client = OpenAI()
   client.models.list()
   ```

### 2. Model Configuration

**Symptoms:**
- Model not available
- Wrong model version
- Performance issues

**Solutions:**
1. Check model availability:
   ```python
   from flujo import list_available_models
   print(list_available_models())
   ```

2. Verify model configuration:
   ```python
   from flujo.recipes.factories import make_default_pipeline

   pipeline = make_default_pipeline(
       review_agent=review_agent,
       solution_agent=solution_agent,
       validator_agent=validator_agent,
       model="openai:gpt-4",
       temperature=0.7,
   )
   print(pipeline)
   ```

## Runtime Issues

### 1. Pipeline Errors

**Symptoms:**
- Pipeline fails to start
- Steps fail unexpectedly
- Wrong output format

**Solutions:**
1. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Check step configuration:
   ```python
   from flujo import Step, Flujo

   pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(make_solution_agent())
    >> Step.validate(make_validator_agent())
)

   # Print pipeline structure
   print(pipeline.structure())
   ```

3. Test steps individually:
   ```python
   # Test review step
   result = make_review_agent().run("Test prompt")
   print(result)
   ```

### 2. Tool Errors

**Symptoms:**
- Tool execution fails
- Wrong tool output
- Timeout errors

**Solutions:**
1. Check tool configuration:
   ```python
   from pydantic_ai import Tool

   tool = Tool(my_function)
   print(tool.config)
   ```

2. Test tool directly:
   ```python
   result = tool.run("test input")
   print(result)
   ```

3. Enable tool debugging:
   ```python
   tool = Tool(my_function, debug=True)
   ```

### 3. Performance Issues

**Symptoms:**
- Slow execution
- High memory usage
- Timeout errors

**Solutions:**
1. Profile execution:
   ```python
   from flujo import enable_profiling

   with enable_profiling():
       result = orchestrator.run("prompt")
   ```

2. Check memory usage:
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())
   print(process.memory_info().rss / 1024 / 1024)  # MB
   ```

3. Optimize configuration:
   ```python
   from flujo.recipes.factories import make_default_pipeline

   pipeline = make_default_pipeline(
       review_agent=review_agent,
       solution_agent=solution_agent,
       validator_agent=validator_agent,
       model="openai:gpt-4",
       max_tokens=1000,  # Limit token usage
       timeout=30,       # Set reasonable timeout
       cache=True,       # Enable caching
   )
   ```

### 4. Usage Governor Breach

**Symptoms:**
- Pipeline stops with `UsageLimitExceededError`
- Unexpected cost or token limits being hit
- Inconsistent cost calculations

**Solutions:**

1. **Increase or remove the limits:**
   ```python
   # Remove all limits
   runner = Flujo(pipeline, usage_limits=None)

   # Increase limits
   limits = UsageLimits(
       total_cost_usd_limit=5.0,    # Increase from $1.0 to $5.0
       total_tokens_limit=10000     # Increase from 5000 to 10000
   )
   runner = Flujo(pipeline, usage_limits=limits)
   ```

2. **Check your cost configuration:**
   ```toml
   # flujo.toml - Verify pricing is correct
   [cost.providers.openai.gpt-4o]
   prompt_tokens_per_1k = 0.005
   completion_tokens_per_1k = 0.015
   ```

3. **Debug cost calculations:**
   ```python
   import logging
   logging.getLogger("flujo.cost").setLevel(logging.DEBUG)

   # Run pipeline to see detailed cost calculation logs
   result = runner.run("Your prompt")
   ```

4. **Reduce costs by using cheaper models:**
   ```python
   # Use cheaper models for cost-sensitive operations
   cheap_agent = make_agent_async("openai:gpt-3.5-turbo", "Simple task", str)
   expensive_agent = make_agent_async("openai:gpt-4o", "Complex task", str)

   # Use cheap agent for simple tasks
   pipeline = Step.solution(cheap_agent) >> Step.validate(expensive_agent)
   ```

5. **Set step-level limits for fine control:**
   ```python
   # Set different limits for different steps
   solution_limits = UsageLimits(total_cost_usd_limit=0.10)
   validation_limits = UsageLimits(total_cost_usd_limit=0.05)

   pipeline = (
       Step.solution(agent, usage_limits=solution_limits)
       >> Step.validate(validator, usage_limits=validation_limits)
   )
   ```

6. **Monitor costs in real-time:**
   ```python
   def log_step_costs(step_result):
       print(f"{step_result.name}: ${step_result.cost_usd:.4f} ({step_result.token_counts} tokens)")

   # Add logging to track costs as they occur
   for step_result in result.step_history:
       log_step_costs(step_result)
   ```

## Telemetry Issues

### 1. Metrics Collection

**Symptoms:**
- Missing metrics
- Wrong metric values
- Export failures

**Solutions:**
1. Check telemetry configuration:
   ```python
   from flujo.infra import init_telemetry

   init_telemetry(
       enable_export=True,
       export_endpoint="http://localhost:4317"
   )
   ```

2. Verify metrics:
   ```python
   from flujo import get_metrics

   metrics = get_metrics()
   print(metrics)
   ```

### 2. Tracing Issues

**Symptoms:**
- Missing traces
- Incomplete traces
- Export errors

**Solutions:**
1. Enable tracing:
   ```python
   from flujo import enable_tracing

   with enable_tracing():
       result = orchestrator.run("prompt")
   ```

2. Check trace export:
   ```python
   from flujo import get_traces

   traces = get_traces()
   print(traces)
   ```

## Common Error Messages

### 1. Authentication Errors

```
AuthenticationError: Invalid API key
```

**Solutions:**
1. Check API key format
2. Verify key is active
3. Ensure key has correct permissions

### 2. Model Errors

```
ModelError: Model not found
```

**Solutions:**
1. Verify model name
2. Check model availability
3. Update to latest version

### 3. Validation Errors

```
ValidationError: Invalid input
```

**Solutions:**
1. Check input format
2. Verify required fields
3. Review validation rules

### 4. Timeout Errors

```
TimeoutError: Operation timed out
```

**Solutions:**
1. Increase timeout
2. Check network
3. Optimize request

### 5. `TypeError: Step '...' returned a Mock object.`

This error almost always occurs during unit testing when a mock agent is not configured to return a concrete value.

**Solution:** Set a return value on your mock agent:

```python
from unittest.mock import AsyncMock

agent = AsyncMock()
agent.run.return_value = "expected"
```
See the [Testing Guide](testing_guide.md) for more examples.

## Getting Help

### 1. Debugging Tools

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable profiling
from flujo import enable_profiling
with enable_profiling():
    result = orchestrator.run("prompt")

# Get detailed error info
from flujo import get_error_details
details = get_error_details(error)
print(details)
```

### 2. Support Resources

1. **Documentation**
   - [Installation Guide](../getting-started/installation.md)
   - [Usage Guide](../user_guide/usage.md)
   - [API Reference](../api/index.md)
   - [SQLite Backend Guide](../guides/sqlite_backend_guide.md) - For persistence and observability issues

2. **Community**
   - [GitHub Issues](https://github.com/aandresalvarez/flujo/issues)
   - [Discussions](https://github.com/aandresalvarez/flujo/discussions)

3. **Development**
   - [Contributing Guide](../development/dev.md)
   - [Development Guide](../development/dev.md)

### 3. Reporting Issues

When reporting an issue, include:

1. **Environment**
   ```bash
   python --version
   pip freeze
   ```

2. **Error Details**
   ```python
   import traceback
   print(traceback.format_exc())
   ```

3. **Reproduction Steps**
   - Minimal code example
   - Expected vs actual behavior
   - Relevant logs

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md)
- Check [Advanced Topics](extending.md)
- Explore [Use Cases](../user_guide/use_cases.md)
- Review the [Adapter Step recipe](../cookbook/adapter_step.md) for data-shaping tips
