# Troubleshooting Guide

This guide helps you resolve common issues when using `flujo`.

## Installation Issues

### 1. Package Installation Fails

**Symptoms:**
- `pip install` fails with dependency errors
- Version conflicts
- Missing system dependencies

**Solutions:**
1. Ensure Python 3.11+ is installed:
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
   from flujo.recipes import Default

   orchestrator = Default(
       model="openai:gpt-4",
       temperature=0.7
   )
   print(orchestrator.model_config)
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
       Step.review(review_agent)
       >> Step.solution(solution_agent)
       >> Step.validate(validator_agent)
   )
   
   # Print pipeline structure
   print(pipeline.structure())
   ```

3. Test steps individually:
   ```python
   # Test review step
   result = review_agent.run("Test prompt")
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
   orchestrator = Default(
       model="openai:gpt-4",
       max_tokens=1000,  # Limit token usage
       timeout=30,       # Set reasonable timeout
       cache=True        # Enable caching
   )
   ```

### 4. Usage Governor Breach

**Symptoms:**
- Pipeline stops with `UsageLimitExceededError`

**Solutions:**
1. Increase or remove the limits:
   ```python
   runner = Flujo(pipeline, usage_limits=None)
   ```
2. Reduce cost by using cheaper models or fewer iterations.

## Telemetry Issues

### 1. Metrics Collection

**Symptoms:**
- Missing metrics
- Wrong metric values
- Export failures

**Solutions:**
1. Check telemetry configuration:
   ```python
   from flujo import init_telemetry
   
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
   - [Installation Guide](installation.md)
   - [Usage Guide](usage.md)
   - [API Reference](api_reference.md)

2. **Community**
   - [GitHub Issues](https://github.com/aandresalvarez/flujo/issues)
   - [Discussions](https://github.com/aandresalvarez/flujo/discussions)

3. **Development**
   - [Contributing Guide](dev.md)
   - [Development Guide](dev.md)

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

- Read the [Usage Guide](usage.md)
- Check [Advanced Topics](extending.md)
- Explore [Use Cases](use_cases.md) 