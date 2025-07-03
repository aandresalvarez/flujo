# Command Line Interface Reference

Flujo provides a powerful command-line interface for common operations, testing, and development workflows.

## Installation

The CLI is automatically installed with Flujo:

```bash
pip install flujo
```

## Basic Commands

### `flujo solve` - Solve a Task

The primary command for running AI workflows from the command line.

```bash
flujo solve "Your prompt here"
```

#### Options

- `--recipe, -r`: Choose the recipe to use (default: "default")
  ```bash
  flujo solve "Write a Python function" --recipe agentic_loop
  ```

- `--max-iters, -m`: Maximum iterations for agentic loops (default: 10)
  ```bash
  flujo solve "Complex task" --max-iters 20
  ```

- `--scorer, -s`: Scoring method to use
  ```bash
  flujo solve "Task" --scorer weighted
  ```

- `--output, -o`: Output format (json, text, yaml)
  ```bash
  flujo solve "Task" --output json
  ```

- `--verbose, -v`: Enable verbose output
  ```bash
  flujo solve "Task" --verbose
  ```

#### Examples

```bash
# Basic usage
flujo solve "Write a Python function to calculate Fibonacci numbers"

# With custom recipe
flujo solve "Research quantum computing" --recipe agentic_loop

# JSON output for programmatic use
flujo solve "Generate a poem" --output json

# Verbose output for debugging
flujo solve "Complex task" --verbose --max-iters 15
```

### `flujo bench` - Benchmark Workflows

Run performance benchmarks and quality evaluations.

```bash
flujo bench "Your benchmark prompt"
```

#### Options

- `--rounds, -r`: Number of benchmark rounds (default: 3)
  ```bash
  flujo bench "Task" --rounds 10
  ```

- `--recipe, -R`: Recipe to benchmark
  ```bash
  flujo bench "Task" --recipe agentic_loop
  ```

- `--output, -o`: Output format for results
  ```bash
  flujo bench "Task" --output csv
  ```

- `--compare`: Compare multiple recipes
  ```bash
  flujo bench "Task" --compare default,agentic_loop
  ```

#### Examples

```bash
# Basic benchmark
flujo bench "Write a sorting algorithm"

# Multiple rounds
flujo bench "Complex task" --rounds 5

# Compare recipes
flujo bench "Task" --compare default,agentic_loop --rounds 3

# Save results
flujo bench "Task" --output csv > results.csv
```

### `flujo show-config` - Display Configuration

Show current configuration settings and environment.

```bash
flujo show-config
```

#### Options

- `--format, -f`: Output format (table, json, yaml)
  ```bash
  flujo show-config --format json
  ```

- `--include-secrets`: Include API keys (use with caution)
  ```bash
  flujo show-config --include-secrets
  ```

#### Examples

```bash
# Basic config display
flujo show-config

# JSON format for scripting
flujo show-config --format json

# Check API key configuration
flujo show-config --include-secrets
```

### `flujo explain` - Explain Pipeline Behavior

Get explanations of how pipelines work and their components.

```bash
flujo explain [component]
```

#### Components

- `pipeline`: Explain pipeline DSL concepts
- `agents`: Explain agent types and roles
- `scoring`: Explain scoring methods
- `tools`: Explain tool integration
- `recipes`: Explain built-in recipes

#### Examples

```bash
# General explanation
flujo explain

# Specific component
flujo explain pipeline

# Multiple components
flujo explain agents scoring
```

### `flujo improve` - Generate Improvements

Analyze code or data and suggest improvements.

```bash
flujo improve [file_or_directory]
```

#### Options

- `--type, -t`: Type of improvement (code, docs, tests)
  ```bash
  flujo improve myfile.py --type code
  ```

- `--focus, -f`: Focus area for improvements
  ```bash
  flujo improve . --focus performance
  ```

- `--output, -o`: Output format for suggestions
  ```bash
  flujo improve file.py --output markdown
  ```

#### Examples

```bash
# Improve a single file
flujo improve my_script.py

# Improve entire project
flujo improve . --type code

# Focus on specific area
flujo improve . --focus documentation
```

## Development Commands

### `flujo test` - Run Tests

Execute test suites and validation checks.

```bash
flujo test [test_path]
```

#### Options

- `--unit`: Run only unit tests
- `--integration`: Run only integration tests
- `--e2e`: Run end-to-end tests
- `--coverage`: Generate coverage report
- `--verbose, -v`: Verbose output

#### Examples

```bash
# Run all tests
flujo test

# Run specific test file
flujo test tests/test_pipeline.py

# Run with coverage
flujo test --coverage

# Run integration tests only
flujo test --integration
```

### `flujo validate` - Validate Configuration

Validate configuration files and settings.

```bash
flujo validate [config_file]
```

#### Options

- `--strict`: Strict validation mode
- `--fix`: Auto-fix issues where possible
- `--output, -o`: Output format for issues

#### Examples

```bash
# Validate current config
flujo validate

# Validate specific file
flujo validate my_config.yaml

# Auto-fix issues
flujo validate --fix
```

### `flujo generate` - Generate Code

Generate code templates and boilerplate.

```bash
flujo generate [template] [name]
```

#### Templates

- `pipeline`: Generate a new pipeline
- `agent`: Generate a custom agent
- `tool`: Generate a custom tool
- `validator`: Generate a custom validator
- `recipe`: Generate a custom recipe

#### Examples

```bash
# Generate a new pipeline
flujo generate pipeline my_pipeline

# Generate a custom agent
flujo generate agent my_agent

# Generate a tool
flujo generate tool my_tool
```

## Utility Commands

### `flujo version` - Show Version

Display Flujo version and dependency information.

```bash
flujo version
```

### `flujo help` - Show Help

Display help information for commands.

```bash
flujo help [command]
```

#### Examples

```bash
# General help
flujo help

# Command-specific help
flujo help solve

# Recipe help
flujo help recipes
```

### `flujo completion` - Shell Completion

Generate shell completion scripts.

```bash
flujo completion [shell]
```

#### Supported Shells

- `bash`: Bash completion script
- `zsh`: Zsh completion script
- `fish`: Fish completion script

#### Examples

```bash
# Generate bash completion
flujo completion bash

# Generate zsh completion
flujo completion zsh

# Install completion (bash)
flujo completion bash > ~/.local/share/bash-completion/completions/flujo
```

## Configuration

### Environment Variables

The CLI respects the same environment variables as the Python API:

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Configuration
export FLUJO_TELEMETRY_ENABLED="true"
export FLUJO_LOG_LEVEL="INFO"
export FLUJO_CACHE_ENABLED="true"
```

### Configuration Files

The CLI can use configuration files for persistent settings:

```yaml
# ~/.flujo/config.yaml
telemetry:
  enabled: true
  endpoint: "https://telemetry.example.com"

logging:
  level: INFO
  format: json

cache:
  enabled: true
  ttl: 3600

recipes:
  default:
    max_iterations: 10
    reflection_enabled: true
```

## Output Formats

### JSON Output

```bash
flujo solve "Task" --output json
```

```json
{
  "success": true,
  "solution": "Generated solution...",
  "score": 0.85,
  "metadata": {
    "cost_usd": 0.023,
    "duration_ms": 1250,
    "steps": 3
  }
}
```

### YAML Output

```bash
flujo solve "Task" --output yaml
```

```yaml
success: true
solution: Generated solution...
score: 0.85
metadata:
  cost_usd: 0.023
  duration_ms: 1250
  steps: 3
```

### Table Output

```bash
flujo bench "Task" --output table
```

```
┌─────────┬────────────┬──────────┬─────────────┬──────────────┐
│ Round   │ Recipe     │ Score    │ Cost (USD)  │ Duration (ms) │
├─────────┼────────────┼──────────┼─────────────┼──────────────┤
│ 1       │ default    │ 0.85     │ 0.023       │ 1250          │
│ 2       │ default    │ 0.87     │ 0.025       │ 1300          │
│ 3       │ default    │ 0.83     │ 0.022       │ 1200          │
└─────────┴────────────┴──────────┴─────────────┴──────────────┘
```

## Error Handling

The CLI provides clear error messages and exit codes:

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: API error
- `4`: Validation error

### Common Error Scenarios

```bash
# Missing API key
flujo solve "Task"
# Error: OPENAI_API_KEY not found in environment

# Invalid recipe
flujo solve "Task" --recipe invalid
# Error: Unknown recipe 'invalid'

# Network error
flujo solve "Task"
# Error: Failed to connect to OpenAI API
```

## Scripting and Automation

The CLI is designed for scripting and automation:

```bash
#!/bin/bash
# Example script using Flujo CLI

# Solve a task and capture output
result=$(flujo solve "Generate a function" --output json)

# Extract solution using jq
solution=$(echo "$result" | jq -r '.solution')

# Save to file
echo "$solution" > generated_function.py

# Run benchmark
flujo bench "Test the function" --rounds 5 --output csv > benchmark_results.csv
```

## Best Practices

### 1. Use Configuration Files
Store common settings in configuration files rather than command-line options.

### 2. Leverage Output Formats
Use JSON/YAML output for programmatic integration and automation.

### 3. Monitor Costs
Use the `--verbose` flag to monitor API costs during development.

### 4. Test with Benchmarks
Use `flujo bench` to evaluate different approaches before production deployment.

### 5. Validate Configurations
Run `flujo validate` regularly to ensure configuration integrity.

This CLI reference provides comprehensive coverage of all available commands and their usage patterns for effective Flujo workflow management. 