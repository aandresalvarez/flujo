# Streamlined Configuration Management

This document describes the new streamlined configuration management system in flujo, which provides a unified approach to configuring the framework using `flujo.toml` files as the primary configuration source.

## Overview

The new configuration system addresses the complexity of managing multiple configuration sources by:

1. **Prioritizing File-Based Configuration**: `flujo.toml` files serve as the primary configuration method
2. **Centralizing CLI Configuration**: CLI commands load their defaults from configuration files
3. **Maintaining Backward Compatibility**: Environment variables and existing settings continue to work
4. **Providing Clear Precedence**: Configuration sources are applied in a predictable order

## Configuration Sources and Precedence

Configuration is loaded from multiple sources in the following order (later sources override earlier ones):

1. **Default Settings**: Built-in defaults in the `Settings` class
2. **Environment Variables**: `.env` files and system environment variables
3. **Configuration Files**: `flujo.toml` files (new)
4. **CLI Arguments**: Command-line options (highest precedence)

## Configuration File Format

The `flujo.toml` file uses TOML format and supports the following sections:

### Global Settings

```toml
# State backend configuration
state_uri = "sqlite:///flujo_ops.db"

# Global settings overrides
[settings]
# Feature toggles
reflection_enabled = true
reward_enabled = true
telemetry_export_enabled = false
otlp_export_enabled = false
state_backend_span_export_enabled = true

# Default models for each agent
default_solution_model = "openai:gpt-4o"
default_review_model = "openai:gpt-4o"
default_validator_model = "openai:gpt-4o"
default_reflection_model = "openai:gpt-4o"
default_self_improvement_model = "openai:gpt-4o"
default_repair_model = "openai:gpt-4o"

# Orchestrator tuning
max_iters = 5
k_variants = 3
reflection_limit = 3
scorer = "ratio"
t_schedule = [1.0, 0.8, 0.5, 0.2]
agent_timeout = 60
```

### CLI Command Defaults

```toml
# Default parameters for the solve command
[solve]
max_iters = 3
k = 1
reflection = true
scorer = "ratio"
solution_model = "openai:gpt-4o"
review_model = "openai:gpt-4o"
validator_model = "openai:gpt-4o"
reflection_model = "openai:gpt-4o"

# Default parameters for the bench command
[bench]
rounds = 10

# Default parameters for the run command
[run]
pipeline_name = "pipeline"
json_output = false
```

## Configuration File Discovery

The system automatically discovers configuration files by searching:

1. **Current Directory**: `./flujo.toml`
2. **Parent Directories**: Searches upward through parent directories
3. **Custom Path**: Can be specified via the configuration manager

## Usage Examples

### Basic Configuration

Create a `flujo.toml` file in your project root:

```toml
[settings]
default_solution_model = "openai:gpt-4o"
max_iters = 5

[solve]
max_iters = 3
k = 2
```

Now when you run `flujo solve "your prompt"`, it will use:
- `max_iters = 3` (from solve section)
- `k = 2` (from solve section)
- `default_solution_model = "openai:gpt-4o"` (from settings section)

### Environment-Specific Configuration

You can use different configuration files for different environments:

```bash
# Development
FLUJO_CONFIG_PATH=./config/dev.toml flujo solve "test prompt"

# Production
FLUJO_CONFIG_PATH=./config/prod.toml flujo solve "production prompt"
```

### CLI Override

CLI arguments always take precedence over configuration files:

```bash
# This will use max_iters = 5, overriding the config file
flujo solve "prompt" --max-iters 5
```

## API Usage

### Programmatic Configuration

```python
from flujo.infra.config_manager import get_config_manager, load_settings

# Load settings with configuration file overrides
settings = load_settings()

# Get CLI defaults for a specific command
config_manager = get_config_manager()
cli_defaults = config_manager.get_cli_defaults("solve")

# Get state URI from configuration
state_uri = config_manager.get_state_uri()
```

### Process-Local Caching and Test Hygiene

`get_config_manager()` now returns a cached instance per process. This avoids re-reading
`flujo.toml` on every call. When you change `FLUJO_CONFIG_PATH` or mutate the environment
within a test, either force a refresh or clear the cache:

```python
from flujo.infra.config_manager import (
    get_config_manager,
    invalidate_config_cache,
)

# Respect a new FLUJO_CONFIG_PATH for this process
cfg_mgr = get_config_manager(force_reload=True)

# Or clear the cache entirely (useful in test teardown)
invalidate_config_cache()
```

### Custom Configuration Path

```python
from flujo.infra.config_manager import ConfigManager

# Use a custom configuration file
config_manager = ConfigManager("path/to/custom.toml")
settings = config_manager.get_settings()
```

## Migration Guide

### From Environment Variables

If you're currently using environment variables, you can migrate to configuration files:

**Before:**
```bash
export FLUJO_MAX_ITERS=5
export FLUJO_DEFAULT_SOLUTION_MODEL=openai:gpt-4o
flujo solve "prompt"
```

**After:**
```toml
# flujo.toml
[settings]
max_iters = 5
default_solution_model = "openai:gpt-4o"

[solve]
max_iters = 3  # Override for solve command
```

```bash
flujo solve "prompt"  # No environment variables needed
```

### From Multiple .env Files

Instead of managing multiple `.env` files, use a single `flujo.toml`:

**Before:**
```bash
# .env.dev
FLUJO_MAX_ITERS=3
FLUJO_DEFAULT_SOLUTION_MODEL=openai:gpt-3.5-turbo

# .env.prod
FLUJO_MAX_ITERS=5
FLUJO_DEFAULT_SOLUTION_MODEL=openai:gpt-4o
```

**After:**
```toml
# config/dev.toml
[settings]
max_iters = 3
default_solution_model = "openai:gpt-3.5-turbo"

# config/prod.toml
[settings]
max_iters = 5
default_solution_model = "openai:gpt-4o"
```

## Best Practices

### 1. Version Control Configuration

- **Include in version control**: `flujo.toml` files should be committed to version control
- **Exclude secrets**: Never commit API keys or sensitive information
- **Use environment variables for secrets**: Keep API keys in environment variables or `.env` files

### 2. Project Structure

```
my-flujo-project/
├── flujo.toml          # Project configuration
├── .env                # Secrets (not in version control)
├── pipelines/
│   └── my_pipeline.py
└── config/
    ├── dev.toml        # Development overrides
    └── prod.toml       # Production overrides
```

### 3. Configuration Hierarchy

1. **Project-level**: `./flujo.toml` for project-specific settings
2. **Environment-specific**: `./config/{env}.toml` for environment overrides
3. **User-specific**: `~/.flujo.toml` for personal preferences (future feature)

### 4. Sensitive Information

```toml
# flujo.toml - Safe to commit
[settings]
default_solution_model = "openai:gpt-4o"
max_iters = 5

# .env - Not committed to version control
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Troubleshooting

### Configuration Not Loading

1. **Check file location**: Ensure `flujo.toml` is in the current directory or a parent directory
2. **Check file format**: Verify the TOML syntax is correct
3. **Check precedence**: Remember CLI arguments override configuration files

### Debugging Configuration

Use the `show-config` command to see the effective configuration:

```bash
flujo show-config
```

This will display all settings with configuration file overrides applied.

### Common Issues

1. **Configuration ignored**: CLI arguments always take precedence
2. **File not found**: Check the file path and TOML syntax
3. **Settings not applied**: Verify the section names match the expected format

## Future Enhancements

The configuration system is designed to be extensible. Future enhancements may include:

- **User-specific configuration**: `~/.flujo.toml` for personal defaults
- **Configuration validation**: Schema validation for configuration files
- **Configuration inheritance**: Hierarchical configuration with inheritance
- **Dynamic configuration**: Runtime configuration updates
- **Configuration templates**: Pre-built configuration templates for common use cases
