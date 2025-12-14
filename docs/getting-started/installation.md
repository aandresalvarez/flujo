# Installation Guide

This guide will help you install and set up `flujo` for your project.

## Prerequisites

- Python 3.13 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Basic Installation

The simplest way to install the package is using pip:

```bash
pip install flujo
```

## Installation with Extras

The package includes several optional extras that provide additional functionality:

```bash
# For development (includes testing tools, linting, etc.)
pip install "flujo[dev]"

# For benchmarking (includes numpy for statistical analysis)
pip install "flujo[bench]"

# For documentation building
pip install "flujo[docs]"

# For OpenTelemetry support
pip install "flujo[opentelemetry]"

# For Logfire-based telemetry
pip install "flujo[logfire]"

# For the SQL syntax validator plugin
pip install "flujo[sql]"

# Install all extras
pip install "flujo[dev,docs,opentelemetry,logfire,sql,bench]"
```

## Development Installation

For development work, you can install the package in editable mode:

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/flujo.git
cd flujo

# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with development extras
pip install -e ".[dev]"
```

## Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```env
   # Required: At least one provider key
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here

   # Optional: Logfire for advanced telemetry
   # LOGFIRE_API_KEY=your_logfire_key

   # Optional: Configuration overrides
   REFLECTION_ENABLED=true
   REWARD_ENABLED=false
   AGENT_TIMEOUT=60
   TELEMETRY_EXPORT_ENABLED=false

   # Optional: Model overrides
   DEFAULT_SOLUTION_MODEL=openai:gpt-4o
   DEFAULT_REVIEW_MODEL=openai:gpt-4o
   DEFAULT_VALIDATOR_MODEL=openai:gpt-4o
   DEFAULT_REFLECTION_MODEL=openai:gpt-4o
   DEFAULT_SELF_IMPROVEMENT_MODEL=openai:gpt-4o
   ```

## Verifying Installation

To verify your installation:

```bash
# Check version
flujo version-cmd

# Test basic functionality
flujo solve "Write a hello world function in Python"

# Show current configuration
flujo show-config
```

You can also verify programmatically:

```python
import flujo
print(f"Version: {flujo.__version__}")

# Test basic import
from flujo.recipes.factories import make_default_pipeline
from flujo import Task
print("✅ Installation successful!")
```

## Troubleshooting

### Common Issues

1. **Python Version Error**
   - Ensure you're using Python 3.13 or higher
   - Check with: `python --version`

2. **Missing Dependencies**
   - Try reinstalling with: `pip install --upgrade flujo`
   - For development: `pip install -e ".[dev]"`

3. **API Key Issues**
   - Verify your `.env` file exists and contains valid API keys
   - Check that the keys are properly formatted
   - Ensure at least one provider key (OpenAI, Anthropic, or Google) is set

4. **Import Errors**
   - Make sure you're in the correct virtual environment
   - Try: `pip list | grep flujo`

5. **Permission Errors**
   - On Unix systems, you might need: `pip install --user flujo`
   - Or use a virtual environment (recommended)

### Getting Help

If you encounter any issues:
1. Check the [troubleshooting guide](../advanced/troubleshooting.md)
2. Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
3. Create a new issue if needed

## Next Steps

- Read the [Quickstart Guide](quickstart.md) to get started
- Explore the [Tutorial](tutorial.md) for a guided tour
- Check out [Use Cases](../user_guide/use_cases.md) for inspiration
