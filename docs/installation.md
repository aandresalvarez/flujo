# Installation Guide

This guide will help you install and set up `pydantic-ai-orchestrator` for your project.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Basic Installation

The simplest way to install the package is using pip:

```bash
pip install pydantic-ai-orchestrator
```

## Installation with Extras

The package includes several optional extras that provide additional functionality:

```bash
# For development (includes testing tools)
pip install "pydantic-ai-orchestrator[dev]"

# For benchmarking
pip install "pydantic-ai-orchestrator[bench]"

# For documentation
pip install "pydantic-ai-orchestrator[docs]"

# Install all extras
pip install "pydantic-ai-orchestrator[all]"
```

## Development Installation

For development work, you can install the package in editable mode:

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/rloop.git
cd rloop

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with all extras
pip install -e ".[all]"
```

## Environment Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```env
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   ```

## Verifying Installation

To verify your installation:

```python
from pydantic_ai_orchestrator import Orchestrator
print(Orchestrator.__version__)  # Should print the installed version
```

## Troubleshooting

### Common Issues

1. **Python Version Error**
   - Ensure you're using Python 3.11 or higher
   - Check with: `python --version`

2. **Missing Dependencies**
   - Try reinstalling with: `pip install --upgrade pydantic-ai-orchestrator`

3. **API Key Issues**
   - Verify your `.env` file exists and contains valid API keys
   - Check that the keys are properly formatted

### Getting Help

If you encounter any issues:
1. Check the [troubleshooting guide](troubleshooting.md)
2. Search [existing issues](https://github.com/aandresalvarez/rloop/issues)
3. Create a new issue if needed

## Next Steps

- Read the [Quickstart Guide](quickstart.md) to get started
- Explore the [Tutorial](tutorial.md) for a guided tour
- Check out [Use Cases](use_cases.md) for inspiration 