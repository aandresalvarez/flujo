# Flujo Installation Guide

This guide provides comprehensive instructions for installing Flujo and its dependencies.

## Quick Start

### For Developers

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/flujo.git
cd flujo

# Install with robust verification (recommended)
make install-robust

# Or install with basic setup
make install
```

### For Users

```bash
# Install from PyPI
pip install flujo

# Install with optional extras
pip install "flujo[dev,prometheus,logfire]"
```

## Prerequisites

### Required
- **Python 3.11+** - Flujo requires Python 3.11 or higher
- **uv** - Fast Python package installer (recommended)
  ```bash
  # Install uv
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Optional
- **Git** - For development
- **Make** - For using Makefile commands (macOS/Linux)

## Installation Methods

### Method 1: Robust Installation (Recommended for Developers)

This method includes comprehensive verification and testing:

```bash
make install-robust
```

This will:
- ✅ Check if uv is installed
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Verify critical dependencies
- ✅ Run a quick functionality test
- ✅ Provide clear error messages

### Method 2: Basic Installation

```bash
make install
```

### Method 3: Manual Installation

```bash
# Create virtual environment
uv venv

# Install dependencies
uv sync --all-extras

# Activate environment
source .venv/bin/activate
```

### Method 4: Using pip (for users)

```bash
# Basic installation
pip install flujo

# With optional extras
pip install "flujo[dev,prometheus,logfire,sql]"
```

## Dependency Groups

Flujo uses optional dependency groups to keep the core installation lightweight:

### Core Dependencies (always installed)
- `pydantic` - Data validation
- `pydantic-ai` - AI integration
- `pydantic-settings` - Configuration management
- `aiosqlite` - Async SQLite support
- `tenacity` - Retry logic
- `typer` - CLI framework
- `rich` - Terminal formatting
- `pydantic-evals` - Intelligent evaluations

### Development Dependencies (`[dev]`)
- `ruff` - Code formatting and linting
- `mypy` - Static type checking
- `pytest` - Testing framework
- `hypothesis` - Property-based testing
- `prometheus-client` - Metrics collection
- `httpx` - HTTP client for tests
- And more...

### Optional Extras
- `[prometheus]` - Prometheus metrics
- `[logfire]` - Logfire telemetry
- `[sql]` - SQL validation
- `[opentelemetry]` - OpenTelemetry support
- `[lens]` - Lens CLI tools
- `[docs]` - Documentation tools
- `[bench]` - Benchmarking tools

## Verification

After installation, verify everything works:

```bash
# Run all tests
make test

# Run quality checks
make all

# Try the quickstart example
python examples/00_quickstart.py
```

## Troubleshooting

### Common Issues

#### 1. "prometheus_client is not installed"
**Solution**: Install with dev dependencies
```bash
uv sync --extra dev
```

#### 2. "uv is not installed"
**Solution**: Install uv first
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. "Python 3.11+ required"
**Solution**: Upgrade Python or use pyenv
```bash
pyenv install 3.11.0
pyenv local 3.11.0
```

#### 4. Import errors in tests
**Solution**: Ensure all test dependencies are installed
```bash
uv sync --all-extras
```

### Dependency Verification

The robust installation script automatically verifies:

**Critical Dependencies:**
- ✅ pydantic
- ✅ pydantic_ai
- ✅ pydantic_settings
- ✅ aiosqlite
- ✅ tenacity
- ✅ typer
- ✅ rich
- ✅ pydantic_evals

**Optional Dependencies:**
- ⚠️ prometheus_client
- ⚠️ httpx
- ⚠️ logfire
- ⚠️ sqlvalidator

### Manual Verification

```bash
# Test basic import
python -c "import flujo; print('✅ Basic import works')"

# Test pipeline functionality
python -c "from flujo import Pipeline, step; print('✅ Pipeline import works')"

# Test optional dependencies
python -c "from flujo.telemetry.prometheus import PrometheusCollector; print('✅ Prometheus works')" 2>/dev/null || echo "⚠️ Prometheus not available"
```

## Development Setup

For contributors:

```bash
# Install with all development dependencies
make install-robust

# Run tests
make test

# Run quality checks
make all

# Run specific test categories
make test-unit
make test-integration
make test-e2e
make test-bench
```

## CI/CD

The project includes comprehensive CI/CD setup that ensures:

- ✅ All dependencies are properly declared
- ✅ Tests run with all required dependencies
- ✅ Optional dependencies are tested separately
- ✅ Installation works across different Python versions

## Environment Variables

Set these for advanced configuration:

```bash
# Disable telemetry
export FLUJO_DISABLE_TELEMETRY=1

# Set log level
export FLUJO_LOG_LEVEL=DEBUG

# Configure database path
export FLUJO_DB_PATH=/path/to/database.db
```

## Next Steps

After successful installation:

1. **Read the documentation**: https://aandresalvarez.github.io/flujo/
2. **Try examples**: `python examples/00_quickstart.py`
3. **Run tests**: `make test`
4. **Start developing**: Check out the contributing guide

## Support

If you encounter issues:

1. Check this installation guide
2. Review the troubleshooting section
3. Check the [GitHub issues](https://github.com/aandresalvarez/flujo/issues)
4. Create a new issue with detailed information

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.
