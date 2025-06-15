# Contributing & Local Dev Guide

Thanks for helping improve **flujo**! This guide will help you set up a fully-featured development environment for Python 3.11+.

---

## 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/rloop.git
cd rloop

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python --version                    # should print 3.11.x
```

> **Note:** If `python3.11` isn't available, install it via:
> - macOS: `brew install python@3.11`
> - Linux: Use your distribution's package manager
> - Windows: Download from python.org
> - Or use pyenv: `pyenv install 3.11.x`

---

## 2. Development Environment Setup

The project supports multiple package managers. Choose your preferred one:

### Option A: Poetry (Recommended)
```bash
# Install Poetry
pip install --upgrade poetry

# Install dependencies (includes runtime, dev, docs, and bench extras)
make poetry-dev
```

### Option B: UV (Fastest)
```bash
# Install UV
pip install --upgrade uv

# Install dependencies
make uv-dev
```

### Option C: Standard pip
```bash
# Install dependencies
make pip-dev
```

> **Important Notes:**
> - All options install the package in editable mode (`-e`)
> - Code changes are picked up instantly without reinstalling
> - `poetry.lock` is tracked in git for reproducible builds
> - Run `make help` to see all available commands

---

## 3. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Add your API keys to .env
# - OpenAI
# - Cohere
# - Vertex AI
# - Other providers as needed
```

The orchestrator automatically loads this file via **python-dotenv**.

---

## 4. Testing

The project includes comprehensive tests with different runners:

### Full Test Suite
```bash
# With Poetry
make poetry-test                    # includes coverage report
make poetry-test-fast              # without coverage

# With UV (fastest)
make uv-test                       # includes coverage
make uv-test-fast                 # without coverage

# With pip (default)
make test                         # includes coverage
make test-fast                    # without coverage
```

### Specific Test Types
```bash
# Unit Tests
make test-unit                    # or poetry-test-unit / uv-test-unit

# End-to-End Tests
make test-e2e                     # or poetry-test-e2e / uv-test-e2e

# Benchmark Tests
make test-bench                   # or poetry-test-bench / uv-test-bench
```

> **Note:** Async tests are handled automatically by **pytest-asyncio**

---

## 5. Code Quality

### All-in-One Quality Check
```bash
make quality                      # runs all quality checks
```

### Individual Quality Checks
```bash
# Code Style
make lint                        # check with Ruff
make format                      # format code
make format-check               # check formatting (CI)

# Type Safety
make type-check                 # static type checking with MyPy

# Security
make security                   # vulnerability check with pip-audit
```

---

## 6. Documentation

```bash
# Local Development
make docs-serve                 # start docs server at http://127.0.0.1:8000

# Build
make docs-build                # generate static site
```

---

## 7. Package Management

### Building
```bash
# Build package
make package                   # creates wheel and sdist in dist/

# Clean build artifacts
make clean-package            # removes build artifacts
```

### Publishing (Maintainers Only)
```bash
# Test PyPI
make publish-test             # builds and uploads to TestPyPI

# PyPI
make publish                  # builds and uploads to PyPI
```

> **Release Process:**
> 1. Update version in `pyproject.toml`
> 2. `git commit -am "release: vX.Y.Z"`
> 3. `git tag vX.Y.Z && git push --tags`
> 4. GitHub Actions will handle the release

---

## 8. Maintenance

### Cleanup Commands
```bash
# Comprehensive cleanup
make clean                     # removes all artifacts and caches

# Selective cleanup
make clean-pyc                # Python cache files
make clean-build             # build/dist artifacts
make clean-test             # test artifacts
make clean-docs            # documentation
make clean-cache          # tool caches (Ruff, MyPy)
```

### Cache Management
```bash
# Clear specific tool caches
make clean-ruff             # Ruff cache
make clean-mypy            # MyPy cache
make clean-cache          # All tool caches
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ERROR: Package requires a different Python` | Ensure Python 3.11+ is active (`python --version`) |
| Async test failures | Verify `pytest-asyncio` is installed (included in `[dev]`) |
| Poetry cache permission errors | `sudo chown -R $USER ~/Library/Caches/pypoetry` |
| Make not found | Install: `brew install make` (macOS) or `apt install make` (Ubuntu) |
| Tool-specific errors | Run `make clean-cache` and retry |

For all available commands:
```bash
make help
```

Happy coding! 🚀
