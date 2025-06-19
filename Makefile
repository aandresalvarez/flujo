.PHONY: help install quality test cov bandit cyclonedx pip-dev pip-install clean

help:
	@echo "Commands:"
	@echo "  install    : Create a hatch environment and install dependencies."
	@echo "  pip-dev    : Install development dependencies using pip."
	@echo "  pip-install: Install the package in development mode using pip."
	@echo "  quality    : Run all code quality checks (format, lint, types, security)."
	@echo "  test       : Run all tests with pytest."
	@echo "  cov        : Run tests and report code coverage."
	@echo "  bandit     : Run Bandit security scan."
	@echo "  cyclonedx  : Generate a CycloneDX SBOM."
	@echo "  clean      : Clean up build artifacts and caches."

install:
	@hatch env create

pip-dev:
	@echo "📦 Installing development dependencies with pip..."
	@python -m pip install --upgrade pip
	@python -m pip install -e ".[dev]"

pip-install:
	@echo "📦 Installing package in development mode with pip..."
	@python -m pip install --upgrade pip
	@python -m pip install -e .

quality:
	@echo "✅ Running all quality checks..."
	@hatch run quality

test:
	@echo "🧪 Running tests..."
	@hatch run test

cov:
	@echo "📊 Running tests with coverage..."
	@hatch run cov

bandit:
	@echo "🔍 Running Bandit security scan..."
	@hatch run bandit-check

cyclonedx:
	@echo "📦 Generating CycloneDX SBOM..."
	@hatch run cyclonedx-py environment --pyproject pyproject.toml --output-file sbom.json --output-format JSON

clean:
	@echo "🧹 Cleaning up build artifacts and caches..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf htmlcov/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
