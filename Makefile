.DEFAULT_GOAL := help

args = $(filter-out $@,$(MAKECMDGOALS))

.PHONY: help install quality lint format type-check test cov bandit sbom pip-dev pip-install clean

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies for development."
	@echo "  quality       - Run all code quality checks."
	@echo "  lint          - Run Ruff linting."
	@echo "  format        - Auto-format the code."
	@echo "  type-check    - Run MyPy type checking."
	@echo "  test          - Run the test suite (use 'make test args=\"-k expr\"')."
	@echo "  cov           - Run tests with coverage (uses args too)."
	@echo "  bandit        - Run Bandit security scan."
	@echo "  sbom          - Generate a CycloneDX SBOM."
	@echo "  clean         - Remove build artifacts and caches."

install:
	@pip install hatch
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
	@hatch run quality

lint:
	@hatch run lint

format:
	@hatch run format

type-check:
	@hatch run type-check

test:
	@hatch run test $(args)

cov:
	@hatch run cov $(args)

bandit:
	@hatch run bandit-check

sbom:
	@hatch run cyclonedx

cyclonedx: sbom

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
