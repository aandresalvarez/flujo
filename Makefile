# Makefile for pydantic-ai-orchestrator

# Shell to use, with flags for robustness:
# -e: exit immediately if a command exits with a non-zero status.
# -u: treat unset variables as an error when substituting.
# -o pipefail: the return value of a pipeline is the status of the last command to exit with a non-zero status,
#             or zero if no command exited with a non-zero status.
SHELL := /bin/bash
SHELLFLAGS := -euo pipefail -c

# Variables
PACKAGE_NAME = pydantic_ai_orchestrator
PYTHON_PATHS = $(PACKAGE_NAME) tests examples # Directories for linting/type-checking
VENV_DIR ?= .venv # Allow overriding venv directory

# Default target when `make` is run without arguments
.DEFAULT_GOAL := help

# Runner prefixes
PIP_RUN =
POETRY_RUN = poetry run
UV_RUN = uv run

# Common Pytest commands (DRY)
PYTEST_CMD = pytest
PYTEST_COV_CMD = $(PYTEST_CMD) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=xml # Added XML for CI
PYTEST_FAST_CMD = $(PYTEST_CMD)
PYTEST_E2E_CMD = $(PYTEST_CMD) -m e2e
PYTEST_UNIT_CMD = $(PYTEST_CMD) -m "not e2e and not benchmark" # Exclude benchmark too
PYTEST_BENCH_CMD = $(PYTEST_CMD) -m benchmark

# Phony targets (targets that are not actual files)
.PHONY: help all dev poetry-dev pip-dev uv-dev \
        test test-fast test-e2e test-unit test-bench \
        poetry-test poetry-test-fast poetry-test-e2e poetry-test-unit poetry-test-bench \
        pip-test pip-test-fast pip-test-e2e pip-test-unit pip-test-bench \
        uv-test uv-test-fast uv-test-e2e uv-test-unit uv-test-bench \
        quality lint format format-check type-check security \
        docs-serve docs-build \
        build clean clean-pyc clean-build clean-test clean-docs \
        requirements

#------------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------------
help:
	@echo "pydantic-ai-orchestrator Makefile"
	@echo "---------------------------------"
	@echo "Targets:"
	@echo ""
	@echo "  Development Environment Setup:"
	@echo "    make dev              - Set up dev environment (defaults to pip-dev)"
	@echo "    make poetry-dev       - Set up dev environment with Poetry"
	@echo "    make pip-dev          - Set up dev environment with pip"
	@echo "    make uv-dev           - Set up dev environment with uv (fastest)"
	@echo "    make requirements     - Generate requirements.txt from poetry.lock (for non-poetry users)"
	@echo ""
	@echo "  Code Quality (use 'make quality' to run all):"
	@echo "    make lint             - Check code style with Ruff"
	@echo "    make format           - Format code automatically with Ruff"
	@echo "    make format-check     - Check formatting without changing files (for CI)"
	@echo "    make type-check       - Run type checking with MyPy"
	@echo "    make security         - Check for security vulnerabilities with pip-audit"
	@echo ""
	@echo "  Testing:"
	@echo "    make test             - Run all tests with coverage (default runner: pip)"
	@echo "    make test-fast        - Run tests without coverage (default: pip)"
	@echo "    make test-e2e         - Run only end-to-end tests (default: pip)"
	@echo "    make test-unit        - Run only unit tests (default: pip)"
	@echo "    make test-bench       - Run only benchmark tests (default: pip)"
	@echo "    ---"
	@echo "    For other runners, prefix with 'poetry-' or 'uv-'. Examples:"
	@echo "    make poetry-test      - Run all tests with coverage using Poetry"
	@echo "    make uv-test-fast     - Run tests without coverage using uv"
	@echo ""
	@echo "  Build & All-in-One:"
	@echo "    make build            - Build the package using Hatch"
	@echo "    make all              - Run lint, type-check, tests (pip), and build docs"
	@echo ""
	@echo "  Documentation:"
	@echo "    make docs-serve       - Start local documentation server"
	@echo "    make docs-build       - Build documentation"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            - Remove all build artifacts, caches, and pyc files"
	@echo "    make clean-pyc        - Remove Python __pycache__ directories and .pyc files"
	@echo "    make clean-build      - Remove build/dist artifacts"
	@echo "    make clean-test       - Remove test artifacts (coverage, pytest cache)"
	@echo "    make clean-docs       - Remove built documentation (site/)"
	@echo ""

#------------------------------------------------------------------------------------
# Default Convenience Targets
#------------------------------------------------------------------------------------
all: quality pip-test docs-build
dev: pip-dev

#------------------------------------------------------------------------------------
# Development Environment Setup
#------------------------------------------------------------------------------------
poetry-dev:
	@echo "Setting up development environment with Poetry..."
	$(POETRY_RUN) install --with dev --with docs --with bench

pip-dev:
	@echo "Setting up development environment with pip..."
	pip install -e ".[dev,docs,bench]"

uv-dev:
	@echo "Setting up development environment with uv..."
	$(UV_RUN) pip install -e ".[dev,docs,bench]"

requirements:
	@echo "Generating requirements.txt from poetry.lock..."
	$(POETRY_RUN) export --format requirements.txt --output requirements.txt --without-hashes --with dev --with docs --with bench

#------------------------------------------------------------------------------------
# Testing Commands
#------------------------------------------------------------------------------------
# Default test runner is pip
test: pip-test
test-fast: pip-test-fast
test-e2e: pip-test-e2e
test-unit: pip-test-unit
test-bench: pip-test-bench

# Poetry
poetry-test:
	@echo "Running tests with Poetry (with coverage)..."
	$(POETRY_RUN) $(PYTEST_COV_CMD)

poetry-test-fast:
	@echo "Running tests with Poetry (fast mode)..."
	$(POETRY_RUN) $(PYTEST_FAST_CMD)

poetry-test-e2e:
	@echo "Running end-to-end tests with Poetry..."
	$(POETRY_RUN) $(PYTEST_E2E_CMD)

poetry-test-unit:
	@echo "Running unit tests with Poetry..."
	$(POETRY_RUN) $(PYTEST_UNIT_CMD)

poetry-test-bench:
	@echo "Running benchmark tests with Poetry..."
	$(POETRY_RUN) $(PYTEST_BENCH_CMD)

# Pip
pip-test:
	@echo "Running tests with pip (with coverage)..."
	$(PIP_RUN) $(PYTEST_COV_CMD)

pip-test-fast:
	@echo "Running tests with pip (fast mode)..."
	$(PIP_RUN) $(PYTEST_FAST_CMD)

pip-test-e2e:
	@echo "Running end-to-end tests with pip..."
	$(PIP_RUN) $(PYTEST_E2E_CMD)

pip-test-unit:
	@echo "Running unit tests with pip..."
	$(PIP_RUN) $(PYTEST_UNIT_CMD)

pip-test-bench:
	@echo "Running benchmark tests with pip..."
	$(PIP_RUN) $(PYTEST_BENCH_CMD)

# UV
uv-test:
	@echo "Running tests with uv (with coverage)..."
	$(UV_RUN) $(PYTEST_COV_CMD)

uv-test-fast:
	@echo "Running tests with uv (fast mode)..."
	$(UV_RUN) $(PYTEST_FAST_CMD)

uv-test-e2e:
	@echo "Running end-to-end tests with uv..."
	$(UV_RUN) $(PYTEST_E2E_CMD)

uv-test-unit:
	@echo "Running unit tests with uv..."
	$(UV_RUN) $(PYTEST_UNIT_CMD)

uv-test-bench:
	@echo "Running benchmark tests with uv..."
	$(UV_RUN) $(PYTEST_BENCH_CMD)

#------------------------------------------------------------------------------------
# Code Quality
#------------------------------------------------------------------------------------
quality: lint format-check type-check security

lint:
	@echo "Linting with Ruff..."
	ruff check $(PYTHON_PATHS)

format:
	@echo "Formatting with Ruff..."
	ruff format $(PYTHON_PATHS)

format-check:
	@echo "Checking formatting with Ruff..."
	ruff format --check $(PYTHON_PATHS)

type-check:
	@echo "Type checking with MyPy..."
	mypy $(PYTHON_PATHS)

security:
	@echo "Checking security with pip-audit..."
	pip-audit

#------------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------------
build:
	@echo "Building package with Hatch..."
	hatch build

#------------------------------------------------------------------------------------
# Documentation
#------------------------------------------------------------------------------------
docs-serve:
	@echo "Serving documentation at http://127.0.0.1:8000 ..."
	mkdocs serve

docs-build:
	@echo "Building documentation..."
	mkdocs build

#------------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------------
clean: clean-build clean-pyc clean-test clean-docs
	@echo "All build artifacts, caches, and .pyc files removed."

clean-pyc:
	@echo "Removing Python __pycache__ directories and .pyc files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-build:
	@echo "Removing build/dist artifacts..."
	rm -rf build/ dist/ *.egg-info/

clean-test:
	@echo "Removing test artifacts..."
	rm -rf .coverage coverage.xml .pytest_cache/ htmlcov/

clean-docs:
	@echo "Removing built documentation..."
	rm -rf site/

# Tool-specific cache cleanup
clean-ruff:
	@echo "Removing Ruff cache..."
	rm -rf .ruff_cache/

clean-mypy:
	@echo "Removing MyPy cache..."
	rm -rf .mypy_cache/

clean-cache: clean-ruff clean-mypy
	@echo "All tool caches removed." 