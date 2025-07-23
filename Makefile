# Makefile for the Flujo project
# Provides a consistent set of commands for development, testing, and quality checks.

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Tooling Checks
# ------------------------------------------------------------------------------

# Ensure uv is installed, as it's the core tool for this workflow.
.PHONY: .uv
.uv:
	@uv --version > /dev/null 2>&1 || (printf "\033[0;31m✖ Error: uv is not installed.\033[0m\n  Please install it via: https://github.com/astral-sh/uv\n" && exit 1)


# ------------------------------------------------------------------------------
# Project Setup & Dependency Management
# ------------------------------------------------------------------------------

.PHONY: install
install: .uv ## Install dependencies into a virtual environment
	@echo "🚀 Creating virtual environment and installing dependencies..."
	@uv venv
	@uv sync --all-extras
	@echo "\n✅ Done! Activate the environment with 'source .venv/bin/activate'"

.PHONY: install-robust
install-robust: .uv ## Install dependencies with robust verification
	@echo "🔧 Installing dependencies with robust verification..."
	@python scripts/install_dependencies.py dev

.PHONY: sync
sync: .uv ## Update dependencies based on pyproject.toml
	@echo "🔄 Syncing dependencies..."
	@uv sync --all-extras

.PHONY: pip-dev
pip-dev: ## Install development dependencies (for CI/CD)
	@echo "📦 Installing development dependencies..."
	pip install -e ".[dev,bench,docs]"

# ------------------------------------------------------------------------------
# Code Quality & Formatting
# ------------------------------------------------------------------------------

.PHONY: format
format: .uv ## Auto-format the code with ruff
	@echo "🎨 Formatting code..."
	@uv run ruff format flujo/ tests/

.PHONY: lint
lint: .uv ## Lint the code for issues
	@echo "🔎 Linting code..."
	@uv run ruff check flujo/ tests/ scripts/

.PHONY: typecheck
typecheck: .uv ## Run static type checking with mypy
	@echo "🧐 Running static type checking..."
	@uv run mypy flujo/


# ------------------------------------------------------------------------------
# Testing & Coverage
# ------------------------------------------------------------------------------

.PHONY: test
test: .uv test-cli-integration ## Run all tests (including CLI integration)
	@echo "🧪 Running tests..."
	CI=1 uv run pytest tests/

.PHONY: test-fast
test-fast: .uv test-cli-integration ## Run fast tests in parallel (and CLI integration)
	@echo "⚡ Running fast tests in parallel..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" -n auto

.PHONY: test-slow
test-slow: .uv ## Run slow tests serially
	@echo "🐌 Running slow tests serially..."
	CI=1 uv run pytest tests/ -m "slow or serial or benchmark"

.PHONY: test-parallel
test-parallel: .uv ## Run all tests in parallel (excludes serial tests)
	@echo "🚀 Running tests in parallel..."
	CI=1 uv run pytest tests/ -m "not serial" -n auto

.PHONY: test-unit
test-unit: .uv ## Run unit tests only
	@echo "🧪 Running unit tests..."
	CI=1 uv run pytest tests/unit/

.PHONY: test-integration
test-integration: .uv ## Run integration tests only
	@echo "🔗 Running integration tests..."
	CI=1 uv run pytest tests/integration/

.PHONY: test-bench
test-bench: .uv ## Run benchmark tests only
	@echo "📊 Running benchmark tests..."
	CI=1 uv run pytest tests/benchmarks/

.PHONY: test-e2e
test-e2e: .uv ## Run end-to-end tests only
	@echo "🌐 Running end-to-end tests..."
	CI=1 uv run pytest tests/e2e/

.PHONY: testcov
testcov: .uv ## Run tests and generate an HTML coverage report
	@echo "🧪 Running tests with coverage..."
	@uv run coverage run --source=flujo -m pytest tests/
	@uv run coverage html
	@echo "\n✅ Coverage report generated in 'htmlcov/'. Open htmlcov/index.html to view."

.PHONY: testcov-fast
testcov-fast: .uv ## Run fast tests with coverage in parallel
	@echo "⚡ Running fast tests with coverage in parallel..."
	@uv run coverage run --source=flujo --parallel-mode -m pytest tests/ -m "not slow and not serial and not benchmark" -n auto
	@uv run coverage html
	@echo "\n✅ Coverage report generated in 'htmlcov/'. Open htmlcov/index.html to view."

.PHONY: test-perf
test-perf: .uv ## Run test performance analysis
	@echo "📊 Analyzing test performance..."
	@uv run python tests/performance_monitor.py


# ------------------------------------------------------------------------------
# Package Building
# ------------------------------------------------------------------------------

.PHONY: package
package: .uv ## Build package distribution files
	@echo "📦 Building package distribution..."
	@uv run python -m build
	@echo "\n✅ Package built in dist/ directory."


# ------------------------------------------------------------------------------
# All-in-one & Help
# ------------------------------------------------------------------------------

.PHONY: all
all: format lint typecheck test ## Run all quality checks (format, lint, typecheck, test, CLI integration)
	@echo "\n✅ All local checks passed! You are ready to push."

.PHONY: help
help: ## ✨ Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)

.PHONY: test-cli-integration
# Run all CLI integration test scripts in tests/cli_integration/
test-cli-integration:
	@echo "🧪 Running CLI integration tests..."
	@for script in tests/cli_integration/*.py; do \
	  echo "Running $$script..."; \
	  python3 "$$script"; \
	done
