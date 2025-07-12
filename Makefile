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

.PHONY: sync
sync: .uv ## Update dependencies based on pyproject.toml
	@echo "🔄 Syncing dependencies..."
	@uv sync --all-extras


# ------------------------------------------------------------------------------
# Code Quality & Formatting
# ------------------------------------------------------------------------------

.PHONY: format
format: .uv ## Auto-format the code with ruff
	@echo "🎨 Formatting code..."
	@uv run ruff format flujo/ tests/

.PHONY: lint
lint: .uv ## Lint the code and check for formatting issues
	@echo "🔎 Linting code..."
	@uv run ruff format --check flujo/ tests/
	@uv run ruff check flujo/ tests/

.PHONY: typecheck
typecheck: .uv ## Run static type checking with mypy
	@echo "🧐 Running static type checking..."
	@uv run mypy flujo/


# ------------------------------------------------------------------------------
# Testing & Coverage
# ------------------------------------------------------------------------------

.PHONY: test
test: .uv ## Run all tests
	@echo "🧪 Running tests..."
	@uv run pytest tests/

.PHONY: testcov
testcov: .uv ## Run tests and generate an HTML coverage report
	@echo "🧪 Running tests with coverage..."
	@uv run coverage run --source=flujo -m pytest tests/
	@uv run coverage html
	@echo "\n✅ Coverage report generated in 'htmlcov/'. Open htmlcov/index.html to view."


# ------------------------------------------------------------------------------
# All-in-one & Help
# ------------------------------------------------------------------------------

.PHONY: all
all: format lint typecheck test ## Run all quality checks (format, lint, typecheck, test)
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
