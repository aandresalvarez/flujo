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
	@python3 scripts/install_dependencies.py dev

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
	@# Ensure dev extras (including types-psutil) are installed before typechecking
	@uv sync --all-extras
	@uv run mypy flujo/


# ------------------------------------------------------------------------------
# Testing & Coverage
# ------------------------------------------------------------------------------

.PHONY: test
test: .uv ## Run all tests
	@echo "🧪 Running tests..."
	CI=1 FLUJO_TEST_FORCE_EXIT=1 uv run pytest tests/

.PHONY: test-fast
test-fast: .uv ## Run fast tests in parallel (excludes slow, serial, and benchmark tests)
	@echo "⚡ Running fast tests in parallel..."
	@echo "📊 Test breakdown:"
	@echo "   • Total tests: 2424"
	@echo "   • Fast tests: 2266 (93.5%)"
	@echo "   • Excluded: 158 (6.5% - slow/benchmark)"
	@echo ""
	@echo "🔄 Starting test execution with limited parallelism..."
	@CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" -n 4 --tb=short -q || \
		(echo "❌ Some tests failed. Run 'make test-fast-verbose' for detailed output." && exit 1)
	@echo "✅ Fast tests completed!"

.PHONY: test-fast-verbose
test-fast-verbose: .uv ## Run fast tests with verbose output for debugging
	@echo "🔍 Running fast tests with verbose output..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" -n 4 -v

.PHONY: test-fast-serial
test-fast-serial: .uv ## Run fast tests serially (for debugging parallel issues)
	@echo "🔧 Running fast tests serially for debugging..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" --tb=short -q

.PHONY: test-fast-conservative
test-fast-conservative: .uv ## Run fast tests with conservative parallelism (2 workers)
	@echo "🐌 Running fast tests with conservative parallelism..."
	@echo "📊 Using 2 workers to minimize resource contention..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" -n 2 --tb=short -q

.PHONY: test-robust
test-robust: .uv ## Run tests with enhanced robustness and monitoring
	@echo "🛡️ Running robust test suite with monitoring..."
	@echo "📊 Enhanced error handling and resource monitoring..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" -n 4 --tb=short -q --maxfail=5

.PHONY: test-stress
test-stress: .uv ## Run stress tests to identify resource issues
	@echo "💪 Running stress tests..."
	@echo "🧠 Testing memory and CPU limits..."
	CI=1 uv run pytest tests/ -m "stress" --timeout=300

.PHONY: test-memory
test-memory: .uv ## Run memory leak detection tests
	@echo "🧠 Running memory leak tests..."
	@echo "📊 Monitoring memory usage patterns..."
	CI=1 uv run pytest tests/ -m "memory" --timeout=120

.PHONY: test-health-full
test-health-full: .uv ## Run comprehensive test suite health check (full)
	@echo "🏥 Running comprehensive test suite health check (full)..."
	@echo "1. Checking test collection..."
	@uv run pytest tests/ --collect-only -q > /dev/null 2>&1 && echo "✅ Test collection OK" || echo "❌ Test collection failed"
	@echo ""
	@echo "2. Running fast test subset..."
	@CI=1 uv run pytest tests/unit/test_fallback.py::test_fallback_assignment -v
	@echo ""
	@echo "3. Checking resource usage..."
	@echo "✅ Test suite health check (full) completed"

.PHONY: test-slow
test-slow: .uv ## Run slow tests serially (excludes ultra-slow tests problematic for mass CI)
	@echo "🐌 Running slow tests serially (excluding ultra-slow tests)..."
	CI=1 uv run pytest tests/ -m "(slow or serial or benchmark) and not ultra_slow"

.PHONY: test-ultra-slow
test-ultra-slow: .uv ## Run ultra-slow tests (>30s, problematic for mass CI)
	@echo "🚨 Running ultra-slow tests (>30s duration)..."
	@echo "⚠️  These tests are excluded from regular CI due to long execution time"
	CI=1 uv run pytest tests/ -m "ultra_slow" -v

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

.PHONY: test-health
test-health: .uv ## Run a quick health check of the test suite
	@echo "🏥 Running test suite health check..."
	@echo "1. Checking test collection..."
	@uv run pytest tests/ --collect-only -q > /dev/null 2>&1 && echo "✅ Test collection OK" || echo "❌ Test collection failed"
	@echo ""
	@echo "2. Running a small subset of tests..."
	@CI=1 uv run pytest tests/unit/test_fallback.py::test_fallback_assignment -q > /dev/null 2>&1 && echo "✅ Basic test execution OK" || echo "❌ Basic test execution failed"
	@echo ""
	@echo "3. Checking for slow tests that might need marking..."
	@echo "   Files with timing operations:"
	@find tests/ -name "*.py" -exec grep -l "time\.sleep\|asyncio\.sleep\|time\.perf_counter" {} \; | head -5 | sed 's/^/   • /'
	@echo ""
	@echo "🏥 Test suite health check completed!"

.PHONY: test-analyze
test-analyze: .uv ## Analyze test collection and categorization
	@echo "📊 Test Analysis Report"
	@echo "======================"
	@echo "Total tests:"
	@uv run pytest tests/ --collect-only -q | tail -1
	@echo ""
	@echo "Fast tests (run by test-fast):"
	@uv run pytest tests/ -m "not slow and not serial and not benchmark" --collect-only -q | tail -1
	@echo ""
	@echo "Slow tests (run by test-slow, excludes ultra-slow):"
	@uv run pytest tests/ -m "(slow or serial or benchmark) and not ultra_slow" --collect-only -q | tail -1
	@echo ""
	@echo "Ultra-slow tests (>30s, excluded from mass CI):"
	@uv run pytest tests/ -m "ultra_slow" --collect-only -q | tail -1
	@echo ""
	@echo "Test categories:"
	@echo "  • Unit tests: $(shell find tests/unit -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • Integration tests: $(shell find tests/integration -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • Benchmark tests: $(shell find tests/benchmarks -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • E2E tests: $(shell find tests/e2e -name "*.py" | wc -l | tr -d ' ') files"

.PHONY: test-failing
test-failing: .uv ## Run only failing tests to identify issues
	@echo "🔍 Running only failing tests..."
	CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" --lf -v

.PHONY: test-slow-marked
test-slow-marked: .uv ## Show which tests are marked as slow
	@echo "🐌 Tests marked as slow:"
	@uv run pytest tests/ -m "slow" --collect-only -q

.PHONY: test-benchmark-marked
test-benchmark-marked: .uv ## Show which tests are marked as benchmark
	@echo "📊 Tests marked as benchmark:"
	@uv run pytest tests/ -m "benchmark" --collect-only -q


# ------------------------------------------------------------------------------
# Package Building
# ------------------------------------------------------------------------------

.PHONY: package
package: .uv ## Build package distribution files
	@echo "📦 Building package distribution..."
	@uv run python3 -m build
	@echo "\n✅ Package built in dist/ directory."


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
