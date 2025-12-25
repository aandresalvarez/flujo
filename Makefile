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

PYTHON_VERSION ?= 3.13

.PHONY: install
install: .uv ## Install dependencies into a virtual environment
	@echo "🚀 Creating virtual environment and installing dependencies..."
	@uv venv --python $(PYTHON_VERSION)
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
	@uv run python scripts/lint_type_safety.py
	@uv run python scripts/lint_adapter_allowlist.py

.PHONY: precommit
precommit: .uv ## Run pre-commit hooks on staged files and typecheck (install hooks with `pre-commit install`)
	@echo "🛡️  Running pre-commit hooks on staged files..."
	@uv run pre-commit run
	@echo "🧐 Running typecheck (pre-commit)..."
	@$(MAKE) typecheck

.PHONY: typecheck
typecheck: .uv ## Run static type checking with mypy
	@echo "🧐 Running static type checking..."
	@# Ensure dev extras (including types-psutil) are installed before typechecking
	@uv sync --all-extras
	@# Clear mypy cache to avoid cross-Python stdlib stub mismatches (e.g., urllib.parse)
	@rm -rf .mypy_cache || true
	@uv run mypy --clear-cache >/dev/null 2>&1 || true
	@# Run mypy non-interactively to avoid hanging on missing stub prompts
	@uv run mypy flujo/ --install-types --non-interactive

.PHONY: typecheck-ci
typecheck-ci: .uv ## Run static type checking without re-syncing deps (CI)
	@echo "🧐 Running static type checking (CI, no sync)..."
	@# Clear mypy cache to avoid cross-Python stdlib stub mismatches (e.g., urllib.parse)
	@rm -rf .mypy_cache || true
	@uv run mypy --clear-cache >/dev/null 2>&1 || true
	@# Run mypy non-interactively to avoid hanging on missing stub prompts
	@uv run mypy flujo/ --install-types --non-interactive

.PHONY: typecheck-fast
typecheck-fast: .uv ## Typecheck without syncing deps (faster local loop)
	@echo "🧐 Running static type checking (no sync)..."
	@uv run mypy flujo/ --install-types --non-interactive

.PHONY: typecheck-verbose
typecheck-verbose: .uv ## Typecheck with verbose output to trace slow modules
	@echo "🧐 Running mypy in verbose mode..."
	@uv run mypy flujo/ --install-types --non-interactive -v

.PHONY: typecheck-profile
typecheck-profile: .uv ## Profile mypy to locate hotspots (writes .mypy.cprof)
	@echo "🧐 Profiling mypy..."
	@uv run mypy flujo/ --install-types --non-interactive --cprofile .mypy.cprof


# ------------------------------------------------------------------------------
# Testing & Coverage
# ------------------------------------------------------------------------------

# Global environment and safety defaults for pytest
# Note: We disable plugin autoload to control which plugins are loaded
# but we must explicitly load pytest-asyncio for async test support
# For now, let's enable autoload to ensure pytest-asyncio works
# export PYTEST_DISABLE_PLUGIN_AUTOLOAD ?= 1

FAST_KEXPR := not bug_reports and not manual_testing and not scripts
# Architecture/type-safety compliance tests run via `make test-architecture`
# and are excluded from the fast subset by default. Set INCLUDE_ARCHITECTURE_IN_FAST=1 to opt in.
ifndef INCLUDE_ARCHITECTURE_IN_FAST
FAST_KEXPR := $(FAST_KEXPR) and not architecture
endif
TEST_SHARD_MARKERS ?= not slow and not veryslow and not serial and not benchmark
TEST_SHARD_KEXPR ?= $(FAST_KEXPR)

.PHONY: test
test: .uv ## Run all tests via enhanced runner (robust, two-phase)
	@echo "🧪 Running full test suite (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --workers auto --timeout 60 --split-slow --slow-workers 1 --slow-timeout 900

.PHONY: test-architecture
test-architecture: .uv ## Run architecture and type-safety compliance tests
	@echo "🏛️ Running architecture compliance tests..."
	CI=1 uv run pytest tests/architecture -vv --tb=short --durations=0 --color=yes

.PHONY: test-srp
test-srp: .uv ## Run Single Responsibility Principle compliance tests
	@echo "📏 Running SRP compliance tests..."
	CI=1 uv run pytest tests/architecture/test_srp_compliance.py tests/architecture/test_srp_semantic_analysis.py -vv --tb=short --color=yes

.PHONY: test-fast
test-fast: .uv ## Run fast tests in parallel with hang guards (excludes slow, veryslow, serial, and benchmark tests)
	@echo "⚡ Running fast tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --disable-plugin-autoload --markers "not slow and not veryslow and not serial and not benchmark" --kexpr "$(FAST_KEXPR)" --workers auto --timeout 90 || (echo "❌ Some tests failed. Run 'make test-fast-verbose' for detailed output." && exit 1)

.PHONY: test-fast-ci
test-fast-ci: .uv ## Run fast tests via pytest-xdist (CI-friendly)
	@echo "⚡ Running fast tests (CI-friendly pytest)..."
	CI=1 uv run pytest tests/ -n auto -p no:randomly -m "not slow and not veryslow and not serial and not benchmark" -k "$(FAST_KEXPR)"

.PHONY: test-fast-verbose
test-fast-verbose: .uv ## Run fast tests with verbose output for debugging
	@echo "🔍 Running fast tests with verbose output (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --disable-plugin-autoload --markers "not slow and not veryslow and not serial and not benchmark" --kexpr "$(FAST_KEXPR)" --workers auto --timeout 90 --tb --pytest-args=-vv

.PHONY: test-fast-serial
test-fast-serial: .uv ## Run fast tests serially with hang guard (debug parallel issues)
	@echo "🔧 Running fast tests serially (enhanced runner)..."
	CI=1 FLUJO_TEST_FORCE_EXIT=1 uv run python scripts/run_targeted_tests.py --full-suite --disable-plugin-autoload --markers "not slow and not veryslow and not serial and not benchmark" --kexpr "$(FAST_KEXPR)" --workers 1 --timeout 90 --faulthandler-timeout 60

.PHONY: test-fast-conservative
test-fast-conservative: .uv ## Run fast tests with conservative parallelism (2 workers + hang guard)
	@echo "🐌 Running fast tests with conservative parallelism (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --disable-plugin-autoload --markers "not slow and not veryslow and not serial and not benchmark" --kexpr "$(FAST_KEXPR)" --workers 2 --timeout 90 --faulthandler-timeout 60

.PHONY: test-robust
test-robust: .uv ## Run tests with enhanced robustness and monitoring
	@echo "🛡️ Running robust test suite (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "not slow and not veryslow and not serial and not benchmark" --kexpr "$(FAST_KEXPR)" --workers auto --timeout 90 --pytest-args="--maxfail=5 -q"

.PHONY: test-stress
test-stress: .uv ## Run stress tests to identify resource issues
	@echo "💪 Running stress tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "stress" --workers 1 --timeout 300 --tb

.PHONY: test-memory
test-memory: .uv ## Run memory leak detection tests
	@echo "🧠 Running memory leak tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "memory" --workers 1 --timeout 120 --tb

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
test-slow: .uv ## Run slow tests serially
	@echo "🐌 Running slow tests serially (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "slow or veryslow or serial or benchmark" --workers 1 --timeout 180 --tb

.PHONY: test-ultra-slow
test-ultra-slow: .uv ## Run ultra-slow stress tests (>30s each) separately
	@echo "⚠️ Running ultra-slow tests (>30s each)"
	@echo "These tests are excluded from regular CI due to long execution time"
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "ultra_slow" --workers 1 --timeout 300 --tb --no-split-slow

.PHONY: test-parallel
test-parallel: .uv ## Run all tests in parallel (excludes serial tests)
	@echo "🚀 Running tests in parallel (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "not serial" --workers auto --timeout 60

.PHONY: test-unit
test-unit: .uv ## Run unit tests only
	@echo "🧪 Running unit tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py tests/unit --workers auto --timeout 60

.PHONY: test-integration
test-integration: .uv ## Run integration tests only
	@echo "🔗 Running integration tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py tests/integration --workers auto --timeout 60

.PHONY: test-bench
test-bench: .uv ## Run benchmark tests only
	@echo "📊 Running benchmark tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py tests/benchmarks --workers 1 --timeout 300 --tb

.PHONY: test-e2e
test-e2e: .uv ## Run end-to-end tests only
	@echo "🌐 Running end-to-end tests (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py tests/e2e --workers auto --timeout 120 --tb

.PHONY: testcov
testcov: .uv ## Run tests and generate an HTML coverage report
	@echo "🧪 Running tests with coverage..."
	@uv run coverage run --source=flujo -m pytest tests/
	@uv run coverage html
	@echo "\n✅ Coverage report generated in 'htmlcov/'. Open htmlcov/index.html to view."

.PHONY: testcov-fast
testcov-fast: .uv ## Run fast tests with coverage in parallel
	@echo "⚡ Running fast tests with coverage in parallel..."
	@uv run pytest --cov=flujo --cov-report=html tests/ -m "not slow and not veryslow and not serial and not benchmark" -n auto
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
	@uv run pytest tests/ -m "not slow and not veryslow and not serial and not benchmark" --collect-only -q | tail -1
	@echo ""
	@echo "Slow/Benchmark tests (excluded):"
	@uv run pytest tests/ -m "slow or serial or benchmark" --collect-only -q | tail -1
	@echo ""
	@echo "Test categories:"
	@echo "  • Unit tests: $(shell find tests/unit -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • Integration tests: $(shell find tests/integration -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • Benchmark tests: $(shell find tests/benchmarks -name "*.py" | wc -l | tr -d ' ') files"
	@echo "  • E2E tests: $(shell find tests/e2e -name "*.py" | wc -l | tr -d ' ') files"

.PHONY: test-workers
test-workers: .uv ## Show optimal worker count recommendations for your system
	@uv run python scripts/check_optimal_workers.py

.PHONY: test-failing
test-failing: .uv ## Run only failing tests to identify issues
	@echo "🔍 Running only failing tests..."
	CI=1 uv run pytest tests/ -m "not slow and not veryslow and not serial and not benchmark" --lf -v

.PHONY: test-slow-marked
test-slow-marked: .uv ## Show which tests are marked as slow
	@echo "🐌 Tests marked as slow:"
	@uv run pytest tests/ -m "slow" --collect-only -q

.PHONY: test-benchmark-marked
test-benchmark-marked: .uv ## Show which tests are marked as benchmark
	@echo "📊 Tests marked as benchmark:"
	@uv run pytest tests/ -m "benchmark" --collect-only -q

# ------------------------------------------------------------------------------
# Advanced Test Execution & Debugging Targets
# ------------------------------------------------------------------------------

.PHONY: test-hang-guard
test-hang-guard: .uv ## Find what's hanging: hard timeouts + stack dumps on stall (enhanced runner)
	@echo "🧯 Hang guard: hard timeouts + stack dumps (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --workers 4 --timeout 60 --faulthandler-timeout 60 --tb

.PHONY: test-top-slowest
test-top-slowest: .uv ## Show the slowest test offenders (fast subset)
	@echo "🐢 Top slow tests (running fast subset for quick analysis):"
	@echo "Reporting top slow tests with durations..."
	CI=1 uv run pytest tests/ \
		-m "not slow and not veryslow and not serial and not benchmark" \
		-q -k "" -p pytest_asyncio

.PHONY: test-loop
test-loop: .uv ## Loop on first failure (tight local debug loop)
	@echo "🔁 Loop on first failure (xdist -f)..."
	CI=1 uv run pytest tests/ -f -x -q

.PHONY: test-no-leaks
test-no-leaks: .uv ## Run a subset with RuntimeWarning/ResourceWarning as errors (xdist disabled)
	@echo "🧹 Running no-leaks check (warnings as errors, serial)..."
	CI=1 FLUJO_NO_LEAKS=1 uv run pytest \
		tests/integration/test_pipeline_hooks.py::test_pipeline_aborts_gracefully_from_hook \
		tests/cli/test_create_sanitizer.py::test_create_moves_content_input_into_params_for_fs_write \
		tests/state/test_sqlite_async_bridge.py \
		tests/unit/application/test_runner_shutdown.py \
		tests/integration/test_async_runner_detection.py \
		-n 0 -q \
		-W error::RuntimeWarning -W error::ResourceWarning


.PHONY: test-changed
test-changed: .uv ## Run only tests impacted by recent changes (testmon)
	@echo "🧩 Running tests impacted by recent changes (testmon)..."
	CI=1 uv run pytest tests/ --testmon -q \
		-p pytest_testmon

.PHONY: test-shard
test-shard: .uv ## Run deterministic shard (for CI: make test-shard SHARD_INDEX=0 SHARD_COUNT=4)
	@echo "🧱 Running shard $(SHARD_INDEX)/$(SHARD_COUNT) (0-indexed input) ..."
	@GROUP_IDX=$$(( $(SHARD_INDEX) + 1 )); \
	CI=1 uv run pytest tests/ \
		--splits $(SHARD_COUNT) --group $$GROUP_IDX \
		-m "$(TEST_SHARD_MARKERS)" -k "$(TEST_SHARD_KEXPR)" \
		-n auto --dist=loadfile \
		-p pytest_split $(if $(TEST_SHARD_COVERAGE),--cov=flujo --cov-report=,) -q

.PHONY: test-serial
test-serial: .uv ## Run serial/slow/benchmark tests in isolation (fixes parallel issues)
	@echo "🧵 Running serial/slow/benchmark tests in isolation (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --markers "slow or serial or benchmark" --workers 1 --timeout 180 --tb

.PHONY: test-flake-harden
test-flake-harden: .uv ## Auto-rerun flaky test failures up to 2x (CI-only recommended)
	@echo "🧷 Rerunning flaky test failures up to 2x..."
	CI=1 uv run pytest tests/ --reruns 2 --reruns-delay 1 -q \
		-p pytest_rerunfailures

.PHONY: test-deadfixtures
test-deadfixtures: .uv ## Find unused/overlapping fixtures that slow collection/execution
	@echo "🧟 Finding dead fixtures..."
	CI=1 uv run pytest tests/ --deadfixtures -q \
		-p pytest_deadfixtures

.PHONY: test-profile
test-profile: .uv ## Profile test execution to find hot spots (cProfile)
	@echo "📊 Profiling test execution..."
	CI=1 uv run python -m cProfile -o test_profile.pstats -m pytest tests/ -q

.PHONY: test-random-order
test-random-order: .uv ## Run tests in random order to reveal order dependencies
	@echo "🎲 Running tests in random order..."
	CI=1 uv run pytest tests/ --randomly-seed=auto -n auto \
		-p pytest_randomly

.PHONY: test-forked
test-forked: .uv ## Run tests serially for process isolation (pytest-forked removed due to CVE)
	@echo "🔀 Running tests serially for isolation (pytest-forked unavailable)..."
	@echo "   Note: pytest-forked was removed due to CVE PYSEC-2022-42969 in 'py' package."
	@echo "   Using serial execution with strict timeouts as alternative."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --workers 1 --timeout 60 --faulthandler-timeout 60 --tb

.PHONY: test-timeout-strict
test-timeout-strict: .uv ## Run with strict timeouts for debugging hangs (enhanced runner)
	@echo "⏰ Running with strict timeouts (enhanced runner)..."
	CI=1 uv run python scripts/run_targeted_tests.py --full-suite --workers auto --timeout 30 --faulthandler-timeout 60 --tb

.PHONY: test-collection-only
test-collection-only: .uv ## Check test collection without running (fast validation)
	@echo "📋 Checking test collection..."
	uv run pytest tests/ --collect-only -q

.PHONY: test-analyze-performance
test-analyze-performance: .uv ## Comprehensive performance analysis
	@echo "📊 Comprehensive test performance analysis..."
	@echo "1. Test collection time..."
	@time uv run pytest tests/ --collect-only -q > /dev/null
	@echo ""
	@echo "2. Top slow tests (unit tests only for speed)..."
	@CI=1 uv run pytest tests/unit/ --durations=20 --durations-min=0.1 -q -k "" \
		-p pytest_asyncio
	@echo ""
	@echo "3. Memory usage check..."
	@CI=1 uv run pytest tests/ -m "memory" --timeout=60 -q \
		-p pytest_asyncio

.PHONY: test-slow-analysis
test-slow-analysis: .uv ## Quick analysis of slow tests without running them
	@echo "🐌 Analyzing slow tests without execution..."
	@echo "1. Tests marked as slow:"
	@uv run pytest tests/ -m "slow" --collect-only -q
	@echo ""
	@echo "2. Tests marked as benchmark:"
	@uv run pytest tests/ -m "benchmark" --collect-only -q
	@echo ""
	@echo "3. Tests marked as serial:"
	@uv run pytest tests/ -m "serial" --collect-only -q
	@echo ""
	@echo "4. Tests with timeout markers:"
	@uv run pytest tests/ -m "timeout" --collect-only -q

.PHONY: test-quick-check
test-quick-check: .uv ## Quick test to verify pytest-asyncio is working (enhanced runner)
	@echo "🔍 Quick async test via enhanced runner..."
	CI=1 uv run python scripts/run_targeted_tests.py tests/unit/test_validation.py::test_base_validator_initialization --timeout 60 --tb --fail-fast --pytest-args=-vv


# ------------------------------------------------------------------------------
# Package Building
# ------------------------------------------------------------------------------

.PHONY: package
package: .uv ## Build package distribution files
	@echo "📦 Building package distribution..."
	@uv run python3 -m build
	@echo "\n✅ Package built in dist/ directory."


# ------------------------------------------------------------------------------
# Documentation
# ------------------------------------------------------------------------------

.PHONY: docs-build
docs-build: .uv ## Build documentation site with MkDocs
	@echo "📚 Building docs..."
	@uv sync --all-extras
	@uv run mkdocs build
	@echo "\n✅ Docs built in 'site/'"

.PHONY: docs-check
docs-check: ## Check docs for broken relative links (offline)
	@echo "🔗 Checking docs for broken links..."
	@python3 scripts/check_docs_links.py

.PHONY: docs-ci
docs-ci: docs-build docs-check ## Build docs and run link checks
	@echo "\n✅ Docs CI checks passed"


# ------------------------------------------------------------------------------
# All-in-one & Help
# ------------------------------------------------------------------------------

ifdef FAST_ALL
TEST_GATE_TARGET := test-fast-ci
TYPECHECK_TARGET := typecheck-fast
else
TEST_GATE_TARGET := test
TYPECHECK_TARGET := typecheck
endif

ifdef SKIP_ARCHITECTURE_TESTS
ALL_QUALITY_TARGETS := format lint $(TYPECHECK_TARGET) $(TEST_GATE_TARGET)
else
ALL_QUALITY_TARGETS := format lint $(TYPECHECK_TARGET) $(TEST_GATE_TARGET) test-architecture test-srp
endif

.PHONY: all
all: $(ALL_QUALITY_TARGETS) ## Run all quality checks (format, lint, typecheck, test, test-architecture)
	@echo "\n✅ make all completed (format, lint, typecheck, $(TEST_GATE_TARGET), test-architecture). Ready to push."

.PHONY: help
help: ## ✨ Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo ""
	@echo "🚀 Development & Setup:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ && /install|sync|format|lint|typecheck/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "🧪 Core Testing:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ && /test[^-]/ && !/test-analyze|test-deadfixtures|test-profile|test-random-order|test-forked|test-timeout-strict|test-collection-only/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "🔧 Advanced Testing & Debugging:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ && /test-analyze|test-deadfixtures|test-profile|test-random-order|test-forked|test-timeout-strict|test-collection-only|test-hang-guard|test-top-slowest|test-loop|test-changed|test-shard|test-serial|test-flake-harden/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
	@echo ""
	@echo "📦 Package & Quality:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ && /package|all/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
