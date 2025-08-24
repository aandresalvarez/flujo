 

# 🧪 Controlled Test Runner — User Manual

This document describes how to use the **enhanced pytest wrapper** (`scripts/run_targeted_tests.py`) to run, debug, and analyze tests in large suites (5k+ tests).

It adds **strict timeouts, parallel isolation, structured logs, and fail-fast** execution — solving common problems with hanging tests and hard-to-find failures.

---

## 🚀 Quick Start

Run the full suite with default settings:

```bash
# Use the project virtualenv to ensure plugins are available
.venv/bin/python scripts/run_targeted_tests.py --full-suite
```

Run only specific tests:

```bash
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result
```

---

## ⚙️ Command-Line Options

### Test Selection

* `nodeids [NODEID ...]`
  Run specific tests by nodeid or file path.
  Example:

  ```bash
  python scripts/run_targeted_tests.py tests/unit/test_example.py::test_function
  ```

* `--full-suite`
  Discover and run the full test suite automatically (honors `-m` and `-k` filters).

* `--markers "EXPR"`
  Pytest `-m` expression to filter tests.
  Example: `--markers "not slow and not serial"`

* `--kexpr "EXPR"`
  Pytest `-k` expression for substring or boolean match.
  Example: `--kexpr "pipeline and not benchmark"`

* `--limit N`
  Limit the number of discovered tests (useful for sampling/debugging).

---

### Timeouts & Safety

* `--timeout N`
  Per-test timeout in seconds (default: `60`). Enforced by **pytest-timeout**.

* `--outer-timeout N`
  Hard failsafe timeout per test subprocess. Defaults to `3 × timeout`.
  (Ensures hung pytest processes get killed.)

---

### Output & Reporting

* `--tb`
  Show full tracebacks (default: hidden for brevity).

* `--show-only-failures`
  Detailed log only includes failing tests, not all.

* Logs are written to `output/`:

  * Full log: `controlled_test_run_YYYYMMDD_HHMMSS.log`
  * Failure summary: `failure_summary_YYYYMMDD_HHMMSS.txt`
  * JSON results: `results_YYYYMMDD_HHMMSS.json`

---

### Execution Control

* `--fail-fast`
  Stop immediately after the first failure. Prints failure details.

* `--workers N | auto`
  Run multiple tests in parallel (isolated subprocess per test).

  * `--workers auto` → half your CPUs, capped at 8
  * `--workers 4` → 4 concurrent tests
  * Default: `1` (serial)

* `--pytest-args "ARGS"`
  Forward additional arguments to pytest.
  Example:

  ```bash
  --pytest-args "--maxfail=1 -q -s"
  ```

* `--disable-plugin-autoload`
  Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to prevent slow/unwanted plugins from auto-loading. Useful in CI.
  When set, the runner explicitly loads the `pytest-timeout` and `pytest-asyncio` plugins.

* `--faulthandler-timeout N`
  Enable pytest faulthandler stack dumps if a test process stalls for `N` seconds.
  Leave unset if your pytest version does not support this option.

* `--split-slow` / `--no-split-slow`
  For `--full-suite` runs, execute non-slow tests first (in parallel), then slow tests in a separate phase with reduced workers. Enabled by default.

* `--slow-workers N | auto`
  Workers used for the slow phase. Defaults to `1`. Use `auto` to use half the CPUs.

* `--slow-timeout N`
  Per-test timeout for the slow phase. Defaults to `2 × --timeout`.

---

## 📊 Example Workflows

### Run all fast tests with guards

```bash
.venv/bin/python scripts/run_targeted_tests.py \
  --full-suite \
  --markers "not slow and not serial and not benchmark" \
  --timeout 60 \
  --workers auto
```

### Debug a flaky test with verbose output

```bash
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_pipeline_runner.py::test_timeout_and_redirect_loop_detection \
  --timeout 120 --tb
```

### Sample 100 tests for a quick smoke run

```bash
.venv/bin/python scripts/run_targeted_tests.py --full-suite --limit 100 --workers 4
```

### Stop immediately on first failure

```bash
.venv/bin/python scripts/run_targeted_tests.py --full-suite --fail-fast
```

### Run all tests optimally (recommended)

```bash
.venv/bin/python scripts/run_targeted_tests.py \
  --full-suite \
  --workers auto \
  --timeout 60 \
  --split-slow \
  --slow-workers 1 \
  --slow-timeout 120
```

---

## 🛠️ Troubleshooting

- Duplicate plugin error (pytest-timeout):
  If you see a message like "Plugin already registered under a different name: timeout=pytest_timeout",
  you are likely loading plugins via autoload and explicitly via `-p`. The runner now avoids that duplication.
  If you pass `--disable-plugin-autoload`, it will explicitly load the timeout and asyncio plugins.

- Unknown option `--faulthandler-timeout`:
  Some pytest versions do not support this option. Omit `--faulthandler-timeout` (or leave the runner’s flag unset).

- Non-test files getting collected:
  Full-suite discovery targets the official `tests/` directories. If a stray file outside this folder
  matches the `test_*.py` pattern, use `--kexpr` to exclude it (e.g., `--kexpr "not bug_reports"`).

- Benchmarks or sustained-load tests fail under parallel load:
  Use the default two-phase run (`--split-slow`) so slow tests run serialized with higher timeouts.
  Tune with `--slow-workers` and `--slow-timeout` as needed.

- HITL or stateful (SQLite-backed) tests time out or contend when run in fast mode:
  Mark them with `@pytest.mark.slow` and `@pytest.mark.serial`, or apply module-level `pytestmark`.
  The fast subset is marker-based (excludes `slow/serial/benchmark`). Avoid kexpr-based exclusions.

---

## ✅ Exit Codes

* `0` → All tests passed
* `2` → At least one test failed
* `3` → At least one test timed out
* `4` → Execution error in runner or test subprocess

---

## 🧰 Tips & Best Practices

* Mark genuine long-running tests with `@pytest.mark.timeout(N)` to override default.
* Use `--workers auto` for day-to-day runs; fall back to serial (`--workers 1`) if debugging order-dependent issues.
* Use `--disable-plugin-autoload` to skip noisy/unneeded pytest plugins that slow collection.
* Review `output/failure_summary_*.txt` for quick triage.
* Parse `results_*.json` in scripts/CI to surface failures in dashboards.

 
