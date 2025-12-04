#!/usr/bin/env python3
"""Pre-commit hook: Detect common test isolation anti-patterns.

This hook catches patterns that cause "passes in PR, fails in main" scenarios:

1. TIGHT_THRESHOLD: Performance assertions with thresholds < 1.0s (too tight for CI)
2. MISSING_SERIAL: Tests checking `current_state` in scratchpad without @pytest.mark.serial
3. CONFLICTING_MARKERS: Tests with both @pytest.mark.slow and @pytest.mark.fast

Run manually: python scripts/hooks/check_test_isolation.py
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
from typing import Final

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]

# Only check test files
TEST_DIRS: Final = {"tests"}
EXCLUDE_DIRS: Final[set[str]] = {".venv", "site-packages", ".git", "__pycache__"}

# Pattern 1: Tight performance thresholds (< 1.0s)
# Matches: assert time < 0.01, assert execution_time < 0.1, assert elapsed < 0.5
TIGHT_THRESHOLD_PATTERN: Final = re.compile(
    r"assert\s+\w*(?:time|elapsed|duration)\w*\s*<\s*(0\.[0-9]+)",
    re.IGNORECASE,
)

# Pattern 2: current_state check without serial marker
CURRENT_STATE_CHECK_PATTERN: Final = re.compile(
    r'scratchpad.*(?:get|\.)\s*[\[\(]?\s*["\']current_state["\']',
    re.IGNORECASE,
)
SERIAL_MARKER_PATTERN: Final = re.compile(
    r"@pytest\.mark\.serial|pytestmark.*serial",
    re.MULTILINE,
)

# Pattern 3: Conflicting slow/fast markers in same file
SLOW_MARKER_PATTERN: Final = re.compile(r"@pytest\.mark\.slow|pytestmark.*slow")
FAST_MARKER_PATTERN: Final = re.compile(r"@pytest\.mark\.fast")


def iter_test_files() -> list[pathlib.Path]:
    """Yield test files from tests/ directory."""
    results: list[pathlib.Path] = []
    for test_dir in TEST_DIRS:
        test_path = REPO_ROOT / test_dir
        if not test_path.exists():
            continue
        for root, dirs, files in os.walk(test_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                if fname.startswith("test_") and fname.endswith(".py"):
                    results.append(pathlib.Path(root, fname))
    return results


def check_tight_thresholds(file: pathlib.Path, text: str) -> list[str]:
    """Check for tight performance thresholds (< 1.0s)."""
    issues: list[str] = []

    # Skip files in benchmarks directory - they're already excluded from fast runs
    # and intentionally measure tight timing for logging
    if "benchmarks" in str(file):
        return issues

    # Skip files with "performance" in name - these are often benchmarks
    if "performance" in file.name.lower():
        return issues

    for match in TIGHT_THRESHOLD_PATTERN.finditer(text):
        threshold = float(match.group(1))
        if threshold < 1.0:
            # Find line number
            line_num = text[: match.start()].count("\n") + 1
            issues.append(
                f"{file}:{line_num}: Tight performance threshold ({threshold}s). "
                f"Use >= 1.0s or convert to benchmark-only pattern."
            )
    return issues


def check_missing_serial_marker(file: pathlib.Path, text: str) -> list[str]:
    """Check for current_state checks without @pytest.mark.serial."""
    issues: list[str] = []

    # Only check integration tests - unit tests with isolated contexts are safe
    # The concern is integration tests where state machine context may persist
    if "integration" not in str(file):
        return issues

    # Skip files that only test state machine *policy* logic (not full integration)
    if "policy" in file.name.lower():
        return issues

    has_current_state_check = CURRENT_STATE_CHECK_PATTERN.search(text) is not None
    has_serial_marker = SERIAL_MARKER_PATTERN.search(text) is not None

    if has_current_state_check and not has_serial_marker:
        issues.append(
            f"{file}: Checks 'current_state' in scratchpad but missing @pytest.mark.serial. "
            f"Add marker to prevent race conditions under xdist."
        )
    return issues


def check_conflicting_markers(file: pathlib.Path, text: str) -> list[str]:
    """Check for conflicting @pytest.mark.slow and @pytest.mark.fast in same file."""
    issues: list[str] = []

    has_slow = SLOW_MARKER_PATTERN.search(text) is not None
    has_fast = FAST_MARKER_PATTERN.search(text) is not None

    if has_slow and has_fast:
        issues.append(
            f"{file}: Has both @pytest.mark.slow and @pytest.mark.fast markers. "
            f"Remove conflicting markers (typically remove @pytest.mark.fast)."
        )
    return issues


def main() -> int:
    """Run all test isolation checks."""
    all_issues: list[str] = []

    for file in iter_test_files():
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Skip files with explicit opt-out comment
        if "# noqa: test-isolation" in text:
            continue

        all_issues.extend(check_tight_thresholds(file, text))
        all_issues.extend(check_missing_serial_marker(file, text))
        all_issues.extend(check_conflicting_markers(file, text))

    if all_issues:
        print("\n❌ Test isolation issues detected:\n", file=sys.stderr)
        for issue in all_issues:
            print(f"  • {issue}", file=sys.stderr)
        print(
            "\nSee docs/testing.md 'CI Stability & Test Isolation' section for guidance.",
            file=sys.stderr,
        )
        return 1

    print("✅ No test isolation issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
