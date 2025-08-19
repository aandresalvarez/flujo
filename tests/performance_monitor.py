#!/usr/bin/env python3
"""
Test Performance Monitor

This script helps identify slow tests and provides performance insights.
Run with: python tests/performance_monitor.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def run_test_with_timing(test_path: str = "tests/") -> Tuple[float, str]:
    """Run a subset of tests with timing and return execution time and output.

    Defaults to running the fast subset to avoid CI timeouts. You can override
    the selection with the FLUJO_PERF_MARKERS env var.
    """
    start_time = time.time()

    # Use a fast subset by default to keep CI under time limits
    # Parse timeout robustly and fall back to 600 on invalid values
    raw_timeout = os.getenv("FLUJO_PERF_TIMEOUT", "").strip()
    try:
        timeout_sec = int(raw_timeout) if raw_timeout else 600
    except ValueError:
        timeout_sec = 600

    # Normalize markers: default to fast subset if unset/empty after strip
    default_markers = "not slow and not serial and not benchmark"
    raw_markers = os.getenv("FLUJO_PERF_MARKERS", "")
    markers = raw_markers.strip() or default_markers

    use_uv = shutil.which("uv") is not None
    base_cmd = ["uv", "run", "pytest"] if use_uv else ["python", "-m", "pytest"]
    cmd_parts = base_cmd + [test_path, "-v", "--tb=no", "--durations=25"]
    if markers.strip():
        cmd_parts += ["-m", markers]
    cmd = cmd_parts

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = result.stdout

        # Fallback: if nothing parsed, try without -v formatting quirks
        if not parse_test_output(stdout):
            fallback_cmd = base_cmd + [test_path, "--tb=no", "--durations=25"]
            if markers.strip():
                fallback_cmd += ["-m", markers]
            result_fb = subprocess.run(
                fallback_cmd,
                capture_output=True,
                text=True,
                timeout=max(60, timeout_sec // 4),  # quick fallback
            )
            stdout = result_fb.stdout

        return time.time() - start_time, stdout
    except subprocess.TimeoutExpired:
        return timeout_sec, "Tests timed out after configured limit"


def parse_test_output(output: str) -> List[Dict[str, str]]:
    """Parse pytest output to extract test names and durations."""
    tests = []
    lines = output.split("\n")

    # Look for the durations section at the end
    in_durations = False
    for line in lines:
        if line.startswith("=") and "durations" in line.lower():
            in_durations = True
            continue
        elif in_durations and line.startswith("="):
            break
        elif in_durations and line.strip():
            # Parse duration line like "1.23s call     tests/unit/test_example.py::test_function"
            parts = line.split()
            if len(parts) >= 3 and "s" in parts[0]:
                try:
                    duration_str = parts[0].replace("s", "")
                    duration = float(duration_str)
                    test_name = " ".join(parts[2:])
                    tests.append(
                        {
                            "name": test_name,
                            "duration": duration,
                            "status": "PASSED",  # Assume passed if in durations
                        }
                    )
                except ValueError:
                    continue

    # If no durations found, try parsing individual test lines
    if not tests:
        for line in lines:
            if line.startswith("tests/") and "::" in line and "PASSED" in line:
                # Extract test path and duration
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    # Look for duration in format like "PASSED [ 0.12s ]"
                    for part in parts:
                        if "s]" in part and "[" in part:
                            duration_str = part.strip("[]").replace("s", "")
                            try:
                                duration = float(duration_str)
                                tests.append(
                                    {"name": test_name, "duration": duration, "status": "PASSED"}
                                )
                            except ValueError:
                                continue

    return tests


def analyze_test_performance(tests: List[Dict[str, str]]) -> Dict:
    """Analyze test performance and return statistics."""
    if not tests:
        return {}

    durations = [t["duration"] for t in tests]
    total_time = sum(durations)
    avg_time = total_time / len(durations)

    # Sort by duration (slowest first)
    sorted_tests = sorted(tests, key=lambda x: x["duration"], reverse=True)

    # Find slow tests (top 10% or > 1 second)
    slow_threshold = max(1.0, sorted(durations)[int(len(durations) * 0.9)])
    slow_tests = [t for t in tests if t["duration"] > slow_threshold]

    return {
        "total_tests": len(tests),
        "total_time": total_time,
        "average_time": avg_time,
        "slowest_tests": sorted_tests[:10],
        "slow_tests": slow_tests,
        "slow_threshold": slow_threshold,
        "test_categories": categorize_tests(tests),
    }


def categorize_tests(tests: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize tests by type."""
    categories = {
        "unit": [],
        "integration": [],
        "benchmark": [],
        "e2e": [],
        "security": [],
        "smoke": [],
        "processors": [],
    }

    for test in tests:
        name = test["name"]
        if "unit/" in name:
            categories["unit"].append(test)
        elif "integration/" in name:
            categories["integration"].append(test)
        elif "benchmarks/" in name:
            categories["benchmark"].append(test)
        elif "e2e/" in name:
            categories["e2e"].append(test)
        elif "security/" in name:
            categories["security"].append(test)
        elif "smoke/" in name:
            categories["smoke"].append(test)
        elif "processors/" in name:
            categories["processors"].append(test)

    return categories


def print_performance_report(stats: Dict) -> None:
    """Print a formatted performance report."""
    if not stats:
        print("❌ No test data available")
        return

    print("📊 Test Performance Report")
    print("=" * 50)
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Average Time: {stats['average_time']:.3f}s")
    print(f"Slow Threshold: {stats['slow_threshold']:.3f}s")
    print()

    # Category breakdown
    print("📁 Test Categories:")
    for category, tests in stats["test_categories"].items():
        if tests:
            total_time = sum(t["duration"] for t in tests)
            print(f"  {category.capitalize()}: {len(tests)} tests, {total_time:.2f}s")
    print()

    # Slowest tests
    print("🐌 Top 10 Slowest Tests:")
    for i, test in enumerate(stats["slowest_tests"][:10], 1):
        print(f"  {i:2d}. {test['name']} ({test['duration']:.3f}s)")
    print()

    # Recommendations
    print("💡 Performance Recommendations:")
    if stats["slow_tests"]:
        print(
            f"  • {len(stats['slow_tests'])} tests are slower than {stats['slow_threshold']:.3f}s"
        )
        print("  • Consider marking slow tests with @pytest.mark.slow")
        print("  • Use 'make test-fast' for faster feedback during development")
    else:
        print("  • All tests are performing well!")

    print("  • Use 'make test-parallel' for parallel execution")
    print("  • Use 'make test-unit' for quick unit test feedback")


def main():
    """Main function to run performance analysis."""
    print("🔍 Analyzing test performance...")
    print()

    # Run tests and capture timing
    duration, output = run_test_with_timing()

    # Parse test results
    tests = parse_test_output(output)

    if not tests:
        print("❌ No test results found. Make sure tests are passing.")
        sys.exit(1)

    # Analyze performance
    stats = analyze_test_performance(tests)

    # Print report
    print_performance_report(stats)

    # Save detailed results
    output_file = Path("test_performance_report.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "total_duration": duration,
                "statistics": stats,
                "all_tests": tests,
            },
            f,
            indent=2,
        )

    print(f"\n📄 Detailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
