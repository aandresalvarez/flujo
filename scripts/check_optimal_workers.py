#!/usr/bin/env python3
"""
Helper script to determine optimal worker count for test execution.

This script analyzes your system's CPU count and provides recommendations
for the maximum number of workers you can use for parallel test execution.
"""

import os
import sys
from typing import Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_cpu_count() -> int:
    """Get the number of CPU cores available."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_auto_workers() -> int:
    """Calculate auto workers using the same logic as run_targeted_tests.py."""
    # Check if we're in CI environment
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")

    if is_ci:
        # CI environments use fixed conservative count
        ci_workers = os.environ.get("CI_TEST_WORKERS")
        if ci_workers:
            try:
                return max(1, int(ci_workers))
            except (ValueError, TypeError):
                pass
        # Default CI worker count (matches GitHub Actions -n 2)
        return 2

    # Local development: adapt to CPU count
    n = get_cpu_count()
    # Keep it conservative to avoid overloading local dev
    return max(1, min(8, (n + 1) // 2))


def get_memory_info() -> Optional[dict]:
    """Get memory information if psutil is available."""
    if not HAS_PSUTIL:
        return None
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": mem.percent,
        }
    except Exception:
        return None


def main() -> int:
    cpu_count = get_cpu_count()
    auto_workers = get_auto_workers()
    memory_info = get_memory_info()
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")

    print("=" * 60)
    print("System Resource Analysis for Test Workers")
    print("=" * 60)
    print()
    if is_ci:
        print("⚠️  CI Environment Detected")
        print()
    print(f"CPU Cores Available: {cpu_count}")
    if memory_info:
        print(f"Total Memory: {memory_info['total_gb']:.1f} GB")
        print(f"Available Memory: {memory_info['available_gb']:.1f} GB")
        print(f"Memory Used: {memory_info['percent_used']:.1f}%")
    print()
    print("-" * 60)
    print("Worker Count Recommendations:")
    print("-" * 60)
    print()
    if is_ci:
        print(f"Auto (CI fixed): {auto_workers} workers")
        print("  CI uses fixed count for consistency across runners")
        ci_override = os.environ.get("CI_TEST_WORKERS")
        if ci_override:
            print(f"  Override via CI_TEST_WORKERS: {ci_override}")
        else:
            print("  Default: 2 workers (matches GitHub Actions)")
        print()
        print("Note: CI environments use fixed worker counts to ensure")
        print("      consistent, reproducible test runs across different")
        print("      runner configurations.")
    else:
        print(f"Auto (conservative): {auto_workers} workers")
        print("  Formula: min(8, (CPU_count + 1) // 2)")
        print()
        print(f"Maximum (aggressive): {cpu_count} workers")
        print("  Uses all CPU cores - may overload system")
        print()
        print(f"Balanced (recommended): {min(cpu_count, auto_workers + 2)} workers")
        print("  Slightly above auto for better throughput")
        print()
        print(f"Safe maximum: {min(cpu_count, 8)} workers")
        print("  Capped at 8 to avoid system overload")
    print()
    print("-" * 60)
    print("CI vs Local Differences:")
    print("-" * 60)
    print()
    if is_ci:
        # Calculate what local would be
        local_auto = max(1, min(8, (cpu_count + 1) // 2))
        print("Local Development (when CI=0):")
        print("  • Uses --workers auto → adapts to your CPU count")
        print(f"  • Would use: {local_auto} workers (from {cpu_count} cores)")
        print("  • Can use --workers 8 for better throughput")
        print()
        print("CI Environment (current):")
        print(f"  • Uses --workers auto → fixed at {auto_workers} workers (for consistency)")
        print("  • Can override with CI_TEST_WORKERS environment variable")
        print("  • Different CI runners may have different CPU counts,")
        print("    but auto ensures consistent behavior")
    else:
        print("Local Development (current):")
        print("  • Uses --workers auto → adapts to your CPU count")
        print(f"  • Your system: {auto_workers} workers (from {cpu_count} cores)")
        print("  • Can use --workers 8 for better throughput")
        print()
        print("CI Environment (when CI=1):")
        print("  • Uses --workers auto → fixed at 2 workers (for consistency)")
        print("  • Can override with CI_TEST_WORKERS environment variable")
        print("  • Different CI runners may have different CPU counts,")
        print("    but auto ensures consistent behavior")
    print()
    print("-" * 60)
    print("Usage Examples:")
    print("-" * 60)
    print()
    print("  # Use auto (recommended for most cases):")
    print("  make test-fast  # Uses --workers auto")
    print()
    if not is_ci:
        print("  # Use specific count:")
        print(f"  uv run python scripts/run_targeted_tests.py --workers {auto_workers} ...")
        print()
        print("  # Test with maximum (may be slower due to overhead):")
        print(f"  uv run python scripts/run_targeted_tests.py --workers {cpu_count} ...")
    else:
        print("  # Override CI worker count:")
        print("  CI_TEST_WORKERS=4 make test-fast")
    print()
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
