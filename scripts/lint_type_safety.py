#!/usr/bin/env python3
"""Fail CI when new `Any` or `cast()` usages are introduced in core/DSL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "scripts/type_safety_baseline.json"

TARGETS = {
    "core": ROOT / "flujo" / "application" / "core",
    "dsl": ROOT / "flujo" / "domain" / "dsl",
}


def _count_token(path: Path, token: str) -> int:
    total = 0
    for file_path in path.rglob("*.py"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        total += len(re.findall(re.escape(token), text))
    return total


def _load_baseline() -> Dict[str, Dict[str, int]]:
    if not BASELINE_PATH.exists():
        sys.stderr.write(
            "Missing type-safety baseline. Commit scripts/type_safety_baseline.json.\n"
        )
        sys.exit(1)
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Unable to read baseline: {exc}\n")
        sys.exit(1)


def _format_delta(current: int, baseline: int) -> str:
    """Format the delta between current and baseline values."""
    delta = current - baseline
    if delta > 0:
        return f"+{delta}"
    elif delta < 0:
        return str(delta)
    return "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Type-safety lint for core/DSL.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline file with current counts (use intentionally).",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        default=True,
        help="Show delta from baseline for each metric (default: True).",
    )
    args = parser.parse_args()

    baseline = _load_baseline()
    current: Dict[str, Dict[str, int]] = {
        scope: {"cast": _count_token(path, "cast("), "Any": _count_token(path, "Any")}
        for scope, path in TARGETS.items()
    }

    # Update baseline if requested
    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"✅ Updated baseline file: {BASELINE_PATH.name}")
        print("  New baseline values:")
        for scope, metrics in current.items():
            for metric, count in metrics.items():
                print(f"    {scope}.{metric}: {count}")
        return

    # Show delta report
    print("Type-safety metrics (current vs baseline):")
    for scope, metrics in current.items():
        for metric, count in metrics.items():
            allowed = baseline.get(scope, {}).get(metric, 0)
            delta = _format_delta(count, allowed)
            status = "✓" if count <= allowed else "✗"
            print(f"  {status} {scope}.{metric}: {count} (baseline: {allowed}, delta: {delta})")
    print()

    failures: list[str] = []
    for scope, metrics in current.items():
        for metric, count in metrics.items():
            allowed = baseline.get(scope, {}).get(metric)
            if allowed is None:
                failures.append(
                    f"No baseline for {scope}.{metric}; add it to {BASELINE_PATH.name}."
                )
                continue
            if count > allowed:
                delta = _format_delta(count, allowed)
                failures.append(
                    f"{scope}.{metric}: found {count}, baseline {allowed} ({delta}) "
                    "(reduce usages or run with --update-baseline)"
                )

    if failures:
        sys.stderr.write("Type-safety lint failures detected:\n")
        for failure in failures:
            sys.stderr.write(f" - {failure}\n")
        sys.exit(1)

    print(
        "✅ Type-safety lint passed "
        f"(casts/Any not above baseline; baseline file: {BASELINE_PATH.name})"
    )


if __name__ == "__main__":
    main()
