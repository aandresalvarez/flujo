#!/usr/bin/env python3
"""Fail CI when new `Any` or `cast()` usages are introduced in core/DSL."""

from __future__ import annotations

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


def main() -> None:
    baseline = _load_baseline()
    current: Dict[str, Dict[str, int]] = {
        scope: {"cast": _count_token(path, "cast("), "Any": _count_token(path, "Any")}
        for scope, path in TARGETS.items()
    }

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
                failures.append(
                    f"{scope}.{metric}: found {count}, baseline {allowed} "
                    "(reduce usages or update baseline intentionally)"
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
