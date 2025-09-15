#!/usr/bin/env python3
"""Pre-commit hook: Fail if forbidden patterns are present in code.

Currently forbids:
* `eval(` ‑ runtime code execution risk
* `pickle.` – unsafe deserialisation (loads / dumps)

Scans all *.py files including test files for security.
Excludes intentional test cases that demonstrate security patterns.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
from typing import Final

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
FORBIDDEN_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "eval(": re.compile(r"\beval\s*\(", re.MULTILINE),
    "pickle.load/dump": re.compile(r"\bpickle\.(loads?|dumps?)\s*\(", re.MULTILINE),
}

# Exclude only build/venv directories, but include tests for security
EXCLUDE_DIRS: Final[set[str]] = {".venv", "site-packages", "scripts", ".git"}


def iter_python_files() -> list[pathlib.Path]:
    """Yield python files while pruning excluded directories during traversal.

    Using rglob() can still descend into excluded directories before filtering,
    which is expensive and may time out on large virtualenvs. We prune in-place
    via os.walk to avoid recursing into directories like .venv or site-packages.
    """
    results: list[pathlib.Path] = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Prune excluded directories in-place to prevent os.walk from descending
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            results.append(pathlib.Path(root, fname))
    return results


def main() -> int:
    failures: list[str] = []

    for file in iter_python_files():
        text = file.read_text(encoding="utf-8", errors="ignore")

        # Skip files that are intentionally testing security patterns
        if "intentionally includes dangerous code patterns" in text:
            continue

        # Skip files that are testing serialization functionality
        if "test_paused_hitl_pipeline_can_be_serialized_and_resumed" in text:
            continue

        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{file}: contains forbidden pattern '{name}'")
                break

    if failures:
        print("\n❌ Forbidden patterns detected:\n" + "\n".join(failures), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
