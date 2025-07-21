#!/usr/bin/env python3
"""Pre-commit hook: Fail if forbidden patterns are present in non-test code.

Currently forbids:
* `eval(` ‑ runtime code execution risk
* `pickle.` – unsafe deserialisation (loads / dumps)

Only *.py files **outside** the `tests/` directory are scanned.
"""
from __future__ import annotations

import pathlib
import re
import sys
from typing import Final

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
FORBIDDEN_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "eval(": re.compile(r"\beval\s*\(", re.MULTILINE),
    "pickle.load/dump": re.compile(r"\bpickle\.(loads?|dumps?)\s*\(", re.MULTILINE),
}

# Exclude any path under tests/ or .venv etc.
EXCLUDE_DIRS: Final[set[str]] = {"tests", ".venv", "site-packages", "scripts"}


def iter_python_files() -> list[pathlib.Path]:
    return [
        p
        for p in REPO_ROOT.rglob("*.py")
        if not any(part in EXCLUDE_DIRS for part in p.parts)
    ]


def main() -> int:
    failures: list[str] = []

    for file in iter_python_files():
        text = file.read_text(encoding="utf-8", errors="ignore")
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
