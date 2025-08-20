#!/usr/bin/env python3
"""Pre-commit hook: Fail if any Python file ends with the duplicate suffix ' 2.py'.

This typically indicates accidental copies (e.g., from editors) that confuse pytest discovery.
Scans the repository for files matching '* 2.py' under tests/ or flujo/.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Final

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    offenders: list[pathlib.Path] = []
    for base in (REPO_ROOT / "tests", REPO_ROOT / "flujo"):
        if not base.exists():
            continue
        for p in base.rglob("* 2.py"):
            if p.is_file():
                offenders.append(p)

    if offenders:
        print(
            "\n❌ Duplicate-suffixed files detected (likely accidental copies):\n", file=sys.stderr
        )
        for p in offenders:
            print(f" - {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "\nRun: uv run python scripts/cleanup_duplicate_tests.py to move/delete them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
