#!/usr/bin/env python3
"""Codemod to rewrite scratchpad accesses to typed context fields.

This is a conservative search/replace helper meant for Phase 1 migration.
It rewrites patterns of the form:
    <ctx>.scratchpad["foo"]  -> <ctx>.foo
    <ctx>.scratchpad.get("foo") -> getattr(<ctx>, "foo", None)

Usage:
    python scripts/codemods/scratchpad_to_typed.py --apply path/to/file.py
    python scripts/codemods/scratchpad_to_typed.py --dry-run src/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


SCRATCHPAD_INDEX = re.compile(r"(?P<prefix>\b\w+)\.scratchpad\[\s*['\"](?P<key>[\w_]+)['\"]\s*\]")
SCRATCHPAD_GET = re.compile(
    r"(?P<prefix>\b\w+)\.scratchpad\.get\(\s*['\"](?P<key>[\w_]+)['\"]\s*(?:,\s*(?P<default>[^)]*))?\)"
)


def iter_targets(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".py":
        yield path
        return
    for file_path in path.rglob("*.py"):
        yield file_path


def rewrite_text(text: str) -> tuple[str, int]:
    """Return rewritten text and replacements count."""
    count = 0

    def _sub_index(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{match.group('prefix')}.{match.group('key')}"

    def _sub_get(match: re.Match) -> str:
        nonlocal count
        count += 1
        prefix = match.group("prefix")
        key = match.group("key")
        default = match.group("default")
        if default is None:
            default_str = "None"
        else:
            default_str = default.strip()
        return f'getattr({prefix}, "{key}", {default_str})'

    text = SCRATCHPAD_INDEX.sub(_sub_index, text)
    text = SCRATCHPAD_GET.sub(_sub_get, text)
    return text, count


def process(path: Path, apply: bool) -> int:
    original = path.read_text(encoding="utf-8")
    rewritten, count = rewrite_text(original)
    if count and apply:
        path.write_text(rewritten, encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to rewrite.")
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for default (no apply).")
    args = parser.parse_args()

    total = 0
    for target in args.paths:
        for file_path in iter_targets(target):
            total += process(file_path, apply=args.apply)

    action = "Rewritten" if args.apply else "Detected"
    print(f"{action} {total} scratchpad occurrence(s)")


if __name__ == "__main__":
    main()
