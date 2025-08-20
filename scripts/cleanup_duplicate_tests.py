#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List


def find_dupes(root: Path) -> List[Path]:
    matches: List[Path] = []
    for base in (root / "tests", root / "flujo"):
        if not base.exists():
            continue
        for p in base.rglob("* 2.py"):
            if p.is_file():
                matches.append(p)
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and optionally move/delete duplicate test files ending with ' 2.py'"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete duplicates instead of moving to backup folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files; do not move or delete",
    )
    args = parser.parse_args()

    root = Path.cwd()
    dupes = find_dupes(root)
    if not dupes:
        print("No files found matching '* 2.py' under tests/ or flujo/.")
        return 0

    print(f"Found {len(dupes)} duplicate-looking files:\n")
    for p in dupes:
        print(f" - {p}")

    if args.dry_run:
        return 0

    if args.delete:
        for p in dupes:
            try:
                p.unlink()
            except Exception as e:
                print(f"Failed to delete {p}: {e}", file=sys.stderr)
        print("\nDeleted duplicates.")
        return 0

    # Move to backup folder preserving relative structure
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / "output" / f"duplicate_test_backups_{ts}"
    for p in dupes:
        rel = p.relative_to(root)
        dest = backup_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(p), str(dest))
        except Exception as e:
            print(f"Failed to move {p} -> {dest}: {e}", file=sys.stderr)
    print(f"\nMoved duplicates to: {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
