#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict
import sys

try:
    import yaml
except Exception:
    print("This script requires PyYAML. Install with: pip install PyYAML", file=sys.stderr)
    sys.exit(1)


DEFAULT_PROCESSING: Dict[str, Any] = {
    "structured_output": "auto",  # off | auto | openai_json | outlines | xgrammar
    "aop": "minimal",  # off | minimal | full
    "coercion": {
        "tolerant_level": 0,  # 0=off, 1=json5, 2=json-repair
        "allow": {
            "integer": ["str->int"],
            "number": ["str->float"],
            "boolean": ["str->bool"],
        },
    },
    # Do not enable reasoning_precheck by default; teams can opt-in
}


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[index]
        else:
            out[k] = v
    return out


def migrate_pipeline_yaml(path: Path, dry_run: bool = False, backup: bool = True) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[skip] Failed to read {path}: {e}")
        return False
    try:
        data = yaml.safe_load(text)
    except Exception as e:
        print(f"[skip] Not valid YAML {path}: {e}")
        return False

    if not isinstance(data, dict):
        print(f"[skip] Unexpected YAML root (not a mapping): {path}")
        return False

    steps = data.get("steps")
    if not isinstance(steps, list):
        print(f"[skip] No steps list: {path}")
        return False

    changed = False
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pmeta = step.get("processing")
        if isinstance(pmeta, dict):
            merged = deep_merge(pmeta, {})  # copy
            # Merge default knobs only when absent to avoid overriding explicit user config
            for k, v in DEFAULT_PROCESSING.items():
                if k not in merged:
                    merged[k] = v
            if merged != pmeta:
                data["steps"][i]["processing"] = merged
                changed = True
        else:
            data["steps"][i]["processing"] = dict(DEFAULT_PROCESSING)
            changed = True

    if not changed:
        print(f"[ok] {path} already has AROS processing config")
        return False

    if dry_run:
        print(f"[dry-run] Would update {path}")
        return True

    try:
        if backup:
            path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"[updated] {path}")
        return True
    except Exception as e:
        print(f"[error] Failed to write {path}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Add AROS processing defaults to pipeline YAML files")
    ap.add_argument("root", help="Root directory to scan (recursively)")
    ap.add_argument(
        "--ext",
        default=",.yaml,.yml",
        help="Comma-separated list of file extensions to include (default: .yaml,.yml)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Show files that would change without writing"
    )
    ap.add_argument(
        "--no-backup", action="store_true", help="Do not create .bak files before writing"
    )
    ap.add_argument("--include", default="", help="Substring to include in file path")
    ap.add_argument("--exclude", default="", help="Substring to exclude from file path")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        print(f"Root directory not found: {root}", file=sys.stderr)
        sys.exit(2)

    changed_any = False
    exts = [e for e in args.ext.split(",") if e]
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not any(fn.endswith(e) for e in exts):
                continue
            p = Path(dirpath) / fn
            sp = str(p)
            if args.include and args.include not in sp:
                continue
            if args.exclude and args.exclude in sp:
                continue
            if migrate_pipeline_yaml(p, dry_run=args.dry_run, backup=(not args.no_backup)):
                changed_any = True
    if args.dry_run and not changed_any:
        print("No files would be updated.")


if __name__ == "__main__":
    main()
