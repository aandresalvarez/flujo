#!/usr/bin/env python3
"""Scaffold lint to ensure adapters are explicitly allowlisted."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = ROOT / "scripts" / "adapter_allowlist.json"
ADAPTER_CLASSES = (
    "DictContextAdapter",
    "SchemaAdapter",
    "TypedAny",
    "TypedAnyAdapter",
    "UnknownAdapter",
)


def _load_allowlist() -> Dict[str, Dict[str, str]]:
    if not ALLOWLIST_PATH.exists():
        sys.stderr.write(
            "Adapter allowlist missing. Commit scripts/adapter_allowlist.json to proceed.\n"
        )
        sys.exit(1)
    try:
        loaded = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Unable to parse adapter allowlist: {exc}\n")
        sys.exit(1)
    return loaded if isinstance(loaded, dict) else {"allowed": {}}


def _find_adapter_usages() -> List[Tuple[Path, str]]:
    usages: list[tuple[Path, str]] = []
    for file_path in (ROOT / "flujo").rglob("*.py"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for adapter in ADAPTER_CLASSES:
            if adapter in text:
                usages.append((file_path, adapter))
    return usages


def _has_token(text: str, token: str) -> bool:
    return f"ADAPTER_ALLOW:{token}" in text


def main() -> None:
    allowlist = _load_allowlist().get("allowed", {})
    usages = _find_adapter_usages()

    violations: list[str] = []
    for file_path, adapter in usages:
        token = allowlist.get(adapter)
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if token is None:
            violations.append(
                f"{adapter} used in {file_path.relative_to(ROOT)} without allowlist entry"
            )
            continue
        if not _has_token(text, token):
            violations.append(
                f"{adapter} in {file_path.relative_to(ROOT)} missing token ADAPTER_ALLOW:{token}"
            )

    if violations:
        sys.stderr.write("Adapter allowlist violations detected:\n")
        for violation in violations:
            sys.stderr.write(f" - {violation}\n")
        sys.exit(1)

    print("✅ Adapter allowlist lint passed (no unapproved adapter usage found).")


if __name__ == "__main__":
    main()
