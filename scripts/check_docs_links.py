from __future__ import annotations

import os
import re
from typing import Iterable


DOCS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


def iter_markdown_files(root: str) -> Iterable[str]:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".md"):
                yield os.path.join(dirpath, f)


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ASSET_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".bmp", ".ico"}


def main() -> int:
    broken: list[tuple[str, str, str, str]] = []
    for path in iter_markdown_files(DOCS_ROOT):
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception as e:  # pragma: no cover - defensive
            print(f"WARN: failed reading {path}: {e}")
            continue
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            # Ignore external links and site-rooted paths
            if target.startswith(("http://", "https://", "mailto:", "tel:", "/")):
                continue
            # Strip title suffix in parens if present: (path "title")
            if '"' in target or "'" in target:
                # take token before first space
                target = target.split(" ", 1)[0]
            # Strip anchors
            t0 = target.split("#", 1)[0]
            if not t0:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), t0))
            ext = os.path.splitext(resolved)[1].lower()
            if ext and ext not in (".md", *ASSET_EXTS):
                if not os.path.exists(resolved):
                    broken.append((path, target, resolved, "non-md asset missing"))
                continue
            if not os.path.exists(resolved):
                broken.append((path, target, resolved, "missing"))

    if broken:
        print("Broken links detected:")
        for src, raw, resolved, why in broken:
            print(f"- in {src}: {raw} -> {resolved} [{why}]")
        return 1
    print("No broken relative links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
