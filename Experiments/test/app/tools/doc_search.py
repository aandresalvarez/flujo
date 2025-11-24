from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_ROOTS: Tuple[str, ...] = ("docs", "examples", "README.md")


def search_local_docs(
    query: str, limit: int = 4, *, roots: Sequence[str] = DEFAULT_ROOTS
) -> List[Dict[str, str]]:
    """Return simple text matches for the query in project docs/examples."""
    needle = query.lower().strip()
    if not needle:
        return []
    matches: List[Dict[str, str]] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for file_path in _iter_candidate_files(path):
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(
                        {
                            "path": str(file_path),
                            "line": str(line_no),
                            "snippet": line.strip(),
                        }
                    )
                    if len(matches) >= limit:
                        return matches
    return matches


def _iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in {".md", ".txt"} or path.name.endswith("README.md"):
            yield path
        return
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in {".md", ".txt"}:
            yield child
