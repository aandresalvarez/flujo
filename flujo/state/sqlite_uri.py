"""SQLite URI helpers.

Flujo uses a custom `sqlite:///...` URI form for state backend configuration.
This module centralizes the URI → Path normalization so non-CLI code does not
need to import from `flujo.cli`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def normalize_sqlite_path(uri: str, cwd: Path, *, config_dir: Path | None = None) -> Path:
    """Normalize a Flujo SQLite URI into a concrete filesystem path.

    Flujo accepts the following forms:
    - Absolute: ``sqlite:////abs/path.db`` → ``/abs/path.db``
    - Relative: ``sqlite:///foo.db`` → ``<base_dir>/foo.db`` where ``base_dir`` is
      ``config_dir`` (if provided) or ``cwd``.

    The normalization is intentionally tolerant of non-standard forms like
    ``sqlite://foo.db`` (netloc present) to preserve backward compatibility.
    """

    parsed = urlparse(uri)
    base_dir = config_dir if config_dir is not None else cwd

    # Case 1: Non-standard sqlite://path (netloc present) or Windows drive in netloc
    if parsed.netloc:
        path_str = parsed.netloc + parsed.path
        # Windows drive logic (C:/... or C:\\...)
        try:
            if ":" in path_str and Path(path_str).is_absolute():
                return Path(path_str).resolve()
        except Exception:
            pass

        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()

        return (base_dir / path_str).resolve()

    # Case 2: Standard URI path
    path_str = parsed.path
    if not path_str or not path_str.strip():
        raise ValueError(
            "Malformed SQLite URI: empty path. Use 'sqlite:///file.db' or 'sqlite:////abs/path.db'."
        )

    # Check for // implying absolute path (sqlite:////abs...)
    if path_str.startswith("//"):
        return Path(path_str[1:]).resolve()

    # Strip standard leading slash for processing
    clean_path_str = path_str[1:] if path_str.startswith("/") else path_str

    # Check if the remaining part is absolute (e.g. Windows /C:/...)
    try:
        p = Path(clean_path_str)
        if p.is_absolute():
            return p.resolve()
    except Exception:
        pass

    # Default to specific behavior: sqlite:///foo.db is relative
    return (base_dir / clean_path_str).resolve()
