"""SQLite URI helpers.

Flujo uses a custom `sqlite:///...` URI form for state backend configuration.
This module centralizes the URI → Path normalization so non-CLI code does not
need to import from `flujo.cli`.

This utility follows FLUJO_TEAM_GUIDE Section 12 (Type Safety) and Section 13
(Change Localization) by providing a well-typed, testable path resolution utility.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import ParseResult, urlparse


def normalize_sqlite_path(uri: str, cwd: Path, *, config_dir: Path | None = None) -> Path:
    """Normalize a Flujo SQLite URI into a concrete filesystem path.

    This function handles various SQLite URI formats including standard forms,
    non-standard forms with netloc, and Windows path edge cases. It provides
    robust path resolution while maintaining backward compatibility.

    Flujo accepts the following forms:
    - Absolute: ``sqlite:////abs/path.db`` → ``/abs/path.db``
    - Relative: ``sqlite:///foo.db`` → ``<base_dir>/foo.db`` where ``base_dir`` is
      ``config_dir`` (if provided) or ``cwd``.
    - Non-standard: ``sqlite://foo.db`` (netloc present) → handled for backward compatibility

    Args:
        uri: SQLite URI string (e.g., 'sqlite:///file.db', 'sqlite:////abs/path.db')
        cwd: Current working directory as Path object for relative path resolution
        config_dir: Optional config directory Path. If provided, relative paths are
            resolved against this directory instead of cwd. If None, uses cwd.

    Returns:
        Resolved absolute Path object. The path is normalized and resolved to an
        absolute filesystem path.

    Raises:
        ValueError: If the URI is malformed (e.g., empty path after parsing).
            The error message includes guidance on correct URI format.

    Examples:
        >>> from pathlib import Path
        >>> normalize_sqlite_path("sqlite:///foo.db", Path.cwd())
        Path('/current/working/dir/foo.db')

        >>> normalize_sqlite_path("sqlite:////abs/path.db", Path.cwd())
        Path('/abs/path.db')

        >>> normalize_sqlite_path("sqlite:///foo.db", Path.cwd(), config_dir=Path("/config"))
        Path('/config/foo.db')
    """
    parsed: ParseResult = urlparse(uri)
    base_dir: Path = config_dir if config_dir is not None else cwd

    # Case 1: Non-standard sqlite://path (netloc present) or Windows drive in netloc
    if parsed.netloc:
        path_str: str = parsed.netloc + parsed.path
        # Windows drive logic (C:/... or C:\\...)
        try:
            test_path: Path = Path(path_str)
            if ":" in path_str and test_path.is_absolute():
                return test_path.resolve()
        except (OSError, ValueError):
            # Path construction failed, continue with normal resolution
            pass

        p: Path = Path(path_str)
        if p.is_absolute():
            return p.resolve()

        return (base_dir / path_str).resolve()

    # Case 2: Standard URI path
    path_str = parsed.path
    stripped_path = path_str.strip() if path_str else ""
    # Check for empty path: urlparse("sqlite:///") returns path="/", which we treat as empty
    if not path_str or not stripped_path or stripped_path == "/":
        raise ValueError(
            "Malformed SQLite URI: empty path. Use 'sqlite:///file.db' or 'sqlite:////abs/path.db'."
        )

    # Check for // implying absolute path (sqlite:////abs...)
    if path_str.startswith("//"):
        return Path(path_str[1:]).resolve()

    # Strip standard leading slash for processing
    clean_path_str: str = path_str[1:] if path_str.startswith("/") else path_str

    # Check if the remaining part is absolute (e.g. Windows /C:/...)
    try:
        p = Path(clean_path_str)
        if p.is_absolute():
            return p.resolve()
    except (OSError, ValueError):
        # Path construction failed, continue with relative resolution
        pass

    # Default to specific behavior: sqlite:///foo.db is relative
    return (base_dir / clean_path_str).resolve()
