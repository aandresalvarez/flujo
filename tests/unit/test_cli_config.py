import tempfile
from pathlib import Path
import pytest
from flujo.cli.config import _normalize_sqlite_path


@pytest.mark.parametrize(
    "uri,expected_rel",
    [
        ("sqlite:///foo.db", "foo.db"),
        ("sqlite:///./foo.db", "./foo.db"),
        ("sqlite:////abs/path.db", "/abs/path.db"),
        ("sqlite:///../data/ops.db", "../data/ops.db"),
        ("sqlite:///subdir/bar.db", "subdir/bar.db"),
        ("sqlite:///./subdir/bar.db", "./subdir/bar.db"),
    ],
)
def test_normalize_sqlite_path_relative(uri, expected_rel):
    """
    _normalize_sqlite_path should resolve relative URIs to cwd, and absolute URIs as-is.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        result = _normalize_sqlite_path(uri, cwd)
        if expected_rel.startswith("/"):
            # Absolute path
            assert result.resolve() == Path(expected_rel).resolve(), (
                f"Expected absolute path {Path(expected_rel).resolve()}, got {result.resolve()}"
            )
        else:
            # Relative path
            assert result.resolve() == (cwd / Path(expected_rel)).resolve(), (
                f"Expected {(cwd / Path(expected_rel)).resolve()}, got {result.resolve()}"
            )


def test_normalize_sqlite_path_absolute():
    """
    _normalize_sqlite_path should return absolute paths as-is.
    """
    uri = "sqlite:////tmp/abs.db"
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        result = _normalize_sqlite_path(uri, cwd)
        assert result.resolve() == Path("/tmp/abs.db").resolve(), (
            f"Expected {Path('/tmp/abs.db').resolve()}, got {result.resolve()}"
        )


def test_normalize_sqlite_path_edge_cases():
    """
    _normalize_sqlite_path should handle edge cases like double slashes and /./ correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        # sqlite:///./foo.db -> ./foo.db
        uri = "sqlite:///./foo.db"
        result = _normalize_sqlite_path(uri, cwd)
        assert result.resolve() == (cwd / "./foo.db").resolve()
        # sqlite:////foo.db -> /foo.db
        uri = "sqlite:////foo.db"
        result = _normalize_sqlite_path(uri, cwd)
        assert result.resolve() == Path("/foo.db").resolve()
        # sqlite:///foo.db -> foo.db
        uri = "sqlite:///foo.db"
        result = _normalize_sqlite_path(uri, cwd)
        assert result.resolve() == (cwd / "foo.db").resolve()
        # sqlite:///../foo.db -> ../foo.db
        uri = "sqlite:///../foo.db"
        result = _normalize_sqlite_path(uri, cwd)
        assert result.resolve() == (cwd / "../foo.db").resolve()
