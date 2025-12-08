import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from flujo.cli.config import _normalize_sqlite_path, load_backend_from_config


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
            assert (
                result.resolve() == Path(expected_rel).resolve()
            ), f"Expected absolute path {Path(expected_rel).resolve()}, got {result.resolve()}"
        else:
            # Relative path
            assert (
                result.resolve() == (cwd / Path(expected_rel)).resolve()
            ), f"Expected {(cwd / Path(expected_rel)).resolve()}, got {result.resolve()}"


def test_normalize_sqlite_path_absolute():
    """
    _normalize_sqlite_path should return absolute paths as-is.
    """
    uri = "sqlite:////tmp/abs.db"
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        result = _normalize_sqlite_path(uri, cwd)
        assert (
            result.resolve() == Path("/tmp/abs.db").resolve()
        ), f"Expected {Path('/tmp/abs.db').resolve()}, got {result.resolve()}"


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


class TestConfigWarning:
    """Test that warning messages are displayed correctly when no state URI is configured."""

    def test_warning_displayed_when_no_env_and_no_config(self, caplog):
        """Test that warning is displayed when neither env var nor config file provides state URI."""
        # Mock environment to have no FLUJO_STATE_URI
        with patch.dict(os.environ, {}, clear=True):
            # Mock get_state_uri to return None (no config file or no state_uri in config)
            with patch("flujo.cli.config.get_state_uri", return_value=None):
                # Mock the file operations to avoid actual file system calls
                with (
                    patch("builtins.open", MagicMock()),
                    patch("pathlib.Path.exists", return_value=True),
                    patch("os.access", return_value=True),
                ):
                    # This should trigger the warning and use default
                    with caplog.at_level("WARNING"):
                        backend = load_backend_from_config()
                    # Check that warning was logged
                    assert (
                        "[flujo.config] FLUJO_STATE_URI not set, using default 'sqlite:///flujo_ops.db'"
                        in caplog.text
                    )
                    # Verify backend was created successfully
                    assert backend is not None

    def test_no_warning_when_env_var_set(self, capsys):
        """Test that no warning is displayed when FLUJO_STATE_URI environment variable is set."""
        # Mock environment to have FLUJO_STATE_URI set
        with patch.dict(os.environ, {"FLUJO_STATE_URI": "sqlite:///test.db"}):
            # Mock the file operations to avoid actual file system calls
            with (
                patch("builtins.open", MagicMock()),
                patch("pathlib.Path.exists", return_value=True),
                patch("os.access", return_value=True),
            ):
                backend = load_backend_from_config()

                # Check that no warning was printed
                captured = capsys.readouterr()
                assert "[flujo.config] Warning: FLUJO_STATE_URI not set" not in captured.err

                # Verify backend was created successfully
                assert backend is not None

    def test_no_warning_when_config_file_has_state_uri(self, capsys):
        """Test that no warning is displayed when config file provides state URI."""
        # Mock environment to have no FLUJO_STATE_URI
        with patch.dict(os.environ, {}, clear=True):
            # Mock get_state_uri to return a value (config file has state_uri)
            with (
                patch("flujo.cli.config.get_state_uri", return_value="sqlite:///config.db"),
                patch("builtins.open", MagicMock()),
                patch("pathlib.Path.exists", return_value=True),
                patch("os.access", return_value=True),
            ):
                backend = load_backend_from_config()

                # Check that no warning was printed
                captured = capsys.readouterr()
                assert "[flujo.config] Warning: FLUJO_STATE_URI not set" not in captured.err

                # Verify backend was created successfully
                assert backend is not None
