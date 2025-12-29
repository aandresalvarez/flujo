from __future__ import annotations

from pathlib import Path

import pytest


def _clear_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_env_uri_precedence_over_test_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given an explicit FLUJO_STATE_URI, it should take precedence even in test mode
    env_db = tmp_path / "env_override.db"
    monkeypatch.setenv("FLUJO_STATE_URI", f"sqlite:///{env_db.as_posix()}")
    monkeypatch.setenv("FLUJO_TEST_MODE", "1")
    monkeypatch.setenv("FLUJO_TEST_STATE_DIR", (tmp_path / "ignored").as_posix())

    from flujo.cli.config import load_backend_from_config
    from flujo.state.backends.sqlite import SQLiteBackend

    backend = load_backend_from_config()
    assert isinstance(backend, SQLiteBackend)
    assert backend.db_path == env_db


def test_test_mode_uses_isolated_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # When no FLUJO_STATE_URI and test mode is on, use FLUJO_TEST_STATE_DIR/flujo_ops.db
    _clear_env(monkeypatch, "FLUJO_STATE_URI")
    monkeypatch.setenv("FLUJO_TEST_MODE", "1")
    monkeypatch.setenv("FLUJO_TEST_STATE_DIR", tmp_path.as_posix())

    from flujo.cli.config import load_backend_from_config
    from flujo.state.backends.sqlite import SQLiteBackend

    backend = load_backend_from_config()
    assert isinstance(backend, SQLiteBackend)
    assert backend.db_path.parent == tmp_path
    assert backend.db_path.name == "flujo_ops.db"


def test_default_non_test_mode_uses_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate non-test mode by ensuring PYTEST_CURRENT_TEST is not set and FLUJO_TEST_MODE=0
    _clear_env(monkeypatch, "FLUJO_STATE_URI", "PYTEST_CURRENT_TEST")
    monkeypatch.setenv("FLUJO_TEST_MODE", "0")
    monkeypatch.chdir(tmp_path)

    from flujo.cli.config import load_backend_from_config
    from flujo.state.backends.sqlite import SQLiteBackend

    backend = load_backend_from_config()
    assert isinstance(backend, SQLiteBackend)
    # Defaults to sqlite:///flujo_ops.db in CWD
    assert backend.db_path == tmp_path / "flujo_ops.db"


def test_postgres_uri_returns_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def mock_find_spec(name: str):
        if name == "asyncpg":
            return importlib.util.spec_from_loader("asyncpg", loader=None)
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)
    monkeypatch.setenv("FLUJO_STATE_URI", "postgres://user:pass@localhost:5432/flujo")
    monkeypatch.setenv("FLUJO_TEST_MODE", "0")

    from flujo.cli.config import load_backend_from_config
    from flujo.state.backends.postgres import PostgresBackend

    backend = load_backend_from_config()
    assert isinstance(backend, PostgresBackend)
