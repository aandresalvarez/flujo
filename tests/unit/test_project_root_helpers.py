from __future__ import annotations

from pathlib import Path
import sys
import pytest

from flujo.cli.helpers import (
    resolve_project_root,
    find_project_root,
    ensure_project_root_on_sys_path,
    scaffold_project,
)
from typer import Exit as TyperExit


def test_resolve_project_root_explicit_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises(TyperExit):
        resolve_project_root(tmp_path / "does_not_exist", allow_missing=False)


def test_resolve_project_root_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create marker file in tmp_path to look like a project
    (tmp_path / "flujo.toml").write_text('state_uri = "memory://"\n')
    monkeypatch.setenv("FLUJO_PROJECT_ROOT", str(tmp_path))
    root = resolve_project_root(None, allow_missing=False)
    assert root == tmp_path


def test_resolve_project_root_marker_search_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tree: tmp/a/b; place flujo.toml in tmp
    (tmp_path / "flujo.toml").write_text('state_uri = "memory://"\n')
    a = tmp_path / "a"
    b = a / "b"
    b.mkdir(parents=True)
    monkeypatch.chdir(b)
    root = resolve_project_root(None, allow_missing=False)
    assert root == tmp_path


def test_resolve_project_root_allow_missing_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_project_root(None, allow_missing=True) is None
    with pytest.raises(TyperExit):
        resolve_project_root(None, allow_missing=False)


def test_find_project_root_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pipeline.yaml").write_text('version: "0.1"\nsteps: []\n')
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == tmp_path


def test_find_project_root_raises_when_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TyperExit):
        _ = find_project_root()


def test_ensure_project_root_on_sys_path_injects_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure path not currently in sys.path
    p = str(tmp_path)
    try:
        sys.path.remove(p)
    except ValueError:
        pass

    ensure_project_root_on_sys_path(tmp_path)
    assert sys.path[0] == p

    # Calling again should not duplicate
    ensure_project_root_on_sys_path(tmp_path)
    assert sys.path.count(p) == 1


def test_scaffold_project_fallback_writes_minimal_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force importlib.resources.files to raise to trigger fallback branch
    import importlib.resources as resources

    def boom(*_a, **_k):  # always raise
        raise RuntimeError("no resources")

    monkeypatch.setattr(resources, "files", boom)
    scaffold_project(tmp_path)

    flujo_toml = (tmp_path / "flujo.toml").read_text()
    pipeline_yaml = (tmp_path / "pipeline.yaml").read_text()
    assert 'state_uri = "sqlite:///.flujo/state.db"' in flujo_toml
    assert 'version: "0.1"' in pipeline_yaml
    assert (tmp_path / "skills" / "__init__.py").exists()
    assert (tmp_path / "skills" / "custom_tools.py").exists()
