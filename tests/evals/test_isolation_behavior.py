from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("sentinel", ["one", "two"])
def test_skills_base_dir_stack_is_isolated(sentinel: str) -> None:
    from flujo.domain.blueprint.loader_resolution import _skills_base_dir_stack

    # Autouse fixture should leave stack empty at test start
    assert list(_skills_base_dir_stack) == []

    _skills_base_dir_stack.append(sentinel)
    assert sentinel in _skills_base_dir_stack


@pytest.mark.parametrize("value", ["file:///tmp/a", "memory://b"])
def test_flujo_env_is_cleared_between_tests(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # Autouse fixture should clear mutable FLUJO_* env vars and set FLUJO_STATE_URI
    assert os.environ.get("FLUJO_STATE_URI") == "memory://"

    monkeypatch.setenv("FLUJO_STATE_URI", value)
    assert os.environ["FLUJO_STATE_URI"] == value


def test_sys_path_repo_root_is_first() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    assert sys.path[0] == repo_root
