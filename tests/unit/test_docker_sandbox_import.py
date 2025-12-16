from __future__ import annotations

from typing import Any

import pytest


def test_docker_sandbox_get_client_errors_when_docker_missing(monkeypatch: Any) -> None:
    from flujo.infra.sandbox.docker_sandbox import DockerSandbox

    real_import = __import__

    def fake_import(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:  # noqa: A002,E501
        if name == "docker":
            raise ImportError("no docker")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    sandbox = DockerSandbox.__new__(DockerSandbox)
    with pytest.raises(RuntimeError, match="Docker client unavailable"):
        sandbox._get_client()
