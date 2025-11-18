from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from flujo import Flujo, Pipeline, Step
from flujo.domain.models import PipelineContext
from flujo.domain.resources import AppResources
from flujo.state.backends.memory import InMemoryBackend
from flujo.state.backends.sqlite import SQLiteBackend


async def _echo_agent(
    data: str,
    *,
    context: PipelineContext | None = None,
    resources: AppResources | None = None,
    **_: Any,
) -> str:
    return data.upper()


def _make_runner(**kwargs: Any) -> Flujo[str, Any, PipelineContext]:
    step = Step.from_callable(_echo_agent, name="echo")
    pipeline = Pipeline.from_step(step)
    return Flujo(pipeline, enable_tracing=False, **kwargs)


def test_runner_shuts_down_default_state_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner-owned SQLite backends should shut down and leave no worker threads."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLUJO_TEST_MODE", "0")
    runner = _make_runner()
    result = runner.run("hello world")
    assert result.output == "HELLO WORLD"
    backend = runner.state_backend
    assert isinstance(backend, SQLiteBackend)
    assert getattr(backend, "_connection_pool", None) is None
    lingering = [
        t
        for t in threading.enumerate()
        if "flujo-sqlite" in (t.name or "").lower() and not t.daemon
    ]
    assert not lingering


class _RecordingBackend(InMemoryBackend):
    """Test double to record shutdown invocations."""
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls = 0

    async def shutdown(self) -> None:  # type: ignore[override]
        self.shutdown_calls += 1


def test_runner_does_not_shutdown_injected_backend() -> None:
    """Runner should not manage the lifecycle of injected state backends."""
    backend = _RecordingBackend()
    runner = _make_runner(state_backend=backend)
    runner.run("custom backend")
    assert backend.shutdown_calls == 0
    # Manual close should be a no-op for injected backends.
    # Users remain responsible for their lifetime management.
    asyncio.run(runner.aclose())
    assert backend.shutdown_calls == 0
