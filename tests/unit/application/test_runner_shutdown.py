from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import pytest

from flujo import Flujo, Pipeline, Step
from flujo.domain.models import PipelineContext
from flujo.domain.resources import AppResources
from flujo.state.backends.memory import InMemoryBackend
from flujo.state.backends.sqlite import SQLiteBackend


async def _echo_agent(
    data: str,
    *,
    context: Optional[PipelineContext] = None,
    resources: Optional[AppResources] = None,
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
    monkeypatch.delenv("FLUJO_STATE_URI", raising=False)
    monkeypatch.delenv("FLUJO_CONFIG_PATH", raising=False)
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


@pytest.mark.asyncio
async def test_runner_does_not_shutdown_injected_backend() -> None:
    """Runner should not manage the lifecycle of injected state backends."""
    backend = _RecordingBackend()
    runner = _make_runner(state_backend=backend)
    await runner.run_async("custom backend")
    assert backend.shutdown_calls == 0
    # Manual close should be a no-op for injected backends.
    # Users remain responsible for their lifetime management.
    await runner.aclose()
    assert backend.shutdown_calls == 0


@pytest.mark.asyncio
async def test_runner_close_raises_inside_running_loop() -> None:
    """Calling `runner.close()` from async contexts should fail fast.

    This prevents silent, best-effort background cleanup that can leak resources and
    cause teardown-time warnings/flakes.
    """
    runner = _make_runner(state_backend=_RecordingBackend())
    with pytest.raises(TypeError, match="cannot be called from a running event loop"):
        runner.close()


@pytest.mark.asyncio
async def test_runner_sync_context_manager_raises_inside_running_loop() -> None:
    runner = _make_runner(state_backend=_RecordingBackend())
    with pytest.raises(TypeError, match="cannot be called from a running event loop"):
        with runner:
            pass
