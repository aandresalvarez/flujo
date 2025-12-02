from __future__ import annotations

import asyncio

from flujo.application.runner_components import StateBackendManager
from flujo.state.backends.memory import InMemoryBackend


class _DummyBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.deleted: list[str] = []

    async def shutdown(self) -> None:  # pragma: no cover - simple counter
        self.shutdown_calls += 1

    async def delete_state(self, run_id: str) -> None:  # pragma: no cover - simple counter
        self.deleted.append(run_id)


def test_default_backend_uses_in_memory_in_test_mode(monkeypatch) -> None:
    monkeypatch.setenv("FLUJO_TEST_MODE", "1")

    manager = StateBackendManager()

    assert isinstance(manager.backend, InMemoryBackend)

    # Should no-op even when backend lacks shutdown
    asyncio.run(manager.shutdown())


def test_custom_backend_not_owned(monkeypatch) -> None:
    dummy = _DummyBackend()

    manager = StateBackendManager(state_backend=dummy)

    asyncio.run(manager.shutdown())
    assert dummy.shutdown_calls == 0  # not owned, so skip shutdown


def test_delete_on_completion_invokes_backend(monkeypatch) -> None:
    dummy = _DummyBackend()

    manager = StateBackendManager(state_backend=dummy, delete_on_completion=True)

    asyncio.run(manager.delete_state("run-123"))

    assert dummy.deleted == ["run-123"]


def test_disable_backend_skips_creation(monkeypatch) -> None:
    manager = StateBackendManager(enable_backend=False)

    assert manager.backend is None
    # Should no-op
    asyncio.run(manager.shutdown())
    asyncio.run(manager.delete_state("run-ignored"))
