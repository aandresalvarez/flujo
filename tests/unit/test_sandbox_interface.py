import pytest
from typing import Any

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.runtime_builder import FlujoRuntimeBuilder
from flujo.domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult
from flujo.infra.sandbox import NullSandbox, RemoteSandbox


class DummySandbox(SandboxProtocol):
    def __init__(self) -> None:
        self.calls: list[SandboxExecution] = []

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        self.calls.append(request)
        return SandboxResult(
            stdout="ok",
            stderr="",
            exit_code=0,
            artifacts=None,
            sandbox_id="dummy",
            timed_out=False,
            error=None,
        )


@pytest.mark.asyncio
async def test_null_sandbox_returns_error() -> None:
    sandbox = NullSandbox()
    request = SandboxExecution(code="print('hi')", language="python")
    result = await sandbox.exec_code(request)
    assert result.error
    assert result.exit_code == 0
    assert result.succeeded is False


def test_runtime_builder_defaults_to_null_sandbox() -> None:
    deps = FlujoRuntimeBuilder().build()
    assert isinstance(deps.sandbox, NullSandbox)


def test_runtime_builder_accepts_custom_sandbox() -> None:
    custom = DummySandbox()
    deps = FlujoRuntimeBuilder().build(sandbox=custom)
    assert deps.sandbox is custom


def test_executor_core_exposes_sandbox() -> None:
    custom = DummySandbox()
    deps = FlujoRuntimeBuilder().build(sandbox=custom)
    core = ExecutorCore(deps=deps)
    assert core.sandbox is custom


def test_runtime_builder_selects_remote_sandbox(monkeypatch: Any) -> None:
    monkeypatch.setenv("FLUJO_SANDBOX_MODE", "remote")
    monkeypatch.setenv("FLUJO_SANDBOX_API_URL", "https://sandbox")
    deps = FlujoRuntimeBuilder().build()
    assert isinstance(deps.sandbox, RemoteSandbox)
