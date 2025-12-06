import pytest

from flujo.builtins import code_interpreter
from flujo.domain.models import PipelineContext
from flujo.domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult
from flujo.infra.sandbox import NullSandbox
from flujo.infra.skill_registry import get_skill_registry


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


class ErrorSandbox(SandboxProtocol):
    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        raise RuntimeError("boom")


def test_code_interpreter_registered() -> None:
    reg = get_skill_registry()
    assert reg.get("flujo.builtins.code_interpreter") is not None


@pytest.mark.asyncio
async def test_code_interpreter_uses_context_sandbox() -> None:
    sandbox = DummySandbox()
    ctx = PipelineContext()
    object.__setattr__(ctx, "_sandbox", sandbox)

    result = await code_interpreter(
        "print('hi')",
        language="python",
        files={"main.py": "print('hi')"},
        environment={"ENV": "1"},
        arguments=["--flag"],
        timeout_s=1.5,
        context=ctx,
    )

    assert result["succeeded"] is True
    assert result["stdout"] == "ok"
    assert sandbox.calls, "sandbox was not invoked"
    call = sandbox.calls[0]
    assert call.code == "print('hi')"
    assert call.language == "python"
    assert call.files == {"main.py": "print('hi')"}
    assert call.environment == {"ENV": "1"}
    assert tuple(call.arguments) == ("--flag",)
    assert call.timeout_s == 1.5


@pytest.mark.asyncio
async def test_code_interpreter_falls_back_to_null_sandbox() -> None:
    ctx = PipelineContext()

    result = await code_interpreter("print('hi')", context=ctx)

    assert result["succeeded"] is False
    assert result["exit_code"] == 0
    assert result["error"] == NullSandbox._SANDBOX_DISABLED_MSG


@pytest.mark.asyncio
async def test_code_interpreter_surfaces_sandbox_errors() -> None:
    ctx = PipelineContext()
    object.__setattr__(ctx, "_sandbox", ErrorSandbox())

    result = await code_interpreter("print('hi')", context=ctx)

    assert result["succeeded"] is False
    assert "boom" in (result["error"] or "")
