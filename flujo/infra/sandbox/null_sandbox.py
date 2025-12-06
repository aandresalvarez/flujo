from __future__ import annotations

from typing import Final

from ...domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult


class NullSandbox(SandboxProtocol):
    """No-op sandbox used as a safe default when sandboxing is not configured."""

    _SANDBOX_DISABLED_MSG: Final[str] = "Sandbox execution disabled (NullSandbox)."

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        del request
        return SandboxResult(
            stdout="",
            stderr="",
            exit_code=0,
            artifacts=None,
            sandbox_id=None,
            timed_out=False,
            error=self._SANDBOX_DISABLED_MSG,
        )
