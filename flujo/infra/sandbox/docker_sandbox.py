from __future__ import annotations

from typing import Final

from ...domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult


class DockerSandbox(SandboxProtocol):
    """Placeholder Docker sandbox. Implementation pending optional docker extra."""

    _NOT_IMPLEMENTED: Final[str] = "Docker sandbox not implemented (placeholder)."

    async def exec_code(
        self, request: SandboxExecution
    ) -> SandboxResult:  # pragma: no cover - placeholder
        return SandboxResult(
            stdout="",
            stderr="",
            exit_code=1,
            artifacts=None,
            sandbox_id=None,
            timed_out=False,
            error=self._NOT_IMPLEMENTED,
        )
