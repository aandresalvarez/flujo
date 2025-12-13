from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Protocol, Sequence, runtime_checkable


@dataclass(slots=True)
class SandboxExecution:
    """A single sandboxed code execution request."""

    code: str
    language: str
    files: Mapping[str, str] | None = None
    environment: Mapping[str, str] | None = None
    arguments: Sequence[str] = field(default_factory=tuple)
    timeout_s: float | None = None


@dataclass(slots=True)
class SandboxResult:
    """Result of sandboxed execution."""

    stdout: str
    stderr: str
    exit_code: int
    artifacts: MutableMapping[str, bytes] | None = None
    sandbox_id: str | None = None
    timed_out: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


@runtime_checkable
class SandboxProtocol(Protocol):
    """Protocol for executing untrusted code in an isolated sandbox."""

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        """Execute code in an isolated sandbox and return structured result."""
