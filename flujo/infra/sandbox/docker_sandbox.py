from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from ...domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult


class DockerSandbox(SandboxProtocol):
    """Docker-based sandbox for local isolated execution (python-focused)."""

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        pull: bool = True,
        timeout_s: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self._image = image
        self._pull = pull
        self._timeout_s = timeout_s
        self._client: Any = client or self._get_client()
        if self._pull:
            try:
                self._client.images.pull(self._image)
            except Exception:
                # Non-fatal: proceed with whatever is available locally
                pass

    def _get_client(self) -> object:
        try:
            import docker  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time path
            raise RuntimeError(f"Docker client unavailable: {exc}") from exc
        try:
            return docker.from_env()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Docker from_env failed: {exc}") from exc

    def _build_command(self, request: SandboxExecution) -> list[str]:
        if request.language.lower() != "python":
            return []
        args: list[str] = ["python", "main.py"]
        if request.arguments:
            args.extend(list(request.arguments))
        return args

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        command = self._build_command(request)
        if not command:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=1,
                artifacts=None,
                sandbox_id=None,
                timed_out=False,
                error=f"Unsupported language: {request.language}",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            try:
                (workdir / "main.py").write_text(request.code, encoding="utf-8")
                for name, content in (request.files or {}).items():
                    dest = workdir / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
            except Exception as exc:
                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=1,
                    artifacts=None,
                    sandbox_id=None,
                    timed_out=False,
                    error=f"Failed to prepare files: {exc}",
                )

            try:
                container = self._client.containers.run(
                    self._image,
                    command,
                    working_dir="/workspace",
                    environment=request.environment or {},
                    volumes={str(workdir): {"bind": "/workspace", "mode": "rw"}},
                    network_disabled=True,
                    detach=True,
                    tty=False,
                )
            except Exception as exc:
                return SandboxResult(
                    stdout="",
                    stderr="",
                    exit_code=1,
                    artifacts=None,
                    sandbox_id=None,
                    timed_out=False,
                    error=f"Failed to start docker container: {exc}",
                )

            timed_out = False
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(container.wait),
                    timeout=request.timeout_s or self._timeout_s,
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    container.kill()
                except Exception:
                    pass
            except Exception:
                pass

            try:
                logs = container.logs(stdout=True, stderr=True) or b""
                stdout = logs.decode("utf-8", errors="replace")
                stderr = ""
            except Exception:
                stdout = ""
                stderr = ""

            try:
                status = container.wait()
                exit_code = int(status.get("StatusCode", 1))
            except Exception:
                exit_code = 1

            try:
                container.remove(force=True)
            except Exception:
                pass

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                artifacts=None,
                sandbox_id=None,
                timed_out=timed_out,
                error="Execution timed out" if timed_out else None,
            )
