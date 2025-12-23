from __future__ import annotations

import asyncio
import importlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from ...domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult

logger = logging.getLogger(__name__)


class DockerSandbox(SandboxProtocol):
    """Docker-based sandbox for local isolated execution (python-focused)."""

    def __init__(
        self,
        *,
        image: str = "python:3.13-slim",
        pull: bool = True,
        timeout_s: float = 60.0,
        mem_limit: str | int | None = None,
        pids_limit: int | None = None,
        network_mode: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._image = image
        self._pull = pull
        self._timeout_s = timeout_s
        if isinstance(mem_limit, str):
            mem_limit = mem_limit.strip() or None
        self._mem_limit = mem_limit
        self._pids_limit = pids_limit if isinstance(pids_limit, int) and pids_limit > 0 else None
        if isinstance(network_mode, str):
            network_mode = network_mode.strip() or None
        self._network_mode = network_mode
        self._client: Any = client or self._get_client()
        if self._pull:
            try:
                self._client.images.pull(self._image)
            except Exception:  # noqa: BLE001,S110
                logger.warning(
                    "Failed to pull Docker image %s; using local if available", self._image
                )

    def _get_client(self) -> object:
        try:
            docker: Any = importlib.import_module("docker")
        except Exception as exc:  # pragma: no cover - import-time path
            raise RuntimeError(f"Docker client unavailable: {exc}") from exc  # noqa: TRY003
        try:
            return docker.from_env()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Docker from_env failed: {exc}") from exc  # noqa: TRY003

    def _build_command(self, request: SandboxExecution) -> tuple[list[str], str]:
        lang = request.language.lower()
        if lang != "python":
            return [], ""
        args: list[str] = ["python", "main.py"]
        if request.arguments:
            args.extend(list(request.arguments))
        return args, "main.py"

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        command, entry_name = self._build_command(request)
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
                target_entry = workdir / entry_name
                target_entry.write_text(request.code, encoding="utf-8")
                for name, content in (request.files or {}).items():
                    dest = (workdir / name).resolve()
                    if not str(dest).startswith(str(workdir.resolve())):
                        return SandboxResult(
                            stdout="",
                            stderr="",
                            exit_code=1,
                            artifacts=None,
                            sandbox_id=None,
                            timed_out=False,
                            error=f"Invalid file path: {name}",
                        )
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
                run_kwargs: dict[str, object] = {
                    "working_dir": "/workspace",
                    "environment": request.environment or {},
                    "volumes": {str(workdir): {"bind": "/workspace", "mode": "rw"}},
                    "detach": True,
                    "tty": False,
                }
                if self._network_mode:
                    run_kwargs["network_mode"] = self._network_mode
                else:
                    run_kwargs["network_disabled"] = True
                if self._mem_limit is not None:
                    run_kwargs["mem_limit"] = self._mem_limit
                if self._pids_limit is not None:
                    run_kwargs["pids_limit"] = self._pids_limit
                container = self._client.containers.run(self._image, command, **run_kwargs)
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
            wait_result: dict[str, int] | None = None
            container_obj: Any = container
            try:
                try:
                    wait_result = await asyncio.wait_for(
                        asyncio.to_thread(container_obj.wait),
                        timeout=request.timeout_s or self._timeout_s,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    try:
                        container_obj.kill()
                    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                        logger.debug("Failed to kill timed-out container: %s", exc)
                except asyncio.CancelledError:
                    try:
                        container_obj.kill()
                    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                        logger.debug("Failed to kill cancelled container: %s", exc)
                    raise
                except Exception as exc:  # noqa: BLE001 - unexpected wait failure
                    logger.debug("Container wait failed: %s", exc)
                    return SandboxResult(
                        stdout="",
                        stderr="",
                        exit_code=1,
                        artifacts=None,
                        sandbox_id=None,
                        timed_out=False,
                        error=f"Container wait failed: {exc}",
                    )

                try:
                    logs = container_obj.logs(stdout=True, stderr=True) or b""
                    stdout = logs.decode("utf-8", errors="replace")
                    stderr = ""
                except Exception as exc:  # noqa: BLE001 - logs are best-effort
                    logger.debug("Failed to read container logs: %s", exc)
                    stdout = ""
                    stderr = ""

                try:
                    status = wait_result if wait_result is not None else {"StatusCode": 1}
                    exit_code = int(status.get("StatusCode", 1))
                except Exception as exc:  # noqa: BLE001 - fall back to exit_code=1
                    logger.debug("Failed to extract container exit code: %s", exc)
                    exit_code = 1

                artifacts = self._collect_artifacts(workdir / "artifacts")

                return SandboxResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    artifacts=artifacts,
                    sandbox_id=None,
                    timed_out=timed_out,
                    error="Execution timed out" if timed_out else None,
                )
            finally:
                try:
                    container_obj.remove(force=True)
                except Exception as exc:  # noqa: BLE001 - cleanup best-effort
                    logger.debug("Failed to remove container: %s", exc)

    def _collect_artifacts(self, path: Path) -> dict[str, bytes] | None:
        if not path.exists():
            return None
        collected: dict[str, bytes] = {}
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    rel = item.relative_to(path).as_posix()
                    collected[rel] = item.read_bytes()
                except Exception:
                    continue
        return collected or None
