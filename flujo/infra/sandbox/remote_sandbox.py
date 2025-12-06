from __future__ import annotations

import base64
from typing import Any, Final, Mapping, MutableMapping

import httpx

from ...domain.sandbox import SandboxExecution, SandboxProtocol, SandboxResult


async def _materialize_artifacts(
    data: Mapping[str, Any] | None,
    client: httpx.AsyncClient,
    timeout_s: float,
) -> MutableMapping[str, bytes] | None:
    if not data:
        return None
    decoded: dict[str, bytes] = {}
    for key, value in data.items():
        try:
            if isinstance(value, str):
                try:
                    decoded[key] = base64.b64decode(value)
                except Exception:
                    decoded[key] = value.encode("utf-8")
                continue
            if isinstance(value, Mapping):
                if "base64" in value:
                    try:
                        decoded[key] = base64.b64decode(str(value["base64"]))
                        continue
                    except Exception:
                        decoded[key] = b""
                        continue
                if "data" in value:
                    decoded[key] = str(value["data"]).encode("utf-8")
                    continue
                if "url" in value:
                    url = str(value["url"])
                    try:
                        resp = await client.get(url, timeout=timeout_s)
                        decoded[key] = resp.content
                        continue
                    except Exception:
                        decoded[key] = b""
                        continue
            decoded[key] = b""
        except Exception:
            decoded[key] = b""
    return decoded


class RemoteSandbox(SandboxProtocol):
    """HTTP-based sandbox that proxies code execution to a remote API."""

    _DEFAULT_PATH: Final[str] = "/execute"

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        verify_ssl: bool = True,
        endpoint_path: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_url:
            raise ValueError("RemoteSandbox requires a non-empty api_url")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._endpoint_path = endpoint_path or self._DEFAULT_PATH
        self._client = client or httpx.AsyncClient(
            base_url=api_url, timeout=timeout_s, verify=verify_ssl
        )

    async def exec_code(self, request: SandboxExecution) -> SandboxResult:
        payload = {
            "code": request.code,
            "language": request.language,
            "files": request.files or {},
            "environment": request.environment or {},
            "arguments": list(request.arguments or ()),
            "timeout_s": request.timeout_s or self._timeout_s,
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = await self._client.post(self._endpoint_path, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=1,
                artifacts=None,
                sandbox_id=None,
                timed_out=True,
                error=str(exc),
            )
        except httpx.RequestError as exc:
            return SandboxResult(
                stdout="",
                stderr="",
                exit_code=1,
                artifacts=None,
                sandbox_id=None,
                timed_out=False,
                error=str(exc),
            )

        try:
            data = resp.json()
        except Exception:
            data = {}

        stdout = str(data.get("stdout", ""))
        stderr = str(data.get("stderr", ""))
        exit_code = int(data.get("exit_code", resp.status_code))
        sandbox_id = data.get("sandbox_id")
        timed_out = bool(data.get("timed_out", False))
        error = data.get("error")
        artifacts = await _materialize_artifacts(
            data.get("artifacts"), self._client, self._timeout_s
        )

        if resp.status_code >= 400 and error is None:
            error = f"Remote sandbox HTTP {resp.status_code}"

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            artifacts=artifacts,
            sandbox_id=sandbox_id,
            timed_out=timed_out,
            error=str(error) if error is not None else None,
        )
