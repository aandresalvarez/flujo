import httpx
import pytest

from flujo.domain.sandbox import SandboxExecution
from flujo.infra.sandbox import RemoteSandbox


@pytest.mark.asyncio
async def test_remote_sandbox_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stdout": "ok",
                "stderr": "",
                "exit_code": 0,
                "artifacts": {"file.txt": "ZGF0YQ=="},
                "sandbox_id": "abc",
                "timed_out": False,
                "error": None,
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://sandbox")
    sandbox = RemoteSandbox(api_url="https://sandbox", client=client)

    result = await sandbox.exec_code(
        SandboxExecution(code="print('hi')", language="python", arguments=("--x",))
    )

    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.artifacts and result.artifacts["file.txt"] == b"data"


@pytest.mark.asyncio
async def test_remote_sandbox_artifact_variants() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "artifacts": {
                    "plain.txt": "hello",  # utf-8 string
                    "b64.txt": {"base64": "ZGF0YQ=="},
                    "url.txt": {"url": "https://sandbox/file"},
                },
            },
        )

    async def artifact_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"from-url")

    transport = httpx.MockTransport(
        lambda req: handler(req) if req.url.path == "/execute" else artifact_handler(req)
    )
    client = httpx.AsyncClient(transport=transport, base_url="https://sandbox")
    sandbox = RemoteSandbox(api_url="https://sandbox", client=client)

    result = await sandbox.exec_code(SandboxExecution(code="x", language="python"))

    assert result.artifacts is not None
    assert result.artifacts["plain.txt"] == b"hello"
    assert result.artifacts["b64.txt"] == b"data"
    assert result.artifacts["url.txt"] == b"from-url"


@pytest.mark.asyncio
async def test_remote_sandbox_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://sandbox")
    sandbox = RemoteSandbox(api_url="https://sandbox", client=client)

    result = await sandbox.exec_code(SandboxExecution(code="x", language="python"))

    assert result.exit_code == 500
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_remote_sandbox_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://sandbox")
    sandbox = RemoteSandbox(api_url="https://sandbox", client=client, timeout_s=0.01)

    result = await sandbox.exec_code(SandboxExecution(code="x", language="python"))

    assert result.timed_out is True
    assert result.exit_code == 1
    assert "timeout" in (result.error or "")
