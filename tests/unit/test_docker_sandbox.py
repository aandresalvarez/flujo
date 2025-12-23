import asyncio
import pytest

from flujo.domain.sandbox import SandboxExecution, SandboxResult
from flujo.infra.sandbox.docker_sandbox import DockerSandbox


class _FakeContainer:
    def __init__(self, *, status: int = 0, logs: bytes = b"ok") -> None:
        self._status = status
        self._logs = logs
        self.killed = False
        self.removed = False

    def wait(self) -> dict[str, int]:
        return {"StatusCode": self._status}

    def kill(self) -> None:
        self.killed = True

    def remove(self, force: bool = False) -> None:
        self.removed = True

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        return self._logs


class _FakeImages:
    def __init__(self) -> None:
        self.pulled = []

    def pull(self, image: str) -> None:
        self.pulled.append(image)


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.last_args = None

    def run(self, *args, **kwargs):
        self.last_args = (args, kwargs)
        return self._container


class _FakeClient:
    def __init__(self, *, container: _FakeContainer) -> None:
        self.images = _FakeImages()
        self.containers = _FakeContainers(container)


@pytest.mark.asyncio
async def test_docker_sandbox_runs_python() -> None:
    container = _FakeContainer(status=0, logs=b"hello")
    client = _FakeClient(container=container)
    sandbox = DockerSandbox(client=client, image="python:3.13-slim", pull=False)

    result = await sandbox.exec_code(SandboxExecution(code="print('hi')", language="python"))

    assert isinstance(result, SandboxResult)
    assert result.stdout == "hello"
    assert result.exit_code == 0
    assert container.removed is True
    # With pull disabled in the test client, pulled list may be empty
    assert client.images.pulled in ([], ["python:3.13-slim"])


@pytest.mark.asyncio
async def test_docker_sandbox_applies_limits() -> None:
    container = _FakeContainer(status=0, logs=b"ok")
    client = _FakeClient(container=container)
    sandbox = DockerSandbox(
        client=client,
        image="python:3.13-slim",
        pull=False,
        mem_limit="256m",
        pids_limit=128,
        network_mode="none",
    )

    await sandbox.exec_code(SandboxExecution(code="print('hi')", language="python"))

    assert client.containers.last_args is not None
    _, kwargs = client.containers.last_args
    assert kwargs["mem_limit"] == "256m"
    assert kwargs["pids_limit"] == 128
    assert kwargs["network_mode"] == "none"
    assert "network_disabled" not in kwargs


@pytest.mark.asyncio
async def test_docker_sandbox_timeout() -> None:
    class SlowContainer(_FakeContainer):
        def wait(self) -> dict[str, int]:
            raise asyncio.TimeoutError()

    container = SlowContainer()
    client = _FakeClient(container=container)
    sandbox = DockerSandbox(client=client, image="python:3.13-slim", pull=False, timeout_s=0.01)

    result = await sandbox.exec_code(SandboxExecution(code="print('hi')", language="python"))

    assert result.timed_out is True
    assert container.killed is True
    assert container.removed is True


@pytest.mark.asyncio
async def test_docker_sandbox_rejects_non_python() -> None:
    container = _FakeContainer()
    client = _FakeClient(container=container)
    sandbox = DockerSandbox(client=client, image="python:3.13-slim", pull=False)

    result = await sandbox.exec_code(SandboxExecution(code="echo hi", language="bash"))

    assert result.exit_code == 1
    assert "Unsupported language" in (result.error or "")


@pytest.mark.asyncio
async def test_docker_sandbox_cleanup_on_cancellation() -> None:
    """Container should be removed even when execution is cancelled."""

    class BlockingContainer(_FakeContainer):
        def wait(self) -> dict[str, int]:
            import time

            time.sleep(1.0)  # Block long enough for cancellation
            return {"StatusCode": 137}

    container = BlockingContainer()
    client = _FakeClient(container=container)
    sandbox = DockerSandbox(client=client, image="python:3.13-slim", pull=False, timeout_s=5)

    task = asyncio.create_task(
        sandbox.exec_code(SandboxExecution(code="print('hi')", language="python"))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert container.killed is True
    assert container.removed is True
