from pathlib import Path

import pytest

from flujo.state.backends.file import FileBackend

MALICIOUS_PAYLOADS = [
    "../outside_file.txt",
    "..%2foutside_file.txt",
    "....//outside_file.txt",
    "/etc/hostname",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
]


@pytest.fixture()
def sandbox(tmp_path: Path):
    safe_zone = tmp_path / "safe_zone"
    safe_zone.mkdir()
    outside = tmp_path / "outside_file.txt"
    outside.write_text("secret")
    backend = FileBackend(safe_zone)
    return backend, outside


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", MALICIOUS_PAYLOADS)
async def test_save_state_path_traversal(payload: str, sandbox) -> None:
    backend, outside = sandbox
    before = outside.read_text()
    try:
        await backend.save_state(payload, {"data": "malicious"})
    except ValueError:
        pass
    assert outside.read_text() == before
    files_outside = [
        p for p in outside.parent.rglob("*") if p.is_file() and not p.is_relative_to(backend.path)
    ]
    assert files_outside == [outside]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", MALICIOUS_PAYLOADS)
async def test_load_state_path_traversal(payload: str, sandbox) -> None:
    backend, _ = sandbox
    try:
        result = await backend.load_state(payload)
    except ValueError:
        return
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", MALICIOUS_PAYLOADS)
async def test_delete_state_path_traversal(payload: str, sandbox) -> None:
    backend, outside = sandbox
    before = outside.read_text()
    try:
        await backend.delete_state(payload)
    except ValueError:
        pass
    assert outside.read_text() == before
    assert outside.exists()
