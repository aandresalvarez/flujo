import pytest

from flujo.state.backends.memory import InMemoryBackend


@pytest.mark.asyncio
async def test_inmemory_backend_roundtrip() -> None:
    backend = InMemoryBackend()
    await backend.save_state("run1", {"foo": 1})
    loaded = await backend.load_state("run1")
    assert loaded == {"foo": 1}
    await backend.delete_state("run1")
    assert await backend.load_state("run1") is None
