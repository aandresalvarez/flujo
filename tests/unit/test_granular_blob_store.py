"""Unit tests for GranularBlobStore."""

import pytest

from flujo.state.granular_blob_store import (
    GranularBlobStore,
    BlobRef,
    BlobNotFoundError,
)


class MockBackend:
    """Mock state backend for testing."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def save_state(self, key: str, data: object) -> None:
        self._store[key] = data

    async def load_state(self, key: str) -> object | None:
        return self._store.get(key)


class TestBlobRef:
    """Tests for BlobRef marker handling."""

    def test_to_marker(self) -> None:
        ref = BlobRef("abc123", 1024)
        marker = ref.to_marker()
        assert marker == "<<FL_BLOB_REF:abc123:size=1024>>"

    def test_from_marker_valid(self) -> None:
        marker = "<<FL_BLOB_REF:abc123:size=1024>>"
        ref = BlobRef.from_marker(marker)
        assert ref is not None
        assert ref.blob_id == "abc123"
        assert ref.size == 1024

    def test_from_marker_invalid(self) -> None:
        assert BlobRef.from_marker("not a marker") is None
        assert BlobRef.from_marker("<<FL_BLOB_REF:missing_size>>") is None
        assert BlobRef.from_marker("<<FL_BLOB_REF:id:invalid>>") is None

    def test_is_marker(self) -> None:
        assert BlobRef.is_marker("<<FL_BLOB_REF:abc:size=100>>") is True
        assert BlobRef.is_marker("regular string") is False
        assert BlobRef.is_marker(12345) is False
        assert BlobRef.is_marker(None) is False

    def test_roundtrip(self) -> None:
        original = BlobRef("test_id", 5000)
        marker = original.to_marker()
        parsed = BlobRef.from_marker(marker)
        assert parsed is not None
        assert parsed.blob_id == original.blob_id
        assert parsed.size == original.size


class TestGranularBlobStore:
    """Tests for GranularBlobStore operations."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend()

    @pytest.fixture
    def store(self, backend: MockBackend) -> GranularBlobStore:
        return GranularBlobStore(backend, threshold_bytes=100)

    def test_should_offload_small(self, store: GranularBlobStore) -> None:
        small_payload = {"key": "value"}
        assert store.should_offload(small_payload) is False

    def test_should_offload_large(self, store: GranularBlobStore) -> None:
        large_payload = {"data": "x" * 200}
        assert store.should_offload(large_payload) is True

    def test_should_offload_non_serializable(self, store: GranularBlobStore) -> None:
        non_serializable = {"func": lambda x: x}
        assert store.should_offload(non_serializable) is False

    @pytest.mark.asyncio
    async def test_offload_and_hydrate(self, store: GranularBlobStore) -> None:
        payload = {"large_data": "x" * 500, "nested": {"key": "value"}}

        ref = await store.offload("run1", "step1", 0, payload)

        assert ref.blob_id is not None
        assert ref.size > 0

        hydrated = await store.hydrate(ref)
        assert hydrated == payload

    @pytest.mark.asyncio
    async def test_hydrate_not_found(self, store: GranularBlobStore) -> None:
        ref = BlobRef("nonexistent_id", 100)

        with pytest.raises(BlobNotFoundError) as exc_info:
            await store.hydrate(ref)

        assert "nonexistent_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_hydrate_marker(self, store: GranularBlobStore) -> None:
        payload = {"test": "data" * 50}
        ref = await store.offload("run1", "step1", 0, payload)
        marker = ref.to_marker()

        hydrated = await store.hydrate_marker(marker)
        assert hydrated == payload

    @pytest.mark.asyncio
    async def test_hydrate_invalid_marker(self, store: GranularBlobStore) -> None:
        with pytest.raises(ValueError):
            await store.hydrate_marker("not a marker")

    @pytest.mark.asyncio
    async def test_process_history_entry_offload(self, store: GranularBlobStore) -> None:
        entry = {
            "turn_index": 0,
            "input": "small input",
            "output": {"large": "x" * 500},
        }

        processed = await store.process_history_entry(entry, "run1", "step1", 0)

        # Input should be unchanged (small)
        assert processed["input"] == "small input"
        # Output should be a marker (large)
        assert BlobRef.is_marker(processed["output"])

    @pytest.mark.asyncio
    async def test_hydrate_history_entry(self, store: GranularBlobStore) -> None:
        # First offload
        large_output = {"large": "x" * 500}
        ref = await store.offload("run1", "step1", 0, large_output)

        entry = {
            "turn_index": 0,
            "input": "small input",
            "output": ref.to_marker(),
        }

        hydrated = await store.hydrate_history_entry(entry)

        assert hydrated["input"] == "small input"
        assert hydrated["output"] == large_output

    def test_generate_blob_id_uniqueness(self, store: GranularBlobStore) -> None:
        id1 = store.generate_blob_id("run1", "step1", 0, "hash1")
        id2 = store.generate_blob_id("run1", "step1", 0, "hash2")
        id3 = store.generate_blob_id("run1", "step1", 1, "hash1")

        # All should be unique due to timestamp component
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
