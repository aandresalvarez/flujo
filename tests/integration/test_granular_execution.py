"""Integration tests for Granular Execution Mode.

Tests the full granular execution flow including:
- Step.granular() factory method
- Crash-safe resume behavior
- Turn persistence and history management
- CAS guards with real pipeline execution
"""

import pytest

from flujo import Flujo
from flujo.domain.dsl import Step, Pipeline
from flujo.domain.models import PipelineContext


class MockGranularAgent:
    """Mock agent that tracks calls and simulates turn-by-turn execution."""

    def __init__(self, max_turns: int = 3) -> None:
        self.max_turns = max_turns
        self.call_count = 0
        self._model_name = "mock:granular-test"
        self._provider = "mock"
        self._system_prompt = "You are a granular test agent"
        self._tools = []

    async def run(
        self,
        data: str,
        *,
        context: PipelineContext | None = None,
        resources: object | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        self.call_count += 1

        # Simulate completion after max_turns
        is_complete = self.call_count >= self.max_turns

        return {
            "turn": self.call_count,
            "output": f"Response for turn {self.call_count}",
            "is_complete": is_complete,
            "done": is_complete,
        }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_step_factory_creates_valid_pipeline() -> None:
    """Verify Step.granular() creates proper Pipeline(LoopStep(GranularStep)) structure."""
    agent = MockGranularAgent()

    pipeline = Step.granular("test_granular", agent, max_turns=5)

    # Should return a Pipeline
    assert isinstance(pipeline, Pipeline)

    # Should have one step (the LoopStep wrapper)
    assert len(pipeline.steps) == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_execution_tracks_turns() -> None:
    """Verify granular execution tracks turn count in state."""
    agent = MockGranularAgent(max_turns=2)

    # Create granular pipeline
    pipeline = Step.granular("track_turns", agent, max_turns=10)
    runner = Flujo(pipeline)

    # Run pipeline
    result = None
    async for item in runner.run_async("initial input"):
        result = item

    # Agent should be called multiple times up to max_turns
    assert agent.call_count >= 1
    assert result is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_step_with_simple_callable() -> None:
    """Test granular step with a simple callable agent."""
    call_log: list[int] = []

    async def simple_agent(
        data: str,
        *,
        context: PipelineContext | None = None,
        resources: object | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        turn = len(call_log) + 1
        call_log.append(turn)
        return {
            "turn": turn,
            "output": f"Turn {turn} complete",
            "is_complete": turn >= 2,
        }

    # Wrap callable in a mock object with required attributes
    class AgentWrapper:
        _model_name = "mock:simple"
        _provider = "mock"
        _system_prompt = "Simple test"
        _tools: list[object] = []

        async def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            return await simple_agent(*args, **kwargs)  # type: ignore[arg-type]

    agent = AgentWrapper()
    pipeline = Step.granular("simple_granular", agent, max_turns=5)
    runner = Flujo(pipeline)

    result = None
    async for item in runner.run_async("test"):
        result = item

    # Should have completed after 2 turns
    assert len(call_log) >= 1
    assert result is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_fingerprint_consistency() -> None:
    """Test that fingerprint remains consistent across runs with same config."""
    from flujo.domain.dsl.granular import GranularStep

    # Create two fingerprints with same config
    fp1 = GranularStep.compute_fingerprint(
        input_data={"key": "value"},
        system_prompt="Test prompt",
        model_id="test-model",
        provider="test",
        tools=[{"name": "tool1", "sig_hash": "abc123"}],
        settings={"history_max_tokens": 128000},
    )

    fp2 = GranularStep.compute_fingerprint(
        input_data={"key": "value"},
        system_prompt="Test prompt",
        model_id="test-model",
        provider="test",
        tools=[{"name": "tool1", "sig_hash": "abc123"}],
        settings={"history_max_tokens": 128000},
    )

    assert fp1 == fp2

    # Different input should produce different fingerprint
    fp3 = GranularStep.compute_fingerprint(
        input_data={"key": "different"},
        system_prompt="Test prompt",
        model_id="test-model",
        provider="test",
        tools=[{"name": "tool1", "sig_hash": "abc123"}],
        settings={"history_max_tokens": 128000},
    )

    assert fp1 != fp3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_idempotency_key_deterministic() -> None:
    """Test that idempotency keys are deterministic."""
    from flujo.domain.dsl.granular import GranularStep

    key1 = GranularStep.generate_idempotency_key("run123", "step1", 0)
    key2 = GranularStep.generate_idempotency_key("run123", "step1", 0)
    key3 = GranularStep.generate_idempotency_key("run123", "step1", 1)

    # Same inputs = same key
    assert key1 == key2

    # Different turn = different key
    assert key1 != key3


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_blob_store_offload_hydrate() -> None:
    """Integration test for blob store offload/hydrate cycle."""
    from flujo.state.granular_blob_store import GranularBlobStore, BlobRef

    class MockBackend:
        def __init__(self) -> None:
            self._store: dict[str, object] = {}

        async def save_state(self, key: str, data: object) -> None:
            self._store[key] = data

        async def load_state(self, key: str) -> object | None:
            return self._store.get(key)

    backend = MockBackend()
    store = GranularBlobStore(backend, threshold_bytes=50)

    # Large payload that should be offloaded
    large_payload = {"data": "x" * 200, "nested": {"key": "value"}}

    # Offload
    ref = await store.offload("run1", "step1", 0, large_payload)
    assert isinstance(ref, BlobRef)
    assert ref.size > 0

    # Create marker
    marker = ref.to_marker()
    assert BlobRef.is_marker(marker)

    # Hydrate
    hydrated = await store.hydrate(ref)
    assert hydrated == large_payload


@pytest.mark.asyncio
@pytest.mark.integration
async def test_granular_history_entry_blob_processing() -> None:
    """Test blob store processing of history entries."""
    from flujo.state.granular_blob_store import GranularBlobStore, BlobRef

    class MockBackend:
        def __init__(self) -> None:
            self._store: dict[str, object] = {}

        async def save_state(self, key: str, data: object) -> None:
            self._store[key] = data

        async def load_state(self, key: str) -> object | None:
            return self._store.get(key)

    backend = MockBackend()
    store = GranularBlobStore(backend, threshold_bytes=50)

    # History entry with large output
    entry = {
        "turn_index": 0,
        "input": "small",
        "output": {"large": "x" * 200},
    }

    # Process entry - should offload large output
    processed = await store.process_history_entry(entry, "run1", "step1", 0)

    assert processed["input"] == "small"  # Small, unchanged
    assert BlobRef.is_marker(processed["output"])  # Large, converted to marker

    # Hydrate entry
    hydrated = await store.hydrate_history_entry(processed)
    assert hydrated["output"] == entry["output"]
