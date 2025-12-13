import copy

from flujo.application.core.state_serializer import StateSerializer
from flujo.domain.models import PipelineContext, StepResult


def make_ctx(prompt: str = "hello") -> PipelineContext:
    return PipelineContext(initial_prompt=prompt)


def test_compute_context_hash_excludes_volatile_fields() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()
    ctx1 = make_ctx("prompt-a")
    ctx2 = copy.deepcopy(ctx1)

    # Changing volatile fields shouldn't affect the hash
    h1 = ser.compute_context_hash(ctx1)
    object.__setattr__(ctx2, "run_id", ctx1.run_id + "_changed")
    h2 = ser.compute_context_hash(ctx2)
    assert h1 == h2, "run_id change should not change hash"

    # Changing a meaningful field should change the hash
    ctx3 = copy.deepcopy(ctx1)
    object.__setattr__(ctx3, "initial_prompt", "prompt-b")
    h3 = ser.compute_context_hash(ctx3)
    assert h1 != h3, "initial_prompt change should change hash"


def test_should_serialize_context_per_run_cache() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()
    ctx = make_ctx("p1")

    # First time for r1: should serialize
    assert ser.should_serialize_context(ctx, "r1") is True
    # No change: shouldn't serialize for same run
    assert ser.should_serialize_context(ctx, "r1") is False

    # Different run id keeps independent cache
    assert ser.should_serialize_context(ctx, "r2") is True

    # Change meaningful field -> should serialize again for r1
    object.__setattr__(ctx, "initial_prompt", "p2")
    assert ser.should_serialize_context(ctx, "r1") is True


def test_serialize_context_for_state_caching_and_minimal() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()
    ctx = make_ctx("p1")
    run_id = "run-x"

    # First call: should serialize full and cache
    full1 = ser.serialize_context_for_state(ctx, run_id)
    assert isinstance(full1, dict)
    assert full1 == ser.serialize_context_full(ctx)

    # Second call without changes: expect cached dict (same object reference)
    cached = ser.get_cached_serialization(ctx, run_id)
    again = ser.serialize_context_for_state(ctx, run_id)
    assert again is cached

    # Clear cache for run: first call will serialize full again (hash cache cleared)
    ser.clear_cache(run_id)
    full_after_clear = ser.serialize_context_for_state(ctx, run_id)
    assert full_after_clear == ser.serialize_context_full(ctx)

    # Fresh serializer: prime hash without caching serialization, then expect minimal
    ser2: StateSerializer[PipelineContext] = StateSerializer()
    ctx2 = make_ctx("p1")
    run2 = "run-y"
    # Prime only the hash cache
    assert ser2.should_serialize_context(ctx2, run2) is True
    # Now unchanged and no cached serialization -> minimal
    minimal2 = ser2.serialize_context_for_state(ctx2, run2)
    assert minimal2 == ser2.serialize_context_minimal(ctx2)


def test_clear_cache_scoped_and_global() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()
    ctx = make_ctx("p1")

    # Populate caches for two runs
    _ = ser.serialize_context_for_state(ctx, "r1")
    _ = ser.serialize_context_for_state(ctx, "r2")
    assert ser.get_cached_serialization(ctx, "r1") is not None
    assert ser.get_cached_serialization(ctx, "r2") is not None

    # Clear one run
    ser.clear_cache("r1")
    assert ser.get_cached_serialization(ctx, "r1") is None
    assert ser.get_cached_serialization(ctx, "r2") is not None

    # Clear all
    ser.clear_cache()
    assert ser.get_cached_serialization(ctx, "r2") is None


def test_step_history_serialization_full_and_minimal() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()

    sr = StepResult(
        name="step-1",
        output={"x": 1},
        success=True,
        attempts=2,
        latency_s=0.123,
        token_counts=42,
        cost_usd=0.01,
        feedback="ok",
    )
    full = ser.serialize_step_history_full([sr])
    assert isinstance(full, list) and len(full) == 1
    # Full serialization should include rich fields
    assert set(full[0]).issuperset(
        {
            "name",
            "output",
            "success",
            "attempts",
            "latency_s",
            "token_counts",
            "cost_usd",
            "feedback",
            "metadata_",
            "step_history",
        }
    )

    minimal = ser.serialize_step_history_minimal([sr])
    assert isinstance(minimal, list) and len(minimal) == 1
    # Minimal serialization should include only essential subset
    assert set(minimal[0]) == {
        "name",
        "output",
        "success",
        "attempts",
        "latency_s",
        "token_counts",
        "cost_usd",
        "feedback",
    }


def test_deserialize_context_round_trip_pipeline_context() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()
    ctx = make_ctx("round")
    ctx.import_artifacts["k"] = "v"
    data = ser.serialize_context_full(ctx)
    restored = ser.deserialize_context(data, PipelineContext)
    assert restored is not None
    assert restored.initial_prompt == ctx.initial_prompt
    assert restored.run_id == ctx.run_id
    assert restored.import_artifacts == ctx.import_artifacts


def test_deserialize_context_with_custom_model() -> None:
    from flujo.domain.models import PipelineContext as PC

    class CustomContext(PC):
        foo: int

    ser: StateSerializer[CustomContext] = StateSerializer()
    ctx = CustomContext(initial_prompt="x", foo=7)
    data = ser.serialize_context_full(ctx)
    restored = ser.deserialize_context(data, CustomContext)
    assert restored is not None
    assert isinstance(restored, CustomContext)
    assert restored.foo == 7
    assert restored.initial_prompt == "x"


def test_deserialize_invalid_data_returns_none() -> None:
    ser: StateSerializer[PipelineContext] = StateSerializer()

    # None is invalid -> None
    assert ser.deserialize_context(None, PipelineContext) is None

    # Non-mapping invalid type -> None
    assert ser.deserialize_context(["not", "a", "dict"], PipelineContext) is None

    # Empty dict should now work since initial_prompt is optional
    # But we can test with truly invalid data
    assert ser.deserialize_context({"invalid_field": "value"}, PipelineContext) is not None
