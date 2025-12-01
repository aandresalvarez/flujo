import time
from typing import Any
from pydantic import BaseModel
from flujo.application.core.state_serializer import StateSerializer


class MockContext(BaseModel):
    run_id: str = "run_123"
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


def benchmark_hashing():
    serializer = StateSerializer()

    # Small context
    small_ctx = MockContext(data={"a": 1, "b": "test"}, metadata={"step": 1})

    # Medium context (borderline large)
    medium_ctx = MockContext(
        data={f"k{i}": f"v{i}" for i in range(20)},
        metadata={"history": ["step1", "step2", "step3"]},
    )

    # Large context
    large_ctx = MockContext(
        data={f"k{i}": {"nested": f"v{i}" * 10} for i in range(100)},
        metadata={"logs": ["log" * 50 for _ in range(50)]},
    )

    print("Benchmarking compute_context_hash...")

    start = time.perf_counter()
    for _ in range(1000):
        serializer.compute_context_hash(small_ctx)
    end = time.perf_counter()
    print(f"Small context (1000 iter): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        serializer.compute_context_hash(medium_ctx)
    end = time.perf_counter()
    print(f"Medium context (1000 iter): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        serializer.compute_context_hash(large_ctx)
    end = time.perf_counter()
    print(f"Large context (1000 iter): {end - start:.4f}s")

    # Benchmark StateManager's inefficient logic
    print("\nBenchmarking StateManager logic (str(v))...")

    def state_manager_hash_logic(context):
        data = context.model_dump()
        # The problematic line from StateManager
        if len(data) > 10 or any(
            isinstance(v, (list, dict)) and len(str(v)) > 1000 for v in data.values()
        ):
            return "large"
        return "small"

    start = time.perf_counter()
    for _ in range(1000):
        state_manager_hash_logic(small_ctx)
    end = time.perf_counter()
    print(f"Small context (StateManager): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        state_manager_hash_logic(medium_ctx)
    end = time.perf_counter()
    print(f"Medium context (StateManager): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        state_manager_hash_logic(large_ctx)
    end = time.perf_counter()
    print(f"Large context (StateManager): {end - start:.4f}s")


if __name__ == "__main__":
    benchmark_hashing()
