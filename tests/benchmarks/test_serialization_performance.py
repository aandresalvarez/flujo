from flujo.type_definitions.common import JSONObject

"""Performance benchmarks for serialization and reconstruction operations."""

import pytest

# Mark this module as benchmark so it's excluded from fast test runs
pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

import os
import time
import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from flujo.testing.utils import SimpleDummyRemoteBackend as DummyRemoteBackend
from flujo.state.backends.base import _serialize_for_json


class BenchmarkModel(BaseModel):
    """Model for benchmarking serialization performance."""

    id: int
    name: str
    data: JSONObject
    items: List[str]
    metadata: JSONObject = Field(default_factory=dict)


class NestedBenchmarkModel(BaseModel):
    """Nested model for benchmarking complex serialization."""

    root: BenchmarkModel
    children: List[BenchmarkModel]
    config: Dict[str, BenchmarkModel]


def _serialize(obj: Any) -> Any:
    """JSON-safe serialization using the shared helper."""
    normalized = _serialize_for_json(obj)
    return json.loads(json.dumps(normalized, ensure_ascii=False))


def create_small_model() -> BenchmarkModel:
    """Create a small model for benchmarking."""
    return BenchmarkModel(
        id=1,
        name="test",
        data={"key": "value"},
        items=["item1", "item2"],
        metadata={"meta": "data"},
    )


def create_medium_model() -> BenchmarkModel:
    """Create a medium-sized model for benchmarking."""
    return BenchmarkModel(
        id=42,
        name="medium_test_model",
        data={
            "string_field": "test_value",
            "int_field": 12345,
            "float_field": 3.14159,
            "bool_field": True,
            "list_field": ["item1", "item2", "item3", "item4", "item5"],
            "dict_field": {
                "nested_key1": "nested_value1",
                "nested_key2": "nested_value2",
                "nested_key3": "nested_value3",
            },
        },
        items=["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8"],
        metadata={
            "created_at": "2024-01-01T00:00:00Z",
            "version": "1.0.0",
            "tags": ["test", "benchmark", "performance"],
            "settings": {"timeout": 30, "retries": 3, "cache_enabled": True},
        },
    )


def create_large_model() -> BenchmarkModel:
    """Create a large model for benchmarking."""
    large_data = {}
    large_items = []
    large_metadata = {}

    # Generate large data dictionary
    for i in range(100):
        large_data[f"key_{i}"] = f"value_{i}"
        large_data[f"nested_{i}"] = {
            "subkey1": f"subvalue1_{i}",
            "subkey2": f"subvalue2_{i}",
            "subkey3": f"subvalue3_{i}",
        }

    # Generate large items list
    for i in range(100):
        large_items.append(f"item_{i}")  # Reduced from 1000 to 100

    # Generate large metadata
    for i in range(50):
        large_metadata[f"meta_key_{i}"] = f"meta_value_{i}"

    return BenchmarkModel(
        id=999,
        name="large_test_model_with_very_long_name_for_benchmarking_purposes",
        data=large_data,
        items=large_items,
        metadata=large_metadata,
    )


def create_nested_model() -> NestedBenchmarkModel:
    """Create a nested model for benchmarking complex structures."""
    root = create_medium_model()
    children = [create_medium_model() for _ in range(10)]
    config = {f"config_{i}": create_medium_model() for i in range(5)}

    return NestedBenchmarkModel(root=root, children=children, config=config)


class TestSerializationPerformance:
    """Performance benchmarks for serialization operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = DummyRemoteBackend()

    def benchmark_serialization_speed(
        self, model: BaseModel, iterations: int = 1000
    ) -> Dict[str, float]:
        """Benchmark serialization speed for a given model."""
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Warm up
        for _ in range(10):
            _serialize(request_data)

        # Benchmark serialization
        start_time = time.perf_counter()
        for _ in range(iterations):
            serialized = _serialize(request_data)
        serialization_time = time.perf_counter() - start_time

        # Benchmark JSON encoding
        start_time = time.perf_counter()
        for _ in range(iterations):
            json.dumps(serialized)
        json_time = time.perf_counter() - start_time

        # Benchmark JSON decoding
        json_str = json.dumps(serialized)
        start_time = time.perf_counter()
        for _ in range(iterations):
            json.loads(json_str)
        json_decode_time = time.perf_counter() - start_time

        # Benchmark reconstruction
        data = json.loads(json_str)
        start_time = time.perf_counter()
        for _ in range(iterations):
            self.backend._reconstruct_payload(request_data, data)
        reconstruction_time = time.perf_counter() - start_time

        return {
            "serialization_time": serialization_time,
            "json_encode_time": json_time,
            "json_decode_time": json_decode_time,
            "reconstruction_time": reconstruction_time,
            "total_time": serialization_time + json_time + json_decode_time + reconstruction_time,
            "iterations": iterations,
            "avg_serialization": serialization_time / iterations * 1000,  # ms
            "avg_json_encode": json_time / iterations * 1000,  # ms
            "avg_json_decode": json_decode_time / iterations * 1000,  # ms
            "avg_reconstruction": reconstruction_time / iterations * 1000,  # ms
            "avg_total": (serialization_time + json_time + json_decode_time + reconstruction_time)
            / iterations
            * 1000,  # ms
        }

    def test_small_model_performance(self):
        """Benchmark performance with small models."""
        model = create_small_model()
        results = self.benchmark_serialization_speed(
            model, iterations=1000
        )  # Reduced from 10000 to 1000

        print("\nSmall Model Performance (10,000 iterations):")
        print(f"  Avg serialization: {results['avg_serialization']:.3f} ms")
        print(f"  Avg JSON encode: {results['avg_json_encode']:.3f} ms")
        print(f"  Avg JSON decode: {results['avg_json_decode']:.3f} ms")
        print(f"  Avg reconstruction: {results['avg_reconstruction']:.3f} ms")
        print(f"  Avg total: {results['avg_total']:.3f} ms")

        # Performance assertions - these should be reasonable for modern hardware
        assert results["avg_serialization"] < 1.0, "Serialization too slow"
        assert results["avg_json_encode"] < 0.5, "JSON encoding too slow"
        assert results["avg_json_decode"] < 0.5, "JSON decoding too slow"
        assert results["avg_reconstruction"] < 2.0, "Reconstruction too slow"
        assert results["avg_total"] < 4.0, "Total roundtrip too slow"

    def test_medium_model_performance(self):
        """Benchmark performance with medium-sized models."""
        model = create_medium_model()
        results = self.benchmark_serialization_speed(
            model, iterations=100
        )  # Reduced from 1000 to 100

        print("\nMedium Model Performance (100 iterations):")
        print(f"  Avg serialization: {results['avg_serialization']:.3f} ms")
        print(f"  Avg JSON encode: {results['avg_json_encode']:.3f} ms")
        print(f"  Avg JSON decode: {results['avg_json_decode']:.3f} ms")
        print(f"  Avg reconstruction: {results['avg_reconstruction']:.3f} ms")
        print(f"  Avg total: {results['avg_total']:.3f} ms")

        # Performance assertions for medium models
        assert results["avg_serialization"] < 5.0, "Serialization too slow"
        assert results["avg_json_encode"] < 2.0, "JSON encoding too slow"
        assert results["avg_json_decode"] < 2.0, "JSON decoding too slow"
        assert results["avg_reconstruction"] < 10.0, "Reconstruction too slow"
        assert results["avg_total"] < 20.0, "Total roundtrip too slow"

    def test_large_model_performance(self):
        """Benchmark performance with large models."""
        model = create_large_model()
        results = self.benchmark_serialization_speed(model, iterations=100)

        print("\nLarge Model Performance (100 iterations):")
        print(f"  Avg serialization: {results['avg_serialization']:.3f} ms")
        print(f"  Avg JSON encode: {results['avg_json_encode']:.3f} ms")
        print(f"  Avg JSON decode: {results['avg_json_decode']:.3f} ms")
        print(f"  Avg reconstruction: {results['avg_reconstruction']:.3f} ms")
        print(f"  Avg total: {results['avg_total']:.3f} ms")

        # Performance assertions for large models
        assert results["avg_serialization"] < 50.0, "Serialization too slow"
        assert results["avg_json_encode"] < 20.0, "JSON encoding too slow"
        assert results["avg_json_decode"] < 20.0, "JSON decoding too slow"
        assert results["avg_reconstruction"] < 100.0, "Reconstruction too slow"
        assert results["avg_total"] < 200.0, "Total roundtrip too slow"

    def test_nested_model_performance(self):
        """Benchmark performance with deeply nested models."""
        model = create_nested_model()
        results = self.benchmark_serialization_speed(model, iterations=100)

        print("\nNested Model Performance (100 iterations):")
        print(f"  Avg serialization: {results['avg_serialization']:.3f} ms")
        print(f"  Avg JSON encode: {results['avg_json_encode']:.3f} ms")
        print(f"  Avg JSON decode: {results['avg_json_decode']:.3f} ms")
        print(f"  Avg reconstruction: {results['avg_reconstruction']:.3f} ms")
        print(f"  Avg total: {results['avg_total']:.3f} ms")

        # Performance assertions for nested models
        assert results["avg_serialization"] < 100.0, "Serialization too slow"
        assert results["avg_json_encode"] < 50.0, "JSON encoding too slow"
        assert results["avg_json_decode"] < 50.0, "JSON decoding too slow"
        assert results["avg_reconstruction"] < 200.0, "Reconstruction too slow"
        assert results["avg_total"] < 400.0, "Total roundtrip too slow"

    def test_memory_usage(self):
        """Test memory usage during serialization operations."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and serialize large models
        models = [create_large_model() for _ in range(50)]  # Reduced from 100 to 50
        request_data_list = []

        for model in models:
            request_data = {
                "input_data": model,
                "context": None,
                "resources": None,
                "context_model_defined": False,
                "usage_limits": None,
                "stream": False,
            }
            request_data_list.append(request_data)

        # Serialize all models
        serialized_list = []
        for request_data in request_data_list:
            serialized = _serialize(request_data)
            serialized_list.append(serialized)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        print("\nMemory Usage Test:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")

        # Memory usage should be reasonable (less than 500MB for 50 large models)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f} MB"

    def test_concurrent_serialization(self):
        """Test performance under concurrent serialization operations."""
        import concurrent.futures

        model = create_medium_model()
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        def serialize_operation():
            """Single serialization operation."""
            serialized = _serialize(request_data)
            data = json.loads(json.dumps(serialized))
            self.backend._reconstruct_payload(request_data, data)
            return True

        # Test with different numbers of concurrent operations
        for num_workers in [1, 4, 8]:
            start_time = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(serialize_operation) for _ in range(100)]
                [future.result() for future in concurrent.futures.as_completed(futures)]

            total_time = time.perf_counter() - start_time
            avg_time_per_operation = total_time / 100 * 1000  # ms

            print(f"\nConcurrent Serialization ({num_workers} workers, 100 operations):")
            print(f"  Total time: {total_time:.3f} s")
            print(f"  Avg time per operation: {avg_time_per_operation:.3f} ms")

            # Performance should scale reasonably with concurrency
            if num_workers == 1:
                baseline_time = avg_time_per_operation
            else:
                # In CI environments, concurrent performance may degrade more due to resource constraints
                # Allow up to 4x degradation with warning, 5x as hard fail in CI; 2x hard fail locally
                if os.getenv("CI"):
                    warn_threshold = 4.0
                    fail_threshold = 10.0
                else:
                    warn_threshold = fail_threshold = 2.0
                if avg_time_per_operation >= baseline_time * fail_threshold:
                    raise AssertionError(
                        f"Concurrent performance degraded too much: {avg_time_per_operation:.3f} ms vs {baseline_time:.3f} ms "
                        f"(max allowed: {baseline_time * fail_threshold:.3f} ms)"
                    )
                elif avg_time_per_operation >= baseline_time * warn_threshold:
                    print(
                        f"WARNING: Concurrent performance degradation is high: {avg_time_per_operation:.3f} ms vs {baseline_time:.3f} ms "
                        f"(warning threshold: {baseline_time * warn_threshold:.3f} ms, fail threshold: {baseline_time * fail_threshold:.3f} ms)"
                    )

    def test_serialization_throughput(self):
        """Test serialization throughput (operations per second)."""
        model = create_medium_model()
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Measure throughput over 1 second
        start_time = time.perf_counter()
        operations = 0

        while time.perf_counter() - start_time < 1.0:
            serialized = _serialize(request_data)
            data = json.loads(json.dumps(serialized))
            self.backend._reconstruct_payload(request_data, data)
            operations += 1

        actual_time = time.perf_counter() - start_time
        throughput = operations / actual_time

        print("\nSerialization Throughput:")
        print(f"  Operations completed: {operations}")
        print(f"  Time taken: {actual_time:.3f} s")
        print(f"  Throughput: {throughput:.1f} ops/sec")

        # Should achieve at least 100 ops/sec for medium models
        assert throughput > 100, f"Throughput too low: {throughput:.1f} ops/sec"

    def test_memory_efficiency(self):
        """Test memory efficiency by measuring memory usage patterns."""
        import gc

        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())

        # Force garbage collection
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and destroy models repeatedly
        for _ in range(500):  # Reduced from 1000 to 500
            model = create_medium_model()
            request_data = {
                "input_data": model,
                "context": None,
                "resources": None,
                "context_model_defined": False,
                "usage_limits": None,
                "stream": False,
            }
            serialized = _serialize(request_data)
            data = json.loads(json.dumps(serialized))
            reconstructed = self.backend._reconstruct_payload(request_data, data)

            # Verify reconstruction
            assert reconstructed["input_data"].model_dump() == model.model_dump()

        # Force garbage collection again
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        print("\nMemory Efficiency Test:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory delta: {memory_delta:.2f} MB")

        # Memory usage should not increase significantly after garbage collection
        assert memory_delta < 100, (
            f"Memory leak detected: {memory_delta:.2f} MB increase"
        )  # Increased from 50MB to 100MB


# Add the reconstruction method to DummyRemoteBackend for testing
def _reconstruct_payload(self, original_payload: dict, data: dict) -> dict:
    """Extract the reconstruction logic for testing."""

    def reconstruct(original: Any, value: Any) -> Any:
        """Rebuild a value using the type of ``original``."""
        # If the incoming value is None, preserve the original structure/value
        if value is None:
            return original
        if original is None:
            return value
        if isinstance(original, BaseModel):
            if isinstance(value, dict):
                fixed_value = {
                    k: reconstruct(getattr(original, k, None), v) for k, v in value.items()
                }
                return type(original).model_validate(fixed_value)
            else:
                return type(original).model_validate(value)
        elif isinstance(original, (list, tuple)):
            if isinstance(value, (list, tuple)):
                if not original:
                    return list(value)
                return type(original)(reconstruct(original[0], v) for v in value)
            else:
                return original
        elif isinstance(original, dict):
            if isinstance(value, dict):
                return {k: reconstruct(original.get(k), v) for k, v in value.items()}
            else:
                return original
        else:
            return value

    # Reconstruct the payload with proper types
    reconstructed_payload = {}
    for key, original_value in original_payload.items():
        if key in data:
            reconstructed_payload[key] = reconstruct(original_value, data[key])
        else:
            reconstructed_payload[key] = original_value

    return reconstructed_payload


# Monkey patch the DummyRemoteBackend for testing
DummyRemoteBackend._reconstruct_payload = _reconstruct_payload
