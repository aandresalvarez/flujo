"""CI Environment Compatibility Tests.

This module contains tests specifically designed to catch issues that might
occur in CI environments but not locally, such as:

1. Memory constraints
2. Different Python versions
3. Different operating systems
4. Resource limitations
5. Timing issues
"""

import pytest
import sys
import platform
import gc
import time
import json

from flujo.type_definitions.common import JSONObject
from flujo.state.backends.base import _serialize_for_json
from flujo.utils.serialization import _robust_serialize_internal


class TestCISerializationCompatibility:
    """Test serialization compatibility in CI environments."""

    def test_serialization_consistency_across_environments(self):
        """Test that serialization produces consistent results across environments."""
        test_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "unicode": "🎉🚀💻",
            "special_chars": "line1\nline2\r\nline3",
        }

        serialized = json.loads(
            json.dumps(test_data, default=_serialize_for_json, ensure_ascii=False)
        )
        assert isinstance(serialized, dict)
        assert serialized["string"] == "test"
        assert serialized["number"] == 42
        assert serialized["boolean"] is True
        assert serialized["none"] is None

        # Test robust_serialize (legacy compatibility)
        robust_result = _robust_serialize_internal(test_data)
        assert isinstance(robust_result, dict)
        assert robust_result["string"] == "test"
        assert robust_result["number"] == 42
        assert robust_result["boolean"] is True
        assert robust_result["none"] is None

    def test_memory_efficient_serialization(self):
        """Test that serialization doesn't consume excessive memory in CI."""
        # Create a moderately large data structure
        large_data = {"items": [{"id": i, "data": f"item_{i}" * 10} for i in range(1000)]}

        # Force garbage collection before test
        gc.collect()

        # Perform serialization
        start_time = time.perf_counter()
        result = _robust_serialize_internal(large_data)
        serialization_time = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert serialization_time < 5.0, f"Serialization took {serialization_time:.3f}s"
        assert isinstance(result, dict)
        assert len(result["items"]) == 1000

        # Force garbage collection after test
        gc.collect()

    def test_error_handling_robustness(self):
        """Test that serialization handles errors gracefully in CI."""

        class ProblematicObject:
            def __init__(self):
                self._problematic_attr = None

            def __getattr__(self, name):
                if name == "_problematic_attr":
                    raise RuntimeError("Simulated CI environment error")
                return super().__getattr__(name)

        obj = ProblematicObject()

        # Should not raise exception
        result = _robust_serialize_internal(obj)
        assert isinstance(result, str)
        assert "ProblematicObject" in result

    def test_circular_reference_in_ci(self):
        """Test circular reference handling in CI environment."""
        # Create circular reference
        data = {"self": None}
        data["self"] = data

        # Should handle without stack overflow
        result = _robust_serialize_internal(data, circular_ref_placeholder=None)
        assert isinstance(result, dict)
        assert result["self"] is None  # Circular ref should be None


class TestCIResourceConstraints:
    """Test behavior under CI resource constraints."""

    def test_low_memory_serialization(self):
        """Test serialization under memory pressure."""
        # Create data that might cause memory issues
        nested_data = {}
        current = nested_data
        for i in range(100):  # Reasonable depth for CI
            current["nested"] = {"level": i, "data": f"level_{i}"}
            current = current["nested"]

        # Should handle without memory issues
        result = _robust_serialize_internal(nested_data)
        assert isinstance(result, dict)
        assert "nested" in result

    def test_concurrent_access_safety(self):
        """Test that serialization is safe under concurrent access."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id: int):
            try:
                data = {
                    "worker_id": worker_id,
                    "data": f"worker_{worker_id}_data",
                    "nested": {"value": worker_id * 10},
                }
                result = _robust_serialize_internal(data)
                results.put((worker_id, result))
            except Exception as e:
                errors.put((worker_id, e))

        # Start multiple threads
        threads = []
        for i in range(3):  # Reasonable number for CI
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert errors.empty(), (
            f"Errors in concurrent test: {[errors.get() for _ in range(errors.qsize())]}"
        )
        assert results.qsize() == 3, f"Expected 3 results, got {results.qsize()}"


class TestCIPlatformCompatibility:
    """Test compatibility across different platforms in CI."""

    def test_platform_independent_serialization(self):
        """Test that serialization works consistently across platforms."""
        # Test data that might behave differently on different platforms
        test_cases = [
            {"empty": ""},
            {"whitespace": " \t\n\r"},
            {"unicode": "🎉🚀💻"},
            {"bytes": b"bytes_data"},
            {"complex": 3 + 4j},
            {"inf": float("inf")},
            {"nan": float("nan")},
        ]

        for test_case in test_cases:
            result = _robust_serialize_internal(test_case)
            assert isinstance(result, dict)
            # Basic validation that serialization completed
            assert len(result) > 0

    def test_python_version_compatibility(self):
        """Test compatibility with different Python versions."""
        # Test features that might behave differently across Python versions
        python_version = sys.version_info

        test_data = {
            "version": f"{python_version.major}.{python_version.minor}",
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        }

        result = _robust_serialize_internal(test_data)
        assert isinstance(result, dict)
        assert result["version"] == test_data["version"]
        assert result["platform"] == test_data["platform"]


class TestCITimingIssues:
    """Test for timing-related issues that might occur in CI."""

    def test_slow_serialization_handling(self):
        """Test that slow serialization doesn't cause timeouts."""
        # Create data that might be slow to serialize
        complex_data = {
            "deep_nested": self._create_deep_nested_structure(50),  # Reasonable depth
            "large_list": [{"item": i, "data": f"item_{i}" * 5} for i in range(500)],
        }

        start_time = time.perf_counter()
        result = _robust_serialize_internal(complex_data)
        serialization_time = time.perf_counter() - start_time

        # Should complete within reasonable time
        assert serialization_time < 10.0, f"Serialization took {serialization_time:.3f}s"
        assert isinstance(result, dict)

    def _create_deep_nested_structure(self, depth: int) -> JSONObject:
        """Create a deeply nested structure for testing."""
        if depth <= 0:
            return {"leaf": "value"}

        return {"level": depth, "nested": self._create_deep_nested_structure(depth - 1)}

    def test_timeout_robustness(self):
        """Test that serialization doesn't hang or timeout."""
        # Create data that might cause timing issues
        data = {
            "recursive": None,
            "large_string": "x" * 10000,  # Large but not excessive
            "mixed_types": [1, "string", True, None, 3.14, {"key": "value"}],
        }

        # Set a timeout for the operation
        start_time = time.perf_counter()
        result = _robust_serialize_internal(data)
        execution_time = time.perf_counter() - start_time

        # Should complete quickly
        assert execution_time < 1.0, f"Serialization took {execution_time:.3f}s"
        assert isinstance(result, dict)


class TestCIErrorRecovery:
    """Test error recovery in CI environments."""

    def test_graceful_degradation(self):
        """Test that serialization degrades gracefully on errors."""

        class FailingObject:
            def __init__(self, fail_on_attr: str):
                self.fail_on_attr = fail_on_attr

            def __getattr__(self, name):
                if name == self.fail_on_attr:
                    raise RuntimeError(f"Simulated failure on {name}")
                return super().__getattr__(name)

        # Test object that fails on attribute access
        obj = FailingObject("problematic_attr")

        # Should handle gracefully
        result = _robust_serialize_internal(obj)
        assert isinstance(result, str)
        assert "FailingObject" in result

    def test_partial_failure_handling(self):
        """Test handling of partial failures in complex objects."""
        data = {
            "normal": "value",
            "problematic": FailingObject("any_attr"),
            "another_normal": {"key": "value"},
        }

        # Enhanced: Robust serialize may return string for entire object in enhanced system
        result = _robust_serialize_internal(data)
        if isinstance(result, dict):
            # Legacy behavior: Partial serialization succeeded
            assert result["normal"] == "value"
            assert result["another_normal"]["key"] == "value"
            assert isinstance(result["problematic"], str)
        else:
            # Enhanced behavior: Entire object converted to string representation
            assert isinstance(result, str)
            assert "<unserializable:" in result or "FailingObject" in result


class FailingObject:
    """Helper class for testing error scenarios."""

    def __init__(self, fail_on_attr: str):
        self.fail_on_attr = fail_on_attr

    def __getattr__(self, name):
        if name == self.fail_on_attr:
            raise RuntimeError(f"Simulated failure on {name}")
        return super().__getattr__(name)


if __name__ == "__main__":
    pytest.main([__file__])
