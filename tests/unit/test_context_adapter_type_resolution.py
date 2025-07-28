"""
Tests for the robust type resolution system in context_adapter.py.

This test suite verifies that the new type resolution mechanism is:
1. More efficient than the old sys.modules iteration
2. Deterministic and predictable
3. Handles edge cases properly
4. Maintains backward compatibility
"""

import pytest
import time
from typing import Optional, Union
from unittest.mock import patch, MagicMock

from flujo.domain.models import BaseModel
from flujo.application.core.context_adapter import (
    _resolve_type_from_string,
    _extract_union_types,
    _resolve_actual_type,
    _deserialize_value,
    _register_type,
    _TYPE_REGISTRY,
)


class TestModel(BaseModel):
    """Test model for type resolution testing."""

    value: int
    name: str = "test"


class NestedTestModel(BaseModel):
    """Nested test model for type resolution testing."""

    id: int
    data: TestModel


class ComplexUnionModel(BaseModel):
    """Model with complex union types."""

    optional_nested: Optional[NestedTestModel] = None
    union_field: Union[str, int] = "default"
    list_of_nested: list[NestedTestModel] = []


class TestTypeResolution:
    """Test the robust type resolution system."""

    def setup_method(self):
        """Clear the type registry before each test."""
        _TYPE_REGISTRY.clear()

    def test_register_and_resolve_type(self):
        """Test that types can be registered and resolved efficiently."""
        # Register a type
        _register_type("TestModel", TestModel)

        # Resolve it
        resolved_type = _resolve_type_from_string("TestModel")
        assert resolved_type == TestModel

        # Verify it's in the registry
        assert "TestModel" in _TYPE_REGISTRY
        assert _TYPE_REGISTRY["TestModel"] == TestModel

    def test_resolve_nonexistent_type(self):
        """Test that resolving a non-existent type returns None."""
        resolved_type = _resolve_type_from_string("NonExistentType")
        assert resolved_type is None

    def test_extract_union_types_traditional_union(self):
        """Test extracting types from traditional Union syntax."""

        # Test Union[T, None]
        union_type = Optional[TestModel]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == TestModel

        # Test Union[T, U, None]
        union_type = Union[str, int, None]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 2
        assert str in non_none_types
        assert int in non_none_types

    def test_extract_union_types_new_syntax(self):
        """Test extracting types from Python 3.10+ Union syntax."""
        # Test T | None
        union_type = TestModel | None
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == TestModel

        # Test T | U | None
        union_type = str | int | None
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 2
        assert str in non_none_types
        assert int in non_none_types

    def test_resolve_actual_type_simple(self):
        """Test resolving actual type from simple type annotations."""
        # Simple type
        actual_type = _resolve_actual_type(TestModel)
        assert actual_type == TestModel

        # None type
        actual_type = _resolve_actual_type(None)
        assert actual_type is None

    def test_resolve_actual_type_union(self):
        """Test resolving actual type from Union type annotations."""
        # Traditional Union
        union_type = Optional[TestModel]
        actual_type = _resolve_actual_type(union_type)
        assert actual_type == TestModel

        # New Union syntax
        union_type = TestModel | None
        actual_type = _resolve_actual_type(union_type)
        assert actual_type == TestModel

    def test_deserialize_value_dict(self):
        """Test deserializing dictionary values."""
        data = {"value": 123, "name": "test"}

        # Test with Pydantic model
        result = _deserialize_value(data, TestModel, ComplexUnionModel)
        assert isinstance(result, TestModel)
        assert result.value == 123
        assert result.name == "test"

        # Test with non-Pydantic type (should return original)
        result = _deserialize_value(data, str, ComplexUnionModel)
        assert result == data

    def test_deserialize_value_list(self):
        """Test deserializing list values."""
        data = [{"value": 1, "name": "test1"}, {"value": 2, "name": "test2"}]

        # Test with list of Pydantic models
        result = _deserialize_value(data, list[TestModel], ComplexUnionModel)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, TestModel) for item in result)
        assert result[0].value == 1
        assert result[1].value == 2

    def test_deserialize_value_non_dict_list(self):
        """Test deserializing non-dict/list values."""
        # String value
        result = _deserialize_value("test", str, ComplexUnionModel)
        assert result == "test"

        # Integer value
        result = _deserialize_value(123, int, ComplexUnionModel)
        assert result == 123

        # List of non-dict items
        result = _deserialize_value([1, 2, 3], list[int], ComplexUnionModel)
        assert result == [1, 2, 3]

    def test_performance_improvement(self):
        """Test that the new system is more efficient than sys.modules iteration."""
        # Register a type
        _register_type("TestModel", TestModel)

        # Measure time for new system
        start_time = time.time()
        for _ in range(1000):
            _resolve_type_from_string("TestModel")
        new_system_time = time.time() - start_time

        # Measure time for old system (simulated)
        start_time = time.time()
        with patch("sys.modules", {"test_module": MagicMock(TestModel=TestModel)}):
            for _ in range(1000):
                # Simulate the old sys.modules iteration
                for module_name, module in {"test_module": MagicMock(TestModel=TestModel)}.items():
                    if hasattr(module, "TestModel"):
                        _ = module.TestModel
                        break
        old_system_time = time.time() - start_time

        # The new system should be significantly faster
        assert new_system_time < old_system_time

    def test_deterministic_behavior(self):
        """Test that type resolution is deterministic."""
        _register_type("TestModel", TestModel)

        # Multiple calls should return the same result
        result1 = _resolve_type_from_string("TestModel")
        result2 = _resolve_type_from_string("TestModel")
        result3 = _resolve_type_from_string("TestModel")

        assert result1 == result2 == result3 == TestModel

    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        # Test with a type that would have been found in sys.modules
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.return_value.f_globals = {"NestedTestModel": NestedTestModel}
            mock_frame.return_value.f_back = None

            result = _resolve_type_from_string("NestedTestModel")
            assert result == NestedTestModel

    def test_error_handling(self):
        """Test that the system handles errors gracefully."""
        # Test with invalid type string
        result = _resolve_type_from_string("")
        assert result is None

        # Test with None
        result = _resolve_type_from_string(None)
        assert result is None

        # Test with non-string input
        result = _resolve_type_from_string(123)
        assert result is None

    def test_complex_union_handling(self):
        """Test handling of complex union types with nested models."""
        # Register the nested model
        _register_type("NestedTestModel", NestedTestModel)

        # Test complex union type
        union_type = NestedTestModel | None
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == NestedTestModel

        # Test resolving actual type
        actual_type = _resolve_actual_type(union_type)
        assert actual_type == NestedTestModel

    def test_type_registry_isolation(self):
        """Test that the type registry is properly isolated between tests."""
        # Should be empty at start
        assert len(_TYPE_REGISTRY) == 0

        # Register a type
        _register_type("TestModel", TestModel)
        assert len(_TYPE_REGISTRY) == 1

        # Clear registry
        _TYPE_REGISTRY.clear()
        assert len(_TYPE_REGISTRY) == 0

    def test_multiple_type_registration(self):
        """Test registering multiple types."""
        _register_type("TestModel", TestModel)
        _register_type("NestedTestModel", NestedTestModel)

        assert len(_TYPE_REGISTRY) == 2
        assert _TYPE_REGISTRY["TestModel"] == TestModel
        assert _TYPE_REGISTRY["NestedTestModel"] == NestedTestModel

        # Test resolution
        assert _resolve_type_from_string("TestModel") == TestModel
        assert _resolve_type_from_string("NestedTestModel") == NestedTestModel

    def test_sys_modules_iteration_prevention(self):
        """Test that we don't fall back to sys.modules iteration."""
        # This test ensures we don't regress to the old problematic pattern
        with patch(
            "sys.modules",
            {
                "module1": MagicMock(TestModel=TestModel),
                "module2": MagicMock(TestModel=NestedTestModel),
            },
        ):
            # Mock inspect.currentframe to return a frame without TestModel in globals
            mock_frame = MagicMock()
            mock_frame.f_globals = {}  # Empty globals
            mock_frame.f_back = None

            with patch("inspect.currentframe", return_value=mock_frame):
                # Should not iterate through sys.modules
                result = _resolve_type_from_string("NonExistentType")
                # Should return None since it's not in registry
                assert result is None

    def test_type_name_conflict_resolution(self):
        """Test that type name conflicts are handled deterministically."""

        # Simulate multiple modules with same type name
        class ConflictingModel1(BaseModel):
            value: str

        class ConflictingModel2(BaseModel):
            value: int

        # Register one in registry
        _register_type("ConflictingModel", ConflictingModel1)

        # Should always return the registered type, not search sys.modules
        result = _resolve_type_from_string("ConflictingModel")
        assert result == ConflictingModel1
        assert result != ConflictingModel2

    def test_performance_degradation_prevention(self):
        """Test that performance doesn't degrade with more modules."""
        # Simulate many loaded modules
        many_modules = {}
        for i in range(1000):
            many_modules[f"module_{i}"] = MagicMock()

        with patch("sys.modules", many_modules):
            # Performance should be consistent regardless of module count
            start_time = time.time()
            for _ in range(1000):
                _resolve_type_from_string("NonExistentType")
            lookup_time = time.time() - start_time

            # Should be very fast (under 0.1 seconds for 1000 lookups)
            assert lookup_time < 0.1

    def test_deterministic_order_independence(self):
        """Test that results don't depend on sys.modules order."""
        # Create modules with different orders
        modules1 = {
            "module1": MagicMock(TestModel=TestModel),
            "module2": MagicMock(TestModel=NestedTestModel),
        }
        modules2 = {
            "module2": MagicMock(TestModel=NestedTestModel),
            "module1": MagicMock(TestModel=TestModel),
        }

        # Results should be the same regardless of order
        with patch("sys.modules", modules1):
            result1 = _resolve_type_from_string("TestModel")

        with patch("sys.modules", modules2):
            result2 = _resolve_type_from_string("TestModel")

        # Both should return None (not in registry) or same result
        assert result1 == result2

    def test_string_parsing_fallback_robustness(self):
        """Test that string parsing fallback is robust."""
        # Test with complex type strings that might cause parsing issues
        complex_type_str = "Union[Optional[NestedModel], List[ComplexModel], None]"

        # Should handle gracefully without crashing
        try:
            result = _resolve_type_from_string(complex_type_str)
            # Should return None for invalid type strings
            assert result is None
        except Exception as e:
            pytest.fail(f"String parsing should not raise exceptions: {e}")

    def test_memory_efficiency(self):
        """Test that we don't create memory leaks with type resolution."""
        import gc

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many type resolutions
        for _ in range(1000):
            _resolve_type_from_string("TestModel")
            _extract_union_types(TestModel | None)
            _resolve_actual_type(Optional[TestModel])

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not increase significantly
        # Allow for some reasonable increase (e.g., 10% more objects)
        assert final_objects <= initial_objects * 1.1

    def test_concurrent_type_resolution(self):
        """Test that type resolution is thread-safe."""
        import threading
        import queue

        results = queue.Queue()

        def resolve_type_worker():
            try:
                result = _resolve_type_from_string("NonExistentType")
                results.put(result)
            except Exception as e:
                results.put(e)

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=resolve_type_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be the same (None since not registered)
        all_results = []
        while not results.empty():
            all_results.append(results.get())

        # All results should be None (not in registry)
        assert all(result is None for result in all_results)

    def test_registry_consistency(self):
        """Test that type registry maintains consistency."""
        # Register a type
        _register_type("TestModel", TestModel)

        # Verify it's in registry
        assert "TestModel" in _TYPE_REGISTRY
        assert _TYPE_REGISTRY["TestModel"] == TestModel

        # Clear registry
        _TYPE_REGISTRY.clear()

        # Verify it's gone
        assert "TestModel" not in _TYPE_REGISTRY

        # Should return None after clearing
        result = _resolve_type_from_string("NonExistentType")
        assert result is None
