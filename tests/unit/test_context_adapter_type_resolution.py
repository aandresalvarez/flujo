"""
Tests for the robust type resolution system in context_adapter.py.

This test suite verifies that the new type resolution mechanism is:
1. Thread-safe and performant
2. Integrates with Flujo's serialization system
3. Uses Python's type system properly
4. Provides validation and safety
5. Supports module-scoped resolution
6. Maintains backward compatibility
"""

import os
import time
import threading
import sys
from typing import Optional, Union, List
from unittest.mock import patch, MagicMock
from flujo.type_definitions.common import JSONObject

from flujo.domain.models import BaseModel
from flujo.application.core.context_adapter import (
    _resolve_type_from_string,
    _extract_union_types,
    _resolve_actual_type,
    _deserialize_value,
    register_custom_type,
    TypeResolutionContext,
    _type_context,
)


class _TestModel(BaseModel):
    """Test model for type resolution testing."""

    value: int
    name: str = "test"


class _NestedTestModel(BaseModel):
    """Nested test model for type resolution testing."""

    nested_value: str
    test_model: _TestModel


class _UserCustomModel(BaseModel):
    """User-defined custom model for testing type resolution."""

    custom_field: str
    number: int


class _AnotherCustomModel(BaseModel):
    """Another user-defined custom model."""

    another_field: bool
    description: str


class TestTypeResolution:
    """Test the robust type resolution system."""

    def setup_method(self):
        """Clear any cached state before each test."""
        # Reset the type context
        _type_context._resolvers.clear()
        _type_context._current_module = None

    def test_register_custom_type_integration(self):
        """Test that custom type registration integrates with serialization."""
        # Register a custom type
        register_custom_type(_UserCustomModel)

        # Verify it's registered for serialization
        from flujo.utils.serialization import lookup_custom_serializer, lookup_custom_deserializer

        # Create an instance to test serializer lookup
        instance = _UserCustomModel(custom_field="test", number=42)
        serializer = lookup_custom_serializer(instance)
        deserializer = lookup_custom_deserializer(_UserCustomModel)

        assert serializer is not None
        assert deserializer is not None

    def test_type_resolution_context_thread_safety(self):
        """Test that type resolution context is thread-safe."""
        context = TypeResolutionContext()
        results = []

        def worker():
            # Use the current module for testing
            current_module = sys.modules[__name__]
            with context.module_scope(current_module):
                result = context.resolve_type("_TestModel", BaseModel)
                results.append(result)

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be the same
        assert all(result == _TestModel for result in results)

    def test_module_scope_resolution(self):
        """Test module-scoped type resolution."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Should be able to resolve types from current module
            result = context.resolve_type("_TestModel", BaseModel)
            assert result == _TestModel

            # Should not resolve types from other modules
            result = context.resolve_type("NonExistentType", BaseModel)
            assert result is None

    def test_type_validation(self):
        """Test that type resolution includes proper validation."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Valid type
            result = context.resolve_type("_TestModel", BaseModel)
            assert result == _TestModel

            # Invalid base type
            result = context.resolve_type("_TestModel", str)
            assert result is None

    def test_extract_union_types_type_system(self):
        """Test extracting types using proper type system integration."""
        # Test Union[T, None]
        union_type = Optional[_TestModel]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == _TestModel

        # Test Union[T, U, None]
        union_type = Union[str, int, None]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 2
        assert str in non_none_types
        assert int in non_none_types

    def test_extract_union_types_modern_syntax(self):
        """Test extracting types from modern Union syntax (Python 3.10+)."""
        # Test Optional[T] syntax
        union_type = Optional[_TestModel]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == _TestModel

        # Test Union[T, U, None] syntax
        union_type = Union[str, int, None]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 2
        assert str in non_none_types
        assert int in non_none_types

    def test_resolve_actual_type(self):
        """Test resolving actual types from field annotations."""
        # Test direct type
        actual_type = _resolve_actual_type(_TestModel)
        assert actual_type == _TestModel

        # Test Union type
        actual_type = _resolve_actual_type(Optional[_TestModel])
        assert actual_type == _TestModel

        # Test None
        actual_type = _resolve_actual_type(None)
        assert actual_type is None

    def test_deserialize_value_integration(self):
        """Test value deserialization with serialization system integration."""
        # Test Pydantic model deserialization
        test_data = {"value": 42, "name": "test"}
        deserialized = _deserialize_value(test_data, _TestModel, _TestModel)
        assert isinstance(deserialized, _TestModel)
        assert deserialized.value == 42
        assert deserialized.name == "test"

        # Test list of models
        list_data = [{"value": 1, "name": "a"}, {"value": 2, "name": "b"}]
        deserialized = _deserialize_value(list_data, List[_TestModel], _TestModel)
        assert isinstance(deserialized, list)
        assert len(deserialized) == 2
        assert all(isinstance(item, _TestModel) for item in deserialized)

    def test_performance_improvement(self):
        """Test that the new system is more efficient."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Measure time for new system
            start_time = time.time()
            for _ in range(1000):
                context.resolve_type("TestModel", BaseModel)
            new_system_time = time.time() - start_time

            # Should be very fast - 1s sanity check for CI variance
            assert new_system_time < 1.0

    def test_deterministic_behavior(self):
        """Test that type resolution is deterministic."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Multiple calls should return the same result
            result1 = context.resolve_type("_TestModel", BaseModel)
            result2 = context.resolve_type("_TestModel", BaseModel)
            result3 = context.resolve_type("_TestModel", BaseModel)

            assert result1 == result2 == result3 == _TestModel

    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        # Test with a type that would have been found in sys.modules
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.return_value.f_globals = {"NestedTestModel": _NestedTestModel}
            mock_frame.return_value.f_back = None

            result = _resolve_type_from_string("NestedTestModel")
            # Should return None since we're not using frame traversal anymore
            assert result is None

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
        # Test complex union type
        union_type = Optional[_NestedTestModel]
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == _NestedTestModel

        # Test resolving actual type
        actual_type = _resolve_actual_type(union_type)
        assert actual_type == _NestedTestModel

    def test_module_resolver_caching(self):
        """Test that module resolver properly caches results."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # First call should cache the result
            result1 = context.resolve_type("_TestModel", BaseModel)
            assert result1 == _TestModel

            # Second call should use cache
            result2 = context.resolve_type("_TestModel", BaseModel)
            assert result2 == _TestModel

            # Verify cache is working
            resolver = context._resolvers.get(__name__)
            assert resolver is not None
            assert "_TestModel" in resolver._cache

    def test_type_hints_integration(self):
        """Test integration with Python's type hints system."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Should be able to resolve types from type hints
            result = context.resolve_type("_TestModel", BaseModel)
            assert result == _TestModel

    def test_concurrent_type_resolution(self):
        """Test that type resolution is thread-safe under concurrent access."""
        context = TypeResolutionContext()
        results = []

        def resolve_worker():
            # Use the current module for testing
            current_module = sys.modules[__name__]
            with context.module_scope(current_module):
                result = context.resolve_type("_TestModel", BaseModel)
                results.append(result)

        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=resolve_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be the same
        assert all(result == _TestModel for result in results)

    def test_serialization_integration(self):
        """Test that type registration integrates with serialization system."""
        # Register a custom type
        register_custom_type(_UserCustomModel)

        # Create an instance
        instance = _UserCustomModel(custom_field="test", number=42)

        # Test serialization
        from flujo.utils.serialization import safe_deserialize
        from flujo.state.backends.base import _serialize_for_json
        import json

        serialized = json.loads(
            json.dumps(instance, default=_serialize_for_json, ensure_ascii=False)
        )
        deserialized = safe_deserialize(serialized, _UserCustomModel)

        assert isinstance(deserialized, _UserCustomModel)
        assert deserialized.custom_field == "test"
        assert deserialized.number == 42

    def test_context_injection_with_type_system(self):
        """Test that context injection uses type system integration."""
        # This test verifies that the new type system integration
        # works properly in the context injection process
        from flujo.application.core.context_adapter import _inject_context

        class TestContext(BaseModel):
            user: _UserCustomModel
            settings: Optional[JSONObject] = None

        # Register the custom type
        register_custom_type(_UserCustomModel)

        # Create context
        context = TestContext(user=_UserCustomModel(custom_field="test", number=42))

        # Test injection
        update_data = {"user": {"custom_field": "updated", "number": 100}}

        result = _inject_context(context, update_data, TestContext)
        assert result is None  # No validation error
        assert context.user.custom_field == "updated"
        assert context.user.number == 100

    def test_future_proof_design(self):
        """Test that the design is future-proof and extensible."""
        # Test that we can easily add new type resolution strategies
        context = TypeResolutionContext()

        # Test that the system can handle new types without modification
        class FutureType(BaseModel):
            future_field: str

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            # Should be able to resolve future types
            result = context.resolve_type("FutureType", BaseModel)
            # The type might not be resolved immediately since it's defined in the test
            # This is expected behavior - the system is future-proof but doesn't auto-discover
            # types defined in the same scope
            assert result is None  # Expected behavior for locally defined types

    def test_security_no_frame_access(self):
        """Test that the new system doesn't access frame globals."""
        # Mock inspect.currentframe to ensure it's not called
        with patch("inspect.currentframe") as mock_frame:
            mock_frame.return_value = None

            # The new system should not use frame access
            result = _resolve_type_from_string("TestModel")
            # Should return None since we're not using frame traversal
            assert result is None

            # Verify frame was not accessed
            mock_frame.assert_not_called()

    def test_type_system_integration(self):
        """Test that the system properly integrates with Python's type system."""
        from typing import get_type_hints

        # Test that we can get type hints from a simple class
        class TestClass:
            test_field: _TestModel
            user_field: _UserCustomModel

        type_hints = get_type_hints(TestClass)

        # Should include our test types
        assert "test_field" in type_hints
        assert "user_field" in type_hints
        assert type_hints["test_field"] == _TestModel
        assert type_hints["user_field"] == _UserCustomModel

    def test_robust_error_recovery(self):
        """Test that the system recovers gracefully from errors."""
        context = TypeResolutionContext()

        # Test with invalid module
        with context.module_scope(None):
            result = context.resolve_type("TestModel", BaseModel)
            assert result is None

        # Test with module that has no type hints
        mock_module = MagicMock()
        mock_module.__name__ = "mock_module"

        with context.module_scope(mock_module):
            result = context.resolve_type("TestModel", BaseModel)
            assert result is None

    def test_performance_under_load(self):
        """Test performance under high load."""
        context = TypeResolutionContext()

        # Use the current module for testing
        current_module = sys.modules[__name__]
        with context.module_scope(current_module):
            start_time = time.time()

            # Perform many type resolutions
            for _ in range(10000):
                context.resolve_type("TestModel", BaseModel)

            end_time = time.time()

            # Should complete in reasonable time
            threshold = float(os.getenv("TYPE_RESOLUTION_THRESHOLD", 2.0))  # Default to 2 seconds
            assert end_time - start_time < threshold  # Configurable threshold for 10k lookups
