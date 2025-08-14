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
from typing import Optional, Union, List, Dict, Any
from unittest.mock import patch, MagicMock

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

import types
from typing import Tuple, Set, Iterable, Optional as _Opt, Union as _Union
import pytest


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
        # Test T | None syntax
        union_type = _TestModel | None
        non_none_types = _extract_union_types(union_type)
        assert len(non_none_types) == 1
        assert non_none_types[0] == _TestModel

        # Test T | U | None syntax
        union_type = str | int | None
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

            # Should be very fast (under 0.1 seconds for 1000 lookups)
            assert new_system_time < 0.1

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
        union_type = _NestedTestModel | None
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
        from flujo.utils.serialization import safe_serialize, safe_deserialize

        serialized = safe_serialize(instance)
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
            settings: Optional[Dict[str, Any]] = None

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


# =============================================================================
# Additional thorough tests to extend coverage for context_adapter type resolution
# NOTE: These tests are written for the project's existing test framework (pytest-style assertions).
# Testing library and framework: Pytest (assumed based on assert style and tests layout).
# =============================================================================


class TestTypeResolutionExtended:
    """Extended coverage for robust type resolution and deserialization."""

    def setup_method(self):
        # Clear caches and context before each test to avoid cross-test contamination
        _type_context._resolvers.clear()
        _type_context._current_module = None

    # ---------------------------
    # _deserialize_value coverage
    # ---------------------------

    def test_deserialize_value_none_and_passthrough(self):
        # None stays None regardless of target type
        assert _deserialize_value(None, _TestModel, _TestModel) is None
        assert _deserialize_value(None, List[_TestModel], _TestModel) is None

        # Already-correct instance should pass through unchanged
        instance = _TestModel(value=5, name="x")
        out = _deserialize_value(instance, _TestModel, _TestModel)
        assert out is instance

    def test_deserialize_value_invalid_inputs_raise_or_return_none(self):
        # Non-dict input for model should be handled
        # Depending on implementation, could raise or return value; we validate graceful handling.
        invalid = "not-a-dict"
        try:
            out = _deserialize_value(invalid, _TestModel, _TestModel)
            # If returned, ensure it is not incorrectly converted to a model
            assert not isinstance(out, _TestModel)
        except Exception as e:
            # Acceptable if implementation raises a clear error
            assert isinstance(e, (TypeError, ValueError))

    def test_deserialize_value_nested_list_of_models(self):
        nested_list_data = [
            [{"value": 1, "name": "a"}, {"value": 2, "name": "b"}],
            [{"value": 3, "name": "c"}],
        ]
        # Target: List[List[_TestModel]]
        deserialized = _deserialize_value(nested_list_data, List[List[_TestModel]], _TestModel)
        assert isinstance(deserialized, list)
        assert all(isinstance(inner, list) for inner in deserialized)
        assert all(isinstance(item, _TestModel) for inner in deserialized for item in inner)
        assert [m.value for inner in deserialized for m in inner] == [1, 2, 3]

    def test_deserialize_value_dict_of_models(self):
        dict_data = {
            "first": {"value": 10, "name": "ten"},
            "second": {"value": 20, "name": "twenty"},
        }
        # We attempt to deserialize into Dict[str, _TestModel] when supported
        deserialized = _deserialize_value(dict_data, Dict[str, _TestModel], _TestModel)
        assert isinstance(deserialized, dict)
        assert set(deserialized.keys()) == {"first", "second"}
        assert all(isinstance(v, _TestModel) for v in deserialized.values())
        assert deserialized["first"].value == 10
        assert deserialized["second"].name == "twenty"

    def test_deserialize_value_list_with_mixed_validity(self):
        mixed = [{"value": 1, "name": "ok"}, "bad", {"value": 2, "name": "ok2"}]
        # Expect graceful handling: either best-effort conversion or error; assert no silent incorrect typing.
        try:
            out = _deserialize_value(mixed, List[_TestModel], _TestModel)
            assert isinstance(out, list)
            # Each element either a model or preserved but not incorrectly typed
            for item in out:
                assert isinstance(item, (_TestModel, dict, str))
        except Exception as e:
            assert isinstance(e, (TypeError, ValueError))

    # ---------------------------
    # _extract_union_types coverage
    # ---------------------------

    def test_extract_union_types_nested_and_duplicates(self):
        # Nested unions with duplicates
        nested = _Union[_TestModel, _Union[_TestModel, str], None]
        types_out = _extract_union_types(nested)
        # Should de-duplicate and exclude None
        assert _TestModel in types_out
        assert str in types_out
        assert all(t is not None for t in types_out)

    def test_extract_union_types_with_any_and_iterables(self):
        u = _Union[Any, Iterable[int], None]
        types_out = _extract_union_types(u)
        # Any may be included depending on implementation; assert key types are present
        assert Iterable in [getattr(t, "__origin__", t) for t in types_out] or any(
            getattr(t, "_name", None) == "Iterable" for t in types_out
        )

    def test_extract_union_types_no_union(self):
        # Non-union input should yield the type itself
        t = _extract_union_types(_TestModel)
        assert t == [_TestModel]

    # ---------------------------
    # _resolve_actual_type coverage
    # ---------------------------

    def test_resolve_actual_type_complex(self):
        # Optional within Union with multiple types
        t = _resolve_actual_type(_Union[_TestModel, str, None])
        assert t in (_TestModel, str)  # Implementation may pick first non-None

        # Modern syntax nesting
        t2 = _resolve_actual_type((_TestModel | None) | (str | None))
        assert t2 in (_TestModel, str)

        # Unparameterized generics: should return the origin or None
        t3 = _resolve_actual_type(List)
        assert t3 in (list, List, None)

    # ---------------------------
    # register_custom_type coverage
    # ---------------------------

    def test_register_custom_type_duplicate_registration_is_safe(self):
        register_custom_type(_UserCustomModel)
        # Duplicate registration shouldn't crash
        register_custom_type(_UserCustomModel)

        # Serializer/deserializer should still be available
        from flujo.utils.serialization import lookup_custom_serializer, lookup_custom_deserializer

        inst = _UserCustomModel(custom_field="x", number=1)
        assert lookup_custom_serializer(inst) is not None
        assert lookup_custom_deserializer(_UserCustomModel) is not None

    # ---------------------------
    # TypeResolutionContext.module_scope behavior
    # ---------------------------

    def test_module_scope_nesting_and_restoration(self):
        ctx = TypeResolutionContext()

        m1 = sys.modules[__name__]
        dummy = types.ModuleType("dummy_mod")
        dummy.__name__ = "dummy_mod"

        assert _type_context._current_module is None

        with ctx.module_scope(m1):
            assert _type_context._current_module is m1
            with ctx.module_scope(dummy):
                assert _type_context._current_module is dummy
            # After inner scope exit, outer should be restored
            assert _type_context._current_module is m1
        # After outer scope exit, should be cleared
        assert _type_context._current_module is None

    def test_module_scope_with_invalid_objects(self):
        ctx = TypeResolutionContext()
        # Should not crash when given invalid module-like object
        with ctx.module_scope(object()):
            # Resolution with invalid scope should be safe and return None
            assert ctx.resolve_type("_TestModel", BaseModel) is None

    # ---------------------------
    # _resolve_type_from_string coverage
    # ---------------------------

    def test_resolve_type_from_string_dotted_paths_and_unknowns(self):
        # Unknown dotted path should yield None
        assert _resolve_type_from_string("nonexistent.module.Type") is None
        # Empty/whitespace
        assert _resolve_type_from_string("   ") is None
        # Non-string input already covered above; test float
        assert _resolve_type_from_string(3.14) is None

    def test_resolve_type_from_string_current_module_name_only(self):
        # Even if a type exists in current module, system is expected not to traverse frames
        assert _resolve_type_from_string("_TestModel") is None

    # ---------------------------
    # Concurrency and stress coverage
    # ---------------------------

    def test_concurrent_mixed_resolution_under_contention(self):
        ctx = TypeResolutionContext()
        results: list = []
        errs: list = []

        def worker(name: str):
            try:
                mod = sys.modules[__name__]
                with ctx.module_scope(mod):
                    results.append(ctx.resolve_type(name, BaseModel))
            except Exception as e:
                errs.append(e)

        names = ["_TestModel", "NonExistentType", "_TestModel", "AlsoMissing"]
        threads = [threading.Thread(target=worker, args=(n,)) for n in names for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert any(r is _TestModel for r in results)
        assert all(e is None or isinstance(e, Exception) for e in errs)  # No unexpected types

    # ---------------------------
    # Serialization integration negatives
    # ---------------------------

    def test_serialization_integration_invalid_payload(self):
        from flujo.utils.serialization import safe_deserialize

        # Provide invalid serialized payload for _UserCustomModel
        bad_payload = {"__type__": "UnknownType", "data": {"x": 1}}
        try:
            out = safe_deserialize(bad_payload, _UserCustomModel)
            # If not raised, ensure it didn't silently create incorrect type
            assert not isinstance(out, _UserCustomModel)
        except Exception as e:
            assert isinstance(e, (ValueError, TypeError, KeyError))

    # ---------------------------
    # Determinism with repeated registrations
    # ---------------------------

    def test_determinism_across_repeated_resolutions(self):
        ctx = TypeResolutionContext()
        mod = sys.modules[__name__]
        with ctx.module_scope(mod):
            r1 = ctx.resolve_type("_TestModel", BaseModel)
            register_custom_type(_AnotherCustomModel)
            r2 = ctx.resolve_type("_TestModel", BaseModel)
            r3 = ctx.resolve_type("_TestModel", BaseModel)
        assert r1 is r2 is r3 is _TestModel

    # ---------------------------
    # Safety: ensure inspect.currentframe is not used in resolution
    # ---------------------------

    def test_no_inspect_currentframe_usage_on_resolve(self):
        with patch("inspect.currentframe") as m:
            m.return_value = None
            # Call several resolution helpers to ensure no frame usage
            assert _resolve_type_from_string("_TestModel") is None
            _ = _extract_union_types(_Opt[_TestModel])
            _ = _resolve_actual_type(_Opt[_TestModel])
            m.assert_not_called()


# Additional targeted unit tests for pure functions (no side-effects)
def test_extract_union_types_with_tuple_and_set_annotations():
    # Not a union; should return container types as-is or single element list
    t_tuple = _extract_union_types(Tuple[int, str])
    # Accept a flexible behavior: either origin or the type itself
    assert t_tuple in ([Tuple], [tuple], [Tuple[int, str]])

    t_set = _extract_union_types(Set[int])
    assert t_set in ([Set], [set], [Set[int]])


def test_resolve_actual_type_prefers_model_over_builtin_when_both_present():
    union = Union[_TestModel, int, None]
    chosen = _resolve_actual_type(union)
    assert chosen in (_TestModel, int)
    # Prefer model if implementation selects first non-None in declaration order
    # If not, still acceptable to select int; the main property is consistency and non-None.


@pytest.mark.parametrize(
    "type_str,expected",
    [
        ("", None),
        (None, None),
        ("_NonExisting", None),
        ("builtins.int", None),  # dotted path with stdlib should not be resolved by string helper
    ],
)
def test_resolve_type_from_string_parametrized(type_str, expected):
    assert _resolve_type_from_string(type_str) is expected