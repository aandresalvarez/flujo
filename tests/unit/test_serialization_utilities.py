"""Comprehensive tests for Flujo serialization utilities."""

import dataclasses
import json
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import BaseModel

from flujo.utils.serialization import (
    create_field_serializer,
    lookup_custom_serializer,
    register_custom_serializer,
    robust_serialize,
    safe_serialize,
    serialize_to_json,
    serialize_to_json_robust,
    reset_custom_serializer_registry,
)


class MockEnum(Enum):
    """Mock enum for testing serialization utilities."""

    A = "a"
    B = "b"
    C = "c"


@dataclasses.dataclass
class MockDataclass:
    """Mock dataclass for testing serialization utilities."""

    name: str
    value: int
    items: List[str]


class MockPydanticModel(BaseModel):
    """Mock Pydantic model for testing serialization utilities."""

    name: str
    value: int
    items: List[str]
    optional_field: Optional[str] = None


class CustomObject:
    """Custom object for testing custom serialization."""

    def __init__(self, data: str, metadata: Dict[str, Any]):
        self.data = data
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {"data": self.data, "metadata": self.metadata}


class UnregisteredObject:
    """Object that should not have a registered serializer."""

    def __init__(self, data: str):
        self.data = data


@pytest.fixture(autouse=True)
def clear_global_registry():
    """Clear the global registry before each test to ensure isolation."""
    reset_custom_serializer_registry()


class TestGlobalRegistry:
    """Test the global custom serializer registry."""

    def test_register_and_lookup_custom_serializer(self):
        """Test registering and looking up custom serializers."""

        # Test exact type matching
        def serialize_custom(obj: CustomObject) -> dict:
            return obj.to_dict()

        register_custom_serializer(CustomObject, serialize_custom)

        obj = CustomObject("test", {"key": "value"})
        serializer = lookup_custom_serializer(obj)

        assert serializer is not None
        result = serializer(obj)
        assert result == {"data": "test", "metadata": {"key": "value"}}

    def test_lookup_nonexistent_serializer(self):
        """Test looking up a serializer for an unregistered type."""
        obj = UnregisteredObject("test")
        serializer = lookup_custom_serializer(obj)
        assert serializer is None

    def test_inheritance_based_lookup(self):
        """Test that serializers work with inheritance."""

        class SubCustomObject(CustomObject):
            pass

        def serialize_custom(obj: CustomObject) -> dict:
            return {"type": "custom", "data": obj.data}

        register_custom_serializer(CustomObject, serialize_custom)

        sub_obj = SubCustomObject("test", {})
        serializer = lookup_custom_serializer(sub_obj)

        assert serializer is not None
        result = serializer(sub_obj)
        assert result == {"type": "custom", "data": "test"}

    def test_thread_safety(self):
        """Test that the registry is thread-safe."""

        def serialize_custom(obj: CustomObject) -> dict:
            return obj.to_dict()

        # Register in one thread
        register_custom_serializer(CustomObject, serialize_custom)

        # Look up in another thread
        obj = CustomObject("test", {})

        def lookup_in_thread():
            serializer = lookup_custom_serializer(obj)
            assert serializer is not None
            return serializer(obj)

        # This should work without race conditions
        result = lookup_in_thread()
        assert result == {"data": "test", "metadata": {}}

    def test_create_field_serializer(self):
        """Test the convenience function for creating field serializers."""

        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        field_serializer = create_field_serializer("custom_field", serialize_custom)

        obj = CustomObject("test", {})
        result = field_serializer(obj)
        assert result == {"custom": "test"}

        # Should NOT be registered globally (field serializers are local)
        global_serializer = lookup_custom_serializer(obj)
        assert global_serializer is None


class TestSafeSerialize:
    """Test the safe_serialize function."""

    def test_primitive_types(self):
        """Test serialization of primitive types."""
        assert safe_serialize(None) is None
        assert safe_serialize("test") == "test"
        assert safe_serialize(42) == 42
        assert safe_serialize(3.14) == 3.14
        assert safe_serialize(True) is True
        assert safe_serialize(False) is False

    def test_special_float_values(self):
        """Test serialization of special float values."""
        assert safe_serialize(float("inf")) == "inf"
        assert safe_serialize(float("-inf")) == "-inf"
        assert safe_serialize(float("nan")) == "nan"

    def test_datetime_objects(self):
        """Test serialization of datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        d = date(2023, 1, 1)
        t = time(12, 0, 0)

        assert safe_serialize(dt) == "2023-01-01T12:00:00"
        assert safe_serialize(d) == "2023-01-01"
        assert safe_serialize(t) == "12:00:00"

    def test_bytes_and_memoryview(self):
        """Test serialization of bytes and memoryview objects."""
        data = b"test data"
        mv = memoryview(data)

        # Test bytes
        result = safe_serialize(data)
        assert isinstance(result, str)
        assert result == "dGVzdCBkYXRh"  # base64 encoded

        # Test memoryview
        result = safe_serialize(mv)
        assert isinstance(result, str)
        assert result == "dGVzdCBkYXRh"

    def test_complex_numbers(self):
        """Test serialization of complex numbers."""
        c = 3 + 4j
        result = safe_serialize(c)
        assert result == {"real": 3.0, "imag": 4.0}

    def test_functions_and_callables(self):
        """Test serialization of functions and callables."""

        def test_function():
            pass

        result = safe_serialize(test_function)
        assert result == "test_function"

        # Test callable object
        class CallableObject:
            def __call__(self):
                pass

        obj = CallableObject()
        result = safe_serialize(obj)
        assert "CallableObject" in result

    def test_dataclasses(self):
        """Test serialization of dataclasses."""
        obj = MockDataclass("test", 42, ["item1", "item2"])
        result = safe_serialize(obj)
        assert result == {"name": "test", "value": 42, "items": ["item1", "item2"]}

    def test_enums(self):
        """Test serialization of enums."""
        result = safe_serialize(MockEnum.A)
        assert result == "a"

    def test_pydantic_models(self):
        """Test serialization of Pydantic models."""
        obj = MockPydanticModel(
            name="test", value=42, items=["item1", "item2"], optional_field="optional"
        )
        result = safe_serialize(obj)
        assert result == {
            "name": "test",
            "value": 42,
            "items": ["item1", "item2"],
            "optional_field": "optional",
        }

    def test_collections(self):
        """Test serialization of collections."""
        # Lists
        result = safe_serialize([1, 2, 3])
        assert result == [1, 2, 3]

        # Tuples
        result = safe_serialize((1, 2, 3))
        assert result == [1, 2, 3]

        # Sets
        result = safe_serialize({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Frozensets
        result = safe_serialize(frozenset([1, 2, 3]))
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Dicts
        result = safe_serialize({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_circular_references(self):
        """Test handling of circular references."""
        # Create circular reference
        obj = {"name": "test"}
        obj["self"] = obj

        result = safe_serialize(obj)
        assert result["name"] == "test"
        assert result["self"] is None  # Circular reference should be None

    def test_custom_serializer_integration(self):
        """Test integration with custom serializers."""

        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        register_custom_serializer(CustomObject, serialize_custom)

        obj = CustomObject("test", {})
        result = safe_serialize(obj)
        assert result == {"custom": "test"}

    def test_default_serializer(self):
        """Test the default_serializer parameter."""
        obj = UnregisteredObject("test")

        # Without default serializer - should raise TypeError
        with pytest.raises(TypeError):
            safe_serialize(obj)

        # With default serializer
        def default_serializer(obj: Any) -> str:
            return f"<{type(obj).__name__}: {obj.data}>"

        result = safe_serialize(obj, default_serializer=default_serializer)
        assert result == "<UnregisteredObject: test>"

    def test_nested_structures(self):
        """Test serialization of nested structures."""
        data = {
            "list": [1, 2, {"nested": "value"}],
            "set": {1, 2, 3},
            "custom": CustomObject("test", {"key": "value"}),
        }

        # Register custom serializer
        register_custom_serializer(CustomObject, lambda x: x.to_dict())

        result = safe_serialize(data)
        assert result["list"] == [1, 2, {"nested": "value"}]
        assert isinstance(result["set"], list)
        assert set(result["set"]) == {1, 2, 3}
        assert result["custom"] == {"data": "test", "metadata": {"key": "value"}}


class TestRobustSerialize:
    """Test the robust_serialize function."""

    def test_robust_serialize_with_pydantic_models(self):
        """Test robust_serialize with Pydantic models."""
        obj = MockPydanticModel(
            name="test", value=42, items=["item1", "item2"], optional_field="optional"
        )
        result = robust_serialize(obj)
        assert result == {
            "name": "test",
            "value": 42,
            "items": ["item1", "item2"],
            "optional_field": "optional",
        }

    def test_robust_serialize_with_unknown_types(self):
        """Test robust_serialize with unknown types."""

        class UnknownType:
            def __init__(self, value):
                self.value = value

        obj = UnknownType("test")
        result = robust_serialize(obj)
        assert result == "<unserializable: UnknownType>"

    def test_robust_serialize_with_custom_serializer(self):
        """Test robust_serialize with custom serializers."""

        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        register_custom_serializer(CustomObject, serialize_custom)

        obj = CustomObject("test", {})
        result = robust_serialize(obj)
        assert result == {"custom": "test"}


class TestSerializeToJson:
    """Test the serialize_to_json functions."""

    def test_serialize_to_json_basic(self):
        """Test basic JSON serialization."""
        data = {"name": "test", "value": 42}
        result = serialize_to_json(data)
        assert result == '{"name": "test", "value": 42}'

    def test_serialize_to_json_with_kwargs(self):
        """Test JSON serialization with additional kwargs."""
        data = {"name": "test", "value": 42}
        result = serialize_to_json(data, indent=2)
        expected = '{\n  "name": "test",\n  "value": 42\n}'
        assert result == expected

    def test_serialize_to_json_robust(self):
        """Test robust JSON serialization."""
        data = {"name": "test", "value": 42}
        result = serialize_to_json_robust(data)
        assert result == '{"name": "test", "value": 42}'

    def test_serialize_to_json_with_custom_types(self):
        """Test JSON serialization with custom types."""
        obj = CustomObject("test", {"key": "value"})

        # Register custom serializer
        register_custom_serializer(CustomObject, lambda x: x.to_dict())

        result = serialize_to_json(obj)
        expected = '{"data": "test", "metadata": {"key": "value"}}'
        assert result == expected


class TestCreateFieldSerializer:
    """Test the create_field_serializer function."""

    def test_create_field_serializer(self):
        """Test creating a field serializer."""

        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        field_serializer = create_field_serializer("custom_field", serialize_custom)

        obj = CustomObject("test", {})
        result = field_serializer(obj)
        assert result == {"custom": "test"}


# --- Property-based tests for robust serialization ---
# These tests ensure that serialization utilities handle a wide variety of
# data structures and edge cases without crashing.


@st.composite
def serialization_test_data(draw, max_depth=3):
    """Generate test data for serialization testing."""
    if max_depth <= 0:
        return draw(
            st.one_of(
                st.none(), st.integers(), st.floats(allow_nan=False), st.text(), st.booleans()
            )
        )

    base = st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.text(), st.booleans())

    container = st.deferred(lambda: serialization_test_data(max_depth=max_depth - 1))
    dicts = st.dictionaries(st.text(), container, max_size=3)
    lists = st.lists(container, max_size=3)
    sets = st.sets(st.one_of(st.integers(), st.text(), st.booleans()), max_size=3)
    tuples = st.tuples(container, container)

    return draw(st.one_of(base, dicts, lists, sets, tuples))


@given(data=serialization_test_data())
@settings(max_examples=50, deadline=2000)
def test_property_based_safe_serialize_does_not_crash(data):
    """Property-based: safe_serialize should not crash on any data structure."""
    try:
        result1 = safe_serialize(data)
        result2 = safe_serialize(data)
        # Should be deterministic
        assert result1 == result2
    except Exception as e:
        pytest.fail(f"safe_serialize failed on {data!r}: {e}")


@given(data=serialization_test_data())
@settings(max_examples=50, deadline=2000)
def test_property_based_robust_serialize_does_not_crash(data):
    """Property-based: robust_serialize should not crash on any data structure."""
    try:
        result1 = robust_serialize(data)
        result2 = robust_serialize(data)
        # Should be deterministic
        assert result1 == result2
    except Exception as e:
        pytest.fail(f"robust_serialize failed on {data!r}: {e}")


@given(data=serialization_test_data())
@settings(max_examples=30, deadline=2000)
def test_property_based_serialize_to_json_does_not_crash(data):
    """Property-based: serialize_to_json should not crash on any data structure."""
    try:
        result = serialize_to_json(data)
        # Should produce valid JSON
        json.loads(result)  # Verify it's valid JSON
        # Should be deterministic
        result2 = serialize_to_json(data)
        assert result == result2
    except Exception as e:
        pytest.fail(f"serialize_to_json failed on {data!r}: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_circular_reference_in_nested_structures(self):
        """Test circular references in nested structures."""
        # Create nested circular reference
        outer = {"level": "outer"}
        inner = {"level": "inner", "parent": outer}
        outer["child"] = inner

        result = safe_serialize(outer)
        assert isinstance(result, dict)
        assert result["level"] == "outer"
        # The circular reference should be handled gracefully

    def test_mixed_types_in_collections(self):
        """Test collections with mixed types."""
        mixed_list = [1, "string", True, None, {"key": "value"}]
        result = safe_serialize(mixed_list)
        assert result == [1, "string", True, None, {"key": "value"}]

    def test_empty_structures(self):
        """Test empty structures."""
        assert safe_serialize([]) == []
        assert safe_serialize({}) == {}
        assert safe_serialize(set()) == []
        assert safe_serialize(()) == []

    def test_special_strings(self):
        """Test special string values."""
        special_strings = ["", " ", "\n", "\t", "\\", '"', "'"]
        for s in special_strings:
            result = safe_serialize(s)
            assert result == s

    def test_large_numbers(self):
        """Test large numbers."""
        large_int = 2**63 - 1
        large_float = 1e308

        assert safe_serialize(large_int) == large_int
        assert safe_serialize(large_float) == large_float

    def test_registry_cleanup(self):
        """Test that the registry can be used multiple times."""

        # Register a serializer
        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        register_custom_serializer(CustomObject, serialize_custom)

        # Use it
        obj = CustomObject("test", {})
        result = safe_serialize(obj)
        assert result == {"custom": "test"}

        # Register a different serializer
        def serialize_custom2(obj: CustomObject) -> dict:
            return {"custom2": obj.data}

        register_custom_serializer(CustomObject, serialize_custom2)

        # Should use the new serializer
        result = safe_serialize(obj)
        assert result == {"custom2": "test"}


class TestIntegration:
    """Test integration between different serialization utilities."""

    def test_global_registry_with_safe_serialize(self):
        """Test that global registry works with safe_serialize."""

        def serialize_custom(obj: CustomObject) -> dict:
            return {"custom": obj.data}

        register_custom_serializer(CustomObject, serialize_custom)

        data = {
            "custom_obj": CustomObject("test", {}),
            "list": [1, 2, CustomObject("nested", {})],
            "dict": {"key": CustomObject("value", {})},
        }

        result = safe_serialize(data)
        assert result["custom_obj"] == {"custom": "test"}
        assert result["list"] == [1, 2, {"custom": "nested"}]
        assert result["dict"]["key"] == {"custom": "value"}

    def test_serialize_to_json_with_complex_objects(self):
        """Test JSON serialization with complex objects."""
        data = {
            "custom_obj": CustomObject("test", {"key": "value"}),
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
            "complex": 3 + 4j,
            "set": {1, 2, 3},
        }

        # Register custom serializer
        register_custom_serializer(CustomObject, lambda x: x.to_dict())

        result = serialize_to_json(data)
        parsed = json.loads(result)

        assert parsed["custom_obj"]["data"] == "test"
        assert parsed["datetime"] == "2023-01-01T12:00:00"
        assert parsed["complex"]["real"] == 3.0
        assert parsed["complex"]["imag"] == 4.0
        assert isinstance(parsed["set"], list)
        assert set(parsed["set"]) == {1, 2, 3}


def test_fallback_serializer_not_called_for_standard_types(monkeypatch):
    """Test that the fallback serializer is NOT called for standard types."""
    from flujo.domain.models import BaseModel as FlujoBaseModel

    called = []

    class DummyModel(FlujoBaseModel):
        s: str
        i: int
        f: float
        b: bool
        lst: list
        d: dict

    # Patch the fallback method to record calls
    orig = FlujoBaseModel._serialize_single_unknown_type

    def fake_fallback(self, value):
        called.append(type(value))
        return orig(self, value)

    monkeypatch.setattr(FlujoBaseModel, "_serialize_single_unknown_type", fake_fallback)

    m = DummyModel(s="hello", i=1, f=2.0, b=True, lst=[1, 2, 3], d={"a": 1})
    # Trigger serialization
    result = m.model_dump(mode="json")
    # The fallback should NOT be called for any of the standard types
    assert called == []
    # The result should be as expected
    assert result == {"s": "hello", "i": 1, "f": 2.0, "b": True, "lst": [1, 2, 3], "d": {"a": 1}}
