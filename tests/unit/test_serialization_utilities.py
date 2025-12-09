"""Unit tests for serialization utilities."""

from flujo.type_definitions.common import JSONObject
import json
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Optional
from unittest.mock import patch

from hypothesis import given, strategies as st
from pydantic import BaseModel

from flujo.utils.serialization import (
    create_field_serializer,
    create_serializer_for_type,
    lookup_custom_deserializer,
    lookup_custom_serializer,
    register_custom_deserializer,
    register_custom_serializer,
    reset_custom_serializer_registry,
    _robust_serialize_internal,
    safe_deserialize,
    _serialize_for_json,
    _serialize_to_json_internal,
)


# Rename helper classes to avoid Pytest collection
class MyEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


class MyPydanticModel(BaseModel):
    name: str
    value: int
    optional_field: Optional[str] = None


class MyCustomObject:
    """Test custom object for serialization tests."""

    def __init__(self, data: str):
        self.data = data

    def to_dict(self) -> JSONObject:
        return {"data": self.data}


class TestSerializationRegistry:
    """Test custom serializer/deserializer registry."""

    def test_register_custom_serializer(self):
        """Test registering a custom serializer."""

        def custom_serializer(obj: MyCustomObject) -> JSONObject:
            return {"custom_data": obj.data}

        register_custom_serializer(MyCustomObject, custom_serializer)

        # Test lookup
        serializer = lookup_custom_serializer(MyCustomObject("test"))
        assert serializer is not None
        result = serializer(MyCustomObject("test"))
        assert result == {"custom_data": "test"}

    def test_register_custom_deserializer(self):
        """Test registering a custom deserializer."""

        def custom_deserializer(data: JSONObject) -> MyCustomObject:
            return MyCustomObject(data["custom_data"])

        register_custom_deserializer(MyCustomObject, custom_deserializer)

        # Test lookup
        deserializer = lookup_custom_deserializer(MyCustomObject)
        assert deserializer is not None
        result = deserializer({"custom_data": "test"})
        assert isinstance(result, MyCustomObject)
        assert result.data == "test"

    def test_lookup_custom_serializer_not_found(self):
        """Test looking up a non-existent custom serializer."""
        result = lookup_custom_serializer("not_registered")
        assert result is None

    def test_lookup_custom_deserializer_not_found(self):
        """Test looking up a non-existent custom deserializer."""
        result = lookup_custom_deserializer(str)
        assert result is None

    def test_reset_custom_serializer_registry(self):
        """Test resetting the custom serializer registry."""

        def custom_serializer(obj: MyCustomObject) -> JSONObject:
            return {"data": obj.data}

        register_custom_serializer(MyCustomObject, custom_serializer)

        # Verify it's registered
        assert lookup_custom_serializer(MyCustomObject("test")) is not None

        # Reset
        reset_custom_serializer_registry()

        # Verify it's gone
        assert lookup_custom_serializer(MyCustomObject("test")) is None


class TestSerializerFactories:
    """Test serializer factory functions."""

    def test_create_serializer_for_type(self):
        """Test creating a serializer for a specific type."""

        def custom_serializer(obj: MyCustomObject) -> JSONObject:
            return {"serialized": obj.data}

        serializer = create_serializer_for_type(MyCustomObject, custom_serializer)

        # Test with correct type
        result = serializer(MyCustomObject("test"))
        assert result == {"serialized": "test"}

        # Test with different type (should return unchanged)
        result = serializer("not_custom_object")
        assert result == "not_custom_object"

    def test_create_field_serializer(self):
        """Test creating a field serializer."""

        def custom_serializer(value: str) -> str:
            return value.upper()

        field_serializer = create_field_serializer("test_field", custom_serializer)

        result = field_serializer("hello")
        assert result == "HELLO"


class TestSerializeForJson:
    """Test _serialize_for_json function."""

    def test__serialize_for_json_basic_types(self):
        """Test serializing basic types."""
        # String
        result = _serialize_for_json("hello")
        assert result == "hello"

        # Integer
        result = _serialize_for_json(42)
        assert result == 42

        # Float
        result = _serialize_for_json(3.14)
        assert result == 3.14

        # Boolean
        result = _serialize_for_json(True)
        assert result is True

        # None
        result = _serialize_for_json(None)
        assert result is None

    def test__serialize_for_json_datetime_objects(self):
        """Test serializing datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = _serialize_for_json(dt)
        assert isinstance(result, str)
        assert "2023-01-01T12:00:00" in result

        d = date(2023, 1, 1)
        result = _serialize_for_json(d)
        assert isinstance(result, str)
        assert "2023-01-01" in result

        t = time(12, 0, 0)
        result = _serialize_for_json(t)
        assert isinstance(result, str)
        assert "12:00:00" in result

    def test__serialize_for_json_enum(self):
        """Test serializing enum values."""
        # Should serialize to .value, not .name
        result = _serialize_for_json(MyEnum.VALUE1)
        assert result == "value1"

    def test__serialize_for_json_pydantic_model(self):
        """Test serializing Pydantic models."""
        model = MyPydanticModel(name="test", value=42)
        result = _serialize_for_json(model)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test__serialize_for_json_list(self):
        """Test serializing lists."""
        data = [1, "hello", MyEnum.VALUE1]
        result = _serialize_for_json(data)
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == "hello"
        assert result[2] == "value1"

    def test__serialize_for_json_dict(self):
        """Test serializing dictionaries."""
        data = {
            "string": "hello",
            "number": 42,
            "enum": MyEnum.VALUE1,
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
        }
        result = _serialize_for_json(data)
        assert isinstance(result, dict)
        assert result["string"] == "hello"
        assert result["number"] == 42
        assert result["enum"] == "value1"
        assert isinstance(result["datetime"], str)

    def test__serialize_for_json_set(self):
        """Test serializing sets."""
        data = {1, 2, 3}
        result = _serialize_for_json(data)
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test__serialize_for_json_custom_object(self):
        """Test serializing custom objects."""
        obj = MyCustomObject("test")
        # Without custom serializer we fall back to string representation
        result = _serialize_for_json(obj)
        assert isinstance(result, str)

    def test__serialize_for_json_with_custom_serializer(self):
        """Test serializing with custom serializer."""

        def custom_serializer(obj: MyCustomObject) -> JSONObject:
            return {"custom_data": obj.data}

        register_custom_serializer(MyCustomObject, custom_serializer)

        obj = MyCustomObject("test")
        result = _serialize_for_json(obj)
        assert result == {"custom_data": "test"}

        # Clean up
        reset_custom_serializer_registry()

    def test__serialize_for_json_circular_reference(self):
        """Test serializing objects with circular references."""
        # Create circular reference
        data = {"key": "value"}
        data["self"] = data

        result = _serialize_for_json(data)
        # Should handle circular reference gracefully
        assert isinstance(result, dict)
        assert result["key"] == "value"
        # The circular reference should be handled (might be string representation)
        assert "self" in result

    def test__serialize_for_json_with_default_serializer(self):
        """Test serializing with default serializer."""

        # Only used for unknown types, not primitives
        def default_serializer(obj: Any) -> str:
            return f"custom_{obj!s}"

        # For a primitive, should return as is
        result = _serialize_for_json("test", default_serializer=default_serializer)
        assert result == "test"

        # For unknown type, should use default_serializer
        class Unknown:
            pass

        unknown = Unknown()
        result = _serialize_for_json(unknown, default_serializer=default_serializer)
        assert result == f"custom_{unknown!s}"

    def test__serialize_for_json_nested_structures(self):
        """Test serializing nested data structures."""
        data = {
            "list": [1, 2, {"nested": "value"}],
            "dict": {"key": [MyEnum.VALUE1, datetime(2023, 1, 1)]},
            "set": {1, 2, 3},
        }

        result = _serialize_for_json(data)
        assert isinstance(result, dict)
        assert isinstance(result["list"], list)
        assert isinstance(result["dict"], dict)
        assert isinstance(result["set"], list)

    def test__serialize_for_json_deep_circular_reference(self):
        """Test that deep nested circular references are handled and _seen is only cleared at the top level."""
        # Create a deeply nested structure with a circular reference
        a = {}
        b = {"child": a}
        a["parent"] = b  # Circular reference
        a["self"] = a  # Self-reference
        # Should not raise RecursionError or leak _seen state
        result = _serialize_for_json(a)
        assert isinstance(result, dict)
        assert "parent" in result
        assert "self" in result
        # The circular reference should be handled (should be None or similar for the cycle)
        assert result["parent"]["child"] is not None
        # Serializing again should not be affected by previous call
        result2 = _serialize_for_json(a)
        assert isinstance(result2, dict)
        assert result2["parent"]["child"] is not None


class TestSafeDeserialize:
    """Test safe_deserialize function."""

    def test_safe_deserialize_basic_types(self):
        """Test deserializing basic types."""
        # String
        result = safe_deserialize("hello")
        assert result == "hello"

        # Integer
        result = safe_deserialize(42)
        assert result == 42

        # Float
        result = safe_deserialize(3.14)
        assert result == 3.14

        # Boolean
        result = safe_deserialize(True)
        assert result is True

        # None
        result = safe_deserialize(None)
        assert result is None

    def test_safe_deserialize_datetime_strings(self):
        """Test deserializing datetime strings."""
        # Should return the string, not a datetime object, unless a custom deserializer is registered
        dt_str = "2023-01-01T12:00:00"
        result = safe_deserialize(dt_str, target_type=datetime)
        assert result == dt_str
        date_str = "2023-01-01"
        result = safe_deserialize(date_str, target_type=date)
        assert result == date_str
        time_str = "12:00:00"
        result = safe_deserialize(time_str, target_type=time)
        assert result == time_str

    def test_safe_deserialize_enum(self):
        """Test deserializing enum values."""
        # Should return the string, not the enum, unless a custom deserializer is registered
        result = safe_deserialize("value1", target_type=MyEnum)
        assert result == "value1"

    def test_safe_deserialize_pydantic_model(self):
        """Test deserializing Pydantic models."""
        # Should return the dict, not the model, unless a custom deserializer is registered
        data = {"name": "test", "value": 42}
        result = safe_deserialize(data, target_type=MyPydanticModel)
        assert result == data

    def test_safe_deserialize_with_custom_deserializer(self):
        """Test deserializing with custom deserializer."""

        def custom_deserializer(data: JSONObject) -> MyCustomObject:
            return MyCustomObject(data["custom_data"])

        register_custom_deserializer(MyCustomObject, custom_deserializer)

        data = {"custom_data": "test"}
        result = safe_deserialize(data, target_type=MyCustomObject)
        assert isinstance(result, MyCustomObject)
        assert result.data == "test"

        # Clean up
        reset_custom_serializer_registry()

    def test_safe_deserialize_with_default_deserializer(self):
        """Test deserializing with default deserializer."""

        # Only used for unknown types
        def default_deserializer(data: Any) -> str:
            return f"deserialized_{str(data)}"

        # For a primitive, should return as is
        result = safe_deserialize("test", default_deserializer=default_deserializer)
        assert result == "test"

        # For unknown type, should use default_deserializer
        class Unknown:
            pass

        unknown = Unknown()
        result = safe_deserialize(unknown, default_deserializer=default_deserializer)
        assert result == f"deserialized_{str(unknown)}"

    def test_safe_deserialize_list(self):
        """Test deserializing lists."""
        data = [1, "hello", "VALUE1"]
        result = safe_deserialize(data)
        assert isinstance(result, list)
        assert result == [1, "hello", "VALUE1"]

    def test_safe_deserialize_dict(self):
        """Test deserializing dictionaries."""
        data = {"key": "value", "number": 42}
        result = safe_deserialize(data)
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42


class TestRobustSerialize:
    """Test robust_serialize function."""

    def test_robust_serialize_basic_types(self):
        """Test robust serialization of basic types."""
        result = _robust_serialize_internal("hello")
        assert result == "hello"

        result = _robust_serialize_internal(42)
        assert result == 42

    def test_robust_serialize_complex_objects(self):
        """Test robust serialization of complex objects."""
        obj = MyCustomObject("test")
        result = _robust_serialize_internal(obj)
        # Should handle unknown objects gracefully
        assert isinstance(result, str)

    def test_robust_serialize_with_fallback(self):
        """Test robust serialization with fallback."""

        # Should return fallback string if fallback also fails
        def fallback_serializer(obj: Any) -> str:
            return f"fallback_{str(obj)}"

        with patch(
            "flujo.utils.serialization._json_serialize_impl",
            side_effect=Exception("Test error"),
        ):
            result = _robust_serialize_internal(object())
            assert result.startswith("<unserializable: ")


class TestSerializeToJson:
    """Test serialize_to_json function."""

    def test_serialize_to_json_basic(self):
        """Test basic JSON serialization."""
        data = {"key": "value", "number": 42}
        result = _serialize_to_json_internal(data)
        assert isinstance(result, str)

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_serialize_to_json_with_kwargs(self):
        """Test JSON serialization with kwargs."""
        data = {"key": "value"}
        result = _serialize_to_json_internal(data, indent=2)
        assert isinstance(result, str)
        assert "  " in result  # Should have indentation

    def test_serialize_to_json_robust(self):
        """Test robust JSON serialization."""
        data = {"key": "value", "datetime": datetime(2023, 1, 1, 12, 0, 0)}
        result = _serialize_to_json_internal(data)
        assert isinstance(result, str)

        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert "datetime" in parsed


# Hypothesis-based property tests
class TestSerializationProperties:
    """Property-based tests for serialization functions."""

    @given(st.text())
    def test__serialize_for_json_text_roundtrip(self, text):
        """Test that text serialization is idempotent."""
        result = _serialize_for_json(text)
        assert result == text

    @given(st.integers())
    def test__serialize_for_json_integer_roundtrip(self, integer):
        """Test that integer serialization is idempotent."""
        result = _serialize_for_json(integer)
        assert result == integer

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test__serialize_for_json_float_roundtrip(self, float_val):
        """Test that float serialization is idempotent."""
        result = _serialize_for_json(float_val)
        assert result == float_val

    @given(st.booleans())
    def test__serialize_for_json_boolean_roundtrip(self, boolean):
        """Test that boolean serialization is idempotent."""
        result = _serialize_for_json(boolean)
        assert result == boolean

    @given(st.lists(st.text()))
    def test__serialize_for_json_list_roundtrip(self, text_list):
        """Test that list serialization preserves structure."""
        result = _serialize_for_json(text_list)
        assert isinstance(result, list)
        assert len(result) == len(text_list)

    @given(st.dictionaries(st.text(), st.text()))
    def test__serialize_for_json_dict_roundtrip(self, text_dict):
        """Test that dict serialization preserves structure."""
        result = _serialize_for_json(text_dict)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(text_dict.keys())

    @given(st.text())
    def test_serialize_to_json_roundtrip(self, text):
        """Test that JSON serialization can be parsed back."""
        data = {"text": text}
        json_str = _serialize_to_json_internal(data)
        parsed = json.loads(json_str)
        assert parsed["text"] == text

    @given(st.text())
    def test_serialize_to_json_robust_roundtrip(self, text):
        """Test that robust JSON serialization can be parsed back."""
        data = {"text": text}
        json_str = _serialize_to_json_internal(data)
        parsed = json.loads(json_str)
        assert parsed["text"] == text
