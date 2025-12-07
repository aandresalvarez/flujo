"""Tests for the serialization registry functionality."""

import pytest
from dataclasses import dataclass
from typing import Any, List
from flujo.type_definitions.common import JSONObject

from flujo.utils.serialization import (
    register_custom_serializer,
    register_custom_deserializer,
    lookup_custom_serializer,
    lookup_custom_deserializer,
    safe_deserialize,
    reset_custom_serializer_registry,
)
from flujo.state.backends.base import _serialize_for_json
import json


def _serialize(obj: Any) -> Any:
    normalized = _serialize_for_json(obj)
    return json.loads(json.dumps(normalized, ensure_ascii=False))


class MockCustomType:
    """A custom type for testing serialization."""

    def __init__(self, value: str, metadata: JSONObject):
        self.value = value
        self.metadata = metadata

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockCustomType):
            return False
        return self.value == other.value and self.metadata == other.metadata


class MockEnum:
    """A simple enum-like class for testing."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MockEnum):
            return False
        return self.name == other.name and self.value == other.value


@dataclass
class MockDataclass:
    """A dataclass for testing serialization."""

    field1: str
    field2: int
    field3: List[str]


class TestSerializationRegistry:
    """Test the serialization registry functionality."""

    def setup_method(self):
        """Reset the registry before each test."""
        reset_custom_serializer_registry()

    def teardown_method(self):
        """Reset the registry after each test."""
        reset_custom_serializer_registry()

    def test_register_custom_serializer(self):
        """Test registering a custom serializer."""

        # Create a custom serializer
        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value, "metadata": obj.metadata}

        # Register the serializer
        register_custom_serializer(MockCustomType, serialize_custom_type)

        # Verify it's registered - lookup_custom_serializer expects an instance, not a type
        obj = MockCustomType("test_value", {"key": "value"})
        serializer = lookup_custom_serializer(obj)
        assert serializer is not None
        assert serializer == serialize_custom_type

        # Test with an instance
        obj = MockCustomType("test_value", {"key": "value"})
        serializer = lookup_custom_serializer(obj)
        assert serializer is not None
        assert serializer == serialize_custom_type

    def test_register_custom_deserializer(self):
        """Test registering a custom deserializer."""

        # Create a custom deserializer
        def deserialize_custom_type(data: JSONObject) -> MockCustomType:
            return MockCustomType(data["value"], data["metadata"])

        # Register the deserializer
        register_custom_deserializer(MockCustomType, deserialize_custom_type)

        # Verify it's registered
        deserializer = lookup_custom_deserializer(MockCustomType)
        assert deserializer is not None
        assert deserializer == deserialize_custom_type

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization work together."""

        # Create serializers and deserializers
        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value, "metadata": obj.metadata}

        def deserialize_custom_type(data: JSONObject) -> MockCustomType:
            return MockCustomType(data["value"], data["metadata"])

        # Register both
        register_custom_serializer(MockCustomType, serialize_custom_type)
        register_custom_deserializer(MockCustomType, deserialize_custom_type)

        # Create test object
        original_obj = MockCustomType("test_value", {"key": "value", "number": 42})

        # Serialize
        serialized = _serialize(original_obj)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "MockCustomType"
        assert serialized["value"] == "test_value"
        assert serialized["metadata"] == {"key": "value", "number": 42}

        # Deserialize
        deserialized = safe_deserialize(serialized, MockCustomType)
        assert isinstance(deserialized, MockCustomType)
        assert deserialized == original_obj

    def test_serialization_with_nested_objects(self):
        """Test serialization with nested custom objects."""

        # Create serializers and deserializers for both types
        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value, "metadata": obj.metadata}

        def deserialize_custom_type(data: JSONObject) -> MockCustomType:
            return MockCustomType(data["value"], data["metadata"])

        def serialize_enum(obj: MockEnum) -> JSONObject:
            return {"type": "MockEnum", "name": obj.name, "value": obj.value}

        def deserialize_enum(data: JSONObject) -> MockEnum:
            return MockEnum(data["name"], data["value"])

        # Register all serializers and deserializers
        register_custom_serializer(MockCustomType, serialize_custom_type)
        register_custom_deserializer(MockCustomType, deserialize_custom_type)
        register_custom_serializer(MockEnum, serialize_enum)
        register_custom_deserializer(MockEnum, deserialize_enum)

        # Create nested object
        enum_obj = MockEnum("test_enum", 42)
        custom_obj = MockCustomType("test_value", {"enum": enum_obj})

        # Serialize
        serialized = _serialize(custom_obj)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "MockCustomType"
        assert serialized["metadata"]["enum"]["type"] == "MockEnum"

        # Deserialize - need to handle nested objects manually
        deserialized = safe_deserialize(serialized, MockCustomType)
        assert isinstance(deserialized, MockCustomType)
        # The nested enum will be a dict, not a MockEnum instance
        # This is expected behavior since safe_deserialize doesn't automatically
        # deserialize nested custom types
        assert isinstance(deserialized.metadata["enum"], dict)
        assert deserialized.metadata["enum"]["name"] == "test_enum"
        assert deserialized.metadata["enum"]["value"] == 42

    def test_serialization_with_dataclass(self):
        """Test serialization with dataclasses."""

        # Create serializers and deserializers for dataclass
        def serialize_dataclass(obj: MockDataclass) -> JSONObject:
            return {
                "type": "MockDataclass",
                "field1": obj.field1,
                "field2": obj.field2,
                "field3": obj.field3,
            }

        def deserialize_dataclass(data: JSONObject) -> MockDataclass:
            return MockDataclass(data["field1"], data["field2"], data["field3"])

        # Register serializers and deserializers
        register_custom_serializer(MockDataclass, serialize_dataclass)
        register_custom_deserializer(MockDataclass, deserialize_dataclass)

        # Create test object
        original_obj = MockDataclass("test", 42, ["item1", "item2"])

        # Serialize
        serialized = _serialize(original_obj)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "MockDataclass"
        assert serialized["field1"] == "test"
        assert serialized["field2"] == 42
        assert serialized["field3"] == ["item1", "item2"]

        # Deserialize
        deserialized = safe_deserialize(serialized, MockDataclass)
        assert isinstance(deserialized, MockDataclass)
        assert deserialized == original_obj

    def test_lookup_with_inheritance(self):
        """Test that lookup works with inheritance."""

        class SubMockCustomType(MockCustomType):
            pass

        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value, "metadata": obj.metadata}

        # Register serializer for base class
        register_custom_serializer(MockCustomType, serialize_custom_type)

        # Test that subclass instances can find the serializer
        obj = SubMockCustomType("test_value", {"key": "value"})
        serializer = lookup_custom_serializer(obj)
        assert serializer is not None
        assert serializer == serialize_custom_type

    def test_lookup_with_deserializer_inheritance(self):
        """Test that deserializer lookup works with inheritance."""

        class SubMockCustomType(MockCustomType):
            pass

        def deserialize_custom_type(data: JSONObject) -> MockCustomType:
            return MockCustomType(data["value"], data["metadata"])

        # Register deserializer for base class
        register_custom_deserializer(MockCustomType, deserialize_custom_type)

        # Test that subclass can find the deserializer
        deserializer = lookup_custom_deserializer(SubMockCustomType)
        assert deserializer is not None
        assert deserializer == deserialize_custom_type

    def test_registry_reset(self):
        """Test that the registry can be reset."""

        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value}

        # Register a serializer
        register_custom_serializer(MockCustomType, serialize_custom_type)
        obj = MockCustomType("test_value", {"key": "value"})
        assert lookup_custom_serializer(obj) is not None

        # Reset the registry
        reset_custom_serializer_registry()

        # Verify it's gone
        obj = MockCustomType("test_value", {"key": "value"})
        assert lookup_custom_serializer(obj) is None

    def test_thread_safety(self):
        """Test that the registry is thread-safe."""
        import threading
        import time

        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value}

        def register_in_thread():
            """Register a serializer in a separate thread."""
            time.sleep(0.01)  # Small delay to increase chance of race condition
            register_custom_serializer(MockCustomType, serialize_custom_type)

        def lookup_in_thread():
            """Look up a serializer in a separate thread."""
            time.sleep(0.01)  # Small delay to increase chance of race condition
            return lookup_custom_serializer(MockCustomType)

        # Start multiple threads
        threads = []
        for _ in range(5):
            t1 = threading.Thread(target=register_in_thread)
            t2 = threading.Thread(target=lookup_in_thread)
            threads.extend([t1, t2])

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify the registry is in a consistent state
        obj = MockCustomType("test_value", {"key": "value"})
        serializer = lookup_custom_serializer(obj)
        assert serializer is not None
        assert serializer == serialize_custom_type

    def test_serialization_error_handling(self):
        """Test that serialization errors are handled gracefully."""

        def bad_serializer(obj: MockCustomType) -> Any:
            raise ValueError("Serialization failed")

        # Register a bad serializer
        register_custom_serializer(MockCustomType, bad_serializer)

        # Try to serialize - should raise ValueError from the bad serializer
        obj = MockCustomType("test_value", {"key": "value"})
        with pytest.raises(ValueError, match="Serialization failed"):
            _serialize(obj)

    def test_deserialization_error_handling(self):
        """Test that deserialization errors are handled gracefully."""

        def bad_deserializer(data: JSONObject) -> MockCustomType:
            raise ValueError("Deserialization failed")

        # Register a bad deserializer
        register_custom_deserializer(MockCustomType, bad_deserializer)

        # Try to deserialize - should fall back to dict reconstruction
        data = {"value": "test_value", "metadata": {"key": "value"}}
        result = safe_deserialize(data, MockCustomType)
        # Should fall back to dict reconstruction
        assert isinstance(result, dict)
        assert result["value"] == "test_value"

    def test_circular_reference_handling(self):
        """Test that circular references are handled correctly."""

        def serialize_custom_type(obj: MockCustomType) -> JSONObject:
            return {"type": "MockCustomType", "value": obj.value, "metadata": obj.metadata}

        # Register serializer
        register_custom_serializer(MockCustomType, serialize_custom_type)

        # Create circular reference
        obj1 = MockCustomType("value1", {})
        obj2 = MockCustomType("value2", {})
        obj1.metadata["ref"] = obj2
        obj2.metadata["ref"] = obj1

        # Serialize - should handle circular reference gracefully
        serialized = _serialize(obj1)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "MockCustomType"
        assert serialized["value"] == "value1"
        # The circular reference should be replaced with a placeholder
        assert serialized["metadata"]["ref"]["type"] == "MockCustomType"
        assert serialized["metadata"]["ref"]["value"] == "value2"
        # The nested ref should be a placeholder
        assert serialized["metadata"]["ref"]["metadata"]["ref"] == "<circular-ref>"
