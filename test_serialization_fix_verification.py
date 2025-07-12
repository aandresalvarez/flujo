#!/usr/bin/env python3
"""Test script to verify serialization improvements."""

import json
from flujo.utils.serialization import safe_serialize, _serialize_for_key


def test_dictionary_key_serialization():
    """Test that dictionary keys are properly serialized as strings."""
    # Test with various key types
    test_dict = {
        "string_key": "value1",
        123: "value2",  # int key
        (1, 2, 3): "value3",  # tuple key
        None: "value5",  # None key
    }

    serialized = safe_serialize(test_dict)

    # All keys should be strings
    for key in serialized.keys():
        assert isinstance(key, str), f"Key {key} is not a string: {type(key)}"

    print("✅ Dictionary key serialization test passed")


def test_serialize_for_key_returns_string():
    """Test that _serialize_for_key always returns a string."""
    test_cases = [
        "string",
        123,
        (1, 2, 3),
        None,
        [1, 2, 3],
        {"a": 1, "b": 2},
    ]

    for obj in test_cases:
        result = _serialize_for_key(obj)
        assert isinstance(result, str), f"Expected string, got {type(result)} for {obj}"

    print("✅ _serialize_for_key string conversion test passed")


def test_json_compatibility():
    """Test that serialized objects are JSON compatible."""
    test_obj = {
        "string": "value",
        123: "int_key",
        (1, 2): "tuple_key",
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
    }

    serialized = safe_serialize(test_obj)

    # Should be JSON serializable
    json_str = json.dumps(serialized)
    assert isinstance(json_str, str)

    # Should be JSON deserializable
    deserialized = json.loads(json_str)
    assert isinstance(deserialized, dict)

    print("✅ JSON compatibility test passed")


if __name__ == "__main__":
    print("🧪 Running serialization improvement verification tests...")

    test_dictionary_key_serialization()
    test_serialize_for_key_returns_string()
    test_json_compatibility()

    print("\n✅ All tests passed! Serialization improvements are working correctly.")
