#!/usr/bin/env python3
"""Test script to verify the serialization functions work as documented."""

import json
from datetime import datetime
from enum import Enum
from typing import Any

from flujo.utils.serialization import (
    register_custom_serializer,
    lookup_custom_serializer,
    create_serializer_for_type,
    create_field_serializer,
    serializable_field,
    safe_serialize,
    serialize_to_json,
)


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class DatabaseConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port


def test_register_custom_serializer():
    """Test register_custom_serializer functionality."""
    print("Testing register_custom_serializer...")

    # Test datetime serialization
    def serialize_datetime(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    register_custom_serializer(datetime, serialize_datetime)

    # Test that it works
    dt = datetime(2023, 1, 1, 12, 30, 45)
    result = safe_serialize(dt)
    expected = "2023-01-01 12:30:45"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ register_custom_serializer works correctly")


def test_lookup_custom_serializer():
    """Test lookup_custom_serializer functionality."""
    print("Testing lookup_custom_serializer...")

    # Test with registered type
    dt = datetime(2023, 1, 1, 12, 30, 45)
    serializer = lookup_custom_serializer(dt)
    assert serializer is not None, "Should find serializer for datetime"

    # Test with unregistered type
    serializer = lookup_custom_serializer("string")
    assert serializer is None, "Should not find serializer for string"
    print("✓ lookup_custom_serializer works correctly")


def test_create_serializer_for_type():
    """Test create_serializer_for_type functionality."""
    print("Testing create_serializer_for_type...")

    def serialize_priority(p: Priority) -> str:
        return p.name.lower()

    MyTypeSerializer = create_serializer_for_type(Priority, serialize_priority)

    # Test that it works
    priority = Priority.HIGH
    result = safe_serialize(priority)
    expected = "high"
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ create_serializer_for_type works correctly")


def test_create_field_serializer():
    """Test create_field_serializer functionality."""
    print("Testing create_field_serializer...")

    def serialize_db_connection(conn: DatabaseConnection) -> dict:
        return {
            "type": "database_connection",
            "host": conn.host,
            "port": conn.port
        }

    field_serializer = create_field_serializer('db', serialize_db_connection)

    # Test that it works
    db = DatabaseConnection("localhost", 5432)
    result = field_serializer(db)
    expected = {"type": "database_connection", "host": "localhost", "port": 5432}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ create_field_serializer works correctly")


def test_serializable_field():
    """Test serializable_field functionality (deprecated but should not raise)."""
    print("Testing serializable_field...")

    # This should not raise an error (it's deprecated but should be a no-op)
    @serializable_field(lambda x: x.to_dict())
    def some_function():
        pass

    print("✓ serializable_field works correctly (no-op for deprecated function)")


def test_complex_serialization():
    """Test complex serialization scenarios."""
    print("Testing complex serialization...")

    # Register multiple serializers
    register_custom_serializer(complex, lambda c: f"{c.real:.2f} + {c.imag:.2f}i")
    register_custom_serializer(set, list)  # Convert sets to lists

    # Test complex object
    data = {
        "timestamp": datetime(2023, 1, 1, 12, 30, 45),
        "priority": Priority.HIGH,
        "complex_num": 3.14159 + 2.71828j,
        "set_data": {1, 2, 3, 4, 5},
        "db_connection": DatabaseConnection("localhost", 5432)
    }

    # Register serializer for DatabaseConnection
    def serialize_db_connection(conn: DatabaseConnection) -> dict:
        return {"host": conn.host, "port": conn.port}

    register_custom_serializer(DatabaseConnection, serialize_db_connection)

    # Serialize the complex object
    result = safe_serialize(data)

    # Verify the result
    assert result["timestamp"] == "2023-01-01 12:30:45"
    assert result["priority"] == "high"
    assert result["complex_num"] == "3.14 + 2.72i"
    assert result["set_data"] == [1, 2, 3, 4, 5]
    assert result["db_connection"] == {"host": "localhost", "port": 5432}

    print("✓ Complex serialization works correctly")


def test_json_serialization():
    """Test JSON serialization."""
    print("Testing JSON serialization...")

    data = {
        "timestamp": datetime(2023, 1, 1, 12, 30, 45),
        "priority": Priority.HIGH,
    }

    json_str = serialize_to_json(data)
    parsed = json.loads(json_str)

    assert parsed["timestamp"] == "2023-01-01 12:30:45"
    assert parsed["priority"] == "high"

    print("✓ JSON serialization works correctly")


def main():
    """Run all tests."""
    print("Testing serialization functions...\n")

    try:
        test_register_custom_serializer()
        test_lookup_custom_serializer()
        test_create_serializer_for_type()
        test_create_field_serializer()
        test_serializable_field()
        test_complex_serialization()
        test_json_serialization()

        print("\n🎉 All tests passed! The serialization functions work as documented.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
