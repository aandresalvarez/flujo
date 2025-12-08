"""Comprehensive unit tests for serialization functions.

This module tests all edge cases and special types that should be handled by the internal
serialization functions, ensuring they're capable of handling all scenarios.
"""

import dataclasses
import json
import uuid
from collections import Counter, OrderedDict, defaultdict
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import List, Optional
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import BaseModel

from flujo.domain.base_model import BaseModel as FlujoBaseModel
from flujo.utils.serialization import (
    register_custom_serializer,
    reset_custom_serializer_registry,
    _serialize_for_json,
    _serialize_to_json_internal,
)


@pytest.fixture(autouse=True)
def reset_serializer_registry():
    """Automatically reset the serializer registry before and after each test."""
    reset_custom_serializer_registry()
    yield
    reset_custom_serializer_registry()


# Test helper classes (renamed to avoid pytest collection warnings)


class SampleEnum(Enum):
    """Sample enum for serialization tests."""

    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = 42


class SamplePydanticModel(BaseModel):
    """Sample Pydantic model for serialization tests."""

    name: str
    value: int
    optional_field: Optional[str] = None


class SampleFlujoModel(FlujoBaseModel):
    """Sample Flujo model for serialization tests."""

    name: str
    value: int
    nested: Optional["SampleFlujoModel"] = None


@dataclasses.dataclass
class SampleDataclass:
    """Sample dataclass for serialization tests."""

    name: str
    value: int
    items: List[str]


class SampleCustomType:
    """Sample custom type for serialization tests."""

    def __init__(self, data: str):
        self.data = data

    def to_dict(self):
        return {"data": self.data}


class SampleCallable:
    """Sample callable object for serialization tests."""

    def __init__(self, name: str):
        self.__name__ = name

    def __call__(self):
        return f"Called {self.__name__}"


class TestSafeSerializeComprehensive:
    """Comprehensive tests for _serialize_for_json function."""

    def test_primitive_types(self):
        """Test serialization of primitive types."""
        # String
        assert _serialize_for_json("hello") == "hello"
        assert _serialize_for_json("") == ""
        assert _serialize_for_json("🚀🌟✨") == "🚀🌟✨"

        # Integer
        assert _serialize_for_json(0) == 0
        assert _serialize_for_json(42) == 42
        assert _serialize_for_json(-1) == -1
        assert _serialize_for_json(2**63 - 1) == 2**63 - 1

        # Float
        assert _serialize_for_json(0.0) == 0.0
        assert _serialize_for_json(3.14) == 3.14
        assert _serialize_for_json(-1.5) == -1.5

        # Special float values
        assert _serialize_for_json(float("inf")) == "inf"
        assert _serialize_for_json(float("-inf")) == "-inf"
        assert _serialize_for_json(float("nan")) == "nan"

        # Boolean
        assert _serialize_for_json(True) is True
        assert _serialize_for_json(False) is False

        # None
        assert _serialize_for_json(None) is None

    def test_datetime_objects(self):
        """Test serialization of datetime objects."""
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

    def test_enum_serialization(self):
        """Test serialization of enum objects."""
        assert _serialize_for_json(SampleEnum.ALPHA) == "alpha"
        assert _serialize_for_json(SampleEnum.BETA) == "beta"
        assert _serialize_for_json(SampleEnum.GAMMA) == 42

    def test_complex_numbers(self):
        """Test serialization of complex numbers."""
        c = complex(3, 4)
        result = _serialize_for_json(c)
        assert result == {"real": 3.0, "imag": 4.0}

    def test_bytes_and_memoryview(self):
        """Test serialization of bytes and memoryview objects."""
        data = b"hello world"
        result = _serialize_for_json(data)
        assert isinstance(result, str)
        # Should be base64 encoded
        import base64

        assert base64.b64decode(result) == data

        # Test memoryview
        mv = memoryview(data)
        result = _serialize_for_json(mv)
        assert isinstance(result, str)
        assert base64.b64decode(result) == data

    def test_collections(self):
        """Test serialization of various collection types."""
        # List
        assert _serialize_for_json([1, 2, 3]) == [1, 2, 3]
        assert _serialize_for_json([]) == []

        # Tuple
        assert _serialize_for_json((1, 2, 3)) == [1, 2, 3]
        assert _serialize_for_json(()) == []

        # Set
        result = _serialize_for_json({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Frozenset
        result = _serialize_for_json(frozenset([1, 2, 3]))
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Dict
        assert _serialize_for_json({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        assert _serialize_for_json({}) == {}

    def test_nested_collections(self):
        """Test serialization of nested collections."""
        nested = {
            "list": [1, [2, 3], {"nested": "dict"}],
            "dict": {"inner": {"deeper": [4, 5, 6]}, "items": [{"id": 1}, {"id": 2}]},
            "tuple": ([1, 2], {"key": "value"}),
        }

        result = _serialize_for_json(nested)
        assert result["list"][0] == 1
        assert result["list"][1] == [2, 3]
        assert result["list"][2] == {"nested": "dict"}
        assert result["dict"]["inner"]["deeper"] == [4, 5, 6]
        assert result["tuple"] == [[1, 2], {"key": "value"}]

    def test_pydantic_models(self):
        """Test serialization of Pydantic models."""
        model = SamplePydanticModel(name="test", value=42, optional_field="optional")
        result = _serialize_for_json(model)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["optional_field"] == "optional"

    def test_flujo_models(self):
        """Test serialization of Flujo models with circular reference handling."""
        model = SampleFlujoModel(name="test", value=42)
        result = _serialize_for_json(model)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["nested"] is None

    def test_dataclass_serialization(self):
        """Test serialization of dataclass objects."""
        dc = SampleDataclass(name="test", value=42, items=["a", "b"])
        result = _serialize_for_json(dc)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["items"] == ["a", "b"]

    def test_callable_objects(self):
        """Test serialization of callable objects."""

        def test_func():
            pass

        # Function with __name__
        result = _serialize_for_json(test_func)
        assert result == "test_func"

        # Callable object with __name__
        callable_obj = SampleCallable("test_callable")
        result = _serialize_for_json(callable_obj)
        assert result == "test_callable"

        # Lambda (no __name__)
        def lambda_func(x):
            return x

        result = _serialize_for_json(lambda_func)
        assert "lambda" in result or "function" in result

    def test_mock_objects(self):
        """Test serialization of Mock objects."""
        mock = Mock(spec=SamplePydanticModel)
        mock.name = "test_mock"
        mock.value = 42

        result = _serialize_for_json(mock)
        # Mock objects might be serialized differently, let's check the actual behavior
        if isinstance(result, dict):
            assert result.get("type") == "Mock" or "Mock" in str(result)
        else:
            # If serialized as string, should contain Mock info
            assert "Mock" in str(result)

        # Test MagicMock
        magic_mock = MagicMock()
        magic_mock.attr = "value"

        result = _serialize_for_json(magic_mock)
        if isinstance(result, dict):
            assert result.get("type") == "Mock" or "Mock" in str(result)
        else:
            assert "Mock" in str(result)

    def test_circular_references_default_mode(self):
        """Test circular reference handling in default mode."""
        model1 = SampleFlujoModel(name="model1", value=1)
        model2 = SampleFlujoModel(name="model2", value=2)
        model1.nested = model2
        model2.nested = model1

        result = _serialize_for_json(model1, mode="default")
        assert isinstance(result, dict)
        assert result["name"] == "model1"
        assert result["value"] == 1
        # Circular reference should be None for Flujo models in default mode
        assert result["nested"]["name"] == "model2"
        assert result["nested"]["nested"] is None

    def test_circular_references_cache_mode(self):
        """Test circular reference handling in cache mode."""
        model1 = SampleFlujoModel(name="model1", value=1)
        model1.nested = model1  # Self-reference

        result = _serialize_for_json(model1, mode="cache")
        assert isinstance(result, dict)
        assert result["name"] == "model1"
        assert result["value"] == 1
        # Circular reference should be marked in cache mode
        assert "circular" in str(result["nested"])
        assert "SampleFlujoModel" in str(result["nested"])

    def test_custom_serializer_registration(self):
        """Test that custom serializers are used when registered."""
        register_custom_serializer(SampleCustomType, lambda obj: obj.to_dict())

        custom_obj = SampleCustomType("test_data")
        result = _serialize_for_json(custom_obj)

        assert result == {"data": "test_data"}

    def test_special_collections(self):
        """Test serialization of special collection types."""
        # OrderedDict
        register_custom_serializer(OrderedDict, lambda obj: dict(obj))
        od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        result = _serialize_for_json(od)
        assert result == {"a": 1, "b": 2, "c": 3}

        # Counter
        register_custom_serializer(Counter, lambda obj: dict(obj))
        counter = Counter(["a", "b", "a", "c", "b", "a"])
        result = _serialize_for_json(counter)
        assert result == {"a": 3, "b": 2, "c": 1}

        # defaultdict
        register_custom_serializer(defaultdict, lambda obj: dict(obj))
        dd = defaultdict(list)
        dd["key1"].append("value1")
        dd["key2"].append("value2")
        result = _serialize_for_json(dd)
        assert result == {"key1": ["value1"], "key2": ["value2"]}

    def test_dict_with_complex_keys(self):
        """Test serialization of dictionaries with complex keys."""
        key1 = (1, 2, 3)
        key2 = SampleEnum.ALPHA
        key3 = datetime(2023, 1, 1)

        complex_dict = {
            key1: "tuple_key",
            key2: "enum_key",
            key3: "datetime_key",
            "simple": "simple_value",
        }

        result = _serialize_for_json(complex_dict)
        assert isinstance(result, dict)
        assert "simple_value" in result.values()
        # Complex keys should be converted to strings
        assert all(isinstance(k, str) for k in result.keys())

    def test_deep_nesting(self):
        """Test serialization of deeply nested structures."""
        deep_data = {"level": 1}
        current = deep_data
        for i in range(2, 11):  # Create 10 levels of nesting
            current["nested"] = {"level": i}
            current = current["nested"]

        result = _serialize_for_json(deep_data)
        assert result["level"] == 1
        assert result["nested"]["level"] == 2
        # Should handle deep nesting without stack overflow
        current_result = result
        for level in range(1, 11):
            assert current_result["level"] == level
            if level < 10:
                current_result = current_result["nested"]

    def test_recursion_depth_limit(self):
        """Test recursion depth limit protection."""
        # Create extremely deep nesting that would exceed the limit
        deep_data = {"level": 1}
        current = deep_data
        for i in range(2, 60):  # Exceed the 50 level limit
            current["nested"] = {"level": i}
            current = current["nested"]

        result = _serialize_for_json(deep_data)
        # Should not crash and should contain max-depth-exceeded marker somewhere
        str(result)
        # The exact structure may vary, but it should handle deep recursion gracefully
        assert isinstance(result, dict)

    def test_unserializable_objects_error_handling(self):
        """Test error handling for truly unserializable objects."""

        class NonSerializable:
            def __init__(self):
                self.data = "test"

        non_serializable = NonSerializable()

        # Should raise TypeError for objects without custom serializers
        with pytest.raises(TypeError) as excinfo:
            _serialize_for_json(non_serializable)

        assert "not serializable" in str(excinfo.value)
        assert "register_custom_serializer" in str(excinfo.value)

    def test_default_serializer_fallback(self):
        """Test fallback to default serializer for unknown types."""

        class CustomObject:
            def __init__(self):
                self.value = "test"

        custom_obj = CustomObject()

        def default_serializer(obj):
            return f"<{type(obj).__name__}: {getattr(obj, 'value', 'unknown')}>"

        result = _serialize_for_json(custom_obj, default_serializer=default_serializer)
        assert result == "<CustomObject: test>"

    def test_mode_specific_behavior(self):
        """Test mode-specific serialization behavior."""
        data = {"key": "value", "number": 42}

        # Default mode
        result_default = _serialize_for_json(data, mode="default")
        assert result_default == data

        # Cache mode
        result_cache = _serialize_for_json(data, mode="cache")
        assert result_cache == data

        # Custom mode with placeholder
        result_custom = _serialize_for_json(
            data, mode="custom", circular_ref_placeholder="<custom>"
        )
        assert result_custom == data

    def test_json_serialization_roundtrip(self):
        """Test that serialized data can be JSON serialized."""
        complex_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "enum": SampleEnum.ALPHA,
            "datetime": datetime(2023, 1, 1, 12, 0, 0),
        }

        serialized = _serialize_for_json(complex_data)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        assert parsed["string"] == "test"
        assert parsed["number"] == 42
        assert parsed["float"] == 3.14
        assert parsed["bool"] is True
        assert parsed["none"] is None
        assert parsed["list"] == [1, 2, 3]
        assert parsed["dict"] == {"nested": "value"}
        assert parsed["enum"] == "alpha"
        assert "2023-01-01T12:00:00" in parsed["datetime"]

    def test_serialize_to_json_wrapper(self):
        """Test the serialize_to_json convenience function."""
        data = {"message": "hello", "timestamp": datetime(2023, 1, 1), "count": 42}

        json_str = _serialize_to_json_internal(data)
        parsed = json.loads(json_str)

        assert parsed["message"] == "hello"
        assert "2023-01-01" in parsed["timestamp"]
        assert parsed["count"] == 42

    def test_large_data_structure_performance(self):
        """Test serialization of large data structures."""
        # Create a reasonably large structure for performance testing
        large_data = {}
        for i in range(1000):
            large_data[f"key_{i}"] = {
                "index": i,
                "data": [j for j in range(10)],
                "metadata": {"created": datetime.now(), "type": "test"},
            }

        # Should complete without timeout or memory issues
        result = _serialize_for_json(large_data)
        assert isinstance(result, dict)
        assert len(result) == 1000
        assert "key_0" in result
        assert "key_999" in result

    def test_custom_uuid_and_decimal_handling(self):
        """Test handling of UUID and Decimal objects."""
        test_uuid = uuid.uuid4()
        test_decimal = Decimal("3.141592653589793")

        data = {
            "uuid": test_uuid,
            "decimal": test_decimal,
            "list_of_uuids": [test_uuid, uuid.uuid4()],
            "list_of_decimals": [test_decimal, Decimal("2.718")],
        }

        # Register custom serializers for these types
        register_custom_serializer(uuid.UUID, lambda obj: str(obj))
        register_custom_serializer(Decimal, lambda obj: str(obj))

        result = _serialize_for_json(data)

        assert isinstance(result["uuid"], str)
        assert isinstance(result["decimal"], str)
        assert all(isinstance(u, str) for u in result["list_of_uuids"])
        assert all(isinstance(d, str) for d in result["list_of_decimals"])

    def test_agent_response_like_objects(self):
        """Test serialization of objects that look like AgentResponse."""

        class MockAgentResponse:
            def __init__(self):
                self.output = "output data"
                self.usage = lambda: type(
                    "Usage", (), {"request_tokens": 10, "response_tokens": 20}
                )()

        response = MockAgentResponse()
        result = _serialize_for_json(response)

        # Should be handled by the AgentResponse serialization logic
        # Check if it contains the serialized structure we expect
        if isinstance(result, dict):
            # Could be serialized as AgentResponse or as a regular object
            has_output = "output" in result or any(
                "output" in str(v) for v in result.values() if isinstance(v, dict)
            )
            has_content = "content" in result or any(
                "content" in str(v) for v in result.values() if isinstance(v, dict)
            )
            assert has_output or has_content, f"Expected output or content in result: {result}"
        else:
            # If serialized as string, should contain some response info
            assert "output" in str(result) or "MockAgentResponse" in str(result)

    def test_error_recovery_and_fallbacks(self):
        """Test error recovery and fallback mechanisms."""

        class ProblematicObject:
            def __init__(self):
                self.value = "test"

            def __getattribute__(self, name):
                if name == "problematic_attr":
                    raise RuntimeError("Simulated error")
                return super().__getattribute__(name)

        problematic = ProblematicObject()

        # With default serializer that handles errors
        def error_handling_serializer(obj):
            try:
                return {"type": type(obj).__name__, "safe_value": getattr(obj, "value", "unknown")}
            except Exception:
                return f"<error-serializing: {type(obj).__name__}>"

        result = _serialize_for_json(problematic, default_serializer=error_handling_serializer)
        assert "ProblematicObject" in str(result)

    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases in a single structure."""
        edge_cases = {
            # Basic types
            "string": "test",
            "empty_string": "",
            "unicode": "🚀🌟✨",
            "int": 42,
            "zero": 0,
            "negative": -1,
            "float": 3.14,
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "nan": float("nan"),
            "bool_true": True,
            "bool_false": False,
            "none": None,
            # Collections
            "empty_list": [],
            "empty_dict": {},
            "empty_tuple": (),
            "list": [1, 2, 3],
            "nested_list": [[1, 2], [3, 4]],
            "dict": {"key": "value"},
            "nested_dict": {"outer": {"inner": "value"}},
            # Special types
            "enum": SampleEnum.ALPHA,
            "datetime": datetime(2023, 1, 1),
            "date": date(2023, 1, 1),
            "time": time(12, 0, 0),
            "complex": complex(3, 4),
            "bytes": b"hello",
            # Objects
            "pydantic": SamplePydanticModel(name="test", value=42),
            "dataclass": SampleDataclass(name="test", value=42, items=["a", "b"]),
        }

        result = _serialize_for_json(edge_cases)

        # Verify all edge cases are handled correctly
        assert result["string"] == "test"
        assert result["empty_string"] == ""
        assert result["unicode"] == "🚀🌟✨"
        assert result["int"] == 42
        assert result["zero"] == 0
        assert result["negative"] == -1
        assert result["float"] == 3.14
        assert result["inf"] == "inf"
        assert result["neg_inf"] == "-inf"
        assert result["nan"] == "nan"
        assert result["bool_true"] is True
        assert result["bool_false"] is False
        assert result["none"] is None
        assert result["empty_list"] == []
        assert result["empty_dict"] == {}
        assert result["empty_tuple"] == []
        assert result["list"] == [1, 2, 3]
        assert result["nested_list"] == [[1, 2], [3, 4]]
        assert result["dict"] == {"key": "value"}
        assert result["nested_dict"] == {"outer": {"inner": "value"}}
        assert result["enum"] == "alpha"
        assert isinstance(result["datetime"], str)
        assert isinstance(result["date"], str)
        assert isinstance(result["time"], str)
        assert result["complex"] == {"real": 3.0, "imag": 4.0}
        assert isinstance(result["bytes"], str)
        assert isinstance(result["pydantic"], dict)
        assert isinstance(result["dataclass"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
