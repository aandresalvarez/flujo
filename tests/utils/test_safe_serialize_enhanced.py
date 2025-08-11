"""Comprehensive tests for the enhanced safe_serialize function.

This module tests the consolidated serialization logic that now handles
all edge cases previously handled by BaseModel.model_dump.
"""

import pytest
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from flujo.utils.serialization import (
    safe_serialize,
    register_custom_serializer,
    reset_custom_serializer_registry,
)
from flujo.domain.base_model import BaseModel


class SampleEnum(Enum):
    """Sample enum for serialization."""

    VALUE_A = "value_a"
    VALUE_B = "value_b"
    VALUE_C = 42


@dataclass
class SampleDataclass:
    """Sample dataclass for serialization."""

    name: str
    value: int
    optional_field: Optional[str] = None


class SampleCustomObject:
    """Sample custom object without serialization support."""

    def __init__(self, data: str):
        self.data = data

    def __repr__(self):
        return f"SampleCustomObject(data={self.data!r})"


class SampleBaseModel(BaseModel):
    """Sample BaseModel for circular reference testing."""

    name: str
    value: int
    ref: Optional["SampleBaseModel"] = None
    items: List[str] = []
    metadata: Dict[str, Any] = {}


class CallableObject:
    """Callable object for testing."""

    def __init__(self, name: str):
        self.__name__ = name

    def __call__(self):
        return "called"


class SampleTestObject:
    """Sample object for testing fallback behavior."""

    def __init__(self, attr_value: str):
        self.attr_value = attr_value
        self._private = "private"


class TestSafeSerializeEnhanced:
    """Test the enhanced safe_serialize function with comprehensive edge cases."""

    def setup_method(self):
        """Reset custom serializers before each test."""
        reset_custom_serializer_registry()

    def teardown_method(self):
        """Reset custom serializers after each test."""
        reset_custom_serializer_registry()

    def test_primitive_types(self):
        """Test serialization of primitive types."""
        # Basic primitives
        assert safe_serialize(None) is None
        assert safe_serialize("hello") == "hello"
        assert safe_serialize(42) == 42
        assert safe_serialize(3.14) == 3.14
        assert safe_serialize(True) is True
        assert safe_serialize(False) is False

        # Empty string
        assert safe_serialize("") == ""

        # Unicode strings
        assert safe_serialize("🚀🌟✨") == "🚀🌟✨"

    def test_special_float_values(self):
        """Test serialization of special float values."""
        assert safe_serialize(float("inf")) == "inf"
        assert safe_serialize(float("-inf")) == "-inf"
        assert safe_serialize(float("nan")) == "nan"
        assert safe_serialize(0.0) == 0.0
        assert safe_serialize(-0.0) == -0.0

    def test_datetime_objects(self):
        """Test serialization of datetime objects."""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        result = safe_serialize(dt)
        assert isinstance(result, str)
        assert "2023-12-25T15:30:45" in result

        d = date(2023, 12, 25)
        result = safe_serialize(d)
        assert isinstance(result, str)
        assert "2023-12-25" in result

        t = time(15, 30, 45)
        result = safe_serialize(t)
        assert isinstance(result, str)
        assert "15:30:45" in result

    def test_enum_objects(self):
        """Test serialization of enum objects."""
        assert safe_serialize(SampleEnum.VALUE_A) == "value_a"
        assert safe_serialize(SampleEnum.VALUE_B) == "value_b"
        assert safe_serialize(SampleEnum.VALUE_C) == 42

    def test_complex_numbers(self):
        """Test serialization of complex numbers."""
        c = complex(3, 4)
        result = safe_serialize(c)
        assert isinstance(result, dict)
        assert result["real"] == 3.0
        assert result["imag"] == 4.0

    def test_bytes_and_memoryview(self):
        """Test serialization of bytes and memoryview objects."""
        # Test bytes
        data = b"hello world"
        result = safe_serialize(data)
        assert isinstance(result, str)
        # Should be base64 encoded
        import base64

        assert base64.b64decode(result.encode("ascii")) == data

        # Test memoryview
        mv = memoryview(data)
        result = safe_serialize(mv)
        assert isinstance(result, str)
        assert base64.b64decode(result.encode("ascii")) == data

    def test_callable_objects(self):
        """Test serialization of callable objects."""

        def test_func():
            return "test"

        # Function with __name__
        result = safe_serialize(test_func)
        assert result == "test_func"

        # Callable object with __name__
        callable_obj = CallableObject("test_callable")
        result = safe_serialize(callable_obj)
        assert result == "test_callable"

        # Lambda (no __name__)
        def lambda_func(x):
            return x

        result = safe_serialize(lambda_func)
        assert isinstance(result, str)
        assert "lambda" in result.lower() or "function" in result.lower()

    def test_collections(self):
        """Test serialization of collections."""
        # Lists
        assert safe_serialize([1, 2, 3]) == [1, 2, 3]
        assert safe_serialize([]) == []

        # Tuples (converted to lists)
        assert safe_serialize((1, 2, 3)) == [1, 2, 3]
        assert safe_serialize(()) == []

        # Sets (converted to lists)
        result = safe_serialize({1, 2, 3})
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Frozensets (converted to lists)
        result = safe_serialize(frozenset([1, 2, 3]))
        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

        # Dictionaries
        test_dict = {"a": 1, "b": 2}
        assert safe_serialize(test_dict) == test_dict

    def test_nested_collections(self):
        """Test serialization of nested collections."""
        nested = {
            "list": [1, 2, {"nested": "dict"}],
            "tuple": (3, 4, ["nested", "list"]),
            "dict": {"key": "value", "nested": {"deep": "value"}},
            "set": {5, 6, 7},
        }

        result = safe_serialize(nested)
        assert isinstance(result, dict)
        assert result["list"] == [1, 2, {"nested": "dict"}]
        assert result["tuple"] == [3, 4, ["nested", "list"]]
        assert result["dict"] == {"key": "value", "nested": {"deep": "value"}}
        assert isinstance(result["set"], list)
        assert set(result["set"]) == {5, 6, 7}

    def test_dataclass_objects(self):
        """Test serialization of dataclass objects."""
        dc = SampleDataclass("test", 42, "optional")
        result = safe_serialize(dc)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["optional_field"] == "optional"

    def test_basemodel_objects(self):
        """Test serialization of Flujo BaseModel objects."""
        model = SampleBaseModel(
            name="test_model", value=42, items=["item1", "item2"], metadata={"key": "value"}
        )

        result = safe_serialize(model)
        assert isinstance(result, dict)
        assert result["name"] == "test_model"
        assert result["value"] == 42
        assert result["ref"] is None
        assert result["items"] == ["item1", "item2"]
        assert result["metadata"] == {"key": "value"}

    def test_circular_references_default_mode(self):
        """Test circular reference handling in default mode."""
        # Create circular reference with BaseModel
        model1 = SampleBaseModel(name="model1", value=1)
        model2 = SampleBaseModel(name="model2", value=2)
        model1.ref = model2
        model2.ref = model1

        result = safe_serialize(model1, mode="default")
        assert isinstance(result, dict)
        assert result["name"] == "model1"
        assert result["value"] == 1
        assert result["ref"]["name"] == "model2"
        assert result["ref"]["value"] == 2
        # Circular reference should be None in default mode
        assert result["ref"]["ref"] is None

        # Test with dictionary circular reference
        dict1 = {"name": "dict1"}
        dict2 = {"name": "dict2"}
        dict1["ref"] = dict2
        dict2["ref"] = dict1

        result = safe_serialize(dict1, mode="default")
        assert result["name"] == "dict1"
        assert result["ref"]["name"] == "dict2"
        # Circular reference should be empty dict in default mode
        assert result["ref"]["ref"] == {}

        # Test with list circular reference
        list1 = ["item1"]
        list2 = ["item2"]
        list1.append(list2)
        list2.append(list1)

        result = safe_serialize(list1, mode="default")
        assert result[0] == "item1"
        assert result[1][0] == "item2"
        # Circular reference should be empty list in default mode
        assert result[1][1] == []

    def test_circular_references_cache_mode(self):
        """Test circular reference handling in cache mode."""
        # Create circular reference with BaseModel
        model1 = SampleBaseModel(name="model1", value=1)
        model2 = SampleBaseModel(name="model2", value=2)
        model1.ref = model2
        model2.ref = model1

        result = safe_serialize(model1, mode="cache")
        assert isinstance(result, dict)
        assert result["name"] == "model1"
        assert result["value"] == 1
        assert result["ref"]["name"] == "model2"
        assert result["ref"]["value"] == 2
        # Circular reference should have class name in cache mode
        assert result["ref"]["ref"] == "<SampleBaseModel circular>"

        # Test with dictionary circular reference
        dict1 = {"name": "dict1"}
        dict2 = {"name": "dict2"}
        dict1["ref"] = dict2
        dict2["ref"] = dict1

        result = safe_serialize(dict1, mode="cache")
        assert result["name"] == "dict1"
        assert result["ref"]["name"] == "dict2"
        # Circular reference should have class name in cache mode
        assert result["ref"]["ref"] == "<dict circular>"

    def test_circular_references_custom_placeholder(self):
        """Test circular reference handling with custom placeholder."""
        dict1 = {"name": "dict1"}
        dict2 = {"name": "dict2"}
        dict1["ref"] = dict2
        dict2["ref"] = dict1

        result = safe_serialize(dict1, mode="custom", circular_ref_placeholder="CIRCULAR")
        assert result["name"] == "dict1"
        assert result["ref"]["name"] == "dict2"
        assert result["ref"]["ref"] == "CIRCULAR"

    def test_deep_nesting(self):
        """Test serialization with deep nesting."""
        # Create deeply nested structure
        deep_structure = {"level": 0}
        current = deep_structure

        for i in range(1, 20):  # Create 20 levels deep
            current["nested"] = {"level": i}
            current = current["nested"]

        result = safe_serialize(deep_structure)

        # Verify the structure is preserved
        assert result["level"] == 0
        current_result = result
        for i in range(1, 20):
            assert "nested" in current_result
            assert current_result["nested"]["level"] == i
            current_result = current_result["nested"]

    def test_max_recursion_depth(self):
        """Test that max recursion depth is enforced."""
        # Create a structure that would exceed max depth
        deep_list = []
        current = deep_list

        # Create a structure deeper than the limit (50)
        for i in range(60):
            nested = []
            current.append(nested)
            current = nested

        result = safe_serialize(deep_list)

        # Should not crash and should handle the deep structure
        assert isinstance(result, list)

        # The deepest part should be marked as max depth exceeded
        def find_max_depth_marker(obj, depth=0):
            if isinstance(obj, str) and "max-depth-exceeded" in obj:
                return True
            elif isinstance(obj, list) and obj:
                return find_max_depth_marker(obj[0], depth + 1)
            return False

        assert find_max_depth_marker(result)

    def test_custom_serializer_integration(self):
        """Test integration with custom serializers."""

        def serialize_custom_object(obj: SampleCustomObject) -> Dict[str, Any]:
            return {"type": "SampleCustomObject", "data": obj.data}

        register_custom_serializer(SampleCustomObject, serialize_custom_object)

        obj = SampleCustomObject("test_data")
        result = safe_serialize(obj)

        assert isinstance(result, dict)
        assert result["type"] == "SampleCustomObject"
        assert result["data"] == "test_data"

    def test_custom_serializer_with_circular_refs(self):
        """Test custom serializers with circular references."""

        def serialize_custom_object(obj: SampleCustomObject) -> Dict[str, Any]:
            return {
                "type": "SampleCustomObject",
                "data": obj.data,
                "ref": getattr(obj, "ref", None),
            }

        register_custom_serializer(SampleCustomObject, serialize_custom_object)

        obj1 = SampleCustomObject("obj1")
        obj2 = SampleCustomObject("obj2")
        obj1.ref = obj2
        obj2.ref = obj1

        result = safe_serialize(obj1)

        assert result["type"] == "SampleCustomObject"
        assert result["data"] == "obj1"
        assert result["ref"]["type"] == "SampleCustomObject"
        assert result["ref"]["data"] == "obj2"
        # Circular reference should be handled
        assert result["ref"]["ref"] == "<circular-ref>"

    def test_fallback_to_default_serializer(self):
        """Test fallback to default serializer."""

        def default_serializer(obj: Any) -> str:
            return f"FALLBACK:{type(obj).__name__}"

        obj = SampleCustomObject("test")
        result = safe_serialize(obj, default_serializer=default_serializer)

        assert result == "FALLBACK:SampleCustomObject"

    def test_unserializable_objects_error_handling(self):
        """Test error handling for unserializable objects."""
        obj = SampleCustomObject("test")

        # Should raise TypeError for unserializable objects without fallback
        with pytest.raises(TypeError, match="not serializable"):
            safe_serialize(obj)

    def test_mock_object_detection(self):
        """Test mock object detection and serialization."""
        try:
            from unittest.mock import Mock, MagicMock

            # Test Mock object
            mock_obj = Mock()
            mock_obj.test_attr = "test_value"
            result = safe_serialize(mock_obj)

            assert isinstance(result, dict)
            assert result["type"] == "Mock"
            assert "attributes" in result

            # Test MagicMock object
            magic_mock = MagicMock()
            magic_mock.test_attr = "test_value"
            result = safe_serialize(magic_mock)

            assert isinstance(result, dict)
            assert result["type"] == "Mock"
            assert "attributes" in result

        except ImportError:
            # Skip if unittest.mock is not available
            pytest.skip("unittest.mock not available")

    def test_agent_response_like_objects(self):
        """Test serialization of objects that look like AgentResponse."""

        class MockAgentResponse:
            def __init__(self):
                self.output = "test output"
                self.usage = MockUsage()

        class MockUsage:
            def __init__(self):
                self.request_tokens = 100
                self.response_tokens = 50

            def __call__(self):
                return self

        # Register a custom serializer for the MockAgentResponse
        def serialize_agent_response(obj: MockAgentResponse) -> Dict[str, Any]:
            return {
                "content": obj.output,
                "metadata": {
                    "usage": {
                        "request_tokens": obj.usage.request_tokens,
                        "response_tokens": obj.usage.response_tokens,
                    }
                },
            }

        register_custom_serializer(MockAgentResponse, serialize_agent_response)

        response = MockAgentResponse()
        result = safe_serialize(response)

        assert isinstance(result, dict)
        assert result["content"] == "test output"
        assert "metadata" in result
        assert "usage" in result["metadata"]

    def test_objects_with_cost_and_token_counts(self):
        """Test serialization of objects with cost_usd and token_counts."""

        class MockUsageResponse:
            def __init__(self):
                self.cost_usd = 0.05
                self.token_counts = 150
                self.output = "test output"

        # Register a custom serializer
        def serialize_usage_response(obj: MockUsageResponse) -> Dict[str, Any]:
            return {
                "cost_usd": obj.cost_usd,
                "token_counts": obj.token_counts,
                "output": obj.output,
            }

        register_custom_serializer(MockUsageResponse, serialize_usage_response)

        response = MockUsageResponse()
        result = safe_serialize(response)

        assert isinstance(result, dict)
        assert result["cost_usd"] == 0.05
        assert result["token_counts"] == 150
        assert result["output"] == "test output"

    def test_dict_keys_serialization(self):
        """Test serialization of complex dictionary keys."""
        # Test with various key types
        test_dict = {
            "string_key": "value1",
            42: "value2",
            True: "value3",
            None: "value4",
            (1, 2): "value5",  # Tuple key
        }

        result = safe_serialize(test_dict)

        assert isinstance(result, dict)
        assert result["string_key"] == "value1"
        assert result["42"] == "value2"
        assert result["True"] == "value3"
        assert result["None"] == "value4"
        # Tuple key should be converted to string
        assert "(1, 2)" in result or "[1, 2]" in result

    def test_error_recovery(self):
        """Test error recovery in serialization."""

        class ProblematicObject:
            def __init__(self):
                self.normal_attr = "normal"

            def __getattribute__(self, name):
                if name == "problem_attr":
                    raise RuntimeError("Problematic attribute")
                return super().__getattribute__(name)

        # Register a custom serializer that might fail
        def problematic_serializer(obj: ProblematicObject) -> Dict[str, Any]:
            return {
                "normal": obj.normal_attr,
                "problem": obj.problem_attr,  # This will raise
            }

        register_custom_serializer(ProblematicObject, problematic_serializer)

        obj = ProblematicObject()

        # Should raise TypeError because the object can't be serialized after custom serializer fails
        with pytest.raises(TypeError, match="not serializable"):
            safe_serialize(obj)

        # Test with a default serializer for fallback
        def fallback_serializer(obj: Any) -> str:
            return f"FALLBACK:{type(obj).__name__}"

        result = safe_serialize(obj, default_serializer=fallback_serializer)
        assert result == "FALLBACK:ProblematicObject"

    def test_mode_parameter_propagation(self):
        """Test that mode parameter is properly propagated through nested structures."""
        nested_models = [
            SampleBaseModel(name="model1", value=1),
            SampleBaseModel(name="model2", value=2),
        ]

        # Create circular reference in nested structure
        nested_models[0].ref = nested_models[1]
        nested_models[1].ref = nested_models[0]

        container = {"models": nested_models, "metadata": {"mode_test": True}}

        # Test cache mode propagation
        result = safe_serialize(container, mode="cache")
        assert result["models"][0]["ref"]["ref"] == "<SampleBaseModel circular>"

        # Test default mode propagation
        result = safe_serialize(container, mode="default")
        assert result["models"][0]["ref"]["ref"] is None


class TestSafeSerializeCompatibility:
    """Test compatibility with existing serialization behavior."""

    def test_backwards_compatibility_with_basemodel_dump(self):
        """Test that safe_serialize produces same results as old BaseModel.model_dump."""
        model = SampleBaseModel(name="test", value=42, items=["a", "b"], metadata={"key": "value"})

        # Both should produce the same result
        direct_result = safe_serialize(model)
        model_dump_result = model.model_dump()

        assert direct_result == model_dump_result

    def test_mode_compatibility(self):
        """Test that mode parameter works consistently."""
        model1 = SampleBaseModel(name="model1", value=1)
        model2 = SampleBaseModel(name="model2", value=2)
        model1.ref = model2
        model2.ref = model1

        # Test cache mode
        cache_result = safe_serialize(model1, mode="cache")
        model_cache_result = model1.model_dump(mode="cache")

        assert cache_result == model_cache_result

        # Test default mode
        default_result = safe_serialize(model1, mode="default")
        model_default_result = model1.model_dump(mode="default")

        assert default_result == model_default_result
