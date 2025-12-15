"""Edge case tests for serialization and reconstruction."""

import json
import pytest
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, OrderedDict, Counter
import uuid

from flujo.type_definitions.common import JSONObject
from flujo.testing.utils import SimpleDummyRemoteBackend as DummyRemoteBackend
from flujo.state.backends.base import _serialize_for_json
from flujo.utils.serialization import register_custom_serializer


def _serialize(obj: Any) -> Any:
    """JSON-friendly serialization used by tests (replaces _serialize_for_json)."""
    normalized = _serialize_for_json(obj)
    return json.loads(json.dumps(normalized, ensure_ascii=False))


class MockEnum(Enum):
    """Mock enum for edge case testing."""

    A = "a"
    B = "b"
    C = "c"


class EdgeCaseModel(BaseModel):
    """Model with various edge case field types."""

    # Basic types with edge cases
    empty_string: str = ""
    very_long_string: str = "x" * 10000
    unicode_string: str = "🚀🌟✨🎉🎊"
    special_chars: str = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"

    # Numeric edge cases
    zero_int: int = 0
    negative_int: int = -1
    large_int: int = 2**63 - 1
    small_int: int = -(2**63)
    zero_float: float = 0.0
    negative_float: float = -1.0
    large_float: float = 1e308
    small_float: float = 1e-308
    inf_float: float = float("inf")
    neg_inf_float: float = float("-inf")
    nan_float: float = float("nan")

    # Boolean edge cases
    true_bool: bool = True
    false_bool: bool = False

    # Collection edge cases
    empty_list: List[str] = Field(default_factory=list)
    empty_dict: JSONObject = Field(default_factory=dict)
    single_item_list: List[str] = ["single"]
    large_list: List[int] = Field(default_factory=lambda: list(range(1000)))
    nested_empty_list: List[List[str]] = [[]]

    # Optional fields
    none_string: Optional[str] = None
    none_int: Optional[int] = None
    none_list: Optional[List[str]] = None
    none_dict: Optional[JSONObject] = None

    # Enum fields
    enum_field: MockEnum = MockEnum.A

    # Union fields
    union_string: Union[str, int] = "string"
    union_int: Union[str, int] = 42
    union_none: Optional[Union[str, int]] = None

    # Literal fields
    literal_field: Literal["a", "b", "c"] = "a"

    # Complex nested structures
    nested_dict: Dict[str, JSONObject] = Field(default_factory=dict)
    list_of_dicts: List[JSONObject] = Field(default_factory=list)
    dict_of_lists: Dict[str, List[str]] = Field(default_factory=dict)


register_custom_serializer(EdgeCaseModel, lambda obj: obj.__dict__)


class CircularReferenceModel(BaseModel):
    """Model that could potentially create circular references."""

    name: str
    parent: Optional["CircularReferenceModel"] = None
    children: List["CircularReferenceModel"] = Field(default_factory=list)


class CustomTypesModel(BaseModel):
    """Model with custom types and edge cases."""

    uuid_field: uuid.UUID = Field(default_factory=uuid.uuid4)
    datetime_field: datetime = Field(default_factory=datetime.now)
    date_field: date = Field(default_factory=date.today)
    time_field: time = Field(default_factory=lambda: time(12, 0, 0))
    decimal_field: Decimal = Decimal("3.141592653589793")

    # Collections with custom types
    uuid_list: List[uuid.UUID] = Field(default_factory=list)
    datetime_list: List[datetime] = Field(default_factory=list)
    decimal_list: List[Decimal] = Field(default_factory=list)


class RecursiveModel(BaseModel):
    """Model with recursive structure."""

    value: str
    children: List["RecursiveModel"] = Field(default_factory=list)
    metadata: JSONObject = Field(default_factory=dict)


register_custom_serializer(RecursiveModel, lambda obj: obj.__dict__)


register_custom_serializer(OrderedDict, lambda obj: dict(obj))
register_custom_serializer(Counter, lambda obj: dict(obj))


@dataclass
class DataclassModel:
    """Dataclass for testing non-Pydantic models."""

    name: str
    value: int
    items: List[str]


class TestSerializationEdgeCases:
    """Test edge cases in serialization and reconstruction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = DummyRemoteBackend()

    def test_empty_structures(self):
        """Test serialization of empty structures."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, EdgeCaseModel)
        assert reconstructed_input.empty_list == []
        assert reconstructed_input.empty_dict == {}
        assert reconstructed_input.nested_empty_list == [[]]

    def test_none_values(self):
        """Test serialization of None values."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.none_string is None
        assert reconstructed_input.none_int is None
        assert reconstructed_input.none_list is None
        assert reconstructed_input.none_dict is None
        assert reconstructed_input.union_none is None

    def test_numeric_edge_cases(self):
        """Test serialization of numeric edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.zero_int == 0
        assert reconstructed_input.negative_int == -1
        assert reconstructed_input.large_int == 2**63 - 1
        assert reconstructed_input.small_int == -(2**63)
        assert reconstructed_input.zero_float == 0.0
        assert reconstructed_input.negative_float == -1.0
        assert reconstructed_input.true_bool is True
        assert reconstructed_input.false_bool is False

    def test_string_edge_cases(self):
        """Test serialization of string edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.empty_string == ""
        assert reconstructed_input.very_long_string == "x" * 10000
        assert reconstructed_input.unicode_string == "🚀🌟✨🎉🎊"
        assert reconstructed_input.special_chars == "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"

    def test_collection_edge_cases(self):
        """Test serialization of collection edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.empty_list == []
        assert reconstructed_input.empty_dict == {}
        assert reconstructed_input.single_item_list == ["single"]
        assert len(reconstructed_input.large_list) == 1000
        assert reconstructed_input.nested_empty_list == [[]]

    def test_enum_and_union_edge_cases(self):
        """Test serialization of enum and union edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.enum_field == MockEnum.A
        assert reconstructed_input.union_string == "string"
        assert reconstructed_input.union_int == 42
        assert reconstructed_input.literal_field == "a"

    def test_custom_types_edge_cases(self):
        """Test serialization of custom types."""
        model = CustomTypesModel()
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }
        # Our robust serialization system handles custom types gracefully
        result = _serialize(request_data)
        # Pydantic models preserve their original types, so custom types remain as objects
        # This is actually the correct behavior for Pydantic models
        assert isinstance(result["input_data"]["uuid_field"], str)
        assert isinstance(result["input_data"]["datetime_field"], str)
        assert isinstance(result["input_data"]["date_field"], str)
        assert isinstance(result["input_data"]["time_field"], str)
        assert isinstance(result["input_data"]["decimal_field"], str)

    def test_recursive_structures(self):
        """Test serialization of recursive structures."""
        model = RecursiveModel(
            value="root",
            children=[
                RecursiveModel(value="child1", children=[]),
                RecursiveModel(
                    value="child2", children=[RecursiveModel(value="grandchild", children=[])]
                ),
            ],
            metadata={"depth": 2},
        )

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, RecursiveModel)
        assert reconstructed_input.value == "root"
        assert len(reconstructed_input.children) == 2
        assert reconstructed_input.children[0].value == "child1"
        assert reconstructed_input.children[1].value == "child2"
        assert len(reconstructed_input.children[1].children) == 1
        assert reconstructed_input.children[1].children[0].value == "grandchild"

    def test_dataclass_serialization(self):
        """Test serialization of dataclass objects."""
        dataclass_obj = DataclassModel(name="test", value=42, items=["item1", "item2"])

        request_data = {
            "input_data": dataclass_obj,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        # Dataclasses should be serialized as dicts
        assert isinstance(reconstructed_input, dict)
        assert reconstructed_input["name"] == "test"
        assert reconstructed_input["value"] == 42
        assert reconstructed_input["items"] == ["item1", "item2"]

    def test_collections_edge_cases(self):
        """Test serialization of various collection types."""

        # Test defaultdict
        dd = defaultdict(list)
        dd["key1"].append("value1")
        dd["key2"].append("value2")

        # Test OrderedDict
        od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

        # Test Counter
        counter = Counter(["a", "b", "a", "c", "b", "a"])

        request_data = {
            "input_data": {"defaultdict": dd, "ordereddict": od, "counter": counter},
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        # Collections should be converted to regular dicts
        assert isinstance(reconstructed_input["defaultdict"], dict)
        assert isinstance(reconstructed_input["ordereddict"], dict)
        assert isinstance(reconstructed_input["counter"], dict)

    def test_float_edge_cases(self):
        """Test serialization of float edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert reconstructed_input.large_float == 1e308
        assert reconstructed_input.small_float == 1e-308
        # Note: inf, -inf, and nan are not JSON serializable by default
        # They would be converted to strings or other representations

    def test_complex_nested_structures(self):
        """Test serialization of complex nested structures."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "items": [1, 2, 3, 4, 5],
                        "metadata": {"nested": {"deep": {"value": "very_deep"}}},
                    }
                }
            },
            "lists": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "mixed": {
                "strings": ["a", "b", "c"],
                "numbers": [1, 2, 3],
                "booleans": [True, False, True],
                "nested": [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}],
            },
        }

        request_data = {
            "input_data": complex_data,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, dict)
        assert reconstructed_input["level1"]["level2"]["level3"]["items"] == [1, 2, 3, 4, 5]
        assert (
            reconstructed_input["level1"]["level2"]["level3"]["metadata"]["nested"]["deep"]["value"]
            == "very_deep"
        )
        assert reconstructed_input["lists"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert reconstructed_input["mixed"]["strings"] == ["a", "b", "c"]
        assert reconstructed_input["mixed"]["numbers"] == [1, 2, 3]
        assert reconstructed_input["mixed"]["booleans"] == [True, False, True]

    def test_error_handling(self):
        """Test error handling in serialization."""

        class NonSerializable:
            def __init__(self):
                self.data = "test"

        non_serializable = NonSerializable()
        request_data = {
            "input_data": non_serializable,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }
        with pytest.raises(TypeError):
            _serialize(request_data)

    def test_circular_reference_handling(self):
        """
        Test that the serialization system is robust when encountering circular references
        and test-only types like MockEnum.

        This validates Flujo's production-ready design principle of graceful degradation:
        - The system should not crash or hang on pathological cases
        - The system should handle unsupported structures gracefully
        - Output may contain placeholders or error indicators, but should be a valid string

        This is NOT a test of JSON validity for circular references—production systems
        should avoid circular references, and test-only types are not guaranteed to be
        serializable in all contexts.
        """
        model = EdgeCaseModel()
        model.nested_dict = {"self_ref": {"parent": model}, "normal": {"key": "value"}}
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Test that serialization handles circular references gracefully
        serialized = _serialize(request_data)
        assert serialized is not None, (
            "serialization should not return None for circular references"
        )
        # Ensure placeholder was inserted
        assert serialized["input_data"]["nested_dict"]["self_ref"]["parent"] == "<circular>"

    @pytest.mark.slow
    def test_large_data_structures(self):
        """Test serialization of very large data structures."""
        # Create a large nested structure
        large_data = {}
        for i in range(100):
            large_data[f"level1_{i}"] = {
                f"level2_{j}": {
                    f"level3_{k}": {
                        "items": list(range(100)),
                        "metadata": {f"key_{idx}": f"value_{idx}" for idx in range(50)},
                    }
                    for k in range(10)
                }
                for j in range(10)
            }

        request_data = {
            "input_data": large_data,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, dict)
        assert len(reconstructed_input) == 100
        assert "level1_0" in reconstructed_input
        assert "level1_99" in reconstructed_input

    def test_type_preservation_edge_cases(self):
        """Test that types are preserved correctly in edge cases."""
        model = EdgeCaseModel()

        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        # Verify type preservation
        assert isinstance(reconstructed_input, EdgeCaseModel)
        assert isinstance(reconstructed_input.zero_int, int)
        assert isinstance(reconstructed_input.zero_float, float)
        assert isinstance(reconstructed_input.true_bool, bool)
        assert isinstance(reconstructed_input.empty_string, str)
        assert isinstance(reconstructed_input.empty_list, list)
        assert isinstance(reconstructed_input.empty_dict, dict)
        assert isinstance(reconstructed_input.enum_field, MockEnum)


def test_circular_reference_in_dict_keys():
    """Test that circular/self-referential objects as dict keys do not cause infinite recursion."""
    from flujo.utils.serialization import register_custom_serializer

    class Node:
        def __init__(self, name):
            self.name = name
            self.ref = None

    # Register a custom serializer for Node
    register_custom_serializer(Node, lambda obj: {"name": obj.name, "has_ref": obj.ref is not None})

    node1 = Node("node1")
    node2 = Node("node2")
    node1.ref = node2
    node2.ref = node1

    test_dict = {node1: "a", node2: "b", "plain": "c"}
    result = _serialize(test_dict)
    assert isinstance(result, dict)
    # Should not raise RecursionError or stack overflow
    assert any("node1" in str(k) or "node2" in str(k) for k in result.keys())


# Add the reconstruction method to DummyRemoteBackend for testing
def _reconstruct_payload(self, original_payload: dict, data: dict) -> dict:
    """Extract the reconstruction logic for testing."""

    def reconstruct(original: Any, value: Any) -> Any:
        """Rebuild a value using the type of ``original``."""
        if original is None:
            return None
        if isinstance(original, BaseModel):
            if isinstance(value, dict):
                fixed_value = {
                    k: reconstruct(getattr(original, k, None), v) for k, v in value.items()
                }
                return type(original).model_validate(fixed_value)
            else:
                return type(original).model_validate(value)
        elif isinstance(original, (list, tuple)):
            if isinstance(value, (list, tuple)):
                if not original:
                    return list(value)
                return type(original)(reconstruct(original[0], v) for v in value)
            else:
                return original
        elif isinstance(original, dict):
            if isinstance(value, dict):
                return {k: reconstruct(original.get(k), v) for k, v in value.items()}
            else:
                return original
        else:
            return value

    # Reconstruct the payload with proper types
    reconstructed_payload = {}
    for key, original_value in original_payload.items():
        if key in data:
            reconstructed_payload[key] = reconstruct(original_value, data[key])
        else:
            reconstructed_payload[key] = original_value

    return reconstructed_payload


# Monkey patch the DummyRemoteBackend for testing
DummyRemoteBackend._reconstruct_payload = _reconstruct_payload
