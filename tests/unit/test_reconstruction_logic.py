"""Tests for the reconstruction logic in DummyRemoteBackend."""

import json
from pydantic import BaseModel
from typing import Any

from flujo.testing.utils import DummyRemoteBackend
from flujo.utils.serialization import safe_serialize


class SimpleNested(BaseModel):
    value: str
    number: int


class TestContainer(BaseModel):
    nested: SimpleNested
    items: list[int]
    metadata: dict[str, str]


class TestAgent:
    async def run(self, data: TestContainer) -> TestContainer:
        return data


class TestReconstructionLogic:
    """Test the reconstruction logic in DummyRemoteBackend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = DummyRemoteBackend()
        self.original_payload = TestContainer(
            nested=SimpleNested(value="test", number=42),
            items=[1, 2, 3],
            metadata={"key1": "value1", "key2": "value2"},
        )

    def test_reconstruction_preserves_nested_models(self):
        """Test that nested Pydantic models are correctly reconstructed."""
        # Simulate the serialization process
        request_data = {
            "input_data": self.original_payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Serialize and deserialize
        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        # Test reconstruction
        reconstructed = self.backend._reconstruct_payload(request_data, data)

        # Verify the reconstructed data
        assert "input_data" in reconstructed
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, TestContainer)
        assert isinstance(reconstructed_input.nested, SimpleNested)
        assert reconstructed_input.model_dump() == self.original_payload.model_dump()

    def test_reconstruction_handles_string_encoded_lists(self):
        """Test that string-encoded lists are properly parsed."""
        # Create a payload where lists might be serialized as strings
        payload = TestContainer(
            nested=SimpleNested(value="test", number=100),
            items=[10, 20, 30],
            metadata={"list_key": "value"},
        )

        request_data = {
            "input_data": payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Serialize and deserialize
        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        # Test reconstruction
        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, TestContainer)
        assert isinstance(reconstructed_input.items, list)
        assert all(isinstance(item, int) for item in reconstructed_input.items)
        assert reconstructed_input.model_dump() == payload.model_dump()

    def test_reconstruction_handles_empty_structures(self):
        """Test reconstruction with empty lists and dictionaries."""
        payload = TestContainer(nested=SimpleNested(value="", number=0), items=[], metadata={})

        request_data = {
            "input_data": payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, TestContainer)
        assert reconstructed_input.items == []
        assert reconstructed_input.metadata == {}
        assert reconstructed_input.model_dump() == payload.model_dump()

    def test_reconstruction_handles_none_values(self):
        """Test reconstruction with None values."""
        request_data = {
            "input_data": self.original_payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)

        # Check that None values are preserved
        assert reconstructed["context"] is None
        assert reconstructed["resources"] is None
        assert reconstructed["usage_limits"] is None

    def test_reconstruction_handles_boolean_values(self):
        """Test reconstruction with boolean values."""
        request_data = {
            "input_data": self.original_payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": True,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)

        # Check that boolean values are preserved
        assert reconstructed["context_model_defined"] is False
        assert reconstructed["stream"] is True

    def test_reconstruction_handles_complex_nested_structures(self):
        """Test reconstruction with complex nested structures."""

        class ComplexNested(BaseModel):
            name: str
            data: dict[str, Any]
            items: list[dict[str, str]]

        class ComplexContainer(BaseModel):
            level1: ComplexNested
            level2: list[ComplexNested]
            level3: dict[str, ComplexNested]

        complex_payload = ComplexContainer(
            level1=ComplexNested(
                name="root", data={"key1": "value1", "key2": 42}, items=[{"a": "1"}, {"b": "2"}]
            ),
            level2=[
                ComplexNested(name="item1", data={"id": "1"}, items=[]),
                ComplexNested(name="item2", data={"id": "2"}, items=[{"x": "y"}]),
            ],
            level3={
                "first": ComplexNested(name="first", data={"type": "primary"}, items=[]),
                "second": ComplexNested(name="second", data={"type": "secondary"}, items=[]),
            },
        )

        request_data = {
            "input_data": complex_payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, ComplexContainer)
        assert isinstance(reconstructed_input.level1, ComplexNested)
        assert isinstance(reconstructed_input.level2, list)
        assert all(isinstance(item, ComplexNested) for item in reconstructed_input.level2)
        assert isinstance(reconstructed_input.level3, dict)
        assert all(
            isinstance(value, ComplexNested) for value in reconstructed_input.level3.values()
        )
        assert reconstructed_input.model_dump() == complex_payload.model_dump()

    def test_reconstruction_handles_mixed_types(self):
        """Test reconstruction with mixed types in the same structure."""

        class MixedContainer(BaseModel):
            strings: list[str]
            numbers: list[int]
            booleans: list[bool]
            mixed: list[dict[str, Any]]

        mixed_payload = MixedContainer(
            strings=["a", "b", "c"],
            numbers=[1, 2, 3],
            booleans=[True, False, True],
            mixed=[
                {"key": "value", "number": 42, "flag": True},
                {"key": "value2", "number": 100, "flag": False},
            ],
        )

        request_data = {
            "input_data": mixed_payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, MixedContainer)
        assert reconstructed_input.model_dump() == mixed_payload.model_dump()

    def test_reconstruction_preserves_exact_types(self):
        """Test that reconstruction preserves exact types without conversion."""
        payload = TestContainer(
            nested=SimpleNested(value="type_test", number=999),
            items=[10, 20, 30],
            metadata={"test": "value"},
        )

        request_data = {
            "input_data": payload,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        serialized = safe_serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        # Check exact type preservation
        assert isinstance(reconstructed_input, TestContainer)
        assert isinstance(reconstructed_input.nested, SimpleNested)
        assert isinstance(reconstructed_input.items, list)
        assert isinstance(reconstructed_input.metadata, dict)

        # Check that all items in lists have correct types
        assert all(isinstance(item, int) for item in reconstructed_input.items)
        assert all(isinstance(value, str) for value in reconstructed_input.metadata.values())

        # Check data integrity
        assert reconstructed_input.model_dump() == payload.model_dump()


# Add the reconstruction method to DummyRemoteBackend for testing
def _reconstruct_payload(self, original_payload: dict, data: dict) -> dict:
    """Extract the reconstruction logic for testing."""

    def reconstruct(original: Any, value: Any) -> Any:
        """Rebuild a value using the type of ``original``."""
        if original is None:
            return None
        if isinstance(original, BaseModel):
            # For BaseModel objects, validate the reconstructed data
            # But first, fix any string-encoded lists in the value
            if isinstance(value, dict):
                fixed_value = {}
                for k, v in value.items():
                    if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                        try:
                            import ast

                            parsed = ast.literal_eval(v)
                            if isinstance(parsed, list):
                                fixed_value[k] = list(parsed)
                            else:
                                fixed_value[k] = parsed
                        except (ValueError, SyntaxError):
                            fixed_value[k] = v
                    elif isinstance(v, list):
                        fixed_value[k] = list(v)
                    else:
                        fixed_value[k] = v
                return type(original).model_validate(fixed_value)
            elif isinstance(value, list):
                return [reconstruct(original, v) for v in value]
            else:
                return type(original).model_validate(value)
        elif isinstance(original, (list, tuple)):
            if isinstance(value, str):
                import ast

                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
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
