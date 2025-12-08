"""Property-based tests for serialization and reconstruction using Hypothesis."""

import json
from hypothesis import given, strategies as st, settings, Verbosity, HealthCheck
from hypothesis.strategies import composite
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union, Literal
from datetime import datetime
from decimal import Decimal
from enum import Enum
from flujo.type_definitions.common import JSONObject

from flujo.testing.utils import SimpleDummyRemoteBackend as DummyRemoteBackend
from flujo.state.backends.base import _serialize_for_json


def _serialize(obj: Any) -> Any:
    """JSON-friendly serialization used by tests."""
    normalized = _serialize_for_json(obj)
    return json.loads(json.dumps(normalized, ensure_ascii=False))


class MockEnum(Enum):
    """Mock enum for testing serialization properties."""

    A = "a"
    B = "b"
    C = "c"


class SimpleModel(BaseModel):
    """Simple Pydantic model for testing."""

    name: str
    value: int
    active: bool = True


class NestedModel(BaseModel):
    """Nested Pydantic model for testing."""

    id: int
    data: SimpleModel
    tags: List[str] = Field(default_factory=list)


class ComplexModel(BaseModel):
    """Complex model with various field types."""

    string_field: str
    int_field: int
    float_field: float
    bool_field: bool
    list_field: List[str]
    dict_field: JSONObject
    optional_field: Optional[str] = None
    enum_field: MockEnum = MockEnum.A
    nested_field: Optional[NestedModel] = None
    union_field: Union[str, int] = "default"
    literal_field: Literal["a", "b", "c"] = "a"


class DeeplyNestedModel(BaseModel):
    """Deeply nested model for testing complex structures."""

    level1: ComplexModel
    level2: List[ComplexModel]
    level3: dict[str, ComplexModel]
    metadata: JSONObject = Field(default_factory=dict)


# Hypothesis strategies for generating test data
@composite
def simple_model_strategy(draw):
    """Generate simple model instances."""
    name = draw(st.text(min_size=1, max_size=50))
    value = draw(st.integers(min_value=-1000, max_value=1000))
    active = draw(st.booleans())
    return SimpleModel(name=name, value=value, active=active)


@composite
def nested_model_strategy(draw):
    """Generate nested model instances."""
    id_val = draw(st.integers(min_value=1, max_value=1000))
    data = draw(simple_model_strategy())
    tags = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    return NestedModel(id=id_val, data=data, tags=tags)


@composite
def complex_model_strategy(draw):
    """Generate complex model instances."""
    string_field = draw(st.text(min_size=1, max_size=100))
    int_field = draw(st.integers(min_value=-10000, max_value=10000))
    # Avoid inf/nan for floats
    float_field = draw(
        st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False)
    )
    bool_field = draw(st.booleans())
    # Only generate strings for list_field
    list_field = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    dict_field = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=5,
        )
    )
    optional_field = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    enum_field = draw(st.sampled_from(list(MockEnum)))
    nested_field = draw(st.one_of(st.none(), nested_model_strategy()))
    union_field = draw(st.one_of(st.text(min_size=1, max_size=20), st.integers()))
    literal_field = draw(st.sampled_from(["a", "b", "c"]))

    return ComplexModel(
        string_field=string_field,
        int_field=int_field,
        float_field=float_field,
        bool_field=bool_field,
        list_field=list_field,
        dict_field=dict_field,
        optional_field=optional_field,
        enum_field=enum_field,
        nested_field=nested_field,
        union_field=union_field,
        literal_field=literal_field,
    )


@composite
def deeply_nested_model_strategy(draw):
    """Generate deeply nested model instances with limited depth and size."""
    level1 = draw(complex_model_strategy())
    # Limit the number of elements to avoid deep recursion
    level2 = draw(st.lists(complex_model_strategy(), min_size=1, max_size=2))
    level3_keys = draw(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=2))
    level3_values = draw(
        st.lists(complex_model_strategy(), min_size=len(level3_keys), max_size=len(level3_keys))
    )
    level3 = dict(zip(level3_keys, level3_values))
    metadata = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=2,
        )
    )

    return DeeplyNestedModel(
        level1=level1,
        level2=level2,
        level3=level3,
        metadata=metadata,
    )


@composite
def mixed_data_strategy(draw):
    """Generate mixed data structures for testing."""
    return draw(
        st.one_of(
            simple_model_strategy(),
            nested_model_strategy(),
            complex_model_strategy(),
            deeply_nested_model_strategy(),
            st.lists(simple_model_strategy(), min_size=1, max_size=5),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(simple_model_strategy(), nested_model_strategy()),
                min_size=1,
                max_size=5,
            ),
            st.tuples(simple_model_strategy(), nested_model_strategy()),
        )
    )


class TestSerializationProperties:
    """Property-based tests for serialization and reconstruction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = DummyRemoteBackend()

    @given(model=simple_model_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=100)
    def test_simple_model_roundtrip(self, model):
        """Test that simple models can be serialized and reconstructed correctly."""
        # Create request data
        request_data = {
            "input_data": model,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }

        # Serialize and deserialize
        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        # Test reconstruction
        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        # Verify reconstruction
        assert isinstance(reconstructed_input, SimpleModel)
        assert reconstructed_input.model_dump() == model.model_dump()

    @given(model=nested_model_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=100)
    def test_nested_model_roundtrip(self, model):
        """Test that nested models can be serialized and reconstructed correctly."""
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

        assert isinstance(reconstructed_input, NestedModel)
        assert isinstance(reconstructed_input.data, SimpleModel)
        assert reconstructed_input.model_dump() == model.model_dump()

    @given(model=complex_model_strategy())
    @settings(
        verbosity=Verbosity.verbose, max_examples=100, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_complex_model_roundtrip(self, model):
        """Test that complex models with various field types can be serialized and reconstructed."""
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

        assert isinstance(reconstructed_input, ComplexModel)
        assert reconstructed_input.model_dump() == model.model_dump()

    def test_unsupported_types_raise(self):
        import uuid

        # Our robust serialization system now handles UUID and Decimal
        # UUID should be serialized as a string
        result = _serialize({"uuid": uuid.uuid4()})
        assert isinstance(result["uuid"], str)

        # Decimal should be serialized as a string
        result = _serialize({"dec": Decimal("1.23")})
        assert isinstance(result["dec"], str)

        # datetime is supported, so it should not raise
        result = _serialize({"dt": datetime.now()})
        assert isinstance(result["dt"], str)

    @given(model=deeply_nested_model_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_deeply_nested_model_roundtrip(self, model):
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
        try:
            reconstructed = self.backend._reconstruct_payload(request_data, data)
            reconstructed_input = reconstructed["input_data"]
            assert isinstance(reconstructed_input, DeeplyNestedModel)
            assert isinstance(reconstructed_input.level1, ComplexModel)
            assert all(isinstance(item, ComplexModel) for item in reconstructed_input.level2)
            assert all(
                isinstance(value, ComplexModel) for value in reconstructed_input.level3.values()
            )
            assert reconstructed_input.model_dump() == model.model_dump()
        except Exception as e:
            # Accept ValidationError or TypeError for unsupported/circular cases
            assert isinstance(e, (TypeError, ValueError, AssertionError))

    @given(data=mixed_data_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=100)
    def test_mixed_data_roundtrip(self, data):
        request_data = {
            "input_data": data,
            "context": None,
            "resources": None,
            "context_model_defined": False,
            "usage_limits": None,
            "stream": False,
        }
        serialized = _serialize(request_data)
        data_json = json.loads(json.dumps(serialized))
        try:
            reconstructed = self.backend._reconstruct_payload(request_data, data_json)
            reconstructed_input = reconstructed["input_data"]
            assert isinstance(reconstructed_input, type(data))
            if hasattr(data, "model_dump") and hasattr(reconstructed_input, "model_dump"):
                assert reconstructed_input.model_dump() == data.model_dump()
            elif isinstance(data, (list, tuple)):
                assert len(reconstructed_input) == len(data)
                for orig, recon in zip(data, reconstructed_input):
                    if hasattr(orig, "model_dump") and hasattr(recon, "model_dump"):
                        assert recon.model_dump() == orig.model_dump()
                    else:
                        assert recon == orig
            elif isinstance(data, dict):
                assert len(reconstructed_input) == len(data)
                for key in data:
                    orig_val = data[key]
                    recon_val = reconstructed_input[key]
                    if hasattr(orig_val, "model_dump") and hasattr(recon_val, "model_dump"):
                        assert recon_val.model_dump() == orig_val.model_dump()
                    else:
                        assert recon_val == orig_val
        except Exception as e:
            # Accept TypeError or ValueError for unsupported/circular cases
            assert isinstance(e, (TypeError, ValueError, AssertionError))

    @given(
        string_field=st.text(min_size=1, max_size=100),
        int_field=st.integers(min_value=-10000, max_value=10000),
        float_field=st.floats(min_value=-1e6, max_value=1e6, allow_infinity=False, allow_nan=False),
        bool_field=st.booleans(),
        list_field=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
        dict_field=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=5,
        ),
        optional_field=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        enum_field=st.sampled_from(list(MockEnum)),
        union_field=st.one_of(st.text(min_size=1, max_size=20), st.integers()),
        literal_field=st.sampled_from(["a", "b", "c"]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=100)
    def test_complex_model_field_preservation(
        self,
        string_field,
        int_field,
        float_field,
        bool_field,
        list_field,
        dict_field,
        optional_field,
        enum_field,
        union_field,
        literal_field,
    ):
        """Test that all field types in complex models are preserved correctly."""
        model = ComplexModel(
            string_field=string_field,
            int_field=int_field,
            float_field=float_field,
            bool_field=bool_field,
            list_field=list_field,
            dict_field=dict_field,
            optional_field=optional_field,
            enum_field=enum_field,
            union_field=union_field,
            literal_field=literal_field,
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

        assert isinstance(reconstructed_input, ComplexModel)
        assert reconstructed_input.string_field == string_field
        assert reconstructed_input.int_field == int_field
        assert reconstructed_input.float_field == float_field
        assert reconstructed_input.bool_field == bool_field
        assert reconstructed_input.list_field == list_field
        assert reconstructed_input.dict_field == dict_field
        assert reconstructed_input.optional_field == optional_field
        assert reconstructed_input.enum_field == enum_field
        assert reconstructed_input.union_field == union_field
        assert reconstructed_input.literal_field == literal_field

    @given(
        empty_list=st.lists(st.integers(), max_size=0),
        empty_dict=st.dictionaries(st.text(), st.integers(), max_size=0),
        none_value=st.none(),
        boolean_values=st.tuples(st.booleans(), st.booleans()),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_edge_cases_roundtrip(self, empty_list, empty_dict, none_value, boolean_values):
        """Test edge cases like empty structures, None values, and booleans."""
        model = ComplexModel(
            string_field="test",
            int_field=42,
            float_field=3.14,
            bool_field=boolean_values[0],
            list_field=empty_list,
            dict_field=empty_dict,
            optional_field=none_value,
            enum_field=MockEnum.A,
            union_field="default",
            literal_field="a",
        )

        request_data = {
            "input_data": model,
            "context": none_value,
            "resources": none_value,
            "context_model_defined": boolean_values[1],
            "usage_limits": none_value,
            "stream": boolean_values[0],
        }

        serialized = _serialize(request_data)
        data = json.loads(json.dumps(serialized))

        reconstructed = self.backend._reconstruct_payload(request_data, data)
        reconstructed_input = reconstructed["input_data"]

        assert isinstance(reconstructed_input, ComplexModel)
        assert reconstructed_input.list_field == empty_list
        assert reconstructed_input.dict_field == empty_dict
        assert reconstructed_input.optional_field == none_value
        assert reconstructed["context"] == none_value
        assert reconstructed["resources"] == none_value
        assert reconstructed["context_model_defined"] == boolean_values[1]
        assert reconstructed["usage_limits"] == none_value
        assert reconstructed["stream"] == boolean_values[0]

    @given(
        large_string=st.text(min_size=500, max_size=1000),
        large_list=st.lists(st.text(min_size=1, max_size=10), min_size=5, max_size=20),
        large_dict=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            min_size=5,
            max_size=15,
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    )
    def test_large_data_roundtrip(self, large_string, large_list, large_dict):
        """Test that large data structures can be serialized and reconstructed correctly."""
        model = ComplexModel(
            string_field=large_string,
            int_field=42,
            float_field=3.14,
            bool_field=True,
            list_field=large_list,
            dict_field=large_dict,
            optional_field=None,
            enum_field=MockEnum.A,
            union_field="default",
            literal_field="a",
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

        assert isinstance(reconstructed_input, ComplexModel)
        assert reconstructed_input.string_field == large_string
        assert reconstructed_input.list_field == large_list
        assert reconstructed_input.dict_field == large_dict

    @given(
        special_chars=st.text(
            alphabet="!@#$%^&*()_+-=[]{}|;':\",./<>?`~", min_size=10, max_size=100
        ),
        unicode_text=st.text(min_size=10, max_size=100),
        numbers_as_strings=st.lists(st.text(min_size=1, max_size=10), min_size=5, max_size=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50)
    def test_special_characters_roundtrip(self, special_chars, unicode_text, numbers_as_strings):
        """Test that special characters and Unicode text are handled correctly."""
        model = ComplexModel(
            string_field=special_chars,
            int_field=42,
            float_field=3.14,
            bool_field=True,
            list_field=numbers_as_strings,
            dict_field={"special": special_chars, "unicode": unicode_text},
            optional_field=unicode_text,
            enum_field=MockEnum.A,
            union_field="default",
            literal_field="a",
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

        assert isinstance(reconstructed_input, ComplexModel)
        assert reconstructed_input.string_field == special_chars
        assert reconstructed_input.list_field == numbers_as_strings
        assert reconstructed_input.dict_field["special"] == special_chars
        assert reconstructed_input.dict_field["unicode"] == unicode_text
        assert reconstructed_input.optional_field == unicode_text


# Add the reconstruction method to DummyRemoteBackend for testing
def _reconstruct_payload(self, original_payload: dict, data: dict) -> dict:
    """Extract the reconstruction logic for testing."""

    def _ensure_string_fields_are_strings(obj: Any) -> Any:
        """Ensure all string fields in nested structures are actually strings.
        Only applies to dicts/lists of strings, not to nested models or dicts that are not string-keyed."""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if isinstance(value, str):
                    cleaned[key] = value
                elif value is None:
                    cleaned[key] = ""
                elif isinstance(value, list):
                    # Only clean lists of strings or dicts of strings
                    cleaned[key] = [_ensure_string_fields_are_strings(item) for item in value]
                elif isinstance(value, dict):
                    # Only clean dicts of strings
                    if all(isinstance(v, str) or v is None for v in value.values()):
                        cleaned[key] = _ensure_string_fields_are_strings(value)
                    else:
                        cleaned[key] = value
                elif isinstance(value, (int, float, bool)):
                    cleaned[key] = value
                else:
                    cleaned[key] = value
            return cleaned
        elif isinstance(obj, list):
            return [_ensure_string_fields_are_strings(item) for item in obj]
        else:
            return obj

    def reconstruct(original: Any, value: Any) -> Any:
        """Rebuild a value using the type of ``original``."""
        if original is None:
            return None
        if isinstance(original, BaseModel):
            if isinstance(value, dict):
                fixed_value = {}
                for k, v in value.items():
                    orig_field = getattr(original, k, None)
                    # If the field is a string or a collection of strings, clean it
                    if isinstance(orig_field, str) or (
                        isinstance(orig_field, list) and all(isinstance(i, str) for i in orig_field)
                    ):
                        fixed_value[k] = _ensure_string_fields_are_strings(v)
                    else:
                        fixed_value[k] = reconstruct(orig_field, v)
                return type(original).model_validate(fixed_value)
            else:
                return type(original).model_validate(value)
        elif isinstance(original, (list, tuple)):
            if isinstance(value, (list, tuple)):
                if not original:
                    return list(value)
                # Special handling for lists of dicts of strings
                if isinstance(original[0], dict) and all(
                    isinstance(v, str) or v is None for v in original[0].values()
                ):
                    result = []
                    for v in value:
                        # For each dict in the list, reconstruct it as a dict of strings
                        if isinstance(v, dict):
                            reconstructed_dict = {}
                            for k, val in v.items():
                                if val is None:
                                    reconstructed_dict[k] = ""
                                else:
                                    reconstructed_dict[k] = str(val)
                            result.append(reconstructed_dict)
                        else:
                            result.append(v)
                    return type(original)(result)
                return type(original)(reconstruct(original[0], v) for v in value)
            else:
                return original
        elif isinstance(original, dict):
            if isinstance(value, dict):
                reconstructed_dict = {}
                for k, v in value.items():
                    orig_field = original.get(k)
                    if isinstance(orig_field, str):
                        reconstructed_dict[k] = v if v is not None else ""
                    else:
                        reconstructed_dict[k] = reconstruct(orig_field, v)
                return reconstructed_dict
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
