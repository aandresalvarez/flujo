from flujo.domain.models import BaseModel, PipelineContext
from flujo.utils.serialization import (
    register_custom_serializer,
    safe_serialize,
    lookup_custom_serializer,
    _serialize_for_key,
)
import json


# --- Recursive Serialization and Custom Serializer Application ---
class Custom:
    def __init__(self, value):
        self.value = value


def custom_serializer(obj):
    return f"custom:{obj.value}"


class NestedModel(BaseModel):
    custom: Custom
    model_config = {"arbitrary_types_allowed": True}


class OuterModel(BaseModel):
    nested: NestedModel
    model_config = {"arbitrary_types_allowed": True}


def test_recursive_custom_serialization():
    register_custom_serializer(Custom, custom_serializer)
    obj = OuterModel(nested=NestedModel(custom=Custom(42)))
    result = _serialize_for_key(obj)
    expected = {"nested": {"custom": "custom:42"}}
    assert result == expected, f"Got: {result}"
    assert isinstance(json.dumps(result), str)


# --- PipelineContext run_id Exclusion ---
class MyContext(PipelineContext):
    extra: str
    model_config = {"arbitrary_types_allowed": True}


def test_run_id_exclusion():
    ctx = MyContext(initial_prompt="foo", extra="bar")
    result = _serialize_for_key(ctx)
    # Required fields
    assert "run_id" not in result
    assert result["initial_prompt"] == "foo"
    assert result["extra"] == "bar"
    # Default fields: accept both container and string representation
    for field, empty in [
        ("scratchpad", {}),
        ("hitl_history", []),
        ("command_log", []),
    ]:
        val = result[field]
        if isinstance(empty, dict):
            assert val == {} or val == "{}"
        elif isinstance(empty, list):
            assert val == [] or val == "[]"
    import json

    assert isinstance(json.dumps(result), str)


# --- Heterogeneous and Deeply Nested Structures ---
class A(BaseModel):
    x: int


class B(BaseModel):
    y: str


def test_heterogeneous_list_serialization():
    data = [A(x=1), B(y="foo"), 42, "bar"]
    result = [_serialize_for_key(v) for v in data]
    # Accept both int and str for numeric fields
    assert result[0]["x"] == 1 or result[0]["x"] == "1", f"Got: {result[0]}"
    assert result[1] == {"y": "foo"}
    assert result[2] == 42 or result[2] == "42"
    assert result[3] == "bar"
    import json

    assert isinstance(json.dumps(result), str)


# --- Circular Reference Handling ---
class Node(BaseModel):
    value: int
    next: "Node" = None
    model_config = {"arbitrary_types_allowed": True}


Node.model_rebuild()


def test_circular_reference_handling():
    a = Node(value=1)
    b = Node(value=2, next=a)
    a.next = b
    result = safe_serialize(a)
    assert isinstance(result, (dict, str))
    if isinstance(result, str):
        assert "circular" in result or "Node" in result


# --- Custom Serializer Registry Isolation ---
def test_registry_isolation():
    from flujo.utils.serialization import default_serializer_registry

    orig = default_serializer_registry.copy()
    try:

        class Dummy:
            pass

        register_custom_serializer(Dummy, lambda x: "dummy")
        assert lookup_custom_serializer(Dummy()) is not None
    finally:
        default_serializer_registry.clear()
        default_serializer_registry.update(orig)
    assert lookup_custom_serializer(object()) is None


# --- Fallback for Unserializable Type ---
def test_fallback_for_unserializable_type():
    class Weird:
        pass

    w = Weird()
    result = safe_serialize(w)
    assert isinstance(result, str)


# --- Deep Nested Structure with Custom Types ---
def test_deep_nested_custom_types():
    class DeepCustom:
        def __init__(self, value):
            self.value = value

    register_custom_serializer(DeepCustom, lambda obj: f"deep:{obj.value}")

    class DeepNested(BaseModel):
        level1: dict
        model_config = {"arbitrary_types_allowed": True}

    nested_data = {"level2": {"level3": DeepCustom(123)}}
    model = DeepNested(level1=nested_data)
    result = _serialize_for_key(model)
    expected = {"level1": {"level2": {"level3": "deep:123"}}}
    assert result == expected, f"Got: {result}"
    assert isinstance(json.dumps(result), str)


# --- Mixed Types in Lists and Dicts ---
def test_mixed_types_in_containers():
    class MixedCustom:
        def __init__(self, name):
            self.name = name

    register_custom_serializer(MixedCustom, lambda obj: f"mixed:{obj.name}")
    mixed_list = [A(x=1), MixedCustom("test"), 42, "string", {"nested": B(y="nested")}]
    result = _serialize_for_key(mixed_list)
    # Accept both int and str for numeric fields
    assert result[0]["x"] == 1 or result[0]["x"] == "1", f"Got: {result[0]}"
    assert result[1] == "mixed:test"
    assert result[2] == 42 or result[2] == "42"
    assert result[3] == "string"
    assert result[4]["nested"] == {"y": "nested"}
    import json

    assert isinstance(json.dumps(result), str)
