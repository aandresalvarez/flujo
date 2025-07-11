from flujo.domain.models import BaseModel, PipelineContext
from flujo.utils.serialization import (
    register_custom_serializer,
    safe_serialize,
    lookup_custom_serializer,
    _serialize_for_key,
)


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
    """Test that custom serializers are applied recursively to nested models."""
    register_custom_serializer(Custom, custom_serializer)
    obj = OuterModel(nested=NestedModel(custom=Custom(42)))
    d = _serialize_for_key(obj)
    # Accept both string and dict outputs for custom field
    nested_custom = d["nested"]["custom"]
    assert nested_custom == "custom:42"


# --- PipelineContext run_id Exclusion ---
class MyContext(PipelineContext):
    extra: str
    model_config = {"arbitrary_types_allowed": True}


def test_run_id_exclusion():
    """Test that run_id is excluded from PipelineContext serialization."""
    ctx = MyContext(initial_prompt="foo", extra="bar")
    d = _serialize_for_key(ctx)
    assert "run_id" not in d
    assert d["initial_prompt"] == "foo"
    assert d["extra"] == "bar"


# --- Heterogeneous and Deeply Nested Structures ---
class A(BaseModel):
    x: int


class B(BaseModel):
    y: str


def test_heterogeneous_list_serialization():
    """Test that heterogeneous lists with models and primitives are handled correctly."""
    data = [A(x=1), B(y="foo"), 42, "bar"]
    result = [_serialize_for_key(v) for v in data]
    assert result[0]["x"] in [1, "1"]
    assert result[1]["y"] == "foo"
    assert result[2] in [42, "42"]
    assert result[3] == "bar"


# --- Circular Reference Handling ---
class Node(BaseModel):
    value: int
    next: "Node" = None
    model_config = {"arbitrary_types_allowed": True}


Node.model_rebuild()


def test_circular_reference_handling():
    """Test that circular references are handled gracefully without infinite recursion."""
    a = Node(value=1)
    b = Node(value=2, next=a)
    a.next = b
    d = safe_serialize(a)
    assert isinstance(d, (dict, str))
    if isinstance(d, str):
        assert "circular" in d or "Node" in d


# --- Custom Serializer Registry Isolation ---
def test_registry_isolation():
    """Test that custom serializer registry is properly isolated between tests."""
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
    """Test that unserializable types fall back to safe string representation."""

    class Weird:
        pass

    w = Weird()
    result = safe_serialize(w)
    assert isinstance(result, str)


# --- Deep Nested Structure with Custom Types ---
def test_deep_nested_custom_types():
    """Test that deeply nested structures with custom types are handled correctly."""

    class DeepCustom:
        def __init__(self, value):
            self.value = value

    register_custom_serializer(DeepCustom, lambda obj: f"deep:{obj.value}")

    class DeepNested(BaseModel):
        level1: dict
        model_config = {"arbitrary_types_allowed": True}

    nested_data = {"level1": {"level2": {"level3": DeepCustom(123)}}}

    model = DeepNested(level1=nested_data)
    result = _serialize_for_key(model)
    # Accept both string and dict outputs for deep custom field
    # Traverse to the deepest value
    level3 = result["level1"]["level1"]["level2"]["level3"]
    assert level3 == "deep:123"


# --- Mixed Types in Lists and Dicts ---
def test_mixed_types_in_containers():
    """Test that lists and dicts with mixed types (models, primitives, custom) work correctly."""

    class MixedCustom:
        def __init__(self, name):
            self.name = name

    register_custom_serializer(MixedCustom, lambda obj: f"mixed:{obj.name}")

    mixed_list = [A(x=1), MixedCustom("test"), 42, "string", {"nested": B(y="nested")}]

    result = _serialize_for_key(mixed_list)
    assert isinstance(result, list)
    assert result[0]["x"] in [1, "1"]
    assert result[1] == "mixed:test"
    assert result[2] in [42, "42"]
    assert result[3] == "string"
    assert isinstance(result[4], dict)
    assert result[4]["nested"]["y"] == "nested"
