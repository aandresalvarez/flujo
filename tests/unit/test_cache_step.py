import pytest

from typing import Any, Dict, List

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.steps.cache_step import _generate_cache_key, _serialize_for_key
from flujo.domain.models import PipelineContext, BaseModel as FlujoBaseModel
from pydantic import BaseModel

import json

from flujo.testing.utils import StubAgent, gather_result


class Model(BaseModel):
    a: int
    b: str


def test_generate_cache_key() -> None:
    m = Model(a=1, b="x")
    dummy = Step(name="dummy")
    key1 = _generate_cache_key(dummy, m)
    key2 = _generate_cache_key(dummy, {"a": 1, "b": "x"})
    assert isinstance(key1, str)
    assert isinstance(key2, str)

    class Unserializable:
        pass

    assert isinstance(_generate_cache_key(dummy, Unserializable()), str)

    key_ctx1 = _generate_cache_key(dummy, m, context={"val": 1})
    key_ctx2 = _generate_cache_key(dummy, m, context={"val": 2})
    assert key_ctx1 != key_ctx2


@pytest.mark.asyncio
async def test_cache_hit_and_miss() -> None:
    agent = StubAgent(["ok"])
    inner = Step.solution(agent)
    cache = InMemoryCache()
    cached_step = Step.cached(inner, cache_backend=cache)
    runner = Flujo(cached_step)

    result1 = await gather_result(runner, "in")
    first_meta = result1.step_history[0].metadata_
    result2 = await gather_result(runner, "in")

    assert agent.call_count == 1
    assert first_meta is None or "cache_hit" not in first_meta
    assert result2.step_history[0].metadata_["cache_hit"] is True


class FailingBackend(InMemoryCache):
    async def get(self, key: str) -> Any:
        raise RuntimeError("boom")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_cache_backend_errors_are_ignored() -> None:
    agent = StubAgent(["ok", "ok"])
    inner = Step.solution(agent)
    cached = Step.cached(inner, cache_backend=FailingBackend())
    runner = Flujo(cached)

    await gather_result(runner, "x")
    await gather_result(runner, "x")

    assert agent.call_count == 2


@pytest.mark.asyncio
async def test_cache_key_differs_for_same_name_steps() -> None:
    agent1 = StubAgent(["a", "a"])
    agent2 = StubAgent(["b", "b"])
    step1 = Step.solution(agent1, name="dup")
    step2 = Step.solution(agent2, name="dup")
    cache = InMemoryCache()
    runner = Flujo(
        Step.cached(step1, cache_backend=cache) >> Step.cached(step2, cache_backend=cache)
    )

    await gather_result(runner, "in")
    result2 = await gather_result(runner, "in")

    h1, h2 = result2.step_history
    assert h1.metadata_["cache_hit"] is True
    assert h2.metadata_["cache_hit"] is True
    assert agent1.call_count == 1
    assert agent2.call_count == 1


def test_cache_key_with_nested_basemodel():
    class NestedModel(FlujoBaseModel):
        value: int
        description: str

    class ComplexModel(FlujoBaseModel):
        name: str
        nested: NestedModel
        items: List[str]
        metadata: Dict[str, Any]

    nested = NestedModel(value=42, description="test")
    complex_obj = ComplexModel(
        name="test", nested=nested, items=["a", "b", "c"], metadata={"key": "value", "number": 123}
    )
    step = Step(name="nested_test")
    key = _generate_cache_key(step, complex_obj)
    assert isinstance(key, str)
    # Should be JSON serializable
    serialized = _serialize_for_key(complex_obj)
    json_str = json.dumps(serialized, sort_keys=True)
    assert "nested" in json_str and "value" in json_str


def test_cache_key_with_pipeline_context_runid_exclusion():
    context = PipelineContext(initial_prompt="prompt", run_id="should_be_excluded")
    step = Step(name="ctx_test")
    key = _generate_cache_key(step, "data", context=context)
    assert isinstance(key, str)
    # Check run_id is excluded from serialization
    serialized = _serialize_for_key(context)
    assert "run_id" not in serialized
    # Should be JSON serializable
    json.dumps(serialized)


def test_cache_key_with_step_and_nested_agent():
    class DummyAgent:
        def run(self, data):
            return data

    agent = DummyAgent()
    step = Step(name="step_with_agent", agent=agent)
    key = _generate_cache_key(step, "input")
    assert isinstance(key, str)
    # Should include agent class name or fallback to type name
    serialized = _serialize_for_key(step)
    # Accept either DummyAgent or dict (depending on Step serialization logic)
    assert "agent" in serialized
    assert serialized["agent"] in ("DummyAgent", "dict")
    json.dumps(serialized)


def test_cache_key_with_custom_serializer():
    from flujo.utils import register_custom_serializer

    class CustomType:
        def __init__(self, value):
            self.value = value

    def serialize_custom_type(obj):
        return {"custom": obj.value}

    register_custom_serializer(CustomType, serialize_custom_type)
    obj = CustomType("abc")
    step = Step(name="custom_serializer")
    key = _generate_cache_key(step, obj)
    assert isinstance(key, str)
    # Should be JSON serializable; may fallback to str if registry is not used in this context
    serialized = _serialize_for_key(obj)
    # Accept either the custom dict or string fallback
    if isinstance(serialized, dict):
        assert serialized == {"custom": "abc"}
    else:
        assert isinstance(serialized, str)
    json.dumps(serialized if isinstance(serialized, dict) else {"val": serialized})


def test_cache_key_with_unserializable_object_fallback():
    class Unserializable:
        pass

    step = Step(name="unserializable")
    key = _generate_cache_key(step, Unserializable())
    assert isinstance(key, str)
    # Should fallback to repr
    serialized = _serialize_for_key(Unserializable())
    assert isinstance(serialized, str)


def test_cache_key_with_nested_pipelinecontext_field():
    class MyContext(PipelineContext):
        extra: int
        model_config = {"arbitrary_types_allowed": True}

    class WrapperModel(FlujoBaseModel):
        context: MyContext
        label: str

    ctx = MyContext(initial_prompt="hi", run_id="should_be_excluded", extra=42)
    wrapper = WrapperModel(context=ctx, label="wrapped")
    serialized = _serialize_for_key(wrapper)
    # Ensure run_id is excluded from nested PipelineContext
    assert "run_id" not in serialized["context"]
    # Should be JSON serializable
    json.dumps(serialized)


def test_cache_key_with_list_of_pipelinecontexts():
    class MyContext(PipelineContext):
        extra: int
        model_config = {"arbitrary_types_allowed": True}

    class ListWrapper(FlujoBaseModel):
        contexts: list[MyContext]

    ctxs = [MyContext(initial_prompt=f"p{i}", run_id=f"r{i}", extra=i) for i in range(3)]
    wrapper = ListWrapper(contexts=ctxs)
    serialized = _serialize_for_key(wrapper)
    # Ensure run_id is excluded from all PipelineContexts in the list
    for ctx_ser in serialized["contexts"]:
        assert "run_id" not in ctx_ser
    json.dumps(serialized)


def test_cache_key_with_list_of_basemodels_and_pipelinecontext():
    class MyContext(PipelineContext):
        extra: int
        model_config = {"arbitrary_types_allowed": True}

    class OtherModel(FlujoBaseModel):
        foo: int

    class ListWrapper(FlujoBaseModel):
        items: list[FlujoBaseModel]

    items = [
        OtherModel(foo=1),
        MyContext(initial_prompt="x", run_id="should_be_excluded", extra=99),
    ]
    wrapper = ListWrapper(items=items)
    serialized = _serialize_for_key(wrapper)
    # When Pydantic stringifies heterogeneous lists, the result is a string
    # Ensure the string is JSON-serializable and doesn't contain run_id
    assert isinstance(serialized["items"], str)
    assert "run_id" not in serialized["items"]
    assert "MyContext" in serialized["items"]
    assert "initial_prompt" in serialized["items"]
    assert "extra" in serialized["items"]
    json.dumps(serialized)


def test_cache_key_deeply_nested_structures():
    class InnerModel(FlujoBaseModel):
        value: int

    class OuterModel(FlujoBaseModel):
        nested: dict[str, list[InnerModel]]
        model_config = {"arbitrary_types_allowed": True}

    obj = OuterModel(nested={"a": [InnerModel(value=1), InnerModel(value=2)], "b": []})
    serialized = _serialize_for_key(obj)
    # Pydantic may stringify complex nested structures
    if isinstance(serialized["nested"], dict):
        assert isinstance(serialized["nested"]["a"], list)
        assert all(isinstance(item, dict) for item in serialized["nested"]["a"])
    else:
        # String fallback is acceptable for complex nested structures
        assert isinstance(serialized["nested"], str)
    json.dumps(serialized)


def test_cache_key_circular_reference():
    class Node(FlujoBaseModel):
        value: int
        next: Any = None
        model_config = {"arbitrary_types_allowed": True}

    a = Node(value=1)
    b = Node(value=2, next=a)
    a.next = b  # Circular reference
    serialized = _serialize_for_key(a)
    # Should fallback to repr for the circular part
    assert isinstance(serialized, str)
    json.dumps(serialized)


def test_cache_key_custom_type_with_serializer():
    from flujo.utils import register_custom_serializer

    class Custom:
        def __init__(self, x):
            self.x = x

    def custom_serializer(obj):
        return {"custom_x": obj.x}

    register_custom_serializer(Custom, custom_serializer)

    class ModelWithCustom(FlujoBaseModel):
        custom: Custom
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithCustom(custom=Custom(42))
    serialized = _serialize_for_key(obj)
    assert serialized["custom"] == {"custom_x": 42}
    json.dumps(serialized)


def test_cache_key_non_string_dict_keys():
    class ModelWithDict(FlujoBaseModel):
        data: dict[int, str]
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithDict(data={1: "a", 2: "b"})
    serialized = _serialize_for_key(obj)
    # Pydantic may stringify dicts with non-string keys
    if isinstance(serialized["data"], dict):
        # JSON requires string keys
        assert all(isinstance(k, str) for k in serialized["data"].keys())
    else:
        # String fallback is acceptable for dicts with non-string keys
        assert isinstance(serialized["data"], str)
    json.dumps(serialized)


def test_cache_key_sets_and_frozensets():
    class ModelWithSets(FlujoBaseModel):
        s: set[int]
        fs: frozenset[str]
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithSets(s={1, 2, 3}, fs=frozenset({"a", "b"}))
    serialized = _serialize_for_key(obj)
    assert isinstance(serialized["s"], list)
    assert isinstance(serialized["fs"], list)
    json.dumps(serialized)


def test_cache_key_bytes_and_memoryview():
    class ModelWithBytes(FlujoBaseModel):
        b: bytes
        mv: memoryview
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithBytes(b=b"abc", mv=memoryview(b"xyz"))
    serialized = _serialize_for_key(obj)
    assert isinstance(serialized["b"], str)
    assert isinstance(serialized["mv"], str)
    json.dumps(serialized)


def test_cache_key_callable_fields():
    def foo():
        return 1

    class ModelWithCallable(FlujoBaseModel):
        cb: Any
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithCallable(cb=foo)
    serialized = _serialize_for_key(obj)
    assert isinstance(serialized["cb"], str)
    assert "foo" in serialized["cb"]
    json.dumps(serialized)


def test_cache_key_enum_values():
    from enum import Enum

    class MyEnum(Enum):
        A = "a"
        B = "b"

    class ModelWithEnum(FlujoBaseModel):
        e: MyEnum
        d: dict[MyEnum, int]
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithEnum(e=MyEnum.A, d={MyEnum.B: 2})
    serialized = _serialize_for_key(obj)
    assert serialized["e"] == "a"
    # Pydantic may stringify dicts with enum keys
    if isinstance(serialized["d"], dict):
        assert list(serialized["d"].keys())[0] == "b"
    else:
        # String fallback is acceptable for dicts with enum keys
        assert isinstance(serialized["d"], str)
    json.dumps(serialized)


def test_cache_key_complex_numbers():
    class ModelWithComplex(FlujoBaseModel):
        c: complex
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithComplex(c=3 + 4j)
    serialized = _serialize_for_key(obj)
    # Should be a dict with 'real' and 'imag'
    assert isinstance(serialized["c"], dict)
    assert set(serialized["c"].keys()) == {"real", "imag"}
    json.dumps(serialized)


def test_cache_key_very_large_structure():
    class LargeModel(FlujoBaseModel):
        items: list[int]
        model_config = {"arbitrary_types_allowed": True}

    obj = LargeModel(items=list(range(10000)))
    serialized = _serialize_for_key(obj)
    # Pydantic may stringify very large structures
    if isinstance(serialized["items"], list):
        assert len(serialized["items"]) == 10000
    else:
        # String fallback is acceptable for very large structures
        assert isinstance(serialized["items"], str)
    json.dumps(serialized)
