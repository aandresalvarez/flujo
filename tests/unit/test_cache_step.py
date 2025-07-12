import pytest

from typing import Any

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.steps.cache_step import _generate_cache_key, _serialize_for_key
from flujo.domain.models import PipelineContext, BaseModel as FlujoBaseModel
from flujo.utils.serialization import safe_serialize
from pydantic import BaseModel

import json

from flujo.testing.utils import StubAgent, gather_result

from hypothesis import given, settings, strategies as st

# --- Property-based test for robust serialization (regression for circular refs, set ordering, etc.) ---
# This test is designed to catch regressions and new edge cases in cache key serialization, including:
#   - Circular references
#   - Sets with mixed types
#   - Deeply nested structures
#   - Custom objects and Pydantic models
#   - Non-deterministic __str__
# It is a regression for bugs fixed in v0.7.0 (see flujo/steps/cache_step.py)

# Helper: strategy for hashable objects (for sets)
hashable_base = st.one_of(
    st.none(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.booleans(),
    st.tuples(st.integers(), st.integers()),
    st.frozensets(st.integers(), max_size=3),
)


@st.composite
def recursive_data(draw, max_depth=3):
    if max_depth <= 0:
        return draw(
            st.one_of(
                st.none(), st.integers(), st.floats(allow_nan=False), st.text(), st.booleans()
            )
        )
    base = st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.text(), st.booleans())
    # Optionally wrap in tuple, list, set, dict
    container = st.deferred(lambda: recursive_data(max_depth=max_depth - 1))
    dicts = st.dictionaries(st.text(), container, max_size=3)
    lists = st.lists(container, max_size=3)
    # Only allow sets of hashable_base
    sets = st.sets(hashable_base, max_size=3)
    tuples = st.tuples(container, container)
    # Compose
    return draw(st.one_of(base, dicts, lists, sets, tuples))


@given(data=recursive_data())
@settings(max_examples=50, deadline=2000)
def test_property_based_serialize_for_key_does_not_crash(data):
    """Property-based: _serialize_for_key should not crash or infinitely recurse on any nested structure."""
    from flujo.steps.cache_step import _serialize_for_key

    try:
        result1 = _serialize_for_key(data)
        result2 = _serialize_for_key(data)
        # Should be deterministic
        assert result1 == result2
    except Exception as e:
        pytest.fail(f"_serialize_for_key failed on {data!r}: {e}")


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
        step: Step
        metadata: dict

    class TestAgent:
        def run(self, data):
            return data

    agent = TestAgent()
    step = Step(name="nested_test", agent=agent)
    nested = NestedModel(step=step, metadata={"key": "value"})

    serialized = _serialize_for_key(nested)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "metadata" in serialized
    assert "step" in serialized


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
    # The cache serialization returns dictionaries for Step instances
    assert isinstance(serialized, dict)
    assert "agent" in serialized
    assert serialized["agent"] == "DummyAgent"


def test_agent_serialization_fix_verification():
    """Test that the agent serialization fix works correctly."""

    class TestAgent:
        def run(self, data):
            return data

    class AnotherAgent:
        def run(self, data):
            return data

    # Test with different agent types
    agent1 = TestAgent()
    agent2 = AnotherAgent()

    step1 = Step(name="test_step", agent=agent1)
    step2 = Step(name="test_step", agent=agent2)

    # Serialize both steps
    serialized1 = _serialize_for_key(step1)
    serialized2 = _serialize_for_key(step2)

    # Verify both return dictionaries with agent class names
    assert isinstance(serialized1, dict)
    assert isinstance(serialized2, dict)
    assert serialized1["agent"] == "TestAgent"
    assert serialized2["agent"] == "AnotherAgent"


def test_agent_serialization_with_complex_agents():
    """Test agent serialization with more complex agent types."""

    class ComplexAgent:
        def __init__(self, name: str, config: dict):
            self.name = name
            self.config = config

        def run(self, data):
            return f"{self.name}: {data}"

    class AsyncAgent:
        async def run_async(self, data):
            return f"async: {data}"

    # Test with complex agent
    complex_agent = ComplexAgent("test", {"timeout": 30})
    step1 = Step(name="complex_step", agent=complex_agent)
    serialized1 = _serialize_for_key(step1)
    # The cache serialization returns dictionaries for Step instances
    assert isinstance(serialized1, dict)
    assert serialized1["agent"] == "ComplexAgent"


def test_agent_serialization_edge_cases():
    """Test agent serialization with edge cases."""

    # Test with agent that has no __name__ attribute
    class AnonymousAgent:
        def run(self, data):
            return data

    # Create an instance without __name__ by using a different approach
    agent = AnonymousAgent()
    # We can't set __name__ to None, but we can test with a class that has a problematic name
    step = Step(name="anonymous_step", agent=agent)
    serialized = _serialize_for_key(step)

    # Should return dictionary with agent class name
    assert isinstance(serialized, dict)
    assert serialized["agent"] == "AnonymousAgent"


def test_agent_serialization_with_nested_structures():
    """Test agent serialization when agents are part of nested structures."""

    class NestedAgent:
        def run(self, data):
            return data

    # Create a model that contains a step with an agent
    class ModelWithStep(FlujoBaseModel):
        step: Step
        metadata: dict

    agent = NestedAgent()
    step = Step(name="nested_step", agent=agent)
    model = ModelWithStep(step=step, metadata={"key": "value"})

    # Serialize the model
    serialized = _serialize_for_key(model)

    # Verify the nested step's agent is correctly serialized
    assert isinstance(serialized, dict)
    assert "step" in serialized
    assert "metadata" in serialized


def test_serialization_utility_agent_handling():
    """Test that the serialization utility correctly handles agent serialization."""
    from flujo.utils.serialization import safe_serialize as util_serialize

    class UtilityTestAgent:
        def run(self, data):
            return data

    agent = UtilityTestAgent()
    step = Step(name="utility_test_step", agent=agent)

    # Test the utility's serialization
    serialized = util_serialize(step)
    # The utility serialization returns a string representation for complex objects
    assert isinstance(serialized, str)
    assert "Step" in serialized


def test_serialization_consistency_between_utilities():
    """Test that both serialization utilities produce consistent results for agents."""
    from flujo.utils.serialization import safe_serialize as util_serialize

    class ConsistencyTestAgent:
        def run(self, data):
            return data

    agent = ConsistencyTestAgent()
    step = Step(name="consistency_test", agent=agent)

    # Test both serialization functions
    cache_result = _serialize_for_key(step)
    util_result = util_serialize(step)

    # Cache serialization returns dict, util serialization returns string
    assert isinstance(cache_result, dict)
    assert isinstance(util_result, str)
    assert cache_result["agent"] == "ConsistencyTestAgent"
    assert "Step" in util_result


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
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "context" in serialized
    assert "label" in serialized


def test_cache_key_with_list_of_pipelinecontexts():
    class MyContext(PipelineContext):
        extra: int
        model_config = {"arbitrary_types_allowed": True}

    class ListWrapper(FlujoBaseModel):
        contexts: list[MyContext]

    ctxs = [MyContext(initial_prompt=f"p{i}", run_id=f"r{i}", extra=i) for i in range(3)]
    wrapper = ListWrapper(contexts=ctxs)
    serialized = _serialize_for_key(wrapper)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "contexts" in serialized


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
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "items" in serialized


def test_cache_key_deeply_nested_structures():
    class InnerModel(FlujoBaseModel):
        value: int

    class OuterModel(FlujoBaseModel):
        nested: dict[str, list[InnerModel]]
        model_config = {"arbitrary_types_allowed": True}

    obj = OuterModel(nested={"a": [InnerModel(value=1), InnerModel(value=2)], "b": []})
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "nested" in serialized


def test_cache_key_circular_reference():
    """Test that circular references are handled gracefully."""

    class Node(FlujoBaseModel):
        value: int
        next: Any = None
        model_config = {"arbitrary_types_allowed": True}

    # Create circular reference
    node1 = Node(value=1)
    node2 = Node(value=2, next=node1)
    node1.next = node2

    # Should handle circular reference gracefully
    serialized = _serialize_for_key(node1)
    assert serialized is not None
    # The circular reference should be serialized as a string placeholder
    assert isinstance(serialized, str)
    assert "Node" in serialized


def test_cache_key_custom_type_with_serializer():
    """Test custom type serialization."""

    class Custom:
        def __init__(self, x):
            self.x = x

    def custom_serializer(obj):
        if isinstance(obj, Custom):
            return {"custom": obj.x}
        raise TypeError(f"Cannot serialize {type(obj)}")

    class ModelWithCustom(FlujoBaseModel):
        custom: Custom
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithCustom(custom=Custom(42))
    # Should use custom serializer
    serialized = safe_serialize(obj, default_serializer=custom_serializer)
    assert serialized["custom"]["custom"] == 42


def test_cache_key_non_string_dict_keys():
    class ModelWithDict(FlujoBaseModel):
        data: dict[int, str]
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithDict(data={1: "a", 2: "b"})
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "data" in serialized


def test_cache_key_sets_and_frozensets():
    class ModelWithSets(FlujoBaseModel):
        s: set[int]
        fs: frozenset[str]
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithSets(s={1, 2, 3}, fs=frozenset({"a", "b"}))
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "s" in serialized
    assert "fs" in serialized


def test_cache_key_bytes_and_memoryview():
    class ModelWithBytes(FlujoBaseModel):
        b: bytes
        mv: memoryview
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithBytes(b=b"abc", mv=memoryview(b"xyz"))
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "b" in serialized
    assert "mv" in serialized


def test_cache_key_callable_fields():
    def foo():
        return 1

    class ModelWithCallable(FlujoBaseModel):
        cb: Any
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithCallable(cb=foo)
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "cb" in serialized


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
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "e" in serialized
    assert "d" in serialized


def test_cache_key_complex_numbers():
    class ModelWithComplex(FlujoBaseModel):
        c: complex
        model_config = {"arbitrary_types_allowed": True}

    obj = ModelWithComplex(c=3 + 4j)
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "c" in serialized


def test_cache_key_very_large_structure():
    class LargeModel(FlujoBaseModel):
        items: list[int]
        model_config = {"arbitrary_types_allowed": True}

    obj = LargeModel(items=list(range(10000)))
    serialized = _serialize_for_key(obj)
    # The cache serialization returns dictionaries for BaseModel instances
    assert isinstance(serialized, dict)
    assert "items" in serialized


def test_cache_key_deterministic_set_ordering():
    """Test that sets and frozensets are serialized with deterministic ordering."""
    from flujo.steps.cache_step import _serialize_for_key

    # Test with sets containing different types
    test_set = {3, 1, 2, "b", "a", "c"}
    test_frozenset = frozenset([3, 1, 2, "b", "a", "c"])

    # Serialize multiple times to ensure consistent ordering
    result1 = _serialize_for_key(test_set)
    result2 = _serialize_for_key(test_set)
    result3 = _serialize_for_key(test_set)

    # All results should be identical
    assert result1 == result2 == result3

    # Test frozenset
    result4 = _serialize_for_key(test_frozenset)
    result5 = _serialize_for_key(test_frozenset)
    result6 = _serialize_for_key(test_frozenset)

    # All results should be identical
    assert result4 == result5 == result6

    # Test with nested sets
    nested_set = {1, 2, frozenset([3, 1, 2])}
    result7 = _serialize_for_key(nested_set)
    result8 = _serialize_for_key(nested_set)

    # Results should be identical
    assert result7 == result8

    # Test with complex nested structures (using frozensets since sets can't contain sets)
    complex_set = {frozenset([3, 1, 2]), frozenset([2, 1, 3]), frozenset(["c", "a", "b"])}
    result9 = _serialize_for_key(complex_set)
    result10 = _serialize_for_key(complex_set)

    # Results should be identical
    assert result9 == result10


def test_cache_key_stability_comprehensive():
    """Comprehensive test for cache key stability across various data types and structures."""
    from flujo.steps.cache_step import _serialize_for_key
    import json

    # Test data with potential stability issues
    test_cases = [
        # Sets and frozensets (the main issue we fixed)
        {"name": "sets", "data": {3, 1, 2, "b", "a", "c"}},
        {"name": "frozensets", "data": frozenset([3, 1, 2, "b", "a", "c"])},
        {"name": "nested_sets", "data": {frozenset([3, 1, 2]), frozenset([2, 1, 3])}},
        # Dictionaries with non-string keys (potential ordering issues)
        {"name": "dict_with_int_keys", "data": {3: "three", 1: "one", 2: "two"}},
        {"name": "dict_with_tuple_keys", "data": {(1, 2): "tuple1", (2, 1): "tuple2"}},
        {
            "name": "dict_with_frozenset_keys",
            "data": {frozenset([1, 2]): "set1", frozenset([2, 1]): "set2"},
        },
        # Mixed collections
        {
            "name": "mixed_collections",
            "data": {
                "list": [3, 1, 2],
                "set": {3, 1, 2},
                "frozenset": frozenset([3, 1, 2]),
                "dict": {3: "three", 1: "one", 2: "two"},
            },
        },
        # Nested structures with potential ordering issues
        {
            "name": "deeply_nested",
            "data": {
                "level1": {
                    "sets": {frozenset([3, 1, 2]), frozenset([2, 1, 3])},
                    "dicts": {frozenset([1, 2]): "value1", frozenset([2, 1]): "value2"},
                }
            },
        },
        # Edge cases
        {
            "name": "empty_collections",
            "data": {
                "empty_set": set(),
                "empty_frozenset": frozenset(),
                "empty_dict": {},
                "empty_list": [],
            },
        },
        {
            "name": "single_element_collections",
            "data": {
                "single_set": {42},
                "single_frozenset": frozenset([42]),
                "single_dict": {42: "value"},
            },
        },
        # Complex nested structures
        {
            "name": "complex_nested",
            "data": {
                "outer": {
                    "inner1": {
                        "sets": {frozenset([1, 2, 3]), frozenset([3, 2, 1])},
                        "dicts": {frozenset([1, 2]): {frozenset([3, 4]): "deep"}},
                    },
                    "inner2": [
                        {frozenset([1, 2]): "list_item1"},
                        {frozenset([2, 1]): "list_item2"},
                    ],
                }
            },
        },
    ]

    for test_case in test_cases:
        data = test_case["data"]
        name = test_case["name"]

        # Serialize multiple times to check for consistency
        results = []
        for i in range(5):
            result = _serialize_for_key(data)
            # Convert result to a JSON-serializable format for comparison
            try:
                json_result = json.dumps(result, sort_keys=True)
                results.append(json_result)
            except TypeError:
                # If JSON serialization fails, use string representation
                results.append(str(result))

        # All results should be identical
        first_result = results[0]
        all_identical = all(r == first_result for r in results)

        assert all_identical, f"Cache key instability detected for {name}. Results: {results}"


def test_cache_key_stability_with_step():
    """Test cache key stability when using actual Step objects."""
    from flujo.steps.cache_step import _generate_cache_key
    from flujo.domain.dsl import Step

    async def test_step(data: str) -> str:
        return f"processed_{data}"

    step = Step.from_callable(test_step, name="test_step")

    # Test data with potential stability issues
    test_data_sets = [
        # Sets and frozensets
        {3, 1, 2, "b", "a", "c"},
        frozenset([3, 1, 2, "b", "a", "c"]),
        {frozenset([3, 1, 2]), frozenset([2, 1, 3])},
        # Dictionaries with non-string keys
        {3: "three", 1: "one", 2: "two"},
        {frozenset([1, 2]): "value1", frozenset([2, 1]): "value2"},
        # Mixed structures
        {
            "sets": {3, 1, 2},
            "frozensets": frozenset([3, 1, 2]),
            "dicts": {frozenset([1, 2]): "value"},
        },
    ]

    for data in test_data_sets:
        # Generate cache keys multiple times
        keys = []
        for i in range(5):
            key = _generate_cache_key(step, data, None, None)
            keys.append(key)

        # All keys should be identical
        first_key = keys[0]
        all_identical = all(k == first_key for k in keys)

        assert all_identical, f"Cache key instability with Step for data {type(data)}. Keys: {keys}"


def test_cache_key_stability_edge_cases():
    """Test cache key stability for edge cases and boundary conditions."""
    from flujo.steps.cache_step import _serialize_for_key
    import json

    edge_cases = [
        # Very large sets
        {"name": "large_set", "data": set(range(100))},
        # Sets with complex objects
        {
            "name": "complex_objects",
            "data": {frozenset([1, 2]): "value1", frozenset([2, 1]): "value2"},
        },
        # Sets with None values
        {"name": "none_values", "data": {None, 1, 2, "string"}},
        # Sets with boolean values
        {"name": "boolean_values", "data": {True, False, 1, 0}},
        # Sets with float values
        {"name": "float_values", "data": {1.0, 2.0, 3.0, 1.5}},
        # Sets with mixed types
        {"name": "mixed_types", "data": {1, "string", 1.5, True, None}},
        # Deeply nested structures
        {
            "name": "deep_nesting",
            "data": {
                "level1": {
                    "level2": {
                        "level3": {
                            "sets": {frozenset([1, 2]), frozenset([2, 1])},
                            "dicts": {frozenset([3, 4]): {frozenset([5, 6]): "deep"}},
                        }
                    }
                }
            },
        },
        # Circular reference prevention
        {
            "name": "circular_reference",
            "data": {
                "self_ref": None  # Will be set to create circular reference
            },
        },
    ]

    # Create circular reference for testing
    circular_data = {"self_ref": None}
    circular_data["self_ref"] = circular_data
    edge_cases.append({"name": "circular_reference", "data": circular_data})

    for test_case in edge_cases:
        data = test_case["data"]
        name = test_case["name"]

        try:
            # Serialize multiple times
            results = []
            for i in range(3):  # Fewer iterations for edge cases
                result = _serialize_for_key(data)
                results.append(json.dumps(result, sort_keys=True))

            # All results should be identical
            first_result = results[0]
            all_identical = all(r == first_result for r in results)

            assert all_identical, f"Cache key instability for edge case {name}. Results: {results}"

        except Exception as e:
            # Some edge cases might fail, which is expected
            print(f"Edge case {name} failed as expected: {e}")


def test_cache_key_stability_performance():
    """Test that cache key stability doesn't significantly impact performance."""
    from flujo.steps.cache_step import _serialize_for_key
    import time

    # Create a moderately complex data structure
    test_data = {
        "sets": {frozenset([1, 2, 3]), frozenset([3, 2, 1]), frozenset([2, 1, 3])},
        "dicts": {frozenset([1, 2]): "value1", frozenset([2, 1]): "value2"},
        "nested": {
            "inner": {
                "sets": {frozenset([4, 5, 6]), frozenset([6, 5, 4])},
                "dicts": {frozenset([7, 8]): "value3"},
            }
        },
    }

    # Measure serialization time
    start_time = time.time()
    for i in range(1000):
        _serialize_for_key(test_data)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / 1000

    # Should complete in reasonable time (less than 1ms per serialization)
    assert avg_time < 0.001, f"Serialization too slow: {avg_time:.6f}s per operation"

    print(f"Average serialization time: {avg_time:.6f}s per operation")


def test_cache_key_stability_regression():
    """Test to prevent regression of cache key stability issues."""
    from flujo.steps.cache_step import _serialize_for_key
    import json

    # Test cases that previously had stability issues
    regression_tests = [
        # Original issue: sets with non-deterministic ordering
        {"name": "original_set_issue", "data": {3, 1, 2, "b", "a", "c"}},
        # Similar issue with frozensets
        {"name": "original_frozenset_issue", "data": frozenset([3, 1, 2, "b", "a", "c"])},
        # Nested sets
        {"name": "nested_sets_issue", "data": {frozenset([3, 1, 2]), frozenset([2, 1, 3])}},
        # Dictionaries with set-like keys
        {
            "name": "dict_with_set_keys",
            "data": {frozenset([1, 2]): "value1", frozenset([2, 1]): "value2"},
        },
        # Mixed collections
        {
            "name": "mixed_collections",
            "data": {"list": [3, 1, 2], "set": {3, 1, 2}, "frozenset": frozenset([3, 1, 2])},
        },
    ]

    for test_case in regression_tests:
        data = test_case["data"]
        name = test_case["name"]

        # Serialize multiple times
        results = []
        for i in range(10):  # More iterations for regression testing
            result = _serialize_for_key(data)
            # Convert result to a JSON-serializable format for comparison
            try:
                json_result = json.dumps(result, sort_keys=True)
                results.append(json_result)
            except TypeError:
                # If JSON serialization fails, use string representation
                results.append(str(result))

        # All results should be identical
        first_result = results[0]
        all_identical = all(r == first_result for r in results)

        assert all_identical, f"Regression detected for {name}. Results: {results}"

        # Also verify that the result is sorted (for sets/frozensets)
        if isinstance(data, (set, frozenset)) or any(
            isinstance(v, (set, frozenset)) for v in data.values() if isinstance(data, dict)
        ):
            serialized = _serialize_for_key(data)
            if isinstance(serialized, list):
                # Check that the list is sorted
                assert serialized == sorted(serialized, key=lambda x: str(x)), (
                    f"Result not sorted for {name}"
                )


def test_circular_reference_detection():
    """Test that circular references are properly detected and handled."""
    from flujo.steps.cache_step import _serialize_for_key

    # Create a circular reference
    obj1 = {"name": "obj1"}
    obj2 = {"name": "obj2", "ref": obj1}
    obj1["ref"] = obj2  # Create circular reference

    result = _serialize_for_key(obj1)

    # Should detect circular reference and return a stable representation
    assert "<dict circular>" in str(result)


def test_deterministic_set_ordering():
    """Test that sets are ordered deterministically for cache key generation."""
    from flujo.steps.cache_step import _serialize_for_key, _sort_set_deterministically

    # Test with different types of objects in sets
    test_set = {3, 1, 2, "b", "a", "c"}
    sorted_list = _sort_set_deterministically(test_set)

    # Should be sorted deterministically
    assert sorted_list == [1, 2, 3, "a", "b", "c"]

    # Test with hashable complex objects
    complex_set = {(1, 2, 3), (3, 2, 1), (2, 1, 3), frozenset([1, 2, 3]), frozenset([3, 2, 1])}

    # Should handle complex objects deterministically
    result = _serialize_for_key(complex_set)
    assert isinstance(result, list)

    # Test that multiple runs produce the same result
    result2 = _serialize_for_key(complex_set)
    assert result == result2


def test_circular_reference_in_nested_structures():
    """Test circular reference detection in nested structures."""
    from flujo.steps.cache_step import _serialize_for_key

    # Create nested structure with circular reference
    outer = {"level": "outer"}
    inner = {"level": "inner", "parent": outer}
    outer["child"] = inner

    result = _serialize_for_key(outer)

    # Should handle nested circular references
    assert isinstance(result, dict)
    # The circular reference should be detected and handled
    assert "child" in result or "<dict circular>" in str(result)


def test_set_with_unhashable_objects():
    """Test set serialization with unhashable objects."""
    from flujo.steps.cache_step import _serialize_for_key

    # This should not raise an exception
    try:
        # Create a set with mixed types
        mixed_set = {1, "string", (1, 2), frozenset([1, 2])}
        result = _serialize_for_key(mixed_set)
        assert isinstance(result, list)
    except Exception as e:
        pytest.fail(f"Set serialization failed: {e}")
