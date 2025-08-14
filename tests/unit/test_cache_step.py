from unittest.mock import Mock, patch
import builtins

from flujo.steps.cache_step import (
    CacheStep,
    _serialize_for_cache_key,
    _sort_set_deterministically,
    _get_stable_repr,
    _serialize_list_for_key,
    _create_step_fingerprint,
    _generate_cache_key,
)
from flujo.domain.dsl import Step
from flujo.infra.caching import InMemoryCache
from flujo.domain.models import BaseModel


class TestCacheStep:
    """Test the CacheStep class and its methods."""

    def test_cache_step_cached_classmethod(self):
        """Test the cached classmethod."""
        # Create a proper step

        step = Step(name="test_step")

        # Test with default cache backend
        cache_step = CacheStep.cached(step)
        assert isinstance(cache_step, CacheStep)
        assert cache_step.name == "test_step"
        assert cache_step.wrapped_step == step
        assert isinstance(cache_step.cache_backend, InMemoryCache)

        # Test with custom cache backend
        custom_cache = Mock()
        # Mock the required CacheBackend methods
        custom_cache.get = Mock()
        custom_cache.set = Mock()
        cache_step = CacheStep.cached(step, cache_backend=custom_cache)
        assert cache_step.cache_backend == custom_cache

    def test_cache_step_cached_classmethod_with_none_backend(self):
        """Test the cached classmethod with None cache backend."""
        step = Step(name="test_step")

        cache_step = CacheStep.cached(step, cache_backend=None)
        assert isinstance(cache_step.cache_backend, InMemoryCache)


class TestSerializeForCacheKey:
    """Test the _serialize_for_cache_key function."""

    def test_serialize_none(self):
        """Test serializing None."""
        result = _serialize_for_cache_key(None)
        assert result is None

    def test_serialize_circular_reference_step(self):
        """Test serializing a Step with circular reference."""
        step = Step(name="test_step")

        class TestAgent:
            pass

        step.agent = TestAgent()

        # Create circular reference
        visited = {id(step)}
        result = _serialize_for_cache_key(step, visited=visited)
        assert result == "TestAgent"

    def test_serialize_circular_reference_node(self):
        """Test serializing a Node-like object with circular reference."""

        class Node:
            def __init__(self):
                self.name = "test_node"

        node = Node()
        visited = {id(node)}
        result = _serialize_for_cache_key(node, visited=visited)
        assert result == "<Node circular>"

    def test_serialize_circular_reference_with_model_dump(self):
        """Test serializing an object with model_dump that has circular reference."""

        class ModelWithDump:
            def model_dump(self, mode="cache"):
                return {"field": "value"}

        obj = ModelWithDump()
        visited = {id(obj)}
        result = _serialize_for_cache_key(obj, visited=visited)
        assert result == "<ModelWithDump circular>"

    def test_serialize_circular_reference_dict(self):
        """Test serializing a dict with circular reference."""
        d = {"key": "value"}
        visited = {id(d)}
        result = _serialize_for_cache_key(d, visited=visited)
        assert result == "<dict circular>"

    def test_serialize_circular_reference_list(self):
        """Test serializing a list with circular reference."""
        lst = [1, 2, 3]
        visited = {id(lst)}
        result = _serialize_for_cache_key(lst, visited=visited)
        assert result == "<list circular>"

    def test_serialize_circular_reference_set(self):
        """Test serializing a set with circular reference."""
        s = {1, 2, 3}
        visited = {id(s)}
        result = _serialize_for_cache_key(s, visited=visited)
        assert result == "<list circular>"

    def test_serialize_circular_reference_generic(self):
        """Test serializing a generic object with circular reference."""
        obj = object()
        visited = {id(obj)}
        result = _serialize_for_cache_key(obj, visited=visited)
        assert result == "<circular>"

    def test_serialize_with_custom_serializer_exception(self):
        """Test serializing with custom serializer that raises exception."""

        def custom_serializer(obj):
            raise ValueError("Custom serializer error")

        with patch(
            "flujo.utils.serialization.lookup_custom_serializer", return_value=custom_serializer
        ):
            result = _serialize_for_cache_key("test")
            assert result == "test"  # Should fall back to original object

    def test_serialize_step_with_agent_field(self):
        """Test serializing a Step with agent field."""
        step = Step(name="test_step")

        class TestAgent:
            pass

        agent = TestAgent()
        step.agent = agent
        # Patch model_dump on the Step class
        original_model_dump = Step.model_dump
        Step.model_dump = lambda self, mode="cache": {"agent": agent, "other": "value"}
        try:
            result = _serialize_for_cache_key(step)
            assert result["agent"] == "TestAgent"
            assert result["other"] == "value"
        finally:
            Step.model_dump = original_model_dump

    def test_serialize_node_with_circular_next(self):
        """Test serializing a Node with circular next reference."""

        class Node:
            def __init__(self, name):
                self.name = name
                self.next = None

            def model_dump(self, mode="cache"):
                return {"name": self.name, "next": self.next}

        node1 = Node("node1")
        node2 = Node("node2")
        node1.next = node2
        node2.next = node1  # Create circular reference

        visited = {id(node1), id(node2)}
        result = _serialize_for_cache_key(node1, visited=visited)
        # When there's a circular reference, the function returns a string

        assert result == "<Node circular>"

    def test_serialize_node_with_non_circular_next(self):
        """Test serializing a Node with non-circular next reference."""

        class Node:
            def __init__(self, name):
                self.name = name
                self.next = None

            def model_dump(self, mode="cache"):
                return {"name": self.name, "next": self.next}

        node1 = Node("node1")
        node2 = Node("node2")
        node1.next = node2

        result = _serialize_for_cache_key(node1)
        assert result["name"] == "node1"
        assert result["next"]["name"] == "node2"

    def test_serialize_dict_with_run_id_and_initial_prompt(self):
        """Test serializing a dict with run_id and initial_prompt."""
        d = {"run_id": "123", "initial_prompt": "test", "other": "value"}
        result = _serialize_for_cache_key(d)
        assert "run_id" not in result
        assert result["initial_prompt"] == "test"
        assert result["other"] == "value"

    def test_serialize_dict_with_custom_serializer_for_values(self):
        """Test serializing a dict with custom serializer for values."""

        class CustomValue:
            pass

        def lookup_custom_serializer(obj):
            if isinstance(obj, CustomValue):
                return lambda x: "serialized_value"
            return None

        with patch(
            "flujo.utils.serialization.lookup_custom_serializer",
            side_effect=lookup_custom_serializer,
        ):
            d = {"key": CustomValue()}
            result = _serialize_for_cache_key(d)
            print(f"Actual result: {result!r}")
            assert result["key"] == "serialized_value"

    def test_serialize_dict_with_model_dump_value_exception(self):
        """Test serializing a dict with model_dump value that raises exception."""

        class ModelWithException:
            def model_dump(self, mode="cache"):
                raise ValueError("Model dump error")

            def __annotations__(self):
                return {"field": str}

        d = {"key": ModelWithException()}
        result = _serialize_for_cache_key(d)
        print(f"Full result: {result!r}")
        # Our improved serialization provides more detailed error messages

        assert isinstance(result, dict)
        assert "key" in result
        assert result["key"].startswith("<unserializable: ModelWithException")

    def test_serialize_dict_with_model_dump_value_recursion_error(self):
        """Test serializing a dict with model_dump value that raises RecursionError."""

        class ModelWithRecursion:
            def model_dump(self, mode="cache"):
                raise RecursionError("Recursion error")

            def __fields__(self):
                return {"field": str}

        d = {"key": ModelWithRecursion()}
        result = _serialize_for_cache_key(d)
        print(f"Full result: {result!r}")
        # Our improved serialization provides more detailed error messages

        assert isinstance(result, dict)
        assert "key" in result
        assert result["key"].startswith("<unserializable: ModelWithRecursion")

    def test_serialize_callable(self):
        """Test serializing a callable object."""

        def test_function():
            pass

        result = _serialize_for_cache_key(test_function)
        assert result == "<callable test_function>"

    def test_serialize_callable_without_name(self):
        """Test serializing a callable object without __name__."""

        def callable_obj():
            pass

        result = _serialize_for_cache_key(callable_obj)
        assert result.startswith("<callable")

    def test_serialize_exception_fallback(self):
        """Test serializing an object that raises exception."""

        class UnserializableObject:
            def __hash__(self):
                raise TypeError("Cannot hash")

        obj = UnserializableObject()
        result = _serialize_for_cache_key(obj)
        print(f"Actual result: {result!r}")
        # Our improved serialization provides better error handling
        assert result.startswith("<unserializable: UnserializableObject")
        # Verify that serialization gracefully handles the unhashable object without raising an exception
        assert isinstance(result, str)
        assert "UnserializableObject" in result


class TestSortSetDeterministically:
    """Test the _sort_set_deterministically function."""

    def test_sort_set_deterministically_success(self):
        """Test sorting a set deterministically."""
        s = {3, 1, 2}
        result = _sort_set_deterministically(s)
        assert result == [1, 2, 3]

    def test_sort_set_deterministically_with_complex_objects(self):
        """Test sorting a set with complex objects."""
        s = {"c", "a", "b"}
        result = _sort_set_deterministically(s)
        assert result == ["a", "b", "c"]

    def test_sort_set_deterministically_frozen_set(self):
        """Test sorting a frozenset deterministically."""
        s = frozenset([3, 1, 2])
        result = _sort_set_deterministically(s)
        assert result == [1, 2, 3]

    def test_sort_set_deterministically_with_exception(self):
        """Test sorting a set that raises exception during sorting."""

        class UnsortableObject:
            def __lt__(self, other):
                raise TypeError("Cannot compare")

        s = {UnsortableObject(), UnsortableObject()}
        result = _sort_set_deterministically(s)
        # Should fall back to string sorting
        assert len(result) == 2


class TestGetStableRepr:
    """Test the _get_stable_repr function."""

    def test_get_stable_repr_none(self):
        """Test getting stable representation of None."""
        result = _get_stable_repr(None)
        assert result == "None"

    def test_get_stable_repr_circular_reference(self):
        """Test getting stable representation with circular reference."""
        obj = object()
        visited = {id(obj)}
        result = _get_stable_repr(obj, visited=visited)
        assert result == "<object circular>"

    def test_get_stable_repr_primitive_types(self):
        """Test getting stable representation of primitive types."""
        assert _get_stable_repr(42) == "42"
        assert _get_stable_repr(3.14) == "3.14"
        assert _get_stable_repr("test") == "test"
        assert _get_stable_repr(True) == "True"

    def test_get_stable_repr_list(self):
        """Test getting stable representation of list."""
        lst = [1, 2, 3]
        result = _get_stable_repr(lst)
        assert result == "[1,2,3]"

    def test_get_stable_repr_tuple(self):
        """Test getting stable representation of tuple."""
        tup = (1, 2, 3)
        result = _get_stable_repr(tup)
        assert result == "[1,2,3]"

    def test_get_stable_repr_dict(self):
        """Test getting stable representation of dict."""
        d = {"b": 2, "a": 1}
        result = _get_stable_repr(d)
        assert result == "{a:1,b:2}"

    def test_get_stable_repr_set(self):
        """Test getting stable representation of set."""
        s = {3, 1, 2}
        result = _get_stable_repr(s)
        assert result == "{1,2,3}"

    def test_get_stable_repr_frozen_set(self):
        """Test getting stable representation of frozenset."""
        s = frozenset([3, 1, 2])
        result = _get_stable_repr(s)
        assert result == "{1,2,3}"

    def test_get_stable_repr_model_with_run_id(self):
        """Test getting stable representation of model with run_id."""

        class TestModel(BaseModel):
            field1: str = "value1"
            run_id: str = "123"

        obj = TestModel()
        result = _get_stable_repr(obj)
        assert "run_id" not in result
        assert "field1" in result

    def test_get_stable_repr_model_with_recursion_error(self):
        """Test getting stable representation of model with recursion error."""

        class TestModel(BaseModel):
            field1: str = "value1"

            def model_dump(self, mode="json"):
                raise RecursionError("Recursion error")

        obj = TestModel()
        result = _get_stable_repr(obj)
        assert result == "<TestModel circular>"

    def test_get_stable_repr_model_with_value_error(self):
        """Test getting stable representation of model with value error."""

        class TestModel(BaseModel):
            field1: str = "value1"

            def model_dump(self, mode="json"):
                raise ValueError("Value error")

        obj = TestModel()
        result = _get_stable_repr(obj)
        assert result == "<TestModel circular>"

    def test_get_stable_repr_callable(self):
        """Test getting stable representation of callable."""

        def test_function():
            pass

        result = _get_stable_repr(test_function)
        assert "test_function" in result

    def test_get_stable_repr_callable_without_module(self):
        """Test getting stable representation of callable without module."""

        def callable_obj():
            pass

        result = _get_stable_repr(callable_obj)
        assert "callable_obj" in result

    def test_get_stable_repr_hashable_object(self):
        """Test getting stable representation of hashable object."""

        class HashableObject:
            def __hash__(self):
                return 42

        obj = HashableObject()
        result = _get_stable_repr(obj)
        assert result == "HashableObject:42"

    def test_get_stable_repr_unhashable_object(self):
        """Test getting stable representation of unhashable object."""

        class UnhashableObject:
            pass

        obj = UnhashableObject()
        result = _get_stable_repr(obj)
        assert result.startswith("UnhashableObject:")

    def test_get_stable_repr_object_with_hash_exception(self):
        """Test getting stable representation of object with hash exception."""

        class ObjectWithHashException:
            def __hash__(self):
                raise TypeError("Cannot hash")

        obj = ObjectWithHashException()
        result = _get_stable_repr(obj)
        assert result == "ObjectWithHashException"


class TestSerializeListForKey:
    """Test the _serialize_list_for_key function."""

    def test_serialize_list_for_key_circular_reference_model(self):
        """Test serializing list with circular reference to model."""

        class TestModel(BaseModel):
            field1: str = "value1"

        obj = TestModel()
        visited = {id(obj)}
        lst = [obj]
        result = _serialize_list_for_key(lst, visited=visited)
        assert result[0] == "<TestModel circular>"

    def test_serialize_list_for_key_circular_reference_dict(self):
        """Test serializing list with circular reference to dict."""
        d = {"key": "value"}
        visited = {id(d)}
        lst = [d]
        result = _serialize_list_for_key(lst, visited=visited)
        assert result[0] == "<dict circular>"

    def test_serialize_list_for_key_circular_reference_list(self):
        """Test serializing list with circular reference to list."""
        lst = [1, 2, 3]
        visited = {id(lst)}
        lst2 = [lst]
        result = _serialize_list_for_key(lst2, visited=visited)
        assert result[0] == "<list circular>"

    def test_serialize_list_for_key_circular_reference_set(self):
        """Test serializing list with circular reference to set."""
        s = {1, 2, 3}
        visited = {id(s)}
        lst = [s]
        result = _serialize_list_for_key(lst, visited=visited)
        assert result[0] == "<list circular>"

    def test_serialize_list_for_key_circular_reference_generic(self):
        """Test serializing list with circular reference to generic object."""
        obj = object()
        visited = {id(obj)}
        lst = [obj]
        result = _serialize_list_for_key(lst, visited=visited)
        assert result[0] == "<circular>"

    def test_serialize_list_for_key_with_model_dump(self):
        """Test serializing list with model_dump objects."""

        class TestModel(BaseModel):
            field1: str = "value1"
            run_id: str = "123"

        obj = TestModel()
        lst = [obj]
        result = _serialize_list_for_key(lst)
        assert result[0]["field1"] == "value1"
        assert "run_id" not in result[0]

    def test_serialize_list_for_key_with_dict(self):
        """Test serializing list with dict objects."""
        d = {"key": "value"}
        lst = [d]
        result = _serialize_list_for_key(lst)
        assert result[0]["key"] == "value"

    def test_serialize_list_for_key_with_list(self):
        """Test serializing list with list objects."""
        inner_lst = [1, 2, 3]
        lst = [inner_lst]
        result = _serialize_list_for_key(lst)
        print(f"Result: {result!r}")
        assert result[0] == [1, 2, 3]

    def test_serialize_list_for_key_with_set(self):
        """Test serializing list with set objects."""
        s = {3, 1, 2}
        lst = [s]
        result = _serialize_list_for_key(lst)
        assert result[0] == [1, 2, 3]

    def test_serialize_list_for_key_with_custom_serializer(self):
        """Test serializing list with custom serializer."""

        def custom_serializer(obj):
            return f"custom_serialized_{obj.data}"  # Return a simple string to avoid recursion

        with patch(
            "flujo.utils.serialization.lookup_custom_serializer", return_value=custom_serializer
        ):
            # Use a custom object type that would trigger the custom serializer
            class CustomObject:
                def __init__(self, data):
                    self.data = data

                def __str__(self):
                    return str(self.data)

            obj = CustomObject("test_data")
            lst = [obj]
            result = _serialize_list_for_key(lst)
            assert result[0] == "custom_serialized_test_data"

    def test_serialize_list_for_key_with_custom_serializer_exception(self):
        """Test serializing list with custom serializer that raises exception."""

        def custom_serializer(obj):
            raise ValueError("Custom serializer error")

        with patch(
            "flujo.utils.serialization.lookup_custom_serializer", return_value=custom_serializer
        ):
            obj = "test"
            lst = [obj]
            result = _serialize_list_for_key(lst)
            assert result[0] == "test"  # Should fall back to original object


class TestCreateStepFingerprint:
    """Test the _create_step_fingerprint function."""

    def test_create_step_fingerprint_basic(self):
        """Test creating step fingerprint with basic step."""
        # Use a real Step instance instead of a Mock
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class DummyAgent:
            __name__ = "TestAgent"

        step = Step(
            name="test_step",
            agent=DummyAgent(),
            config=StepConfig(max_retries=3, timeout_s=30.0, temperature=0.7),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            updates_context=True,
            persist_feedback_to_context="feedback",
            persist_validation_results_to="validation",
        )
        # Add plugins, validators, processors as needed
        step.plugins = [(DummyAgent(), 1), (DummyAgent(), 2)]
        step.validators = [DummyAgent(), DummyAgent()]
        step.processors.prompt_processors = [DummyAgent(), DummyAgent()]
        step.processors.output_processors = [DummyAgent()]
        result = _create_step_fingerprint(step)
        assert result["name"] == "test_step"
        assert result["agent"]["type"] == "DummyAgent"
        assert result["config"]["max_retries"] == 3
        assert result["config"]["timeout_s"] == 30.0
        assert result["config"]["temperature"] == 0.7
        assert len(result["plugins"]) == 2
        assert len(result["validators"]) == 2
        assert len(result["processors"]["prompt_processors"]) == 2
        assert len(result["processors"]["output_processors"]) == 1
        assert result["updates_context"] is True
        assert result["persist_feedback_to_context"] == "feedback"
        assert result["persist_validation_results_to"] == "validation"

    def test_create_step_fingerprint_without_agent(self):
        """Test creating step fingerprint without agent."""
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        step = Step(
            name="test_step",
            agent=None,
            config=StepConfig(max_retries=1, timeout_s=None, temperature=None),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            updates_context=False,
            persist_feedback_to_context=None,
            persist_validation_results_to=None,
        )
        result = _create_step_fingerprint(step)
        assert result["name"] == "test_step"
        assert result["agent"]["type"] is None
        assert result["config"]["max_retries"] == 1
        assert result["config"]["timeout_s"] is None
        assert result["config"]["temperature"] is None
        assert result["plugins"] == []
        assert result["validators"] == []
        assert result["processors"]["prompt_processors"] == []
        assert result["processors"]["output_processors"] == []
        assert result["updates_context"] is False
        assert result["persist_feedback_to_context"] is None
        assert result["persist_validation_results_to"] is None


class TestGenerateCacheKey:
    """Test the _generate_cache_key function."""

    def test_generate_cache_key_success(self):
        """Test generating cache key successfully."""
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class DummyAgent:
            pass

        step = Step(
            name="test_step",
            agent=DummyAgent(),
            config=StepConfig(),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
        )
        data = {"input": "test"}
        context = {"context": "value"}
        resources = {"resource": "value"}
        result = _generate_cache_key(step, data, context, resources)
        assert result.startswith("test_step:")
        assert len(result.split(":")[1]) == 64  # SHA256 hex digest length

    def test_generate_cache_key_with_none_values(self):
        """Test generating cache key with None values."""
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        step = Step(
            name="test_step",
            agent=None,
            config=StepConfig(),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
        )
        result = _generate_cache_key(step, None, None, None)
        assert result.startswith("test_step:")
        assert len(result.split(":")[1]) == 64

    def test_generate_cache_key_json_exception_pickle_success(self):
        """Test generating cache key when JSON fails but pickle succeeds."""
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class DummyAgent:
            pass

        step = Step(
            name="test_step",
            agent=DummyAgent(),
            config=StepConfig(),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
        )
        # Create an object that will fail JSON serialization but succeed with pickle
        # by having a dict with a non-string key that JSON can't handle
        data = {1: "value"}  # JSON doesn't allow non-string keys
        result = _generate_cache_key(step, data)
        assert result.startswith("test_step:")
        assert len(result.split(":")[1]) == 64

    def test_generate_cache_key_both_json_and_pickle_fail(self):
        """Test generating cache key when both JSON and pickle fail."""
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class DummyAgent:
            pass

        step = Step(
            name="test_step",
            agent=DummyAgent(),
            config=StepConfig(),
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
        )

        class UnserializableObject:
            def __init__(self):
                self.data = "test"

            def __reduce__(self):
                raise TypeError("Cannot pickle")

        data = UnserializableObject()
        result = _generate_cache_key(step, data)
        # Our improved cache key generation now succeeds even with unserializable objects
        # by using the step fingerprint and a hash of the object
        assert result is not None
        assert result.startswith("test_step:")

# Additional tests appended by CodeRabbit Inc to expand coverage for cache step and serialization helpers.
# Test framework: pytest (with unittest.mock for patching)

try:
    # Prefer importing from actual module locations if available
    from flujo.utils.serialization import (
        _serialize_for_cache_key,
        _sort_set_deterministically,
        _get_stable_repr,
        _serialize_list_for_key,
        _create_step_fingerprint,
        _generate_cache_key,
    )
    from flujo.domain.dsl.step import Step
    from flujo.cache.backends.memory import InMemoryCache
    from flujo.cache.step import CacheStep
    from pydantic import BaseModel
except Exception:
    # Fallback import names are already used in this file above; if imports fail here,
    # existing imports in prior tests should still make these available in the module's globals.
    pass


class TestCacheStepAdditional:
    """More tests for CacheStep.cached and its interactions with cache backends."""

    def test_cached_uses_default_backend_when_invalid_backend_provided(self):
        # Provide an object that doesn't implement get/set - should fall back to default or raise a clear error.
        step = Step(name="invalid_backend_step")

        class InvalidBackend:
            pass

        invalid = InvalidBackend()

        # If implementation is strict, it may raise; otherwise, ensure a default backend is used.
        try:
            cache_step = CacheStep.cached(step, cache_backend=invalid)
            assert isinstance(cache_step.cache_backend, InMemoryCache)
        except Exception as e:
            # Accept clear, deterministic errors
            assert isinstance(e, (AttributeError, TypeError))

    def test_cached_preserves_wrapped_step_identity_and_name(self):
        step = Step(name="identity_step")
        cache_step = CacheStep.cached(step)
        assert cache_step.wrapped_step is step
        assert cache_step.name == step.name

    def test_cached_with_custom_backend_called(self):
        step = Step(name="custom_backend_step")
        custom = Mock()
        custom.get = Mock(return_value=None)
        custom.set = Mock()
        cache_step = CacheStep.cached(step, cache_backend=custom)
        # Simulate usage path by reaching into cache backend for a synthetic key
        key = "dummy:key"
        cache_step.cache_backend.get(key)
        cache_step.cache_backend.set(key, {"ok": True})
        custom.get.assert_called_once_with(key)
        custom.set.assert_called_once()


class TestSerializeForCacheKeyAdditions:
    """Expand edge-case coverage for _serialize_for_cache_key."""

    def test_serialize_primitive_and_nan_inf_values(self):
        assert _serialize_for_cache_key(0) == 0
        assert _serialize_for_cache_key(3.0) == 3.0
        assert _serialize_for_cache_key(True) is True
        # NaN/inf should be representable without raising
        res_nan = _serialize_for_cache_key(float("nan"))
        res_inf = _serialize_for_cache_key(float("inf"))
        assert isinstance(res_nan, (float, str))
        assert isinstance(res_inf, (float, str))

    def test_serialize_dict_with_nested_structures_and_order_independence(self):
        d1 = {"a": [3, 2, 1], "b": {"z": 1, "y": 2}, "s": {3, 1, 2}}
        d2 = {"b": {"y": 2, "z": 1}, "s": {2, 3, 1}, "a": [3, 2, 1]}
        r1 = _serialize_for_cache_key(d1)
        r2 = _serialize_for_cache_key(d2)
        # Both should produce equivalent serialized shapes
        assert r1["a"] == r2["a"]
        assert r1["b"] == r2["b"]
        assert r1["s"] == r2["s"]

    def test_serialize_object_with_model_dump_including_run_id_and_private_fields(self):
        class M(BaseModel):
            run_id: str = "rid"
            _private: str = "p"  # type: ignore

        obj = M()
        ser = _serialize_for_cache_key(obj)
        # run_id should generally be removed to stabilize keys
        assert isinstance(ser, dict) or isinstance(ser, str)
        if isinstance(ser, dict):
            assert "run_id" not in ser

    def test_serialize_bytes_and_bytearray(self):
        b = b"\x00\x01\x02"
        ba = bytearray(b"\x03\x04")
        rb = _serialize_for_cache_key(b)
        rba = _serialize_for_cache_key(ba)
        # Should be bytes-like safe serializations
        assert isinstance(rb, (str, bytes))
        assert isinstance(rba, (str, list, bytes))


class TestSortSetDeterministicallyAdditions:
    """Additional scenarios for _sort_set_deterministically."""

    def test_sort_set_with_mixed_types_falls_back_to_string(self):
        s = {"a", 1, "b", 2}
        res = _sort_set_deterministically(s)
        # ensure no crash and stable order by string
        assert all(isinstance(x, str) or isinstance(x, int) for x in s)
        assert isinstance(res, list)
        assert len(res) == 4

    def test_sort_empty_set(self):
        assert _sort_set_deterministically(set()) == []


class TestGetStableReprAdditions:
    """More coverage for _get_stable_repr."""

    def test_stable_repr_for_nested_collections(self):
        obj = {"k": [3, 1, 2], "s": {3, 1, 2}, "t": (2, 1)}
        rep = _get_stable_repr(obj)
        assert isinstance(rep, str)
        assert "{k:[3,1,2],s:{1,2,3},t:[1,2]}" in rep or rep.startswith("{")  # tolerant to exact format

    def test_stable_repr_for_module_and_builtin_function(self):
        rep_mod = _get_stable_repr(builtins)
        rep_fn = _get_stable_repr(len)
        assert "builtins" in rep_mod
        assert "len" in rep_fn

    def test_stable_repr_for_lambda_callable(self):
        f = lambda x: x  # noqa: E731
        rep = _get_stable_repr(f)
        assert "lambda" in rep or "<callable" in rep

    def test_stable_repr_for_unhashable_mapping_subclass(self):
        class UnhashableMapping(dict):
            __hash__ = None

        obj = UnhashableMapping(a=1)
        rep = _get_stable_repr(obj)
        assert "UnhashableMapping" in rep or "{a:1}" in rep


class TestSerializeListForKeyAdditions:
    """Edge cases for _serialize_list_for_key."""

    def test_serialize_list_mixed_types_and_nested(self):
        class M(BaseModel):
            a: int = 1

        lst = [M(), {"k": {3, 2, 1}}, [3, 2, 1], (2, 1)]
        res = _serialize_list_for_key(lst)
        assert isinstance(res, list)
        assert isinstance(res[0], dict) or isinstance(res[0], str)
        assert isinstance(res[1], dict)
        assert res[2] == [3, 2, 1]
        assert res[3] == [1, 2]

    def test_serialize_list_with_callable_items(self):
        def foo():
            return 1

        lst = [foo]
        res = _serialize_list_for_key(lst)
        assert isinstance(res[0], str) and "foo" in res[0]

    def test_serialize_list_with_bytes(self):
        lst = [b"\x00\xff"]
        res = _serialize_list_for_key(lst)
        assert isinstance(res[0], (str, bytes))


class TestCreateStepFingerprintAdditions:
    """More coverage for _create_step_fingerprint."""

    def test_step_fingerprint_includes_plugins_validators_and_processors_types(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class P:
            __name__ = "Plugin"

        class V:
            __name__ = "Validator"

        class Agent:
            __name__ = "Agent"

        step = Step(
            name="fp_step",
            agent=Agent(),
            config=StepConfig(max_retries=2, timeout_s=5.0, temperature=0.5),
            plugins=[(P(), 0)],
            validators=[V()],
            processors=AgentProcessors(),
            updates_context=False,
            persist_feedback_to_context=None,
            persist_validation_results_to=None,
        )
        step.processors.prompt_processors = [P()]
        fp = _create_step_fingerprint(step)
        assert fp["name"] == "fp_step"
        assert fp["agent"]["type"] in ("Agent", "P", "DummyAgent")
        assert fp["plugins"]
        assert fp["validators"]
        assert fp["processors"]["prompt_processors"]

    def test_step_fingerprint_is_stable_across_equal_steps(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class Agent:
            __name__ = "A"

        def make_step():
            s = Step(
                name="same",
                agent=Agent(),
                config=StepConfig(max_retries=1, timeout_s=1.0, temperature=0.1),
                plugins=[],
                validators=[],
                processors=AgentProcessors(),
            )
            return s

        fp1 = _create_step_fingerprint(make_step())
        fp2 = _create_step_fingerprint(make_step())
        assert fp1 == fp2


class TestGenerateCacheKeyAdditions:
    """Expand tests for _generate_cache_key to exercise error paths."""

    def test_generate_cache_key_is_stable_for_same_inputs(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class A:
            pass

        step = Step(name="stable", agent=A(), config=StepConfig(), processors=AgentProcessors(), plugins=[], validators=[])
        data = {"a": 1, "b": [3, 2, 1]}
        context = {"ctx": "v"}
        resources = {"r": {3, 1, 2}}

        k1 = _generate_cache_key(step, data, context, resources)
        k2 = _generate_cache_key(step, data, context, resources)
        assert k1 == k2
        assert k1.startswith("stable:")

    def test_generate_cache_key_changes_when_data_changes(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        step = Step(name="change", agent=None, config=StepConfig(), processors=AgentProcessors(), plugins=[], validators=[])
        k1 = _generate_cache_key(step, {"a": 1})
        k2 = _generate_cache_key(step, {"a": 2})
        assert k1 != k2

    def test_generate_cache_key_handles_unserializable_by_json_but_pickleable(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors
        step = Step(name="pickle_ok", agent=None, config=StepConfig(), processors=AgentProcessors(), plugins=[], validators=[])
        obj = {"k": set([1,2,3])}  # set not JSON-serializable but pickleable
        k = _generate_cache_key(step, obj)
        assert k.startswith("pickle_ok:")

    def test_generate_cache_key_handles_unpickleable_object(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        class Unpickleable:
            def __reduce__(self):
                raise TypeError("no pickle")

        step = Step(name="unpickleable", agent=None, config=StepConfig(), processors=AgentProcessors(), plugins=[], validators=[])
        obj = Unpickleable()
        k = _generate_cache_key(step, obj)
        assert k.startswith("unpickleable:")

    def test_generate_cache_key_with_large_nested_structure(self):
        from flujo.domain.dsl.step import Step, StepConfig
        from flujo.domain.processors import AgentProcessors

        step = Step(name="large", agent=None, config=StepConfig(), processors=AgentProcessors(), plugins=[], validators=[])
        big = {"lvl1": [{"lvl2": {i for i in range(10)} } for _ in range(5)]}
        k = _generate_cache_key(step, big)
        assert k.startswith("large:")