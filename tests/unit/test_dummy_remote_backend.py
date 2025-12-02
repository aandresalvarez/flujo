import pytest
from typing import Optional
from flujo.domain.models import BaseModel

from flujo import Step
from flujo.testing.utils import DummyRemoteBackend, gather_result
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    count: int = 0


class IncrementAgent:
    async def run(self, data: int, *, context: Optional[Ctx] = None) -> int:
        if context is not None:
            context.count += 1
        return data + 1


class Nested(BaseModel):
    foo: str
    bar: int


class Container(BaseModel):
    nested: Nested
    items: list[int]


class EchoAgent:
    async def run(self, data: Container) -> Container:
        return data


class ComplexNested(BaseModel):
    name: str
    metadata: dict[str, str]


class DeepContainer(BaseModel):
    level1: ComplexNested
    level2: list[ComplexNested]
    level3: dict[str, ComplexNested]


class DeepEchoAgent:
    async def run(self, data: DeepContainer) -> DeepContainer:
        return data


class ListContainer(BaseModel):
    simple_list: list[int]
    nested_list: list[Nested]
    mixed_list: list[dict[str, str]]


class ListEchoAgent:
    async def run(self, data: ListContainer) -> ListContainer:
        return data


class DictContainer(BaseModel):
    simple_dict: dict[str, int]
    nested_dict: dict[str, Nested]
    mixed_dict: dict[str, list[int]]

    def model_dump(self, **kwargs):
        """Override to use standard Pydantic serialization for this test case."""
        # Use parent's model_dump to avoid custom circular reference logic
        from pydantic import BaseModel as PydanticBaseModel

        return PydanticBaseModel.model_dump(self, **kwargs)


class DictEchoAgent:
    async def run(self, data: DictContainer) -> DictContainer:
        return data


@pytest.mark.asyncio
async def test_dummy_remote_backend_preserves_context() -> None:
    step1 = Step.model_validate({"name": "a", "agent": IncrementAgent()})
    step2 = Step.model_validate({"name": "b", "agent": IncrementAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(
        step1 >> step2,
        backend=backend,
        context_model=Ctx,
        initial_context_data={"count": 0},
    )
    result = await gather_result(runner, 1)
    assert backend.call_counter == 2
    assert isinstance(result.final_pipeline_context, Ctx)
    assert result.final_pipeline_context.count == 2


@pytest.mark.asyncio
async def test_dummy_remote_backend_roundtrip_complex_input() -> None:
    step = Step.model_validate({"name": "echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)
    payload = Container(nested=Nested(foo="hi", bar=42), items=[1, 2, 3])
    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, Container)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_simple_nested_structures() -> None:
    """Test reconstruction with simple nested structures that we know work."""
    step = Step.model_validate({"name": "simple_echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    # Test with the original working case
    payload = Container(nested=Nested(foo="test", bar=42), items=[1, 2, 3])

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, Container)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_list_of_primitives() -> None:
    """Test reconstruction with lists of primitive types."""
    step = Step.model_validate({"name": "list_echo", "agent": ListEchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    payload = ListContainer(
        simple_list=[1, 2, 3, 4, 5],
        nested_list=[
            Nested(foo="a", bar=1),
            Nested(foo="b", bar=2),
            Nested(foo="c", bar=3),
        ],
        mixed_list=[
            {"key1": "value1", "key2": "value2"},
            {"key3": "value3"},
        ],
    )

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, ListContainer)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_dict_of_primitives() -> None:
    """Test reconstruction with dictionaries of primitive types."""
    step = Step.model_validate({"name": "dict_echo", "agent": DictEchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    payload = DictContainer(
        simple_dict={"a": 1, "b": 2, "c": 3},
        nested_dict={
            "first": Nested(foo="first", bar=10),
            "second": Nested(foo="second", bar=20),
        },
        mixed_dict={
            "list1": [1, 2, 3],
            "list2": [4, 5, 6],
        },
    )

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, DictContainer)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_edge_cases() -> None:
    """Test reconstruction with edge cases like empty structures and None values."""
    step = Step.model_validate({"name": "edge_echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    # Test with empty lists and minimal data
    payload = Container(nested=Nested(foo="", bar=0), items=[])

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, Container)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_string_encoded_lists() -> None:
    """Test that string-encoded lists are properly reconstructed."""
    step = Step.model_validate({"name": "string_echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    # This tests the case where lists might be serialized as strings
    payload = Container(nested=Nested(foo="test", bar=42), items=[1, 2, 3])

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, Container)
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_preserves_types() -> None:
    """Test that the backend preserves exact types and doesn't convert between types."""
    step = Step.model_validate({"name": "type_echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    payload = Container(nested=Nested(foo="type_test", bar=100), items=[1, 2, 3])

    result = await gather_result(runner, payload)
    returned = result.step_history[0].output

    # Check that types are preserved exactly
    assert isinstance(returned, Container)
    assert isinstance(returned.nested, Nested)
    assert isinstance(returned.items, list)
    assert all(isinstance(item, int) for item in returned.items)

    # Check that the data is identical
    assert returned.model_dump() == payload.model_dump()


@pytest.mark.asyncio
async def test_dummy_remote_backend_multiple_roundtrips() -> None:
    """Test that multiple serialization/deserialization cycles work correctly."""
    step = Step.model_validate({"name": "multi_echo", "agent": EchoAgent()})
    backend = DummyRemoteBackend()
    runner = create_test_flujo(step, backend=backend)

    payload = Container(nested=Nested(foo="multi_test", bar=999), items=[10, 20, 30])

    # Run multiple times to ensure consistency
    for i in range(3):
        result = await gather_result(runner, payload)
        returned = result.step_history[0].output
        assert isinstance(returned, Container)
        assert returned.model_dump() == payload.model_dump()
        assert backend.call_counter == i + 1
