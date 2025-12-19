import pytest
from pydantic import BaseModel

from flujo.domain.dsl.cache_step import _generate_cache_key, _serialize_for_cache_key
from flujo.utils.serialization import (
    _serialize_for_json,
    register_custom_serializer,
    reset_custom_serializer_registry,
)
from tests.test_types.fixtures import create_test_step


class SelfRefModel(BaseModel):
    value: int
    next: "SelfRefModel | None" = None

    # model_config inherited from BaseModel


SelfRefModel.model_rebuild()


@pytest.fixture(autouse=True)
def clear_serializers():
    reset_custom_serializer_registry()
    yield
    reset_custom_serializer_registry()


def test_determinism_with_unordered_collections() -> None:
    step = create_test_step(name="determinism")
    data1 = {"a": 1, "b": 2, "set": {3, 1}}
    data2 = {"b": 2, "a": 1, "set": {1, 3}}
    key1 = _generate_cache_key(step, data1)
    key2 = _generate_cache_key(step, data2)
    assert key1 == key2


def test_circular_reference_handling() -> None:
    node = SelfRefModel(value=1)
    node.next = node
    serialized = _serialize_for_cache_key(node)
    assert serialized["next"] == f"<{SelfRefModel.__name__} circular>"


def test_custom_serializer_is_honored() -> None:
    def custom_int(value: int) -> str:
        return f"INT:{value}"

    register_custom_serializer(int, custom_int)
    result = _serialize_for_json({"num": 7})
    assert result["num"] == "INT:7"
