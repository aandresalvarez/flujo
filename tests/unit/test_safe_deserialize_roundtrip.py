import dataclasses
from flujo.type_definitions.common import JSONObject

import pytest

from flujo.utils.serialization import (
    _serialize_for_json,
    safe_deserialize,
    register_custom_serializer,
    register_custom_deserializer,
    reset_custom_serializer_registry,
)


class CustomType:
    def __init__(self, value: int) -> None:
        self.value = value

    def to_dict(self) -> JSONObject:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: JSONObject) -> "CustomType":
        return cls(int(data["value"]))


@dataclasses.dataclass
class DataExample:
    num: int
    text: str


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    reset_custom_serializer_registry()


def test_default_behavior_returns_serialized_data() -> None:
    obj = DataExample(1, "x")
    set = _serialize_for_json(obj)
    assert isinstance(set, dict)
    assert safe_deserialize(set) == set


def test_custom_type_roundtrip_with_registry() -> None:
    item = CustomType(42)
    register_custom_serializer(CustomType, lambda x: x.to_dict())
    register_custom_deserializer(CustomType, CustomType.from_dict)
    serialized = _serialize_for_json(item)
    result = safe_deserialize(serialized, CustomType)
    assert isinstance(result, CustomType)
    assert result.value == 42
