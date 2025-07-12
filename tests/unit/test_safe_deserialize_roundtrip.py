import dataclasses
from typing import Any, Dict

import pytest

from flujo.utils.serialization import (
    safe_serialize,
    safe_deserialize,
    register_custom_serializer,
    register_custom_deserializer,
    reset_custom_serializer_registry,
)


class CustomType:
    def __init__(self, value: int) -> None:
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomType":
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
    ser = safe_serialize(obj)
    assert isinstance(ser, dict)
    assert safe_deserialize(ser) == ser


def test_custom_type_roundtrip_with_registry() -> None:
    item = CustomType(42)
    register_custom_serializer(CustomType, lambda x: x.to_dict())
    register_custom_deserializer(CustomType, CustomType.from_dict)
    serialized = safe_serialize(item)
    result = safe_deserialize(serialized, CustomType)
    assert isinstance(result, CustomType)
    assert result.value == 42
