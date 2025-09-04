from __future__ import annotations

import pytest

from flujo.processors.aros import SmartTypeCoercionProcessor


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_object_integer():
    data = {"count": "42"}
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    proc = SmartTypeCoercionProcessor(
        allow={"integer": ["str->int"]}, schema=schema, anyof_strategy="first-pass"
    )
    out = await proc.process(data)
    assert isinstance(out["count"], int)
    assert out["count"] == 42


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_anyof_boolean_string_prefers_first_branch_when_allowed():
    data = "true"
    schema = {"anyOf": [{"type": "boolean"}, {"type": "string"}]}
    proc = SmartTypeCoercionProcessor(
        allow={"boolean": ["str->bool"]}, schema=schema, anyof_strategy="first-pass"
    )
    out = await proc.process(data)
    # Prefer the first anyOf branch (boolean) when allowed, even if string is valid
    assert out is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_array_from_string():
    data = "[1, 2, 3]"
    schema = {"type": "array", "items": {"type": "integer"}}
    proc = SmartTypeCoercionProcessor(
        allow={"array": ["str->array"], "integer": ["str->int"]}, schema=schema
    )
    out = await proc.process(data)
    assert isinstance(out, list)
    assert out == [1, 2, 3]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_nested_array_of_objects():
    data = {
        "items": [
            {"id": "1", "enabled": "true"},
            {"id": "2", "enabled": "false"},
        ]
    }
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["id", "enabled"],
                },
            }
        },
        "required": ["items"],
    }
    proc = SmartTypeCoercionProcessor(
        allow={"integer": ["str->int"], "boolean": ["str->bool"]}, schema=schema
    )
    out = await proc.process(data)
    assert isinstance(out["items"], list)
    assert out["items"][0]["id"] == 1 and out["items"][0]["enabled"] is True
    assert out["items"][1]["id"] == 2 and out["items"][1]["enabled"] is False


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_oneof_integer_or_string_prefers_integer_when_allowed():
    data = "7"
    schema = {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    proc = SmartTypeCoercionProcessor(
        allow={"integer": ["str->int"]}, schema=schema, anyof_strategy="first-pass"
    )
    out = await proc.process(data)
    assert isinstance(out, int) and out == 7


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_array_items_integer_from_list_of_strings():
    data = ["1", "2", "3"]
    schema = {"type": "array", "items": {"type": "integer"}}
    proc = SmartTypeCoercionProcessor(allow={"integer": ["str->int"]}, schema=schema)
    out = await proc.process(data)
    assert out == [1, 2, 3]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_coercion_nullable_integer_none_passthrough():
    data = None
    schema = {"oneOf": [{"type": "null"}, {"type": "integer"}]}
    proc = SmartTypeCoercionProcessor(allow={"integer": ["str->int"]}, schema=schema)
    out = await proc.process(data)
    assert out is None


@pytest.mark.fast
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "val,expected", [("0", False), ("1", True), ("true", True), ("false", False)]
)
async def test_schema_coercion_boolean_zero_one_and_text(val: str, expected: bool):
    schema = {"type": "boolean"}
    proc = SmartTypeCoercionProcessor(allow={"boolean": ["str->bool"]}, schema=schema)
    out = await proc.process(val)
    assert out is expected


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_anyof_array_or_integer_prefers_array_when_string_json():
    data = "[5,6]"
    schema = {"anyOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "integer"}]}
    proc = SmartTypeCoercionProcessor(
        allow={"array": ["str->array"], "integer": ["str->int"]}, schema=schema
    )
    out = await proc.process(data)
    assert out == [5, 6]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_nested_anyof_in_object_property():
    data = {"v": "[3,4]"}
    schema = {
        "type": "object",
        "properties": {
            "v": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "integer"}}]}
        },
        "required": ["v"],
    }
    proc = SmartTypeCoercionProcessor(
        allow={"array": ["str->array"], "integer": ["str->int"]}, schema=schema
    )
    out = await proc.process(data)
    assert out["v"] == [3, 4]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_schema_oneof_array_or_string_keeps_array_when_already_list():
    data = [1, 2]
    schema = {"oneOf": [{"type": "array", "items": {"type": "integer"}}, {"type": "string"}]}
    proc = SmartTypeCoercionProcessor(schema=schema)
    out = await proc.process(data)
    assert out == [1, 2]
