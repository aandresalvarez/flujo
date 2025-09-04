from __future__ import annotations

import pytest
from pydantic import BaseModel

from flujo.utils.schema_utils import derive_json_schema_from_type
from flujo.processors.aros import SmartTypeCoercionProcessor


class Item(BaseModel):
    id: int
    active: bool


@pytest.mark.fast
def test_derive_json_schema_from_pydantic_model():
    schema = derive_json_schema_from_type(Item)
    assert schema is not None
    assert schema.get("type") == "object"
    assert "properties" in schema
    props = schema["properties"]
    assert "id" in props and props["id"].get("type") in {"integer", "number"}
    assert "active" in props and props["active"].get("type") == "boolean"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_smart_coercion_with_derived_schema():
    schema = derive_json_schema_from_type(Item)
    # Input with strings
    payload = {"id": "7", "active": "true"}
    proc = SmartTypeCoercionProcessor(
        allow={
            "integer": ["str->int"],
            "boolean": ["str->bool"],
        },
        schema=schema,
    )
    out = await proc.process(payload)
    assert out["id"] == 7 and isinstance(out["id"], int)
    assert out["active"] is True
