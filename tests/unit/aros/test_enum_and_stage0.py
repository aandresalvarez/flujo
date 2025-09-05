from __future__ import annotations

import json
import pytest

from flujo.processors.aros import (
    SmartTypeCoercionProcessor,
    JsonRegionExtractorProcessor,
    TolerantJsonDecoderProcessor,
)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_enum_integer_coercion_from_string():
    schema = {"enum": [1, 2, 3]}
    proc = SmartTypeCoercionProcessor(allow={"integer": ["str->int"]}, schema=schema)
    out = await proc.process("2")
    assert out == 2


@pytest.mark.fast
@pytest.mark.asyncio
async def test_enum_boolean_coercion_from_string():
    schema = {"enum": [True, False]}
    proc = SmartTypeCoercionProcessor(allow={"boolean": ["str->bool"]}, schema=schema)
    out = await proc.process("1")
    assert out is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_stage0_extractor_code_fence():
    raw = """
    Some text
    ```json
    {"a": 1, "b": [2,3]}
    ```
    trailing
    """
    proc = JsonRegionExtractorProcessor()
    candidate = await proc.process(raw)
    data = json.loads(candidate)
    assert data == {"a": 1, "b": [2, 3]}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_stage0_unescape_double_encoded():
    raw = '"{\\"x\\": 5}"'
    proc = JsonRegionExtractorProcessor(max_unescape_depth=2)
    candidate = await proc.process(raw)
    # Now decode with tolerant decoder fast path
    dec = TolerantJsonDecoderProcessor()
    obj = await dec.process(candidate)
    assert obj == {"x": 5}
