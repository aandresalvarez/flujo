from __future__ import annotations

import pytest

from flujo.processors.aros import SmartTypeCoercionProcessor
from flujo.tracing.manager import TraceManager, set_active_trace_manager


@pytest.mark.fast
@pytest.mark.asyncio
async def test_anyof_branch_selection_and_branch_choices_recorded():
    # anyOf: integer or boolean; input "1" should coerce to integer (first branch)
    schema = {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}
    proc = SmartTypeCoercionProcessor(
        allow={"integer": ["str->int"], "boolean": ["str->bool"]}, schema=schema
    )

    # Enable tracing to capture coercion event
    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    out = await proc.process("1")
    assert out == 1
    # Inspect last event
    cur = tm._span_stack[-1]
    ev = next((e for e in cur.events if e.get("name") == "output.coercion.success"), None)
    assert ev is not None
    attrs = ev.get("attributes", {})
    # Branch choices should indicate root ("/") index 0
    bc = attrs.get("branch_choices", {})
    assert isinstance(bc, dict)
    assert bc.get("/") == 0
