from __future__ import annotations

import pytest

from flujo.tracing.manager import TraceManager
from flujo.grammars.adapters import compile_outlines_regex, compile_xgrammar


class DummySpan:
    def __init__(self) -> None:
        self.events = []
        self.attributes = {}


@pytest.mark.fast
def test_tracing_aggregation_for_coercion_and_soe_and_reasoning():
    tm = TraceManager()
    # Push a dummy current span
    tm._span_stack = [DummySpan()]

    # Coercion success (semantic)
    tm.add_event(
        "output.coercion.success",
        {"stage": "semantic", "transforms": ["str->int", "json.loads"]},
    )
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.coercion.total") == 1
    assert cur.attributes.get("aros.coercion.stage.semantic") == 1
    xforms = set(cur.attributes.get("aros.coercion.transforms", []))
    assert "str->int" in xforms and "json.loads" in xforms

    # Grammar applied and skipped
    tm.add_event("grammar.applied", {"mode": "openai_json", "schema_hash": None})
    assert cur.attributes.get("aros.soe.count") == 1
    tm.add_event("aros.soe.skipped", {"reason": "unsupported_provider", "mode": "auto"})
    assert cur.attributes.get("aros.soe.skipped") == 1
    assert cur.attributes.get("aros.soe.skipped.unsupported_provider") == 1

    # Reasoning validation
    tm.add_event("reasoning.validation", {"result": "fail", "score": 0.3})
    tm.add_event("reasoning.validation", {"result": "pass", "score": 0.9})
    assert cur.attributes.get("aros.precheck.total") == 2
    assert cur.attributes.get("aros.precheck.fail") == 1
    assert cur.attributes.get("aros.precheck.pass") == 1


@pytest.mark.fast
def test_grammar_adapters_stubs_return_placeholders():
    obj_schema = {"type": "object"}
    arr_schema = {"type": "array"}
    assert "{" in compile_outlines_regex(obj_schema)
    assert "[" in compile_outlines_regex(arr_schema)
    assert "object" in compile_xgrammar(obj_schema)
    assert "array" in compile_xgrammar(arr_schema)
