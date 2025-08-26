from __future__ import annotations

import pytest

from flujo.utils.expressions import compile_expression_to_callable


class _Ctx:
    def __init__(self, scratchpad: dict | None = None) -> None:
        self.scratchpad = scratchpad or {}


def test_allowlisted_dict_get_with_default() -> None:
    expr = compile_expression_to_callable("context.scratchpad.get('missing', 'fallback')")
    ctx = _Ctx({"present": 1})
    out = expr(output=None, context=ctx)
    assert out == "fallback"


def test_allowlisted_string_methods() -> None:
    expr = compile_expression_to_callable("previous_step.message.lower().startswith('hi')")
    prev = {"message": "Hi There"}
    ctx = _Ctx()
    out = expr(prev, ctx)
    assert out is True

    expr2 = compile_expression_to_callable("previous_step.message.strip()")
    prev2 = {"message": "  x  "}
    assert expr2(prev2, ctx) == "x"


def test_disallow_mutating_or_unknown_calls() -> None:
    # pop is not allow-listed
    with pytest.raises(ValueError, match="Unsupported expression element"):
        compile_expression_to_callable("context.scratchpad.pop('k')")(None, _Ctx())

    # bare function calls not allowed
    with pytest.raises(ValueError, match="Unsupported expression element"):
        compile_expression_to_callable("os.system('echo 1')")(None, _Ctx())
