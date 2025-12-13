from __future__ import annotations

import pytest

from flujo.utils.expressions import compile_expression_to_callable


class _Ctx:
    def __init__(self, step_outputs: dict | None = None) -> None:
        self.step_outputs = step_outputs or {}


def test_allowlisted_dict_get_with_default() -> None:
    expr = compile_expression_to_callable("context.step_outputs.get('missing', 'fallback')")
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
        compile_expression_to_callable("context.step_outputs.pop('k')")(None, _Ctx())

    # bare function calls not allowed (may surface as Unknown name or Unsupported expression)
    with pytest.raises(ValueError, match="(Unsupported expression element|Unknown name)"):
        compile_expression_to_callable("os.system('echo 1')")(None, _Ctx())


def test_nested_attribute_and_subscript_and_none_tolerance() -> None:
    # Use boolean conditions with nested access; avoid IfExp (not supported)
    expr = compile_expression_to_callable(
        "context.step_outputs.get('user') and context.step_outputs.user['name'].lower().startswith('ali')"
    )
    ctx = _Ctx({"user": {"name": "ALICE"}})
    assert expr(None, ctx) is True
    ctx2 = _Ctx({})
    # When 'user' missing, whole expression is falsy
    assert not bool(expr(None, ctx2))


def test_invalid_arg_types_to_allowlisted_methods_raise() -> None:
    # dict.get with non-string key should raise per sandbox rules
    with pytest.raises(ValueError, match="Unsupported expression element"):
        compile_expression_to_callable("context.step_outputs.get(123)")(None, _Ctx())

    # startswith requires a string
    with pytest.raises(ValueError, match="Unsupported expression element"):
        compile_expression_to_callable("previous_step.message.startswith(10)")(
            {"message": "x"}, _Ctx()
        )
