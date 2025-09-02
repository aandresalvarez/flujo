from __future__ import annotations

from flujo.utils.expressions import compile_expression_to_callable


def test_boolean_literals_lowercase_true_false_and_null() -> None:
    fn_true = compile_expression_to_callable("true")
    fn_false = compile_expression_to_callable("false")
    fn_null = compile_expression_to_callable("null")

    assert fn_true(None, None) is True
    assert fn_false(None, None) is False
    assert fn_null(None, None) is None


def test_boolean_literals_in_comparison() -> None:
    fn = compile_expression_to_callable("context.scratchpad.get('flag') == true")

    class Ctx:
        def __init__(self) -> None:
            self.scratchpad = {"flag": True}

    assert fn(None, Ctx()) is True
