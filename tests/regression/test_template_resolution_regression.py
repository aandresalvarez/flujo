from __future__ import annotations

from flujo.utils.template_vars import render_template


def test_bracket_indexing_is_not_supported_renders_empty() -> None:
    tpl = "{{ steps['step1'] }} + {{ steps[0] }}"
    out = render_template(tpl, steps={"step1": "Hello"})
    assert out == " + "


def test_unknown_variable_renders_empty() -> None:
    out = render_template("{{ steps.unknown }}", steps={"step1": "Hello"})
    assert out == ""


def test_mixed_supported_and_unsupported_tokens() -> None:
    tpl = "{{ steps.step1 }} {{ steps['step1'] }}"
    out = render_template(tpl, steps={"step1": "Hello"})
    assert out == "Hello "
