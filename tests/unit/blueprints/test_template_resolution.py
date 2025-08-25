from __future__ import annotations

from flujo.utils.template_vars import render_template


def test_dotted_steps_resolution() -> None:
    steps = {"step1": "Hello World"}
    out = render_template("{{ steps.step1 }} - Processed", steps=steps)
    assert out == "Hello World - Processed"


def test_aliases_output_result_value() -> None:
    steps = {"step1": "Hello World"}
    tpl = "A: {{ steps.step1 }}, B: {{ steps.step1.output }}, C: {{ steps.step1.result }}, D: {{ steps.step1.value }}"
    out = render_template(tpl, steps=steps)
    assert out == "A: Hello World, B: Hello World, C: Hello World, D: Hello World"


def test_context_proxy_fallback_to_steps() -> None:
    steps = {"step1": "Hello World"}
    out = render_template("{{ context.step1 }}", context={}, steps=steps)
    assert out == "Hello World"


def test_context_base_overrides_steps() -> None:
    out = render_template(
        "{{ context.step1 }}",
        context={"step1": "FromContext"},
        steps={"step1": "FromSteps"},
    )
    assert out == "FromContext"


def test_previous_step_variable() -> None:
    out = render_template("Prev: {{ previous_step }}", previous_step="Hello World")
    assert out == "Prev: Hello World"
