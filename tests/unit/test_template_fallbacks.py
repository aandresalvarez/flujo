from __future__ import annotations

from flujo.utils.prompting import AdvancedPromptFormatter
from flujo.utils.template_vars import TemplateContextProxy, StepValueProxy
from pydantic import BaseModel as PydanticModel


class Ctx(PydanticModel):
    initial_prompt: str | None = None


def test_fallback_literal_is_used_when_primary_missing() -> None:
    fmt = AdvancedPromptFormatter("{{ context.initial_prompt or 'Fallback: No prompt provided' }}")
    ctx = Ctx(initial_prompt=None)
    proxy = TemplateContextProxy(ctx, steps={})
    out = fmt.format(context=proxy, steps={}, previous_step=None)
    assert out == "Fallback: No prompt provided"


def test_fallback_literal_is_used_when_primary_empty_string() -> None:
    fmt = AdvancedPromptFormatter("{{ context.initial_prompt or 'fallback' }}")
    ctx = Ctx(initial_prompt="")
    proxy = TemplateContextProxy(ctx, steps={})
    out = fmt.format(context=proxy, steps={}, previous_step=None)
    assert out == "fallback"


def test_steps_map_fallback_works() -> None:
    fmt = AdvancedPromptFormatter("{{ steps.check_prompt or 'Fallback' }}")
    steps = {"check_prompt": StepValueProxy("")}
    out = fmt.format(
        context=TemplateContextProxy(Ctx(), steps=steps), steps=steps, previous_step=None
    )
    assert out == "Fallback"


def test_chained_fallback_selects_first_truthy() -> None:
    fmt = AdvancedPromptFormatter("{{ context.a or context.b or 'Z' }}")

    class M(PydanticModel):
        a: str | None = None
        b: str | None = None

    proxy = TemplateContextProxy(M(a=None, b="B"), steps={})
    out = fmt.format(context=proxy, steps={}, previous_step=None)
    assert out == "B"
