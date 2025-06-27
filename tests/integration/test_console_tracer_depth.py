import pytest
from flujo.tracing import ConsoleTracer
from flujo.domain.events import PreStepPayload, PostStepPayload
from flujo.domain.pipeline_dsl import Step
from flujo.domain.models import StepResult


@pytest.mark.asyncio
async def test_console_tracer_indentation(monkeypatch):
    tracer = ConsoleTracer(level="info", colorized=False)
    titles: list[str] = []
    monkeypatch.setattr(
        tracer.console, "print", lambda panel: titles.append(panel.title)
    )

    outer_step = Step("outer")
    inner_step = Step("inner")

    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=outer_step, step_input=None)
    )
    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=inner_step, step_input=None)
    )
    await tracer.hook(
        PostStepPayload(event_name="post_step", step_result=StepResult(name="inner"))
    )
    await tracer.hook(
        PostStepPayload(event_name="post_step", step_result=StepResult(name="outer"))
    )

    assert titles[0] == "Step Start: outer"
    assert titles[1].startswith("  Step Start: inner")
    assert titles[2].startswith("  Step End: inner")
    assert titles[3] == "Step End: outer"
