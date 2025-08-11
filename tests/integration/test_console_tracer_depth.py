import pytest

from flujo.domain.dsl import Step
from flujo.domain.models import StepResult
from flujo.testing.utils import StubAgent
from flujo.infra.console_tracer import ConsoleTracer
from flujo.domain.events import PreStepPayload, PostStepPayload, OnStepFailurePayload


@pytest.mark.asyncio
async def test_console_tracer_indentation(monkeypatch):
    tracer = ConsoleTracer(level="info", colorized=False)
    titles: list[str] = []
    monkeypatch.setattr(tracer.console, "print", lambda panel: titles.append(panel.title))

    outer_step = Step.model_validate({"name": "outer"})
    inner_step = Step.model_validate({"name": "inner"})

    await tracer.hook(PreStepPayload(event_name="pre_step", step=outer_step, step_input=None))
    await tracer.hook(PreStepPayload(event_name="pre_step", step=inner_step, step_input=None))
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="inner")))
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="outer")))

    assert titles[0] == "Step Start: outer"
    assert titles[1].startswith("  Step Start: inner")
    assert titles[2].startswith("  Step End: inner")
    assert titles[3] == "Step End: outer"


@pytest.mark.asyncio
async def test_console_tracer_indentation_reset_on_failure(monkeypatch):
    tracer = ConsoleTracer(level="info", colorized=False)
    titles: list[str] = []
    monkeypatch.setattr(tracer.console, "print", lambda panel: titles.append(panel.title))

    fail_step = Step.model_validate({"name": "fail"})
    next_step = Step.model_validate({"name": "next"})

    await tracer.hook(PreStepPayload(event_name="pre_step", step=fail_step, step_input=None))
    await tracer.hook(
        OnStepFailurePayload(
            event_name="on_step_failure",
            step_result=StepResult(name="fail", success=False, feedback="bad"),
        )
    )
    await tracer.hook(PreStepPayload(event_name="pre_step", step=next_step, step_input=None))

    assert titles[0] == "Step Start: fail"
    assert titles[1] == "Step Failure: fail"
    assert titles[2] == "Step Start: next"


@pytest.mark.asyncio
async def test_console_tracer_pipeline_success(monkeypatch):
    tracer = ConsoleTracer(level="info", colorized=False)
    titles: list[str] = []
    monkeypatch.setattr(tracer.console, "print", lambda panel: titles.append(panel.title))

    pipeline = Step.model_validate(
        {"name": "s1", "agent": StubAgent(["ok"])}
    ) >> Step.model_validate({"name": "s2", "agent": StubAgent(["ok"])})

    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=pipeline.steps[0], step_input=None)
    )
    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=pipeline.steps[1], step_input=None)
    )
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="s2")))
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="s1")))

    assert titles[0] == "Step Start: s1"
    assert titles[1].startswith("  Step Start: s2")
    assert titles[2].startswith("  Step End: s2")
    assert titles[3] == "Step End: s1"


@pytest.mark.asyncio
async def test_console_tracer_pipeline_failure(monkeypatch):
    tracer = ConsoleTracer(level="info", colorized=False)
    titles: list[str] = []
    monkeypatch.setattr(tracer.console, "print", lambda panel: titles.append(panel.title))

    pipeline = Step.model_validate(
        {"name": "s1", "agent": StubAgent(["ok"])}
    ) >> Step.model_validate({"name": "s2", "agent": StubAgent(["fail"])})

    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=pipeline.steps[0], step_input=None)
    )
    await tracer.hook(
        PreStepPayload(event_name="pre_step", step=pipeline.steps[1], step_input=None)
    )
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="s2")))
    await tracer.hook(PostStepPayload(event_name="post_step", step_result=StepResult(name="s1")))

    assert titles[0] == "Step Start: s1"
    assert titles[1].startswith("  Step Start: s2")
    assert titles[2].startswith("  Step End: s2")
    assert titles[3] == "Step End: s1"
