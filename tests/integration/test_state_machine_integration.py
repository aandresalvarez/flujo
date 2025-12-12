from __future__ import annotations

import asyncio
import pytest


from flujo.domain.models import PipelineContext as _PC


async def _set_next_state(_inp: str, *, context: _PC | None = None) -> str:
    try:
        if context is not None:
            context.next_state = "s2"
    except Exception:
        pass
    return "ok"


@pytest.mark.fast
def test_state_machine_dsl_executes_end_to_end() -> None:
    from flujo.domain.dsl import Pipeline
    from flujo.domain.dsl.state_machine import StateMachineStep
    from flujo.domain.dsl import Step
    from flujo.application.core.executor_core import ExecutorCore
    from flujo.domain.models import PipelineContext

    # Build two states where s1 transitions to s2, and s2 is terminal
    s1 = Pipeline.from_step(Step.from_callable(_set_next_state, name="SetNext"))
    s2 = Pipeline.from_step(Step.from_callable(lambda x: x, name="Done"))
    sm = StateMachineStep(
        name="SM",
        states={"s1": s1, "s2": s2},
        start_state="s1",
        end_states=["s2"],
    )
    pipe = Pipeline.from_step(sm)

    class Ctx(PipelineContext):
        pass

    ctx = Ctx(initial_prompt="")
    # Execute via core to ensure policy dispatch path
    core: ExecutorCore[Ctx] = ExecutorCore()

    async def _run() -> None:
        res = await core._execute_pipeline_via_policies(pipe, "hi", ctx, None, None, None)
        assert res is not None
        assert len(res.step_history) >= 1

    asyncio.run(_run())


@pytest.mark.fast
def test_state_machine_yaml_end_state_noop() -> None:
    from flujo.domain.dsl import Pipeline

    yaml_text = (
        'version: "0.1"\n'
        "steps:\n"
        "  - kind: StateMachine\n"
        "    name: SM\n"
        "    start_state: s1\n"
        "    end_states: [s1]\n"
        "    states:\n"
        "      s1: []\n"
    )
    p = Pipeline.from_yaml_text(yaml_text)
    assert p is not None
    assert len(p.steps) == 1
