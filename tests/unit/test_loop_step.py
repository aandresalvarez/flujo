import pytest

from flujo.domain import Step, Pipeline
from flujo.domain.dsl import LoopStep
from flujo.domain.models import PipelineContext
from typing import Optional
from tests.conftest import create_test_flujo


class Ctx(PipelineContext):
    initial_prompt: str = "test"
    counter: int = 0


def test_loop_step_init_validation() -> None:
    with pytest.raises(ValueError):
        LoopStep.model_validate(
            {
                "name": "loop",
                "loop_body_pipeline": Pipeline.from_step(Step.model_validate({"name": "a"})),
                "exit_condition_callable": lambda *_: True,
                "max_loops": 0,
            }
        )


def test_step_factory_loop_until() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "a"}))
    step = Step.loop_until(
        name="loop", loop_body_pipeline=body, exit_condition_callable=lambda *_: True
    )
    assert isinstance(step, LoopStep)
    assert step.max_loops == 5


@pytest.mark.asyncio
async def test_loopstep_context_isolation_unit():
    class IncAgent:
        async def run(self, x: int, *, context: Optional[Ctx] = None) -> int:
            if context:
                context.counter += 1
            return x + 1

    body = Pipeline.from_step(Step.model_validate({"name": "inc", "agent": IncAgent()}))
    loop = Step.loop_until(
        name="loop_ctx_isolation_unit",
        loop_body_pipeline=body,
        exit_condition_callable=lambda out, ctx: out >= 2,
        max_loops=5,
    )
    runner = create_test_flujo(loop, context_model=Ctx)
    result = None
    async for r in runner.run_async(0, initial_context_data={"initial_prompt": "test"}):
        result = r
    assert result is not None, "No result returned from runner.run_async()"
    # FIXED: Context updates are now properly applied between iterations
    # This ensures that context state persists across loop iterations
    # The counter should be incremented exactly twice (once per iteration)
    # since the exit condition is out >= 2 and we start with 0
    assert result.final_pipeline_context.counter == 2, (
        f"Expected counter to be exactly 2 (one increment per iteration), "
        f"but got {result.final_pipeline_context.counter}. "
        f"Context updates should be properly applied between iterations."
    )
