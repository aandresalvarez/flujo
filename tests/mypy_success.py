from typing import Any, cast, Callable, Optional, reveal_type
import asyncio

from flujo.domain import Step, step, Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.testing.utils import StubAgent
from flujo.application.runner import Flujo
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.models import BaseModel


class UserInfo(BaseModel):
    name: str


class Report(BaseModel):
    summary: str


def test_pipeline_type_continuity() -> None:
    agent1 = StubAgent([UserInfo(name="test")])
    agent2 = StubAgent([Report(summary="report")])
    step1: Step[str, UserInfo] = Step.solution(cast(AsyncAgentProtocol[Any, Any], agent1))
    step2: Step[UserInfo, Report] = Step.solution(cast(AsyncAgentProtocol[Any, Any], agent2))
    _pipeline = step1 >> step2

    @step
    async def foo(x: str) -> int:
        return len(x)

    inferred = foo
    reveal_type(inferred)  # noqa: F821

    # pipeline should type check
    # The following should fail mypy if uncommented:
    # agent3 = StubAgent(["raw_string"])
    # step3: Step[int, str] = Step.solution(agent3)
    # bad_pipeline = step1 >> step3


class MyCtx(BaseModel):
    counter: int = 0


def check_result_typing() -> None:
    runner: Flujo[Any, Any, MyCtx] = Flujo(
        Step.solution(cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))),
        context_model=MyCtx,
    )
    result = runner.run("hi")
    reveal_type(result.final_pipeline_context)  # noqa: F821


class LoopCtx(BaseModel):
    is_finished: bool = False


def typed_loop_step() -> None:
    def should_exit(out: int, ctx: LoopCtx | None) -> bool:
        return bool(ctx and ctx.is_finished)

    body: Pipeline[Any, Any] = Pipeline.from_step(Step.model_validate({"name": "a"}))
    _loop_step: LoopStep[LoopCtx] = Step.loop_until(
        name="l",
        loop_body_pipeline=body,
        exit_condition_callable=cast(
            Callable[[Any, Optional[LoopCtx]], bool],
            should_exit,
        ),
    )

    # Uncommenting the following should fail mypy:
    # def bad(out: int, ctx: LoopCtx | None) -> bool:
    #     return ctx.missing  # attr-defined error expected


def typed_conditional_step() -> None:
    def choose(_out: str, ctx: LoopCtx | None) -> str:
        return "a" if ctx and ctx.is_finished else "b"

    branches: dict[str, Pipeline[Any, Any]] = {
        "a": Step.model_validate({"name": "a"}) >> Step.model_validate({"name": "b"})
    }
    _branch_step: ConditionalStep[LoopCtx] = Step.branch_on(
        name="b",
        condition_callable=cast(
            Callable[[Any, Optional[LoopCtx]], str],
            choose,
        ),
        branches=branches,
    )


def typed_arun() -> None:
    length_step: Step[str, int]

    @step
    async def length(x: str) -> int:
        return len(x)

    length_step = length
    out = asyncio.run(length_step.arun("hello"))
    reveal_type(out)  # noqa: F821
