from typing import Any, Dict

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step, HumanInTheLoopStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.dsl.cache_step import CacheStep
from flujo.domain.models import Success, StepResult
from flujo.domain.dsl.pipeline import Pipeline


@pytest.mark.asyncio
async def test_registry_dispatch_invokes_correct_policies() -> None:
    core: ExecutorCore[Any] = ExecutorCore()

    # Monkeypatch each policy's execute to mark invocation
    called: Dict[str, int] = {}

    async def mark(
        name: str, result: StepResult | Success[StepResult]
    ) -> StepResult | Success[StepResult]:
        called[name] = called.get(name, 0) + 1
        return result

    # Agent/default (base Step)
    orig_agent = core.agent_step_executor.execute

    async def agent_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["agent"] = called.get("agent", 0) + 1
        return StepResult(name="s", output=1, success=True)

    core.agent_step_executor.execute = agent_exec  # type: ignore[assignment]

    # Loop
    orig_loop = core.loop_step_executor.execute

    async def loop_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["loop"] = called.get("loop", 0) + 1
        return StepResult(name="l", output=2, success=True)

    core.loop_step_executor.execute = loop_exec  # type: ignore[assignment]

    # Parallel
    orig_par = core.parallel_step_executor.execute

    async def par_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["parallel"] = called.get("parallel", 0) + 1
        return StepResult(name="p", output=3, success=True)

    core.parallel_step_executor.execute = par_exec  # type: ignore[assignment]

    # Conditional
    orig_cond = core.conditional_step_executor.execute

    async def cond_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["conditional"] = called.get("conditional", 0) + 1
        return StepResult(name="c", output=4, success=True)

    core.conditional_step_executor.execute = cond_exec  # type: ignore[assignment]

    # Dynamic Router
    orig_router = core.dynamic_router_step_executor.execute

    async def router_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["router"] = called.get("router", 0) + 1
        return StepResult(name="r", output=5, success=True)

    core.dynamic_router_step_executor.execute = router_exec  # type: ignore[assignment]

    # HITL
    orig_hitl = core.hitl_step_executor.execute

    async def hitl_exec(*args: Any, **kwargs: Any) -> StepResult:
        called["hitl"] = called.get("hitl", 0) + 1
        return StepResult(name="h", output=6, success=True)

    core.hitl_step_executor.execute = hitl_exec  # type: ignore[assignment]

    # Cache
    orig_cache = core.cache_step_executor.execute

    async def cache_exec(*args: Any, **kwargs: Any) -> Success[StepResult]:
        called["cache"] = called.get("cache", 0) + 1
        sr: StepResult = StepResult(name="cache", output=7, success=True)
        return Success(step_result=sr)

    core.cache_step_executor.execute = cache_exec  # type: ignore[method-assign]

    # Trigger for each type
    await core.execute(step=Step(name="base", agent=object()), data=None)
    await core.execute(
        step=LoopStep(
            name="loop",
            loop_body_pipeline=Pipeline(steps=[Step(name="b", agent=object())]),
            exit_condition_callable=lambda _o, _c: True,
        ),
        data=None,
    )
    await core.execute(
        step=ParallelStep(
            name="par", branches={"a": Pipeline.from_step(Step(name="x", agent=object()))}
        ),
        data=None,
    )
    await core.execute(
        step=ConditionalStep(
            name="cond",
            condition_callable=lambda _o, _c: "a",
            branches={"a": Pipeline.from_step(Step(name="xa", agent=object()))},
        ),
        data=None,
    )
    await core.execute(
        step=DynamicParallelRouterStep(
            name="router",
            router_agent=object(),
            branches={"a": Pipeline.from_step(Step(name="ya", agent=object()))},
        ),
        data=None,
    )
    await core.execute(step=HumanInTheLoopStep(name="hitl", message_for_user="ok"), data=None)
    await core.execute(
        step=CacheStep(name="cache", wrapped_step=Step(name="w", agent=object())), data=None
    )

    # Validate each policy was called exactly once
    assert called.get("agent", 0) == 1
    assert called.get("loop", 0) == 1
    assert called.get("parallel", 0) == 1
    assert called.get("conditional", 0) == 1
    assert called.get("router", 0) == 1
    assert called.get("hitl", 0) == 1
    assert called.get("cache", 0) == 1

    # Restore (not strictly necessary in ephemeral test env)
    core.agent_step_executor.execute = orig_agent  # type: ignore[method-assign]
    core.loop_step_executor.execute = orig_loop  # type: ignore[method-assign]
    core.parallel_step_executor.execute = orig_par  # type: ignore[method-assign]
    core.conditional_step_executor.execute = orig_cond  # type: ignore[method-assign]
    core.dynamic_router_step_executor.execute = orig_router  # type: ignore[method-assign]
    core.hitl_step_executor.execute = orig_hitl  # type: ignore[method-assign]
    core.cache_step_executor.execute = orig_cache  # type: ignore[method-assign]
