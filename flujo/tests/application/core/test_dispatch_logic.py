from typing import Any

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step, HumanInTheLoopStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.steps.cache_step import CacheStep
from flujo.domain.models import Success, StepResult


@pytest.mark.asyncio
async def test_registry_dispatch_invokes_correct_policies():
    core = ExecutorCore()

    # Monkeypatch each policy's execute to mark invocation
    called: dict[str, int] = {}

    async def mark(name: str, result: Any) -> Any:
        called[name] = called.get(name, 0) + 1
        return result

    # Agent/default (base Step)
    orig_agent = core.agent_step_executor.execute

    async def agent_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("agent", StepResult(name="s", output=1, success=True))

    core.agent_step_executor.execute = agent_exec  # type: ignore[assignment]

    # Loop
    orig_loop = core.loop_step_executor.execute

    async def loop_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("loop", StepResult(name="l", output=2, success=True))

    core.loop_step_executor.execute = loop_exec  # type: ignore[assignment]

    # Parallel
    orig_par = core.parallel_step_executor.execute

    async def par_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("parallel", StepResult(name="p", output=3, success=True))

    core.parallel_step_executor.execute = par_exec  # type: ignore[assignment]

    # Conditional
    orig_cond = core.conditional_step_executor.execute

    async def cond_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("conditional", StepResult(name="c", output=4, success=True))

    core.conditional_step_executor.execute = cond_exec  # type: ignore[assignment]

    # Dynamic Router
    orig_router = core.dynamic_router_step_executor.execute

    async def router_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("router", StepResult(name="r", output=5, success=True))

    core.dynamic_router_step_executor.execute = router_exec  # type: ignore[assignment]

    # HITL
    orig_hitl = core.hitl_step_executor.execute

    async def hitl_exec(*args: Any, **kwargs: Any) -> StepResult:
        return await mark("hitl", StepResult(name="h", output=6, success=True))

    core.hitl_step_executor.execute = hitl_exec  # type: ignore[assignment]

    # Cache
    orig_cache = core.cache_step_executor.execute

    async def cache_exec(*args: Any, **kwargs: Any) -> Success:
        sr = StepResult(name="cache", output=7, success=True)
        return await mark("cache", Success(step_result=sr))  # type: ignore[return-value]

    core.cache_step_executor.execute = cache_exec  # type: ignore[assignment]

    # Trigger for each type
    await core.execute(step=Step(name="base", agent=object()), data=None)
    await core.execute(step=LoopStep(name="loop", body=Step(name="b", agent=object())), data=None)
    await core.execute(step=ParallelStep(name="par", branches=[]), data=None)
    await core.execute(step=ConditionalStep(name="cond", branches=[]), data=None)
    await core.execute(step=DynamicParallelRouterStep(name="router", routes=[]), data=None)
    await core.execute(step=HumanInTheLoopStep(name="hitl", message_for_user="ok"), data=None)
    await core.execute(step=CacheStep(name="cache"), data=None)

    # Validate each policy was called exactly once
    assert called.get("agent", 0) == 1
    assert called.get("loop", 0) == 1
    assert called.get("parallel", 0) == 1
    assert called.get("conditional", 0) == 1
    assert called.get("router", 0) == 1
    assert called.get("hitl", 0) == 1
    assert called.get("cache", 0) == 1

    # Restore (not strictly necessary in ephemeral test env)
    core.agent_step_executor.execute = orig_agent  # type: ignore[assignment]
    core.loop_step_executor.execute = orig_loop  # type: ignore[assignment]
    core.parallel_step_executor.execute = orig_par  # type: ignore[assignment]
    core.conditional_step_executor.execute = orig_cond  # type: ignore[assignment]
    core.dynamic_router_step_executor.execute = orig_router  # type: ignore[assignment]
    core.hitl_step_executor.execute = orig_hitl  # type: ignore[assignment]
    core.cache_step_executor.execute = orig_cache  # type: ignore[assignment]
