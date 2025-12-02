import pytest
from types import TracebackType
from typing import Optional

from flujo import Step
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.resources import AppResources
from flujo.exceptions import PausedException
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo
from pydantic import Field


class RecordingResource(AppResources):
    enter_calls: int = 0
    exit_calls: int = 0
    commit_count: int = 0
    rollback_count: int = 0
    exc_types: list[Optional[type[BaseException]]] = Field(default_factory=list)
    observed: list[str] = Field(default_factory=list)

    async def __aenter__(self) -> "RecordingResource":
        self.enter_calls += 1
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.exit_calls += 1
        self.exc_types.append(exc_type)
        if exc_type is None:
            self.commit_count += 1
        else:
            self.rollback_count += 1


@pytest.mark.asyncio
async def test_resource_context_manager_commits_on_success() -> None:
    resources = RecordingResource()

    async def agent(data: str, *, resources: RecordingResource) -> str:
        resources.observed.append(data)
        return "ok"

    step = Step.from_callable(agent, name="ctx_success", max_retries=0)
    runner = create_test_flujo(step, resources=resources)

    outcome = await gather_result(runner, "hello")

    assert getattr(outcome, "step_history", []) and outcome.step_history[-1].success is True
    assert resources.enter_calls == resources.exit_calls == 1
    assert resources.commit_count == 1
    assert resources.rollback_count == 0
    assert resources.exc_types == [None]
    assert resources.observed == ["hello"]


@pytest.mark.asyncio
async def test_resource_context_manager_rolls_back_on_failure() -> None:
    resources = RecordingResource()

    async def failing_agent(data: str, *, resources: RecordingResource) -> None:
        raise ValueError("boom")

    step = Step.from_callable(failing_agent, name="ctx_failure", max_retries=0)
    runner = create_test_flujo(step, resources=resources)

    outcome = await gather_result(runner, "input")

    assert getattr(outcome, "step_history", []) and outcome.step_history[-1].success is False
    assert resources.enter_calls == resources.exit_calls == 1
    assert resources.commit_count == 0
    assert resources.rollback_count == 1
    assert resources.exc_types[0] in {RuntimeError, ValueError}


@pytest.mark.asyncio
async def test_resource_context_manager_handles_paused_exception() -> None:
    resources = RecordingResource()

    async def pausing_agent(data: str, *, resources: RecordingResource) -> str:
        raise PausedException("pause-now")

    step = Step.from_callable(pausing_agent, name="ctx_paused", max_retries=0)
    runner = create_test_flujo(step, resources=resources)

    outcome = await gather_result(runner, "anything")

    assert getattr(outcome, "status", None) == "paused"
    assert resources.enter_calls == resources.exit_calls == 1
    assert resources.exc_types and resources.exc_types[0] is PausedException


@pytest.mark.asyncio
async def test_parallel_branches_enter_resources_per_attempt() -> None:
    resources = RecordingResource()

    async def branch(data: str, *, resources: RecordingResource) -> str:
        resources.observed.append(data)
        return f"branch-{data}"

    branch_one = Step.from_callable(branch, name="branch_one", max_retries=0)
    branch_two = Step.from_callable(branch, name="branch_two", max_retries=0)

    parallel = ParallelStep.model_validate(
        {"name": "parallel_step", "branches": {"one": branch_one, "two": branch_two}}
    )

    runner = create_test_flujo(parallel, resources=resources)

    outcome = await gather_result(runner, "data")

    assert resources.enter_calls == resources.exit_calls == 2
    assert resources.rollback_count == 0
    assert resources.commit_count == 2
    assert resources.exc_types == [None, None]
    assert len(resources.observed) == 2
    assert getattr(outcome, "step_history", []) and outcome.step_history[-1].success is True
