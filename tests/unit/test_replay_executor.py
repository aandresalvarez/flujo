from __future__ import annotations

import asyncio
from typing import Any

import pytest

from flujo.application.runner_components import ReplayExecutor
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.exceptions import ReplayError


class _FakeStateBackend:
    def __init__(self) -> None:
        self._details = {"id": "run-1"}
        self._steps = [
            {"step_name": "s1", "output": "foo"},
        ]
        self._trace = {"events": [{"name": "flujo.resumed", "attributes": {"human_input": "bar"}}]}
        self._state = {"pipeline_context": {"status": "running"}}

    async def get_run_details(self, run_id: str) -> Any:
        return self._details

    async def list_run_steps(self, run_id: str) -> Any:
        return self._steps

    async def get_trace(self, run_id: str) -> Any:
        return self._trace

    async def load_state(self, run_id: str) -> Any:
        return self._state


class _Step:
    def __init__(self, name: str) -> None:
        self.name = name
        self.agent = None


class _Pipeline:
    def __init__(self) -> None:
        self.steps = [_Step("s1")]


class _Runner:
    def __init__(self, state_backend: Any) -> None:
        self.state_backend = state_backend
        self.pipeline = None
        self._ensured = False
        self._replayed_inputs: list[Any] = []

    def _ensure_pipeline(self) -> None:
        self._ensured = True
        self.pipeline = _Pipeline()

    async def run_async(self, initial_input: Any, initial_context_data: Any):
        self._replayed_inputs.append(initial_input)
        # Yield a single PipelineResult to mimic minimal contract
        pr: PipelineResult[PipelineContext] = PipelineResult(
            step_history=[],
            final_pipeline_context=PipelineContext(),
        )
        yield pr

    async def resume_async(
        self, paused_result: PipelineResult[PipelineContext], human_input: Any
    ) -> PipelineResult[PipelineContext]:
        pr = PipelineResult(step_history=[], final_pipeline_context=PipelineContext())
        pr.final_pipeline_context.status = "running"
        return pr


def test_replay_executor_requires_state_backend() -> None:
    runner = _Runner(state_backend=None)
    executor = ReplayExecutor(runner)
    with pytest.raises(ReplayError):
        asyncio.run(executor.replay_from_trace("run-1"))


def test_replay_executor_replays_and_drains_pauses() -> None:
    backend = _FakeStateBackend()
    runner = _Runner(state_backend=backend)
    executor = ReplayExecutor(runner)

    result = asyncio.run(executor.replay_from_trace("run-1"))

    assert isinstance(result, PipelineResult)
    assert runner._ensured is True
    assert runner._replayed_inputs[0] is None  # from trace attributes
