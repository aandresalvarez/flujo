from __future__ import annotations

from typing import Any

import pytest

from flujo.application.core.agent_execution_runner import AgentExecutionRunner
from flujo.domain.agent_result import FlujoAgentResult
from flujo.domain.models import StepOutcome, StepResult, Success

pytestmark = pytest.mark.fast


class _StubConfig:
    max_retries = 0
    timeout_s = None
    temperature = None
    top_k = None
    top_p = None


class _StubStep:
    def __init__(self) -> None:
        self.name = "unwrap_step"
        self.agent = object()
        self.config = _StubConfig()
        self.processors: list[tuple[object, ...]] = [("noop",)]
        self.plugins: list[object] = []
        self.validators: list[object] = []
        self.fallback_step = None
        self.meta: dict[str, object] = {}
        self.sink_to = None
        self.updates_context = False


class _StubAgentRunner:
    async def run(self, *args: object, **kwargs: object) -> object:
        _ = (args, kwargs)
        return FlujoAgentResult(output="raw-payload")


class _StubProcessorPipeline:
    def __init__(self) -> None:
        self.last_output_value: object | None = None

    async def apply_prompt(
        self, processors: object, value: object, *, context: object | None = None
    ) -> object:
        _ = (processors, context)
        return value

    async def apply_output(
        self, processors: object, value: object, *, context: object | None = None
    ) -> object:
        _ = (processors, context)
        self.last_output_value = value
        return value


class _StubValidationOrchestrator:
    def __init__(self) -> None:
        self.last_output_value: object | None = None

    async def validate(
        self, *args: object, **kwargs: object
    ) -> StepOutcome[StepResult] | StepResult | None:
        self.last_output_value = kwargs.get("output")
        return None


class _StubUsageMeter:
    async def add(self, *args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        return None


class _StubCore:
    def __init__(self) -> None:
        self._agent_runner = _StubAgentRunner()
        self._processor_pipeline = _StubProcessorPipeline()
        self._validation_orchestrator = _StubValidationOrchestrator()
        self._usage_meter = _StubUsageMeter()

    def _safe_step_name(self, step: Any) -> str:
        return str(getattr(step, "name", "unknown"))


@pytest.mark.asyncio
async def test_agent_execution_runner_unwraps_flujo_agent_result_for_processing() -> None:
    runner = AgentExecutionRunner()
    core = _StubCore()
    step = _StubStep()

    outcome = await runner.execute(
        core=core,  # type: ignore[arg-type]
        step=step,
        data="input",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    assert outcome.step_result.output == "raw-payload"
    assert core._processor_pipeline.last_output_value == "raw-payload"
    assert core._validation_orchestrator.last_output_value == "raw-payload"
