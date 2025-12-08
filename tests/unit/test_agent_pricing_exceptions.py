from typing import Any

import pytest

from flujo.application.core.agent_execution_runner import AgentExecutionRunner
from flujo.exceptions import PricingNotConfiguredError

pytestmark = pytest.mark.fast


class _StubConfig:
    max_retries = 0
    timeout_s = None
    temperature = None
    top_k = None
    top_p = None


class _StubStep:
    def __init__(self) -> None:
        self.name = "pricing_step"
        self.agent = "agent-id"
        self.config = _StubConfig()
        self.processors: list[tuple] = []
        self.plugins: list = []
        self.validators: list = []
        self.fallback_step = None
        self.meta = {}
        self.sink_to = None
        self.updates_context = False


class _StubAgentRunner:
    async def run(self, *args, **kwargs):
        _ = (args, kwargs)
        raise PricingNotConfiguredError(provider="test-provider", model="test-model")


class _StubProcessorPipeline:
    async def apply_prompt(self, processors, value, *, context=None):
        _ = (processors, context)
        return value

    async def apply_output(self, processors, value, *, context=None):
        _ = (processors, context)
        return value


class _StubUsageMeter:
    async def add(self, *args, **kwargs):
        _ = (args, kwargs)
        return None


class _StubCore:
    def __init__(self) -> None:
        self._agent_runner = _StubAgentRunner()
        self._processor_pipeline = _StubProcessorPipeline()
        self._usage_meter = _StubUsageMeter()

    def _safe_step_name(self, step: Any) -> str:
        return getattr(step, "name", "unknown")


@pytest.mark.asyncio
async def test_agent_execution_runner_raises_pricing_error() -> None:
    runner = AgentExecutionRunner()
    core = _StubCore()
    step = _StubStep()

    with pytest.raises(PricingNotConfiguredError):
        await runner.execute(
            core=core,
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
