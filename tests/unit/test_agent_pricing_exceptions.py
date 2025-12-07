import pytest

from flujo.application.core.agent_execution_runner import AgentExecutionRunner
from flujo.exceptions import PricingNotConfiguredError


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
        raise PricingNotConfiguredError(provider="test-provider", model="test-model")


class _StubProcessorPipeline:
    async def apply_prompt(self, processors, value, *, context=None):
        return value

    async def apply_output(self, processors, value, *, context=None):
        return value


class _StubUsageMeter:
    async def add(self, *args, **kwargs):
        return None


class _StubCore:
    def __init__(self) -> None:
        self._agent_runner = _StubAgentRunner()
        self._processor_pipeline = _StubProcessorPipeline()
        self._usage_meter = _StubUsageMeter()

    def _safe_step_name(self, step):
        try:
            return step.name
        except Exception:
            return "unknown"


@pytest.mark.asyncio
async def test_agent_execution_runner_raises_pricing_error():
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
