import pytest
from flujo.domain.dsl import Step, StepConfig
from flujo.testing.utils import DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.application.runner import InfiniteRedirectError
from tests.conftest import create_test_flujo


class UnhashableAgent:
    __hash__ = None

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.call_count = 0

    async def run(self, data: str) -> str:
        out = self.outputs[min(self.call_count, len(self.outputs) - 1)]
        self.call_count += 1
        return out


@pytest.mark.asyncio
async def test_redirect_loop_detected_with_unhashable_agents() -> None:
    a1 = UnhashableAgent(["a1"])
    a2 = UnhashableAgent(["a2"])
    plugin = DummyPlugin(
        outcomes=[
            PluginOutcome(success=False, redirect_to=a2),
            PluginOutcome(success=False, redirect_to=a1),
        ]
    )
    step = Step.model_validate(
        {
            "name": "loop",
            "agent": a1,
            "config": StepConfig(max_retries=3),
            "plugins": [(plugin, 0)],
        }
    )
    runner = create_test_flujo(step)
    with pytest.raises(InfiniteRedirectError):
        await gather_result(runner, "start")
