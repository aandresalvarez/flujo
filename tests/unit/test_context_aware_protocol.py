import warnings
from typing import Any

import pytest
from typing import Optional
from flujo.domain.models import BaseModel

from flujo import Step
from flujo.domain.agent_protocol import ContextAwareAgentProtocol, AsyncAgentProtocol
from flujo.domain.plugins import ContextAwarePluginProtocol, PluginOutcome
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    val: int = 0


class TypedAgent(ContextAwareAgentProtocol[str, str, Ctx]):
    __context_aware__ = True

    async def run(
        self, data: str, *, context: Ctx = None, pipeline_context: Ctx = None, **kwargs
    ) -> str:
        ctx = context or pipeline_context
        if ctx is not None:
            ctx.val += 1
        return data


class LegacyAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, *, context: Optional[Ctx] = None) -> str:
        return data

    async def run_async(self, data: str, *, context: Optional[Ctx] = None) -> str:
        return await self.run(data, context=context)


class TypedPlugin(ContextAwarePluginProtocol[Ctx]):
    async def validate(self, data: dict[str, Any], *, context: Ctx, **kwargs: Any) -> PluginOutcome:
        context.val += 1
        return PluginOutcome(success=True)


@pytest.mark.asyncio
async def test_context_aware_agent_no_warning() -> None:
    step = Step.model_validate({"name": "s", "agent": TypedAgent()})
    runner = create_test_flujo(step, context_model=Ctx, initial_context_data={"val": 0})
    with warnings.catch_warnings(record=True) as rec:
        await gather_result(runner, "in")
    assert not any(isinstance(w.message, DeprecationWarning) for w in rec)


@pytest.mark.asyncio
async def test_legacy_agent_works_with_context() -> None:
    step = Step.model_validate({"name": "s", "agent": LegacyAgent()})
    runner = create_test_flujo(step, context_model=Ctx)
    # Should work without warnings since we now use 'context' parameter
    result = await gather_result(runner, "in")
    assert result.step_history[0].success
