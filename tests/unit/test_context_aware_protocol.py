import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from flujo import Flujo, Step
from flujo.domain.agent_protocol import ContextAwareAgentProtocol, AsyncAgentProtocol
from flujo.domain.plugins import ContextAwarePluginProtocol, PluginOutcome
from flujo.testing.utils import gather_result


class Ctx(BaseModel):
    val: int = 0


class TypedAgent(ContextAwareAgentProtocol[str, str, Ctx]):
    async def run(self, data: str, *, pipeline_context: Ctx, **kwargs: Any) -> str:
        pipeline_context.val += 1
        return data


class LegacyAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, *, pipeline_context: Ctx | None = None) -> str:
        return data

    async def run_async(self, data: str, *, pipeline_context: Ctx | None = None) -> str:
        return await self.run(data, pipeline_context=pipeline_context)


class TypedPlugin(ContextAwarePluginProtocol[Ctx]):
    async def validate(
        self, data: dict[str, Any], *, pipeline_context: Ctx, **kwargs: Any
    ) -> PluginOutcome:
        pipeline_context.val += 1
        return PluginOutcome(success=True)


@pytest.mark.asyncio
async def test_context_aware_agent_no_warning() -> None:
    step = Step("s", TypedAgent())
    runner = Flujo(step, context_model=Ctx)
    with warnings.catch_warnings(record=True) as rec:
        await gather_result(runner, "in")
    assert not any(isinstance(w.message, DeprecationWarning) for w in rec)


@pytest.mark.asyncio
async def test_legacy_agent_triggers_warning() -> None:
    step = Step("s", LegacyAgent())
    runner = Flujo(step, context_model=Ctx)
    with pytest.warns(DeprecationWarning):
        await gather_result(runner, "in")
