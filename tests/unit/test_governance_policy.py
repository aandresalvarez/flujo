from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from flujo.application.core.agent_orchestrator import AgentOrchestrator
from flujo.application.core.governance_policy import (
    GovernanceDecision,
    GovernanceEngine,
)
from flujo.exceptions import ConfigurationError


class DenyPolicy:
    async def evaluate(
        self,
        *,
        core: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
    ) -> GovernanceDecision:
        return GovernanceDecision(allow=False, reason="blocked")


class AllowPolicy:
    async def evaluate(
        self,
        *,
        core: Any,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
    ) -> GovernanceDecision:
        return GovernanceDecision(allow=True)


@pytest.mark.asyncio
async def test_governance_engine_default_allows() -> None:
    engine = GovernanceEngine()
    await engine.enforce(core=Mock(), step=Mock(), data=None, context=None, resources=None)


@pytest.mark.asyncio
async def test_governance_engine_deny_raises_configuration_error() -> None:
    engine = GovernanceEngine(policies=(DenyPolicy(),))
    with pytest.raises(ConfigurationError):
        await engine.enforce(core=Mock(), step=Mock(), data=None, context=None, resources=None)


@pytest.mark.asyncio
async def test_agent_orchestrator_invokes_governance_before_runner() -> None:
    deny_policy = DenyPolicy()
    engine = GovernanceEngine(policies=(deny_policy,))
    orchestrator = AgentOrchestrator()
    orchestrator._execution_runner.execute = AsyncMock()

    class DummyFallbackHandler:
        MAX_CHAIN_LENGTH = 3

        def reset(self) -> None:  # pragma: no cover - not used
            pass

        def is_step_in_chain(self, _: Any) -> bool:
            return False

        def push_to_chain(self, _: Any) -> None:
            pass

    class DummyCore:
        def __init__(self) -> None:
            self._governance_engine = engine
            self._fallback_handler = DummyFallbackHandler()

    core = DummyCore()

    with pytest.raises(ConfigurationError):
        await orchestrator.execute(
            core=core,
            step=Mock(name="s1"),
            data=None,
            context=None,
            resources=None,
            limits=None,
            stream=False,
            on_chunk=None,
            cache_key=None,
            fallback_depth=0,
        )

    orchestrator._execution_runner.execute.assert_not_called()


@pytest.mark.asyncio
async def test_agent_orchestrator_allows_when_policy_allows() -> None:
    allow_policy = AllowPolicy()
    engine = GovernanceEngine(policies=(allow_policy,))
    orchestrator = AgentOrchestrator()
    orchestrator._execution_runner.execute = AsyncMock(return_value="ok")

    class DummyFallbackHandler:
        MAX_CHAIN_LENGTH = 3

        def reset(self) -> None:  # pragma: no cover - not used
            pass

        def is_step_in_chain(self, _: Any) -> bool:
            return False

        def push_to_chain(self, _: Any) -> None:
            pass

    class DummyCore:
        def __init__(self) -> None:
            self._governance_engine = engine
            self._fallback_handler = DummyFallbackHandler()

    core = DummyCore()

    result = await orchestrator.execute(
        core=core,
        step=Mock(name="s1"),
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        fallback_depth=0,
    )

    orchestrator._execution_runner.execute.assert_called_once()
    assert result == "ok"
