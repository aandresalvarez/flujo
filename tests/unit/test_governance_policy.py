from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from flujo.application.core.agent_orchestrator import AgentOrchestrator
from flujo.application.core.governance_policy import (
    GovernanceDecision,
    GovernanceEngine,
    PIIScrubbingPolicy,
    ToolAllowlistPolicy,
)
from flujo.application.core.runtime_builder import FlujoRuntimeBuilder
from flujo.domain.models import StepResult, Success
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
    out = await engine.enforce(core=Mock(), step=Mock(), data=None, context=None, resources=None)
    assert out is None


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
    orchestrator._execution_runner = Mock()
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
            core=cast(Any, core),
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
    orchestrator._execution_runner = Mock()
    orchestrator._execution_runner.execute = AsyncMock(
        return_value=Success(step_result=StepResult(name="s1", output="ok", success=True))
    )

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
        core=cast(Any, core),
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
    assert isinstance(result, Success)
    assert result.step_result.output == "ok"


@pytest.mark.asyncio
async def test_runtime_builder_respects_settings(monkeypatch: Any) -> None:
    monkeypatch.setenv("FLUJO_GOVERNANCE_MODE", "deny_all")
    deps: Any = FlujoRuntimeBuilder().build()
    engine = deps.governance_engine
    assert isinstance(engine, GovernanceEngine)
    with pytest.raises(ConfigurationError):
        await engine.enforce(core=object(), step=Mock(), data=None, context=None, resources=None)


@pytest.mark.asyncio
async def test_pii_scrubbing_policy_replaces_input_data() -> None:
    engine = GovernanceEngine(policies=(PIIScrubbingPolicy(),))
    data = {"email": "user@example.com", "note": "call me at 555-555-5555"}
    out = await engine.enforce(
        core=Mock(), step=Mock(name="s1"), data=data, context=None, resources=None
    )
    assert isinstance(out, dict)
    assert out["email"] == "[REDACTED]"
    assert "[REDACTED]" in str(out["note"])


@pytest.mark.asyncio
async def test_pii_strong_mode_falls_back_when_presidio_missing(monkeypatch: Any) -> None:
    real_import = __import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:  # noqa: A002
        if name.startswith("presidio_"):
            raise ImportError("no presidio")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    engine = GovernanceEngine(policies=(PIIScrubbingPolicy(strong=True),))
    out = await engine.enforce(
        core=Mock(),
        step=Mock(name="s1"),
        data="email user@example.com",
        context=None,
        resources=None,
    )
    assert out == "email [REDACTED]"


@pytest.mark.asyncio
async def test_tool_allowlist_policy_denies_disallowed_tool() -> None:
    def safe_tool() -> None:  # pragma: no cover - used for identity only
        return None

    def dangerous_tool() -> None:  # pragma: no cover - used for identity only
        return None

    class DummyAgent:
        def __init__(self) -> None:
            self.tools = [safe_tool, dangerous_tool]

    step = Mock()
    step.name = "s1"
    step.agent = Mock()
    step.agent._agent = DummyAgent()

    engine = GovernanceEngine(policies=(ToolAllowlistPolicy(allowed=frozenset({"safe_tool"})),))
    with pytest.raises(ConfigurationError):
        await engine.enforce(core=Mock(), step=step, data="hi", context=None, resources=None)
