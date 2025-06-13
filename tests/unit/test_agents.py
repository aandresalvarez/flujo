import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from pydantic import SecretStr

# Ensure environment variable exists before agents are imported
os.environ.setdefault("orch_openai_api_key", "test-key")

from pydantic_ai_orchestrator.infra.agents import AsyncAgentWrapper, NoOpReflectionAgent, get_reflection_agent, LoggingReviewAgent

from pydantic_ai_orchestrator.exceptions import OrchestratorRetryError

@pytest.mark.asyncio
async def test_async_agent_wrapper_success():
    agent = AsyncMock()
    agent.run.return_value = "ok"
    wrapper = AsyncAgentWrapper(agent)
    result = await wrapper.run_async("prompt")
    assert result == "ok"

@pytest.mark.asyncio
async def test_async_agent_wrapper_retry_then_success():
    agent = AsyncMock()
    agent.run.side_effect = [Exception("fail"), "ok"]
    wrapper = AsyncAgentWrapper(agent, max_retries=2)
    result = await wrapper.run_async("prompt")
    assert result == "ok"

@pytest.mark.asyncio
async def test_async_agent_wrapper_timeout():
    agent = AsyncMock()
    async def never_returns(*args, **kwargs):
        await asyncio.sleep(0.05)
    agent.run.side_effect = never_returns
    wrapper = AsyncAgentWrapper(agent, timeout=0.01, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")

@pytest.mark.asyncio
async def test_async_agent_wrapper_exception():
    agent = AsyncMock()
    agent.run.side_effect = Exception("fail")
    wrapper = AsyncAgentWrapper(agent, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")

@pytest.mark.asyncio
async def test_async_agent_wrapper_temperature():
    agent = AsyncMock()
    agent.run.return_value = "ok"
    wrapper = AsyncAgentWrapper(agent)
    await wrapper.run_async("prompt", temperature=0.5)
    # Should set generation_kwargs["temperature"]
    agent.run.assert_called()

@pytest.mark.asyncio
async def test_noop_reflection_agent():
    agent = NoOpReflectionAgent()
    result = await agent.run()
    assert result == ""

def test_get_reflection_agent_disabled(monkeypatch):
    import importlib
    import pydantic_ai_orchestrator.infra.agents as agents_mod
    monkeypatch.setattr("pydantic_ai_orchestrator.infra.settings.settings.reflection_enabled", False)
    importlib.reload(agents_mod)
    agent = agents_mod.get_reflection_agent()
    assert agent.__class__.__name__ == "NoOpReflectionAgent"

def test_get_reflection_agent_creation_failure(monkeypatch):
    monkeypatch.setattr("pydantic_ai_orchestrator.infra.settings.settings.reflection_enabled", True)
    with patch("pydantic_ai_orchestrator.infra.agents.make_agent_async", side_effect=Exception("fail")):
        agent = get_reflection_agent()
        assert agent.__class__.__name__ == "NoOpReflectionAgent"

@pytest.mark.asyncio
async def test_logging_review_agent_success():
    base_agent = AsyncMock()
    base_agent.run.return_value = "ok"
    agent = LoggingReviewAgent(base_agent)
    result = await agent.run("prompt")
    assert result == "ok"

@pytest.mark.asyncio
async def test_logging_review_agent_error():
    base_agent = AsyncMock()
    base_agent.run.side_effect = Exception("fail")
    agent = LoggingReviewAgent(base_agent)
    with pytest.raises(Exception):
        await agent.run("prompt")

@pytest.mark.asyncio
async def test_async_agent_wrapper_agent_failed_string():
    agent = AsyncMock()
    agent.run.return_value = "Agent failed after 3 attempts. Last error: foo"
    wrapper = AsyncAgentWrapper(agent, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")

@pytest.mark.asyncio
async def test_logging_review_agent_run_async_fallback():
    class NoAsyncAgent:
        async def run(self, *args, **kwargs):
            return "ok"
    base_agent = NoAsyncAgent()
    agent = LoggingReviewAgent(base_agent)
    result = await agent._run_async("prompt")
    assert result == "ok"

@pytest.mark.asyncio
async def test_logging_review_agent_run_async_non_callable():
    class WeirdAgent:
        run_async = "not callable"
        async def run(self, *args, **kwargs):
            return "ok"
    base_agent = WeirdAgent()
    agent = LoggingReviewAgent(base_agent)
    result = await agent._run_async("prompt")
    assert result == "ok"

@pytest.mark.asyncio
async def test_async_agent_wrapper_agent_failed_string_only():
    class DummyAgent:
        async def run(self, *args, **kwargs):
            return "Agent failed after 2 attempts. Last error: foo"
    wrapper = AsyncAgentWrapper(DummyAgent(), max_retries=1)
    with pytest.raises(OrchestratorRetryError) as exc:
        await wrapper.run_async("prompt")
    assert "Agent failed after" in str(exc.value)


def test_make_agent_async_injects_key(monkeypatch):
    monkeypatch.setenv("orch_openai_api_key", "test-key")
    from pydantic_ai_orchestrator.infra import settings as settings_mod
    settings_mod.settings.openai_api_key = SecretStr("test-key")
    from pydantic_ai_orchestrator.infra.agents import make_agent_async

    wrapper = make_agent_async("openai:gpt-4o", "sys", str)
    assert wrapper is not None


def test_make_agent_async_missing_key(monkeypatch):
    monkeypatch.delenv("orch_anthropic_api_key", raising=False)
    from pydantic_ai_orchestrator.infra import settings as settings_mod
    settings_mod.settings.anthropic_api_key = None
    from pydantic_ai_orchestrator.infra.agents import make_agent_async
    from pydantic_ai_orchestrator.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        make_agent_async("anthropic:claude-3", "sys", str)
