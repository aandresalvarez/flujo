import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from pydantic import SecretStr

from flujo.infra.agents import (
    AsyncAgentWrapper,
    NoOpReflectionAgent,
    get_reflection_agent,
    LoggingReviewAgent,
)
from flujo.domain.models import Checklist, ChecklistItem

from flujo.exceptions import OrchestratorRetryError
from flujo.infra.settings import settings


@pytest.fixture
def mock_pydantic_ai_agent() -> MagicMock:
    agent = MagicMock()
    agent.model = "test_model"
    return agent


@pytest.mark.asyncio
async def test_async_agent_wrapper_success() -> None:
    agent = AsyncMock()
    agent.run.return_value = "ok"
    wrapper = AsyncAgentWrapper(agent)
    result = await wrapper.run_async("prompt")
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_agent_wrapper_retry_then_success() -> None:
    agent = AsyncMock()
    agent.run.side_effect = [Exception("fail"), "ok"]
    wrapper = AsyncAgentWrapper(agent, max_retries=2)
    result = await wrapper.run_async("prompt")
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_agent_wrapper_timeout() -> None:
    agent = AsyncMock()

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(2)

    agent.run.side_effect = never_returns
    wrapper = AsyncAgentWrapper(agent, timeout=1, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")


@pytest.mark.asyncio
async def test_async_agent_wrapper_exception() -> None:
    agent = AsyncMock()
    agent.run.side_effect = Exception("fail")
    wrapper = AsyncAgentWrapper(agent, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")


@pytest.mark.asyncio
async def test_async_agent_wrapper_temperature() -> None:
    agent = AsyncMock()
    agent.run.return_value = "ok"
    wrapper = AsyncAgentWrapper(agent)
    await wrapper.run_async("prompt", temperature=0.5)
    # Should set generation_kwargs["temperature"]
    agent.run.assert_called()


@pytest.mark.asyncio
async def test_noop_reflection_agent() -> None:
    agent = NoOpReflectionAgent()
    result = await agent.run()
    assert result == ""


def test_get_reflection_agent_disabled(monkeypatch) -> None:
    import importlib
    import flujo.infra.agents as agents_mod

    monkeypatch.setattr("flujo.infra.settings.settings.reflection_enabled", False)
    importlib.reload(agents_mod)
    agent = agents_mod.get_reflection_agent()
    assert agent.__class__.__name__ == "NoOpReflectionAgent"


def test_get_reflection_agent_creation_failure(monkeypatch) -> None:
    monkeypatch.setattr("flujo.infra.settings.settings.reflection_enabled", True)
    with patch("flujo.infra.agents.make_agent_async", side_effect=Exception("fail")):
        agent = get_reflection_agent()
        assert agent.__class__.__name__ == "NoOpReflectionAgent"


@pytest.mark.asyncio
async def test_logging_review_agent_success() -> None:
    base_agent = AsyncMock()
    base_agent.run.return_value = "ok"
    agent = LoggingReviewAgent(base_agent)
    result = await agent.run("prompt")
    assert result == "ok"


@pytest.mark.asyncio
async def test_logging_review_agent_error() -> None:
    base_agent = AsyncMock()
    base_agent.run.side_effect = Exception("fail")
    agent = LoggingReviewAgent(base_agent)
    with pytest.raises(Exception):
        await agent.run("prompt")


@pytest.mark.asyncio
async def test_async_agent_wrapper_agent_failed_string() -> None:
    agent = AsyncMock()
    agent.run.return_value = "Agent failed after 3 attempts. Last error: foo"
    wrapper = AsyncAgentWrapper(agent, max_retries=1)
    with pytest.raises(OrchestratorRetryError):
        await wrapper.run_async("prompt")


@pytest.mark.asyncio
async def test_logging_review_agent_run_async_fallback() -> None:
    class NoAsyncAgent:
        async def run(self, *args, **kwargs):
            return "ok"

    base_agent = NoAsyncAgent()
    agent = LoggingReviewAgent(base_agent)
    result = await agent._run_async("prompt")
    assert result == "ok"


@pytest.mark.asyncio
async def test_logging_review_agent_run_async_non_callable() -> None:
    class WeirdAgent:
        run_async = "not callable"

        async def run(self, *args, **kwargs):
            return "ok"

    base_agent = WeirdAgent()
    agent = LoggingReviewAgent(base_agent)
    result = await agent._run_async("prompt")
    assert result == "ok"


@pytest.mark.asyncio
async def test_async_agent_wrapper_agent_failed_string_only() -> None:
    class DummyAgent:
        async def run(self, *args, **kwargs):
            return "Agent failed after 2 attempts. Last error: foo"

    wrapper = AsyncAgentWrapper(DummyAgent(), max_retries=1)
    with pytest.raises(OrchestratorRetryError) as exc:
        await wrapper.run_async("prompt")
    assert "Agent failed after" in str(exc.value)


def test_make_agent_async_injects_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "openai_api_key", SecretStr("test-key"))
    from flujo.infra.agents import make_agent_async

    wrapper = make_agent_async("openai:gpt-4o", "sys", str)
    assert wrapper is not None


def test_make_agent_async_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("ORCH_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from flujo.infra import settings as settings_mod

    settings_mod.settings.anthropic_api_key = None
    from flujo.infra.agents import make_agent_async
    from flujo.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        make_agent_async("anthropic:claude-3", "sys", str)


def test_async_agent_wrapper_timeout_validation() -> None:
    """Test that AsyncAgentWrapper validates timeout parameter type."""
    agent = AsyncMock()
    with pytest.raises(TypeError, match="timeout must be an integer or None"):
        AsyncAgentWrapper(agent, timeout="not a number")


def test_async_agent_wrapper_with_dummy_agent() -> None:
    class DummyAgent:
        async def run(self, *args, **kwargs):
            return "dummy"

    wrapper = AsyncAgentWrapper(DummyAgent())
    assert isinstance(wrapper, AsyncAgentWrapper)


def test_async_agent_wrapper_init_valid_args(mock_pydantic_ai_agent: MagicMock) -> None:
    wrapper = AsyncAgentWrapper(
        agent=mock_pydantic_ai_agent, max_retries=5, timeout=10, model_name="custom_test_model"
    )
    assert wrapper._max_retries == 5
    assert wrapper._timeout_seconds == 10
    assert wrapper._model_name == "custom_test_model"
    assert wrapper._agent is mock_pydantic_ai_agent


def test_async_agent_wrapper_init_default_timeout(mock_pydantic_ai_agent: MagicMock) -> None:
    wrapper = AsyncAgentWrapper(agent=mock_pydantic_ai_agent)
    assert wrapper._timeout_seconds == settings.agent_timeout


def test_async_agent_wrapper_init_invalid_max_retries_type(
    mock_pydantic_ai_agent: MagicMock,
) -> None:
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        AsyncAgentWrapper(agent=mock_pydantic_ai_agent, max_retries="not_an_int")


def test_async_agent_wrapper_init_negative_max_retries_value(
    mock_pydantic_ai_agent: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
        AsyncAgentWrapper(agent=mock_pydantic_ai_agent, max_retries=-1)


def test_async_agent_wrapper_init_invalid_timeout_type(mock_pydantic_ai_agent: MagicMock) -> None:
    with pytest.raises(TypeError, match="timeout must be an integer or None"):
        AsyncAgentWrapper(agent=mock_pydantic_ai_agent, timeout="not_an_int")


def test_async_agent_wrapper_init_non_positive_timeout_value(
    mock_pydantic_ai_agent: MagicMock,
) -> None:
    with pytest.raises(ValueError, match="timeout must be a positive integer if specified"):
        AsyncAgentWrapper(agent=mock_pydantic_ai_agent, timeout=0)
    with pytest.raises(ValueError, match="timeout must be a positive integer if specified"):
        AsyncAgentWrapper(agent=mock_pydantic_ai_agent, timeout=-10)


@pytest.mark.asyncio
async def test_async_agent_wrapper_runtime_timeout(mock_pydantic_ai_agent: MagicMock) -> None:
    async def slow_run(*args, **kwargs):
        await asyncio.sleep(2)
        return "should_not_reach_here"

    mock_pydantic_ai_agent.run = AsyncMock(side_effect=slow_run)
    wrapper = AsyncAgentWrapper(agent=mock_pydantic_ai_agent, timeout=1, max_retries=1)
    with pytest.raises(OrchestratorRetryError) as exc_info:
        await wrapper.run_async("prompt")
    assert "timed out" in str(exc_info.value).lower() or "TimeoutError" in str(exc_info.value)
    mock_pydantic_ai_agent.run.assert_called_once()


def test_make_self_improvement_agent_uses_settings_default(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.infra.agents.make_agent_async",
        fake_make,
    )
    monkeypatch.setattr(
        "flujo.infra.agents.settings.default_self_improvement_model",
        "model_from_settings",
    )
    from flujo.infra.agents import make_self_improvement_agent

    make_self_improvement_agent()
    assert called["model"] == "model_from_settings"


def test_make_self_improvement_agent_uses_override_model(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.infra.agents.make_agent_async",
        fake_make,
    )
    from flujo.infra.agents import make_self_improvement_agent

    make_self_improvement_agent(model="override_model")
    assert called["model"] == "override_model"


@pytest.mark.asyncio
async def test_async_agent_wrapper_serializes_pydantic_input() -> None:
    mock_agent = AsyncMock()
    wrapper = AsyncAgentWrapper(mock_agent)
    checklist = Checklist(items=[ChecklistItem(description="a")])
    await wrapper.run_async(checklist)
    mock_agent.run.assert_called_once_with(checklist.model_dump())


@pytest.mark.asyncio
async def test_async_agent_wrapper_passthrough_non_model() -> None:
    mock_agent = AsyncMock()
    wrapper = AsyncAgentWrapper(mock_agent)
    await wrapper.run_async("hi")
    mock_agent.run.assert_called_once_with("hi")


@pytest.mark.asyncio
async def test_async_agent_wrapper_serializes_pydantic_kwarg() -> None:
    mock_agent = AsyncMock()
    wrapper = AsyncAgentWrapper(mock_agent)
    checklist = Checklist(items=[ChecklistItem(description="a")])
    await wrapper.run_async(data=checklist)
    mock_agent.run.assert_called_once_with(data=checklist.model_dump())
