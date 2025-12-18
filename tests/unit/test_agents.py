import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pydantic import SecretStr, BaseModel, TypeAdapter
from typing import List, Union, Dict

from flujo.agents import (
    AsyncAgentWrapper,
    NoOpReflectionAgent,
    LoggingReviewAgent,
    make_agent_async,
)
from flujo.domain.models import Checklist, ChecklistItem
from flujo.domain.agent_result import FlujoAgentResult

from flujo.exceptions import OrchestratorRetryError
from flujo.infra.settings import settings
from flujo.domain.processors import AgentProcessors


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
    # Wrapper now returns FlujoAgentResult; access output inside
    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_async_agent_wrapper_retry_then_success(no_wait_backoff) -> None:
    agent = AsyncMock()
    agent.run.side_effect = [Exception("fail"), "ok"]
    wrapper = AsyncAgentWrapper(agent, max_retries=2)
    result = await wrapper.run_async("prompt")
    # Wrapper now returns FlujoAgentResult; access output inside
    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_async_agent_wrapper_timeout(monkeypatch) -> None:
    agent = AsyncMock()

    async def fake_wait_for(awaitable, *args, **kwargs):
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    agent.run.side_effect = AsyncMock(return_value="unused")
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
async def test_async_agent_wrapper_temperature_passed_directly() -> None:
    agent = AsyncMock()
    agent.run.return_value = "ok"
    wrapper = AsyncAgentWrapper(agent)
    await wrapper.run_async("prompt", temperature=0.5)
    agent.run.assert_called_once()
    kwargs = agent.run.call_args.kwargs
    assert kwargs.get("temperature") == 0.5
    assert "generation_kwargs" not in kwargs


@pytest.mark.asyncio
async def test_noop_reflection_agent() -> None:
    agent = NoOpReflectionAgent()
    result = await agent.run()
    # NoOpReflectionAgent returns empty string directly (not wrapped in FlujoAgentResult)
    assert result == ""


def test_get_reflection_agent_disabled(monkeypatch) -> None:
    import importlib
    import flujo.agents as agents_mod
    import sys
    from flujo.infra.settings import Settings

    # Create a new settings instance with reflection disabled
    disabled_settings = Settings(reflection_enabled=False)
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", disabled_settings)
    importlib.reload(agents_mod)

    from flujo.agents import get_reflection_agent, NoOpReflectionAgent

    agent = get_reflection_agent()
    assert isinstance(agent, NoOpReflectionAgent)


def test_get_reflection_agent_creation_failure(monkeypatch) -> None:
    import sys
    from flujo.infra.settings import Settings

    # Create a new settings instance with reflection enabled but no API key
    enabled_settings = Settings(reflection_enabled=True, openai_api_key=None)
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", enabled_settings)


@pytest.mark.asyncio
async def test_logging_review_agent_success() -> None:
    base_agent = AsyncMock()
    base_agent.run.return_value = "ok"
    agent = LoggingReviewAgent(base_agent)
    result = await agent.run("prompt")
    # LoggingReviewAgent returns base agent's result directly (not wrapped in FlujoAgentResult)
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
    # LoggingReviewAgent returns base agent's result directly (not wrapped in FlujoAgentResult)
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
    # LoggingReviewAgent returns base agent's result directly (not wrapped in FlujoAgentResult)
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

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))
    from flujo.agents import make_agent_async

    wrapper = make_agent_async("openai:gpt-4o", "sys", str)
    assert wrapper is not None


def test_make_agent_async_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("ORCH_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from flujo.infra import settings as settings_mod

    settings_mod.anthropic_api_key = None
    from flujo.agents import make_agent_async
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
        agent=mock_pydantic_ai_agent,
        max_retries=5,
        timeout=10,
        model_name="custom_test_model",
    )
    assert wrapper._max_retries == 5
    assert wrapper._timeout_seconds == 10
    assert wrapper._model_name == "custom_test_model"
    assert wrapper._agent is mock_pydantic_ai_agent


def test_async_agent_wrapper_init_default_timeout(
    mock_pydantic_ai_agent: MagicMock,
) -> None:
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


def test_async_agent_wrapper_init_invalid_timeout_type(
    mock_pydantic_ai_agent: MagicMock,
) -> None:
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
async def test_async_agent_wrapper_runtime_timeout(
    mock_pydantic_ai_agent: MagicMock,
    monkeypatch,
) -> None:
    async def fake_wait_for(awaitable, *args, **kwargs):
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    mock_pydantic_ai_agent.run = AsyncMock(return_value="unused")
    wrapper = AsyncAgentWrapper(agent=mock_pydantic_ai_agent, timeout=1, max_retries=1)
    with pytest.raises(OrchestratorRetryError) as exc_info:
        await wrapper.run_async("prompt")
    assert "timed out" in str(exc_info.value).lower() or "TimeoutError" in str(exc_info.value)
    # Note: With the adapter pattern, the agent's run method may not be called
    # when timeout happens immediately (before the adapter can call the agent).
    # The key test is that the timeout is properly caught and raised.


def test_make_self_improvement_agent_uses_settings_default(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.agents.wrapper.make_agent_async",
        fake_make,
    )
    monkeypatch.setattr(
        "flujo.infra.settings.default_self_improvement_model",
        "model_from_settings",
    )
    from flujo.agents import make_self_improvement_agent

    make_self_improvement_agent()
    assert called["model"] == "model_from_settings"


def test_make_repair_agent_uses_settings_default(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type, **kwargs) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.agents.wrapper.make_agent_async",
        fake_make,
    )
    monkeypatch.setattr(
        "flujo.infra.settings.default_repair_model",
        "model_from_settings",
    )
    from flujo.agents import make_repair_agent

    make_repair_agent()
    assert called["model"] == "model_from_settings"


def test_make_self_improvement_agent_uses_override_model(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.agents.wrapper.make_agent_async",
        fake_make,
    )
    from flujo.agents import make_self_improvement_agent

    make_self_improvement_agent(model="override_model")
    assert called["model"] == "override_model"


def test_make_repair_agent_uses_override_model(monkeypatch) -> None:
    called: dict[str, str] = {}

    def fake_make(model: str, system_prompt: str, output_type: type, **kwargs) -> None:
        called["model"] = model
        return MagicMock()

    monkeypatch.setattr(
        "flujo.agents.wrapper.make_agent_async",
        fake_make,
    )
    from flujo.agents import make_repair_agent

    make_repair_agent(model="override_model")
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


@pytest.mark.asyncio
async def test_make_agent_async_no_extra_processors(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    wrapper = make_agent_async("openai:gpt-4o", "sys", Checklist)
    assert wrapper.processors.output_processors == []


@pytest.mark.asyncio
async def test_make_agent_async_custom_processor_order(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    class DummyProc:
        name = "dummy"

        async def process(self, data, context=None):
            return data

    procs = AgentProcessors(output_processors=[DummyProc()])
    wrapper = make_agent_async("openai:gpt-4o", "sys", Checklist, processors=procs)
    names = [p.name for p in wrapper.processors.output_processors]
    assert names == ["dummy"]


@pytest.mark.asyncio
async def test_pydantic_output_parsed_by_agent(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod
    from flujo.agents.adapters.pydantic_ai_adapter import PydanticAIAdapter

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    raw = 'Here you go:\n```json\n{"items": []}\n```'

    class ParsingStubAgent:
        output_type = Checklist

        def __init__(self, text: str) -> None:
            self.text = text

        async def run(self, *_args, **_kwargs):
            import re
            import json

            m = re.search(r"\{.*\}", self.text, re.DOTALL)
            return Checklist(**json.loads(m.group(0)))

    wrapper = make_agent_async("openai:gpt-4o", "sys", Checklist)
    stub_agent = ParsingStubAgent(raw)
    wrapper._agent = stub_agent
    # Also update the adapter to use the stub agent (adapter pattern)
    wrapper._adapter = PydanticAIAdapter(stub_agent)
    result = await wrapper.run_async("prompt")
    # Wrapper now returns FlujoAgentResult; access output inside
    assert isinstance(result, FlujoAgentResult)
    assert isinstance(result.output, Checklist)
    assert result.output.items == []


@pytest.mark.asyncio
async def test_make_agent_async_type_adapter(monkeypatch) -> None:
    """Ensure TypeAdapter instances are unwrapped correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    class MyModel(BaseModel):
        value: int

    wrapper = make_agent_async("openai:gpt-4o", "sys", TypeAdapter(List[MyModel]))
    assert wrapper.target_output_type == List[MyModel]


@pytest.mark.asyncio
async def test_make_agent_async_type_adapter_complex_nested(monkeypatch) -> None:
    """Test TypeAdapter with complex nested types like List[JSONObject]."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    class MyModel(BaseModel):
        value: int
        name: str

    # Test with complex nested type
    complex_type = TypeAdapter(List[Dict[str, MyModel]])
    wrapper = make_agent_async("openai:gpt-4o", "sys", complex_type)
    assert wrapper.target_output_type == List[Dict[str, MyModel]]


@pytest.mark.asyncio
async def test_make_agent_async_type_adapter_union_types(monkeypatch) -> None:
    """Test TypeAdapter with Union types."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from flujo.infra import settings as settings_mod

    monkeypatch.setattr(settings_mod, "openai_api_key", SecretStr("test-key"))

    class ModelA(BaseModel):
        value: int

    class ModelB(BaseModel):
        text: str

    # Test with Union type
    union_type = TypeAdapter(Union[ModelA, ModelB])
    wrapper = make_agent_async("openai:gpt-4o", "sys", union_type)
    assert wrapper.target_output_type == Union[ModelA, ModelB]


def test_unwrap_type_adapter_function() -> None:
    """Test the _unwrap_type_adapter helper function directly."""
    from flujo.agents import _unwrap_type_adapter

    class MyModel(BaseModel):
        value: int

    # Test with TypeAdapter instance
    type_adapter = TypeAdapter(List[MyModel])
    unwrapped = _unwrap_type_adapter(type_adapter)
    assert unwrapped == List[MyModel]

    # Test with regular type (should pass through unchanged)
    regular_type = List[MyModel]
    unwrapped = _unwrap_type_adapter(regular_type)
    assert unwrapped == List[MyModel]

    # Test with primitive type
    unwrapped = _unwrap_type_adapter(str)
    assert unwrapped is str
