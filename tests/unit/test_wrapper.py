import asyncio
import sys
from types import SimpleNamespace
from typing import Any, List

import pytest

# We assume pytest-asyncio is available in the repository.
pytestmark = pytest.mark.asyncio

# Resolve import path for the wrapper under test.
# Prefer the commonly used path; update if your repo uses a different layout.
try:
    from flujo.infra.agents.wrapper import AsyncAgentWrapper, make_agent_async
except Exception:
    # Fallback path if module is nested differently in repo
    from flujo.agents.wrapper import AsyncAgentWrapper, make_agent_async  # type: ignore


class DummyPydanticModel:
    """Simple stand-in for pydantic BaseModel used only for .model_dump behavior."""
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class DummyAgent:
    """A minimal async agent stub exposing .run and attributes used by wrapper."""
    def __init__(self, output=None, raise_exc: Exception | None = None, usage_data=None, output_type=Any):
        self._output = output
        self._raise = raise_exc
        self._usage_data = usage_data
        self.output_type = output_type
        self.model = "dummy:model"

    async def run(self, *args, **kwargs):
        if self._raise:
            raise self._raise
        # Simulate AgentRunResult-like object if output is a structure; else return primitive
        if isinstance(self._output, SimpleNamespace):  # already a prepared object
            return self._output
        return self._output

    def usage(self):
        return self._usage_data


class UsageCarrier:
    """Simulate an AgentRunResult-like response having .output and .usage()."""
    def __init__(self, output, usage):
        self.output = output
        self._usage = usage

    def usage(self):
        return self._usage


class DummyProcessor:
    """Async processor with process method for prompt or output."""
    def __init__(self, name: str, record: List[str], transform=lambda x, ctx: x):
        self.name = name
        self._record = record
        self._transform = transform

    async def process(self, data, context):
        self._record.append(f"{self.name}:{data!r}:{context!r}")
        return self._transform(data, context)


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    # Patch flujo.infra.settings.settings.agent_timeout referenced in __init__
    class S:
        agent_timeout = 30

    monkeypatch.setitem(globals(), "_dummy", None)  # no-op to keep flake8 happy
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "0")
    try:
        import flujo.infra.settings as settings_mod
        monkeypatch.setattr(settings_mod, "settings", S(), raising=False)
    except Exception:
        # If the exact module path doesn't exist during tests, patch import in wrapper import scope
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "flujo.infra.settings":
                mod = SimpleNamespace(settings=S())
                return mod
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
    yield


@pytest.fixture
def processors_record():
    return []


@pytest.fixture
def processors(processors_record):
    # Build AgentProcessors-like object with prompt and output processors lists.
    # We avoid importing actual class to keep tests isolated.
    class P:
        def __init__(self, prompt=None, output=None):
            self.prompt_processors = prompt or []
            self.output_processors = output or []

    return P


@pytest.fixture
def patch_accepts_param(monkeypatch):
    # Patch flujo.application.core.context_manager._accepts_param to control context filtering
    def setter(accepts: bool):
        # We need to ensure that the import path the wrapper uses is patched.
        module_path = "flujo.application.core.context_manager"
        # Create a dummy module-like object
        obj = SimpleNamespace(_accepts_param=lambda fn, name: accepts)
        monkeypatch.setitem(sys.modules, module_path, obj)

    return setter


async def _awaitable(value):
    return value


@pytest.mark.parametrize(
    "max_retries, timeout, exc_type",
    [
        ("3", None, TypeError),
        (-1, None, ValueError),
        (1, 0, ValueError),
        (1, 2.5, TypeError),
    ],
)
async def test_init_validation_errors(max_retries, timeout, exc_type):
    agent = DummyAgent(output="ok")
    with pytest.raises(exc_type):
        AsyncAgentWrapper(agent, max_retries=max_retries, timeout=timeout)


async def test_serializes_pydantic_like_inputs(monkeypatch, processors, processors_record, patch_accepts_param):
    # Arrange
    patch_accepts_param(accepts=True)
    prompt_proc = DummyProcessor("prompt1", processors_record, transform=lambda d, c: {"prompt": d["text"].upper()})
    out_proc = DummyProcessor("out1", processors_record, transform=lambda d, c: {"result": d["value"] * 2})
    procs = processors(prompt=[prompt_proc], output=[out_proc])

    returned = {"value": 21}
    agent_resp = UsageCarrier(output=returned, usage={"tokens": 42})
    agent = DummyAgent(output=agent_resp, output_type=dict)

    # Patch asyncio.wait_for to just await without delay
    monkeypatch.setattr(asyncio, "wait_for", lambda coro, timeout=None: coro)

    w = AsyncAgentWrapper(agent, max_retries=1, processors=procs, auto_repair=False)

    # Inputs include a pydantic-like BaseModel instance which should be dumped
    inp = DummyPydanticModel({"text": "hello"})
    ctx = {"user": "abc"}

    # Act
    result = await w.run_async(inp, context=ctx)

    # Assert: output processors applied to unpacked output,
    # and usage should be preserved via wrapper object
    assert hasattr(result, "output")
    assert result.output == {"result": 42}
    assert callable(getattr(result, "usage"))
    assert result.usage() == {"tokens": 42}

    # Prompt and output processor call records
    assert any("prompt1" in r for r in processors_record)
    assert any("out1" in r for r in processors_record)


async def test_context_filtered_when_not_accepted(monkeypatch, processors, patch_accepts_param):
    # Accepts param returns False so context should be removed before calling agent
    patch_accepts_param(accepts=False)

    seen_kwargs = {}

    class InspectAgent(DummyAgent):
        async def run(self, *args, **kwargs):
            nonlocal seen_kwargs
            seen_kwargs = dict(kwargs)
            return {"value": 1}

    agent = InspectAgent(output={"value": 1})
    monkeypatch.setattr(asyncio, "wait_for", lambda coro, timeout=None: coro)
    w = AsyncAgentWrapper(agent, max_retries=1, processors=processors())

    _ = await w.run_async("ignored", context={"x": 1}, pipeline_context={"y": 2}, other=123)
    # Should not forward context/pipeline_context if not accepted
    assert "context" not in seen_kwargs and "pipeline_context" not in seen_kwargs
    # Other kwargs pass through
    assert seen_kwargs.get("other") == 123


async def test_retry_success_on_first_attempt(monkeypatch, processors):
    calls = {"n": 0}

    class FlakyAgent(DummyAgent):
        async def run(self, *args, **kwargs):
            calls["n"] += 1
            return "ok"

    agent = FlakyAgent(output="ok")
    # Ensure wait_for simply awaits
    monkeypatch.setattr(asyncio, "wait_for", lambda coro, timeout=None: coro)

    w = AsyncAgentWrapper(agent, max_retries=3, processors=processors())
    out = await w.run_async("hi")
    assert out == "ok"
    assert calls["n"] == 1  # no retries needed


async def test_timeout_raises_orchestrator_retry_error(monkeypatch, processors):
    # Simulate asyncio timeout inside wait_for
    async def raiser(coro, timeout=None):
        raise asyncio.TimeoutError("boom")

    monkeypatch.setattr(asyncio, "wait_for", raiser)

    agent = DummyAgent(output="never")
    w = AsyncAgentWrapper(agent, max_retries=1, processors=processors())
    try:
        from flujo.infra.agents.exceptions import OrchestratorRetryError as _ORE
    except ImportError:
        _ORE = None
    # We can't import exceptions reliably here; assert via string matching
    with pytest.raises(Exception) as ei:
        await w.run_async("x")
    msg = str(ei.value)
    assert "timed out" in msg.lower() or "Agent failed after" in msg or "Agent timed out" in msg


async def test_auto_repair_deterministic_success(monkeypatch, processors):
    # Arrange: make wait_for raise ValidationError causing tenacity to retry and end with RetryError
    class FakeValidationError(Exception):
        pass

    async def raising_wait_for(coro, timeout=None):
        raise FakeValidationError("bad output")

    monkeypatch.setattr(asyncio, "wait_for", raising_wait_for)

    # Patch tenacity.AsyncRetrying to run only 1 attempt then raise RetryError with last_exc
    class FakeLastAttempt:
        def exception(self):
            return FakeValidationError("bad output")

    class FakeRetryError(Exception):
        def __init__(self, last_attempt):
            self.last_attempt = last_attempt

    class FakeAsyncRetrying:
        def __aiter__(self):
            # Immediately raise RetryError when consumed
            raise FakeRetryError(FakeLastAttempt())

    import flujo.infra.agents.wrapper as wrapper_mod
    monkeypatch.setattr(wrapper_mod, "AsyncRetrying", FakeAsyncRetrying)

    # Patch agents_utils.get_raw_output_from_exception to return raw string to repair
    monkeypatch.setattr(wrapper_mod.agents_utils, "get_raw_output_from_exception", lambda e: '{"value": 41}')

    # DeterministicRepairProcessor returns cleaned JSON string
    class FakeDeterministicRepair:
        async def process(self, raw):
            return '{"value": 42}'

    monkeypatch.setattr(wrapper_mod, "DeterministicRepairProcessor", FakeDeterministicRepair)

    # TypeAdapter.validate_json should accept and return parsed value; patch _unwrap_type_adapter to pass through
    monkeypatch.setattr(wrapper_mod, "_unwrap_type_adapter", lambda t: t)

    # Build wrapper/agent
    agent = DummyAgent(output_type=dict)
    w = AsyncAgentWrapper(agent, max_retries=1, processors=processors(), auto_repair=True)

    res = await w.run_async("ignored")
    assert isinstance(res, dict)
    assert res.get("value") == 42


async def test_auto_repair_llm_success(monkeypatch, processors):
    # Arrange to force deterministic repair to fail, then succeed via LLM repair
    class FakeValidationError(Exception):
        pass

    async def raising_wait_for(coro, timeout=None):
        raise FakeValidationError("bad output")

    monkeypatch.setattr(asyncio, "wait_for", raising_wait_for)

    class FakeLastAttempt:
        def exception(self):
            return FakeValidationError("bad output")
    class FakeRetryError(Exception):
        def __init__(self, last_attempt):
            self.last_attempt = last_attempt

    class FakeAsyncRetrying:
        def __aiter__(self):
            raise FakeRetryError(FakeLastAttempt())

    import flujo.infra.agents.wrapper as wrapper_mod
    monkeypatch.setattr(wrapper_mod, "AsyncRetrying", FakeAsyncRetrying)
    monkeypatch.setattr(wrapper_mod.agents_utils, "get_raw_output_from_exception", lambda e: "raw")

    # Deterministic repair fails by raising ValidationError
    class FakeDeterministicRepair:
        async def process(self, raw):
            return '{"invalid": true}'

    monkeypatch.setattr(wrapper_mod, "DeterministicRepairProcessor", FakeDeterministicRepair)

    # cause validate_json to raise, forcing LLM repair path
    from pydantic import ValidationError as PydValidationError

    def fake_validate_json(_):
        raise PydValidationError.from_exception_data("X", [])

    class FakeTypeAdapter:
        def __init__(self, _): pass
        def validate_json(self, s): return fake_validate_json(s)
        def json_schema(self): return {"type": "object", "properties": {"value": {"type": "integer"}}}
        def validate_python(self, obj): return obj

    monkeypatch.setattr(wrapper_mod, "TypeAdapter", FakeTypeAdapter)
    monkeypatch.setattr(wrapper_mod, "_unwrap_type_adapter", lambda t: t)
    monkeypatch.setattr(wrapper_mod, "safe_serialize", lambda s: s)

    # Stub _format_repair_prompt to just return a string
    monkeypatch.setattr(wrapper_mod, "_format_repair_prompt", lambda data: "repair prompt")

    # Mock repair agent that returns a JSON string (wrapped in code fences to test stripping)
    class FakeRepairAgent:
        async def run(self, prompt):
            return "```json\n{\"value\": 123}\n```"

    def get_repair_agent():
        return FakeRepairAgent()

    # Patch the imported get_repair_agent symbol path used inside wrapper
    import types
    fake_repair_mod = types.SimpleNamespace(get_repair_agent=get_repair_agent)
    monkeypatch.setitem(sys.modules, "flujo.infra.agents.repair", fake_repair_mod)
    # Also patch local import in wrapper
    monkeypatch.setenv("DUMMY", "1")  # no-op

    # The wrapper imports get_repair_agent via "from .repair import get_repair_agent as repair_get_repair_agent"
    # So we must patch that symbol on wrapper_mod's namespace
    monkeypatch.setattr(wrapper_mod, "repair_get_repair_agent", get_repair_agent, raising=False)

    agent = DummyAgent(output_type=dict)
    w = AsyncAgentWrapper(agent, max_retries=1, processors=processors(), auto_repair=True)
    out = await w.run_async("x")
    assert isinstance(out, dict) and out["value"] == 123


async def test_auto_repair_llm_invalid_json_raises_orchestrator_error(monkeypatch, processors):
    class FakeValidationError(Exception):
        pass

    async def raising_wait_for(coro, timeout=None):
        raise FakeValidationError("bad")
    import flujo.infra.agents.wrapper as wrapper_mod

    class FakeLastAttempt:
        def exception(self): return FakeValidationError("bad")
    class FakeRetryError(Exception):
        def __init__(self, last_attempt): self.last_attempt = last_attempt
    class FakeAsyncRetrying:
        def __aiter__(self): raise FakeRetryError(FakeLastAttempt())

    monkeypatch.setattr(asyncio, "wait_for", raising_wait_for)
    monkeypatch.setattr(wrapper_mod, "AsyncRetrying", FakeAsyncRetrying)
    monkeypatch.setattr(wrapper_mod.agents_utils, "get_raw_output_from_exception", lambda e: "raw")
    class FakeDeterministicRepair:
        async def process(self, raw): return '{"invalid": true}'
    monkeypatch.setattr(wrapper_mod, "DeterministicRepairProcessor", FakeDeterministicRepair)

    from pydantic import ValidationError as PydValidationError

    def fake_validate_json(_): raise PydValidationError.from_exception_data("X", [])
    class FakeTypeAdapter:
        def __init__(self, _): pass
        def validate_json(self, s): return fake_validate_json(s)
        def json_schema(self): return {"type": "object"}
        def validate_python(self, obj): return obj

    monkeypatch.setattr(wrapper_mod, "TypeAdapter", FakeTypeAdapter)
    monkeypatch.setattr(wrapper_mod, "_unwrap_type_adapter", lambda t: t)
    monkeypatch.setattr(wrapper_mod, "safe_serialize", lambda s: s)
    monkeypatch.setattr(wrapper_mod, "_format_repair_prompt", lambda data: "repair prompt")

    class FakeRepairAgent:
        async def run(self, prompt): return "{not json}"  # invalid JSON to trigger json.JSONDecodeError
    monkeypatch.setattr(wrapper_mod, "repair_get_repair_agent", lambda: FakeRepairAgent(), raising=False)

    agent = DummyAgent(output_type=dict)
    w = AsyncAgentWrapper(agent, max_retries=1, processors=SimpleNamespace(prompt_processors=[], output_processors=[]), auto_repair=True)

    with pytest.raises(Exception) as ei:
        await w.run_async("x")
    assert "invalid JSON" in str(ei.value) or "schema validation error" in str(ei.value)


async def test_make_agent_async_prefers_infra_make_agent(monkeypatch):
    # Patch recipes checks
    path = "flujo.infra.agents.recipes"
    import sys
    import types
    fake_recipes = types.SimpleNamespace(
        _is_image_generation_model=lambda model: model.startswith("image:"),
        _attach_image_cost_post_processor=lambda agent, model: setattr(agent, "_image_cost_attached", True),
    )
    sys.modules[path] = fake_recipes

    # Provide infra make_agent returning (agent, processors)
    def infra_make_agent(model, system_prompt, output_type, processors=None, **kwargs):
        agent = SimpleNamespace(run=lambda *a, **k: _awaitable("ok"), output_type=output_type, model=model)
        final_procs = processors or SimpleNamespace(prompt_processors=[], output_processors=[])
        return agent, final_procs

    # Insert flujo.agents with make_agent
    sys.modules["flujo.agents"] = types.SimpleNamespace(make_agent=infra_make_agent)

    wrapper = await make_agent_async(
        model="image:generation-model",
        system_prompt="sys",
        output_type=dict,
        max_retries=2,
        timeout=10,
        processors=None,
        auto_repair=False,
    )
    # Verify wrapper is AsyncAgentWrapper and image post-processor attached
    assert isinstance(wrapper, AsyncAgentWrapper)
    assert getattr(wrapper._agent, "_image_cost_attached", False) is True
    assert wrapper._max_retries == 2
    assert wrapper._timeout_seconds == 10


async def test_make_agent_async_fallback_to_local_factory(monkeypatch):
    # Remove flujo.agents to trigger ImportError and fallback
    import sys
    import types
    sys.modules.pop("flujo.agents", None)

    # Patch local .factory.make_agent and _unwrap_type_adapter indirectly via wrapper module
    import flujo.infra.agents.wrapper as wrapper_mod

    def local_make_agent(model, system_prompt, output_type, processors=None, **kwargs):
        agent = SimpleNamespace(run=lambda *a, **k: _awaitable("ok2"), output_type=output_type, model=model)
        final_procs = processors or SimpleNamespace(prompt_processors=[], output_processors=[])
        return agent, final_procs

    monkeypatch.setattr(wrapper_mod, "make_agent", local_make_agent, raising=True)

    # Patch recipes similarly as previous test
    recipes_mod = types.SimpleNamespace(
        _is_image_generation_model=lambda model: False,
        _attach_image_cost_post_processor=lambda agent, model: setattr(agent, "_image_cost_attached", True),
    )
    sys.modules["flujo.infra.agents.recipes"] = recipes_mod

    wrapper = await make_agent_async(
        model="chat:model",
        system_prompt="sys",
        output_type=list,
        processors=None,
    )
    assert isinstance(wrapper, AsyncAgentWrapper)
    assert not getattr(wrapper._agent, "_image_cost_attached", False)
    # The wrapper should have been created with provided parameters
    assert wrapper._model_name == "chat:model"