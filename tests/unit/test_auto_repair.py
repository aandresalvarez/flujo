import pytest
from pydantic import BaseModel, TypeAdapter
from flujo.processors.repair import DeterministicRepairProcessor
from flujo.infra.agents import AsyncAgentWrapper
from flujo.exceptions import OrchestratorError
from flujo.infra import agents as agents_mod


class Model(BaseModel):
    value: int


class FailAgent:
    output_type = Model

    async def run(self, *_args, **_kwargs):
        TypeAdapter(Model).validate_json('{"value":1} trailing')


class FailAgentEscalate:
    output_type = Model

    async def run(self, *_args, **_kwargs):
        TypeAdapter(Model).validate_json("bad")


@pytest.mark.asyncio
async def test_deterministic_processor_cleans_trailing_text() -> None:
    proc = DeterministicRepairProcessor()
    cleaned = await proc.process('{"a":1} trailing')
    assert cleaned == '{"a":1}'


@pytest.mark.asyncio
async def test_async_agent_wrapper_deterministic_repair(monkeypatch) -> None:
    wrapper = AsyncAgentWrapper(FailAgent(), max_retries=1, auto_repair=True)
    monkeypatch.setattr(
        agents_mod,
        "get_raw_output_from_exception",
        lambda exc: '{"value":1} trailing',
    )
    result = await wrapper.run_async("prompt")
    assert result.value == 1


@pytest.mark.asyncio
async def test_async_agent_wrapper_llm_repair(monkeypatch) -> None:
    wrapper = AsyncAgentWrapper(FailAgentEscalate(), max_retries=1, auto_repair=True)
    monkeypatch.setattr(
        agents_mod,
        "get_raw_output_from_exception",
        lambda exc: "bad",
    )

    async def fail_process(self, _raw):
        raise ValueError("fail")

    class DummyRepairAgent:
        async def run(self, *_a, **_k):
            return '{"value":2}'

    monkeypatch.setattr(DeterministicRepairProcessor, "process", fail_process)
    monkeypatch.setattr(agents_mod, "get_repair_agent", lambda: DummyRepairAgent())

    result = await wrapper.run_async("prompt")
    assert result.value == 2


def test_balance_removes_and_adds_braces() -> None:
    text = DeterministicRepairProcessor._balance('{"a":1}}')
    assert text == '{"a":1}'
    text = DeterministicRepairProcessor._balance('{"a":1')
    assert text == '{"a":1}'


def test_balance_ignores_braces_in_strings() -> None:
    text = DeterministicRepairProcessor._balance('{"a":"}"}')
    assert text == '{"a":"}"}'


@pytest.mark.asyncio
async def test_async_agent_wrapper_llm_repair_invalid_json(monkeypatch) -> None:
    wrapper = AsyncAgentWrapper(FailAgentEscalate(), max_retries=1, auto_repair=True)
    monkeypatch.setattr(agents_mod, "get_raw_output_from_exception", lambda exc: "bad")

    async def fail_process(self, _raw):
        raise ValueError("fail")

    class DummyRepairAgent:
        async def run(self, *_a, **_k):
            return "not json"

    monkeypatch.setattr(DeterministicRepairProcessor, "process", fail_process)
    monkeypatch.setattr(agents_mod, "get_repair_agent", lambda: DummyRepairAgent())

    with pytest.raises(OrchestratorError, match="invalid JSON"):
        await wrapper.run_async("prompt")


@pytest.mark.asyncio
async def test_repair_prompt_handles_braces(monkeypatch) -> None:
    wrapper = AsyncAgentWrapper(FailAgentEscalate(), max_retries=1, auto_repair=True)
    monkeypatch.setattr(agents_mod, "get_raw_output_from_exception", lambda exc: "bad{")

    async def fail_process(self, _raw):
        raise ValueError("fail")

    captured = {}

    class DummyRepairAgent:
        async def run(self, prompt, *_a, **_k):
            captured["prompt"] = prompt
            return '{"value":3}'

    monkeypatch.setattr(DeterministicRepairProcessor, "process", fail_process)
    monkeypatch.setattr(agents_mod, "get_repair_agent", lambda: DummyRepairAgent())

    result = await wrapper.run_async("original { brace }")
    assert result.value == 3
    assert captured["prompt"]
