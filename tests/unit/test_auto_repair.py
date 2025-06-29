import pytest
from pydantic import BaseModel, TypeAdapter
from flujo.processors.repair import DeterministicRepairProcessor
from flujo.infra.agents import AsyncAgentWrapper
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
