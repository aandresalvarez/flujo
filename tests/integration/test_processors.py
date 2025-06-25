import pytest
from pydantic import BaseModel

from flujo import Flujo, Step, AgentProcessors
from flujo.testing.utils import StubAgent, gather_result


class AddWorld:
    name = "AddWorld"

    async def process(self, data: str, context=None) -> str:
        return data + " world"


class DoubleOutput:
    name = "DoubleOutput"

    async def process(self, data: str, context=None) -> str:
        return data * 2


class ContextPrefix:
    name = "CtxPrefix"

    async def process(self, data: str, context=None) -> str:
        prefix = getattr(context, "prefix", "") if context else ""
        return f"{prefix}:{data}"


class FailingProc:
    name = "Fail"

    async def process(self, data, context=None):
        raise RuntimeError("boom")


class Ctx(BaseModel):
    prefix: str = "P"


@pytest.mark.asyncio
async def test_prompt_processor_modifies_input() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[AddWorld()])
    step = Step.solution(agent, processors=procs)
    runner = Flujo(step)
    await gather_result(runner, "hello")
    assert agent.inputs[0] == "hello world"


@pytest.mark.asyncio
async def test_output_processor_modifies_output() -> None:
    agent = StubAgent(["hi"])
    procs = AgentProcessors(output_processors=[DoubleOutput()])
    step = Step.solution(agent, processors=procs)
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].output == "hihi"


@pytest.mark.asyncio
async def test_processor_receives_context() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[ContextPrefix()])
    step = Step.solution(agent, processors=procs)
    runner = Flujo(step, context_model=Ctx, initial_context_data={"prefix": "X"})
    await gather_result(runner, "hello")
    assert agent.inputs[0].startswith("X:")


@pytest.mark.asyncio
async def test_failing_processor_does_not_crash() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[FailingProc()])
    step = Step.solution(agent, processors=procs)
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].success is True
