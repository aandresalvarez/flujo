import pytest
from flujo.domain.models import BaseModel, PipelineContext

from flujo import Step
from flujo.domain.processors import AgentProcessors
from flujo.processors import SerializePydantic
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


class AddWorld:
    name = "AddWorld"

    async def process(self, data: str, context: PipelineContext | None = None) -> str:
        return data + " world"


class DoubleOutput:
    name = "DoubleOutput"

    async def process(self, data: str, context: PipelineContext | None = None) -> str:
        return data * 2


class ContextPrefix:
    name = "CtxPrefix"

    async def process(self, data: str, context: PipelineContext | None = None) -> str:
        prefix = getattr(context, "prefix", "") if context else ""
        return f"{prefix}:{data}"


class FailingProc:
    name = "Fail"

    async def process(self, data, context: PipelineContext | None = None):
        raise RuntimeError("boom")


class Ctx(BaseModel):
    prefix: str = "P"


@pytest.mark.asyncio
async def test_prompt_processor_modifies_input() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[AddWorld()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step)
    await gather_result(runner, "hello")
    assert agent.inputs[0] == "hello world"


@pytest.mark.asyncio
async def test_output_processor_modifies_output() -> None:
    agent = StubAgent(["hi"])
    procs = AgentProcessors(output_processors=[DoubleOutput()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].output == "hihi"


@pytest.mark.asyncio
async def test_processor_receives_context() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[ContextPrefix()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step, context_model=Ctx, initial_context_data={"prefix": "X"})
    await gather_result(runner, "hello")
    assert agent.inputs[0].startswith("X:")


@pytest.mark.asyncio
async def test_failing_processor_does_not_crash() -> None:
    agent = StubAgent(["ok"])
    procs = AgentProcessors(prompt_processors=[FailingProc()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].success is False  # Processor failure should cause step failure


class User(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_serialize_pydantic_output_to_dict() -> None:
    agent = StubAgent([User(name="A", age=1)])
    procs = AgentProcessors(output_processors=[SerializePydantic()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].output == {"name": "A", "age": 1}


@pytest.mark.asyncio
async def test_serialize_pydantic_is_idempotent() -> None:
    agent = StubAgent([{"x": 1}])
    procs = AgentProcessors(output_processors=[SerializePydantic()])
    step = Step.solution(agent, processors=procs)
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    assert result.step_history[0].output == {"x": 1}
