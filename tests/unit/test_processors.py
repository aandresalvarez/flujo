import pytest
from flujo.processors import (
    AddContextVariables,
    StripMarkdownFences,
    EnforceJsonResponse,
    AgentProcessors,
)
from flujo.infra.agents import AsyncAgentWrapper
from flujo.domain.models import BaseModel
import dataclasses


class Ctx(BaseModel):
    user_id: int
    session_id: str


@dataclasses.dataclass
class Result:
    output: str


class DummyAgent:
    async def run(self, data, **kwargs):
        return Result(output=data)


@pytest.mark.asyncio
async def test_add_context_variables():
    ctx = Ctx(user_id=1, session_id="abc")
    proc = AddContextVariables(vars=["user_id", "session_id"])
    out = await proc.process("hello", ctx)
    assert "user_id: 1" in out
    assert out.endswith("hello")


@pytest.mark.asyncio
async def test_strip_markdown_fences():
    proc = StripMarkdownFences(language="json")
    text = 'before ```json\n{"a":1}\n``` after'
    result = await proc.process(text, None)
    assert result == '{"a":1}'


@pytest.mark.asyncio
async def test_enforce_json_response():
    proc = EnforceJsonResponse()
    out = await proc.process('{"a":1}', None)
    assert out == '{"a":1}'
    with pytest.raises(ValueError):
        await proc.process("not json", None)


@pytest.mark.asyncio
async def test_wrapper_applies_processors():
    processors = AgentProcessors(
        prompt_processors=[AddContextVariables(vars=["user_id"])],
        output_processors=[StripMarkdownFences(language="json")],
    )
    wrapper = AsyncAgentWrapper(DummyAgent(), processors=processors, output_type=dict)
    ctx = Ctx(user_id=5, session_id="x")
    res = await wrapper.run_async('```json\n{"k":1}\n```', context=ctx)
    assert dataclasses.is_dataclass(res)
    assert res.output == {"k": 1}
