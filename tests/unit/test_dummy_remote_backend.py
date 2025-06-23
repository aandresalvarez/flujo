import pytest
from pydantic import BaseModel

from flujo import Flujo, Step
from flujo.testing.utils import DummyRemoteBackend, gather_result


class Ctx(BaseModel):
    count: int = 0


class IncrementAgent:
    async def run(self, data: int, *, pipeline_context: Ctx | None = None) -> int:
        if pipeline_context is not None:
            pipeline_context.count += 1
        return data + 1


class Nested(BaseModel):
    foo: str
    bar: int


class Container(BaseModel):
    nested: Nested
    items: list[int]


class EchoAgent:
    async def run(self, data: Container) -> Container:
        return data


@pytest.mark.asyncio
async def test_dummy_remote_backend_preserves_context() -> None:
    step1 = Step("a", IncrementAgent())
    step2 = Step("b", IncrementAgent())
    backend = DummyRemoteBackend()
    runner = Flujo(
        step1 >> step2,
        backend=backend,
        context_model=Ctx,
        initial_context_data={"count": 0},
    )
    result = await gather_result(runner, 1)
    assert backend.call_counter == 2
    assert isinstance(result.final_pipeline_context, Ctx)
    assert result.final_pipeline_context.count == 2


@pytest.mark.asyncio
async def test_dummy_remote_backend_roundtrip_complex_input() -> None:
    step = Step("echo", EchoAgent())
    backend = DummyRemoteBackend()
    runner = Flujo(step, backend=backend)
    payload = Container(nested=Nested(foo="hi", bar=42), items=[1, 2, 3])
    result = await gather_result(runner, payload)
    returned = result.step_history[0].output
    assert isinstance(returned, Container)
    assert returned.model_dump() == payload.model_dump()
