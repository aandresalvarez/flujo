import pytest
from flujo.domain.models import BaseModel
from pydantic import model_validator

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.testing.utils import gather_result, StubAgent


class NestedModel(BaseModel):
    value: int
    name: str = "nested"


class ContextWithNesting(BaseModel):
    counter: int = 0
    nested_item: NestedModel | None = None
    list_of_items: list[NestedModel] = []

    @model_validator(mode="after")
    def check_counter_and_nested(self):
        if self.counter > 10 and self.nested_item is None:
            raise ValueError("Nested item must be present when counter > 10")
        return self


@pytest.mark.asyncio
async def test_update_nested_model() -> None:
    update_step = Step(
        "update",
        StubAgent([{"nested_item": {"value": 123}}]),
        updates_context=True,
    )

    class ReaderAgent:
        async def run(
            self, data: object | None = None, *, pipeline_context: ContextWithNesting | None = None
        ) -> int:
            assert pipeline_context is not None
            assert isinstance(pipeline_context.nested_item, NestedModel)
            return pipeline_context.nested_item.value

    read_step = Step("read", ReaderAgent())
    runner = Flujo(update_step >> read_step, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    ctx = result.final_pipeline_context
    assert isinstance(ctx.nested_item, NestedModel)
    assert ctx.nested_item.value == 123
    assert result.step_history[-1].output == 123


@pytest.mark.asyncio
async def test_update_list_of_nested_models() -> None:
    update_step = Step(
        "update_list",
        StubAgent([{"list_of_items": [{"value": 1}, {"value": 2}]}]),
        updates_context=True,
    )

    class ListReader:
        async def run(
            self, data: object | None = None, *, pipeline_context: ContextWithNesting | None = None
        ) -> list[int]:
            assert pipeline_context is not None
            for item in pipeline_context.list_of_items:
                assert isinstance(item, NestedModel)
            return [i.value for i in pipeline_context.list_of_items]

    reader = Step("reader", ListReader())
    runner = Flujo(update_step >> reader, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    assert all(isinstance(i, NestedModel) for i in result.final_pipeline_context.list_of_items)
    assert result.step_history[-1].output == [1, 2]


@pytest.mark.asyncio
async def test_invalid_field_type_fails() -> None:
    bad_step = Step(
        "bad",
        StubAgent([{"counter": "not-an-int"}]),
        updates_context=True,
    )
    runner = Flujo(bad_step, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "validation error" in step_result.feedback.lower()
    assert result.final_pipeline_context.counter == 0


@pytest.mark.asyncio
async def test_model_level_validation_failure() -> None:
    inc_step = Step(
        "inc",
        StubAgent([{"counter": 11}]),
        updates_context=True,
    )
    runner = Flujo(
        inc_step,
        context_model=ContextWithNesting,
        initial_context_data={"counter": 5},
    )
    result = await gather_result(runner, None)
    step_result = result.step_history[-1]

    assert step_result.success is False
    assert "Nested item must be present" in step_result.feedback
    assert result.final_pipeline_context.counter == 5


@pytest.mark.asyncio
async def test_incompatible_output_type_skips_update() -> None:
    step_no_update = Step("no_update", StubAgent(["hello"]), updates_context=True)
    runner = Flujo(step_no_update, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.counter == 0
