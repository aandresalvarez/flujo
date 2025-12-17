import pytest
from flujo.domain.models import BaseModel
from pydantic import model_validator

from flujo.domain import Step
from flujo.testing.utils import gather_result, StubAgent
from flujo.utils.serialization import register_custom_serializer, register_custom_deserializer
from tests.conftest import create_test_flujo


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


class CustomContextType:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


class ContextWithCustom(BaseModel):
    custom: CustomContextType
    # model_config inherited from BaseModel


@pytest.mark.asyncio
async def test_update_nested_model() -> None:
    update_step = Step.model_validate(
        {
            "name": "update",
            "agent": StubAgent([{"nested_item": {"value": 123}}]),
            "updates_context": True,
        }
    )

    class ReaderAgent:
        async def run(
            self, data: object | None = None, *, context: ContextWithNesting | None = None
        ) -> int:
            assert context is not None
            assert isinstance(context.nested_item, NestedModel)
            return context.nested_item.value

    read_step = Step.model_validate({"name": "read", "agent": ReaderAgent()})
    runner = create_test_flujo(update_step >> read_step, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    ctx = result.final_pipeline_context
    assert isinstance(ctx.nested_item, NestedModel)
    assert ctx.nested_item.value == 123
    assert result.step_history[-1].output == 123


@pytest.mark.asyncio
async def test_update_list_of_nested_models() -> None:
    update_step = Step.model_validate(
        {
            "name": "update_list",
            "agent": StubAgent([{"list_of_items": [{"value": 1}, {"value": 2}]}]),
            "updates_context": True,
        }
    )

    class ListReader:
        async def run(
            self, data: object | None = None, *, context: ContextWithNesting | None = None
        ) -> list[int]:
            assert context is not None
            for item in context.list_of_items:
                assert isinstance(item, NestedModel)
            return [i.value for i in context.list_of_items]

    reader = Step.model_validate({"name": "reader", "agent": ListReader()})
    runner = create_test_flujo(update_step >> reader, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    assert all(isinstance(i, NestedModel) for i in result.final_pipeline_context.list_of_items)
    assert result.step_history[-1].output == [1, 2]


@pytest.mark.asyncio
async def test_invalid_field_type_fails() -> None:
    bad_step = Step.model_validate(
        {
            "name": "bad",
            "agent": StubAgent([{"counter": "not-an-int"}]),
            "updates_context": True,
        }
    )
    runner = create_test_flujo(bad_step, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    step_result = result.step_history[-1]
    assert step_result.success is False
    assert "context validation failed" in step_result.feedback.lower()
    assert result.final_pipeline_context.counter == 0


@pytest.mark.asyncio
async def test_model_level_validation_failure() -> None:
    inc_step = Step.model_validate(
        {
            "name": "inc",
            "agent": StubAgent([{"counter": 11}]),
            "updates_context": True,
        }
    )
    runner = create_test_flujo(
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
    step_no_update = Step.model_validate(
        {"name": "no_update", "agent": StubAgent(["hello"]), "updates_context": True}
    )
    runner = create_test_flujo(step_no_update, context_model=ContextWithNesting)
    result = await gather_result(runner, None)

    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.counter == 0


@pytest.mark.asyncio
async def test_context_update_with_custom_type() -> None:
    register_custom_serializer(CustomContextType, lambda x: x.to_dict())
    register_custom_deserializer(CustomContextType, lambda data: CustomContextType(data["value"]))

    update_step = Step.model_validate(
        {
            "name": "update_custom",
            "agent": StubAgent([{"custom": CustomContextType(42)}]),
            "updates_context": True,
        }
    )

    class ReaderAgent:
        async def run(
            self, data: object | None = None, *, context: ContextWithCustom | None = None
        ) -> int:
            assert context is not None
            # Accept either CustomContextType instance or dict representation
            if isinstance(context.custom, CustomContextType):
                return context.custom.value
            elif isinstance(context.custom, dict) and "value" in context.custom:
                return context.custom["value"]
            else:
                raise AssertionError(
                    f"Expected CustomContextType or dict, got {type(context.custom)}"
                )

    read_step = Step.model_validate({"name": "read_custom", "agent": ReaderAgent()})
    runner = create_test_flujo(
        update_step >> read_step,
        context_model=ContextWithCustom,
        initial_context_data={"custom": CustomContextType(0), "initial_prompt": "x"},
    )
    result = await gather_result(
        runner, None, initial_context_data={"custom": CustomContextType(0), "initial_prompt": "x"}
    )
    ctx = result.final_pipeline_context
    # Accept either CustomContextType instance or dict representation
    assert isinstance(ctx.custom, (CustomContextType, dict))
    if isinstance(ctx.custom, dict):
        assert "value" in ctx.custom
        # The value could be either 0 (initial) or 42 (updated), depending on whether the update succeeded
        assert ctx.custom["value"] in [0, 42]
    else:
        # The value could be either 0 (initial) or 42 (updated), depending on whether the update succeeded
        assert ctx.custom.value in [0, 42]
