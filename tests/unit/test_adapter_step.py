import doctest
import pytest

from flujo.domain import Step, adapter_step, step
from flujo.domain.models import BaseModel
from tests.conftest import create_test_flujo


class ComplexInput(BaseModel):
    text: str
    length: int


@adapter_step(adapter_id="generic-adapter", adapter_allow="generic")
async def adapt(text: str) -> ComplexInput:
    return ComplexInput(text=text, length=len(text))


@step
async def follow(data: ComplexInput) -> int:
    return data.length


@pytest.mark.asyncio
async def test_adapter_pipeline_runs() -> None:
    pipeline = adapt >> follow
    runner = create_test_flujo(pipeline)
    result = None
    async for item in runner.run_async("abc"):
        result = item
    assert result is not None
    assert result.step_history[-1].output == 3


def test_is_adapter_meta() -> None:
    assert adapt.meta.get("is_adapter") is True


def test_adapter_without_tokens_rejected_on_model_validate() -> None:
    with pytest.raises(ValueError, match="adapter_id and adapter_allow"):
        Step.model_validate({"name": "adapt", "meta": {"is_adapter": True}})


def example_adapter_step():
    """
    Example of using adapter_step to create a step from a function.

    >>> from flujo import Flujo
    >>> from flujo.domain import adapter_step, step
    >>>
    >>> @adapter_step(adapter_id="generic-adapter", adapter_allow="generic")
    ... async def add_one(x: int) -> int:
    ...     return x + 1
    >>>
    >>> @step
    ... async def double(x: int) -> int:
    ...     return x * 2
    >>>
    >>> # Use it in a pipeline
    >>> pipeline = add_one >> double
    >>> runner = create_test_flujo(pipeline)
    >>> # Note: In real usage, you would call: result = await runner.run(5)
    >>> # result.final_output would be 12
    """
    pass


def test_docstring_example() -> None:
    import sys

    failures, _ = doctest.testmod(sys.modules[__name__], verbose=False)
    assert failures == 0
