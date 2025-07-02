import doctest
from pydantic import BaseModel
import pytest

from flujo import Flujo
from flujo.domain import adapter_step, step


class ComplexInput(BaseModel):
    text: str
    length: int


@adapter_step
async def adapt(text: str) -> ComplexInput:
    return ComplexInput(text=text, length=len(text))


@step
async def follow(data: ComplexInput) -> int:
    return data.length


@pytest.mark.asyncio
async def test_adapter_pipeline_runs() -> None:
    pipeline = adapt >> follow
    runner = Flujo(pipeline)
    result = None
    async for item in runner.run_async("abc"):
        result = item
    assert result is not None
    assert result.step_history[-1].output == 3


def test_is_adapter_meta() -> None:
    assert adapt.meta.get("is_adapter") is True


def example_adapter_step():
    """
    Example of using adapter_step to create a step from a function.

    >>> from flujo import Flujo
    >>> from flujo.domain import adapter_step, step
    >>>
    >>> @adapter_step
    ... async def add_one(x: int) -> int:
    ...     return x + 1
    >>>
    >>> @step
    ... async def double(x: int) -> int:
    ...     return x * 2
    >>>
    >>> # Use it in a pipeline
    >>> pipeline = add_one >> double
    >>> runner = Flujo(pipeline)
    >>> # Note: In real usage, you would call: result = await runner.run(5)
    >>> # result.final_output would be 12
    """
    pass


def test_docstring_example() -> None:
    import sys

    failures, _ = doctest.testmod(sys.modules[__name__], verbose=False)
    assert failures == 0
