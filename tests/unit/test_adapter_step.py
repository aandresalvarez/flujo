import doctest
from pydantic import BaseModel
import pytest

from flujo import Flujo, adapter_step, step


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


def example_adapter_step() -> None:
    """Docstring used for doctest.

    >>> from pydantic import BaseModel
    >>> from flujo import Flujo, adapter_step, step
    >>> class ComplexInput(BaseModel):
    ...     text: str
    ...     length: int
    >>> @adapter_step
    ... async def build_input(data: str) -> ComplexInput:
    ...     return ComplexInput(text=data, length=len(data))
    >>> @step
    ... async def summarize(inp: ComplexInput) -> str:
    ...     return inp.text[:3]
    >>> Flujo(build_input >> summarize).run("hello").step_history[-1].output
    'hel'
    """


def test_docstring_example() -> None:
    import sys
    failures, _ = doctest.testmod(sys.modules[__name__], verbose=False)
    assert failures == 0
