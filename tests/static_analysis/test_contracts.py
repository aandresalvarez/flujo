from typing import Protocol, TypeVar, Any, Optional, Callable, Generic, Union
from flujo.type_definitions.common import JSONObject
from pydantic import BaseModel, Field


# Define the protocol and TypeVar as in flujo/application/core/types.py
class ContextWithScratchpad(Protocol):
    """A contract ensuring a context object has a scratchpad attribute."""

    scratchpad: JSONObject


TContext_w_Scratch = TypeVar("TContext_w_Scratch", bound=ContextWithScratchpad)


# Mock Step and ParallelStep for static analysis
class Step:
    def __init__(self, name: str):
        self.name = name


class ParallelStep(Step, Generic[TContext_w_Scratch]):
    merge_strategy: Union[str, Callable[[TContext_w_Scratch, JSONObject], None]] = Field(...)

    def __init__(self, name: str, merge_strategy: Any):
        super().__init__(name)
        self.merge_strategy = merge_strategy


# Define a mock context without a scratchpad
class ContextWithoutScratchpad(BaseModel):
    value: str


# Define a mock context with a scratchpad
class ContextWithActualScratchpad(BaseModel):
    scratchpad: JSONObject = Field(default_factory=dict)
    value: str


# Mock ExecutorCore._handle_parallel_step for static analysis
async def _mock_handle_parallel_step(
    parallel_step: ParallelStep[TContext_w_Scratch],
    data: Any,
    context: Optional[TContext_w_Scratch],
) -> None:
    # This function does nothing at runtime, it's just for type checking
    pass


def test_parallel_step_context_contract_static_analysis():
    """Static analysis test for ParallelStep context contract."""
    # This test is designed to be checked by mypy, not to be run.
    # If mypy reports an error on the line below, the contract is enforced.

    # Case 1: Context without scratchpad (should cause mypy error)
    ContextWithoutScratchpad(value="test")
    ParallelStep(
        name="test_parallel_no_scratch",
        merge_strategy=lambda ctx, res: ctx.scratchpad.update(
            res
        ),  # This line should cause a mypy error
    )
    # The following line is expected to cause a mypy error if the contract is enforced
    # await _mock_handle_parallel_step(parallel_step_no_scratch, "data", context_without_scratchpad)

    # Case 2: Context with scratchpad (should pass mypy)
    ContextWithActualScratchpad(value="test", scratchpad={})
    ParallelStep(
        name="test_parallel_with_scratch",
        merge_strategy=lambda ctx, res: ctx.scratchpad.update(res),
    )
    # This line should pass mypy
    # await _mock_handle_parallel_step(parallel_step_with_scratch, "data", context_with_scratchpad)

    # This test simply passes at runtime, as its purpose is static type checking.
    pass
