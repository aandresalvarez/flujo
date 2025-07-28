from hypothesis import given, strategies as st
from flujo.domain.dsl import Step
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo
import pytest


def make_step(name: str) -> Step:
    return Step.model_validate({"name": name, "agent": StubAgent([name])})


@given(st.integers(min_value=1, max_value=5))
@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to hypothesis deadline issues
@pytest.mark.hypothesis(deadline=500)  # Increase deadline to 500ms
async def test_random_linear_pipeline(length: int) -> None:
    steps = [make_step(str(i)) for i in range(length)]
    pipeline = steps[0]
    for step in steps[1:]:
        pipeline = pipeline >> step
    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, None)
    assert len(result.step_history) == length
