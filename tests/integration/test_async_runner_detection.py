import asyncio
import pytest

from flujo.domain import Step
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo

pipeline = Step.model_validate({"name": "test_step", "agent": StubAgent(["OK"] * 5)})


def test_run_succeeds_in_synchronous_context():
    runner = create_test_flujo(pipeline)
    result = runner.run("sync input")
    assert result.step_history[0].success
    assert result.step_history[0].output == "OK"


@pytest.mark.asyncio
async def test_run_raises_type_error_in_asynchronous_context():
    runner = create_test_flujo(pipeline)
    with pytest.raises(TypeError, match=r"Flujo.run\(\) cannot be called"):
        runner.run("async input")


@pytest.mark.asyncio
async def test_run_async_is_unaffected_and_works_correctly():
    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, "async input")
    assert result.step_history[0].success
    assert result.step_history[0].output == "OK"


def test_run_in_simulated_jupyter_environment():
    runner = create_test_flujo(pipeline)

    async def jupyter_cell_execution():
        with pytest.raises(TypeError, match=r"Flujo.run\(\) cannot be called"):
            runner.run("jupyter input")

    asyncio.run(jupyter_cell_execution())
