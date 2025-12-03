import pytest

from unittest.mock import MagicMock
from typing import Any
from flujo.domain.models import BaseModel

from flujo import Step
from flujo.domain import AppResources
from flujo.testing.utils import gather_result
from flujo.domain.plugins import ValidationPlugin, PluginOutcome
from flujo.domain.agent_protocol import AsyncAgentProtocol
from tests.conftest import create_test_flujo

import uuid
from pathlib import Path


class MyResources(AppResources):
    db_conn: MagicMock
    api_client: MagicMock


class MyContext(BaseModel):
    run_id: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, name, value):
        if name == "run_id":
            super().__setattr__(name, value)


class ResourceUsingAgent(AsyncAgentProtocol):
    async def run(self, data: str, *, resources: MyResources, **kwargs) -> str:
        resources.db_conn.query(f"SELECT * FROM {data}")
        return f"queried_{data}"


class ResourceUsingPlugin(ValidationPlugin):
    async def validate(self, data: Any, *, resources: MyResources, **kwargs) -> PluginOutcome:
        # Handle both string and dict inputs
        if isinstance(data, str):
            output_value = data
        elif isinstance(data, dict) and "output" in data:
            output_value = data["output"]
        else:
            output_value = str(data)

        resources.api_client.post("/validate", json=output_value)
        return PluginOutcome(success=True)


class ContextAndResourceAgent(AsyncAgentProtocol):
    async def run(
        self,
        data: str,
        *,
        context: MyContext,
        resources: MyResources,
        **kwargs,
    ) -> str:
        context.run_id = "modified"
        resources.db_conn.query(f"Log from {context.run_id}")
        return "context_and_resource_used"


@pytest.fixture
def mock_resources() -> MyResources:
    return MyResources(db_conn=MagicMock(), api_client=MagicMock())


@pytest.mark.asyncio
async def test_resources_passed_to_agent(mock_resources: MyResources):
    pipeline = Step.model_validate({"name": "query_step", "agent": ResourceUsingAgent()})
    runner = create_test_flujo(pipeline, resources=mock_resources)

    await gather_result(runner, "users")

    mock_resources.db_conn.query.assert_called_once_with("SELECT * FROM users")


@pytest.mark.asyncio
async def test_resources_passed_to_plugin(mock_resources: MyResources):
    plugin = ResourceUsingPlugin()
    step = Step.model_validate(
        {"name": "plugin_step", "agent": ResourceUsingAgent(), "plugins": [(plugin, 0)]}
    )
    runner = create_test_flujo(step, resources=mock_resources)

    result = await gather_result(runner, "products")

    assert result.step_history[0].success
    mock_resources.api_client.post.assert_called_once_with("/validate", json="queried_products")


@pytest.mark.asyncio
async def test_resource_instance_is_shared_across_steps(mock_resources: MyResources):
    pipeline = Step.model_validate(
        {"name": "step1", "agent": ResourceUsingAgent()}
    ) >> Step.model_validate({"name": "step2", "agent": ResourceUsingAgent()})
    runner = create_test_flujo(pipeline, resources=mock_resources)

    await gather_result(runner, "orders")

    assert mock_resources.db_conn.query.call_count == 2
    mock_resources.db_conn.query.assert_any_call("SELECT * FROM orders")
    mock_resources.db_conn.query.assert_any_call("SELECT * FROM queried_orders")


@pytest.mark.asyncio
async def test_pipeline_with_no_resources_succeeds():
    class SimpleAgent(AsyncAgentProtocol):
        async def run(self, data: str, **kwargs) -> str:
            return "ok"

    agent = SimpleAgent()
    pipeline = Step.model_validate({"name": "simple_step", "agent": agent})

    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, "in")

    assert result.step_history[0].success
    assert result.step_history[0].output == "ok"


@pytest.mark.asyncio
@pytest.mark.slow  # Uses SQLite backend file and UUID; can linger in CI/macOS
async def test_mixing_resources_and_context(tmp_path: Path, mock_resources: MyResources):
    # Use a unique run_id and a temporary SQLite backend for isolation
    run_id = f"test_run_{uuid.uuid4()}"
    db_path = tmp_path / f"state_{run_id}.db"
    from flujo.state.backends.sqlite import SQLiteBackend

    # Use async with for proper SQLite connection cleanup
    async with SQLiteBackend(db_path) as backend:
        agent = ContextAndResourceAgent()
        step = Step.model_validate({"name": "mixed_step", "agent": agent})
        pipeline = step
        runner = create_test_flujo(
            pipeline,
            context_model=MyContext,
            initial_context_data={"run_id": run_id},
            resources=mock_resources,
            state_backend=backend,
        )

        result = await gather_result(runner, "data")

        # Check that the context was modified (indicating the agent ran)
        final_context = result.final_pipeline_context
        assert isinstance(final_context, MyContext)
        assert final_context.run_id == "modified"

        # Check that the resource was used
        mock_resources.db_conn.query.assert_called_once_with("Log from modified")

        # If step history is populated, check the output
        if hasattr(result, "step_history") and result.step_history:
            assert result.step_history[0].output == "context_and_resource_used"

    # Connection automatically closed by async with context manager
    # Manual file cleanup no longer needed - handled by context manager
