import pytest
from unittest.mock import MagicMock
from flujo.domain.models import BaseModel

from flujo import Flujo, Step
from flujo.domain import AppResources
from flujo.testing.utils import gather_result
from flujo.domain.plugins import ValidationPlugin, PluginOutcome
from flujo.domain.agent_protocol import AsyncAgentProtocol

import uuid
import os
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
    async def validate(self, data: dict, *, resources: MyResources, **kwargs) -> PluginOutcome:
        resources.api_client.post("/validate", json=data["output"])
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
    runner = Flujo(pipeline, resources=mock_resources)

    await gather_result(runner, "users")

    mock_resources.db_conn.query.assert_called_once_with("SELECT * FROM users")


@pytest.mark.asyncio
async def test_resources_passed_to_plugin(mock_resources: MyResources):
    plugin = ResourceUsingPlugin()
    step = Step.model_validate(
        {"name": "plugin_step", "agent": ResourceUsingAgent(), "plugins": [(plugin, 0)]}
    )
    runner = Flujo(step, resources=mock_resources)

    result = await gather_result(runner, "products")

    assert result.step_history[0].success
    mock_resources.api_client.post.assert_called_once_with("/validate", json="queried_products")


@pytest.mark.asyncio
async def test_resource_instance_is_shared_across_steps(mock_resources: MyResources):
    pipeline = Step.model_validate(
        {"name": "step1", "agent": ResourceUsingAgent()}
    ) >> Step.model_validate({"name": "step2", "agent": ResourceUsingAgent()})
    runner = Flujo(pipeline, resources=mock_resources)

    await gather_result(runner, "orders")

    assert mock_resources.db_conn.query.call_count == 2
    mock_resources.db_conn.query.assert_any_call("SELECT * FROM orders")
    mock_resources.db_conn.query.assert_any_call("SELECT * FROM queried_orders")


@pytest.mark.asyncio
async def test_pipeline_with_no_resources_succeeds():
    agent = MagicMock(spec=AsyncAgentProtocol)
    agent.run.return_value = "ok"
    pipeline = Step.model_validate({"name": "simple_step", "agent": agent})

    runner = Flujo(pipeline)
    result = await gather_result(runner, "in")

    assert result.step_history[0].success
    assert result.step_history[0].output == "ok"


@pytest.mark.asyncio
async def test_mixing_resources_and_context(tmp_path: Path, mock_resources: MyResources):
    # Use a unique run_id and a temporary SQLite backend for isolation
    run_id = f"test_run_{uuid.uuid4()}"
    db_path = tmp_path / f"state_{run_id}.db"
    from flujo.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(db_path)

    agent = ContextAndResourceAgent()
    step = Step.model_validate({"name": "mixed_step", "agent": agent})
    pipeline = step
    runner = Flujo(
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

    # Clean up the state file
    if db_path.exists():
        os.remove(db_path)
