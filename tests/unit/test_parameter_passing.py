"""Test parameter passing to ensure backward compatibility."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from flujo.application.flujo_engine import Flujo
from flujo.domain.pipeline_dsl import Pipeline, Step
from flujo.domain.models import PipelineContext
from flujo.domain.plugins import PluginOutcome


class MockAgentWithContext:
    """Mock agent that expects 'context' parameter."""

    def __init__(self):
        self.run_mock = AsyncMock(return_value={"output": "ok"})

    async def run(self, data, *, context: PipelineContext | None = None, **kwargs):
        """Run method that expects 'context' parameter."""
        await self.run_mock(data, context=context, **kwargs)
        return {"output": f"Processed: {data}"}


class MockAgentWithPipelineContext:
    """Mock agent that expects 'pipeline_context' parameter only."""

    def __init__(self):
        self.called_with = None

    async def run(self, data, *, pipeline_context: PipelineContext | None = None):
        self.called_with = pipeline_context
        return {"output": f"Processed: {data}"}


class MockPluginWithContext:
    """Mock plugin that expects 'context' parameter."""

    def __init__(self):
        self.mock = AsyncMock(return_value=PluginOutcome(success=True, feedback=None))

    async def validate(self, data, *, context: PipelineContext | None = None, **kwargs):
        """Validate method that expects 'context' parameter."""
        await self.mock(data, context=context, **kwargs)
        return PluginOutcome(success=True, feedback=None)


class MockPluginWithPipelineContext:
    """Mock plugin that expects 'pipeline_context' parameter only."""

    def __init__(self):
        self.called_with = None

    async def validate(self, data, *, pipeline_context: PipelineContext | None = None):
        self.called_with = pipeline_context
        return PluginOutcome(success=True, feedback=None)


@pytest.mark.asyncio
async def test_agent_receives_context_parameter():
    """Test that agents receive 'context' parameter when they expect it."""
    agent = MockAgentWithContext()
    step = Step(name="test_step", agent=agent, config=MagicMock(max_retries=1, timeout_s=30))

    pipeline = Pipeline(steps=[step])

    flujo = Flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass  # We only need the first result

    # Verify the agent was called with 'context' parameter
    agent.run_mock.assert_called_once()
    call_args = agent.run_mock.call_args
    assert "context" in call_args[1]
    assert call_args[1]["context"].initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_agent_receives_pipeline_context_parameter():
    """Test that agents receive 'pipeline_context' parameter when they expect it."""
    agent = MockAgentWithPipelineContext()
    step = Step(name="test_step", agent=agent, config=MagicMock(max_retries=1, timeout_s=30))

    pipeline = Pipeline(steps=[step])

    flujo = Flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass  # We only need the first result

    # Verify the agent was called with 'pipeline_context' parameter
    assert agent.called_with is not None
    assert agent.called_with.initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_plugin_receives_context_parameter():
    """Test that plugins receive 'context' parameter when they expect it."""
    agent = MockAgentWithContext()
    plugin = MockPluginWithContext()

    step = Step(
        name="test_step",
        agent=agent,
        plugins=[(plugin, 1)],
        config=MagicMock(max_retries=1, timeout_s=30),
    )

    pipeline = Pipeline(steps=[step])

    flujo = Flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass  # We only need the first result

    # Verify the plugin was called with 'context' parameter
    plugin.mock.assert_called_once()
    call_args = plugin.mock.call_args
    assert "context" in call_args[1]
    assert call_args[1]["context"].initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_plugin_receives_pipeline_context_parameter():
    """Test that plugins receive 'pipeline_context' parameter when they expect it."""
    agent = MockAgentWithContext()
    plugin = MockPluginWithPipelineContext()

    step = Step(
        name="test_step",
        agent=agent,
        plugins=[(plugin, 1)],
        config=MagicMock(max_retries=1, timeout_s=30),
    )

    pipeline = Pipeline(steps=[step])

    flujo = Flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass  # We only need the first result

    # Verify the plugin was called with 'pipeline_context' parameter
    assert plugin.called_with is not None
    assert plugin.called_with.initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_parameter_priority_context_over_pipeline_context():
    """Test that 'context' parameter is prioritized over 'pipeline_context' when both are accepted."""
    agent = MockAgentWithContext()
    # Modify the agent to accept both parameters
    agent.run = AsyncMock()

    # Fix recursion: use a separate mock to track calls
    call_log = {}

    async def run_with_both(
        data,
        *,
        context: PipelineContext | None = None,
        pipeline_context: PipelineContext | None = None,
        **kwargs,
    ) -> dict:
        call_log["context"] = context
        call_log["pipeline_context"] = pipeline_context
        return {"output": f"Processed: {data}"}

    agent.run = run_with_both

    step = Step(name="test_step", agent=agent, config=MagicMock(max_retries=1, timeout_s=30))

    pipeline = Pipeline(steps=[step])

    flujo = Flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass  # We only need the first result

    # Verify 'context' is prioritized
    assert call_log["context"] is not None
    assert call_log["context"].initial_prompt == "test prompt"
    assert call_log["pipeline_context"] is None
