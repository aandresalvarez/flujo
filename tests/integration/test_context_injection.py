"""Integration tests for FSD-11: Signature-Aware Context Injection for Agent Execution.

This test file validates the fix for the critical bug where stateless agents created
with make_agent_async would fail with TypeError due to unconditional context injection.
"""

import pytest
from unittest.mock import AsyncMock
from flujo.domain.dsl import Pipeline, Step, StepConfig
from flujo.domain.models import PipelineContext

from tests.conftest import create_test_flujo


def get_step_result(pipeline_result):
    """Helper function to get the first step result from a pipeline result."""
    assert len(pipeline_result.step_history) > 0
    return pipeline_result.step_history[0]


class StatelessAgent:
    """Mock stateless agent that does NOT accept context parameter."""

    def __init__(self):
        self.run_mock = AsyncMock(return_value={"output": "stateless response"})

    async def run(self, data: str) -> dict:
        """Run method that does NOT accept context parameter."""
        await self.run_mock(data)
        return {"output": f"Stateless processed: {data}"}


class ContextAwareAgent:
    """Mock context-aware agent that explicitly accepts context parameter."""

    def __init__(self):
        self.run_mock = AsyncMock(return_value={"output": "context-aware response"})

    async def run(self, data: str, *, context: PipelineContext) -> dict:
        """Run method that explicitly accepts context parameter."""
        await self.run_mock(data, context=context)
        return {"output": f"Context-aware processed: {data}", "context_received": True}


class KwargsAgent:
    """Mock agent that accepts **kwargs (should receive context)."""

    def __init__(self):
        self.run_mock = AsyncMock(return_value={"output": "kwargs response"})

    async def run(self, data: str, **kwargs) -> dict:
        """Run method that accepts **kwargs."""
        await self.run_mock(data, **kwargs)
        return {
            "output": f"Kwargs processed: {data}",
            "kwargs_received": "context" in kwargs,
        }


class FailingAgent:
    """Mock agent that raises a specific error for testing error propagation."""

    async def run(self, data: str) -> dict:
        """Run method that raises a specific error."""
        raise ValueError("Internal error")


@pytest.mark.asyncio
async def test_stateless_agent_no_context_injection():
    """Test Case 1: Stateless Agent (make_agent_async wrapper)

    Given: A pipeline with a Step using an agent created via make_agent_async that is stateless.
    When: The pipeline is executed by the Flujo runner with a context model configured.
    Then: The pipeline completes successfully, and no TypeError is raised. The context is NOT passed to the agent.
    """

    # Create a stateless agent using a mock that simulates make_agent_async behavior
    # This avoids the need for a real API key in tests
    class MockAsyncAgentWrapper:
        """Mock agent that simulates the behavior of make_agent_async wrapper"""

        def __init__(self):
            self.run_mock = AsyncMock(return_value="mock response")

        async def run(self, data: str) -> str:
            """Run method that does NOT accept context parameter - simulates stateless agent"""
            await self.run_mock(data)
            return f"Mock response to: {data}"

    stateless_agent = MockAsyncAgentWrapper()

    step = Step.model_validate(
        {
            "name": "stateless_step",
            "agent": stateless_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline - this should NOT raise TypeError
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step succeeded
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify the agent was called without context parameter
    stateless_agent.run_mock.assert_called_once()
    call_args = stateless_agent.run_mock.call_args
    assert "context" not in call_args[1]  # No context parameter passed


@pytest.mark.asyncio
async def test_stateless_agent_custom_no_context_injection():
    """Test Case 1b: Custom Stateless Agent

    Given: A pipeline with a Step using a custom stateless agent.
    When: The pipeline is executed with a context model.
    Then: The pipeline completes successfully, and no TypeError is raised.
    """
    stateless_agent = StatelessAgent()

    step = Step.model_validate(
        {
            "name": "custom_stateless_step",
            "agent": stateless_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline - this should NOT raise TypeError
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step succeeded
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify the agent was called without context parameter
    stateless_agent.run_mock.assert_called_once()
    call_args = stateless_agent.run_mock.call_args
    assert "context" not in call_args[1]  # No context parameter passed


@pytest.mark.asyncio
async def test_context_aware_agent_explicit_context():
    """Test Case 2: Context-Aware Agent (Explicit context Param)

    Given: A pipeline with a Step using a custom agent class with explicit context parameter.
    When: The pipeline is executed with a context_model.
    Then: The pipeline completes successfully, and the agent correctly receives and can modify the context.
    """
    context_aware_agent = ContextAwareAgent()

    step = Step.model_validate(
        {
            "name": "context_aware_step",
            "agent": context_aware_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step succeeded
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify the agent was called with context parameter
    context_aware_agent.run_mock.assert_called_once()
    call_args = context_aware_agent.run_mock.call_args
    assert "context" in call_args[1]
    assert call_args[1]["context"].initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_context_aware_agent_kwargs():
    """Test Case 3: Context-Aware Agent (**kwargs)

    Given: A pipeline with a Step using an agent with **kwargs.
    When: The pipeline is executed with a context_model.
    Then: The pipeline completes successfully, and the agent correctly receives the context within its kwargs.
    """
    kwargs_agent = KwargsAgent()

    step = Step.model_validate(
        {
            "name": "kwargs_step",
            "agent": kwargs_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step succeeded
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify the agent was called with context in kwargs
    kwargs_agent.run_mock.assert_called_once()
    call_args = kwargs_agent.run_mock.call_args
    assert "context" in call_args[1]
    assert call_args[1]["context"].initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_error_propagation():
    """Test Case 4: Error Propagation

    Given: A pipeline with an agent that is designed to fail for a reason other than a signature mismatch.
    When: The pipeline is executed.
    Then: The final StepResult.feedback string must contain the substring "ValueError: Internal error".
    """
    failing_agent = FailingAgent()

    step = Step.model_validate(
        {
            "name": "failing_step",
            "agent": failing_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step failed
    step_result = get_step_result(result)
    assert not step_result.success

    # FR-36: Verify the feedback contains the actual error type and message
    assert "ValueError: Internal error" in step_result.feedback
    assert "Agent execution failed with ValueError" in step_result.feedback


@pytest.mark.asyncio
async def test_backward_compatibility_existing_context_aware():
    """Test that existing context-aware agents continue to work correctly.

    This ensures NFR-14: Existing pipelines with correctly defined context-aware agents
    must continue to function without any changes.
    """
    context_aware_agent = ContextAwareAgent()

    step = Step.model_validate(
        {
            "name": "backward_compat_step",
            "agent": context_aware_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the step succeeded (backward compatibility maintained)
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify the agent received context as expected
    context_aware_agent.run_mock.assert_called_once()
    call_args = context_aware_agent.run_mock.call_args
    assert "context" in call_args[1]
    assert call_args[1]["context"].initial_prompt == "test prompt"


@pytest.mark.asyncio
async def test_mixed_pipeline_stateless_and_context_aware():
    """Test a pipeline with both stateless and context-aware agents.

    This ensures that the signature-aware injection works correctly in mixed scenarios.
    """
    stateless_agent = StatelessAgent()
    context_aware_agent = ContextAwareAgent()

    step1 = Step.model_validate(
        {
            "name": "stateless_step",
            "agent": stateless_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    step2 = Step.model_validate(
        {
            "name": "context_aware_step",
            "agent": context_aware_agent,
            "config": StepConfig(max_retries=1, timeout_s=30),
        }
    )

    pipeline = Pipeline(steps=[step1, step2])
    flujo = create_test_flujo(pipeline, context_model=PipelineContext)

    # Run the pipeline
    async for result in flujo.run_async(
        "test_data", initial_context_data={"initial_prompt": "test prompt"}
    ):
        pass

    # Verify the pipeline succeeded
    step_result = get_step_result(result)
    assert step_result.success
    assert step_result.feedback is None

    # Verify stateless agent was called without context
    stateless_agent.run_mock.assert_called_once()
    stateless_call_args = stateless_agent.run_mock.call_args
    assert "context" not in stateless_call_args[1]

    # Verify context-aware agent was called with context
    context_aware_agent.run_mock.assert_called_once()
    context_call_args = context_aware_agent.run_mock.call_args
    assert "context" in context_call_args[1]
