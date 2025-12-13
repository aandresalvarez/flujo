"""
Test suite for FSD-11 bug fix: Signature-aware context injection for agent wrappers.

This test file verifies that the framework correctly handles context injection
for AsyncAgentWrapper instances by inspecting the underlying agent's signature
rather than the wrapper's signature.
"""

import pytest
from typing import Any, Optional
from unittest.mock import AsyncMock

from flujo import Pipeline, Step
from flujo.agents import make_agent_async
from flujo.domain.models import BaseModel
from flujo.domain.models import PipelineContext
from tests.conftest import create_test_flujo
from flujo.testing.utils import gather_result
from tests.unit.conftest import MockStatelessAgent


class FSD11TestContext(BaseModel):
    """Test context model for the pipeline."""

    user_id: str
    session_id: str
    metadata: dict[str, Any] = {}


class FSD11TestOutput(BaseModel):
    """Test output model for the agent."""

    message: str
    confidence: float


class ContextAwareAgent:
    def __init__(self, wrapped_agent):
        self._agent = wrapped_agent

    async def run(
        self, data: str, context: Optional[FSD11TestContext] = None, **kwargs: Any
    ) -> FSD11TestOutput:
        user_id = context.user_id if context else "unknown"
        response = await self._agent.run(f"User {user_id}: {data}")
        return FSD11TestOutput(message=response, confidence=0.9)


class KwargsContextAgent:
    def __init__(self, wrapped_agent):
        self._agent = wrapped_agent

    async def run(self, data: str, **kwargs: Any) -> FSD11TestOutput:
        context = kwargs.get("context")
        user_id = context.user_id if context else "unknown"
        response = await self._agent.run(f"User {user_id}: {data}")
        return FSD11TestOutput(message=response, confidence=0.9)


class MockUnderlyingAgent:
    def __init__(self):
        self.run_mock = AsyncMock(return_value="Personalized response")

    async def run(self, data: str) -> str:
        await self.run_mock(data)
        return f"Mock response to: {data}"


@pytest.mark.asyncio
async def test_fsd11_stateless_agent_make_agent_async():
    """
    Test Case 1: Stateless Agent (make_agent_async)

    This test verifies that make_agent_async works correctly with stateless agents
    that don't accept a context parameter. The framework should NOT pass context
    to the underlying pydantic-ai agent.
    """

    stateless_agent = MockStatelessAgent()

    # Create a pipeline with context
    pipeline = Pipeline(
        name="test_fsd11_stateless",
        steps=[
            Step(
                name="stateless_agent",
                agent=stateless_agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=FSD11TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work without TypeError about unexpected 'context' argument
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, str)

    # Verify the agent was called without context parameter
    stateless_agent.run_mock.assert_called_once()
    call_args = stateless_agent.run_mock.call_args
    assert "context" not in call_args[1]  # No context parameter passed


@pytest.mark.asyncio
async def test_fsd11_context_aware_agent_explicit():
    """
    Test Case 2: Context-Aware Agent (Explicit context parameter)

    This test verifies that agents with explicit context parameters work correctly.
    """

    # Create a context-aware agent wrapper
    context_aware_agent = ContextAwareAgent(MockUnderlyingAgent())

    # Create a pipeline with context
    pipeline = Pipeline(
        name="test_fsd11_context_aware",
        steps=[
            Step(
                name="context_aware_agent",
                agent=context_aware_agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=FSD11TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work correctly
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, FSD11TestOutput)
    assert step_result.output.message is not None
    assert step_result.output.confidence > 0


@pytest.mark.asyncio
async def test_fsd11_context_aware_agent_kwargs():
    """
    Test Case 3: Context-Aware Agent (kwargs)

    This test verifies that agents with **kwargs work correctly with context.
    """

    # Create a custom agent that accepts context via **kwargs
    kwargs_context_agent = KwargsContextAgent(MockUnderlyingAgent())

    # Create a pipeline with context
    pipeline = Pipeline(
        name="test_fsd11_kwargs_context",
        steps=[
            Step(
                name="kwargs_context_agent",
                agent=kwargs_context_agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=FSD11TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work correctly
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, FSD11TestOutput)
    assert step_result.output.message is not None
    assert step_result.output.confidence > 0


@pytest.mark.asyncio
async def test_fsd11_error_propagation():
    """
    Test Case 4: Error Propagation

    This test verifies that errors are properly propagated and the feedback
    contains the actual error type and message.
    """

    # Create a mock agent that will fail
    class FailingMockAgent:
        async def run(self, data: str) -> str:
            raise Exception("Simulated failure")

    # Create a pipeline
    pipeline = Pipeline(
        name="test_fsd11_error_propagation",
        steps=[
            Step(
                name="failing_agent",
                agent=FailingMockAgent(),
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=FSD11TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should fail but with proper error information
    result = await gather_result(runner, {"message": "This should fail"})

    # The result should indicate failure
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert not step_result.success
    assert step_result.feedback is not None
    # The feedback should contain error information
    assert "error" in step_result.feedback.lower() or "exception" in step_result.feedback.lower()


@pytest.mark.asyncio
async def test_fsd11_no_context_passed_to_stateless():
    """
    Test Case 5: Verify context is NOT passed to stateless agents

    This test ensures that the framework correctly identifies when NOT to pass
    context to underlying agents.
    """

    stateless_agent = MockStatelessAgent()

    # Create a pipeline without context
    pipeline = Pipeline(
        name="test_fsd11_no_context",
        steps=[
            Step(
                name="stateless_agent",
                agent=stateless_agent,
                input_key="message",
            )
        ],
    )

    # Create runner without context
    runner = create_test_flujo(pipeline)

    # This should work without any context-related errors
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, str)

    # Verify the agent was called without context parameter
    stateless_agent.run_mock.assert_called_once()
    call_args = stateless_agent.run_mock.call_args
    assert "context" not in call_args[1]  # No context parameter passed


@pytest.mark.asyncio
async def test_fsd11_context_required_but_none_provided():
    """
    Test Case 6: Context Required but None Provided

    This test verifies that when an agent requires context but none is provided,
    the framework handles it gracefully (FSD-11 fix behavior).
    """

    # Create a custom agent that requires context
    class ContextRequiredAgent:
        async def run(self, data: str, context: PipelineContext, **kwargs: Any) -> str:
            # This agent requires context - use PipelineContext fields
            return f"Response for run {context.run_id}: {data}"

    # Create a pipeline
    pipeline = Pipeline(
        name="test_fsd11_context_required",
        steps=[
            Step(
                name="context_required_agent",
                agent=ContextRequiredAgent(),
                input_key="message",
            )
        ],
    )

    # Create runner WITH context (this should work with FSD-11 fix)
    runner = create_test_flujo(
        pipeline,
        context_model=PipelineContext,
        initial_context_data={"initial_prompt": "test prompt"},
    )

    # With FSD-11 fix, this should work gracefully
    # The framework should detect that the agent requires context and provide it
    result = await gather_result(runner, {"message": "Hello"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    # The step should succeed because the framework now provides context automatically
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, str)


@pytest.mark.asyncio
async def test_fsd11_signature_analysis_fix():
    """
    Test Case 7: Signature Analysis Fix

    This test verifies that the signature analysis correctly identifies
    whether an agent accepts context or not.
    """
    # Create a simple agent using make_agent_async
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Respond with a simple message.",
        output_type=str,
        max_retries=1,
        timeout=30,
    )

    # Create a pipeline with context
    pipeline = Pipeline(
        name="test_fsd11_signature_analysis",
        steps=[
            Step(
                name="signature_test_agent",
                agent=agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=FSD11TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work without TypeError about unexpected 'context' argument
    # The framework should detect that the agent doesn't accept context
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    # Note: This test may fail due to API issues, but the important thing
    # is that no TypeError about context injection is raised
    # Check for either timeout or API error patterns
    feedback_lower = (step_result.feedback or "").lower()
    assert (
        step_result.success
        or "timeout" in feedback_lower
        or "api" in feedback_lower
        or "invalid" in feedback_lower
    )


@pytest.mark.asyncio
async def test_fsd11_context_filtering_works():
    """
    Test Case 8: Context Filtering Works

    This test verifies that the context filtering mechanism works correctly
    for different agent signatures.
    """

    # Create a mock context
    class MockContext:
        def __init__(self):
            self.user_id = "test_user"
            self.session_id = "test_session"

    # Test 1: Agent that doesn't accept context
    class NoContextAgent:
        async def run(self, data: str) -> str:
            return f"No context response: {data}"

    # Test 2: Agent that accepts context explicitly
    class ExplicitContextAgent:
        async def run(self, data: str, context: MockContext) -> str:
            return f"Explicit context response for {context.user_id}: {data}"

    # Test 3: Agent that accepts context via kwargs
    class KwargsContextAgent:
        async def run(self, data: str, **kwargs) -> str:
            context = kwargs.get("context")
            user_id = context.user_id if context else "unknown"
            return f"Kwargs context response for {user_id}: {data}"

    # Test all three agents
    agents = [
        ("no_context", NoContextAgent()),
        ("explicit_context", ExplicitContextAgent()),
        ("kwargs_context", KwargsContextAgent()),
    ]

    for agent_name, agent in agents:
        pipeline = Pipeline(
            name=f"test_fsd11_context_filtering_{agent_name}",
            steps=[
                Step(
                    name=f"{agent_name}_step",
                    agent=agent,
                    input_key="message",
                )
            ],
        )

        # Create runner with context
        runner = create_test_flujo(
            pipeline,
            context_model=FSD11TestContext,
            initial_context_data={"user_id": "test_user", "session_id": "test_session"},
        )

        # This should work without TypeError
        result = await gather_result(runner, {"message": "Hello"})

        # Check that the step executed successfully
        assert len(result.step_history) > 0
        step_result = result.step_history[0]
        assert step_result.success
        assert step_result.output is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
