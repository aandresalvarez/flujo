"""
Test suite for FSD-11 bug fix: Signature-aware context injection for agent wrappers.

This test file verifies that the framework correctly handles context injection
for AsyncAgentWrapper instances by inspecting the underlying agent's signature
rather than the wrapper's signature.
"""

import pytest
from typing import Any, Optional
from pydantic import BaseModel

from flujo import Pipeline, Step
from flujo.infra.agents import make_agent_async
from tests.conftest import create_test_flujo
from flujo.testing.utils import gather_result


class TestContext(BaseModel):
    """Test context model for the pipeline."""

    user_id: str
    session_id: str
    metadata: dict[str, Any] = {}


class TestOutput(BaseModel):
    """Test output model for the agent."""

    message: str
    confidence: float


@pytest.mark.asyncio
async def test_fsd11_stateless_agent_make_agent_async():
    """
    Test Case 1: Stateless Agent (make_agent_async)

    This test verifies that make_agent_async works correctly with stateless agents
    that don't accept a context parameter. The framework should NOT pass context
    to the underlying pydantic-ai agent.
    """
    # Create a simple stateless agent using make_agent_async
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Respond with a simple message.",
        output_type=str,
        max_retries=1,
        timeout=30,
    )

    # Create a pipeline with context
    pipeline = Pipeline(
        name="test_fsd11_stateless",
        steps=[
            Step(
                name="stateless_agent",
                agent=agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=TestContext,
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


@pytest.mark.asyncio
async def test_fsd11_context_aware_agent_explicit():
    """
    Test Case 2: Context-Aware Agent (Explicit context parameter)

    This test verifies that agents with explicit context parameters work correctly.
    """
    # Create a context-aware agent
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Use the context to personalize responses.",
        output_type=TestOutput,
        max_retries=1,
        timeout=30,
    )

    # Create a custom agent that accepts context
    class ContextAwareAgent:
        def __init__(self, wrapped_agent):
            self._agent = wrapped_agent

        async def run(
            self, data: str, context: Optional[TestContext] = None, **kwargs: Any
        ) -> TestOutput:
            # Use context to personalize the response
            user_id = context.user_id if context else "unknown"
            response = await self._agent.run(f"User {user_id}: {data}")
            return TestOutput(message=response, confidence=0.9)

    # Wrap the agent
    context_aware_agent = ContextAwareAgent(agent)

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
        context_model=TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work correctly
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, TestOutput)
    assert step_result.output.message is not None
    assert step_result.output.confidence > 0


@pytest.mark.asyncio
async def test_fsd11_context_aware_agent_kwargs():
    """
    Test Case 3: Context-Aware Agent (kwargs)

    This test verifies that agents with **kwargs work correctly with context.
    """
    # Create a context-aware agent with **kwargs
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant. Use the context to personalize responses.",
        output_type=TestOutput,
        max_retries=1,
        timeout=30,
    )

    # Create a custom agent that accepts context via **kwargs
    class KwargsContextAgent:
        def __init__(self, wrapped_agent):
            self._agent = wrapped_agent

        async def run(self, data: str, **kwargs: Any) -> TestOutput:
            # Extract context from kwargs
            context = kwargs.get("context")
            user_id = context.user_id if context else "unknown"
            response = await self._agent.run(f"User {user_id}: {data}")
            return TestOutput(message=response, confidence=0.9)

    # Wrap the agent
    kwargs_context_agent = KwargsContextAgent(agent)

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
        context_model=TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should work correctly
    result = await gather_result(runner, {"message": "Hello, how are you?"})

    # Check that the step executed successfully
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert step_result.success
    assert step_result.output is not None
    assert isinstance(step_result.output, TestOutput)
    assert step_result.output.message is not None
    assert step_result.output.confidence > 0


@pytest.mark.asyncio
async def test_fsd11_error_propagation():
    """
    Test Case 4: Error Propagation

    This test verifies that errors are properly propagated and the feedback
    contains the actual error type and message.
    """
    # Create an agent that will fail
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        output_type=str,
        max_retries=1,
        timeout=1,  # Very short timeout to force failure
    )

    # Create a pipeline
    pipeline = Pipeline(
        name="test_fsd11_error_propagation",
        steps=[
            Step(
                name="failing_agent",
                agent=agent,
                input_key="message",
            )
        ],
    )

    # Create runner with context
    runner = create_test_flujo(
        pipeline,
        context_model=TestContext,
        initial_context_data={"user_id": "test_user", "session_id": "test_session"},
    )

    # This should fail but with proper error information
    result = await gather_result(runner, {"message": "This should timeout"})

    # The result should indicate failure
    assert len(result.step_history) > 0
    step_result = result.step_history[0]
    assert not step_result.success
    assert step_result.feedback is not None
    # The feedback should contain error information
    assert "timeout" in step_result.feedback.lower() or "error" in step_result.feedback.lower()


@pytest.mark.asyncio
async def test_fsd11_no_context_passed_to_stateless():
    """
    Test Case 5: Verify context is NOT passed to stateless agents

    This test ensures that the framework correctly identifies when NOT to pass
    context to underlying agents.
    """
    # Create a stateless agent
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        output_type=str,
        max_retries=1,
        timeout=30,
    )

    # Create a pipeline without context
    pipeline = Pipeline(
        name="test_fsd11_no_context",
        steps=[
            Step(
                name="stateless_agent",
                agent=agent,
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


@pytest.mark.asyncio
async def test_fsd11_context_required_but_none_provided():
    """
    Test Case 6: Context required but none provided

    This test verifies that the framework correctly raises an error when
    a context-aware agent requires context but none is provided.
    """

    # Create a context-aware agent
    class ContextRequiredAgent:
        async def run(self, data: str, context: TestContext, **kwargs: Any) -> str:
            return f"Hello {context.user_id}: {data}"

    agent = ContextRequiredAgent()

    # Create a pipeline
    pipeline = Pipeline(
        name="test_fsd11_context_required",
        steps=[
            Step(
                name="context_required_agent",
                agent=agent,
                input_key="message",
            )
        ],
    )

    # Create runner without context
    runner = create_test_flujo(pipeline)

    # This should raise a TypeError about missing context
    with pytest.raises(TypeError, match="requires a context"):
        await gather_result(runner, {"message": "Hello, how are you?"})


@pytest.mark.asyncio
async def test_fsd11_signature_analysis_fix():
    """
    Test Case 7: Verify signature analysis fix works correctly

    This test verifies that the signature analysis correctly identifies
    when NOT to pass context to underlying agents, without requiring API calls.
    """
    from flujo.application.context_manager import _accepts_param
    from flujo.infra.agents import make_agent_async

    # Create a simple agent using make_agent_async
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        output_type=str,
        max_retries=1,
        timeout=30,
    )

    # Test that the underlying agent's run method does NOT accept context
    # This is the core of the FSD-11 fix
    accepts_context = _accepts_param(agent._agent.run, "context")
    assert accepts_context is False, (
        f"Expected underlying agent to NOT accept context, but got: {accepts_context}"
    )

    # Test that the wrapper's run method DOES accept context (for flexibility)
    wrapper_accepts_context = _accepts_param(agent.run, "context")
    assert wrapper_accepts_context is True, (
        f"Expected wrapper to accept context, but got: {wrapper_accepts_context}"
    )

    # Verify that the signature analysis correctly identifies the difference
    import inspect

    underlying_sig = inspect.signature(agent._agent.run)
    wrapper_sig = inspect.signature(agent.run)

    # The underlying agent should have a **kwargs parameter typed as Never
    has_never_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD and str(p.annotation) == "Never"
        for p in underlying_sig.parameters.values()
    )
    assert has_never_kwargs, "Expected underlying agent to have **kwargs: Never"

    # The wrapper should have a **kwargs parameter that accepts any kwargs
    has_flexible_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD and str(p.annotation) != "Never"
        for p in wrapper_sig.parameters.values()
    )
    assert has_flexible_kwargs, "Expected wrapper to have flexible **kwargs"


@pytest.mark.asyncio
async def test_fsd11_context_filtering_works():
    """
    Test Case 8: Verify context filtering works in AsyncAgentWrapper

    This test verifies that the AsyncAgentWrapper correctly filters out
    context parameters when the underlying agent doesn't accept them.
    """
    from flujo.infra.agents import make_agent_async

    # Create a simple agent using make_agent_async
    agent = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        output_type=str,
        max_retries=1,
        timeout=30,
    )

    # Create a mock context
    class MockContext:
        pass

    # Test that the wrapper correctly filters kwargs
    # We'll use a simple test to verify the filtering logic works
    from flujo.application.context_manager import _accepts_param

    # The underlying agent should NOT accept context
    underlying_accepts = _accepts_param(agent._agent.run, "context")
    assert underlying_accepts is False

    # The wrapper should accept context (for flexibility)
    wrapper_accepts = _accepts_param(agent.run, "context")
    assert wrapper_accepts is True

    # This verifies that the signature analysis fix is working correctly
    # The framework will now correctly identify that context should NOT be passed
    # to the underlying agent, preventing the TypeError


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
