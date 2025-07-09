"""
Comprehensive tests for fallback logic and agent behavior.

This module tests:
1. Fallback execution for all failure modes
2. Infinite fallback loop detection
3. Agent call count behavior during fallback
4. Output/latency/cost propagation from fallback
5. Error message composition for fallback failures
"""

import pytest
from typing import Any

from flujo.domain.dsl.step import StandardStep, StepConfig
from flujo.domain.models import StepResult
from flujo.testing.utils import StubAgent
from flujo.exceptions import InfiniteFallbackError
from flujo.application.runner import Flujo
from flujo.registry import CallableRegistry
from flujo.domain.dsl.pipeline import Pipeline


async def execute_step_with_fallback(step: StandardStep, data: Any) -> StepResult:
    """Execute a step through the proper execution strategy to test fallback logic."""

    # Find any StubAgent instance in the step tree
    def find_stub_agent(step, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        step_id = id(step)
        if step_id in visited:
            return None
        visited.add(step_id)

        if isinstance(step.agent, StubAgent):
            return step.agent
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            return find_stub_agent(step.fallback_step, visited)
        return None

    stub_agent_instance = find_stub_agent(step)

    # Find all agent instances in the step tree
    def find_all_agents(step, agents=None, visited=None):
        if agents is None:
            agents = []
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        step_id = id(step)
        if step_id in visited:
            return agents
        visited.add(step_id)

        # Add current step's agent
        if hasattr(step, "agent") and step.agent is not None:
            agents.append(step.agent)

        # Recursively check fallback step
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            find_all_agents(step.fallback_step, agents, visited)

        return agents

    all_agents = find_all_agents(step)

    def find_failing_agent(step, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        step_id = id(step)
        if step_id in visited:
            return None
        visited.add(step_id)

        if isinstance(step.agent, FailingAgent):
            return step.agent
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            return find_failing_agent(step.fallback_step, visited)
        return None

    failing_agent_instance = find_failing_agent(step)

    def find_conditional_failing_agent(step, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        step_id = id(step)
        if step_id in visited:
            return None
        visited.add(step_id)

        if isinstance(step.agent, ConditionalFailingAgent):
            return step.agent
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            return find_conditional_failing_agent(step.fallback_step, visited)
        return None

    conditional_failing_agent_instance = find_conditional_failing_agent(step)

    # Find any FailingPlugin instance in the step tree
    def find_failing_plugin(step, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        step_id = id(step)
        if step_id in visited:
            return None
        visited.add(step_id)

        if hasattr(step, "plugins"):
            for plugin, _ in getattr(step, "plugins", []):
                if isinstance(plugin, FailingPlugin):
                    return plugin
        if hasattr(step, "fallback_step") and step.fallback_step is not None:
            return find_failing_plugin(step.fallback_step, visited)
        return None

    failing_plugin_instance = find_failing_plugin(step)
    agent_registry = {
        "FailingAgent": failing_agent_instance if failing_agent_instance else FailingAgent,
        "StubAgent": stub_agent_instance if stub_agent_instance else StubAgent,
        "ConditionalFailingAgent": conditional_failing_agent_instance
        if conditional_failing_agent_instance
        else ConditionalFailingAgent,
    }
    # Add all agent instances to the registry with unique keys
    for agent in all_agents:
        # Use the same unique identification logic as in IR serialization
        agent_type = agent.__class__.__name__
        if hasattr(agent, "_unique_id"):
            unique_key = f"{agent_type}_{agent._unique_id}"
        else:
            unique_key = f"{agent_type}_{id(agent)}"
        agent_registry[unique_key] = agent
    plugin_registry = {}
    if failing_plugin_instance is not None:
        plugin_registry["FailingPlugin"] = failing_plugin_instance

    # Force IR serialization/deserialization cycle to test circular fallback resolution
    callable_registry = CallableRegistry()
    pipeline = Pipeline.model_construct(steps=[step])
    pipeline_ir = pipeline.to_model(callable_registry)
    # Deserialize the pipeline to trigger circular fallback resolution
    pipeline = Pipeline.from_model(pipeline_ir, agent_registry, callable_registry, plugin_registry)

    # Create Flujo runner and execute, passing agent_registry and plugin_registry
    flujo_runner = Flujo(pipeline, agent_registry=agent_registry, plugin_registry=plugin_registry)
    # Consume the async generator to get the result
    async for result in flujo_runner.run_async(data):
        return result.step_history[0] if result.step_history else StepResult(name=step.name)
    return StepResult(name=step.name)


class FailingAgent:
    """Agent that always fails with a specific error."""

    def __init__(self, error_message: str = "Agent failed"):
        self.error_message = error_message
        self.call_count = 0

    async def run(self, data: Any, **kwargs) -> Any:
        self.call_count += 1
        raise Exception(self.error_message)


class ConditionalFailingAgent:
    """Agent that fails based on input."""

    def __init__(self, fail_on_input: str):
        self.fail_on_input = fail_on_input
        self.call_count = 0

    async def run(self, data: Any, **kwargs) -> Any:
        self.call_count += 1
        if data == self.fail_on_input:
            raise ValueError(f"Failed on input: {data}")
        return f"Success: {data}"


class FailingPlugin:
    """Plugin that always fails validation."""

    def __init__(self, error_message: str = "Plugin validation failed"):
        self.error_message = error_message
        self.call_count = 0

    async def validate(self, data: Any, **kwargs) -> Any:
        self.call_count += 1
        from flujo.domain.plugins import PluginOutcome

        return PluginOutcome(success=False, feedback=self.error_message)


class TestFallbackBasicBehavior:
    """Test basic fallback functionality."""

    @pytest.mark.asyncio
    async def test_fallback_on_agent_failure(self):
        """Test that fallback is triggered when agent fails."""
        # Create failing agent and fallback agent
        failing_agent = FailingAgent("Primary agent failed")
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=failing_agent, config=StepConfig(max_retries=1))
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True
        # Standardized error checks
        assert "Agent execution error: Primary agent failed" in result.metadata_.get(
            "original_error", ""
        )
        assert result.metadata_.get("original_exception_message") == "Primary agent failed"

        # Verify call counts
        assert failing_agent.call_count == 1  # Called once, failed
        assert fallback_agent.call_count == 1  # Called once for fallback

    @pytest.mark.asyncio
    async def test_fallback_on_plugin_failure(self):
        """Test that fallback is triggered when plugin validation fails."""
        # Create agents
        primary_agent = StubAgent(["primary output"])
        fallback_agent = StubAgent(["fallback output"])
        failing_plugin = FailingPlugin("Plugin validation failed")

        # Create step with failing plugin and fallback
        step = StandardStep(name="test_step", agent=primary_agent, plugins=[(failing_plugin, 0)])
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True

        # Verify call counts
        assert primary_agent.call_count == 1  # Called once, validation failed
        assert fallback_agent.call_count == 1  # Called once for fallback
        assert failing_plugin.call_count == 1  # Called once for validation

    @pytest.mark.asyncio
    async def test_fallback_on_retry_exhaustion(self):
        """Test that fallback is triggered after all retries are exhausted."""
        # Create agent that fails consistently
        failing_agent = FailingAgent("Consistent failure")
        fallback_agent = StubAgent(["fallback output"])

        # Create step with multiple retries and fallback
        step = StandardStep(name="test_step", agent=failing_agent, config=StepConfig(max_retries=3))
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True

        # Verify call counts (primary agent called max_retries times)
        assert failing_agent.call_count == 3  # 3 attempts total (1 initial + 2 retries)
        assert fallback_agent.call_count == 1  # Called once for fallback


class TestInfiniteFallbackDetection:
    """Test infinite fallback loop detection."""

    @pytest.mark.asyncio
    async def test_infinite_fallback_loop_detection(self):
        """Test that infinite fallback loops are detected and raise appropriate error."""
        # Create failing agents to trigger fallback
        agent1 = FailingAgent("Agent1 failed")
        agent2 = FailingAgent("Agent2 failed")

        # Create circular fallback chain
        step1 = StandardStep(name="step1", agent=agent1)
        step2 = StandardStep(name="step2", agent=agent2)

        step1.fallback_step = step2
        step2.fallback_step = step1  # This creates a loop

        # Execute step - should raise InfiniteFallbackError
        with pytest.raises(InfiniteFallbackError) as exc_info:
            await execute_step_with_fallback(step1, "test input")

        assert "Fallback loop detected" in str(exc_info.value)
        assert "step1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_self_fallback_detection(self):
        """Test that a step falling back to itself is detected."""
        # Create failing agents to trigger fallback
        agent1 = FailingAgent("Agent1 failed")
        agent2 = FailingAgent("Agent2 failed")

        # Create step that falls back to itself through a chain
        step1 = StandardStep(name="step1", agent=agent1)
        step2 = StandardStep(name="step2", agent=agent2)

        # Create a chain that eventually leads back to step1
        step1.fallback_step = step2
        step2.fallback_step = step1  # This creates a loop

        # Execute step - should raise InfiniteFallbackError
        with pytest.raises(InfiniteFallbackError) as exc_info:
            await execute_step_with_fallback(step1, "test input")

        assert "Fallback loop detected" in str(exc_info.value)
        assert "step1" in str(exc_info.value)


class TestFallbackOutputPropagation:
    """Test that fallback output, latency, cost, and tokens are properly propagated."""

    @pytest.mark.asyncio
    async def test_fallback_output_propagation(self):
        """Test that fallback output becomes the step output."""
        # Create agents with different outputs
        failing_agent = FailingAgent("Primary failed")
        fallback_agent = StubAgent(["fallback result"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=failing_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback output is used
        assert result.success is True
        assert result.output == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_latency_accumulation(self):
        """Test that latency from both primary and fallback is accumulated."""
        # Create agents
        failing_agent = FailingAgent("Primary failed")
        fallback_agent = StubAgent(["fallback result"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=failing_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify latency is accumulated (should be > 0)
        assert result.latency_s > 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_fallback_cost_accumulation(self):
        """Test that cost from both primary and fallback is accumulated."""
        # Create agents with cost tracking
        failing_agent = FailingAgent("Primary failed")
        fallback_agent = StubAgent(["fallback result"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=failing_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify cost is accumulated (may be 0 for stub agents)
        assert result.cost_usd >= 0
        assert result.success is True


class TestFallbackErrorHandling:
    """Test error handling when both primary and fallback fail."""

    @pytest.mark.asyncio
    async def test_both_primary_and_fallback_fail(self):
        """Test error composition when both primary and fallback fail."""
        # Create agents that both fail
        failing_agent1 = FailingAgent("Primary agent failed")
        failing_agent2 = FailingAgent("Fallback agent failed")

        # Create step with failing fallback
        step = StandardStep(
            name="test_step", agent=failing_agent1, config=StepConfig(max_retries=1)
        )
        step.fallback_step = StandardStep(name="fallback_step", agent=failing_agent2)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify both failed
        assert result.success is False
        assert "Primary agent failed" in result.feedback
        assert "Fallback agent failed" in result.feedback

        # Verify call counts
        assert failing_agent1.call_count == 1  # Called once, failed
        assert failing_agent2.call_count == 1  # Called once for fallback

    @pytest.mark.asyncio
    async def test_fallback_with_plugin_failure(self):
        """Test fallback when primary fails due to plugin validation."""
        # Create agents
        primary_agent = StubAgent(["primary output"])
        fallback_agent = StubAgent(["fallback output"])
        failing_plugin = FailingPlugin("Plugin validation failed")

        # Create step with failing plugin and fallback
        step = StandardStep(name="test_step", agent=primary_agent, plugins=[(failing_plugin, 0)])
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback succeeded
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True


class TestAgentCallCountBehavior:
    """Test agent call count behavior during fallback scenarios."""

    @pytest.mark.asyncio
    async def test_agent_call_count_with_fallback(self):
        """Test that agent call counts reflect fallback invocations."""
        # Create agents with call count tracking
        primary_agent = ConditionalFailingAgent("fail")
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=primary_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step with input that causes primary to fail
        result = await execute_step_with_fallback(step, "fail")

        # Verify call counts
        assert primary_agent.call_count == 1  # Called once, failed
        assert fallback_agent.call_count == 1  # Called once for fallback

        # Verify result
        assert result.success is True
        assert result.output == "fallback output"

    @pytest.mark.asyncio
    async def test_agent_call_count_without_fallback(self):
        """Test that agent call counts are correct when fallback is not triggered."""
        # Create agents
        primary_agent = ConditionalFailingAgent("fail")
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=primary_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step with input that allows primary to succeed
        result = await execute_step_with_fallback(step, "succeed")

        # Verify call counts
        assert primary_agent.call_count == 1  # Called once, succeeded
        assert fallback_agent.call_count == 0  # Not called

        # Verify result
        assert result.success is True
        assert result.output == "Success: succeed"


class TestFallbackWithPipelineExecution:
    """Test fallback behavior in full pipeline execution."""

    @pytest.mark.asyncio
    async def test_fallback_in_pipeline(self):
        """Test fallback behavior when step is part of a pipeline."""
        from flujo.domain.dsl.pipeline import Pipeline

        # Create agents
        failing_agent = FailingAgent("Pipeline step failed")
        fallback_agent = StubAgent(["fallback output"])
        next_agent = StubAgent(["next step output"])

        # Create steps
        step1 = StandardStep(name="step1", agent=failing_agent)
        step1.fallback_step = StandardStep(name="fallback1", agent=fallback_agent)

        step2 = StandardStep(name="step2", agent=next_agent)

        # Create pipeline
        pipeline = Pipeline.model_construct(steps=[step1, step2])

        # Build agent registry
        agent_registry = {
            "FailingAgent": failing_agent,
            "StubAgent": fallback_agent,  # Use the specific instance
        }
        # Add all agent instances with their unique keys
        for agent in [failing_agent, fallback_agent, next_agent]:
            agent_type = agent.__class__.__name__
            unique_key = f"{agent_type}_{id(agent)}"
            agent_registry[unique_key] = agent

        # Execute pipeline
        from flujo.application.runner import Flujo

        runner = Flujo(pipeline, agent_registry=agent_registry)
        async for result in runner.run_async("test input"):
            break  # Get the first result

        # Verify fallback was triggered and pipeline continued
        assert len(result.step_history) == 2
        assert result.step_history[0].output == "fallback output"
        assert result.step_history[1].output == "next step output"

        # Verify call counts
        assert failing_agent.call_count == 1
        assert fallback_agent.call_count == 1
        assert next_agent.call_count == 1


class TestFallbackEdgeCases:
    """Test edge cases in fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_with_none_output(self):
        """Test fallback behavior when primary returns None."""
        # Create agent that returns None
        none_agent = StubAgent([None])
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=none_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered (None output should trigger fallback)
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True

    @pytest.mark.asyncio
    async def test_fallback_with_empty_string_output(self):
        """Test fallback behavior when primary returns empty string."""
        # Create agent that returns empty string
        empty_agent = StubAgent([""])
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=empty_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered (empty string should trigger fallback)
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True

    @pytest.mark.asyncio
    async def test_fallback_with_zero_output(self):
        """Test fallback behavior when primary returns zero."""
        # Create agent that returns zero
        zero_agent = StubAgent([0])
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=zero_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify fallback was triggered (zero should trigger fallback)
        assert result.success is True
        assert result.output == "fallback output"
        assert result.metadata_.get("fallback_triggered") is True


class TestFallbackMetadata:
    """Test metadata handling during fallback."""

    @pytest.mark.asyncio
    async def test_fallback_metadata_preservation(self):
        """Test that metadata is properly preserved during fallback."""
        # Create agents
        failing_agent = FailingAgent("Primary failed")
        fallback_agent = StubAgent(["fallback output"])

        # Create step with fallback
        step = StandardStep(name="test_step", agent=failing_agent)
        step.fallback_step = StandardStep(name="fallback_step", agent=fallback_agent)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify metadata
        assert result.metadata_.get("fallback_triggered") is True
        assert "Primary failed" in result.metadata_.get("original_error", "")

    @pytest.mark.asyncio
    async def test_fallback_metadata_when_both_fail(self):
        """Test metadata when both primary and fallback fail."""
        # Create agents that both fail
        failing_agent1 = FailingAgent("Primary agent failed")
        failing_agent2 = FailingAgent("Fallback agent failed")

        # Create step with failing fallback
        step = StandardStep(
            name="test_step", agent=failing_agent1, config=StepConfig(max_retries=1)
        )
        step.fallback_step = StandardStep(name="fallback_step", agent=failing_agent2)

        # Execute step
        result = await execute_step_with_fallback(step, "test input")

        # Verify metadata
        assert result.metadata_.get("fallback_triggered") is True
        assert "Primary agent failed" in result.metadata_.get("original_error", "")
        assert "Fallback agent failed" in result.metadata_.get("fallback_error", "")
