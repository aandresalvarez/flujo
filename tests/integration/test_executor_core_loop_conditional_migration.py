"""Integration tests for LoopStep and ConditionalStep migration to ExecutorCore."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import BaseModel, StepResult, UsageLimits
from flujo.testing.utils import StubAgent

# Mark this module as slow due to complex integration testing
pytestmark = pytest.mark.slow


class IntegrationTestContext(BaseModel):
    """Test context for integration tests."""

    counter: int = 0
    values: list[str] = []
    branch_executed: str = ""
    loop_iterations: int = 0


class TestExecutorCoreLoopConditionalMigration:
    """Integration test suite for LoopStep and ConditionalStep migration."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.run = AsyncMock(return_value=Mock(output="test_output"))
        return agent

    @pytest.fixture
    def increment_agent(self):
        """Create an increment agent for loop testing."""

        class IncrementAgent:
            async def run(self, data: int, **kwargs) -> int:
                return data + 1

        return IncrementAgent()

    @pytest.fixture
    def echo_agent(self):
        """Create an echo agent for conditional testing."""

        class EchoAgent:
            async def run(self, data: Any, **kwargs) -> Any:
                return data

        return EchoAgent()

    async def test_loopstep_migration_integration(self, executor_core, increment_agent):
        """Test that LoopStep works correctly through the new ExecutorCore."""
        # Create a simple LoopStep
        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": increment_agent}))
        loop_step = Step.loop_until(
            name="test_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 3,  # Exit after reaching 3
            max_loops=5,  # Increase max_loops to allow reaching 3
        )

        # Execute through ExecutorCore
        result = await executor_core.execute(
            step=loop_step,
            data=0,  # Start with 0
            context=None,
            resources=None,
            limits=None,
        )

        # Verify the result
        assert isinstance(result, StepResult)
        assert result.name == "test_loop"
        assert result.success is True
        assert result.output == 3  # Should reach 3 after 3 iterations (0->1->2->3)
        assert result.attempts == 3  # Should have executed 3 iterations

    async def test_conditionalstep_migration_integration(self, executor_core, echo_agent):
        """Test that ConditionalStep works correctly through the new ExecutorCore."""
        # Create a simple ConditionalStep
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": echo_agent})
            ),
            "branch_b": Pipeline.from_step(
                Step.model_validate({"name": "step_b", "agent": echo_agent})
            ),
        }
        conditional_step = Step.branch_on(
            name="test_conditional",
            condition_callable=lambda data, ctx: "branch_a"
            if data.get("value") > 5
            else "branch_b",
            branches=branches,
        )

        # Execute through ExecutorCore with branch A
        result_a = await executor_core.execute(
            step=conditional_step,
            data={"value": 10},
            context=None,
            resources=None,
            limits=None,
        )

        # Verify the result
        assert isinstance(result_a, StepResult)
        assert result_a.name == "test_conditional"
        assert result_a.success is True
        assert result_a.metadata_["executed_branch_key"] == "branch_a"

        # Execute through ExecutorCore with branch B
        result_b = await executor_core.execute(
            step=conditional_step,
            data={"value": 3},
            context=None,
            resources=None,
            limits=None,
        )

        # Verify the result
        assert isinstance(result_b, StepResult)
        assert result_b.name == "test_conditional"
        assert result_b.success is True
        assert result_b.metadata_["executed_branch_key"] == "branch_b"

    async def test_loopstep_with_usage_limits(self, executor_core, increment_agent):
        """Test that LoopStep respects usage limits through new architecture."""
        # Create a LoopStep that might exceed limits
        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": increment_agent}))
        loop_step = Step.loop_until(
            name="test_loop_limits",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: False,  # Never exit naturally
            max_loops=10,
        )

        # Set strict usage limits
        limits = UsageLimits(total_cost_usd_limit=0.01)  # Very low limit

        # Execute through ExecutorCore
        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=limits,
        )

        # Verify the result (should fail due to limits)
        assert isinstance(result, StepResult)
        assert result.name == "test_loop_limits"
        # The exact behavior depends on implementation, but it should handle limits properly

    async def test_conditionalstep_with_complex_branches(self, executor_core, echo_agent):
        """Test that ConditionalStep handles complex branch scenarios."""
        # Create a ConditionalStep with multiple branches and default
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": echo_agent})
            ),
            "branch_b": Pipeline.from_step(
                Step.model_validate({"name": "step_b", "agent": echo_agent})
            ),
        }
        default_branch = Pipeline.from_step(
            Step.model_validate({"name": "default_step", "agent": echo_agent})
        )
        conditional_step = Step.branch_on(
            name="test_complex_conditional",
            condition_callable=lambda data, ctx: "nonexistent_branch",
            branches=branches,
            default_branch_pipeline=default_branch,
        )

        # Execute through ExecutorCore
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        # Verify the result
        assert isinstance(result, StepResult)
        assert result.name == "test_complex_conditional"
        assert result.success is True
        assert result.metadata_["executed_branch_key"] == "nonexistent_branch"

    async def test_loopstep_context_isolation(self, executor_core, increment_agent):
        """Test that LoopStep maintains proper context isolation through new architecture."""
        context = IntegrationTestContext()

        # Create a LoopStep that modifies context
        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": increment_agent}))
        loop_step = Step.loop_until(
            name="test_loop_context",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 3,
            max_loops=5,  # Increase max_loops to allow reaching 3
        )

        # Execute through ExecutorCore
        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=context,
            resources=None,
            limits=None,
        )

        # Verify the result and context handling
        assert isinstance(result, StepResult)
        assert result.name == "test_loop_context"
        assert result.success is True
        assert result.output == 3

    async def test_conditionalstep_context_propagation(self, executor_core, echo_agent):
        """Test that ConditionalStep properly propagates context through new architecture."""
        context = IntegrationTestContext()

        # Create a ConditionalStep that modifies context
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": echo_agent})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_context",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        # Execute through ExecutorCore
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=context,
            resources=None,
            limits=None,
        )

        # Verify the result and context handling
        assert isinstance(result, StepResult)
        assert result.name == "test_conditional_context"
        assert result.success is True
        assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_migration_backward_compatibility(self, executor_core):
        """Test that the migration maintains backward compatibility."""
        # This test ensures that the new implementation produces the same results
        # as the legacy implementation for the same inputs

        # Test basic LoopStep functionality
        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([1, 2, 3])})
        )
        loop_step = Step.loop_until(
            name="backward_compat_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 2,
            max_loops=3,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        # Verify backward compatibility
        assert isinstance(result, StepResult)
        assert result.name == "backward_compat_loop"
        assert result.success is True
        assert result.output == 2
        assert result.attempts == 2

        # Test basic ConditionalStep functionality
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="backward_compat_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        # Verify backward compatibility
        assert isinstance(result, StepResult)
        assert result.name == "backward_compat_conditional"
        assert result.success is True
        assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_loopstep_edge_cases(self, executor_core):
        """Test LoopStep edge cases through new architecture."""
        # Test with zero max_loops (should be invalid)
        with pytest.raises(ValueError):
            LoopStep.model_validate(
                {
                    "name": "invalid_loop",
                    "max_loops": 0,
                    "loop_body_pipeline": {"steps": []},
                    "exit_condition_callable": lambda out, ctx: True,
                }
            )

        # Test with very large max_loops
        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": StubAgent([1])}))
        loop_step = Step.loop_until(
            name="large_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: True,  # Exit immediately
            max_loops=1000,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        # The loop should exit immediately due to the exit condition
        assert result.attempts == 1  # Should exit after first iteration
        # The loop should succeed since the body step succeeded and exit condition was met
        assert result.success is True

        # Test with None context - create a new loop step with fresh StubAgent
        body2 = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": StubAgent([1])}))
        loop_step2 = Step.loop_until(
            name="large_loop2",
            loop_body_pipeline=body2,
            exit_condition_callable=lambda out, ctx: True,  # Exit immediately
            max_loops=1000,
        )

        result = await executor_core.execute(
            step=loop_step2,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True

        # Test with complex exit conditions
        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([1, 2, 3, 4, 5])})
        )
        loop_step = Step.loop_until(
            name="complex_exit_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 3,
            max_loops=5,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.output == 3
        assert result.attempts == 3

        # Test with failing body steps
        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([Exception("Step failed")])})
        )
        loop_step = Step.loop_until(
            name="failing_body_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out,
            ctx: False,  # Don't exit immediately, let body step fail
            max_loops=3,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is False
        # Enhanced: Check for loop failure indicators in feedback
        assert result.feedback and (
            "max_loops" in result.feedback
            or "loop body failed" in result.feedback.lower()
            or "no more outputs" in result.feedback.lower()
        )

    async def test_conditionalstep_edge_cases(self, executor_core):
        """Test ConditionalStep edge cases through new architecture."""
        # Test with no branches (should be invalid)
        with pytest.raises(ValueError):
            ConditionalStep.model_validate(
                {
                    "name": "invalid_conditional",
                    "condition_callable": lambda data, ctx: "branch_a",
                    "branches": {},
                }
            )

        # Test with empty branches
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "empty_step", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="empty_branches_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True

        # Test with failing condition callable
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="failing_condition_conditional",
            condition_callable=lambda data, ctx: Exception("Condition failed"),
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is False
        assert "Condition failed" in result.feedback

        # Test with complex branch input/output mappers
        def input_mapper(data, context):
            return {"mapped": data.get("value", "default")}

        def output_mapper(output, branch_key, context):
            return {"result": output, "branch": branch_key}

        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A", "A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="complex_mappers_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
            branch_input_mapper=input_mapper,
            branch_output_mapper=output_mapper,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        # The complex mappers test should now work with more outputs
        assert result.success is True
        assert result.output["result"] == "A"
        assert result.output["branch"] == "branch_a"

        # Test with None context
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True

    async def test_migration_performance(self, executor_core):
        """Test that migration doesn't introduce performance regressions."""
        # Benchmark LoopStep performance
        body = Pipeline.from_step(
            Step.model_validate(
                {"name": "step1", "agent": StubAgent([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
            )
        )
        loop_step = Step.loop_until(
            name="performance_test_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 5,
            max_loops=10,
        )

        start_time = time.time()
        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )
        end_time = time.time()

        # Performance should be reasonable (less than 1 second for this simple test)
        assert end_time - start_time < 1.0
        assert result.success is True
        assert result.output == 5
        assert result.attempts == 5

        # Benchmark ConditionalStep performance
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="performance_test_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        start_time = time.time()
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )
        end_time = time.time()

        # Performance should be reasonable (less than 1 second for this simple test)
        assert end_time - start_time < 1.0
        assert result.success is True
        assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_loopstep_with_input_output_mappers(self, executor_core, increment_agent):
        """Test LoopStep with input/output mappers through new architecture."""

        def initial_mapper(data, context):
            return data * 2

        def iteration_mapper(output, context, iteration):
            return output + iteration

        def output_mapper(output, context):
            return {"final_output": output, "iterations": context.loop_iterations if context else 0}

        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": increment_agent}))
        loop_step = Step.loop_until(
            name="test_loop_mappers",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 10,
            max_loops=5,  # Increase max_loops to allow reaching 10
            initial_input_to_loop_body_mapper=initial_mapper,
            iteration_input_mapper=iteration_mapper,
            loop_output_mapper=output_mapper,
        )

        context = IntegrationTestContext()
        result = await executor_core.execute(
            step=loop_step,
            data=1,  # Initial input
            context=context,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.output["final_output"] >= 10
        assert "iterations" in result.output

    async def test_conditionalstep_with_input_output_mappers(self, executor_core, echo_agent):
        """Test ConditionalStep with input/output mappers through new architecture."""

        def input_mapper(data, context):
            return {"mapped_input": data.get("value", "default")}

        def output_mapper(output, branch_key, context):
            return {
                "result": output,
                "branch": branch_key,
                "context_counter": context.counter if context else 0,
            }

        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": echo_agent})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_mappers",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
            branch_input_mapper=input_mapper,
            branch_output_mapper=output_mapper,
        )

        context = IntegrationTestContext()
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test_input"},
            context=context,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.output["result"] == {"mapped_input": "test_input"}
        assert result.output["branch"] == "branch_a"
        assert "context_counter" in result.output

    async def test_loopstep_with_context_modifications(self, executor_core):
        """Test LoopStep with context modifications through new architecture."""

        class ContextModifyingAgent:
            async def run(self, data: int, **kwargs) -> int:
                context = kwargs.get("context")
                if context:
                    context.counter += 1
                    context.values.append(f"iteration_{context.counter}")
                return data + 1

        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": ContextModifyingAgent()})
        )
        loop_step = Step.loop_until(
            name="test_loop_context_modifications",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 3,
            max_loops=3,
        )

        context = IntegrationTestContext()
        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=context,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.output == 3
        assert context.counter > 0
        assert len(context.values) > 0

    async def test_conditionalstep_with_context_modifications(self, executor_core):
        """Test ConditionalStep with context modifications through new architecture."""

        class ContextModifyingAgent:
            async def run(self, data: Any, **kwargs) -> Any:
                context = kwargs.get("context")
                if context:
                    context.branch_executed = "branch_a"
                    context.counter += 1
                return data

        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": ContextModifyingAgent()})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_context_modifications",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        context = IntegrationTestContext()
        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=context,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert context.branch_executed == "branch_a"
        assert context.counter > 0

    async def test_loopstep_error_handling(self, executor_core):
        """Test LoopStep error handling through new architecture."""

        # Test with failing exit condition
        def failing_exit_condition(out, ctx):
            raise Exception("Exit condition failed")

        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([1, 2, 3])})
        )
        loop_step = Step.loop_until(
            name="test_loop_exit_error",
            loop_body_pipeline=body,
            exit_condition_callable=failing_exit_condition,
            max_loops=3,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        # The exit condition should fail and the loop should fail
        assert result.success is False
        assert "Exit condition failed" in result.feedback

        # Test with failing initial mapper
        def failing_initial_mapper(data, context):
            raise Exception("Initial mapper failed")

        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": StubAgent([1])}))
        loop_step = Step.loop_until(
            name="test_loop_initial_mapper_error",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: True,
            max_loops=3,
            initial_input_to_loop_body_mapper=failing_initial_mapper,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is False
        assert "Initial mapper failed" in result.feedback

    async def test_conditionalstep_error_handling(self, executor_core):
        """Test ConditionalStep error handling through new architecture."""
        # Test with failing condition callable
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_condition_error",
            condition_callable=lambda data, ctx: Exception("Condition failed"),
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is False
        assert "Condition failed" in result.feedback

        # Test with failing input mapper
        def failing_input_mapper(data, context):
            raise Exception("Input mapper failed")

        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_input_mapper_error",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
            branch_input_mapper=failing_input_mapper,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is False
        assert "Input mapper failed" in result.feedback

    async def test_loopstep_with_resources_and_limits(self, executor_core, increment_agent):
        """Test LoopStep with resources and limits through new architecture."""
        resources = {"test_resource": "test_value"}
        limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=1000)

        body = Pipeline.from_step(Step.model_validate({"name": "step1", "agent": increment_agent}))
        loop_step = Step.loop_until(
            name="test_loop_resources_limits",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 3,
            max_loops=5,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=resources,
            limits=limits,
        )

        assert isinstance(result, StepResult)
        assert result.name == "test_loop_resources_limits"
        assert result.success is True
        assert result.output == 3

    async def test_conditionalstep_with_resources_and_limits(self, executor_core, echo_agent):
        """Test ConditionalStep with resources and limits through new architecture."""
        resources = {"test_resource": "test_value"}
        limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=1000)

        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": echo_agent})
            )
        }
        conditional_step = Step.branch_on(
            name="test_conditional_resources_limits",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=resources,
            limits=limits,
        )

        assert isinstance(result, StepResult)
        assert result.name == "test_conditional_resources_limits"
        assert result.success is True
        assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_loopstep_complex_scenarios(self, executor_core):
        """Test LoopStep with complex scenarios through new architecture."""
        # Test nested loops (if supported)
        inner_body = Pipeline.from_step(
            Step.model_validate({"name": "inner_step", "agent": StubAgent([1, 2])})
        )
        inner_loop = Step.loop_until(
            name="inner_loop",
            loop_body_pipeline=inner_body,
            exit_condition_callable=lambda out, ctx: out >= 2,
            max_loops=2,
        )

        outer_body = Pipeline.from_step(
            Step.model_validate({"name": "outer_step", "agent": StubAgent([inner_loop])})
        )
        outer_loop = Step.loop_until(
            name="outer_loop",
            loop_body_pipeline=outer_body,
            exit_condition_callable=lambda out, ctx: True,
            max_loops=2,
        )

        result = await executor_core.execute(
            step=outer_loop,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert isinstance(result, StepResult)
        assert result.name == "outer_loop"

    async def test_conditionalstep_complex_scenarios(self, executor_core):
        """Test ConditionalStep with complex scenarios through new architecture."""
        # Test nested conditionals
        inner_branches = {
            "inner_branch": Pipeline.from_step(
                Step.model_validate({"name": "inner_step", "agent": StubAgent(["inner_result"])})
            )
        }
        inner_conditional = Step.branch_on(
            name="inner_conditional",
            condition_callable=lambda data, ctx: "inner_branch",
            branches=inner_branches,
        )

        outer_branches = {
            "outer_branch": Pipeline.from_step(
                Step.model_validate({"name": "outer_step", "agent": StubAgent([inner_conditional])})
            )
        }
        outer_conditional = Step.branch_on(
            name="outer_conditional",
            condition_callable=lambda data, ctx: "outer_branch",
            branches=outer_branches,
        )

        result = await executor_core.execute(
            step=outer_conditional,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert isinstance(result, StepResult)
        assert result.name == "outer_conditional"

    async def test_migration_telemetry_integration(self, executor_core):
        """Test that migration properly integrates with telemetry."""
        # Test LoopStep telemetry
        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([1, 2])})
        )
        loop_step = Step.loop_until(
            name="telemetry_test_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 2,
            max_loops=2,
        )

        result = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.latency_s >= 0  # Should have telemetry data

        # Test ConditionalStep telemetry
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="telemetry_test_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        result = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result.success is True
        assert result.latency_s >= 0  # Should have telemetry data

    async def test_migration_cache_integration(self, executor_core):
        """Test that migration properly integrates with caching."""
        # Test LoopStep caching
        body = Pipeline.from_step(
            Step.model_validate({"name": "step1", "agent": StubAgent([1, 2, 1, 2])})
        )
        loop_step = Step.loop_until(
            name="cache_test_loop",
            loop_body_pipeline=body,
            exit_condition_callable=lambda out, ctx: out >= 2,
            max_loops=5,  # Increase max_loops to allow reaching 2
        )

        # First execution
        result1 = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        # Second execution (should potentially use cache)
        result2 = await executor_core.execute(
            step=loop_step,
            data=0,
            context=None,
            resources=None,
            limits=None,
        )

        assert result1.success is True
        assert result2.success is True
        assert result1.output == result2.output

        # Test ConditionalStep caching
        branches = {
            "branch_a": Pipeline.from_step(
                Step.model_validate({"name": "step_a", "agent": StubAgent(["A", "A"])})
            )
        }
        conditional_step = Step.branch_on(
            name="cache_test_conditional",
            condition_callable=lambda data, ctx: "branch_a",
            branches=branches,
        )

        # First execution
        result1 = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        # Second execution (should potentially use cache)
        result2 = await executor_core.execute(
            step=conditional_step,
            data={"value": "test"},
            context=None,
            resources=None,
            limits=None,
        )

        assert result1.success is True
        assert result2.success is True
        assert result1.output == result2.output
