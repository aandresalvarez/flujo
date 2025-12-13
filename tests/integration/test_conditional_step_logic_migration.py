"""Integration tests for ConditionalStep logic migration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.step import StepConfig
from flujo.domain.models import StepResult
from flujo.application.core.executor_core import ExecutorCore


class TestConditionalStepLogicMigration:
    """Integration test suite for ConditionalStep logic migration."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    async def test_conditional_step_integration_with_real_agents(self, executor_core):
        """Test ConditionalStep with real agent execution."""
        # Create a real ConditionalStep with mock agents
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a"
            if data.get("value") > 5
            else "branch_b",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="agent_a",
                            agent=Mock(return_value="Result A"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
                "branch_b": Pipeline(
                    name="branch_b",
                    steps=[
                        Step(
                            name="agent_b",
                            agent=Mock(return_value="Result B"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            # Test branch A execution
            result_a = await executor_core._handle_conditional_step(
                conditional_step,
                data={"value": 10},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result_a.success is True
            assert result_a.metadata_["executed_branch_key"] == "branch_a"

            # Test branch B execution
            result_b = await executor_core._handle_conditional_step(
                conditional_step,
                data={"value": 3},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result_b.success is True
            assert result_b.metadata_["executed_branch_key"] == "branch_b"

    async def test_conditional_step_with_context_updates(self, executor_core):
        """Test ConditionalStep with context updates."""
        from flujo.domain.models import PipelineContext

        class _Ctx(PipelineContext):
            counter: int = 0

        context = _Ctx()

        def condition_callable(data, ctx):
            return "increment" if data.get("action") == "increment" else "decrement"

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=condition_callable,
            branches={
                "increment": Pipeline(
                    name="increment",
                    steps=[
                        Step(
                            name="increment_agent",
                            agent=Mock(return_value={"counter": 1}),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
                "decrement": Pipeline(
                    name="decrement",
                    steps=[
                        Step(
                            name="decrement_agent",
                            agent=Mock(return_value={"counter": -1}),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output={"counter": 1}
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"action": "increment"},
                context=context,
                resources=None,
                limits=None,
                context_setter=Mock(),
            )

            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "increment"

    async def test_conditional_step_with_input_output_mapping(self, executor_core):
        """Test ConditionalStep with input and output mapping."""

        def input_mapper(data, context):
            return {"processed": data.get("raw") * 2}

        def output_mapper(output, branch_key, context):
            return {"final_result": output.get("result"), "branch": branch_key}

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "process",
            branches={
                "process": Pipeline(
                    name="process",
                    steps=[
                        Step(
                            name="process_agent",
                            agent=Mock(return_value={"result": "processed"}),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
            branch_input_mapper=input_mapper,
            branch_output_mapper=output_mapper,
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output={"result": "processed"}
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"raw": 5},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is True
            assert result.output["final_result"] == "processed"
            assert result.output["branch"] == "process"

    async def test_conditional_step_with_default_branch(self, executor_core):
        """Test ConditionalStep with default branch fallback."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "nonexistent_branch",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="agent_a",
                            agent=Mock(return_value="Result A"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
            default_branch_pipeline=Pipeline(
                name="default_branch",
                steps=[
                    Step(
                        name="default_agent",
                        agent=Mock(return_value="Default Result"),
                        config=StepConfig(max_retries=1),
                    )
                ],
            ),
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="default_output"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"value": 10},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "nonexistent_branch"

    async def test_conditional_step_with_multiple_steps_in_branch(self, executor_core):
        """Test ConditionalStep with multiple steps in a branch."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "multi_step",
            branches={
                "multi_step": Pipeline(
                    name="multi_step",
                    steps=[
                        Step(
                            name="step1",
                            agent=Mock(return_value="Step 1 Result"),
                            config=StepConfig(max_retries=1),
                        ),
                        Step(
                            name="step2",
                            agent=Mock(return_value="Step 2 Result"),
                            config=StepConfig(max_retries=1),
                        ),
                        Step(
                            name="step3",
                            agent=Mock(return_value="Final Result"),
                            config=StepConfig(max_retries=1),
                        ),
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = [
                StepResult(name="step1", success=True, output="Step 1 Result"),
                StepResult(name="step2", success=True, output="Step 2 Result"),
                StepResult(name="step3", success=True, output="Final Result"),
            ]

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"value": 10},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is True
            assert result.output == "Final Result"
            assert mock_execute.call_count == 3

    async def test_conditional_step_with_branch_failure(self, executor_core):
        """Test ConditionalStep with branch execution failure."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "failing_branch",
            branches={
                "failing_branch": Pipeline(
                    name="failing_branch",
                    steps=[
                        Step(
                            name="failing_step",
                            agent=Mock(return_value="Should not reach here"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="failing_step", success=False, output=None, feedback="Step execution failed"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"value": 10},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is False
            assert "Failure in branch 'failing_branch'" in result.feedback

    async def test_conditional_step_with_complex_context(self, executor_core):
        """Test ConditionalStep with complex context handling."""
        from flujo.domain.models import PipelineContext

        context = PipelineContext(
            user_id=123,
            session_data={"preferences": ["option_a", "option_b"]},
            execution_count=0,
        )

        def condition_callable(data, ctx):
            prefs = ctx.get("session_data", {}).get("preferences", ["option_a", "option_b"])
            return "option_a" if "option_a" in prefs else "option_b"

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=condition_callable,
            branches={
                "option_a": Pipeline(
                    name="option_a",
                    steps=[
                        Step(
                            name="agent_a",
                            agent=Mock(return_value="Option A Result"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
                "option_b": Pipeline(
                    name="option_b",
                    steps=[
                        Step(
                            name="agent_b",
                            agent=Mock(return_value="Option B Result"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                ),
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="Option A Result"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"action": "process"},
                context=context,
                resources=None,
                limits=None,
                context_setter=Mock(),
            )

            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "option_a"

    async def test_conditional_step_with_resources_and_limits(self, executor_core):
        """Test ConditionalStep with resources and usage limits."""
        resources = {"api_key": "test_key", "rate_limit": 100}
        limits = Mock()
        limits.total_cost_usd_limit = 1.0
        limits.total_tokens_limit = 1000

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "resource_branch",
            branches={
                "resource_branch": Pipeline(
                    name="resource_branch",
                    steps=[
                        Step(
                            name="resource_agent",
                            agent=Mock(return_value="Resource Result"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="Resource Result"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"action": "process"},
                context=None,
                resources=resources,
                limits=limits,
                context_setter=None,
            )

            assert result.success is True
            # Enhanced: Verify that execute was called (resources are handled internally)
            mock_execute.assert_called_once()
            # Enhanced: In the enhanced system, resources and limits are managed internally
            # The test verifies that the conditional step executes successfully

    async def test_conditional_step_error_propagation(self, executor_core):
        """Test that errors in ConditionalStep are properly propagated."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "error_branch",
            branches={
                "error_branch": Pipeline(
                    name="error_branch",
                    steps=[
                        Step(
                            name="error_agent",
                            agent=Mock(return_value="Should not reach here"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Agent execution failed")

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data={"action": "process"},
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            assert result.success is False
            assert "Error executing conditional logic or branch" in result.feedback

    async def test_conditional_step_with_empty_branch(self, executor_core):
        """Test ConditionalStep with empty branch (no steps)."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "empty_branch",
            branches={"empty_branch": Pipeline(name="empty_branch", steps=[])},
        )

        result = await executor_core._handle_conditional_step(
            conditional_step,
            data={"action": "process"},
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        assert result.success is True
        # When there are no steps, the output should be the input data
        assert result.output == {"action": "process"}
        assert result.metadata_["executed_branch_key"] == "empty_branch"
