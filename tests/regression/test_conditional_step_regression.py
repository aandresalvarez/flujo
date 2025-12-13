"""Regression tests for ConditionalStep logic migration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.step import StepConfig
from flujo.domain.models import StepResult
from flujo.application.core.executor_core import ExecutorCore

# Note: Tests that need telemetry capture should use the isolated_telemetry fixture
# which provides per-test isolation without requiring serial execution.


class TestConditionalStepRegression:
    """Regression test suite for ConditionalStep logic migration."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    async def test_legacy_behavior_preserved(self, executor_core):
        """Test that legacy behavior is preserved after migration."""
        # This test ensures that the new implementation produces the same results
        # as the legacy implementation for the same inputs

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify legacy behavior is preserved
            assert result.success is True
            assert result.output == "test_output"
            assert result.name == "test_conditional"
            assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_error_handling_regression(self, executor_core):
        """Test that error handling behavior is preserved."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
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
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify error handling is preserved
            assert result.success is False
            assert "Error executing conditional logic or branch" in result.feedback

    async def test_metrics_accumulation_regression(self, executor_core):
        """Test that metrics accumulation behavior is preserved."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent",
                success=True,
                output="test_output",
                latency_s=1.0,
                cost_usd=0.01,
                token_counts=100,
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify metrics accumulation is preserved
            assert result.latency_s == 1.0
            assert result.cost_usd == 0.01
            assert result.token_counts == 100

    async def test_context_handling_regression(self, executor_core):
        """Test that context handling behavior is preserved."""
        from flujo.domain.models import PipelineContext, ImportArtifacts

        context = PipelineContext(initial_prompt="test", import_artifacts=ImportArtifacts())
        context_setter_called = False

        def mock_context_setter(result, ctx):
            nonlocal context_setter_called
            context_setter_called = True

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=context,
                resources=None,
                limits=None,
                context_setter=mock_context_setter,
            )

        # Verify context handling is preserved
        assert context_setter_called is True

    async def test_branch_output_mapper_regression(self, executor_core):
        """Test that branch output mapper behavior is preserved."""

        def output_mapper(output, branch_key, context):
            return {"mapped": output, "key": branch_key}

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
            branch_output_mapper=output_mapper,
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify output mapper behavior is preserved
            assert result.success is True
            assert result.output["mapped"] == "test_output"
            assert result.output["key"] == "branch_a"

    async def test_branch_input_mapper_regression(self, executor_core):
        """Test that branch input mapper behavior is preserved."""

        def input_mapper(data, context):
            return {"processed": data}

        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
            branch_input_mapper=input_mapper,
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify input mapper was called with correct parameters
            mock_execute.assert_called_once()
            # The step executor should have been called with the mapped input
            # We can't directly verify the mapped input, but we can verify the call was made

    async def test_default_branch_regression(self, executor_core):
        """Test that default branch behavior is preserved."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "nonexistent_branch",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
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
                        agent=Mock(return_value="default_output"),
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
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify default branch behavior is preserved
            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "nonexistent_branch"

    async def test_multiple_steps_regression(self, executor_core):
        """Test that multiple steps execution behavior is preserved."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="step1",
                            agent=Mock(return_value="step1_output"),
                            config=StepConfig(max_retries=1),
                        ),
                        Step(
                            name="step2",
                            agent=Mock(return_value="step2_output"),
                            config=StepConfig(max_retries=1),
                        ),
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = [
                StepResult(name="step1", success=True, output="step1_output"),
                StepResult(name="step2", success=True, output="step2_output"),
            ]

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify multiple steps execution behavior is preserved
            assert result.success is True
            assert result.output == "step2_output"
            assert mock_execute.call_count == 2

    async def test_attempts_field_regression(self, executor_core):
        """Test that attempts field is properly set."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify attempts field is set correctly
            assert result.attempts == 1

    async def test_telemetry_regression(self, executor_core, isolated_telemetry):
        """Test that telemetry logging behavior is preserved."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            await executor_core._handle_conditional_step(
                conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify telemetry logging is preserved
            assert any(
                "Condition evaluated to branch key 'branch_a'" in msg
                for msg in isolated_telemetry.infos
            )

    async def test_null_parameters_regression(self, executor_core):
        """Test that null parameters are handled correctly."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={
                "branch_a": Pipeline(
                    name="branch_a",
                    steps=[
                        Step(
                            name="test_agent",
                            agent=Mock(return_value="test_output"),
                            config=StepConfig(max_retries=1),
                        )
                    ],
                )
            },
        )

        with patch.object(executor_core, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = StepResult(
                name="test_agent", success=True, output="test_output"
            )

            result = await executor_core._handle_conditional_step(
                conditional_step,
                data=None,
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify null parameters are handled correctly
            assert result.success is True
            assert result.metadata_["executed_branch_key"] == "branch_a"

    async def test_empty_branches_regression(self, executor_core):
        """Test that empty branches are handled correctly."""
        conditional_step = ConditionalStep(
            name="test_conditional",
            condition_callable=lambda data, context: "branch_a",
            branches={"branch_a": Pipeline(name="branch_a", steps=[])},
        )

        result = await executor_core._handle_conditional_step(
            conditional_step,
            data="test_data",
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
        )

        # Verify empty branches are handled correctly
        assert result.success is True
        # When there are no steps, the output should be the input data
        assert result.output == "test_data"
        assert result.metadata_["executed_branch_key"] == "branch_a"
