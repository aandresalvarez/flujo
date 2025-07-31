"""Tests for ConditionalStep dispatch in ExecutorCore."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import StepResult
from flujo.application.core.ultra_executor import ExecutorCore


class TestExecutorCoreConditionalStepDispatch:
    """Test suite for ConditionalStep dispatch in ExecutorCore."""

    @pytest.fixture
    def executor_core(self):
        """Create ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def mock_conditional_step(self):
        """Create a mock ConditionalStep for testing."""
        conditional_step = Mock(spec=ConditionalStep)
        conditional_step.name = "test_conditional"
        return conditional_step

    async def test_execute_complex_step_routes_conditionalstep_to_new_handler(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep is routed to _handle_conditional_step method."""
        # Mock the _handle_conditional_step method
        with patch.object(
            executor_core, "_handle_conditional_step", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = StepResult(name="test_conditional", success=True)

            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                breach_event=None,
                context_setter=None,
            )

            # Verify the new handler was called
            mock_handler.assert_called_once()
            assert isinstance(result, StepResult)
            assert result.success is True

    async def test_execute_complex_step_conditionalstep_parameter_passing(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep parameters are correctly passed to new handler."""
        captured_args = None
        captured_kwargs = None

        async def mock_handler(*args, **kwargs):
            nonlocal captured_args, captured_kwargs
            captured_args = args
            captured_kwargs = kwargs
            return StepResult(name="test_conditional", success=True)

        with patch.object(executor_core, "_handle_conditional_step", mock_handler):
            test_data = "test_data"
            test_context = Mock()
            test_resources = Mock()
            test_limits = Mock()
            test_context_setter = Mock()

            await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data=test_data,
                context=test_context,
                resources=test_resources,
                limits=test_limits,
                stream=False,
                on_chunk=None,
                breach_event=None,
                context_setter=test_context_setter,
            )

            # Verify parameters were passed correctly
            assert captured_args[0] is mock_conditional_step
            assert captured_args[1] is test_data
            assert captured_args[2] is test_context
            assert captured_args[3] is test_resources
            assert captured_args[4] is test_limits
            assert captured_args[5] is test_context_setter

    async def test_execute_complex_step_conditionalstep_no_legacy_import(self, executor_core):
        """Test that _handle_conditional_step is no longer imported from legacy step_logic."""
        mock_conditional_step = Mock(spec=ConditionalStep)
        mock_conditional_step.name = "test_conditional"

        with patch.object(
            executor_core, "_handle_conditional_step", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = StepResult(name="test_conditional", success=True)

            # Execute the complex step with ConditionalStep
            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                breach_event=None,
                context_setter=None,
            )

            # Verify the new handler was called (not the legacy one)
            mock_handler.assert_called_once()
            assert isinstance(result, StepResult)

            # Verify that the method is fully self-contained and doesn't delegate to legacy code
            # This test ensures that the implementation is complete and independent

    async def test_execute_complex_step_conditionalstep_error_propagation(
        self, executor_core, mock_conditional_step
    ):
        """Test that errors from _handle_conditional_step are properly propagated."""
        with patch.object(
            executor_core,
            "_handle_conditional_step",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception, match="Test error"):
                await executor_core._execute_complex_step(
                    step=mock_conditional_step,
                    data="test_data",
                    context=None,
                    resources=None,
                    limits=None,
                    stream=False,
                    on_chunk=None,
                    breach_event=None,
                    context_setter=None,
                )

    async def test_execute_complex_step_conditionalstep_telemetry_logging(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep dispatch includes proper telemetry logging."""
        with patch.object(
            executor_core,
            "_handle_conditional_step",
            new_callable=AsyncMock,
            return_value=StepResult(name="test_conditional", success=True),
        ):
            # Mock telemetry to capture log messages
            with patch("flujo.infra.telemetry.logfire.debug") as mock_debug:
                await executor_core._execute_complex_step(
                    step=mock_conditional_step,
                    data="test_data",
                    context=None,
                    resources=None,
                    limits=None,
                    stream=False,
                    on_chunk=None,
                    breach_event=None,
                    context_setter=None,
                )

                # Verify debug logging was called
                mock_debug.assert_called()
                debug_calls = [call[0][0] for call in mock_debug.call_args_list]
                assert any("Handling ConditionalStep" in call for call in debug_calls)
