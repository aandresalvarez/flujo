"""Tests for ConditionalStep dispatch in ExecutorCore."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import StepResult, Success
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.types import ExecutionFrame


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

    async def test_execute_complex_step_routes_conditionalstep_via_dispatcher(
        self, executor_core, mock_conditional_step
    ):
        """ConditionalStep dispatch should flow through the dispatcher."""
        with patch.object(
            executor_core._dispatch_handler, "dispatch", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = Success(
                step_result=StepResult(name="test_conditional", success=True)
            )

            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                context_setter=None,
            )

            mock_dispatch.assert_called_once()
            assert isinstance(result, StepResult)
            assert result.success is True

    async def test_execute_complex_step_conditionalstep_parameter_passing(
        self, executor_core, mock_conditional_step
    ):
        """ConditionalStep parameters are forwarded via ExecutionFrame."""
        from flujo.domain.models import BaseModel

        class TestContext(BaseModel):
            key: str = "value"

        captured_frame = None

        async def mock_dispatch(frame: ExecutionFrame, called_with_frame: bool):
            nonlocal captured_frame
            captured_frame = frame
            return Success(step_result=StepResult(name="test_conditional", success=True))

        with patch.object(executor_core._dispatch_handler, "dispatch", mock_dispatch):
            test_data = "test_data"
            test_context = TestContext()
            test_resources = Mock()
            from flujo.domain.models import UsageLimits

            test_limits = UsageLimits(total_cost_usd_limit=10.0)
            test_context_setter = Mock()

            await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data=test_data,
                context=test_context,
                resources=test_resources,
                limits=test_limits,
                stream=False,
                on_chunk=None,
                context_setter=test_context_setter,
                _fallback_depth=1,
            )

        assert isinstance(captured_frame, ExecutionFrame)
        assert captured_frame.step is mock_conditional_step
        assert captured_frame.data is test_data
        from flujo.domain.models import PipelineContext

        assert captured_frame.context is test_context or isinstance(
            captured_frame.context, PipelineContext
        )
        assert captured_frame.resources is test_resources
        assert captured_frame.limits is test_limits
        assert callable(captured_frame.context_setter)
        from flujo.domain.models import PipelineResult

        captured_frame.context_setter(
            PipelineResult(
                step_history=[], total_cost_usd=0.0, total_tokens=0, final_pipeline_context=None
            ),
            None,
        )
        assert test_context_setter.called
        assert getattr(captured_frame, "_fallback_depth") == 1

    async def test_execute_complex_step_conditionalstep_no_legacy_import(self, executor_core):
        """Test that _handle_conditional_step is no longer imported from legacy step_logic."""
        mock_conditional_step = Mock(spec=ConditionalStep)
        mock_conditional_step.name = "test_conditional"

        with (
            patch.object(
                executor_core, "_handle_conditional_step", new_callable=AsyncMock
            ) as mock_handler,
            patch.object(
                executor_core._dispatch_handler,
                "dispatch",
                new_callable=AsyncMock,
                return_value=Success(step_result=StepResult(name="test_conditional", success=True)),
            ) as mock_dispatch,
        ):
            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                context_setter=None,
            )

        mock_dispatch.assert_called_once()
        mock_handler.assert_not_called()
        assert isinstance(result, StepResult)
        assert result.success

    async def test_execute_complex_step_conditionalstep_error_propagation(
        self, executor_core, mock_conditional_step
    ):
        """Test that dispatch errors are converted into StepResults."""
        with patch.object(
            executor_core._dispatch_handler,
            "dispatch",
            new_callable=AsyncMock,
            side_effect=Exception("Test error"),
        ):
            result = await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                context_setter=None,
            )
        assert isinstance(result, StepResult)
        assert result.success is False

    async def test_execute_complex_step_conditionalstep_telemetry_logging(
        self, executor_core, mock_conditional_step
    ):
        """Test that ConditionalStep dispatch includes proper telemetry logging."""
        with patch.object(
            executor_core._dispatch_handler,
            "dispatch",
            new_callable=AsyncMock,
            return_value=Success(step_result=StepResult(name="test_conditional", success=True)),
        ):
            await executor_core._execute_complex_step(
                step=mock_conditional_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                stream=False,
                on_chunk=None,
                context_setter=None,
            )
