"""Unit tests for the new execution management components."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from flujo.application.core import (
    ExecutionManager,
    StateManager,
    StepCoordinator,
    TypeValidator,
)
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    PipelineResult,
    StepResult,
    UsageLimits,
    PipelineContext,
    Failure,
    Success,
)

from flujo.exceptions import PipelineAbortSignal, TypeMismatchError


class TestStateManager:
    """Test the StateManager component."""

    @pytest.fixture
    def state_manager(self):
        return StateManager()

    @pytest.fixture
    def mock_state_backend(self):
        backend = Mock()
        backend.load_state = AsyncMock()
        backend.save_state = AsyncMock()
        return backend

    @pytest.mark.asyncio
    async def test_load_workflow_state_no_backend(self, state_manager):
        """Test loading state when no backend is configured."""
        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
            step_history,
        ) = await state_manager.load_workflow_state("test-id")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None
        assert step_history == []

    @pytest.mark.asyncio
    async def test_load_workflow_state_no_run_id(self, mock_state_backend):
        """Test loading state when no run_id is provided."""
        state_manager = StateManager(mock_state_backend)
        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
            step_history,
        ) = await state_manager.load_workflow_state("")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None
        assert step_history == []

    @pytest.mark.asyncio
    async def test_load_workflow_state_not_found(self, mock_state_backend):
        """Test loading state when state doesn't exist."""
        mock_state_backend.load_state.return_value = None
        state_manager = StateManager(mock_state_backend)

        (
            context,
            output,
            idx,
            created,
            pipeline_name,
            pipeline_version,
            step_history,
        ) = await state_manager.load_workflow_state("test-id")
        assert context is None
        assert output is None
        assert idx == 0
        assert created is None
        assert pipeline_name is None
        assert pipeline_version is None
        assert step_history == []

    @pytest.mark.asyncio
    async def test_persist_workflow_state_no_backend(self, state_manager):
        """Test persisting state when no backend is configured."""
        await state_manager.persist_workflow_state(
            run_id="test-id",
            context=PipelineContext(initial_prompt="test"),
            current_step_index=1,
            last_step_output="output",
            status="running",
        )
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_persist_workflow_state_no_run_id(self, mock_state_backend):
        """Test persisting state when no run_id is provided."""
        state_manager = StateManager(mock_state_backend)
        await state_manager.persist_workflow_state(
            run_id=None,
            context=PipelineContext(initial_prompt="test"),
            current_step_index=1,
            last_step_output="output",
            status="running",
        )
        mock_state_backend.save_state.assert_not_called()

    def test_get_run_id_from_context(self, state_manager):
        """Test extracting run_id from context."""
        context = PipelineContext(initial_prompt="test")
        context.run_id = "test-run-id"

        run_id = state_manager.get_run_id_from_context(context)
        assert run_id == "test-run-id"

    def test_get_run_id_from_context_none(self, state_manager):
        """Test extracting run_id when context is None."""
        run_id = state_manager.get_run_id_from_context(None)
        assert run_id is None


class TestQuotaMode:
    """Quota-mode replacements for legacy UsageGovernor tests."""

    def test_root_quota_constructed_from_usage_limits(self):
        """ExecutionManager constructs a root quota when usage_limits are provided."""
        limits = UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=1000)
        # Minimal pipeline (no steps needed for this assertion)
        pipeline = Pipeline.model_validate({"steps": []})
        manager = ExecutionManager(pipeline, usage_limits=limits)
        assert getattr(manager, "root_quota", None) is not None

    def test_execution_manager_accepts_quota_without_usage_governor(self):
        """Ensure no legacy UsageGovernor is required; quota object is the control surface."""
        limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=100)
        pipeline = Pipeline.model_validate({"steps": []})
        manager = ExecutionManager(pipeline, usage_limits=limits)
        # Legacy attribute should not exist; root_quota should be present
        assert not hasattr(manager, "usage_governor")
        assert getattr(manager, "root_quota", None) is not None


class TestTypeValidator:
    """Test the TypeValidator component."""

    @pytest.fixture
    def type_validator(self):
        return TypeValidator()

    @pytest.fixture
    def mock_step(self):
        step = Mock(spec=Step)
        step.name = "test_step"
        step.__step_input_type__ = str
        step.__step_output_type__ = str
        return step

    def test_validate_step_output_no_next_step(self, type_validator, mock_step):
        """Test type validation when there's no next step."""
        # Should not raise any exceptions
        type_validator.validate_step_output(mock_step, "output", None)

    def test_validate_step_output_compatible_types(self, type_validator, mock_step):
        """Test type validation with compatible types."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = str

        # Should not raise any exceptions
        type_validator.validate_step_output(mock_step, "string output", next_step)

    def test_validate_step_output_incompatible_types(self, type_validator, mock_step):
        """Test type validation with incompatible types."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = int

        with pytest.raises(TypeMismatchError, match="Type mismatch"):
            type_validator.validate_step_output(mock_step, "string output", next_step)

    def test_validate_step_output_none_value(self, type_validator, mock_step):
        """Test type validation with None values."""
        next_step = Mock(spec=Step)
        next_step.name = "next_step"
        next_step.__step_input_type__ = str

        # Should raise TypeMismatchError when None is passed to str type
        with pytest.raises(TypeMismatchError, match="Type mismatch"):
            type_validator.validate_step_output(mock_step, None, next_step)

    def test_get_step_input_type(self, type_validator, mock_step):
        """Test getting step input type."""
        mock_step.__step_input_type__ = str
        input_type = type_validator.get_step_input_type(mock_step)
        assert input_type is str

    def test_get_step_output_type(self, type_validator, mock_step):
        """Test getting step output type."""
        mock_step.__step_output_type__ = int
        output_type = type_validator.get_step_output_type(mock_step)
        assert output_type is int


class TestStepCoordinator:
    """Test the StepCoordinator component."""

    @pytest.fixture
    def step_coordinator(self):
        return StepCoordinator()

    @pytest.fixture
    def mock_step(self):
        step = Mock(spec=Step)
        step.name = "test_step"
        step.__step_input_type__ = str
        step.__step_output_type__ = str
        return step

    @pytest.mark.asyncio
    async def test_execute_step_success(self, step_coordinator, mock_step):
        """Test successful step execution."""
        step_result = StepResult(name="test_step", output="success", success=True)

        results = []

        class _Backend:
            agent_registry = {}

            async def execute_step(self, _request):
                return Success(step_result=step_result)

        async for item in step_coordinator.execute_step(
            mock_step, "input", None, backend=_Backend()
        ):
            results.append(item)

        assert len(results) == 1
        assert isinstance(results[0], Success)
        assert results[0].step_result == step_result

    @pytest.mark.asyncio
    async def test_execute_step_failure(self, step_coordinator, mock_step):
        """Test failed step execution."""
        step_result = StepResult(name="test_step", output=None, success=False, feedback="error")

        results = []

        class _Backend:
            agent_registry = {}

            async def execute_step(self, _request):
                return Failure(error=Exception("error"), feedback="error", step_result=step_result)

        async for item in step_coordinator.execute_step(
            mock_step, "input", None, backend=_Backend()
        ):
            results.append(item)

        assert len(results) == 1
        assert isinstance(results[0], Failure)
        assert results[0].step_result == step_result

    def test_update_pipeline_result(self, step_coordinator):
        """Test updating pipeline result with step result."""
        result = PipelineResult()
        step_result = StepResult(name="test", output="test", success=True, cost_usd=1.0)

        step_coordinator.update_pipeline_result(result, step_result)

        assert len(result.step_history) == 1
        assert result.step_history[0] == step_result
        assert result.total_cost_usd == 1.0


class TestExecutionManager:
    """Test the ExecutionManager component."""

    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Mock(spec=Pipeline)
        step1 = Mock(spec=Step)
        step1.name = "step1"
        step1.__step_input_type__ = str
        step1.__step_output_type__ = str
        step2 = Mock(spec=Step)
        step2.name = "step2"
        step2.__step_input_type__ = str
        step2.__step_output_type__ = str
        pipeline.steps = [step1, step2]
        return pipeline

    @pytest.fixture
    def execution_manager(self, mock_pipeline):
        return ExecutionManager(mock_pipeline)

    @pytest.mark.asyncio
    async def test_execute_steps_basic(self, mock_pipeline):
        """Test basic step execution."""
        step1_result = StepResult(name="step1", output="output1", success=True)
        step2_result = StepResult(name="step2", output="output2", success=True)

        class _Backend:
            agent_registry = {}

            async def execute_step(self, request):
                if request.step.name == "step1":
                    return Success(step_result=step1_result)
                if request.step.name == "step2":
                    return Success(step_result=step2_result)
                raise AssertionError(f"Unexpected step {request.step.name}")

        execution_manager = ExecutionManager(mock_pipeline, backend=_Backend())
        result = PipelineResult()
        results = []
        async for item in execution_manager.execute_steps(
            start_idx=0,
            data="input",
            context=None,
            result=result,
        ):
            results.append(item)

        assert len(results) == 0  # No streaming output
        assert len(result.step_history) == 2  # Both steps in the pipeline were executed

    @pytest.mark.asyncio
    async def test_pipeline_invariants_fail_after_context_update(self):
        """Invariant violations should fail the step after context updates."""
        step = Step(name="step1", agent=lambda _x: {"mode": "unsafe"}, updates_context=True)
        pipeline = Pipeline(
            steps=[step],
            static_invariants=["context.mode == 'safe'"],
        )

        class _Backend:
            agent_registry = {}

            async def execute_step(self, _request):
                return Success(
                    step_result=StepResult(name="step1", output={"mode": "unsafe"}, success=True)
                )

        execution_manager = ExecutionManager(pipeline, backend=_Backend())
        result = PipelineResult()
        ctx = PipelineContext(initial_prompt="test")
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="input",
            context=ctx,
            result=result,
        ):
            pass

        assert len(result.step_history) == 1
        step_result = result.step_history[0]
        assert step_result.success is False
        assert step_result.feedback is not None
        assert "Invariant violated" in step_result.feedback
        assert step_result.metadata_ is not None
        assert step_result.metadata_["invariant_violations"][0]["rule"] == "context.mode == 'safe'"

    @pytest.mark.asyncio
    async def test_abort_from_on_step_failure_closes_step_iterator(self) -> None:
        """Abort paths must close the in-flight step iterator to avoid asyncgen leaks."""

        closed = False
        gate = asyncio.Event()

        class AbortCoordinator:
            resources = None

            async def execute_step(self, *_a: object, **_k: object):
                nonlocal closed
                try:
                    yield Failure(
                        error=Exception("fail"),
                        feedback="fail",
                        step_result=StepResult(name="s1", success=False, feedback="fail"),
                    )
                    await gate.wait()
                finally:
                    closed = True

            async def _dispatch_hook(self, *_a: object, **_k: object) -> None:
                raise PipelineAbortSignal("Aborted from hook")

            def update_pipeline_result(
                self, result: PipelineResult, step_result: StepResult
            ) -> None:
                result.step_history.append(step_result)

        step = Mock(spec=Step)
        step.name = "s1"
        step.__step_input_type__ = str
        step.__step_output_type__ = str
        pipeline = Mock(spec=Pipeline)
        pipeline.steps = [step]

        manager = ExecutionManager(pipeline, step_coordinator=AbortCoordinator())
        result = PipelineResult()
        outputs: list[object] = []
        async for item in manager.execute_steps(
            start_idx=0,
            data="x",
            context=None,
            result=result,
            stream_last=False,
        ):
            outputs.append(item)

        assert outputs, "Expected a PipelineResult yield on abort"
        assert closed is True

    def test_set_final_context(self, execution_manager):
        """Test setting final context."""
        result = PipelineResult()
        context = PipelineContext(initial_prompt="test")

        execution_manager.set_final_context(result, context)

        assert result.final_pipeline_context == context

    @pytest.mark.asyncio
    async def test_persist_final_state(self, execution_manager):
        """Test persisting final state."""
        result = PipelineResult()
        result.step_history = [StepResult(name="step1", output="final", success=True)]
        context = PipelineContext(initial_prompt="test")

        # Mock the state manager
        execution_manager.state_manager = Mock()
        execution_manager.state_manager.persist_workflow_state = AsyncMock()
        execution_manager.state_manager.record_run_end = AsyncMock()
        execution_manager.state_manager.get_run_id_from_context.return_value = "test-run"

        await execution_manager.persist_final_state(
            run_id="test-run",
            context=context,
            result=result,
            start_idx=0,
            state_created_at=datetime.now(),
            final_status="completed",
        )

        execution_manager.state_manager.persist_workflow_state.assert_called_once()
        execution_manager.state_manager.record_run_end.assert_called_once()
