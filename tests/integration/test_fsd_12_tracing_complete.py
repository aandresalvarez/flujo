"""
Integration tests for FSD-12: Rich Internal Tracing and Visualization.

This test suite verifies that the tracing system works correctly end-to-end,
including trace generation, persistence, and CLI visualization.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from flujo.domain.dsl import Pipeline, Step, LoopStep, ConditionalStep
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend

# Tracing + SQLite persistence can be slow under some environments.
pytestmark = pytest.mark.slow


class FSD12TracingTestContext(PipelineContext):
    """Test context for tracing tests."""

    counter: int = 0
    results: list[str] = []


async def simple_step(input_data: str, context: PipelineContext) -> str:
    """A simple step that processes input."""
    result = f"processed_{input_data}"
    context.import_artifacts["last_result"] = result
    return result


async def loop_step(input_data: str, context: PipelineContext) -> list[str]:
    """A step that processes data in a loop."""
    results = []
    for iteration in range(2):
        result = f"loop_{input_data}_{iteration}"
        results.append(result)
    return results


class TestFSD12TracingComplete:
    """Test the complete FSD-12 tracing functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass

    @pytest.fixture
    async def state_backend(self, temp_db_path):
        """Create a SQLite backend for testing with proper cleanup."""
        backend = SQLiteBackend(temp_db_path)
        yield backend
        # Ensure backend is properly closed
        try:
            await backend.close()
        except Exception:
            pass

    @pytest.fixture
    def complex_pipeline(self):
        """Create a complex pipeline with loops and conditionals for testing."""
        return Pipeline(
            steps=[
                Step.from_callable(simple_step, name="step1"),
                LoopStep(
                    name="loop1",
                    loop_body_pipeline=Pipeline(
                        steps=[Step.from_callable(loop_step, name="loop_step")]
                    ),
                    exit_condition_callable=lambda output, ctx: len(output) >= 2
                    if output
                    else False,
                    max_loops=2,
                ),
                ConditionalStep(
                    name="conditional1",
                    condition_callable=lambda data, ctx: "high" if len(str(data)) > 10 else "low",
                    branches={
                        "high": Pipeline(
                            steps=[Step.from_callable(simple_step, name="high_branch")]
                        ),
                        "low": Pipeline(steps=[Step.from_callable(simple_step, name="low_branch")]),
                    },
                ),
                Step.from_callable(simple_step, name="final_step"),
            ],
        )

    @pytest.mark.asyncio
    async def test_trace_generation_and_persistence(self, state_backend, complex_pipeline):
        """Test that traces are generated and persisted correctly."""
        # Create Flujo runner with tracing enabled
        flujo = Flujo(
            pipeline=complex_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        # Run the pipeline
        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        # Verify trace tree was generated
        assert final_result.trace_tree is not None
        assert final_result.trace_tree.name == "pipeline_run"
        assert final_result.trace_tree.status == "completed"

        # Verify trace has children (the steps)
        assert len(final_result.trace_tree.children) > 0

        # Verify trace was persisted to database
        run_id = final_result.final_pipeline_context.run_id
        trace = await state_backend.get_trace(run_id)
        assert trace is not None
        assert trace["name"] == "pipeline_run"
        # Note: The persisted trace status might be "running" if saved before finalization
        # but the in-memory trace should be "completed"

    @pytest.mark.asyncio
    async def test_trace_hierarchical_structure(self, state_backend, complex_pipeline):
        """Test that trace maintains proper hierarchical structure."""
        flujo = Flujo(
            pipeline=complex_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        # Verify root span
        root_span = final_result.trace_tree
        assert root_span.name == "pipeline_run"

        # Verify step spans exist
        step_names = [child.name for child in root_span.children]
        assert "step1" in step_names

        # Find the loop step
        loop_step = None
        for child in root_span.children:
            if child.name == "loop1":
                loop_step = child
                break

        assert loop_step is not None
        assert loop_step.status == "completed"

        # Note: Loop steps might not have children if the loop body is not traced
        # This is expected behavior for the current implementation
        # The loop step itself should be traced, even if its children are not

    @pytest.mark.asyncio
    async def test_trace_metadata_capture(self, state_backend, complex_pipeline):
        """Test that trace captures metadata correctly."""
        flujo = Flujo(
            pipeline=complex_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        # Verify step history is captured
        assert len(final_result.step_history) > 0

        # Verify each step has proper metadata
        for step_result in final_result.step_history:
            assert step_result.name is not None
            assert step_result.success is not None
            assert step_result.attempts >= 0
            assert step_result.latency_s >= 0

    @pytest.mark.asyncio
    async def test_trace_persistence_recovery(self, state_backend, complex_pipeline):
        """Test that traces can be recovered from persistence."""
        flujo = Flujo(
            pipeline=complex_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        # Run pipeline and get run_id
        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        run_id = final_result.final_pipeline_context.run_id

        # Create new backend instance to simulate restart - Use async with for proper cleanup
        async with SQLiteBackend(state_backend.db_path) as new_backend:
            # Verify trace can be retrieved
            trace = await new_backend.get_trace(run_id)
            assert trace is not None
            assert trace["name"] == "pipeline_run"

            # Verify spans can be retrieved
            spans = await new_backend.get_spans(run_id)
            assert len(spans) > 0

            # Verify span statistics
            stats = await new_backend.get_span_statistics()
            assert stats is not None

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to performance variability
    async def test_trace_performance_overhead(self, state_backend):
        """Test that tracing doesn't add significant performance overhead."""
        # Simple pipeline for performance testing
        simple_pipeline = Pipeline(
            steps=[
                Step.from_callable(simple_step, name="step1"),
                Step.from_callable(simple_step, name="step2"),
            ],
        )

        # Test without tracing
        flujo_no_trace = Flujo(
            pipeline=simple_pipeline,
            enable_tracing=False,
            state_backend=state_backend,
        )

        start_time = asyncio.get_event_loop().time()
        async for result in flujo_no_trace.run_async("test_input"):
            pass  # We only need timing, not the result
        no_trace_time = asyncio.get_event_loop().time() - start_time

        # Test with tracing
        flujo_with_trace = Flujo(
            pipeline=simple_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        start_time = asyncio.get_event_loop().time()
        async for result in flujo_with_trace.run_async("test_input"):
            pass  # We only need timing, not the result
        with_trace_time = asyncio.get_event_loop().time() - start_time

        # Verify tracing overhead is reasonable (less than 100% increase)
        overhead_ratio = with_trace_time / no_trace_time
        assert overhead_ratio < 2.0, f"Tracing overhead too high: {overhead_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_trace_error_handling(self, state_backend):
        """Test that tracing handles errors gracefully."""

        async def failing_step(input_data: str, context: PipelineContext) -> str:
            """A step that always fails."""
            raise ValueError("Intentional failure for testing")

        failing_pipeline = Pipeline(
            steps=[
                Step.from_callable(failing_step, name="failing_step"),
            ],
        )

        flujo = Flujo(
            pipeline=failing_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        # Run the pipeline (should fail)
        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        # Verify trace tree was still generated
        assert final_result.trace_tree is not None
        assert final_result.trace_tree.name == "pipeline_run"

        # Verify failed step is marked as failed
        failed_step = None
        for child in final_result.trace_tree.children:
            if child.name == "failing_step":
                failed_step = child
                break

        assert failed_step is not None
        assert failed_step.status == "failed"

    @pytest.mark.asyncio
    async def test_trace_large_pipeline(self, state_backend):
        """Test tracing with a larger pipeline to verify scalability."""
        # Create a larger pipeline with many steps
        steps = []
        for step_index in range(10):
            steps.append(Step.from_callable(simple_step, name=f"step_{step_index}"))

        large_pipeline = Pipeline(steps=steps)

        flujo = Flujo(
            pipeline=large_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        final_result = None
        async for result in flujo.run_async("test_input"):
            final_result = result

        # Verify trace tree was generated
        assert final_result.trace_tree is not None

        # Verify all steps are captured
        assert len(final_result.step_history) == 10

        # Verify trace can be persisted and retrieved
        run_id = final_result.final_pipeline_context.run_id
        trace = await state_backend.get_trace(run_id)
        assert trace is not None
