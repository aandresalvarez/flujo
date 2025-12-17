"""Tests for persistence edge cases and performance scenarios to prevent future issues."""

import pytest
import time
import asyncio
from typing import Any, List
import uuid
from datetime import datetime, timezone

from flujo.domain.models import PipelineContext, PipelineResult
from flujo.application.core.execution_manager import ExecutionManager
from flujo.application.core.state_manager import StateManager
from flujo.domain import Step
from flujo.testing.utils import StubAgent

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow


class TestPersistenceOptimizationEdgeCases:
    """Test edge cases related to persistence optimizations."""

    @pytest.mark.asyncio
    async def test_persistence_frequency_optimization(self, sqlite_backend_factory):
        """Test that persistence frequency optimization works correctly."""
        backend = sqlite_backend_factory("test.db")
        state_manager = StateManager(state_backend=backend)

        # Create a multi-step pipeline
        steps = [
            Step.model_validate({"name": "step1", "agent": StubAgent(["step1_output"])}),
            Step.model_validate({"name": "step2", "agent": StubAgent(["step2_output"])}),
            Step.model_validate({"name": "step3", "agent": StubAgent(["step3_output"])}),
        ]

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=steps)

        execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_run", str(uuid.uuid4()), "test_pipeline", "1.0", created_at=now, updated_at=now
        )

        # Run the pipeline
        result = PipelineResult()
        context = PipelineContext(initial_prompt="test")
        run_id = "test_run"

        # Execute steps
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify all steps were recorded
        steps = await backend.list_run_steps(run_id)
        assert len(steps) == 3
        assert steps[0]["step_name"] == "step1"
        assert steps[1]["step_name"] == "step2"
        assert steps[2]["step_name"] == "step3"

        # Verify workflow state was only persisted on final step
        # (We can't directly check this, but we can verify the final state is correct)
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        assert workflow_state["current_step_index"] == 3

    @pytest.mark.asyncio
    async def test_persistence_on_step_failure(self, sqlite_backend_factory):
        """Test that persistence works correctly when steps fail."""
        backend = sqlite_backend_factory("test.db")
        state_manager = StateManager(state_backend=backend)

        # Create a failing agent
        class FailingAgent:
            async def run(self, *args, **kwargs) -> Any:
                raise RuntimeError("Simulated failure")

            async def run_async(self, *args, **kwargs) -> Any:
                return await self.run(*args, **kwargs)

        failing_agent = FailingAgent()
        step = Step.model_validate({"name": "failing_step", "agent": failing_agent})

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=[step])

        execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_failure_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run the step
        result = PipelineResult()
        context = PipelineContext(initial_prompt="test")
        run_id = "test_failure_run"

        # Execute steps
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify failed step was recorded
        steps = await backend.list_run_steps(run_id)
        assert len(steps) == 1
        assert steps[0]["step_name"] == "failing_step"
        assert steps[0]["status"] == "failed"

        # Verify workflow state was persisted on failure
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        assert workflow_state["current_step_index"] == 1

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to performance measurement
    async def test_large_context_serialization_performance(self, sqlite_backend_factory):
        """Test that large context serialization doesn't cause performance issues."""
        backend = sqlite_backend_factory("large_context.db")
        state_manager = StateManager(state_backend=backend)

        # Create context with large data
        class LargeContext(PipelineContext):
            large_data: str = "x" * 50000  # 50KB of data
            nested_data: Any = {"deep": {"nested": {"data": "x" * 10000}}}

        agent = StubAgent(["output"])
        step = Step.model_validate({"name": "large_context_step", "agent": agent})

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=[step])

        execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_large_context_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run with large context
        result = PipelineResult()
        context = LargeContext(initial_prompt="test", large_data="y" * 50000)
        run_id = "test_large_context_run"

        # Measure performance
        start_time = time.perf_counter()
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass
        execution_time = time.perf_counter() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 5.0, f"Large context execution took {execution_time:.3f}s"

        # Verify data was persisted correctly
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        assert "large_data" in workflow_state["pipeline_context"]

    @pytest.mark.asyncio
    async def test_serialization_error_handling(self, sqlite_backend_factory):
        """Test that serialization errors are handled gracefully."""
        backend = sqlite_backend_factory("serialization_error.db")
        state_manager = StateManager(state_backend=backend)

        # Create context that might cause serialization issues
        class ProblematicContext(PipelineContext):
            def model_dump(self, **kwargs):
                raise RuntimeError("Simulated serialization error")

        agent = StubAgent(["output"])
        step = Step.model_validate({"name": "problematic_step", "agent": agent})

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=[step])

        execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_serialization_error_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run with problematic context
        result = PipelineResult()
        context = ProblematicContext(initial_prompt="test")
        run_id = "test_serialization_error_run"

        # Should not raise an exception
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify that the error was handled gracefully
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        # Should have fallback error message in context
        assert "error" in workflow_state["pipeline_context"]

    @pytest.mark.asyncio
    async def test_concurrent_persistence_operations(self, sqlite_backend_factory):
        """Test that concurrent persistence operations don't cause issues."""
        backend = sqlite_backend_factory("concurrent.db")
        state_manager = StateManager(state_backend=backend)

        # Create multiple agents, one per step per pipeline
        def make_steps():
            return [
                Step.model_validate({"name": f"step_{i}", "agent": StubAgent([f"output_{i}"])})
                for i in range(3)
            ]

        from flujo.domain.dsl.pipeline import Pipeline

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "concurrent_run_0",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run multiple pipelines concurrently
        async def run_pipeline(run_id: str):
            result = PipelineResult()
            context = PipelineContext(initial_prompt=f"test_{run_id}")
            # Each pipeline gets its own steps/agents
            pipeline = Pipeline(steps=make_steps())
            execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

            async for _ in execution_manager.execute_steps(
                start_idx=0,
                data="test_input",
                context=context,
                result=result,
                run_id=run_id,
            ):
                pass

        # Run multiple pipelines concurrently
        tasks = [run_pipeline(f"concurrent_run_{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all runs were persisted correctly
        for i in range(5):
            workflow_state = await backend.load_state(f"concurrent_run_{i}")
            assert workflow_state is not None
            assert workflow_state["run_id"] == f"concurrent_run_{i}"

    @pytest.mark.asyncio
    async def test_persistence_with_none_context(self, sqlite_backend_factory):
        """Test that persistence works correctly with None context."""
        backend = sqlite_backend_factory("none_context.db")
        state_manager = StateManager(state_backend=backend)

        agent = StubAgent(["output"])
        step = Step.model_validate({"name": "none_context_step", "agent": agent})

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=[step])

        execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_none_context_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run with None context
        result = PipelineResult()
        context = None
        run_id = "test_none_context_run"

        # Should not raise an exception
        async for _ in execution_manager.execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify that None context was handled gracefully
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        assert workflow_state["pipeline_context"] is None

    @pytest.mark.asyncio
    async def test_persistence_with_complex_nested_objects(self, sqlite_backend_factory):
        """Test that persistence works with complex nested objects."""
        backend = sqlite_backend_factory("complex_nested.db")
        state_manager = StateManager(state_backend=backend)

        # Create context with complex nested objects
        class ComplexContext(PipelineContext):
            nested_dict: Any = {
                "level1": {
                    "level2": {"level3": {"data": [1, 2, 3, 4, 5], "nested": {"deep": "value"}}}
                }
            }
            list_of_dicts: List[Any] = [
                {"id": i, "data": f"item_{i}", "nested": {"value": i * 2}} for i in range(10)
            ]

        agent = StubAgent(["output"])
        # step = Step.model_validate({"name": "complex_object_step", "agent": agent})

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_complex_object_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run with complex context
        result = PipelineResult()
        context = ComplexContext(initial_prompt="test")
        run_id = "test_complex_object_run"

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(
            steps=[Step.model_validate({"name": "complex_object_step", "agent": agent})]
        )

        # Should not raise an exception
        async for _ in ExecutionManager(
            pipeline=pipeline, state_manager=state_manager
        ).execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify complex objects were serialized correctly
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        assert "nested_dict" in workflow_state["pipeline_context"]
        assert "list_of_dicts" in workflow_state["pipeline_context"]

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow due to performance measurement
    async def test_persistence_performance_under_load(self, sqlite_backend_factory):
        """Test persistence performance under high load."""
        backend = sqlite_backend_factory("performance_load.db")
        state_manager = StateManager(state_backend=backend)

        # Create a simple agent

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "load_test_run_0",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run many pipelines quickly
        async def run_pipeline(run_id: str):
            result = PipelineResult()
            context = PipelineContext(initial_prompt=f"test_{run_id}")
            # Each pipeline gets its own agent and step
            agent = StubAgent(["output"])
            step = Step.model_validate({"name": "load_test_step", "agent": agent})
            from flujo.domain.dsl.pipeline import Pipeline

            pipeline = Pipeline(steps=[step])
            execution_manager = ExecutionManager(pipeline=pipeline, state_manager=state_manager)

            async for _ in execution_manager.execute_steps(
                start_idx=0,
                data="test_input",
                context=context,
                result=result,
                run_id=run_id,
            ):
                pass

        # Run 50 pipelines quickly
        start_time = time.perf_counter()
        tasks = [run_pipeline(f"load_test_run_{i}") for i in range(50)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert total_time < 10.0, f"Load test took {total_time:.3f}s"

        # Verify all runs were persisted
        for i in range(50):
            workflow_state = await backend.load_state(f"load_test_run_{i}")
            assert workflow_state is not None

    @pytest.mark.asyncio
    async def test_persistence_with_circular_references(self, sqlite_backend_factory):
        """Test that persistence handles circular references gracefully."""
        backend = sqlite_backend_factory("circular_refs.db")
        state_manager = StateManager(state_backend=backend)

        # Create context that might have circular references
        from pydantic import PrivateAttr

        class CircularContext(PipelineContext):
            _self_reference: Any = PrivateAttr()

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._self_reference = self  # Circular reference

        agent = StubAgent(["output"])
        step = Step.model_validate({"name": "circular_ref_step", "agent": agent})

        from flujo.domain.dsl.pipeline import Pipeline

        pipeline = Pipeline(steps=[step])

        # Register the run in the backend
        now = datetime.now(timezone.utc).isoformat()
        await state_manager.record_run_start(
            "test_circular_ref_run",
            str(uuid.uuid4()),
            "test_pipeline",
            "1.0",
            created_at=now,
            updated_at=now,
        )

        # Run with circular reference context
        result = PipelineResult()
        context = CircularContext(initial_prompt="test")
        run_id = "test_circular_ref_run"

        # Should not raise an exception
        async for _ in ExecutionManager(
            pipeline=pipeline, state_manager=state_manager
        ).execute_steps(
            start_idx=0,
            data="test_input",
            context=context,
            result=result,
            run_id=run_id,
        ):
            pass

        # Verify that circular references were handled gracefully
        workflow_state = await backend.load_state(run_id)
        assert workflow_state is not None
        # The circular reference should be handled by robust_serialize
        assert "pipeline_context" in workflow_state
