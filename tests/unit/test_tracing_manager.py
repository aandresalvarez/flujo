"""
Unit tests for the TraceManager hook.

This module contains tests for the TraceManager class, which is responsible for
managing trace spans during the execution of pipelines. The tests cover various
aspects of the TraceManager's functionality, including:
- Initialization and internal state management
- Building and maintaining a nested trace tree structure
- Integration with pipeline execution and step processing
- Handling of step failures and marking spans as failed
- Validation of the Span dataclass attributes

The tests use mock objects and a stub agent to simulate pipeline execution
scenarios and verify the correctness of the TraceManager's behavior.
"""

import pytest
from unittest.mock import Mock

from flujo.tracing.manager import TraceManager, Span
from flujo.domain.models import StepResult, PipelineResult
from flujo.domain.events import (
    PreRunPayload,
    PostRunPayload,
    PreStepPayload,
    PostStepPayload,
    OnStepFailurePayload,
)

from flujo.testing.utils import StubAgent
from flujo import Flujo


class TestTraceManager:
    """Test the TraceManager hook functionality."""

    def test_trace_manager_initialization(self):
        """Test that TraceManager initializes correctly."""
        manager = TraceManager()
        assert manager._span_stack == []
        assert manager._root_span is None

    @pytest.mark.asyncio
    async def test_trace_manager_builds_nested_tree(self):
        """Test that TraceManager builds a nested trace tree correctly."""
        manager = TraceManager()

        # Create a simple pipeline result
        pipeline_result = PipelineResult(step_history=[])

        # Simulate pre_run event
        mock_pipeline = Mock()
        mock_pipeline.name = "test_pipeline"

        pre_run_payload = PreRunPayload(
            event_name="pre_run", initial_input="test_input", context=None, resources=None
        )
        await manager.hook(pre_run_payload)

        # Verify root span was created
        assert manager._root_span is not None
        assert manager._root_span.name == "pipeline_root"
        assert manager._root_span.start_time > 0
        assert manager._root_span.parent_span_id is None
        assert manager._root_span.attributes["initial_input"] == "test_input"
        assert len(manager._span_stack) == 1
        assert manager._span_stack[0] == manager._root_span

        # Simulate pre_step event
        mock_step = Mock()
        mock_step.name = "test_step"

        pre_step_payload = PreStepPayload(
            event_name="pre_step",
            step=mock_step,
            step_input="step_input",
            context=None,
            resources=None,
        )
        await manager.hook(pre_step_payload)

        # Verify child span was created
        assert len(manager._span_stack) == 2
        child_span = manager._span_stack[1]
        assert child_span.name == "test_step"
        assert child_span.parent_span_id == manager._root_span.span_id
        assert child_span.start_time > 0
        assert child_span.attributes["step_type"] == "Mock"
        assert child_span.attributes["step_input"] == "step_input"

        # Simulate post_step event
        step_result = StepResult(
            name="test_step",
            output="test_output",
            success=True,
            attempts=1,
            latency_s=0.1,
            cost_usd=0.01,
            token_counts=100,
        )

        post_step_payload = PostStepPayload(
            event_name="post_step", step_result=step_result, context=None, resources=None
        )
        await manager.hook(post_step_payload)

        # Verify span was finalized
        assert len(manager._span_stack) == 1  # Back to root only
        child_span = manager._root_span.children[0]
        assert child_span.end_time is not None
        assert child_span.status == "completed"
        assert child_span.attributes["success"] is True
        assert child_span.attributes["attempts"] == 1
        assert child_span.attributes["latency_s"] == 0.1
        assert child_span.attributes["cost_usd"] == 0.01
        assert child_span.attributes["token_counts"] == 100

        # Simulate post_run event
        post_run_payload = PostRunPayload(
            event_name="post_run", pipeline_result=pipeline_result, context=None, resources=None
        )
        await manager.hook(post_run_payload)
        # Verify trace tree was built correctly in manager._root_span
        assert manager._root_span is not None
        assert manager._root_span.name == "pipeline_root"
        assert manager._root_span.end_time is not None
        assert manager._root_span.status == "completed"
        assert len(manager._root_span.children) == 1
        child_span = manager._root_span.children[0]
        assert child_span.name == "test_step"
        assert child_span.status == "completed"
        assert child_span.end_time is not None
        assert child_span.attributes["success"] is True
        assert child_span.attributes["attempts"] == 1
        assert child_span.attributes["latency_s"] == 0.1
        assert child_span.attributes["cost_usd"] == 0.01
        assert child_span.attributes["token_counts"] == 100

    @pytest.mark.asyncio
    async def test_trace_manager_integration(self):
        """Test that TraceManager works correctly in a real pipeline execution."""

        from flujo import Step

        # Create a simple pipeline
        step = Step.model_validate({"name": "test_step", "agent": StubAgent(["test_output"])})

        # Create runner with TraceManager
        runner = Flujo(step)

        # Run the pipeline and get the final result
        result = None
        async for r in runner.run_async("test_input"):
            result = r

        # Verify that trace tree was attached
        assert result is not None
        assert result.trace_tree is not None
        assert result.trace_tree.name == "pipeline_root"
        assert len(result.trace_tree.children) == 1
        assert result.trace_tree.children[0].name == "test_step"
        assert result.trace_tree.children[0].status == "completed"
        assert result.trace_tree.children[0].attributes["success"] is True

    @pytest.mark.asyncio
    async def test_step_failure_marks_span_as_failed(self):
        """Test that step failure marks the span as failed."""
        manager = TraceManager()

        # Setup root span and child span
        pre_run_payload = PreRunPayload(
            event_name="pre_run", initial_input="test_input", context=None, resources=None
        )
        await manager.hook(pre_run_payload)

        mock_step = Mock()
        mock_step.name = "test_step"

        pre_step_payload = PreStepPayload(
            event_name="pre_step",
            step=mock_step,
            step_input="step_input",
            context=None,
            resources=None,
        )
        await manager.hook(pre_step_payload)

        # Create failed StepResult
        step_result = StepResult(
            name="test_step",
            output=None,
            success=False,
            attempts=3,
            latency_s=0.5,
            feedback="Test error",
        )

        failure_payload = OnStepFailurePayload(
            event_name="on_step_failure", step_result=step_result, context=None, resources=None
        )
        await manager.hook(failure_payload)

        # Verify span was marked as failed
        assert len(manager._span_stack) == 1  # Back to root only
        child_span = manager._root_span.children[0]
        assert child_span.end_time is not None
        assert child_span.status == "failed"
        assert child_span.attributes["success"] is False
        assert child_span.attributes["attempts"] == 3
        assert child_span.attributes["latency_s"] == 0.5
        assert child_span.attributes["feedback"] == "Test error"
        assert child_span.attributes["token_counts"] == 0

    @pytest.mark.asyncio
    async def test_nested_spans_are_correctly_structured(self):
        """Test that nested spans maintain proper parent-child relationships."""
        manager = TraceManager()

        # Setup root span
        pre_run_payload = PreRunPayload(
            event_name="pre_run", initial_input="test_input", context=None, resources=None
        )
        await manager.hook(pre_run_payload)

        # Create two nested steps
        step1 = Mock()
        step1.name = "step1"

        step2 = Mock()
        step2.name = "step2"

        # Execute step1
        pre_step1_payload = PreStepPayload(
            event_name="pre_step", step=step1, step_input="input1", context=None, resources=None
        )
        await manager.hook(pre_step1_payload)

        # Execute step2 (nested under step1)
        pre_step2_payload = PreStepPayload(
            event_name="pre_step", step=step2, step_input="input2", context=None, resources=None
        )
        await manager.hook(pre_step2_payload)

        # Complete step2
        step2_result = StepResult(
            name="step2", output="output2", success=True, attempts=1, latency_s=0.1
        )
        post_step2_payload = PostStepPayload(
            event_name="post_step", step_result=step2_result, context=None, resources=None
        )
        await manager.hook(post_step2_payload)

        # Complete step1
        step1_result = StepResult(
            name="step1", output="output1", success=True, attempts=1, latency_s=0.2
        )
        post_step1_payload = PostStepPayload(
            event_name="post_step", step_result=step1_result, context=None, resources=None
        )
        await manager.hook(post_step1_payload)

        # Verify structure
        root_span = manager._root_span
        assert len(root_span.children) == 1
        step1_span = root_span.children[0]
        assert step1_span.name == "step1"
        assert len(step1_span.children) == 1
        step2_span = step1_span.children[0]
        assert step2_span.name == "step2"
        assert step2_span.parent_span_id == step1_span.span_id

    def test_span_dataclass_attributes(self):
        """Test that Span dataclass has correct attributes."""
        span = Span(
            span_id="test_span",
            name="test_name",
            start_time=123.456,
            end_time=124.456,
            parent_span_id="parent_span",
            attributes={"key": "value"},
            children=[],
            status="completed",
        )

        assert span.span_id == "test_span"
        assert span.name == "test_name"
        assert span.start_time == 123.456
        assert span.end_time == 124.456
        assert span.parent_span_id == "parent_span"
        assert span.attributes == {"key": "value"}
        assert span.children == []
        assert span.status == "completed"
