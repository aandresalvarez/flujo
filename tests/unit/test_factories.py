"""Unit tests for pipeline factory functions."""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes.factories import (
    make_default_pipeline,
    make_agentic_loop_pipeline,
    make_state_machine_pipeline,
    run_default_pipeline,
    run_agentic_loop_pipeline,
)
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    Task,
    Checklist,
    Candidate,
    ChecklistItem,
    PipelineContext,
    PipelineResult,
)
from flujo.domain.commands import FinishCommand
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo


class TestMakeDefaultPipeline:
    """Test the make_default_pipeline factory function."""

    def test_creates_pipeline_object(self):
        """Test that make_default_pipeline returns a Pipeline object."""
        review_agent = AsyncMock()
        solution_agent = AsyncMock()
        validator_agent = AsyncMock()

        pipeline = make_default_pipeline(
            review_agent=review_agent,
            solution_agent=solution_agent,
            validator_agent=validator_agent,
        )

        assert isinstance(pipeline, Pipeline)

    def test_creates_pipeline_with_reflection(self):
        """Test that make_default_pipeline includes reflection step when provided."""
        review_agent = AsyncMock()
        solution_agent = AsyncMock()
        validator_agent = AsyncMock()
        reflection_agent = AsyncMock()

        pipeline = make_default_pipeline(
            review_agent=review_agent,
            solution_agent=solution_agent,
            validator_agent=validator_agent,
            reflection_agent=reflection_agent,
        )

        assert isinstance(pipeline, Pipeline)
        # The pipeline should have 4 steps: review, solution, validate, reflection
        assert len(pipeline.steps) == 4

    def test_custom_max_retries(self):
        """Test that max_retries parameter is respected."""
        review_agent = AsyncMock()
        solution_agent = AsyncMock()
        validator_agent = AsyncMock()

        pipeline = make_default_pipeline(
            review_agent=review_agent,
            solution_agent=solution_agent,
            validator_agent=validator_agent,
            max_retries=5,
        )

        assert isinstance(pipeline, Pipeline)
        # Check that the first step has the custom max_retries
        assert pipeline.steps[0].config.max_retries == 5


class TestMakeAgenticLoopPipeline:
    """Test the make_agentic_loop_pipeline factory function."""

    def test_creates_pipeline_object(self):
        """Test that make_agentic_loop_pipeline returns a Pipeline object."""
        planner_agent = StubAgent([FinishCommand(final_answer="done")])
        agent_registry = {}

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner_agent,
            agent_registry=agent_registry,
        )

        assert isinstance(pipeline, Pipeline)

    def test_custom_max_loops_and_retries(self):
        """Test that max_loops and max_retries parameters are respected."""
        planner_agent = StubAgent([FinishCommand(final_answer="done")])
        agent_registry = {}

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner_agent,
            agent_registry=agent_registry,
            max_loops=10,
            max_retries=5,
        )

        assert isinstance(pipeline, Pipeline)
        # The pipeline should contain a LoopStep
        loop_step = pipeline.steps[0]
        assert loop_step.max_loops == 10
        assert loop_step.config.max_retries == 5


class TestRunDefaultPipeline:
    """Test the run_default_pipeline convenience function."""

    @pytest.mark.asyncio
    async def test_runs_pipeline_successfully(self):
        """Test that run_default_pipeline executes and returns a Candidate."""
        # Create a simple pipeline
        review_agent = StubAgent(
            [
                Checklist(
                    items=[ChecklistItem(description="item1"), ChecklistItem(description="item2")]
                )
            ]
        )
        solution_agent = StubAgent(["test solution"])
        validator_agent = StubAgent(
            [
                Checklist(
                    items=[
                        ChecklistItem(description="validated_item1"),
                        ChecklistItem(description="validated_item2"),
                    ]
                )
            ]
        )

        pipeline = make_default_pipeline(
            review_agent=review_agent,
            solution_agent=solution_agent,
            validator_agent=validator_agent,
        )

        task = Task(prompt="test task")
        result = await run_default_pipeline(pipeline, task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert isinstance(result.checklist, Checklist)

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        """Test that run_default_pipeline returns None when processing fails."""
        # Create agents that don't produce expected outputs
        review_agent = StubAgent([Checklist(items=[ChecklistItem(description="item1")])])
        solution_agent = StubAgent([None])  # No solution
        validator_agent = StubAgent([Checklist(items=[ChecklistItem(description="item1")])])

        pipeline = make_default_pipeline(
            review_agent=review_agent,
            solution_agent=solution_agent,
            validator_agent=validator_agent,
        )

        task = Task(prompt="test task")
        result = await run_default_pipeline(pipeline, task)

        assert result is None


class TestRunAgenticLoopPipeline:
    """Test the run_agentic_loop_pipeline convenience function."""

    @pytest.mark.asyncio
    async def test_runs_pipeline_successfully(self):
        """Test that run_agentic_loop_pipeline executes and returns a result."""
        planner_agent = StubAgent([FinishCommand(final_answer="final result")])
        agent_registry = {}

        pipeline = make_agentic_loop_pipeline(
            planner_agent=planner_agent,
            agent_registry=agent_registry,
        )

        result = await run_agentic_loop_pipeline(pipeline, "test goal")

        assert isinstance(result, PipelineResult)
        assert result.final_pipeline_context is not None


class TestMakeStateMachinePipeline:
    """Test the make_state_machine_pipeline factory."""

    def test_creates_pipeline_object(self):
        async def _identity(x: str) -> str:
            return x

        class Ctx(PipelineContext):
            next_state: str = "A"
            is_complete: bool = False

        step_a = Step.from_mapper(_identity)
        pipeline = make_state_machine_pipeline(nodes={"A": step_a}, context_model=Ctx)
        assert isinstance(pipeline, Pipeline)

    def test_validates_context_fields(self):
        class BadCtx(PipelineContext):
            pass

        async def noop(x: str) -> str:
            return x

        with pytest.raises(AttributeError):
            make_state_machine_pipeline(
                nodes={"A": Step.from_mapper(noop)},
                context_model=BadCtx,
            )

    @pytest.mark.asyncio
    async def test_runs_until_complete(self):
        class Ctx(PipelineContext):
            next_state: str = "only"
            is_complete: bool = False

        async def only_state(data: str, *, context: Ctx) -> str:
            context.is_complete = True
            return "done"

        pipeline = make_state_machine_pipeline(
            nodes={"only": Step.from_callable(only_state, updates_context=True)},
            context_model=Ctx,
        )

        runner = create_test_flujo(pipeline, context_model=Ctx)
        result = None
        async for item in runner.run_async("go", initial_context_data={"initial_prompt": "go"}):
            result = item

        assert result is not None
        assert result.step_history[-1].output == "done"
