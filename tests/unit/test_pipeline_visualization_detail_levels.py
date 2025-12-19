"""Unit tests for pipeline visualization detail levels."""

import pytest

from flujo.domain import Step, StepConfig, Pipeline
from flujo.visualization.visualize import (
    _calculate_complexity_score,
    _determine_optimal_detail_level,
    visualize,
    visualize_with_detail_level,
)
from typing import Any


class TestAgent:
    """Simple test agent for testing."""

    async def run(self, data: str, **kwargs: Any) -> str:
        return f"processed: {data}"


class TestPipelineVisualizationDetailLevels:
    """Test pipeline visualization with different detail levels."""

    def test_detail_level_validation(self):
        """Test that invalid detail levels raise ValueError."""
        pipeline = Pipeline.from_step(Step.model_validate({"name": "Test", "agent": TestAgent()}))

        with pytest.raises(ValueError, match="Invalid detail_level"):
            visualize_with_detail_level(pipeline, "invalid")

    def test_complexity_score_calculation(self):
        """Test complexity score calculation for different pipeline types."""
        # Simple pipeline
        simple_pipeline = Pipeline.from_step(
            Step.model_validate({"name": "Test", "agent": TestAgent()})
        )
        assert _calculate_complexity_score(simple_pipeline) == 1

        # Pipeline with retries
        retry_pipeline = Pipeline.from_step(
            Step.model_validate(
                {"name": "Test", "agent": TestAgent(), "config": StepConfig(max_retries=3)}
            )
        )
        assert _calculate_complexity_score(retry_pipeline) == 2

        # Pipeline with loop
        loop_body = Pipeline.from_step(
            Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
        )
        loop_step = Step.loop_until(
            name="TestLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=lambda out, ctx: True,
        )
        loop_pipeline = Pipeline.from_step(loop_step)
        # Score: 1 (base) + 3 (loop) + 2 (nested step) = 6
        assert _calculate_complexity_score(loop_pipeline) == 6

        # Pipeline with conditional
        conditional_step = Step.branch_on(
            name="TestBranch",
            condition_callable=lambda out, ctx: "a",
            branches={
                "a": Pipeline.from_step(
                    Step.model_validate({"name": "BranchA", "agent": TestAgent()})
                )
            },
        )
        conditional_pipeline = Pipeline.from_step(conditional_step)
        # Score: 1 (base) + 2 (conditional) + 2 (branch) = 5
        assert _calculate_complexity_score(conditional_pipeline) == 5

    def test_optimal_detail_level_detection(self):
        """Test automatic detail level detection based on complexity."""
        # Low complexity (score < 8) -> high detail
        simple_pipeline = Pipeline.from_step(
            Step.model_validate({"name": "Test", "agent": TestAgent()})
        )
        assert _determine_optimal_detail_level(simple_pipeline) == "high"

        # Medium complexity (score 8-14) -> medium detail
        # Create a pipeline with score around 10
        steps = []
        for i in range(5):
            step = Step.model_validate(
                {"name": f"Step{i}", "agent": TestAgent(), "config": StepConfig(max_retries=2)}
            )
            steps.append(step)

        # Create pipeline by chaining steps
        medium_pipeline = Pipeline.from_step(steps[0])
        for step in steps[1:]:
            medium_pipeline = medium_pipeline >> step

        # Score should be 5 (base) + 5 (retries) = 10
        assert _determine_optimal_detail_level(medium_pipeline) == "medium"

        # High complexity (score >= 15) -> low detail
        # Create a complex pipeline with loops and conditionals
        loop_body = Pipeline.from_step(
            Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
        )
        for i in range(3):
            loop_body = loop_body >> Step.model_validate(
                {"name": f"LoopStep{i}", "agent": TestAgent()}
            )

        loop_step = Step.loop_until(
            name="ComplexLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=lambda out, ctx: True,
        )

        conditional_step = Step.branch_on(
            name="ComplexBranch",
            condition_callable=lambda out, ctx: "a",
            branches={
                "a": Pipeline.from_step(
                    Step.model_validate({"name": "BranchA", "agent": TestAgent()})
                )
                >> Step.model_validate({"name": "BranchA2", "agent": TestAgent()}),
                "b": Pipeline.from_step(
                    Step.model_validate({"name": "BranchB", "agent": TestAgent()})
                )
                >> Step.model_validate({"name": "BranchB2", "agent": TestAgent()}),
            },
        )

        complex_pipeline = Pipeline.from_step(loop_step) >> conditional_step
        # Score should be: 2 (base) + 9 (loop with 3 nested steps) + 6 (conditional with 2 branches) = 17
        assert _determine_optimal_detail_level(complex_pipeline) == "low"

    def test_high_detail_mermaid_generation(self):
        """Test high detail Mermaid generation with all features."""
        # Create a pipeline with various step types
        loop_body = Pipeline.from_step(
            Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
        )
        loop_step = Step.loop_until(
            name="TestLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=lambda out, ctx: True,
        )

        conditional_step = Step.branch_on(
            name="TestBranch",
            condition_callable=lambda out, ctx: "a",
            branches={
                "a": Pipeline.from_step(
                    Step.model_validate({"name": "BranchA", "agent": TestAgent()})
                )
            },
        )

        pipeline = Pipeline.from_step(loop_step) >> conditional_step
        mermaid = visualize_with_detail_level(pipeline, "high")

        # Check for high detail features
        assert "subgraph" in mermaid  # Should have subgraphs
        assert "Loop Body" in mermaid  # Should show loop body
        assert "Branch: a" in mermaid  # Should show branch details
        assert "Exit" in mermaid  # Should show exit conditions
        assert "join" in mermaid  # Should show join nodes

    def test_medium_detail_mermaid_generation(self):
        """Test medium detail Mermaid generation with simplified structure."""
        # Create a pipeline with various step types
        loop_step = Step.loop_until(
            name="TestLoop",
            loop_body_pipeline=Pipeline.from_step(
                Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
            ),
            exit_condition_callable=lambda out, ctx: True,
        )

        conditional_step = Step.branch_on(
            name="TestBranch",
            condition_callable=lambda out, ctx: "a",
            branches={
                "a": Pipeline.from_step(
                    Step.model_validate({"name": "BranchA", "agent": TestAgent()})
                )
            },
        )

        pipeline = Pipeline.from_step(loop_step) >> conditional_step
        mermaid = visualize_with_detail_level(pipeline, "medium")

        # Check for medium detail features
        assert "🔄" in mermaid  # Should have emojis
        assert "🔀" in mermaid  # Should have emojis
        assert "subgraph" not in mermaid  # Should not have subgraphs
        assert "Loop Body" not in mermaid  # Should not show detailed structure

    def test_low_detail_mermaid_generation(self):
        """Test low detail Mermaid generation with minimal information."""
        # Create a pipeline with various step types
        simple_steps = [
            Step.model_validate({"name": f"Step{i}", "agent": TestAgent()}) for i in range(3)
        ]
        loop_step = Step.loop_until(
            name="TestLoop",
            loop_body_pipeline=Pipeline.from_step(
                Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
            ),
            exit_condition_callable=lambda out, ctx: True,
        )

        # Create pipeline by chaining
        pipeline = Pipeline.from_step(simple_steps[0])
        for step in simple_steps[1:]:
            pipeline = pipeline >> step
        pipeline = pipeline >> loop_step

        mermaid = visualize_with_detail_level(pipeline, "low")

        # Check for low detail features
        assert "Processing:" in mermaid  # Should group simple steps
        assert "🔄" in mermaid  # Should have emojis
        # Should not have individual nodes for Step0, Step1, Step2
        assert '["Step0"]' not in mermaid
        assert '["Step1"]' not in mermaid
        assert '["Step2"]' not in mermaid

    def test_auto_detail_level_selection(self):
        """Test that auto detail level selects the appropriate level."""
        # Simple pipeline should select high detail
        simple_pipeline = Pipeline.from_step(
            Step.model_validate({"name": "Test", "agent": TestAgent()})
        )
        mermaid_auto = visualize_with_detail_level(simple_pipeline, "auto")
        mermaid_high = visualize_with_detail_level(simple_pipeline, "high")
        assert mermaid_auto == mermaid_high

        # Complex pipeline should select low detail
        loop_body = Pipeline.from_step(
            Step.model_validate({"name": "LoopStep", "agent": TestAgent()})
        )
        for i in range(5):
            loop_body = loop_body >> Step.model_validate(
                {"name": f"LoopStep{i}", "agent": TestAgent()}
            )

        loop_step = Step.loop_until(
            name="ComplexLoop",
            loop_body_pipeline=loop_body,
            exit_condition_callable=lambda out, ctx: True,
        )

        complex_pipeline = Pipeline.from_step(loop_step)
        mermaid_auto = visualize_with_detail_level(complex_pipeline, "auto")
        mermaid_low = visualize_with_detail_level(complex_pipeline, "low")
        assert mermaid_auto == mermaid_low

    def test_visualize_uses_auto(self):
        """Test that visualize() uses auto detail level."""
        pipeline = Pipeline.from_step(Step.model_validate({"name": "Test", "agent": TestAgent()}))
        mermaid_default = visualize(pipeline)
        mermaid_auto = visualize_with_detail_level(pipeline, "auto")
        assert mermaid_default == mermaid_auto

    def test_retry_annotations_in_different_levels(self):
        """Test that retry annotations appear appropriately in different detail levels."""
        step1 = Step.model_validate({"name": "A", "agent": TestAgent()})
        step2 = Step.model_validate(
            {"name": "B", "agent": TestAgent(), "config": StepConfig(max_retries=3)}
        )
        pipeline = Pipeline.from_step(step1) >> step2

        # High detail should show retry edges
        high_mermaid = visualize_with_detail_level(pipeline, "high")
        assert "-.->" in high_mermaid  # Dashed edge for retries

        # Medium detail should not show retry edges
        medium_mermaid = visualize_with_detail_level(pipeline, "medium")
        assert "-.->" not in medium_mermaid
        assert "-->" in medium_mermaid  # Regular edges only

        # Low detail should not show retry edges
        low_mermaid = visualize_with_detail_level(pipeline, "low")
        assert "-.->" not in low_mermaid

    def test_validation_annotations_in_different_levels(self):
        """Test that validation annotations appear appropriately in different detail levels."""
        # Create a step with plugins/validators (simulated)
        step = Step.model_validate({"name": "Test", "agent": TestAgent()})
        step.plugins = ["plugin1"]  # Simulate having plugins
        pipeline = Pipeline.from_step(step)

        # High detail should show validation annotations
        high_mermaid = visualize_with_detail_level(pipeline, "high")
        assert "🛡️" in high_mermaid

        # Medium detail should show validation annotations
        medium_mermaid = visualize_with_detail_level(pipeline, "medium")
        assert "🛡️" in medium_mermaid

        # Low detail should not show validation annotations (grouped steps)
        low_mermaid = visualize_with_detail_level(pipeline, "low")
        assert "🛡️" not in low_mermaid

    def test_parallel_step_visualization(self):
        """Test parallel step visualization in different detail levels."""
        parallel_step = Step.parallel(
            name="TestParallel",
            branches={
                "Branch1": Pipeline.from_step(
                    Step.model_validate({"name": "Branch1Step", "agent": TestAgent()})
                ),
                "Branch2": Pipeline.from_step(
                    Step.model_validate({"name": "Branch2Step", "agent": TestAgent()})
                ),
            },
        )
        pipeline = Pipeline.from_step(parallel_step)

        # High detail should show subgraphs for each branch
        high_mermaid = visualize_with_detail_level(pipeline, "high")
        assert "Parallel: Branch1" in high_mermaid
        assert "Parallel: Branch2" in high_mermaid
        assert "join" in high_mermaid

        # Medium detail should show parallel step without subgraphs
        medium_mermaid = visualize_with_detail_level(pipeline, "medium")
        assert "⚡" in medium_mermaid
        assert "Parallel: Branch1" not in medium_mermaid
        assert "Parallel: Branch2" not in medium_mermaid

        # Low detail should show parallel step
        low_mermaid = visualize_with_detail_level(pipeline, "low")
        assert "⚡" in low_mermaid

    def test_human_in_the_loop_visualization(self):
        """Test human-in-the-loop step visualization in different detail levels."""
        hitl_step = Step.human_in_the_loop("TestHITL", "Please review")
        pipeline = Pipeline.from_step(hitl_step)

        # High detail should show human step with special shape
        high_mermaid = visualize_with_detail_level(pipeline, "high")
        assert "/Human:" in high_mermaid

        # Medium detail should show human step with emoji
        medium_mermaid = visualize_with_detail_level(pipeline, "medium")
        assert "👤" in medium_mermaid

        # Low detail should show human step with emoji
        low_mermaid = visualize_with_detail_level(pipeline, "low")
        assert "👤" in low_mermaid
