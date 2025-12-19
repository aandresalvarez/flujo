"""Tests for pipeline visualization functionality."""

from typing import Any
from flujo.domain import Step, Pipeline, StepConfig
from flujo.visualization.visualize import visualize, visualize_with_detail_level
from flujo.testing import DummyPlugin
from flujo.domain.plugins import PluginOutcome


class TestAgent:
    """Simple test agent for visualization tests."""

    async def run(self, data: Any, **kwargs: Any) -> str:
        return f"processed: {data}"


def test_simple_pipeline_visualization():
    """Test visualization of a simple linear pipeline."""
    step1 = Step.model_validate({"name": "Extract", "agent": TestAgent()})
    step2 = Step.model_validate({"name": "Transform", "agent": TestAgent()})
    step3 = Step.model_validate({"name": "Load", "agent": TestAgent()})

    pipeline = step1 >> step2 >> step3
    mermaid = visualize(pipeline)

    # Check that it's a valid Mermaid graph
    assert mermaid.startswith("graph TD")

    # Check that all steps are represented
    assert "Extract" in mermaid
    assert "Transform" in mermaid
    assert "Load" in mermaid

    # Check that edges connect the steps
    assert "-->" in mermaid


def test_pipeline_with_validation_steps():
    """Test visualization of steps with plugins/validators."""
    plugin = DummyPlugin(outcomes=[PluginOutcome(success=True, feedback="ok")])
    step1 = Step.model_validate(
        {"name": "Validate", "agent": TestAgent(), "plugins": [(plugin, 0)]}
    )
    step2 = Step.model_validate({"name": "Process", "agent": TestAgent(), "validators": []})

    pipeline = step1 >> step2
    mermaid = visualize(pipeline)

    # Check for validation annotation
    assert "🛡️" in mermaid


def test_pipeline_with_retry_config():
    """Test visualization of steps with retry configuration."""
    step1 = Step.model_validate({"name": "NormalStep", "agent": TestAgent()})
    step2 = Step.model_validate(
        {"name": "RetryStep", "agent": TestAgent(), "config": StepConfig(max_retries=3)}
    )

    pipeline = step1 >> step2
    mermaid = visualize(pipeline)

    # Check for dashed edge (retry indicator)
    assert "-.->" in mermaid


def test_loop_step_visualization():
    """Test visualization of LoopStep with nested pipeline."""
    inner_step = Step.model_validate({"name": "Inner", "agent": TestAgent()})
    inner_pipeline = Pipeline.from_step(inner_step)

    loop_step = Step.loop_until(
        name="RefineLoop",
        loop_body_pipeline=inner_pipeline,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=3,
    )

    pipeline = Pipeline.from_step(loop_step)
    mermaid = visualize(pipeline)

    # Check for loop node shape
    assert '("Loop: RefineLoop")' in mermaid

    # Check for subgraph
    assert 'subgraph "Loop Body: RefineLoop"' in mermaid

    # Check for exit path
    assert "Exit" in mermaid


def test_conditional_step_visualization():
    """Test visualization of ConditionalStep with branches."""
    branch1_step = Step.model_validate({"name": "CodeGen", "agent": TestAgent()})
    branch2_step = Step.model_validate({"name": "QAStep", "agent": TestAgent()})

    branch1_pipeline = Pipeline.from_step(branch1_step)
    branch2_pipeline = Pipeline.from_step(branch2_step)

    conditional_step = Step.branch_on(
        name="Router",
        condition_callable=lambda out, ctx: "code" if "code" in str(out) else "qa",
        branches={
            "code": branch1_pipeline,
            "qa": branch2_pipeline,
        },
    )

    pipeline = Pipeline.from_step(conditional_step)
    mermaid = visualize(pipeline)

    # Check for conditional node shape
    assert '{"Branch: Router"}' in mermaid

    # Check for branch subgraphs
    assert 'subgraph "Branch: code"' in mermaid
    assert 'subgraph "Branch: qa"' in mermaid

    # Check for branch labels on edges
    assert '|"code"|' in mermaid
    assert '|"qa"|' in mermaid


def test_parallel_step_visualization():
    """Test visualization of ParallelStep with concurrent branches."""
    branch1_step = Step.model_validate({"name": "ProcessA", "agent": TestAgent()})
    branch2_step = Step.model_validate({"name": "ProcessB", "agent": TestAgent()})

    parallel_step = Step.parallel(
        name="ParallelProcess",
        branches={
            "A": Pipeline.from_step(branch1_step),
            "B": Pipeline.from_step(branch2_step),
        },
    )

    pipeline = Pipeline.from_step(parallel_step)
    mermaid = visualize(pipeline)

    # Check for parallel node shape
    assert '{{"Parallel: ParallelProcess"}}' in mermaid

    # Check for parallel subgraphs
    assert 'subgraph "Parallel: A"' in mermaid
    assert 'subgraph "Parallel: B"' in mermaid


def test_human_in_the_loop_visualization():
    """Test visualization of HumanInTheLoopStep."""
    hitl_step = Step.human_in_the_loop("UserApproval", "Please review the result")

    pipeline = Pipeline.from_step(hitl_step)
    mermaid = visualize(pipeline)

    # Check for human step shape
    assert "[/Human: UserApproval/]" in mermaid


def test_complex_nested_pipeline_visualization():
    """Test visualization of a complex pipeline with nested structures."""
    # Create a complex pipeline: loop -> conditional -> parallel
    inner_step = Step.model_validate({"name": "Inner", "agent": TestAgent()})
    inner_pipeline = Pipeline.from_step(inner_step)

    # Loop step
    loop_step = Step.loop_until(
        name="MainLoop",
        loop_body_pipeline=inner_pipeline,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=2,
    )

    # Conditional step
    branch_step = Step.model_validate({"name": "BranchStep", "agent": TestAgent()})
    conditional_step = Step.branch_on(
        name="Decision",
        condition_callable=lambda out, ctx: "branch",
        branches={"branch": Pipeline.from_step(branch_step)},
    )

    # Parallel step
    parallel_step = Step.parallel(
        name="Concurrent",
        branches={
            "X": Pipeline.from_step(Step.model_validate({"name": "TaskX", "agent": TestAgent()})),
            "Y": Pipeline.from_step(Step.model_validate({"name": "TaskY", "agent": TestAgent()})),
        },
    )

    # Combine all
    complex_pipeline = loop_step >> conditional_step >> parallel_step

    # Test auto-detection (should choose low detail for complex pipeline)
    mermaid_auto = visualize(complex_pipeline)

    # Verify auto-detection chose low detail (emoji format)
    assert '("🔄 MainLoop")' in mermaid_auto
    assert '{"🔀 Decision"}' in mermaid_auto
    assert "{{⚡ Concurrent}}" in mermaid_auto

    # Test high detail explicitly
    mermaid_high = visualize_with_detail_level(complex_pipeline, "high")

    # Verify high detail format
    assert '("Loop: MainLoop")' in mermaid_high
    assert '{"Branch: Decision"}' in mermaid_high
    assert '{{"Parallel: Concurrent"}}' in mermaid_high

    # Verify subgraphs in high detail
    assert 'subgraph "Loop Body: MainLoop"' in mermaid_high
    assert 'subgraph "Branch: branch"' in mermaid_high
    assert 'subgraph "Parallel: X"' in mermaid_high
    assert 'subgraph "Parallel: Y"' in mermaid_high


def test_pipeline_with_mixed_configurations():
    """Test visualization of pipeline with mixed step configurations."""
    # Step with validation and retries
    validated_step = Step.model_validate(
        {
            "name": "ValidatedStep",
            "agent": TestAgent(),
            "plugins": [(DummyPlugin(outcomes=[PluginOutcome(success=True)]), 0)],
            "config": StepConfig(max_retries=3),
        }
    )

    # Step with just retries
    retry_step = Step.model_validate(
        {"name": "RetryStep", "agent": TestAgent(), "config": StepConfig(max_retries=2)}
    )

    # Normal step
    normal_step = Step.model_validate({"name": "NormalStep", "agent": TestAgent()})

    pipeline = validated_step >> retry_step >> normal_step
    mermaid = visualize(pipeline)

    # Check for validation annotation
    assert "ValidatedStep 🛡️" in mermaid

    # Check for retry edges (dashed lines)
    assert "-.->" in mermaid

    # Check that normal step doesn't have validation annotation
    assert "NormalStep ️" not in mermaid
