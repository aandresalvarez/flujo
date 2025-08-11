"""
Dynamic Parallel Router Golden Transcript Test

This test locks in the behavior of the Step.dynamic_parallel_branch primitive,
which is a specialized and powerful feature for runtime branch selection.
"""

import pytest
from typing import Any, List

from flujo.domain import Step, Pipeline
from flujo.domain.models import PipelineContext
from flujo.domain.dsl import MergeStrategy
from tests.conftest import create_test_flujo


class DynamicParallelContext(PipelineContext):
    """Context for dynamic parallel testing."""

    initial_prompt: str = "test"
    executed_branches: List[str] = []
    branch_results: dict = {}
    total_failures: int = 0


class StubRouterAgent:
    """Deterministic router agent for testing."""

    def __init__(self, branch_names: List[str]):
        self.branch_names = branch_names

    async def run(self, data: Any, *, context: DynamicParallelContext = None) -> List[str]:
        """Return the list of branch names to execute."""
        return self.branch_names


class StubBranchAgent:
    """Deterministic branch agent for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail

    async def run(self, data: Any, *, context: DynamicParallelContext = None) -> str:
        """Execute the branch logic."""
        print(f"[DEBUG] Branch {self.name} called with data: {data}")
        print(f"[DEBUG] Branch {self.name} context is None: {context is None}")

        if self.should_fail:
            print(f"[DEBUG] Branch {self.name} will fail intentionally")
            raise RuntimeError(f"Intentional failure in {self.name}")

        result = f"{self.name}_processed_{data}"

        if context:
            print(
                f"[DEBUG] Branch {self.name} context.executed_branches before: {context.executed_branches}"
            )
            context.executed_branches.append(self.name)
            context.branch_results[self.name] = result
            print(
                f"[DEBUG] Branch {self.name} context.executed_branches after: {context.executed_branches}"
            )
            print(f"[DEBUG] Branch {self.name} context.branch_results: {context.branch_results}")
        else:
            print(f"[DEBUG] Branch {self.name} no context provided")

        return result


@pytest.mark.asyncio
async def test_golden_transcript_dynamic_parallel():
    """Test the dynamic parallel router with deterministic behavior."""

    # Create the router agent
    router_agent = StubRouterAgent(["branch1", "branch2", "branch3"])

    # Create branch agents
    branch_agents = {
        "branch1": StubBranchAgent("branch1"),
        "branch2": StubBranchAgent("branch2"),
        "branch3": StubBranchAgent("branch3", should_fail=True),  # One failure
    }

    # Create the dynamic parallel pipeline
    dynamic_parallel_pipeline = Step.dynamic_parallel_branch(
        name="test_dynamic_parallel",
        router_agent=router_agent,
        branches={
            "branch1": Pipeline.from_step(
                Step.from_callable(branch_agents["branch1"].run, name="branch1")
            ),
            "branch2": Pipeline.from_step(
                Step.from_callable(branch_agents["branch2"].run, name="branch2")
            ),
            "branch3": Pipeline.from_step(
                Step.from_callable(branch_agents["branch3"].run, name="branch3")
            ),
        },
        on_branch_failure="ignore",
        merge_strategy=MergeStrategy.OVERWRITE,
    )

    # Initialize Flujo runner
    runner = create_test_flujo(dynamic_parallel_pipeline, context_model=DynamicParallelContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        "test_input",
        initial_context_data={
            "initial_prompt": "test",
            "executed_branches": [],
            "branch_results": {},
            "total_failures": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context and output
    final_context = result.final_pipeline_context
    final_output = result.step_history[-1].output

    # Dynamic parallel assertions
    # Only one branch should be executed successfully
    assert len(final_context.executed_branches) == 1
    assert "branch2" in final_context.executed_branches

    # Verify branch results
    assert "branch2" in final_context.branch_results
    assert final_context.branch_results["branch2"] == "branch2_processed_test_input"

    # Verify the output structure
    assert isinstance(final_output, dict)
    assert "branch1" in final_output
    assert "branch2" in final_output
    assert "branch3" in final_output

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")


@pytest.mark.asyncio
async def test_golden_transcript_dynamic_parallel_selective():
    """Test the dynamic parallel router with selective branch execution."""

    # Create the router agent that only selects some branches
    router_agent = StubRouterAgent(["branch1", "branch3"])

    # Create branch agents
    branch_agents = {
        "branch1": StubBranchAgent("branch1"),
        "branch2": StubBranchAgent("branch2"),
        "branch3": StubBranchAgent("branch3"),
    }

    # Create the dynamic parallel pipeline
    dynamic_parallel_pipeline = Step.dynamic_parallel_branch(
        name="test_dynamic_parallel_selective",
        router_agent=router_agent,
        branches={
            "branch1": Pipeline.from_step(
                Step.from_callable(branch_agents["branch1"].run, name="branch1")
            ),
            "branch2": Pipeline.from_step(
                Step.from_callable(branch_agents["branch2"].run, name="branch2")
            ),
            "branch3": Pipeline.from_step(
                Step.from_callable(branch_agents["branch3"].run, name="branch3")
            ),
        },
        on_branch_failure="ignore",
        merge_strategy=MergeStrategy.OVERWRITE,
    )

    # Initialize Flujo runner
    runner = create_test_flujo(dynamic_parallel_pipeline, context_model=DynamicParallelContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        "selective_input",
        initial_context_data={
            "initial_prompt": "test",
            "executed_branches": [],
            "branch_results": {},
            "total_failures": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context
    final_context = result.final_pipeline_context

    # Only selected branches should be executed
    assert len(final_context.executed_branches) == 1
    assert "branch3" in final_context.executed_branches
    assert "branch1" not in final_context.executed_branches  # Not selected

    # Verify branch results
    assert "branch3" in final_context.branch_results
    assert "branch1" not in final_context.branch_results

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")


@pytest.mark.asyncio
async def test_golden_transcript_dynamic_parallel_empty():
    """Test the dynamic parallel router with no branches selected."""

    # Create the router agent that selects no branches
    router_agent = StubRouterAgent([])

    # Create branch agents
    branch_agents = {
        "branch1": StubBranchAgent("branch1"),
        "branch2": StubBranchAgent("branch2"),
    }

    # Create the dynamic parallel pipeline
    dynamic_parallel_pipeline = Step.dynamic_parallel_branch(
        name="test_dynamic_parallel_empty",
        router_agent=router_agent,
        branches={
            "branch1": Pipeline.from_step(
                Step.from_callable(branch_agents["branch1"].run, name="branch1")
            ),
            "branch2": Pipeline.from_step(
                Step.from_callable(branch_agents["branch2"].run, name="branch2")
            ),
        },
        on_branch_failure="ignore",
        merge_strategy=MergeStrategy.OVERWRITE,
    )

    # Initialize Flujo runner
    runner = create_test_flujo(dynamic_parallel_pipeline, context_model=DynamicParallelContext)

    # Run the pipeline
    result = None
    async for r in runner.run_async(
        "empty_input",
        initial_context_data={
            "initial_prompt": "test",
            "executed_branches": [],
            "branch_results": {},
            "total_failures": 0,
        },
    ):
        result = r

    assert result is not None, "No result returned from runner.run_async()"

    # Get the final context
    final_context = result.final_pipeline_context

    # No branches should be executed
    assert len(final_context.executed_branches) == 0
    assert len(final_context.branch_results) == 0

    # Verify step history structure
    assert len(result.step_history) > 0
    for step_result in result.step_history:
        assert hasattr(step_result, "name")
        assert hasattr(step_result, "success")
        assert hasattr(step_result, "output")
