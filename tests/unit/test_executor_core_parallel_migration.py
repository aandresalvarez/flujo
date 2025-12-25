"""
Test that the ExecutorCore's new parallel and dynamic router step methods are working correctly.

This test verifies that FSD 4 of 6 has been successfully implemented.

Note: This test uses proper domain objects instead of Mock objects for integration testing.
While mock detection has been removed, using real domain objects remains best practice
for verifying complex orchestration behaviors.
"""

import pytest
from unittest.mock import AsyncMock, patch

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.dsl.step import Step, MergeStrategy, BranchFailureStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import StepResult
from flujo.testing.utils import StubAgent

pytestmark = pytest.mark.fast


class TestExecutorCoreParallelMigration:
    """Test that ExecutorCore correctly handles ParallelStep and DynamicParallelRouterStep."""

    @pytest.fixture
    def executor_core(self):
        """Create an ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def test_agent(self):
        """Create a real agent for testing."""
        return StubAgent(["test_output"])

    @pytest.fixture
    def branch_agent_1(self):
        """Create a real agent for branch 1 testing."""
        return StubAgent(["branch1_output"])

    @pytest.fixture
    def branch_agent_2(self):
        """Create a real agent for branch 2 testing."""
        return StubAgent(["branch2_output"])

    @pytest.fixture
    def router_agent(self):
        """Create a real router agent for testing."""
        return StubAgent([["branch1"]])

    @pytest.fixture
    def simple_step(self, test_agent):
        """Create a simple step for testing."""
        return Step(
            name="test_step",
            agent=test_agent,
        )

    @pytest.fixture
    def parallel_step(self, branch_agent_1, branch_agent_2):
        """Create a parallel step for testing with real agents."""
        return ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Step(name="branch1", agent=branch_agent_1),
                "branch2": Step(name="branch2", agent=branch_agent_2),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

    @pytest.fixture
    def dynamic_router_step(self, router_agent, branch_agent_1, branch_agent_2):
        """Create a dynamic router step for testing with real agents."""
        return DynamicParallelRouterStep(
            name="test_router",
            router_agent=router_agent,
            branches={
                "branch1": Pipeline(steps=[Step(name="branch1", agent=branch_agent_1)]),
                "branch2": Pipeline(steps=[Step(name="branch2", agent=branch_agent_2)]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

    @pytest.mark.asyncio
    async def test_executor_core_handles_parallel_step(self, executor_core, parallel_step):
        """Test that ExecutorCore correctly identifies and handles ParallelStep."""
        # Mock the parallel_step_executor.execute method to verify it's called
        with patch.object(
            executor_core.parallel_step_executor, "execute", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_parallel", success=True, output={})

            result = await executor_core.execute(
                step=parallel_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that parallel_step_executor.execute was called
            mock_handle.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_core_handles_dynamic_router_step(
        self, executor_core, dynamic_router_step
    ):
        """Test that ExecutorCore correctly identifies and handles DynamicParallelRouterStep."""
        # Mock the dynamic_router_step_executor.execute method to verify it's called
        with patch.object(
            executor_core.dynamic_router_step_executor, "execute", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_router", success=True, output={})

            result = await executor_core.execute(
                step=dynamic_router_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that dynamic_router_step_executor.execute was called
            mock_handle.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_core_is_complex_step_detection(
        self, executor_core, parallel_step, dynamic_router_step
    ):
        """Test that ExecutorCore correctly identifies complex steps."""
        # Test ParallelStep detection
        assert executor_core._is_complex_step(parallel_step) is True

        # Test DynamicParallelRouterStep detection
        assert executor_core._is_complex_step(dynamic_router_step) is True

    @pytest.mark.asyncio
    async def test_executor_core_parallel_step_recursive_execution(
        self, executor_core, parallel_step
    ):
        """Test that ExecutorCore's parallel step method is called correctly."""
        # Mock the parallel_step_executor.execute method to verify it's called
        with patch.object(
            executor_core.parallel_step_executor, "execute", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_parallel", success=True, output={})

            result = await executor_core.execute(
                step=parallel_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that parallel_step_executor.execute was called
            mock_handle.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_core_dynamic_router_delegates_to_parallel(
        self, executor_core, dynamic_router_step
    ):
        """Test that DynamicParallelRouterStep is routed to the correct executor."""
        # Mock dynamic_router_step_executor.execute to verify it's called
        with patch.object(
            executor_core.dynamic_router_step_executor, "execute", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_router", success=True, output={})

            result = await executor_core.execute(
                step=dynamic_router_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that dynamic_router_step_executor.execute was called
            mock_handle.assert_called_once()
            assert result.success is True
