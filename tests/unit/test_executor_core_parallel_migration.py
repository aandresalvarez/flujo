"""
Test that the ExecutorCore's new parallel and dynamic router step methods are working correctly.

This test verifies that FSD 4 of 6 has been successfully implemented.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.dsl.step import Step, MergeStrategy, BranchFailureStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import StepResult


class TestExecutorCoreParallelMigration:
    """Test that ExecutorCore correctly handles ParallelStep and DynamicParallelRouterStep."""

    @pytest.fixture
    def executor_core(self):
        """Create an ExecutorCore instance for testing."""
        return ExecutorCore()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.run = AsyncMock(return_value="test_output")
        return agent

    @pytest.fixture
    def simple_step(self, mock_agent):
        """Create a simple step for testing."""
        return Step(
            name="test_step",
            agent=mock_agent,
        )

    @pytest.fixture
    def parallel_step(self, simple_step):
        """Create a parallel step for testing."""
        return ParallelStep(
            name="test_parallel",
            branches={
                "branch1": Step(name="branch1", agent=Mock()),
                "branch2": Step(name="branch2", agent=Mock()),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

    @pytest.fixture
    def dynamic_router_step(self, mock_agent):
        """Create a dynamic router step for testing."""
        router_agent = Mock()
        router_agent.run = AsyncMock(return_value=["branch1"])

        return DynamicParallelRouterStep(
            name="test_router",
            router_agent=router_agent,
            branches={
                "branch1": Pipeline(steps=[Step(name="branch1", agent=Mock())]),
                "branch2": Pipeline(steps=[Step(name="branch2", agent=Mock())]),
            },
            merge_strategy=MergeStrategy.NO_MERGE,
            on_branch_failure=BranchFailureStrategy.PROPAGATE,
        )

    @pytest.mark.asyncio
    async def test_executor_core_handles_parallel_step(self, executor_core, parallel_step):
        """Test that ExecutorCore correctly identifies and handles ParallelStep."""
        # Mock the _handle_parallel_step method to verify it's called
        with patch.object(
            executor_core, "_handle_parallel_step", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_parallel", success=True, output={})

            result = await executor_core.execute(
                step=parallel_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that _handle_parallel_step was called
            mock_handle.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_core_handles_dynamic_router_step(
        self, executor_core, dynamic_router_step
    ):
        """Test that ExecutorCore correctly identifies and handles DynamicParallelRouterStep."""
        # Mock the _handle_dynamic_router_step method to verify it's called
        with patch.object(
            executor_core, "_handle_dynamic_router_step", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_router", success=True, output={})

            result = await executor_core.execute(
                step=dynamic_router_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that _handle_dynamic_router_step was called
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
        # Mock the entire _handle_parallel_step method to verify it's called
        with patch.object(
            executor_core, "_handle_parallel_step", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_parallel", success=True, output={})

            result = await executor_core.execute(
                step=parallel_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
            )

            # Verify that _handle_parallel_step was called
            mock_handle.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_core_dynamic_router_delegates_to_parallel(
        self, executor_core, dynamic_router_step
    ):
        """Test that DynamicParallelRouterStep delegates to ParallelStep correctly."""
        # Mock the router agent to return specific branches
        dynamic_router_step.router_agent.run = AsyncMock(return_value=["branch1"])

        # Mock _handle_parallel_step to verify it's called
        with patch.object(
            executor_core, "_handle_parallel_step", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = StepResult(name="test_router", success=True, output={})

            result = await executor_core._handle_dynamic_router_step(
                router_step=dynamic_router_step,
                data="test_data",
                context=None,
                resources=None,
                limits=None,
                context_setter=None,
            )

            # Verify that _handle_parallel_step was called
            mock_handle.assert_called_once()
            assert result.success is True
            assert result.metadata_["executed_branches"] == ["branch1"]
