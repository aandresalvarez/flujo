"""Test cumulative usage limit enforcement in UltraStepExecutor."""

import pytest
from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor, _UsageTracker
from flujo.domain.models import UsageLimits


class TestUltraExecutorCumulativeLimits:
    """Test that UltraStepExecutor correctly implements cumulative usage tracking."""

    @pytest.fixture
    def executor(self):
        """Create an ultra executor for testing."""
        return UltraStepExecutor(enable_cache=False)

    @pytest.fixture
    def usage_tracker(self):
        """Create a usage tracker for direct testing."""
        return _UsageTracker()

    @pytest.fixture
    def cost_output_class(self):
        """Create a class that returns cost information."""

        class CostOutput:
            def __init__(self, cost: float = 0.1, tokens: int = 100):
                self.output = "test_output"
                self.cost_usd = cost
                self.token_counts = tokens

        return CostOutput

    @pytest.mark.asyncio
    async def test_usage_tracker_cumulative_tracking(self, usage_tracker):
        """Test that the usage tracker correctly accumulates costs and tokens."""

        # Add multiple usage increments
        await usage_tracker.add(0.1, 50)
        await usage_tracker.add(0.2, 75)
        await usage_tracker.add(0.05, 25)

        # Check cumulative totals
        total_cost, total_tokens = await usage_tracker.get_current_totals()
        assert abs(total_cost - 0.35) < 1e-10  # 0.1 + 0.2 + 0.05
        assert total_tokens == 150  # 50 + 75 + 25

    @pytest.mark.asyncio
    async def test_usage_tracker_limit_checking(self, usage_tracker):
        """Test that the guard method exists for backward compatibility."""

        # Add some usage
        await usage_tracker.add(0.1, 50)

        # Test that guard method exists and doesn't raise (since limit checking is now handled by governor)
        limits = UsageLimits(total_cost_usd_limit=0.2, total_tokens_limit=100)
        await usage_tracker.guard(limits)  # Should not raise since governor handles limits

        # Verify that usage tracking still works
        cost, tokens = await usage_tracker.get_current_totals()
        assert cost == 0.1
        assert tokens == 50

    @pytest.mark.asyncio
    async def test_usage_tracker_thread_safety(self, usage_tracker):
        """Test that the usage tracker is thread-safe for concurrent operations."""

        import asyncio

        # Simulate concurrent usage updates
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                task = usage_tracker.add(0.01, 10)  # Each adds $0.01 and 10 tokens
                tasks.append(task)

            # Run all tasks concurrently
            await asyncio.gather(*tasks)

            # Get final totals
            return await usage_tracker.get_current_totals()

        # Run concurrent updates
        final_cost, final_tokens = await add_usage_concurrently()

        # Verify totals are correct (10 * 0.01 = 0.1, 10 * 10 = 100)
        # Use approximate comparison for floating point precision
        assert abs(final_cost - 0.1) < 1e-10
        assert final_tokens == 100

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_usage_tracker_multiple_limit_checks(self, usage_tracker):
        """Test that multiple usage additions work correctly."""

        # Add usage incrementally
        await usage_tracker.add(0.05, 25)

        # Check totals after first addition
        cost, tokens = await usage_tracker.get_current_totals()
        assert cost == 0.05
        assert tokens == 25

        # Add more usage
        await usage_tracker.add(0.06, 25)

        # Check totals after second addition
        cost, tokens = await usage_tracker.get_current_totals()
        assert cost == 0.11  # 0.05 + 0.06
        assert tokens == 50  # 25 + 25

        # Add even more usage
        await usage_tracker.add(0.01, 10)

        # Check totals after third addition
        cost, tokens = await usage_tracker.get_current_totals()
        assert cost == 0.12  # 0.05 + 0.06 + 0.01
        assert tokens == 60  # 25 + 25 + 10

    @pytest.mark.asyncio
    async def test_usage_tracker_zero_limits(self, usage_tracker):
        """Test behavior with zero and None limits (guard is a no-op)."""

        await usage_tracker.add(0.1, 50)

        # Test with None limits (should not affect usage tracking)
        limits_none = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=None)
        await usage_tracker.guard(limits_none)  # Should not raise

        # Test with zero limits (guard is a no-op; enforcement happens via quota reservation)
        limits_zero = UsageLimits(total_cost_usd_limit=0.0, total_tokens_limit=0)
        await usage_tracker.guard(limits_zero)  # Should not raise

        # Verify that usage tracking still works
        cost, tokens = await usage_tracker.get_current_totals()
        assert cost == 0.1
        assert tokens == 50

    @pytest.mark.asyncio
    async def test_usage_tracker_precision_handling(self, usage_tracker):
        """Test that floating point precision is handled correctly."""

        # Add very small amounts
        await usage_tracker.add(0.0001, 1)
        await usage_tracker.add(0.0001, 1)
        await usage_tracker.add(0.0001, 1)

        # Check that small amounts accumulate correctly
        total_cost, total_tokens = await usage_tracker.get_current_totals()
        assert abs(total_cost - 0.0003) < 1e-10
        assert total_tokens == 3

        # Guard is a no-op; this should not raise.
        limits = UsageLimits(total_cost_usd_limit=0.0002, total_tokens_limit=2)
        await usage_tracker.guard(limits)

        # Verify that precision is maintained
        total_cost, total_tokens = await usage_tracker.get_current_totals()
        assert abs(total_cost - 0.0003) < 1e-10
        assert total_tokens == 3
