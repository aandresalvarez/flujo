"""Stress test for thread-safe parallel usage governance.

This test verifies that the _ParallelUsageGovernor correctly handles high concurrency
and maintains accurate cost tracking across multiple concurrent branches.
"""

import asyncio
import pytest
from typing import Dict, Any

from flujo.application.core.ultra_executor import ExecutorCore
from flujo.domain.models import StepResult, UsageLimits, PipelineResult
from flujo.exceptions import UsageLimitExceededError
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import StubAgent


class TestParallelUsageGovernorStress:
    """Stress test for _ParallelUsageGovernor thread safety."""

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_high_concurrency(self):
        """Test that _ParallelUsageGovernor handles high concurrency correctly."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor with a high limit to test accuracy
        limits = UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=10000)
        governor = ExecutorCore._ParallelUsageGovernor(limits)

        # Create a proper StepResult for testing
        step_result = StepResult(name="stress_test_step")

        # Define known costs for each concurrent operation
        # Each operation will add a small, known cost
        operation_costs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        operation_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Expected total cost and tokens
        expected_total_cost = sum(operation_costs)
        expected_total_tokens = sum(operation_tokens)

        # Simulate concurrent usage updates with 10 branches
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                cost = operation_costs[i]
                tokens = operation_tokens[i]
                # Each task adds a known cost and token amount
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent usage updates
        results = await add_usage_concurrently()

        # Verify that no breach occurred (since we're well under the limit)
        assert not any(results), "No breach should occur with these costs"

        # Verify that the governor detected no breach
        assert not governor.breached()
        assert governor.get_error() is None

        # CRITICAL: Verify that the final aggregated cost is exactly the sum of individual costs
        # This tests that no data was lost due to race conditions
        assert governor.total_cost == expected_total_cost, (
            f"Expected total cost {expected_total_cost}, got {governor.total_cost}"
        )
        assert governor.total_tokens == expected_total_tokens, (
            f"Expected total tokens {expected_total_tokens}, got {governor.total_tokens}"
        )

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_breach_detection(self):
        """Test that _ParallelUsageGovernor correctly detects breaches under high concurrency."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor with a low limit to trigger breach
        limits = UsageLimits(total_cost_usd_limit=0.15, total_tokens_limit=500)
        governor = ExecutorCore._ParallelUsageGovernor(limits)

        # Create a proper StepResult for testing
        step_result = StepResult(name="breach_test_step")

        # Define costs that will exceed the limit when combined
        # Each operation adds $0.03, total will be $0.30 which exceeds $0.15
        operation_costs = [0.03] * 10  # 10 operations of $0.03 each
        operation_tokens = [50] * 10   # 10 operations of 50 tokens each

        # Simulate concurrent usage updates
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                cost = operation_costs[i]
                tokens = operation_tokens[i]
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent usage updates
        results = await add_usage_concurrently()

        # Verify that at least one operation detected the breach
        assert any(results), "At least one operation should detect the breach"

        # Verify that the governor detected the breach
        assert governor.breached()
        assert governor.get_error() is not None
        assert "Cost limit of $0.15 exceeded" in str(governor.get_error())

        # Verify that the total cost is correct (should be $0.30)
        expected_total_cost = sum(operation_costs)
        assert governor.total_cost == expected_total_cost, (
            f"Expected total cost {expected_total_cost}, got {governor.total_cost}"
        )

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_mixed_operations(self):
        """Test _ParallelUsageGovernor with mixed operations of different sizes."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor with a moderate limit
        limits = UsageLimits(total_cost_usd_limit=1.0, total_tokens_limit=2000)
        governor = ExecutorCore._ParallelUsageGovernor(limits)

        # Create a proper StepResult for testing
        step_result = StepResult(name="mixed_test_step")

        # Define mixed operation costs (some small, some large)
        operation_costs = [0.001, 0.01, 0.1, 0.05, 0.02, 0.08, 0.03, 0.07, 0.04, 0.06]
        operation_tokens = [1, 10, 100, 50, 20, 80, 30, 70, 40, 60]

        # Expected total cost and tokens
        expected_total_cost = sum(operation_costs)
        expected_total_tokens = sum(operation_tokens)

        # Simulate concurrent usage updates with mixed operations
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                cost = operation_costs[i]
                tokens = operation_tokens[i]
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent usage updates
        results = await add_usage_concurrently()

        # Verify that no breach occurred
        assert not any(results), "No breach should occur with these costs"

        # Verify that the governor detected no breach
        assert not governor.breached()
        assert governor.get_error() is None

        # CRITICAL: Verify that the final aggregated cost is exactly the sum of individual costs
        assert governor.total_cost == expected_total_cost, (
            f"Expected total cost {expected_total_cost}, got {governor.total_cost}"
        )
        assert governor.total_tokens == expected_total_tokens, (
            f"Expected total tokens {expected_total_tokens}, got {governor.total_tokens}"
        )

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_rapid_succession(self):
        """Test _ParallelUsageGovernor with rapid succession of operations."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor
        limits = UsageLimits(total_cost_usd_limit=5.0, total_tokens_limit=5000)
        governor = ExecutorCore._ParallelUsageGovernor(limits)

        # Create a proper StepResult for testing
        step_result = StepResult(name="rapid_test_step")

        # Simulate rapid succession of operations
        async def rapid_usage_updates():
            tasks = []
            for i in range(15):  # More operations for stress testing
                cost = 0.01 * (i + 1)  # Increasing costs
                tokens = 10 * (i + 1)  # Increasing tokens
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run rapid usage updates
        results = await rapid_usage_updates()

        # Verify that no breach occurred
        assert not any(results), "No breach should occur with these costs"

        # Verify that the governor detected no breach
        assert not governor.breached()
        assert governor.get_error() is None

        # Calculate expected totals
        expected_total_cost = sum(0.01 * (i + 1) for i in range(15))
        expected_total_tokens = sum(10 * (i + 1) for i in range(15))

        # CRITICAL: Verify that the final aggregated cost is exactly the sum of individual costs
        assert governor.total_cost == expected_total_cost, (
            f"Expected total cost {expected_total_cost}, got {governor.total_cost}"
        )
        assert governor.total_tokens == expected_total_tokens, (
            f"Expected total tokens {expected_total_tokens}, got {governor.total_tokens}"
        )

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_token_limits(self):
        """Test _ParallelUsageGovernor with token limit breaches."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor with token limits
        limits = UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=150)
        governor = ExecutorCore._ParallelUsageGovernor(limits)

        # Create a proper StepResult for testing
        step_result = StepResult(name="token_test_step")

        # Define token-heavy operations that will exceed the token limit
        operation_costs = [0.01] * 10  # Small costs
        operation_tokens = [20] * 10   # 20 tokens each, total 200 exceeds 150

        # Simulate concurrent usage updates
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                cost = operation_costs[i]
                tokens = operation_tokens[i]
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent usage updates
        results = await add_usage_concurrently()

        # Verify that at least one operation detected the breach
        assert any(results), "At least one operation should detect the token breach"

        # Verify that the governor detected the breach
        assert governor.breached()
        assert governor.get_error() is not None
        assert "Token limit of 150 exceeded" in str(governor.get_error())

        # Verify that the total tokens is correct (should be 200)
        expected_total_tokens = sum(operation_tokens)
        assert governor.total_tokens == expected_total_tokens, (
            f"Expected total tokens {expected_total_tokens}, got {governor.total_tokens}"
        )

    @pytest.mark.asyncio
    async def test_stress_parallel_usage_governor_no_limits(self):
        """Test _ParallelUsageGovernor with no limits (edge case)."""
        from flujo.application.core.ultra_executor import ExecutorCore

        # Create a usage governor with no limits
        governor = ExecutorCore._ParallelUsageGovernor(None)

        # Create a proper StepResult for testing
        step_result = StepResult(name="no_limit_test_step")

        # Define operations
        operation_costs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        operation_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Expected total cost and tokens
        expected_total_cost = sum(operation_costs)
        expected_total_tokens = sum(operation_tokens)

        # Simulate concurrent usage updates
        async def add_usage_concurrently():
            tasks = []
            for i in range(10):
                cost = operation_costs[i]
                tokens = operation_tokens[i]
                task = governor.add_usage(cost, tokens, step_result)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent usage updates
        results = await add_usage_concurrently()

        # Verify that no breach occurred (no limits set)
        assert not any(results), "No breach should occur when no limits are set"

        # Verify that the governor detected no breach
        assert not governor.breached()
        assert governor.get_error() is None

        # CRITICAL: Verify that the final aggregated cost is exactly the sum of individual costs
        assert governor.total_cost == expected_total_cost, (
            f"Expected total cost {expected_total_cost}, got {governor.total_cost}"
        )
        assert governor.total_tokens == expected_total_tokens, (
            f"Expected total tokens {expected_total_tokens}, got {governor.total_tokens}"
        ) 