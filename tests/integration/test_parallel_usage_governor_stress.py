"""Legacy governor stress tests - REMOVED

This file tested the _ParallelUsageGovernor which has been removed in favor of
the pure quota system. The tests are no longer applicable since:

1. There is no governor class anymore
2. Budgeting is now handled via proactive quota reservations
3. Parallel execution uses Quota.split() and deterministic limit checks

See the new quota system tests for current parallel budgeting behavior.
"""
