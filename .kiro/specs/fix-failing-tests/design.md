# Design Document

## Overview

This document outlines the design for fixing the 6 remaining failing tests in the Flujo test suite. The failures are in three categories:

1. **Parallel Step Cancellation Timing** (2 tests) - Tests expect proactive cancellation to complete within specific time thresholds
2. **Loop Step Attempt Counting** (2 tests) - Tests expect loop steps to report specific attempt counts when usage limits are exceeded
3. **Usage Cost Calculations** (2 tests) - Tests expect specific cost calculations for nested parallel loops and usage enforcement

## Architecture

The fixes will target three main components:

1. **Parallel Step Execution** - Enhance proactive cancellation mechanism
2. **Loop Step Execution** - Fix attempt counting logic
3. **Usage Governor** - Correct cost calculation and limit enforcement

## Components and Interfaces

### 1. Parallel Step Proactive Cancellation

**Component**: `flujo/application/core/ultra_executor.py` (or related parallel execution logic)

**Current Issue**:
- `test_proactive_cancellation_with_multiple_branches` expects execution < 0.3s but takes ~0.52s
- `test_proactive_cancellation_token_limits` expects execution < 0.4s but takes ~0.55s

**Root Cause**: The proactive cancellation mechanism is not efficiently cancelling parallel branches when usage limits are breached.

**Design Solution**:
- Improve the `breach_event` handling in parallel step execution
- Ensure that when one branch breaches limits, other branches are cancelled immediately
- Optimize the cancellation propagation to reduce delay

### 2. Loop Step Attempt Counting

**Component**: Loop step execution logic

**Current Issue**:
- `test_governor_with_loop_step` expects 3 attempts but gets 4
- `test_governor_halts_loop_step_mid_iteration` expects 3 attempts but gets 5

**Root Cause**: The attempt counting logic is including additional iterations beyond what the tests expect.

**Design Solution**:
- Review the loop iteration counting logic
- Ensure attempts are counted correctly when usage limits cause early termination
- The tests expect exactly 3 attempts when cost limit of $0.25 is breached with $0.1 per iteration

### 3. Usage Cost Calculations

**Component**: Usage governor and cost accumulation logic

**Current Issues**:
- `test_governor_loop_with_nested_parallel_limit` expects ~$0.6 but gets ~$2.0
- `test_usage_limits_enforcement_loop_steps` expects ~$1.2 but gets ~$3.0

**Root Cause**: Cost calculations are being multiplied or accumulated incorrectly, possibly double-counting costs in nested scenarios.

**Design Solution**:
- Review cost accumulation in nested parallel/loop scenarios
- Ensure costs are not double-counted when steps are executed within loops
- Verify that usage limit enforcement stops execution at the correct cost threshold

## Data Models

No changes to data models are required. The fixes will focus on execution logic and cost calculation algorithms.

## Error Handling

The error handling for `UsageLimitExceededError` should remain unchanged. The fixes will ensure that:
- Errors are raised at the correct cost/token thresholds
- The result objects contain accurate cost and attempt information
- Timing constraints are met for proactive cancellation

## Testing Strategy

The testing strategy will be:
1. **No test modifications** - All fixes must be made to implementation code only
2. **Preserve existing behavior** - Ensure other tests continue to pass
3. **Focus on timing and counting accuracy** - Address the specific numerical expectations in the failing tests

### Test Analysis

#### Parallel Cancellation Tests
- Both tests use `CostlyAgent` with specific delays and costs
- Tests measure execution time and expect fast cancellation
- The `breach_event` parameter should enable early termination

#### Loop Attempt Tests
- Both tests use `UsageLimits(total_cost_usd_limit=0.25)`
- Both use agents with `cost=0.1` per iteration
- Mathematical expectation: 3 iterations × $0.1 = $0.3 (exceeds $0.25 limit)
- Tests expect exactly 3 attempts when limit is breached

#### Cost Calculation Tests
- `test_governor_loop_with_nested_parallel_limit`: Expects $0.6 total cost
- `test_usage_limits_enforcement_loop_steps`: Expects $1.2 total cost
- Both involve loop steps with specific cost-per-iteration agents
- Current implementation appears to be over-calculating costs

## Implementation Plan

The implementation will focus on three key areas:

1. **Optimize Parallel Cancellation**: Improve the speed of proactive cancellation when usage limits are breached
2. **Fix Loop Attempt Counting**: Ensure loop steps report accurate attempt counts
3. **Correct Cost Calculations**: Fix cost accumulation logic to match test expectations

Each fix will be implemented carefully to avoid breaking existing functionality while meeting the specific requirements of the failing tests.
