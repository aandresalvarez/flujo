# Requirements Document

## Introduction

This document outlines the requirements for fixing the 6 remaining failing tests in the Flujo test suite. The failing tests are related to parallel step cancellation timing and usage governance (loop step counting and cost calculations). These tests need to be fixed without modifying the test code itself, only the implementation.

## Requirements

### Requirement 1: Fix Parallel Step Cancellation Timing

**User Story:** As a developer running the test suite, I want the parallel step cancellation tests to pass consistently, so that the build pipeline remains stable.

#### Acceptance Criteria

1. WHEN `test_proactive_cancellation_with_multiple_branches` runs THEN the execution time SHALL be less than 0.3 seconds
2. WHEN `test_proactive_cancellation_token_limits` runs THEN the execution time SHALL be less than 0.4 seconds
3. WHEN proactive cancellation is triggered THEN it SHALL cancel parallel branches efficiently
4. WHEN token limits are reached THEN cancellation SHALL occur within the expected timeframe

### Requirement 2: Fix Loop Step Attempt Counting

**User Story:** As a developer using loop steps with usage governance, I want the attempt counting to be accurate, so that usage limits are enforced correctly.

#### Acceptance Criteria

1. WHEN `test_governor_with_loop_step` runs THEN the loop step SHALL report exactly 3 attempts
2. WHEN `test_governor_halts_loop_step_mid_iteration` runs THEN the loop step SHALL report exactly 3 attempts
3. WHEN a loop step reaches its exit condition THEN the attempt count SHALL match the number of iterations executed
4. WHEN a loop step is halted mid-iteration THEN the attempt count SHALL reflect the actual iterations completed

### Requirement 3: Fix Usage Cost Calculations

**User Story:** As a developer using nested parallel loops with usage limits, I want cost calculations to be accurate, so that usage governance works correctly.

#### Acceptance Criteria

1. WHEN `test_governor_loop_with_nested_parallel_limit` runs THEN the total cost SHALL be approximately 0.6
2. WHEN `test_usage_limits_enforcement_loop_steps` runs THEN the total cost SHALL be approximately 1.2
3. WHEN nested parallel steps execute within loops THEN costs SHALL be calculated correctly
4. WHEN usage limits are enforced on loop steps THEN the cost accumulation SHALL be accurate
