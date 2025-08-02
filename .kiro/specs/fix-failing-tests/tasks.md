# Implementation Plan

- [x] 1. Fix parallel step proactive cancellation timing
  - Analyze current parallel step execution logic to identify cancellation delays
  - Optimize breach_event handling to ensure immediate cancellation of parallel branches
  - Ensure CostlyAgent respects breach_event for early termination
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Investigate and fix loop step attempt counting logic
  - Examine loop step execution to understand why attempts are over-counted
  - Identify where attempt counting occurs during usage limit enforcement
  - Correct the counting logic to match test expectations (3 attempts for $0.25 limit with $0.1 per iteration)
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 3. Debug and fix usage cost calculation issues
  - Analyze cost accumulation in nested parallel/loop scenarios
  - Identify where costs are being double-counted or over-calculated
  - Fix cost calculation logic for test_governor_loop_with_nested_parallel_limit (expect $0.6)
  - Fix cost calculation logic for test_usage_limits_enforcement_loop_steps (expect $1.2)
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4. Verify all fixes work together
  - Run the specific failing tests to confirm they now pass
  - Run the full test suite to ensure no regressions were introduced
  - Validate that timing constraints are consistently met
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2_