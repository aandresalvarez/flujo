# Implementation Plan

- [x] 1. Restore ExecutorCore with proper step handling and failure domain separation
  - ✅ Implement consistent step routing in the main execute() method
  - ✅ Create proper _execute_agent_step with separated try-catch blocks for validators, plugins, and agents
  - ✅ Implement missing _execute_simple_step method with comprehensive fallback support
  - ✅ Fix _is_complex_step logic to properly categorize steps with plugins and fallbacks
  - ✅ Ensure all step handlers follow the recursive execution model consistently
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Fix parallel step feedback handling and error propagation
  - Fix parallel step to properly set feedback when branches fail
  - Ensure parallel step result feedback is not None when failures occur
  - Fix error message formatting in parallel step execution
  - Fix branch failure propagation to include proper error details
  - Ensure parallel step success/failure determination works correctly
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 3. Fix loop step context propagation and iteration management
  - Fix context accumulation across loop iterations in _handle_loop_step
  - Fix iteration counting logic to be accurate and consistent
  - Fix exit condition evaluation that works even when iterations fail
  - Fix max iterations logic to stop at the correct count
  - Ensure iteration input/output mappers are called at the correct times
  - Fix loop step attempt counting for usage governance integration
  - _Requirements: 2.1, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 4. Fix conditional step execution and context handling
  - ✅ Fix branch execution logic in _handle_conditional_step
  - ✅ Fix context isolation for branch execution using deep copy
  - ✅ Fix context capture and merging logic to preserve branch modifications
  - ✅ Fix mapper context handling to call mappers on main context, not branch context
  - ✅ Ensure conditional step error handling properly propagates branch failures
  - ✅ Fix feedback messages to accurately reflect what actually happened
  - _Requirements: 2.3, 1.1, 1.3_

- [x] 5. Fix agent step feedback and validation handling
  - ✅ Fix _execute_agent_step to ensure feedback is never None when step fails
  - ✅ Fix validation error handling to preserve proper feedback messages
  - ✅ Fix plugin error handling to maintain error context
  - ✅ Ensure successful steps have empty string feedback, not None
  - ✅ Fix retry logic to properly accumulate feedback across attempts
  - _Requirements: 1.1, 1.2, 1.3, 8.1, 8.2_

- [x] 6. Fix serialization system edge cases
  - ✅ Fix AgentResponse serialization with proper field extraction
  - ✅ Add better support for Mock objects in test scenarios
  - ✅ Fix custom object serialization with circular reference handling
  - ✅ Add helpful error messages for unknown types with custom serializer suggestions
  - ✅ Fix Enum serialization edge cases
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7. Fix usage governance and cost tracking integration
  - ✅ Fix order of operations in ExecutionManager.execute_steps
  - ✅ Fix loop step exception handling to properly re-raise UsageLimitExceededError
  - ✅ Fix step history population to add results even when exceptions occur
  - ✅ Fix cost limit error message formatting to be consistent
  - ✅ Fix parallel step cost aggregation to avoid double-counting
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 8. Fix HITL step integration and context management
  - ✅ Fix _handle_hitl_step method signature consistency
  - ✅ Fix context status update before raising PausedException
  - ✅ Fix HITL step message formatting consistency
  - ✅ Ensure HITL steps integrate properly with telemetry and usage limits
  - ✅ Fix HITL step error handling consistency
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Fix context management system integration
  - ✅ Implement proper context isolation using deep copy for branch execution
  - ✅ Fix context merging logic using safe_merge_context_updates
  - ✅ Fix context accumulation for loop iterations
  - ✅ Ensure context state transitions are handled correctly
  - ✅ Fix context updates to preserve modifications from successful branches
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - All previous tasks remain unbroken and verified by the test suite.

- [x] 10. Fix fallback system edge cases and error handling
  - ✅ Fix fallback cost accumulation to not double-count costs
  - ✅ Fix fallback feedback formatting to include proper error context
  - ✅ Fix fallback with None and empty string feedback handling
  - ✅ Fix fallback retry scenarios to have correct attempt counts
  - ✅ Fix fallback metadata to preserve original error information
  - _Requirements: 1.4, 8.5_

- [x] 11. Fix pipeline runner integration and retry logic
  - ✅ Fix pipeline runner retry logic to respect max_retries parameter
  - ✅ Fix feedback enrichment to properly accumulate across retries
  - ✅ Fix conditional redirection logic to trigger appropriate agent calls
  - ✅ Fix on_failure callback integration to be called when expected
  - ✅ Fix agent result unpacking to handle wrapped results correctly
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 12. Fix mock output handling and type error detection
  - ✅ Fix mock output detection to properly identify Mock objects
  - ✅ Fix error message patterns to match actual error messages from Mock operations
  - ✅ Ensure pipeline stops correctly when Mock objects are detected
  - ✅ Fix type error handling for Mock object operations
  - _Requirements: 3.5, 10.4_

- [ ] 13. Fix nested DSL construct error propagation
  - Fix deeply nested error propagation to properly bubble up failures
  - Ensure loop steps with failing bodies handle exit conditions correctly
  - Fix error handling in complex nested scenarios
  - Ensure error messages accurately reflect the source of failures in nested constructs
  - _Requirements: 1.3, 6.2_

- [ ] 14. Fix explicit cost integration and image cost tracking
  - Fix MockImageResult and MockResponseWithBoth to have proper output attributes
  - Fix AgentResponse serialization for image cost tracking scenarios
  - Ensure explicit cost protocol works correctly with usage limits
  - Fix cost tracking for image generation scenarios
  - Fix explicit cost handling with zero, negative, and None values
  - _Requirements: 3.1, 4.3_

- [ ] 15. Fix performance and persistence issues
  - Optimize persistence operations to reduce overhead below 35% limit
  - Fix default backend configuration logic to use correct backends
  - Fix proper error handling for persistence operations
  - Optimize large context persistence to maintain acceptable performance
  - Fix backend configuration adaptation when settings change
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 16. Validate all fixes work together and run comprehensive test suite
  - Run make test-fast-verbose to verify all architectural fixes are working
  - Ensure no regressions were introduced by the architectural changes
  - Validate that performance remains acceptable after all fixes
  - Confirm that failing tests now pass
  - Verify that the total test count remains consistent
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5_