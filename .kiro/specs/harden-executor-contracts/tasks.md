# Implementation Plan

- [x] 1. Create type contract system
  - ✅ Created flujo/application/core/types.py with ContextWithScratchpad protocol and bounded TypeVar
  - ✅ Defined Protocol interface for contexts with scratchpad attribute
  - _Requirements: 1.1, 1.2_

- [x] 2. Update ParallelStep type contracts
  - [x] 2.1 Update ParallelStep class definition in flujo/domain/dsl/parallel.py
    - ✅ Imported TContext_w_Scratch from types module
    - ✅ Updated merge_strategy field to use bounded TypeVar
    - _Requirements: 1.1, 1.3_

  - [x] 2.2 Update ExecutorCore._handle_parallel_step method signature
    - ✅ Imported TContext_w_Scratch in flujo/application/core/ultra_executor.py
    - ✅ Updated method signature to use bounded TypeVar for context parameter
    - ✅ Updated parallel_step parameter type annotation
    - _Requirements: 1.2, 1.4_

- [x] 3. Implement explicit plugin failure handling
  - [x] 3.1 Modify DefaultPluginRunner.run_plugins method
    - ✅ Updated exception handling to re-raise plugin exceptions
    - ✅ Added telemetry logging before re-raising exceptions
    - ✅ Plugin failures now cause step failures
    - _Requirements: 2.1, 2.2, 2.4_

- [x] 4. Implement object-oriented complex step detection
  - [x] 4.1 Add is_complex property to base Step class
    - ✅ Added property to flujo/domain/dsl/step.py returning False by default
    - ✅ Documented the property purpose and usage
    - _Requirements: 3.4_

  - [x] 4.2 Override is_complex property in complex step subclasses
    - ✅ Updated LoopStep in flujo/domain/dsl/loop.py to return True
    - ✅ Updated ParallelStep in flujo/domain/dsl/parallel.py to return True
    - ✅ Updated ConditionalStep in flujo/domain/dsl/conditional.py to return True
    - ✅ Updated DynamicParallelRouterStep in flujo/domain/dsl/dynamic_router.py to return True
    - ✅ Updated CacheStep in flujo/steps/cache_step.py to return True
    - ✅ Updated HumanInTheLoopStep in flujo/domain/dsl/step.py to return True
    - _Requirements: 3.5_

  - [x] 4.3 Simplify ExecutorCore._is_complex_step implementation
    - ✅ Updated method in flujo/application/core/ultra_executor.py to use getattr approach
    - ✅ Replaced existing complex step type checking logic
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 5. Create static analysis tests
  - [x] 5.1 Create tests/static_analysis/test_contracts.py
    - ✅ Created test that defines context without scratchpad attribute
    - ✅ Created test that attempts to use invalid context with ParallelStep
    - ✅ Verified mypy catches type violations (test passes if mypy reports errors)
    - _Requirements: 5.2, 5.3_

- [ ] 6. Fix failing tests from FSD 10 implementation
  - [x] 6.1 Fix plugin failure test issues
    - Investigate and fix validation persistence tests that expect plugin failures to be stored
    - Fix hybrid validation tests that expect specific plugin error messages
    - _Requirements: 2.3, 4.1_

  - [x] 6.2     Fix step logic accounting tests
    - Fix test_multiple_retries_preserve_last_attempt_metrics feedback issue
    - Ensure retry feedback is properly preserved
    - _Requirements: 4.2_

  - [x] 6.3 Fix pipeline runner integration tests
    - Fix conditional redirection test call count issue
    - Fix failure handler and timeout detection tests
    - Fix agent result unpacking test
    - _Requirements: 4.3_

  - [x] 6.4 Fix executor core unit tests
    - Fix validation failure test feedback expectations
    - Fix usage limit exception propagation test
    - Fix max retries exceeded feedback test
    - _Requirements: 4.1, 4.2_

- [-] 7. Run regression testing and achieve 100% pass rate
  - [x] 7.1 Run full test suite with make test-fast
    - Execute complete test suite to verify 100% pass rate
    - Investigate and fix any remaining test failures
    - Ensure no regressions in existing functionality
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 7.2 Run mypy type checking
    - Execute mypy on entire codebase
    - Verify no new type errors are introduced
    - Confirm type contract enforcement is working
    - _Requirements: 5.1, 5.4_
