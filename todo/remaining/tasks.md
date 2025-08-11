# Extract ALL Step Execution Logic from ExecutorCore into Policy Classes

## Overview
This task list provides detailed instructions for completing the full refactoring of the monolithic ExecutorCore into a policy-injection architecture. The goal is to eliminate all interleaved logic and achieve true separation of concerns.

## Current State Analysis
- ✅ Policy injection architecture is set up
- ✅ Protocol definitions exist for all step executors
- ✅ Default implementations exist (but are incomplete)
- ❌ Monolithic handlers still contain execution logic
- ❌ 163 test failures indicate incomplete refactoring
- ❌ Interleaved loops and special cases still exist

## Task List

### Phase 1: Extract Simple Step Execution Logic

#### Task 1.1: Extract `_execute_simple_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultSimpleStepExecutor`

**Instructions**:
1. Move the entire `_execute_simple_step` method (lines 693-1203) from `ultra_executor.py` into `DefaultSimpleStepExecutor.execute()`
2. Extract helper functions:
   - `_unpack_agent_result` → `DefaultAgentResultUnpacker.unpack()`
   - `_detect_mock_objects` → `MockDetectionPolicy`
   - `_format_feedback` → `FeedbackFormatter`
3. Extract retry logic into `RetryPolicy`
4. Extract fallback logic into `FallbackPolicy`
5. Extract usage tracking into `UsageTrackingPolicy`
6. Extract cache logic into `CachePolicy`

**Dependencies**: Tasks 1.2-1.6 must be completed first

#### Task 1.2: Create RetryPolicy
**File**: `flujo/application/core/retry_policy.py`

**Instructions**:
1. Create `RetryPolicy` protocol with `execute_with_retries()` method
2. Create `DefaultRetryPolicy` implementation
3. Extract retry loop logic from `_execute_simple_step`
4. Handle different error types (ValidationError, PluginError, AgentError)
5. Support configurable max_retries and retry conditions
6. Integrate with telemetry for retry logging

#### Task 1.3: Create FallbackPolicy
**File**: `flujo/application/core/fallback_policy.py`

**Instructions**:
1. Create `FallbackPolicy` protocol with `execute_with_fallback()` method
2. Create `DefaultFallbackPolicy` implementation
3. Extract fallback chain tracking logic
4. Extract fallback loop detection
5. Extract fallback metrics accumulation
6. Handle fallback error formatting and metadata

#### Task 1.4: Create UsageTrackingPolicy
**File**: `flujo/application/core/usage_tracking_policy.py`

**Instructions**:
1. Create `UsageTrackingPolicy` protocol
2. Create `DefaultUsageTrackingPolicy` implementation
3. Extract usage meter integration
4. Extract cost and token tracking
5. Extract usage limit enforcement
6. Support usage snapshot and reporting

#### Task 1.5: Create CachePolicy
**File**: `flujo/application/core/cache_policy.py`

**Instructions**:
1. Create `CachePolicy` protocol with `get_cached_result()` and `cache_result()` methods
2. Create `DefaultCachePolicy` implementation
3. Extract cache key generation logic
4. Extract cache TTL handling
5. Extract cache hit/miss logging
6. Support cache invalidation

#### Task 1.6: Create MockDetectionPolicy
**File**: `flujo/application/core/mock_detection_policy.py`

**Instructions**:
1. Create `MockDetectionPolicy` protocol
2. Create `DefaultMockDetectionPolicy` implementation
3. Extract mock object detection logic
4. Support different mock types (Mock, MagicMock, AsyncMock)
5. Handle mock detection in nested structures
6. Provide clear error messages for mock detection

### Phase 2: Extract Agent Step Execution Logic

#### Task 2.1: Extract `_execute_agent_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultAgentStepExecutor`

**Instructions**:
1. Move the entire `_execute_agent_step` method from `ultra_executor.py` into `DefaultAgentStepExecutor.execute()`
2. Extract agent execution orchestration:
   - Processor pipeline integration
   - Agent runner integration
   - Plugin execution
   - Validator execution
3. Extract error handling for different failure types
4. Extract streaming support
5. Extract usage metrics extraction

#### Task 2.2: Create AgentExecutionOrchestrator
**File**: `flujo/application/core/agent_execution_orchestrator.py`

**Instructions**:
1. Create `AgentExecutionOrchestrator` protocol
2. Create `DefaultAgentExecutionOrchestrator` implementation
3. Extract the orchestration logic:
   - Apply prompt processors
   - Execute agent
   - Apply output processors
   - Execute plugins
   - Execute validators
4. Handle streaming vs non-streaming execution
5. Support different agent types and configurations

#### Task 2.3: Create PluginExecutionPolicy
**File**: `flujo/application/core/plugin_execution_policy.py`

**Instructions**:
1. Create `PluginExecutionPolicy` protocol
2. Create `DefaultPluginExecutionPolicy` implementation
3. Extract plugin execution logic from `_execute_plugins_with_redirects`
4. Handle plugin redirects and loops
5. Support plugin outcomes (success, failure, redirect, new_solution)
6. Integrate with timeout handling

#### Task 2.4: Create ValidatorExecutionPolicy
**File**: `flujo/application/core/validator_execution_policy.py`

**Instructions**:
1. Create `ValidatorExecutionPolicy` protocol
2. Create `DefaultValidatorExecutionPolicy` implementation
3. Extract validator execution logic from `_execute_validators`
4. Handle validation failures and retries
5. Support multiple validators per step
6. Integrate with timeout handling

### Phase 3: Extract Loop Step Execution Logic

#### Task 3.1: Extract `_execute_loop` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultLoopStepExecutor`

**Instructions**:
1. Move the entire `_execute_loop` method (lines 2576-2818) from `ultra_executor.py` into `DefaultLoopStepExecutor.execute()`
2. Extract loop iteration logic into `LoopIterationPolicy`
3. Extract loop exit condition logic into `LoopExitPolicy`
4. Extract loop context management into `LoopContextPolicy`
5. Extract loop metrics accumulation into `LoopMetricsPolicy`

#### Task 3.2: Create LoopIterationPolicy
**File**: `flujo/application/core/loop_iteration_policy.py`

**Instructions**:
1. Create `LoopIterationPolicy` protocol
2. Create `DefaultLoopIterationPolicy` implementation
3. Extract iteration execution logic
4. Handle iteration input/output mappers
5. Support different iteration strategies
6. Handle iteration failures and fallbacks

#### Task 3.3: Create LoopExitPolicy
**File**: `flujo/application/core/loop_exit_policy.py`

**Instructions**:
1. Create `LoopExitPolicy` protocol
2. Create `DefaultLoopExitPolicy` implementation
3. Extract exit condition evaluation
4. Handle max_loops enforcement
5. Support custom exit conditions
6. Handle exit condition errors

#### Task 3.4: Create LoopContextPolicy
**File**: `flujo/application/core/loop_context_policy.py`

**Instructions**:
1. Create `LoopContextPolicy` protocol
2. Create `DefaultLoopContextPolicy` implementation
3. Extract context isolation logic
4. Extract context accumulation logic
5. Handle context updates between iterations
6. Support context inheritance and merging

### Phase 4: Extract Parallel Step Execution Logic

#### Task 4.1: Extract `_handle_parallel_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultParallelStepExecutor`

**Instructions**:
1. Move the entire `_handle_parallel_step` method from `ultra_executor.py` into `DefaultParallelStepExecutor.execute()`
2. Extract parallel execution orchestration into `ParallelExecutionOrchestrator`
3. Extract branch execution logic into `BranchExecutionPolicy`
4. Extract parallel usage governance into `ParallelUsageGovernor`
5. Extract parallel result aggregation into `ParallelResultAggregator`

#### Task 4.2: Create ParallelExecutionOrchestrator
**File**: `flujo/application/core/parallel_execution_orchestrator.py`

**Instructions**:
1. Create `ParallelExecutionOrchestrator` protocol
2. Create `DefaultParallelExecutionOrchestrator` implementation
3. Extract parallel execution coordination
4. Handle concurrent branch execution
5. Support different concurrency strategies
6. Handle parallel execution failures

#### Task 4.3: Create BranchExecutionPolicy
**File**: `flujo/application/core/branch_execution_policy.py`

**Instructions**:
1. Create `BranchExecutionPolicy` protocol
2. Create `DefaultBranchExecutionPolicy` implementation
3. Extract individual branch execution logic
4. Handle branch context isolation
5. Support branch failure strategies
6. Handle branch result collection

#### Task 4.4: Create ParallelUsageGovernor
**File**: `flujo/application/core/parallel_usage_governor.py`

**Instructions**:
1. Create `ParallelUsageGovernor` protocol
2. Create `DefaultParallelUsageGovernor` implementation
3. Extract parallel usage tracking
4. Handle proactive cancellation
5. Support usage limit enforcement across branches
6. Handle usage breach detection and response

### Phase 5: Extract Conditional Step Execution Logic

#### Task 5.1: Extract `_handle_conditional_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultConditionalStepExecutor`

**Instructions**:
1. Move the entire `_handle_conditional_step` method from `ultra_executor.py` into `DefaultConditionalStepExecutor.execute()`
2. Extract condition evaluation logic into `ConditionEvaluationPolicy`
3. Extract branch selection logic into `BranchSelectionPolicy`
4. Extract branch execution logic into `ConditionalBranchExecutionPolicy`
5. Extract conditional result formatting into `ConditionalResultFormatter`

#### Task 5.2: Create ConditionEvaluationPolicy
**File**: `flujo/application/core/condition_evaluation_policy.py`

**Instructions**:
1. Create `ConditionEvaluationPolicy` protocol
2. Create `DefaultConditionEvaluationPolicy` implementation
3. Extract condition evaluation logic
4. Handle different condition types
5. Support condition error handling
6. Support condition caching

#### Task 5.3: Create BranchSelectionPolicy
**File**: `flujo/application/core/branch_selection_policy.py`

**Instructions**:
1. Create `BranchSelectionPolicy` protocol
2. Create `DefaultBranchSelectionPolicy` implementation
3. Extract branch selection logic
4. Handle default branch fallback
5. Support branch not found scenarios
6. Handle branch selection errors

### Phase 6: Extract Dynamic Router Step Execution Logic

#### Task 6.1: Extract `_handle_dynamic_router_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultDynamicRouterStepExecutor`

**Instructions**:
1. Move the entire `_handle_dynamic_router_step` method from `ultra_executor.py` into `DefaultDynamicRouterStepExecutor.execute()`
2. Extract router evaluation logic into `RouterEvaluationPolicy`
3. Extract dynamic branch creation into `DynamicBranchCreationPolicy`
4. Extract router result aggregation into `RouterResultAggregator`

#### Task 6.2: Create RouterEvaluationPolicy
**File**: `flujo/application/core/router_evaluation_policy.py`

**Instructions**:
1. Create `RouterEvaluationPolicy` protocol
2. Create `DefaultRouterEvaluationPolicy` implementation
3. Extract router evaluation logic
4. Handle router function execution
5. Support router error handling
6. Support router result validation

### Phase 7: Extract HITL Step Execution Logic

#### Task 7.1: Extract `_handle_hitl_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultHitlStepExecutor`

**Instructions**:
1. Move the entire `_handle_hitl_step` method from `ultra_executor.py` into `DefaultHitlStepExecutor.execute()`
2. Extract HITL interaction logic into `HitlInteractionPolicy`
3. Extract HITL state management into `HitlStatePolicy`
4. Extract HITL command execution into `HitlCommandExecutionPolicy`

#### Task 7.2: Create HitlInteractionPolicy
**File**: `flujo/application/core/hitl_interaction_policy.py`

**Instructions**:
1. Create `HitlInteractionPolicy` protocol
2. Create `DefaultHitlInteractionPolicy` implementation
3. Extract human interaction logic
4. Handle interaction state management
5. Support different interaction types
6. Handle interaction timeouts

### Phase 8: Extract Cache Step Execution Logic

#### Task 8.1: Extract `_handle_cache_step` Logic
**File**: `flujo/application/core/step_policies.py`
**Target**: `DefaultCacheStepExecutor`

**Instructions**:
1. Move the entire `_handle_cache_step` method from `ultra_executor.py` into `DefaultCacheStepExecutor.execute()`
2. Extract cache key generation into `CacheKeyGenerationPolicy`
3. Extract cache hit/miss handling into `CacheHitMissPolicy`
4. Extract cache result formatting into `CacheResultFormatter`

#### Task 8.2: Create CacheKeyGenerationPolicy
**File**: `flujo/application/core/cache_key_generation_policy.py`

**Instructions**:
1. Create `CacheKeyGenerationPolicy` protocol
2. Create `DefaultCacheKeyGenerationPolicy` implementation
3. Extract cache key generation logic
4. Support different key generation strategies
5. Handle key collision resolution
6. Support cache key customization

### Phase 9: Extract Context Management Logic

#### Task 9.1: Extract Context Management Methods
**File**: `flujo/application/core/context_management_policy.py`

**Instructions**:
1. Create `ContextManagementPolicy` protocol
2. Create `DefaultContextManagementPolicy` implementation
3. Extract `_isolate_context` method
4. Extract `_merge_context_updates` method
5. Extract `_accumulate_loop_context` method
6. Extract `_update_context_state` method
7. Extract `_preserve_branch_modifications` method

#### Task 9.2: Create ContextIsolationPolicy
**File**: `flujo/application/core/context_isolation_policy.py`

**Instructions**:
1. Create `ContextIsolationPolicy` protocol
2. Create `DefaultContextIsolationPolicy` implementation
3. Extract context isolation logic
4. Handle deep copy vs shallow copy strategies
5. Support different isolation levels
6. Handle isolation errors

#### Task 9.3: Create ContextMergingPolicy
**File**: `flujo/application/core/context_merging_policy.py`

**Instructions**:
1. Create `ContextMergingPolicy` protocol
2. Create `DefaultContextMergingPolicy` implementation
3. Extract context merging logic
4. Handle merge conflicts
5. Support different merge strategies
6. Handle merge errors

### Phase 10: Extract Pipeline Execution Logic

#### Task 10.1: Extract `_execute_pipeline` Logic
**File**: `flujo/application/core/pipeline_execution_policy.py`

**Instructions**:
1. Create `PipelineExecutionPolicy` protocol
2. Create `DefaultPipelineExecutionPolicy` implementation
3. Extract the entire `_execute_pipeline` method
4. Extract step-by-step execution logic
5. Extract pipeline result aggregation
6. Handle pipeline failures and rollback

#### Task 10.2: Create StepSequentialExecutionPolicy
**File**: `flujo/application/core/step_sequential_execution_policy.py`

**Instructions**:
1. Create `StepSequentialExecutionPolicy` protocol
2. Create `DefaultStepSequentialExecutionPolicy` implementation
3. Extract sequential step execution logic
4. Handle step-to-step data flow
5. Support step failure handling
6. Handle step result propagation

### Phase 11: Extract Utility Methods

#### Task 11.1: Extract Helper Methods
**File**: `flujo/application/core/execution_utilities.py`

**Instructions**:
1. Create utility classes for common operations:
   - `ExecutionUtilities` for general execution helpers
   - `StepValidationUtilities` for step validation
   - `ResultFormattingUtilities` for result formatting
2. Extract `_safe_step_name` method
3. Extract `_format_feedback` method
4. Extract `_default_set_final_context` method
5. Extract `_is_complex_step` method

#### Task 11.2: Create ErrorHandlingPolicy
**File**: `flujo/application/core/error_handling_policy.py`

**Instructions**:
1. Create `ErrorHandlingPolicy` protocol
2. Create `DefaultErrorHandlingPolicy` implementation
3. Extract error classification logic
4. Handle different error types
5. Support error recovery strategies
6. Handle error reporting and logging

### Phase 12: Update ExecutorCore to Use Policies

#### Task 12.1: Update ExecutorCore Constructor
**File**: `flujo/application/core/ultra_executor.py`

**Instructions**:
1. Add all new policy dependencies to `__init__` method
2. Create default instances for all policies
3. Remove old monolithic method implementations
4. Update method signatures to use policy injection
5. Ensure backward compatibility

#### Task 12.2: Update Handler Methods
**File**: `flujo/application/core/ultra_executor.py`

**Instructions**:
1. Update `_handle_loop_step` to use `LoopStepExecutor`
2. Update `_handle_parallel_step` to use `ParallelStepExecutor`
3. Update `_handle_conditional_step` to use `ConditionalStepExecutor`
4. Update `_handle_dynamic_router_step` to use `DynamicRouterStepExecutor`
5. Update `_handle_hitl_step` to use `HitlStepExecutor`
6. Update `_handle_cache_step` to use `CacheStepExecutor`
7. Update `_execute_agent_step` to use `AgentStepExecutor`
8. Update `_execute_simple_step` to use `SimpleStepExecutor`

#### Task 12.3: Remove Monolithic Methods
**File**: `flujo/application/core/ultra_executor.py`

**Instructions**:
1. Remove `_execute_simple_step` method (moved to policy)
2. Remove `_execute_agent_step` method (moved to policy)
3. Remove `_execute_loop` method (moved to policy)
4. Remove `_execute_pipeline` method (moved to policy)
5. Remove `_ParallelUsageGovernor` class (moved to policy)
6. Remove context management methods (moved to policies)
7. Remove utility methods (moved to utilities)

### Phase 13: Update Policy Implementations

#### Task 13.1: Complete DefaultSimpleStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with all required policies:
   - RetryPolicy
   - FallbackPolicy
   - UsageTrackingPolicy
   - CachePolicy
   - MockDetectionPolicy
3. Ensure proper error handling and logging
4. Maintain backward compatibility

#### Task 13.2: Complete DefaultAgentStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with AgentExecutionOrchestrator
3. Integrate with PluginExecutionPolicy
4. Integrate with ValidatorExecutionPolicy
5. Ensure proper streaming support
6. Maintain backward compatibility

#### Task 13.3: Complete DefaultLoopStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with LoopIterationPolicy
3. Integrate with LoopExitPolicy
4. Integrate with LoopContextPolicy
5. Integrate with LoopMetricsPolicy
6. Ensure proper fallback support in loops

#### Task 13.4: Complete DefaultParallelStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with ParallelExecutionOrchestrator
3. Integrate with BranchExecutionPolicy
4. Integrate with ParallelUsageGovernor
5. Integrate with ParallelResultAggregator
6. Ensure proper concurrency control

#### Task 13.5: Complete DefaultConditionalStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with ConditionEvaluationPolicy
3. Integrate with BranchSelectionPolicy
4. Integrate with ConditionalBranchExecutionPolicy
5. Integrate with ConditionalResultFormatter
6. Ensure proper branch execution

#### Task 13.6: Complete DefaultDynamicRouterStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with RouterEvaluationPolicy
3. Integrate with DynamicBranchCreationPolicy
4. Integrate with RouterResultAggregator
5. Ensure proper dynamic routing

#### Task 13.7: Complete DefaultHitlStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with HitlInteractionPolicy
3. Integrate with HitlStatePolicy
4. Integrate with HitlCommandExecutionPolicy
5. Ensure proper human interaction handling

#### Task 13.8: Complete DefaultCacheStepExecutor
**File**: `flujo/application/core/step_policies.py`

**Instructions**:
1. Implement the complete `execute()` method using extracted logic
2. Integrate with CacheKeyGenerationPolicy
3. Integrate with CacheHitMissPolicy
4. Integrate with CacheResultFormatter
5. Ensure proper cache hit/miss handling

### Phase 14: Testing and Validation

#### Task 14.1: Fix Test Failures
**Instructions**:
1. Run `make test-fast` to identify current failures
2. Fix each test failure systematically
3. Ensure all extracted policies work correctly
4. Verify backward compatibility
5. Test all step types with new architecture

#### Task 14.2: Create Policy Tests
**File**: `tests/application/core/test_step_policies.py`

**Instructions**:
1. Create comprehensive tests for each policy
2. Test policy integration and composition
3. Test error handling in policies
4. Test performance characteristics
5. Test backward compatibility

#### Task 14.3: Create Integration Tests
**File**: `tests/integration/test_policy_integration.py`

**Instructions**:
1. Test complete policy integration
2. Test end-to-end execution flows
3. Test complex step combinations
4. Test error scenarios
5. Test performance under load

### Phase 15: Documentation and Cleanup

#### Task 15.1: Update Documentation
**Instructions**:
1. Update architecture documentation
2. Document all new policies
3. Update API documentation
4. Create migration guide
5. Update examples

#### Task 15.2: Cleanup Legacy Code
**Instructions**:
1. Remove unused imports
2. Remove dead code
3. Clean up comments
4. Update type hints
5. Ensure code quality standards

## Success Criteria

1. **Zero Test Failures**: All 163 current test failures must be resolved
2. **Complete Separation**: No monolithic methods remain in ExecutorCore
3. **Policy Injection**: All step execution uses injected policies
4. **Single Responsibility**: Each policy has a single, clear responsibility
5. **Backward Compatibility**: All existing APIs continue to work
6. **Performance**: No performance regression from refactoring
7. **Maintainability**: Code is easier to understand and modify

## Dependencies

- Tasks must be completed in order within each phase
- Phase 1 must be completed before Phase 2
- All phases must be completed before testing
- Testing must pass before documentation updates

## Estimated Effort

- **Phase 1-8**: 2-3 days each (extraction work)
- **Phase 9-11**: 1-2 days each (utility extraction)
- **Phase 12-13**: 2-3 days (integration work)
- **Phase 14**: 3-5 days (testing and fixing)
- **Phase 15**: 1-2 days (documentation)
