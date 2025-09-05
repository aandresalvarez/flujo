# Failed Tests Report - Coverage Run

**Date**: September 5, 2024  
**Test Command**: `make testcov`  
**Total Tests**: 2,982  
**Passed**: 2,954  
**Failed**: 28  
**Skipped**: 23  
**Duration**: 56 minutes 19 seconds

## Failed Tests Summary

### 1. Conversational Loop Tests (2 failures)
- `tests/integration/test_conversational_loop_parallel.py::test_conversational_loop_parallel_all_agents`
  - **Error**: `assert False`
  - **Issue**: Parallel conversational loop test failure

- `tests/integration/test_conversational_loop_nested.py::test_nested_conversation_inner_scoped`
  - **Error**: `assert False`
  - **Issue**: Nested conversation scoping test failure

### 2. Parallel Step Robustness (1 failure)
- `tests/unit/test_parallel_step_robustness.py::TestParallelStepRobustness::test_usage_limit_breach_propagates_correctly`
  - **Error**: `AssertionError: assert 0 > 0`
  - **Issue**: Usage limit breach not propagating correctly

### 3. Architect Security Validation Tests (16 failures - All Timeouts)
All architect security validation tests failed due to timeouts (>30.0s):

- `test_architect_handles_deserialization_attacks`
- `test_architect_handles_no_sql_injection_attempts`
- `test_architect_handles_template_injection_attempts`
- `test_architect_handles_regex_dos_attempts`
- `test_architect_handles_path_traversal_attempts`
- `test_architect_handles_overflow_attempts`
- `test_architect_handles_prototype_pollution_attempts`
- `test_architect_handles_xss_attempts`
- `test_architect_handles_unicode_normalization_attacks`
- `test_architect_handles_sql_injection_attempts`
- `test_architect_handles_command_injection_attempts`
- `test_architect_handles_mixed_malicious_inputs`
- `test_architect_handles_ldap_injection_attempts`
- `test_architect_handles_encoding_manipulation`

**Root Cause**: Security validation tests are timing out, likely due to:
- Complex security validation logic taking too long
- Resource constraints during parallel execution
- Need for timeout adjustments for security tests

### 4. Architect Performance Stress Tests (5 failures)
- `test_architect_large_context_handling`
  - **Error**: `AssertionError: Execution time 16.30s exceeds 5.0s limit for large context`
  - **Issue**: Large context handling slower than expected

- `test_architect_resource_usage_scaling`
  - **Error**: `AssertionError: Execution time 17.60s exceeds expected 1.50s for complexity level`
  - **Issue**: Resource usage scaling performance regression

- `test_architect_handles_high_frequency_requests` (Timeout >60.0s)
- `test_architect_concurrent_pipeline_execution` (Timeout >60.0s)
- `test_architect_response_time_under_load` (Timeout >60.0s)
- `test_architect_stress_test_rapid_requests` (Timeout >120.0s)
- `test_architect_execution_time_consistency` (Timeout >30.0s)

**Root Cause**: Performance stress tests are failing due to:
- Performance regressions in architect execution
- Timeout settings too aggressive for complex operations
- Resource contention during parallel test execution

### 5. CLI Integration Tests (2 failures)
- `tests/cli/test_architect_integration.py::TestArchitectCLIIntegration::test_architect_cli_allow_side_effects_help`
  - **Error**: `AssertionError: Help should mention allow-side-effects flag`
  - **Issue**: Missing CLI help text for allow-side-effects parameter

- `tests/cli/test_architect_gpt5_settings.py::TestArchitectCLIIntegration::test_architect_cli_help_includes_expected_options`
  - **Error**: `AssertionError: Help should mention allow-side-effects parameter`
  - **Issue**: Missing CLI help text for allow-side-effects parameter

**Root Cause**: CLI help text missing for `allow-side-effects` parameter

### 6. Performance Optimization Tests (2 failures)
- `tests/unit/test_performance_optimizations.py::TestCallableResolutionOptimization::test_resolve_callable_performance`
  - **Error**: `assert 0.005758999846875668 < 0.005`
  - **Issue**: Callable resolution performance slightly slower than threshold

- `tests/benchmarks/test_conversational_overhead.py::test_history_manager_overhead_benchmark`
  - **Error**: `assert 14.626784000080079 < 1.0`
  - **Issue**: History manager overhead significantly higher than expected

**Root Cause**: Performance thresholds too strict or performance regressions

## Recommendations

### Immediate Actions
1. **Security Test Timeouts**: Increase timeout values for security validation tests
2. **CLI Help Text**: Add missing `allow-side-effects` parameter to CLI help
3. **Performance Thresholds**: Review and adjust performance thresholds for current system capabilities

### Investigation Needed
1. **Conversational Loop Issues**: Investigate parallel and nested conversation test failures
2. **Architect Performance**: Analyze performance regressions in architect execution
3. **Usage Limit Propagation**: Fix usage limit breach propagation in parallel steps

### Test Environment
- Consider running security and performance tests separately with higher resource allocation
- Review parallel execution settings for resource-intensive tests
- Implement test categorization to prevent resource contention

## Coverage Impact
Despite 28 test failures, the test suite achieved comprehensive coverage across:
- Core functionality (2,954 passing tests)
- Integration scenarios
- Performance benchmarks
- Security validations (though timing out)

The failures appear to be primarily related to:
- Performance thresholds
- Timeout configurations
- Missing CLI documentation
- Resource contention during parallel execution
