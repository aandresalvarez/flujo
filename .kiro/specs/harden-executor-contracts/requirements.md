# Requirements Document

## Introduction

This feature focuses on hardening the internal contracts of the ExecutorCore and its related components, making them statically verifiable and more robust. This implements the core principles from the "Philosophical Shift" document by moving from runtime assumptions to explicit, type-checked contracts, making plugin failures explicit, and simplifying complex step detection through object-oriented design.

## Requirements

### Requirement 1: Harden ParallelStep Type Contracts

**User Story:** As a developer working with ParallelStep components, I want the type contracts to be statically verifiable, so that I can catch type errors at development time rather than runtime.

#### Acceptance Criteria

1. WHEN defining a ParallelStep merge_strategy THEN the system SHALL enforce that the context parameter has a scratchpad attribute through static typing
2. WHEN ExecutorCore handles a ParallelStep THEN the context parameter SHALL be bounded by a ContextWithScratchpad protocol
3. WHEN running mypy on the codebase THEN type violations for ParallelStep context usage SHALL be detected
4. WHEN a context without scratchpad is passed to ParallelStep operations THEN mypy SHALL report a type error

### Requirement 2: Make Plugin Failures Explicit

**User Story:** As a developer using plugins in my pipeline steps, I want plugin failures to be explicit and cause step failure, so that no failures are silently ignored.

#### Acceptance Criteria

1. WHEN a plugin raises an exception during execution THEN the DefaultPluginRunner SHALL re-raise the exception
2. WHEN a plugin returns a PluginOutcome with success=False THEN the system SHALL raise a ValueError with the feedback message
3. WHEN plugin execution fails THEN the step execution SHALL fail with the plugin's error information
4. WHEN analyzing step failures THEN plugin failure details SHALL be available in telemetry logs

### Requirement 3: Simplify Complex Step Detection

**User Story:** As a developer extending Flujo with new step types, I want complex step detection to be object-oriented and extensible, so that I can add new complex step types without modifying ExecutorCore dispatch logic.

#### Acceptance Criteria

1. WHEN defining a new step type THEN the system SHALL use an is_complex property to determine if it needs complex handling
2. WHEN ExecutorCore checks if a step is complex THEN it SHALL use the step's is_complex property
3. WHEN adding new complex step types THEN no modifications to ExecutorCore._is_complex_step SHALL be required
4. WHEN base Step instances are created THEN they SHALL have is_complex=False by default
5. WHEN complex step subclasses are created THEN they SHALL override is_complex to return True

### Requirement 4: Maintain Backward Compatibility

**User Story:** As a developer with existing Flujo pipelines, I want the contract hardening changes to maintain full backward compatibility, so that my existing code continues to work without modifications.

#### Acceptance Criteria

1. WHEN upgrading to hardened contracts THEN all existing functionality SHALL remain unchanged
2. WHEN running the existing test suite THEN 100% of tests SHALL pass
3. WHEN using existing APIs THEN the system SHALL maintain identical behavior
4. WHEN handling existing step types THEN the system SHALL process them correctly with the new contract system

### Requirement 5: Ensure Type Safety Verification

**User Story:** As a developer, I want static analysis to verify the new type contracts, so that I can be confident the hardening is effective.

#### Acceptance Criteria

1. WHEN running mypy on the codebase THEN no new type errors SHALL be introduced
2. WHEN creating test cases with invalid context types THEN mypy SHALL detect and report type violations
3. WHEN using the ContextWithScratchpad protocol THEN type checking SHALL enforce the scratchpad attribute requirement
4. WHEN bounded TypeVars are used THEN mypy SHALL verify type constraints are satisfied
