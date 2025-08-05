# Requirements Document

## Introduction

This document outlines the requirements for systematically restoring the Flujo test suite to a fully passing state. Currently, 282 out of 2087 tests are failing, indicating fundamental architectural issues that need to be addressed using first principles analysis based on the Flujo architecture described in `flujo.md`.

The failures span multiple architectural layers and indicate systemic issues in:
- Execution Core logic (ExecutorCore, step handling)
- Context management and propagation
- Serialization and type system
- Usage governance and cost tracking
- Parallel and loop step execution
- HITL (Human-in-the-Loop) integration
- Plugin and validation systems

## Requirements

### Requirement 1: Fix Core Execution Engine Issues

**User Story:** As a developer using Flujo, I want the core execution engine to properly handle all step types and execution patterns, so that pipelines execute reliably and predictably.

#### Acceptance Criteria

1. WHEN any step type is executed THEN the ExecutorCore SHALL handle it according to the Flujo architecture's recursive execution model
2. WHEN complex steps (loops, conditionals, parallel) are executed THEN they SHALL properly delegate to their respective handlers
3. WHEN step execution fails THEN error handling SHALL be consistent and informative
4. WHEN fallback logic is triggered THEN it SHALL execute without causing agent exhaustion
5. WHEN steps have plugins or validators THEN they SHALL be executed in the correct order without causing retries

### Requirement 2: Fix Context Management and Propagation

**User Story:** As a developer building stateful pipelines, I want context to be properly managed across all step types, so that state modifications are preserved and isolated correctly.

#### Acceptance Criteria

1. WHEN loop steps execute THEN context modifications from each iteration SHALL be properly accumulated
2. WHEN parallel steps execute THEN each branch SHALL receive isolated context and updates SHALL be merged correctly
3. WHEN conditional steps execute THEN branch context updates SHALL be preserved in the final result
4. WHEN HITL steps pause execution THEN context status SHALL be updated to 'paused' before raising PausedException
5. WHEN nested control flow steps execute THEN context isolation and merging SHALL work correctly at all levels

### Requirement 3: Fix Serialization and Type System

**User Story:** As a developer working with complex data types, I want all objects to be properly serializable, so that state persistence and debugging work correctly.

#### Acceptance Criteria

1. WHEN AgentResponse objects are serialized THEN they SHALL be converted to a serializable format
2. WHEN Enum types are serialized THEN they SHALL return their .value property
3. WHEN custom objects are serialized THEN circular references SHALL be handled gracefully
4. WHEN unknown types are encountered THEN serialization SHALL provide helpful error messages
5. WHEN Mock objects are used in tests THEN they SHALL be properly handled by the serialization system

### Requirement 4: Fix Usage Governance and Cost Tracking

**User Story:** As a developer implementing cost controls, I want usage limits to be enforced accurately and consistently, so that budget overruns are prevented.

#### Acceptance Criteria

1. WHEN usage limits are checked THEN they SHALL be evaluated after step costs are added to totals
2. WHEN loop steps execute with usage limits THEN attempt counting SHALL be accurate
3. WHEN parallel steps execute within loops THEN cost calculations SHALL not be double-counted
4. WHEN usage limits are exceeded THEN UsageLimitExceededError SHALL be raised with correct cost information
5. WHEN proactive cancellation is triggered THEN parallel branches SHALL be cancelled efficiently

### Requirement 5: Fix Parallel Step Execution

**User Story:** As a developer using parallel execution, I want parallel steps to handle context isolation, error propagation, and cancellation correctly, so that concurrent operations work reliably.

#### Acceptance Criteria

1. WHEN parallel steps execute THEN each branch SHALL receive properly isolated context
2. WHEN parallel branches fail THEN error handling SHALL determine overall step success/failure correctly
3. WHEN parallel steps are cancelled THEN cancellation SHALL propagate to all branches efficiently
4. WHEN parallel branches complete THEN context updates SHALL be merged using the correct strategy
5. WHEN parallel steps have mixed success/failure THEN the overall result SHALL reflect the correct state

### Requirement 6: Fix Loop Step Logic

**User Story:** As a developer using iterative processes, I want loop steps to handle iteration counting, exit conditions, and context updates correctly, so that loops behave predictably.

#### Acceptance Criteria

1. WHEN loop steps execute THEN iteration counting SHALL be accurate and consistent
2. WHEN exit conditions are evaluated THEN they SHALL work correctly even when individual iterations fail
3. WHEN max iterations are reached THEN loops SHALL stop at the correct count
4. WHEN loop bodies fail THEN error handling SHALL continue evaluating exit conditions appropriately
5. WHEN iteration mappers are used THEN they SHALL be called at the correct times with proper context

### Requirement 7: Fix HITL Step Integration

**User Story:** As a developer implementing human-in-the-loop workflows, I want HITL steps to integrate properly with the execution engine, so that human interaction points work seamlessly.

#### Acceptance Criteria

1. WHEN HITL steps are executed THEN method signatures SHALL be consistent across all handlers
2. WHEN HITL steps pause execution THEN messages SHALL be formatted correctly for different step types
3. WHEN HITL steps resume THEN context state SHALL be preserved accurately
4. WHEN HITL steps integrate with other components THEN they SHALL work with telemetry, usage limits, and other features
5. WHEN HITL steps handle errors THEN error propagation SHALL be consistent with other step types

### Requirement 8: Fix Plugin and Validation Systems

**User Story:** As a developer extending Flujo with custom validation and plugins, I want these systems to integrate properly with the execution engine, so that custom logic works reliably.

#### Acceptance Criteria

1. WHEN validation plugins execute THEN they SHALL not trigger unnecessary agent retries
2. WHEN plugin failures occur THEN they SHALL be handled separately from agent failures
3. WHEN validation fails THEN error messages SHALL accurately reflect the validation issue
4. WHEN multiple plugins are used THEN they SHALL execute in the correct order
5. WHEN plugin results redirect execution THEN the redirection SHALL be handled properly

### Requirement 9: Fix Performance and Persistence

**User Story:** As a developer deploying Flujo in production, I want performance to be acceptable and persistence to work correctly, so that applications run efficiently.

#### Acceptance Criteria

1. WHEN persistence operations execute THEN overhead SHALL be within acceptable limits (< 35%)
2. WHEN default backends are configured THEN the correct backend SHALL be used in different contexts
3. WHEN large contexts are persisted THEN performance SHALL remain acceptable
4. WHEN persistence errors occur THEN they SHALL be handled gracefully
5. WHEN backend configuration changes THEN the system SHALL adapt correctly

### Requirement 10: Fix Integration and End-to-End Scenarios

**User Story:** As a developer building complete applications, I want all components to work together seamlessly, so that end-to-end workflows function correctly.

#### Acceptance Criteria

1. WHEN pipeline runners execute THEN they SHALL handle retries and feedback correctly
2. WHEN complex nested scenarios execute THEN all components SHALL integrate properly
3. WHEN end-to-end tests run THEN they SHALL validate complete pipeline flows accurately
4. WHEN integration points are tested THEN they SHALL demonstrate proper component interaction
5. WHEN regression scenarios are tested THEN they SHALL prevent previously fixed issues from recurring