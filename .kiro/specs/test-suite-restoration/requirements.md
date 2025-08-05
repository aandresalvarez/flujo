# Requirements Document

## Introduction

The Flujo test suite has 241 failing tests that need to be systematically diagnosed and fixed to ensure the framework's reliability and prevent regressions. The failures span multiple core areas including usage governance, parallel execution, serialization, database backends, and CLI functionality. This feature will restore the test suite to a passing state while maintaining the integrity of the existing functionality.

## Requirements

### Requirement 1: Usage Governor Test Failures

**User Story:** As a developer, I want the usage governor tests to pass so that cost and token limits are properly enforced in production.

#### Acceptance Criteria

1. WHEN the usage governor checks cost limits THEN it SHALL format cost values consistently with test expectations
2. WHEN the usage governor checks token limits THEN it SHALL properly aggregate token counts from step history
3. WHEN usage limits are exceeded THEN the system SHALL raise UsageLimitExceededError with correct error messages
4. WHEN no usage limits are configured THEN the system SHALL not raise any exceptions

### Requirement 2: Parallel Step Execution Failures

**User Story:** As a developer, I want parallel step execution tests to pass so that concurrent pipeline branches work correctly.

#### Acceptance Criteria

1. WHEN parallel steps execute THEN they SHALL properly validate context objects
2. WHEN parallel branches complete THEN they SHALL correctly merge context state
3. WHEN parallel execution encounters errors THEN it SHALL provide meaningful error messages
4. WHEN usage limits are breached during parallel execution THEN it SHALL properly detect and prevent the breach

### Requirement 3: Serialization Edge Cases

**User Story:** As a developer, I want serialization tests to pass so that complex objects can be properly stored and retrieved.

#### Acceptance Criteria

1. WHEN serializing mock objects THEN the system SHALL use registered custom serializers
2. WHEN encountering unknown types THEN the system SHALL provide fallback serialization
3. WHEN serializing edge cases (None, empty collections, etc.) THEN it SHALL handle them gracefully
4. WHEN circular references exist THEN the system SHALL detect and handle them appropriately

### Requirement 4: SQLite Backend Issues

**User Story:** As a developer, I want SQLite backend tests to pass so that state persistence works reliably.

#### Acceptance Criteria

1. WHEN database corruption occurs THEN the system SHALL handle it gracefully
2. WHEN schema migrations are needed THEN they SHALL execute successfully
3. WHEN backup operations fail THEN the system SHALL provide appropriate fallbacks
4. WHEN concurrent access occurs THEN the system SHALL maintain data integrity

### Requirement 5: CLI and Lens Command Failures

**User Story:** As a developer, I want CLI tests to pass so that command-line tools work correctly.

#### Acceptance Criteria

1. WHEN lens commands are executed THEN they SHALL handle database schema mismatches
2. WHEN querying run history THEN the system SHALL use correct column names
3. WHEN displaying results THEN it SHALL format output consistently
4. WHEN database is empty THEN commands SHALL handle the case gracefully

### Requirement 6: Fallback Logic Issues

**User Story:** As a developer, I want fallback logic tests to pass so that error recovery works as expected.

#### Acceptance Criteria

1. WHEN fallback loops are detected THEN the system SHALL raise InfiniteFallbackError
2. WHEN fallback agents fail THEN error messages SHALL be properly formatted
3. WHEN retry scenarios occur THEN attempt counts SHALL be accurate
4. WHEN complex metadata is involved THEN it SHALL be preserved correctly

### Requirement 7: Test Infrastructure Robustness

**User Story:** As a developer, I want the test infrastructure to be robust so that tests run reliably and provide accurate results.

#### Acceptance Criteria

1. WHEN tests run THEN they SHALL use proper isolation mechanisms
2. WHEN mock objects are used THEN they SHALL be properly serialized
3. WHEN test fixtures are created THEN they SHALL match production behavior
4. WHEN tests complete THEN they SHALL clean up resources properly