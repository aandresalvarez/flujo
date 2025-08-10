1. Rationale & First Principles
Goal: To decompose the monolithic flujo/application/core/ultra_executor.py file into smaller, single-responsibility modules.
Why: This applies the Separation of Concerns principle at the module level. It improves code organization, readability, and maintainability, making the new architecture easier to understand and extend. It reduces cognitive load and aligns the codebase with standard Python best practices.
2. Scope of Work
Create New Files:
flujo/application/core/executor_protocols.py: This file will contain all the I... protocol definitions (ISerializer, IHasher, ICacheBackend, etc.).
flujo/application/core/default_components.py: This file will contain all the default concrete implementations (OrjsonSerializer, Blake3Hasher, InMemoryLRUBackend, ThreadSafeMeter, DefaultAgentRunner, etc.).
flujo/application/core/executor_core.py: This file will contain the ExecutorCore class itself.
Move Code:
Cut all Protocol definitions from ultra_executor.py and paste them into executor_protocols.py.
Cut all default implementation classes from ultra_executor.py and paste them into default_components.py.
Cut the ExecutorCore class from ultra_executor.py and paste it into executor_core.py.
Update Imports:
In executor_core.py, update the imports to pull the protocols and default components from their new locations.
In flujo/application/runner.py (the Composition Root), update the imports to pull ExecutorCore and the default components from their new files.
In flujo/infra/backends.py, update the import for ExecutorCore.
Run a project-wide search for any other files that import from ultra_executor and update their import paths accordingly.
Delete ultra_executor.py:
Once all code has been moved and all imports have been updated, the now-empty flujo/application/core/ultra_executor.py file will be deleted.
3. Implementation Details
flujo/application/core/executor_protocols.py: Will contain only Protocol definitions.
flujo/application/core/default_components.py: Will contain the concrete classes like OrjsonSerializer, Blake3Hasher, etc. It will import from executor_protocols.py.
flujo/application/core/executor_core.py: Will contain the ExecutorCore class. It will import from both executor_protocols.py and default_components.py (for the default values in its __init__ signature).
4. Testing Strategy
No New Tests Needed: This is a pure refactoring of file structure. No logic is changing.
Static Analysis: After moving the code and updating imports, run mypy. It is the primary tool for verifying that all the import paths have been corrected and the type relationships are still understood.
Regression Tests (Existing):
Run the entire test suite.
A 100% pass rate is the definitive proof that the file decomposition was successful and did not break any functionality. The system's runtime behavior should be absolutely identical.
5. Acceptance Criteria
The new files (executor_protocols.py, default_components.py, executor_core.py) are created and populated with the correct classes.
The old flujo/application/core/ultra_executor.py file has been deleted.
All imports across the entire project that previously referenced ultra_executor have been updated to point to the new, more specific files.
The project passes all static analysis checks (mypy) with no import errors.
100% of the existing test suite passes, confirming the refactor was purely structural and introduced no regressions.