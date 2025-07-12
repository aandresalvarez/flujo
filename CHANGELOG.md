# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2025-02-20

### Added
- `run_id` parameter for `Flujo.run()` and `run_async()` simplifies durable workflow APIs.
- `serializer_default` on `StateBackend` implementations for advanced serialization.

### Changed
- Upgraded to Pydantic 2.0.

### Fixed
- Nested Pydantic models persist correctly in workflow state.

## [0.6.1] - 2025-01-15

### Added
- **Optimized ParallelStep Context Copying**: New `context_include_keys` parameter for `Step.parallel()` to selectively copy only needed context fields
  - Significantly reduces memory usage and overhead when working with large context objects
  - Allows developers to specify which context fields are required by parallel branches
  - Maintains backward compatibility - omitting the parameter copies the entire context
  - Performance improvement scales with context size and number of parallel branches
- **Proactive Governor Cancellation**: Enhanced `ParallelStep` with immediate sibling task cancellation
  - When any branch exceeds usage limits (cost or token limits), all sibling branches are immediately cancelled
  - Prevents wasted resources and time by stopping unnecessary work early
  - Uses `asyncio.Event` for efficient coordination between parallel tasks
  - Improves cost efficiency and reduces execution time for usage-limited scenarios
- **Comprehensive Benchmark Tests**: Added performance validation for new ParallelStep features
  - Integration tests verify selective context copying behavior
  - Benchmark tests measure performance improvements with large context objects
  - Cancellation tests ensure proper cleanup when usage limits are exceeded
  - Example script demonstrates practical usage of new features

### Changed
- **Enhanced ParallelStep Implementation**: Refactored `_execute_parallel_step_logic` for better performance and resource management
  - Optimized context copying strategy with selective field inclusion
  - Improved error handling and cancellation logic
  - Better resource cleanup and task management
  - More efficient coordination between parallel branches

### Fixed
- **Test Context Model Inheritance**: Fixed test context models to inherit from `flujo.domain.models.BaseModel`
  - Resolves Pydantic model inheritance issues in test suite
  - Ensures proper type compatibility with Flujo's domain models
  - Maintains test isolation and reliability
- **Pydantic-AI Compatibility:** Fixed a `TypeError` by updating how generation parameters like `temperature` are passed to the underlying `pydantic-ai` agent, ensuring compatibility with `pydantic-ai>=0.4.1`.
- **Dependencies:** Updated `pyproject.toml` to require `pydantic-ai>=0.4.1`.
- **Deprecated Recipes:** Marked `AgenticLoop` and `Default` classes as deprecated. Use the factory functions in `flujo.recipes.factories`.

## [0.6.0] - 2025-01-15

### Added
- **Curated Layered Public API**: Complete architectural refactor with organized, layered import structure
  - Core types (`Pipeline`, `Step`, `Context`, `Result`) available at top level (`from flujo import Pipeline`)
  - Related components grouped into logical submodules (`recipes`, `testing`, `plugins`, `processors`, `models`, `exceptions`, `validation`, `tracing`, `utils`, `domain`, `application`, `infra`)
  - Improved discoverability and reduced import complexity
  - Enhanced developer experience with clear module boundaries
- **ContextAwareAgentProtocol**: Type-safe context handling for agents
  - New protocol for agents that need typed pipeline context
  - Eliminates runtime errors and provides better IDE support
  - Maintains backward compatibility with AsyncAgentProtocol
- **Comprehensive Test Suite**: Robust testing infrastructure with 359 passing tests
  - Fixed all import errors and circular dependency issues
  - Resolved context mutation and agent protocol signature mismatches
  - Implemented proper settings patching for isolated test execution
  - Added systematic test fixes for all submodules and components
- **Enhanced Code Quality**: Production-ready codebase with comprehensive quality checks
  - All linting errors resolved (`ruff` compliance)
  - Complete type checking compliance (`mypy` success)
  - Security scanning passed (`bandit` validation)
  - Removed unused imports and dead code
  - Improved error handling and validation patterns

### Changed
- **BREAKING CHANGE**: Complete API restructuring for better organization and maintainability
  - Moved from flat import structure to curated, layered public API
  - Core types remain at top level for backward compatibility
  - Related functionality grouped into logical submodules
  - Updated all examples and documentation to use new import structure
  - Added migration guide for users transitioning from flat imports
- **BREAKING CHANGE**: Standardized context parameter injection
  - Unified context parameter injection to use `context` exclusively
  - Removed support for `pipeline_context` parameter in step functions, agents, and plugins
  - All context injection now uses the `context` parameter name
  - This aligns the implementation with the documented API contract
- **Improved Module Organization**: Better separation of concerns and encapsulation
  - Domain models and business logic separated from infrastructure
  - Application services isolated from domain logic
  - Infrastructure concerns properly abstracted
  - Clear boundaries between different architectural layers
- **Enhanced Error Handling**: More robust error management throughout the codebase
  - Consistent error patterns and exception handling
  - Better error messages and debugging information
  - Improved validation error reporting
  - Structured exception mechanisms for better error recovery

### Fixed
- **Import System**: Resolved all circular dependency and import issues
  - Fixed module import errors in test suite
  - Eliminated circular dependencies between submodules
  - Proper module initialization and attribute access
  - Consistent import patterns across the codebase
- **TypeAdapter Handling**: Enhanced `make_agent_async` to seamlessly handle `pydantic.TypeAdapter` instances
  - Automatically unwraps TypeAdapter instances to extract underlying types
  - Supports complex nested types like `List[Dict[str, MyModel]]`
  - Supports Union types like `Union[ModelA, ModelB]`
  - Maintains backward compatibility with regular types
- **Test Infrastructure**: Comprehensive test suite fixes and improvements
  - Fixed settings singleton patching for isolated test execution
  - Resolved context mutation issues in test scenarios
  - Fixed agent protocol signature mismatches
  - Corrected custom context model usage in tests
  - Implemented robust test isolation and cleanup
- **Documentation and Examples**: Updated all documentation to reflect new API structure
  - Fixed import statements in all examples
  - Updated documentation to use new submodule structure
  - Corrected example execution paths and import patterns
  - Enhanced documentation clarity and accuracy
  - **Updated "The Flujo Way" guide** with current API structure and ContextAwareAgentProtocol
- **Development Workflow**: Improved development and testing experience
  - Fixed `make quality` command for comprehensive quality checks
  - Enhanced `make test` and `make cov` commands
  - Improved development environment setup
  - Better error reporting and debugging tools

### Removed
- **Obsolete Submodules**: Cleaned up problematic module structure
  - Removed empty `__init__.py` files that caused import issues
  - Eliminated redundant module hierarchies
  - Streamlined module organization for better maintainability
  - Reduced complexity in import resolution
- **Repository Artifacts**: Cleaned up development artifacts
  - Removed obsolete backup files (`*.orig`) and temporary documentation
  - Eliminated patch files and standalone debug scripts
  - Improved contributor onboarding experience with cleaner repository

## [0.5.0] - 2025-07-02

### Added
- **Robust TypeAdapter Support**: Enhanced `make_agent_async` to seamlessly handle `pydantic.TypeAdapter` instances
  - Automatically unwraps TypeAdapter instances to extract underlying types
  - Supports complex nested types like `List[Dict[str, MyModel]]`
  - Supports Union types like `Union[ModelA, ModelB]`
  - Maintains backward compatibility with regular types
  - Enables modern Pydantic v2 patterns for non-BaseModel types
- **Enhanced CLI User Experience**: Improved command-line interface robustness and usability
  - Added `typer.Choice` validation for `--scorer` option with automatic tab completion
  - Enhanced help text generation for scoring strategy options
  - Removed manual validation logic in favor of built-in Typer validation
- **Comprehensive Type Safety**: Enabled full type checking for CLI module
  - Removed global `# type: ignore` directive from CLI module
  - Added proper generic type annotations for Pipeline and Step types
  - Enhanced type safety throughout the command-line interface

### Changed
- **BREAKING CHANGE**: Unified context parameter injection to use `context` exclusively
  - Removed support for `pipeline_context` parameter in step functions, agents, and plugins
  - All context injection now uses the `context` parameter name
  - This aligns the implementation with the documented API contract
  - Users who relied on `pipeline_context` parameter must update their code to use `context`
  - Removed deprecation warnings and backward compatibility logic for `pipeline_context`
- **Enhanced Documentation**: Improved clarity and discoverability of validation features
  - Added comprehensive documentation for `strict` parameter in `Step.validate_step`
  - Clarified difference between strict and non-strict validation modes
  - Added practical examples showing audit vs. blocking validation patterns
  - Updated Pipeline DSL guide with validation best practices

### Fixed
- **Repository Hygiene**: Cleaned up development artifacts and improved project structure
  - Removed obsolete backup files (`*.orig`) and temporary documentation
  - Eliminated patch files and standalone debug scripts
  - Improved contributor onboarding experience with cleaner repository
- **Test Suite Stability**: Fixed test failures related to context parameter migration
  - Updated test assertions to use new `context` parameter consistently
  - Ensured all integration tests pass with unified parameter naming
- **Code Quality**: Addressed linting and type checking issues
  - Removed unused imports and variables
  - Fixed type comparison issues in test code
  - Enhanced overall code quality and maintainability

## [0.4.24] - 2025-06-30

### Added
- Pre-flight pipeline validation with `Pipeline.validate()` returning a detailed report.
- New `flujo validate` CLI command to check pipelines from the command line.

## [0.4.25] - 2025-07-01

### Fixed
- `make_agent_async` now accepts `pydantic.TypeAdapter` instances for
  `output_type`, unwrapping them for proper schema generation and validation.

## [0.4.23] - 2025-06-27

### Fixed
- Loop iteration spans now wrap each iteration, eliminating redundant spans
- Conditional branch spans record the executed branch key for clarity
- Console tracer tracks nesting depth, indenting start/end messages accordingly

## [0.4.22] - 2025-06-23

### Added
- Distributed `py.typed` for PEP 561 type hint compatibility.

### Fixed
- Improved CI/CD workflows to gracefully handle Git tag conflicts.

## [0.4.18] - 2024-12-19

### Fixed
- Fixed parameter passing to prioritize 'context' over 'pipeline_context' for backward compatibility
- Ensures step functions receive the parameter name they expect, maintaining compatibility with existing code
- Resolves issue where Flujo engine was passing 'pipeline_context' instead of 'context' to step functions

## [0.4.15] - 2024-12-19

### Changed
- Version bump for release

## [Unreleased]

## [0.4.14] - 2024-12-19

### Changed
- Version bump for release

## [0.4.13] - 2025-06-19

### Added
- Enhanced Makefile with pip-based development workflow support
- New `pip-dev` target for installing development dependencies with pip
- New `pip-install` target for installing package in development mode
- New `clean` target for cleaning build artifacts and caches

### Changed
- Improved development environment setup with better tooling support
- Enhanced project documentation and build system configuration

## [0.4.12] - 2024-12-19

### Changed
- Version bump for release

## [0.4.11] - 2024-12-19

### Changed
- Additional improvements and fixes

## [0.4.1] - 2024-12-19

### Fixed
- Fixed step retry logic to properly handle max_retries configuration
- Fixed pipeline execution to allow step retries before halting
- Fixed plugin validation loop to correctly handle retries and redirections
- Fixed failure handler execution during retry attempts
- Fixed redirect loop detection for unhashable agent objects
- Added usage limits support to loop and conditional step execution
- Improved error handling in streaming pipeline execution
- Fixed token and cost accumulation in step results

## [0.4.0] - 2024-12-19

### Added
- Intelligent evaluation system with traceability
- Pluggable execution backends for enhanced flexibility
- Streaming support with async generators
- Human-in-the-loop (HITL) support for interactive workflows
- Usage governor with cost and token limits
- Managed resource injection system
- Benchmark harness for performance testing
- Comprehensive cookbook documentation with examples
- Lifecycle hooks and callbacks system
- Agentic loop recipe for exploration workflows
- Step factory and fluent builder patterns
- Enhanced error handling and validation

### Changed
- Improved step execution request handling
- Enhanced backend dispatch for nested steps
- Better context passing between pipeline components
- Updated documentation and examples
- Improved type safety and validation

### Fixed
- Step output handling issues
- Parameter detection cache for unhashable callables
- Agent wrapper compatibility with Pydantic models
- Various linting and formatting issues

## [0.3.6] - 2024-01-XX

### Fixed
- Changelog generation and version management
- Documentation formatting and references

## [0.3.5] - 2024-01-XX

### Fixed
- Workflow syntax and version management

## [0.3.4] - 2024-01-XX

### Added
- Initial release with core orchestration features

## [0.3.3] - 2024-01-XX

### Added
- Basic pipeline execution framework

## [0.3.2] - 2024-01-XX

### Added
- Initial project structure and core components
