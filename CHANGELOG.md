# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-07-02

### Added
- **Robust TypeAdapter Support**: Enhanced `make_agent_async` to seamlessly handle `pydantic.TypeAdapter` instances
  - Automatically unwraps TypeAdapter instances to extract underlying types
  - Supports complex nested types like `List[Dict[str, MyModel]]`
  - Supports Union types like `Union[ModelA, ModelB]`
  - Maintains backward compatibility with regular types
  - Enables modern Pydantic v2 patterns for non-BaseModel types

### Changed
- **BREAKING CHANGE**: Unified context parameter injection to use `context` exclusively
  - Removed support for `pipeline_context` parameter in step functions, agents, and plugins
  - All context injection now uses the `context` parameter name
  - This aligns the implementation with the documented API contract
  - Users who relied on `pipeline_context` parameter must update their code to use `context`
  - Removed deprecation warnings and backward compatibility logic for `pipeline_context`

### Fixed
- Resolved API inconsistency between documentation and implementation
- Eliminated developer confusion caused by parameter name mismatch
- Improved code clarity and reduced technical debt

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