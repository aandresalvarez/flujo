# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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