# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.4] - 2026-01-19

### Fixed

- **PostgreSQL Backend**: Fixed type cast bug in `save_state` where `memory_usage_mb` parameter was incorrectly cast to `jsonb` instead of `REAL`
- **PostgreSQL Backend**: Enhanced `_parse_timestamp` to log warnings when timestamp strings are unparseable, aiding in identifying serialization issues
- **Test Suite**: Fixed migration version collision by renaming `004_fix_vector_dimensions.sql` to `005_fix_vector_dimensions.sql`
- **Test Suite**: Fixed crash recovery test to properly document state persistence validation instead of asserting subprocess exit codes

### Added

- **Auto-wiring**: `Flujo()` initialization now automatically respects `FLUJO_STATE_URI` environment variable when `state_backend` is not explicitly provided (skips in test mode to preserve test isolation)
- **Vector Store**: Added runtime validation to `PostgresVectorStore.add()` ensuring vectors match expected 1536 dimensions with clear error messages

### Changed

- **CLI Tests**: Updated help snapshot tests to reflect current command structure
- **Crash Recovery**: Improved test robustness by using context managers for SQLite connections and moving imports to module level

## [0.6.3] - 2025-12-29

### Fixed

- Runner: automatically shut down the default SQLite state backend when a run finishes (sync or async) and mark aiosqlite worker threads as daemonized to prevent lingering processes.

### Added

- Runner now exposes `close()/aclose()` plus context manager helpers so callers can explicitly release resources when reusing the same instance.

### Changed

- Sync cleanup now fails fast inside running event loops: `runner.close()` and `with Flujo(...)` raise `TypeError` when called inside `async def`; use `await runner.aclose()` / `async with Flujo(...)` instead.

### Docs

- Documented the new runner cleanup pattern in the user guide.

### Removed

- Optimization layer deleted (including `adaptive_resource_manager`, `circuit_breaker`, `performance_monitor`, and `optimized_error_handler`); `psutil` dependency dropped. `ExecutorCore` still accepts `optimization_config` via the stub for backward compatibility but it is ignored, and module isolation is verified via subprocess tests.

## [0.4.38] - 2025-10-04

### Fixed

- Loop/HITL: Prevent nested loop creation on resume in `DefaultLoopStepExecutor`. Resuming from a HITL pause advances within the current loop iteration using scratchpad indices and evaluates exit conditions in the parent loop context.

### Added

- DSL/Blueprint: `sink_to` forwarded for YAML steps using `uses: "pkg.mod:fn"` callable import path to persist scalar outputs to context.

### Docs

- Team Guide: Added “HITL In Loops – Resume Semantics (Updated)” clarifying pause propagation, idempotent iteration context, exit condition timing, and `sink_to` parity.

### Added

#### Framework Ergonomics Improvements (2025-10-02)

- **HITL Enhancements**:
  - `sink_to` field for HITL steps to automatically store human responses to context paths (e.g., `sink_to: "scratchpad.user_name"`), eliminating boilerplate passthrough steps
  - `resume_input` template variable available after HITL steps, containing the most recent human response
  - Works in templates (`{{ resume_input }}`), expressions (`condition_expression: "resume_input.lower() == 'yes'"`), and agent prompts

- **Context Helpers** - New built-in skills for type-safe context manipulation:
  - `flujo.builtins.context_set` - Set a single context field at a dot-separated path
  - `flujo.builtins.context_merge` - Merge a dictionary into context at a path
  - `flujo.builtins.context_get` - Get a context field with optional default fallback
  - Replaces error-prone passthrough patterns and eliminates `Any` type usage

- **Validation Rules** - Three new linting rules to prevent common mistakes:
  - **V-EX1**: Control Flow Exception Handling - Warns when custom skills catch `PausedException`, `PipelineAbortSignal`, or `InfiniteRedirectError` without re-raising them (prevents broken pause/resume workflows)
  - **V-CTX1**: Context Isolation - Warns when loops or parallel steps use custom skills without `ContextManager.isolate()`, which can break idempotency
  - **V-T5/V-T6**: Template Expression Validation - Detects missing model fields and invalid JSON in templates (already implemented, now documented)

- **Blueprint Loading**:
  - Sync/async validation for `exit_condition` and `condition` functions - Async functions now raise clear `BlueprintError` at load time instead of cryptic runtime TypeErrors

- **Performance**:
  - Fixed systematic SQLite resource leaks in 5 test files, reducing test times from 181s to ~1-5s each (18x-148x speedup, ~15 min CI time saved)

- **Documentation**:
  - Comprehensive updates to `llm.md` with all new patterns, best practices, and anti-patterns
  - Added examples to `docs/hitl.md`, `docs/expression_language.md`, and `docs/user_guide/pipeline_context.md`
  - All validation rules documented in `docs/validation_rules.md` with fix examples
  - Three new example pipelines: `hitl_sink_demo.yaml`, `context_helpers_demo.yaml`, and integration tests

- Validate: Pluggable linter architecture (always‑on) covering Templates (V‑T1..T6), Schema (V‑S1..S3), Context (V‑C1..C3), Agents (V‑A6..A8), Orchestration (V‑P1..P3, V‑L1, V‑CF1, V‑SM1), and Imports (V‑I1..I6). Inline duplicates removed from Pipeline.
- Validate: Import validation performance — path‑keyed caching and realpath cycle detection for `--imports` recursion.
- Validate: New fixers and UX
  - Registry with preview/prompt flow and metrics.
  - `--fix`, `--yes`, `--fix-rules` (per‑rule/glob), `--fix-dry-run` (patch preview).
  - V‑T1 (rewrite `previous_step.output` → `previous_step | tojson`).
  - V‑T3 (correct common filter typos like `to_json`→`tojson`, `lowercase`→`lower`).
  - V‑C2 (map `parent: scratchpad` to `parent: scratchpad.<key>`).
- Validate: JSON output includes `counts` (when `FLUJO_CLI_TELEMETRY=1`), `baseline` deltas, and `fixes` metrics. Adds `fixes_dry_run` when using `--fix-dry-run`.
- SARIF: Enriched rule metadata with stable `helpUri` and rule names from the catalog.
- Docs: `docs/cli/validate.md` updated with fixer options; added `docs/reference/validation_rules.md`.
- Conversational Loops (FSD-033):
  - `conversation: true` loop mode with automatic conversation_history capture and prompt injection via processors
  - History management strategies: `truncate_tokens`, `truncate_turns`, `summarize` with centralized defaults and loop-level overrides
  - Selection controls: `ai_turn_source` (last/all_agents/named_steps), `user_turn_sources` (hitl and/or named steps)
  - Lens trace shows `agent.prompt` events with a sanitized preview of rendered history
  - Wizard support to scaffold conversation blocks and presets
  - Persistence: conversation_history survives pause/resume and project restarts (SQLite backend tested)

### Fixed

- StateMachine YAML loader now compiles `states` that use `uses: imports.<alias>` into first-class `ImportStep`s. This preserves policy-driven execution and prevents fallback to the default Step policy (which could trigger `MissingAgentError` if no agent is present on the import step). Tests include unit, integration, and a regression covering the original scenario.

### Added

- Declarative LoopStep enhancements in YAML loader:
  - `loop.init` (runs once on isolated iteration context)
  - `loop.propagation.next_input` (presets: `context` | `previous_output` | `auto` or template)
  - `loop.output_template` and `loop.output` (object mapping) compiled to `loop_output_mapper`
- Friendly presets for domain users:
  - `conversation: true`, `stop_when: agent_finished`, `propagation: context|previous_output`,
    `output: text: conversation_history`, and simple `init.history.start_with` helpers
- MapStep sugars:
  - `map.init` (pre-run init ops) and `map.finalize` (post-aggregation output mapping)
    with the same templating semantics as loop output.
- Parallel reduce sugar:
  - `reduce: keys|values|union|concat|first|last` to post-process branch outputs while preserving
    input branch order; default remains branch-output mapping.
- CLI improvements:
  - `flujo create --wizard` to generate a natural, friendly YAML without running the Architect.
  - `flujo explain <path>` to summarize a YAML's structure in plain language.
- Policy hook in `DefaultLoopStepExecutor` to execute compiled init ops at iteration 1

### Notes

- Aligns with FLUJO_TEAM_GUIDE policy-driven architecture: control-flow exceptions re-raised,
  context idempotency preserved via isolation, and quotas unchanged.

## [0.6.3] - 2025-08-10

### Added

- `core/default_components.py`: Centralized default implementations for executor composition
  - `OrjsonSerializer`, `Blake3Hasher`, `InMemoryLRUBackend`, `ThreadSafeMeter`
  - `DefaultAgentRunner`, `DefaultProcessorPipeline`, `DefaultValidatorRunner`, `DefaultPluginRunner`
  - `DefaultTelemetry`, `DefaultCacheKeyGenerator`
- `__all__` in `core/default_components.py` and `core/executor_protocols.py` for explicit public surfaces.

### Changed

- Consolidated Protocol interfaces into `core/executor_protocols.py` as the single source of truth; removed duplicates from `ultra_executor.py`.
- Removed the legacy `core/ultra_executor.py` re-export surface and `_UsageTracker` shim; import from
  `core/executor_core.py` and `core/default_components.py` instead. Quota enforcement is handled by
  policies using `ThreadSafeMeter`.
- `application/runner.py` composition updated to import defaults from `core/default_components.py`.
- Classified `PipelineAbortSignal` as a control-flow category in `core/optimized_error_handler.py` to align with FSD-009 (non-retryable control flow).

### Migration

- Recommended imports:
  - Defaults: `from flujo.application.core.default_components import OrjsonSerializer, ...`
  - Interfaces: `from flujo.application.core.executor_protocols import IAgentRunner, ...`
- Backward compatibility: `core/ultra_executor.py` re-exports were removed; update imports to the
  modules above for quota-only execution.

### Notes

- This change aligns with the policy-driven architecture in `FLUJO_TEAM_GUIDE.md` and the FSD for decomposing the ultra executor. No runtime behavior changes are intended.

## [0.4.37] - 2025-08-14

### Added

- Project scaffolding via `flujo init` with templates (`flujo.toml`, `pipeline.yaml`, `skills/`, `.flujo/`).
- Conversational `flujo create` enhancements: optional goal prompt, pipeline name prompt (injected as top-level `name:`), and per-run budget prompt appended to `flujo.toml` under `[budgets.pipeline."<name>"]`.
- Project-aware defaults: `flujo run` and `flujo validate` now infer the project’s `pipeline.yaml` when no file path is provided.
- `lens replay` now looks for `pipeline.yaml` in the project when `--file` is omitted; still supports `--file` for `.yaml` or Python definitions.
- Template `flujo.toml` sets `state_uri = "sqlite:///.flujo/state.db"` so lens and telemetry use project-local state by default.

### Changed

- Inside a project, `flujo create` overwrites `pipeline.yaml` by default (no `--force` needed). For non-project output directories, original `--force` behavior remains.
- Documentation updated to reflect the new project-based journey and project-aware commands.

### Migration Guidance

- Existing flows that passed explicit file paths continue to work unchanged.
- Recommended: initialize a project (`flujo init`), then run `flujo create` and `flujo run` from inside the project.
- For `lens` tooling, the new template sets a project-local `state_uri`. If you used a global DB, you can keep using `FLUJO_STATE_URI` or set `state_uri` in your `flujo.toml`.

## [0.4.35] - 2025-01-15

### Added

- **Performance Optimizations**: Enhanced execution efficiency and resource management
  - Improved parallel step execution with better resource allocation
  - Optimized memory usage patterns for large-scale workflows
  - Enhanced caching mechanisms for better performance
  - Streamlined context handling for improved throughput

### Changed

- **Stability Improvements**: Enhanced error handling and recovery mechanisms
  - Improved error recovery and resilience patterns
  - Better exception handling across pipeline components
  - Enhanced validation and error reporting
  - More robust state management and persistence

### Fixed

- **Bug Fixes**: Resolved various edge cases and issues
  - Fixed context serialization issues in complex workflows
  - Resolved race conditions in parallel execution
  - Corrected memory leaks in long-running pipelines
  - Fixed edge cases in error recovery mechanisms

## [0.4.34] - 2025-01-15

### Added

- **Enhanced Documentation**: Improved documentation structure and content
  - Fixed broken internal links in documentation pages
  - Updated navigation structure for better user experience
  - Enhanced cookbook examples with current API patterns
  - Improved documentation coverage for new features

### Fixed

- **Documentation Build Warnings**: Resolved mkdocs build warnings and issues
  - Fixed missing cookbook pages and broken internal links
  - Corrected navigation structure and page references
  - Improved documentation build process reliability
  - Enhanced user experience with better documentation organization

### Changed

- **Code Quality Improvements**: Enhanced codebase maintainability and reliability
  - Improved error handling and validation patterns
  - Enhanced test coverage and reliability
  - Better code organization and documentation
  - Streamlined development workflow

## [0.4.33] - 2025-01-15

### Added

- **Budget-Aware Workflows**: Enhanced workflow execution with cost and token budget management
  - New budget-aware execution strategies for cost-effective AI workflows
  - Token usage tracking and optimization across pipeline steps
  - Cost monitoring and alerting capabilities for production deployments
  - Dynamic resource allocation based on budget constraints

### Changed

- **Performance Optimizations**: Improved execution efficiency and resource utilization
  - Enhanced parallel step execution with better resource management
  - Optimized context handling for large-scale workflows
  - Improved memory usage patterns for long-running pipelines
  - Better error recovery and resilience mechanisms

### Fixed

- **Documentation Updates**: Resolved documentation build warnings and link issues
  - Fixed broken internal links in documentation
  - Updated navigation structure for better user experience
  - Enhanced cookbook examples with current API patterns
  - Improved documentation coverage for new features

## [0.4.32] - 2025-07-14

### Fixed

- **CI/CD Workflow Improvements**: Enhanced GitHub Actions release workflow for robust PyPI publishing
  - Added `uv` installation step to fix missing dependency in CI environment
  - Simplified release workflow to industry-standard approach without automatic version bumping
  - Fixed permission issues by removing automatic tag creation and pushing
  - Improved changelog generation with manual control over release process
  - Added proper error handling and dependency management for reliable builds

### Changed

- **Release Process**: Streamlined release workflow for better reliability and control
  - Manual version management in `pyproject.toml` for explicit control
  - Tag-based triggers only (no automatic version bumping)
  - Simplified changelog generation without external dependencies
  - Enhanced build and test process with proper dependency installation

## [0.4.31] - 2025-07-14

### Fixed

- **Makefile Enhancements**: Added missing targets for CI/CD pipeline
  - Added `pip-dev` target for installing development dependencies
  - Added `package` target for building distribution files
  - Improved development workflow with comprehensive build system support
  - Enhanced CI/CD compatibility with proper dependency management

## [0.4.29] - 2025-07-14

### Added

- **Documentation Updates**: Enhanced project documentation and branding
  - Updated main documentation page with new headline "The Framework for AI Systems That Learn"
  - Improved documentation structure with comprehensive navigation
  - Enhanced mkdocs configuration for better user experience
  - Updated contact information and licensing details

### Changed

- **License Management**: Updated commercial licensing and contact information
  - Enhanced commercial license agreement with comprehensive legal terms
  - Updated contact email to aandresalvarez@gmail.com
  - Improved dual licensing strategy (AGPL-3.0 + Commercial)
  - Enhanced intellectual property protection and usage terms

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
