## FSD-002: Policy-Driven Executor Logic Migration — Status

### Scope
Migrate step-execution logic out of the monolithic `ExecutorCore` in `flujo/application/core/ultra_executor.py` into specialized policies in `flujo/application/core/step_policies.py`, turning the core into a pure dispatcher. Targeted to resolve Category 2 (Signature Mismatches) and Category 3 (Logic Bugs) by improving modularity, testability, and isolation.

### Completed Work
- **Dispatcher wiring**
  - `ExecutorCore.execute` delegates to policy executors for `SimpleStep`, `AgentStep`, and others (previous groundwork retained).
  - `DefaultDynamicRouterStepExecutor` kept back-compat expectations (optional `step` param; delegation via `core._handle_parallel_step(...)`).

- **SimpleStep policy inputs (Phase-1)**
  - `DefaultSimpleStepExecutor.execute` now:
    - Preprocesses input via processor pipeline (prompt processing) when possible.
    - Builds agent options from `Step.config` (`temperature`, `top_k`, `top_p`).
    - Uses a shallow-copy + shim (`_policy_processed_payload`, `_policy_agent_options`) to pass preprocessed data/options through `core._execute_simple_step(..., _from_policy=True)` without public signature changes.

- **Plugin redirector policy**
  - `DefaultPluginRedirector.run` raises a generic `Exception` on plugin failure. This lets `ExecutorCore` re-wrap it as its local `PluginError` to satisfy tests and preserve legacy error surfaces and messages.

- **Validator invoker policy**
  - `DefaultValidatorInvoker.validate` invokes validators and raises a generic `Exception` on first invalid result. Core now delegates validator checks through the policy for consistent wrapping/retry/fallback orchestration.

- **Validation step strictness (hybrid check)**
  - For DSL-built validation steps (`meta.is_validation_step`):
    - Added strict vs non-strict handling through `meta.strict_validation`.
    - Strict: fail and drop output; Non-strict: pass-through output and set `metadata_["validation_passed"] = False`.
    - Uses `run_hybrid_check` to combine plugin and validator feedback when applicable.

- **Dynamic Router back-compat fixes**
  - Ensured optional `step` param in handler/policy signatures where tests inspect legacy signatures.
  - Delegation path matches legacy expectations to avoid signature-related regressions.

- **Loop migration attempt and rollback**
  - Attempted to move `_execute_loop` logic into `DefaultLoopStepExecutor` → caused broad loop-path regressions.
  - Reverted: `DefaultLoopStepExecutor.execute` delegates to `core._execute_loop(...)`.
  - Restored core loop implementation parity: `ExecutorCore._execute_loop = OptimizedExecutorCore._execute_loop`.

- **SimpleStep parity adjustments (Phase-2, incremental)**
  - Unpacked final outputs at policy boundary so `StepResult.output` carries the primitive/result, not wrapper objects.
  - Normalized plugin IO in `DefaultPluginRedirector` so plugins consistently receive `{ "output": ... }` and dict-based outcomes with `{"output": x}` are flattened back to `x`.
  - Treated plugin-originated failures as non-retryable at policy layer to match legacy fallback semantics; proceed directly to fallback when present.
  - Cached successful fallback results when cache is enabled, marking `metadata_["fallback_triggered"] = True` and setting `feedback = None` on success.

### Notable Fixes/Adjustments
- Removed legacy monkey-patching for `_execute_agent_step_fn` and `_handle_loop_step_fn` (prior groundwork).
- Corrected parameter names and delegation for dynamic router and parallel step pathways.
- Consolidated plugin/validator behavior so core owns error wrapping (policies raise generic exceptions by design).
- Improved redirect loop detection via `DefaultPluginRedirector` by tracking visited agents and raising on cycles.
- Ensured processor pipelines are applied symmetrically around agent execution and that cost/token accounting flows through policy-owned paths.

### Current State (high level)
- Policies are in place and own SimpleStep orchestration end-to-end (preprocessing, options, agent run, plugins, validators, retries, fallback, metrics, cache), implemented in `_execute_simple_step_policy_impl` and wired via `DefaultSimpleStepExecutor`.
- Core remains the dispatcher and retains loop execution; Loop policy still delegates to core during migration.

### Pending Tasks
- **Task 1.2: SimpleStep migration (Phase-2 parity validation)**
  - [x] Unpack final outputs and normalize plugin IO contracts.
  - [x] Treat plugin-originated errors as non-retryable and cache successful fallbacks.
  - [ ] Validate and tune attempt/metrics accounting (tokens, cost, latency) for edge cases (multi-retry + fallback chains) to match assertions.
  - [ ] Align specific error message text where tests assert substrings (plugin vs agent prefixes) without weakening semantics.
  - [ ] Keep strict/non-strict validation semantics intact after migration.

- **Task 2.1: LoopStep migration**
  - [ ] Migrate `_execute_loop` body into `DefaultLoopStepExecutor.execute`, parameterize all internal calls through `core`.
  - [ ] Validate error messages, iteration mappers, context isolation/merging, exit conditions, and fallback semantics.
  - [ ] Re-switch `ExecutorCore._execute_loop` to delegate to the policy once behavior parity is confirmed.

- **Task 3.1: Handler purity audit**
  - [ ] Verify `_handle_parallel_step`, `_handle_conditional_step`, `_handle_dynamic_router_step`, `_handle_hitl_step` contain no business logic and exclusively delegate to policies.
  - [ ] Remove any residual logic from handlers uncovered during the audit.

- **Task 4.1: Test updates for signature inspection**
  - [ ] Update tests that introspect legacy `ExecutorCore` method signatures to instead target the policy `execute` methods (public surface), maintaining meaningful coverage.

- **Task 4.2: Final cleanup**
  - [ ] Remove unused private methods from `ExecutorCore` after migration.
  - [ ] Remove shim attributes (`_policy_*`) once public signatures are updated and back-compat is no longer required.

### Guardrails and Approach
- Follow first principles: single responsibility per policy; strict typing and encapsulation.
- Maintain behavior parity during each step; run fast tests after each edit.
- Use shim attributes only during transition; avoid public API changes until final cleanup.
- Favor raising generic exceptions in policies; let core re-wrap into domain-specific error types for consistent messaging.

### File References
- Core: `flujo/application/core/ultra_executor.py`
- Policies: `flujo/application/core/step_policies.py`
- Hybrid validation: `flujo/application/core/hybrid_check.py`

### Next Execution Slice
- Focus on parity validation for SimpleStep policy:
  - Run targeted tests to validate recent changes without long suite runs:
    - `pytest -q tests/integration/test_pipeline_runner.py::test_runner_unpacks_agent_result`
    - `pytest -q tests/integration/test_resilience_features.py::test_cached_fallback_result_is_reused`
    - `pytest -q tests/integration/test_pipeline_runner.py::test_timeout_and_redirect_loop_detection`
  - If deltas remain, adjust error text and attempt aggregation in policy, keeping metrics accounting consistent.
  - Once parity is green for these scenarios, expand to broader SimpleStep-related suites, then resume LoopStep migration (Task 2.1).