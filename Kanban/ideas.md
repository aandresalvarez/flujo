This document replaces the old "complexity ranking" with a proposal that is aligned
to AGENTS.md and FLUJO_TEAM_GUIDE.md. The new version prioritizes changes that
reinforce the policy-driven architecture, strict typing, and centralized config,
while avoiding API-breaking refactors that violate the core dispatcher contract.

## Proposal: Excellence Roadmap (Aligned)

### 1) Security: Hardened Sandboxing (Config-first)
**Goal:** Apply Docker resource limits via configuration, not hard-coded values.

**Work:**
- Add settings in `flujo.infra.config_manager` for sandbox limits.
- Wire Docker limits in `flujo/infra/sandbox/docker_sandbox.py` using settings.
- Add tests that validate limits are applied and defaults are used correctly.

**Why this aligns:** Centralized configuration is mandatory; policies and core logic
must not read env or config directly. This preserves team rules while improving safety.

### 2) Code Hygiene: Typed Test Contexts
**Goal:** Remove runtime test dependencies and enforce typed fixtures.

**Work:**
- Remove `unittest.mock` fallback code in runtime modules.
- Add a typed `TestContext` in `tests/test_types/` (or reuse existing fixtures).
- Migrate tests to typed fixtures (`tests/test_types/fixtures.py`).

**Why this aligns:** The guide mandates strict typing and typed fixtures. This
reduces technical debt without touching execution semantics.

### 3) Observability: SQLite Span Exporter (Config-gated)
**Goal:** Provide zero-config local tracing without breaking optimized telemetry.

**Work:**
- Implement a SQLite OpenTelemetry exporter that writes to the existing `spans` table.
- Gate it behind telemetry config (opt-in or safe default).
- Preserve batching/buffering to avoid per-span write overhead.

**Why this aligns:** Telemetry is part of the runtime, but must not bypass the
optimized telemetry pipeline or add hot-path cost without configuration.

### 4) Structural Integrity: Remove Runtime Imports in DSL
**Goal:** Break circular imports using `TYPE_CHECKING` and forward references.

**Work:**
- Move imports to top-level with `if TYPE_CHECKING:`.
- Convert type hints to string references.
- Extract traversal helpers where cycles persist.

**Why this aligns:** Keeps runtime clean, supports strict typing, and avoids
implicit imports inside business logic.

### 5) Deferred: Spec/Runtime Split
**Status:** Defer unless there is a hard requirement to serialize pipelines across
process boundaries.

**Why defer:** The change breaks public APIs, disrupts policies, and risks
introducing step-specific logic into `ExecutorCore`, which is explicitly forbidden.

## Why the Proposal Changed

1) **Policy-driven execution is non-negotiable.**
   The previous plan risked rewriting `ExecutorCore` to handle step-specific
   runtime building. This violates the dispatcher-only rule.

2) **Centralized configuration is required.**
   Sandboxing changes must be routed through `config_manager`, not hard-coded.

3) **Type safety is a hard gate.**
   The team guide mandates typed fixtures and strict mypy compliance, so test
   hygiene must be typed-first.

4) **Avoid breaking API surface without a migration plan.**
   The spec/runtime split is rewrite territory and does not align with the
   current "policy-first" execution model.

## Recommendation

If time is limited:
1) Do **#1 Security** and **#2 Test Hygiene** first.
2) Do **#3 Observability** next, gated by config.
3) Tackle **#4 Imports** when bandwidth allows.
4) **Defer #5** until a serialization requirement exists.
