---
name: flujotg
description: Ensure changes in the Flujo repository follow FLUJO_TEAM_GUIDE.md. Use when modifying Flujo core code, policies, executor/runner/CLI behavior, quota or context handling, agent/config management, tests/markers, or SQLite schema in /Users/alvaro1/Documents/Coral/Code/flujo/flujo (or any Flujo repo checkout).
---

# Flujo Team Guide

## Overview

Use this skill to align code changes with the Flujo Team Developer Guide. Read the guide, map the change to relevant sections, and verify the non-negotiable architectural rules before editing.

## Workflow

1. Locate the guide
   - Open `FLUJO_TEAM_GUIDE.md` at the repo root.
   - List headings to find relevant sections: `rg -n "^## " FLUJO_TEAM_GUIDE.md`.

2. Apply non-negotiables
   - Keep step execution logic in `flujo/application/core/step_policies.py`.
   - Re-raise control flow exceptions (`PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError`).
   - Use `ContextManager.isolate()` for complex steps and merge only on success.
   - Enforce quotas with Reserve -> Execute -> Reconcile and use `Quota.split()` for branches.
   - Access configuration via `flujo.infra.config_manager` helpers only.
   - Create agents via `flujo.agents` factories.

3. Map changes to guide sections
   - Executor/dispatcher changes -> "Executor Dispatch Guarantees", "Policy Registration Pattern", "Advanced Performance Architecture".
   - Step policies or new step types -> "Golden Rule", "Policy Implementation Patterns", "Adding New Step Types", "Idempotency in Step Policies".
   - Exception handling -> "Exception Handling: The Architectural Way".
   - Quota/usage/budget changes -> "Budgeting & Quotas (Pure Quota Mode)".
   - Context isolation or retries -> "Context Isolation Requirements".
   - CLI input/IO changes -> "CLI I/O Semantics (Team Standard)".
   - Runner/resume/HITL changes -> "Runner API Usage", "HITL In Loops - Resume Semantics".
   - Agent/config changes -> "Agent and Configuration Management".
   - Tests/markers changes -> "Testing Standards (Markers & Fast/Slow Split)".
   - SQLite schema changes -> "SQLite Schema Safeguards".

4. Verify anti-patterns are avoided
   - Do not add `isinstance` routing in `ExecutorCore`.
   - Do not convert control flow exceptions into `StepResult(success=False)`.
   - Do not read `flujo.toml` or environment variables directly in policies or domain logic.
   - Do not introduce reactive quota checks or legacy governor/breach_event patterns.

5. Record guide references in the change rationale
   - Note the sections consulted when summarizing or reviewing the change.
