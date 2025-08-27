# FSD-033: Intelligent, Scoped Conversational History Management

Status: Approved for Implementation
Owner: Flujo Core Architecture Team
Last Updated: 2025-08-26
Supersedes: FSD-032

## 1) Purpose & Scope

Deliver a zero‑boilerplate, production‑ready conversational loop for Flujo that:
- Automatically manages conversation history during iterative loops.
- Injects history into agent prompts safely and transparently.
- Preserves strict architectural principles (policy-driven, idempotent, proactive quota, centralized config).
- Provides robust cost controls (truncation + summarization) and full observability (lens/tracing).

Out of scope: Non-loop conversational primitives and third‑party chat history backends.

## 2) First Principles

- Explicit over implicit: Feature is opt‑in and scoped to a loop step.
- Strong invariants: Control‑flow exceptions re‑raised; context updates idempotent; configs centralized.
- Composition over mutation: Augment agents via processors/wrappers, not by rewriting agent definitions.
- Production‑grade UX: Defaults that “just work,” with clear overrides and complete lens visibility.
- Governance first: Cost and token limits enforced proactively; histories bounded and auditable.

## 3) Requirements

### 3.1 Functional
- Add `loop.conversation: true` to enable conversational behavior within that loop’s body only.
- Maintain `conversation_history` on the `PipelineContext` as first‑class state.
- Populate conversation turns automatically:
  - Initial user turn from initial input (or explicit init mapping).
  - HITL step outputs count as user turns.
  - AI turns come from agent steps per policy (configurable source).
- Inject conversation into agent prompts of loop body steps by default.
- Allow step‑instance opt‑out from history injection.
- Provide history management strategies: truncate by tokens, truncate by turns, summarize.
- Full prompt/messages observability in lens/tracing after all processors applied (with redaction).

### 3.2 Non‑Functional
- MyPy strict typing compliance and complete test coverage for added logic.
- Minimal overhead (< 5 ms per loop iteration excluding model calls).
- Deterministic behavior on pause/resume and restarts (SQLite backend persistence).
- Centralized configuration via `infra.config_manager` only.

## 4) Architecture Overview

- Policy-driven: Implement entirely in `DefaultLoopStepExecutor` within `flujo/application/core/step_policies.py`.
- Context state: Extend `PipelineContext` with `conversation_history: list[ConversationTurn]`.
- Injection path: Add a prompt processor (ConversationHistoryPromptProcessor) dynamically to step execution when inside a `conversation: true` loop. No mutation of agent definitions; use `AgentProcessors` on the step instance.
- Cost controls: A `HistoryManager` prepares a bounded/summarized history slice per step based on configured limits and target model context.
- Observability: Emit sanitized final prompt/messages via tracing hook; enhance `flujo lens` to render them.
- Persistence: `conversation_history` is pydantic-serializable and stored with the context for pause/resume.

## 5) Data Model

```python
# flujo/domain/models.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class ConversationRole(str, Enum):
    user = "user"
    assistant = "assistant"

class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str

class PipelineContext(BaseModel):
    # existing fields ...
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
```

Notes:
- Roles intentionally use `user|assistant` to match common chat semantics and future API portability.
- Content is `str` only in v1; rich content (images/files) can be added later in a backward-compatible way.

## 6) DSL Additions

```yaml
- kind: loop
  name: clarification_loop
  loop:
    conversation: true
    history_management:
      strategy: truncate_tokens   # truncate_tokens | truncate_turns | summarize
      max_tokens: 4096            # when truncate_tokens
      max_turns: 20               # when truncate_turns
      summarizer_agent: architect_summarizer  # when summarize
      summarize_ratio: 0.5        # optional; proportion of older turns to condense
    ai_turn_source: last          # last | all_agents | named_steps
    user_turn_sources: [hitl]     # hitl | named_steps
    history_template: |
      {{#each history}}
      {{ this.role }}: {{ this.content }}
      {{/each}}
    body:
      - kind: step
        name: clarify
        use_history: true         # step-instance override (default true when conversation: true)
      - kind: step
        name: extract_entities
        use_history: false        # example opt-out for this step only
```

Rules:
- `conversation: true` enables history capture and injection for this loop’s body.
- `use_history` applies per step instance, not globally to agent definitions.
- `ai_turn_source` determines which agent outputs become assistant turns:
  - `last`: final agent step in body (default)
  - `all_agents`: every agent-producing step in body
  - `named_steps`: only steps listed in `named_steps` (array)
- `user_turn_sources` controls which steps contribute user turns:
  - `hitl`: any `HumanInTheLoopStep` output becomes a user turn
  - `named_steps`: only steps explicitly listed

## 7) Execution Semantics (Loop Policy)

In `DefaultLoopStepExecutor`:
- Idempotency: wrap each iteration in `ContextManager.isolate()`. Merge back on success. On `PausedException`, re‑raise after copying safe iteration state into parent context (preserving resume correctness) without poisoning other state.
- Turn updates:
  - First iteration: seed `conversation_history` with initial input when present (configurable via optional mapper, see §9).
  - After a HITL step in the iteration: append a `user` turn.
  - After iteration finishes: append an `assistant` turn based on `ai_turn_source`.
- Quotas: Reserve at iteration start from the loop’s quota; reconcile at end. Inside the body, parallel steps must split quota using branch policy helpers; the loop policy itself remains in single-quota mode.
- Exceptions: Never convert control‑flow exceptions (PausedException, PipelineAbortSignal, InfiniteRedirectError) into data failures. Re‑raise immediately after safe context handling.

## 8) Prompt Injection Path (Processor)

Implement `ConversationHistoryPromptProcessor`:
- Inputs: `context`, step instance config (`use_history`, `history_template`), effective `HistoryManager` (see §10), and model ID (for token estimation).
- Behavior: Before calling the agent, compute the bounded/summarized slice of `context.conversation_history`, render via template (default or custom), and inject as chat messages rather than mutating `system_prompt`.
  - If the underlying agent supports chat messages: prepend rendered turns as message history (`[ {role, content}, ... ]`).
  - If the agent is prompt‑only: prepend a formatted textual history block to the prompt input (clearly delimited), preserving the agent’s system prompt unchanged.
- Safety: Filter tool/function-call artifacts; only natural language content is included by default. Allow opt‑in for structured inclusion in future versions.
- Redaction: Apply `utils.redact` to sensitive data before emitting trace/lens artifacts.

Wiring:
- The loop policy, when `conversation: true`, augments the step instance’s `AgentProcessors.prompt_processors` with this processor for the duration of that step execution only (no mutation of underlying agent definition).

## 9) Initial Input and Custom Seeding (Optional)

Provide an optional loop `init.history` mapper for explicit control:

```yaml
loop:
  conversation: true
  init:
    history:
      start_with:
        from_initial_input: true   # default behavior
        prefix: "User: "           # optional formatting hint for prompt-only models
```

If unspecified, the first non-empty initial input is seeded as the first `user` turn.

## 10) History Management (Cost Control)

Introduce `HistoryManager` with strategies:
- `truncate_tokens`: Use centralized token estimator (existing cost/token utilities) based on the target model context window. Keep most recent turns within `max_tokens`.
- `truncate_turns`: Keep the last `max_turns` turns.
- `summarize`: When the raw history exceeds thresholds, summarize the oldest portion using a configured `summarizer_agent` (created via `flujo.agents` factory) into a compact `assistant` turn and retain the most recent turns. Summarization obeys the same quota rules and is traced.

Implementation details:
- Token estimation is model-aware and routed via centralized config (`infra.config_manager` → `get_settings()` and pricing/model metadata).
- Strategies can be combined via a two‑phase pass: (1) summarize older chunk if present and configured; (2) enforce final token/turn bounds.

## 11) Observability & Lens

- Tracing: Add a step event (e.g., `agent.prompt`) recording the sanitized, fully rendered message set (after processors). Never crash if tracing fails.
- Lens: Enhance `flujo lens trace/show` to display the final rendered prompt/messages for each step when available, with expand/collapse and redaction. Clearly badge “conversation injected” when `conversation: true`.

## 12) Persistence & Resume

- `PipelineContext.conversation_history` is persisted alongside the rest of context (SQLite backend compatibility retained).
- On `PausedException`, loop policy merges safe iteration state to the parent context and re‑raises. Upon resume, the runner restores the context and continues the loop with correct history and iteration counters.

## 13) Configuration

- All configuration is accessed via `flujo.infra.config_manager`:
  - Defaults for history strategy and limits.
  - Model context windows and token pricing.
  - Summarizer agent defaults (if not provided at loop level).
- No direct reads from `flujo.toml` or environment in policies/processors.

## 14) Failure Modes & Edge Cases

- Non‑agent final step: If the last body step is not agent-driven, `ai_turn_source=last` contributes no assistant turn; prefer `named_steps` to specify which agent step contributes.
- Multi‑agent bodies: `ai_turn_source=all_agents` will append an assistant turn after each agent step in order.
- Parallel bodies: Each branch can read from the same bounded history; only the selected `ai_turn_source` steps contribute assistant turns when the iteration completes. Quotas are split at branch executors, not in the loop policy.
- Streaming: Assistant turns are appended only after the agent’s stream completes. If streaming fails mid-way, no partial turn is added.
- Tool/function calls: Omit tool call JSON from history; only natural text content is injected by default.
- Nested loops: `conversation: true` scoped to the innermost loop. Inner loop history does not leak to outer loop unless explicitly merged by the parent pipeline logic.

## 15) Security & Privacy

- Redact secrets and tokens via `utils.redact` before tracing/lens output.
- Bound maximum history size to prevent prompt injection and cost blowups.
- Template rendering is sandboxed; only the provided history fields are exposed.

## 16) Performance Targets

- History preparation + injection adds < 5 ms per iteration average in benchmarks (excluding model calls) on a standard developer machine.
- Summarization work is amortized and only triggered when bounds are exceeded.

## 17) Testing Strategy

Unit (`@pytest.mark.fast`):
- Token and turn truncation correctness.
- Summarization path triggers and output size limits.
- Template rendering default and custom template.
- Step-level `use_history` overrides.

Integration (`@pytest.mark.slow`, `@pytest.mark.serial` where applicable):
- Happy‑path multi‑turn conversation (HITL + agent) with correct turns.
- Pause and resume with SQLite backend; history preserved and correct.
- Multi‑agent body with `ai_turn_source` variants.
- Non‑agent final step behavior.
- Nested `conversation: true` loops isolation.

Benchmark (`@pytest.mark.benchmark`):
- 100-iteration conversational loop measuring overhead of history management + injection.

Lens/Tracing:
- Rendered message sets appear in lens; secrets are redacted.

## 18) Migration & Backward Compatibility

- Feature is off by default; enabling requires `loop.conversation: true`.
- No changes to existing agents or pipelines unless opted in.
- `hitl_history` remains; when `conversation: true`, HITL outputs are also reflected as user turns in `conversation_history`.

## 19) Implementation Plan

1. Data model: add `ConversationTurn` + `conversation_history` to `PipelineContext`.
2. HistoryManager: token/turn truncation; summarization with optional summarizer agent.
3. Prompt processor: `ConversationHistoryPromptProcessor` with chat/prompt modes and redaction.
4. Loop policy wiring: attach processor per step in `conversation: true` loops; manage turn creation per §7.
5. Tracing & lens: add post-processor prompt/messages trace event; lens render support.
6. Configuration plumbing: defaults via `infra.config_manager`.
7. Tests: unit, integration, benchmarks as described.
8. Docs: update `docs/` and CLI wizard hints; add examples.

## 20) Acceptance Criteria

- All new code passes `make all` (format, lint, typecheck, tests) with zero errors.
- Demonstrated e2e pipeline with `conversation: true` where lens shows final injected messages and history evolves correctly across pause/resume.
- Benchmarks show < 5 ms overhead per iteration (excluding model calls).
- Summarization strategy demonstrably bounds long histories without regressions.
- No direct env/config reads from policies/processors; only via config manager APIs.

## 21) Examples

Minimal conversational loop:
```yaml
version: "0.1"
steps:
  - kind: step
    name: get_goal
  - kind: loop
    name: clarification_loop
    loop:
      conversation: true
      history_management:
        strategy: truncate_tokens
        max_tokens: 4096
      ai_turn_source: last
      body:
        - kind: step
          name: clarify
          use_history: true
    
  - kind: step
    name: planner
```

Named sources and custom template:
```yaml
loop:
  conversation: true
  ai_turn_source: named_steps
  named_steps: ["clarify", "finalize"]
  user_turn_sources: [hitl]
  history_management:
    strategy: summarize
    summarizer_agent: convo_summarizer
    summarize_ratio: 0.4
  history_template: |
    <history>
    {{#each history}}
    <turn role="{{ this.role }}">{{ this.content }}</turn>
    {{/each}}
    </history>
```

---

Notes to Implementers
- Keep changes minimal and localized; do not alter `ExecutorCore` semantics.
- Prefer composition using `AgentProcessors` on step instances.
- Ensure strict typing and no leaking of internal objects to public APIs.

## 22) Milestones & Open Tasks

M1. Data Model
- [x] Add `ConversationRole` enum and `ConversationTurn` model.
- [x] Extend `PipelineContext` with `conversation_history: list[ConversationTurn]`.
- [x] Ensure serialization/persistence compatibility (SQLite backend).
- [x] MyPy strict typing for new symbols and references.
- [x] Update docs/comments for new fields.

M2. HistoryManager (Cost Control)
- [x] Integrate model-aware token estimator (best-effort with tiktoken fallback).
- [x] Implement `truncate_tokens` strategy with safe bounds.
- [x] Implement `truncate_turns` strategy.
- [x] Implement `summarize` strategy with two-phase bound enforcement.
- [ ] Wire configuration via `infra.config_manager` (no direct env/file reads) — planned when policy wiring is added.
- [ ] Unit tests for each strategy and combined behavior.

M3. Prompt Processor
- [x] Implement `ConversationHistoryPromptProcessor` with prompt-only injection path.
- [x] Provide prompt-only fallback (text block) without mutating `system_prompt`.
- [ ] Filter tool/function-call artifacts from history by default (stubbed pass-through; refine in M4).
- [x] Integrate `utils.redact` before tracing/lens emission.
- [ ] Unit tests for injection modes, redaction, and filtering.

M4. Loop Policy Wiring
- [ ] Attach prompt processor per step when `loop.conversation: true` and `use_history != false`.
- [ ] Seed initial user turn from initial input or explicit init mapping.
- [ ] Append user turns from HITL outputs.
- [ ] Append assistant turns per `ai_turn_source` (last|all_agents|named_steps).
- [ ] Ensure idempotency via `ContextManager.isolate()` and safe merge.
- [ ] Reserve/reconcile quota per iteration; ensure branch policies split quota in parallel bodies.
- [ ] Re-raise control-flow exceptions after safe state handling (no data-failure conversion).
- [ ] Integration tests for happy path, AI-turn variants, and error propagation.

M5. Tracing & Lens
- [ ] Emit `agent.prompt` trace event with sanitized, final messages post-processor.
- [ ] Add lens rendering for final prompts/messages with expand/collapse and redaction.
- [ ] Tests for lens visibility and redaction behavior.

M6. Persistence & Resume
- [ ] Verify `conversation_history` persists across pause/resume and process restarts.
- [ ] E2E test with SQLite backend and HITL pause/resume.

M7. Configuration & Defaults
- [ ] Define defaults for history strategies and limits in settings via `config_manager`.
- [ ] Allow loop-level overrides to take precedence over global defaults.
- [ ] Update CLI wizard to hint/show `conversation: true` and `history_management` scaffold.
- [ ] Tests for precedence and default resolution.

M8. Performance & Benchmarks
- [ ] Add benchmark test (100-iteration loop) measuring history prep + injection overhead.
- [ ] Validate < 5 ms/iteration overhead (excluding model calls) on reference machine.

M9. Documentation & Samples
- [ ] Update docs with “Conversational Loops” guide and DSL reference.
- [ ] Provide examples under `examples/` showcasing summarization and named sources.
- [ ] Update README and changelog with feature overview and migration notes.

M10. Release Readiness
- [ ] `make all` passes: format, lint, typecheck, unit/integration tests.
- [ ] Security/privacy review for redaction coverage and bounds.
- [ ] Final acceptance demo: lens shows injected history; pause/resume correctness; cost bounded.
