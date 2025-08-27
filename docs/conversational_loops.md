# Conversational Loops (conversation: true)

This guide explains how to enable and control conversational behavior within a loop body in Flujo.

Core concepts
- Scoping: `conversation: true` applies only to the loop’s body.
- History: `PipelineContext.conversation_history` stores ordered turns (`user|assistant`), persisted to state backends.
- Injection: History is injected via a prompt processor attached to steps at runtime — agents themselves are not mutated.
- Governance: Token/turn bounds and summarization ensure predictable costs.
- Observability: A sanitized preview of the rendered history is emitted as an `agent.prompt` event in the trace tree.

DSL fields
- loop.conversation: boolean
  - Enables conversational mode for that loop.
- loop.history_management: object
  - strategy: `truncate_tokens` | `truncate_turns` | `summarize`
  - max_tokens: integer (when `truncate_tokens`)
  - max_turns: integer (when `truncate_turns`)
  - summarize_ratio: float 0..1 (portion of older turns to summarize)
- loop.history_template: string (optional)
  - Custom template to render history into the prompt.
- step.use_history: boolean
  - Per-step opt-out/in for injection on that step instance.
- loop.ai_turn_source: string
  - `last` (default): assistant turn from last step output.
  - `all_agents`: assistant turns from outputs of every non-complex agent step in the body.
  - `named_steps`: assistant turns only from steps listed in `loop.named_steps`.
- loop.user_turn_sources: list[string]
  - Accepts `'hitl'` and/or specific step names to append user turns from.
- loop.named_steps: list[string]
  - Used with `ai_turn_source: named_steps` to select assistant-turn steps.

Examples

Minimal
```yaml
- kind: loop
  name: clarify
  loop:
    conversation: true
    history_management:
      strategy: truncate_tokens
      max_tokens: 4096
    body:
      - kind: step
        name: clarify
```

Named sources and custom template
```yaml
- kind: loop
  name: guided
  loop:
    conversation: true
    ai_turn_source: named_steps
    named_steps: ["planner"]
    user_turn_sources: [hitl]
    history_management:
      strategy: summarize
      summarize_ratio: 0.4
    history_template: |
      <history>
      {{#each history}}
      <turn role="{{ this.role }}">{{ this.content }}</turn>
      {{/each}}
      </history>
    body:
      - kind: step
        name: clarify
      - kind: hitl
        name: ask_user
      - kind: step
        name: planner
```

All agents as assistant sources
```yaml
- kind: loop
  name: multi
  loop:
    conversation: true
    ai_turn_source: all_agents
    body:
      - kind: step
        name: a1
      - kind: step
        name: a2
```

Behavior
- Seeding: On iteration 1, if history is empty, the loop seeds a `user` turn from the iteration input.
- HITL: `user_turn_sources` includes `'hitl'` by default; HITL outputs contribute `user` turns when listed.
- Assistant turns: Selected per `ai_turn_source` after each iteration.
- Parallel branches: Loop policy preserves existing quota and isolation semantics. Assistant/user turn collection is based on top-level step results; complex nested branches may require tailored selection rules.

Notes
- Centralized configuration can provide defaults for `history_management`; loop-level values override defaults.
- Prompt tracing: The injected history preview is redacted and truncated before emitting to the trace tree.

