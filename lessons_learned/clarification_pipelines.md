# Clarification Pipelines — Lessons Learned, Patterns, and Anti‑Patterns

This document captures practical guidance for building conversational clarification loops in Flujo YAML. It distills what worked, what failed, and why, so future pipelines converge quickly to natural, non‑repetitive behavior.

## Goals
- Ask one good clarifying question per turn.
- Never re‑ask already answered slots.
- Exit cleanly when complete (agent finishes or human says done).
- Preserve useful context across turns without bloating prompts.

---

## Essential Patterns

- Goal anchoring: Always reference the original goal, not the most recent human reply.
  - Use `{{ steps.get_initial_goal.output }}` in agent inputs.

- Embed conversation history into the agent’s input when building YAML.
  - Use HITL transcript for deterministic context:
    ```yaml
    input: |
      --- Conversation So Far ---
      {{#each context.hitl_history}}
      assistant: {{ this.message_to_human }}
      user: {{ this.human_response }}
      {{/each}}

      --- Instruction ---
      Initial goal: {{ steps.get_initial_goal.output }}
      Task: Ask ONE clarifying question that explicitly references the initial goal and conversation so far.
      - Do not ask again for any slot already answered in the conversation above.
      - If all required slots are filled, output {"action":"finish"}.
    ```

- Conversation loop configuration (strong defaults):
  ```yaml
  loop:
    conversation: true
    history_management:
      strategy: truncate_tokens
      max_tokens: 4000
    ai_turn_source: named_steps
    user_turn_sources: ["hitl"]
    named_steps: ["check_if_more_info_needed"]
  ```
  - Rationale: token truncation keeps earlier Q/A when short; named_steps makes assistant questions visible; HITL captures human replies.

- One‑question policy in the agent system prompt:
  ```yaml
  agents:
    clarification_agent:
      system_prompt: |
        You are an expert data analyst. Maintain a running slot map from prior turns; never re‑ask a filled slot. Ask ONE clarifying question that explicitly mentions the initial goal. Only return {"action":"finish"} when all slots are filled or the user says they’re done.
      output_schema:
        type: object
        properties:
          action: { type: string, enum: [ask, finish] }
          question: { type: string }
        required: [action]
  ```

- Exit expression that the safe evaluator accepts:
  ```yaml
  exit_expression: |
    (
      steps['check_if_more_info_needed'] and
      steps['check_if_more_info_needed'].action == 'finish'
    ) or (
      'finish' in steps.get('ask_user_for_clarification', '').lower() or
      'done' in steps.get('ask_user_for_clarification', '').lower() or
      'go ahead' in steps.get('ask_user_for_clarification', '').lower() or
      "that's all" in steps.get('ask_user_for_clarification', '').lower()
    )
  ```
  - Allowed names: `previous_step`, `output`, `context`, `steps`
  - Allowed ops: and/or, not, comparisons, `in`/`not in`, attribute/subscript access, limited calls: `dict.get`, `str.lower|upper|strip|startswith|endswith`
  - Prefer attribute access (`.action`) for Pydantic outputs; `dict.get` works for dict outputs.

- Planner uses clarifications, not just the goal:
  ```yaml
  steps:
    - kind: step
      name: generate_plan
      uses: agents.planner
      input: |
        Goal: {{ steps.get_initial_goal.output }}

        Clarifications:
        {{#each context.hitl_history}}
        Q: {{ this.message_to_human }}
        A: {{ this.human_response }}
        {{/each}}

        Task: Produce a data analysis execution plan (not medical advice).
        - Identify dataset(s) and access assumptions.
        - Define cohort criteria, time window, and filters as specified.
        - Specify metric computation (count/rate) and grouping/dimensions.
        - Outline validation checks, edge cases, and next actions.
        - Keep it concise and actionable.
  ```

---

## Anti‑Patterns (Avoid These)

- Treating the last human reply as the goal.
  - Leads to repeated “time window?” / “metric?” churn. Always anchor to the original goal step.

- Over‑aggressive history truncation (e.g., `truncate_turns: 3`).
  - Drops necessary context; prefer token-based truncation with a reasonable budget.

- Exit expressions with unsupported calls or paths (e.g., `previous_step.output.action.lower().contains('finish')`).
  - Use the safe evaluator’s limited methods, and attribute/subscript access only.

- Not capturing assistant questions in history.
  - Without `ai_turn_source: named_steps` (or a processor), the agent lacks awareness of what it asked.

- Re‑asking filled slots due to vague prompts.
  - Always instruct the agent to maintain a running slot map and skip already answered slots.

- Mixing step‑specific logic into core executor policy.
  - All execution logic must live in policy classes per Flujo’s architectural principles.

---

## Verification Checklist

- Debug exports contain a natural Q/A sequence and no repeated prompts.
- `agent.input` shows the embedded “Conversation So Far” block.
- `conversation_history` has assistant questions (no trailing `action='finish'` artifacts) and user answers.
- `exit_expression` compiles and the loop exits when `action == 'finish'` or the user says “done/finish/go ahead/that’s all”.
- Planner output reflects clarified slots (goal, cohort, time window, metric, grouping, filters).

---

## Example Skeleton

```yaml
version: "0.1"
name: "clarification_conversation"

agents:
  clarification_agent:
    model: "openai:gpt-4o"
    system_prompt: |
      You are an expert data analyst. Maintain a running slot map from prior turns; never re‑ask a filled slot. Ask ONE clarifying question that explicitly mentions the initial goal. Only return {"action":"finish"} when all slots are filled or the user says they’re done.
    output_schema:
      type: object
      properties:
        action: { type: string, enum: [ask, finish] }
        question: { type: string }
      required: [action]

steps:
  - kind: hitl
    name: get_initial_goal
    message: "What would you like to accomplish?"
    updates_context: true

  - kind: loop
    name: clarification_loop
    loop:
      conversation: true
      history_management:
        strategy: truncate_tokens
        max_tokens: 4000
      ai_turn_source: named_steps
      user_turn_sources: ["hitl"]
      named_steps: ["check_if_more_info_needed"]
      body:
        - kind: step
          name: check_if_more_info_needed
          uses: agents.clarification_agent
          input: |
            --- Conversation So Far ---
            {{#each context.hitl_history}}
            assistant: {{ this.message_to_human }}
            user: {{ this.human_response }}
            {{/each}}

            --- Instruction ---
            Initial goal: {{ steps.get_initial_goal.output }}
            Task: Ask ONE clarifying question that explicitly references the initial goal and conversation so far.
            - Do not ask again for any slot already answered above.
            - If all required slots are filled, output {"action":"finish"}.

        - kind: conditional
          name: ask_question_if_needed
          condition_expression: "previous_step.action == 'ask'"
          branches:
            true:
              - kind: hitl
                name: ask_user_for_clarification
                message: "{{ steps.check_if_more_info_needed.output.question }}"
                updates_context: true
            false:
              - kind: step
                name: passthrough
                agent: "flujo.builtins.passthrough"

      exit_expression: |
        (
          steps['check_if_more_info_needed'] and
          steps['check_if_more_info_needed'].action == 'finish'
        ) or (
          'finish' in steps.get('ask_user_for_clarification', '').lower() or
          'done' in steps.get('ask_user_for_clarification', '').lower() or
          'go ahead' in steps.get('ask_user_for_clarification', '').lower() or
          "that's all" in steps.get('ask_user_for_clarification', '').lower()
        )
      max_loops: 4

  - kind: step
    name: generate_plan
    uses: agents.planner
    input: |
      Goal: {{ steps.get_initial_goal.output }}

      Clarifications:
      {{#each context.hitl_history}}
      Q: {{ this.message_to_human }}
      A: {{ this.human_response }}
      {{/each}}

      Task: Produce a data analysis execution plan (not medical advice).
      - Identify dataset(s) and access assumptions.
      - Define cohort criteria, time window, and filters as specified.
      - Specify metric computation (count/rate) and grouping/dimensions.
      - Outline validation checks, edge cases, and next actions.
      - Keep it concise and actionable.
```

---

## Optional Enhancements

- Slot state synthesis: Add a small step to parse `context.hitl_history` into a structured slot map and store it under `context.scratchpad.slots`.
  - Provided helper: `uses: "flujo.helpers.slot_synthesis:synthesize_slots"` with `updates_context: true`.
  - Example:
    ```yaml
    - kind: step
      name: synthesize_slots
      uses: "flujo.helpers.slot_synthesis:synthesize_slots"
      updates_context: true
    ```
  - Access in prompts: `{{ context.scratchpad.slots | tojson }}` or reference fields like `{{ context.scratchpad.slots.metric }}`.
- Summarization strategy: Switch to `strategy: summarize` with a summarizer agent for very long conversations.
- Lens/trace checks: Use debug exports to verify that `agent.input` includes the final rendered prompt with history before shipping.

---

## Engine Behavior Notes

- The loop policy now skips logging assistant turns when the agent output indicates `action == 'finish'`, avoiding noisy artifacts in `conversation_history`.
- Assistant question extraction supports dict and Pydantic outputs (`.action`/`.question`), with safe JSON/string fallbacks.

This guidance should help future YAML pipelines behave naturally, minimize repetition, and exit reliably when done.
