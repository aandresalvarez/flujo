# Template Variables in Nested Contexts

**Last Updated**: October 3, 2025  
**Version**: Flujo 0.4.38+

---

## Overview

When using template variables (like `{{ context.field }}` or `{{ previous_step.output }}`) inside nested contexts (loops, conditionals, parallel branches), special scoping rules apply. This guide explains those rules and shows correct patterns to avoid silent failures.

---

## Quick Reference

| What You Want | Correct Pattern | Example |
|---------------|----------------|---------|
| **Previous step output** | `{{ previous_step.field }}` | `{{ previous_step.question }}` |
| **Named step output** | `{{ steps.step_name.output.field }}` | `{{ steps.clarify.output.question }}` |
| **Explicit context field** | `{{ context.import_artifacts.field }}` | `{{ context.import_artifacts.user_goal }}` |

---

## The Problem: Silent Template Failures

### What Goes Wrong

When HITL steps or agent steps use templates with undefined variables, the behavior depends on your configuration:

**Default Behavior** (`undefined_variables = "warn"`):
- Template resolves to **empty string**
- Warning logged to debug output
- Pipeline continues (backward compatible)

**Strict Mode** (`undefined_variables = "strict"`):
- Raises `TemplateResolutionError` immediately
- Shows available variables
- Pipeline fails fast with clear error

**Example of the Bug:**

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: agent
        uses: agents.my_agent
        # Outputs: {"action": "ask", "question": "What is X?"}
      
      - kind: conditional
        condition_expression: "previous_step.action == 'ask'"
        branches:
          true:
            - kind: hitl
              name: ask_user
              message: "{{ context.question }}"  # ❌ UNDEFINED!
```

**What happens:**
- Agent outputs: `{"action": "ask", "question": "What is X?"}`
- Template tries to access: `context.question` (doesn't exist!)
- **Default mode**: Empty message `""` → User sees blank prompt
- **Strict mode**: `TemplateResolutionError` → Pipeline fails with clear error

---

## Configuration: Enable Strict Mode

To catch template errors early, enable strict mode in `flujo.toml`:

```toml
[template]
undefined_variables = "strict"  # Raise error on undefined variables
log_resolution = true            # Log template resolution for debugging
```

**Options for `undefined_variables`:**
- `"strict"` - Raise `TemplateResolutionError` (recommended for development)
- `"warn"` - Log warning, return empty string (default, backward compatible)
- `"ignore"` - Silent, return empty string (not recommended)

**Environment variable override:**
```bash
export FLUJO_TEMPLATE_UNDEFINED="strict"
uv run flujo run pipeline.yaml
```

---

## Correct Patterns

### Pattern 1: Use `previous_step` (Recommended)

**When to use**: Reference the immediate previous step in the same scope.

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: decide
        uses: agents.decision_maker
        # Outputs: {"action": "ask", "question": "What next?"}
      
      - kind: hitl
        name: ask_user
        message: "{{ previous_step.question }}"  # ✅ CORRECT
```

**Pros:**
- Simple and intuitive
- Always works in nested contexts
- Recommended for most cases

**Cons:**
- Only works for immediate previous step
- Can't reference steps from earlier in the pipeline

---

### Pattern 2: Use Named Step References

**When to use**: Reference any named step from anywhere in the pipeline.

```yaml
- kind: step
  name: initialize
  agent: { id: "flujo.builtins.passthrough" }
  input: '{"goal": "analyze data"}'
  updates_context: true

- kind: loop
  loop:
    body:
      - kind: step
        name: agent
        uses: agents.analyzer
      
      - kind: conditional
        branches:
          true:
            - kind: hitl
              # ✅ Reference step from outside the loop
              message: "Goal: {{ steps.initialize.output.goal }}, Continue?"
```

**Pros:**
- Can reference any named step
- Explicit and clear
- Works across nesting levels

**Cons:**
- More verbose
- Requires steps to have names

---

### Pattern 3: Explicit Context Storage

**When to use**: Need to carry data explicitly through context.

```yaml
- kind: step
  name: agent
  uses: agents.my_agent
  updates_context: true
  sink_to: "import_artifacts.current_question"  # ✅ Explicit storage
  # Stores output at context.import_artifacts.current_question

- kind: conditional
  branches:
    true:
      - kind: hitl
        message: "{{ context.import_artifacts.current_question }}"  # ✅ Works!
```

**Pros:**
- Full control over context structure
- Clear data flow
- Good for complex pipelines

**Cons:**
- Most verbose
- Requires understanding context structure

---

### Sink-to Pattern

Use `sink_to` when a step returns a scalar or small object that you want stored at a predictable
context path without writing a custom processor.

```yaml
- kind: hitl
  name: ask_user
  message: "Name?"
  sink_to: "import_artifacts.user_name"

- kind: step
  name: greet
  input: "Hello {{ context.import_artifacts.user_name }}"
```

- Paths are dotted (e.g., `import_artifacts.user_name`) and are created automatically under `import_artifacts`.
- Works alongside `updates_context: true`; `sink_to` just controls the destination.
- Useful for HITL prompts or guardrail steps that capture simple fields.

---

## Common Mistakes

### ❌ Mistake 1: Direct Context Field Access

```yaml
- kind: step
  name: agent
  uses: agents.my_agent
  updates_context: true
  # Outputs: {"question": "What is X?"}

- kind: hitl
  message: "{{ context.question }}"  # ❌ WRONG! Field doesn't exist
```

**Why it fails:** With `updates_context: true`, the output is recorded under `steps.agent.output.question` (in `context.step_outputs`), NOT `context.question`.

**Fix:** Use `{{ previous_step.question }}` or `{{ steps.agent.output.question }}`

---

### ❌ Mistake 2: Step References in Loop Bodies

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: process
        uses: agents.processor
      
      - kind: conditional
        condition_expression: "steps['process'].output.done == true"  # ❌ WRONG!
```

**Why it fails:** Loop body steps are scoped to the current iteration. `steps['process']` doesn't reference the step in the current iteration.

**Fix:** Use `previous_step.done == true` (see [Loop Scoping Guide](loop_step_scoping.md))

---

### ❌ Mistake 3: Assuming Agent Output Structure

```yaml
- kind: step
  name: agent
  uses: agents.my_agent
  # Agent outputs: "Just a string"

- kind: hitl
  message: "{{ previous_step.question }}"  # ❌ WRONG! No .question field
```

**Why it fails:** If agent outputs a string, not a dict, there's no `.question` field.

**Fix in strict mode:** Error shows: `Undefined template variable: 'previous_step.question'`

**Fix:** Check agent output structure, use `{{ previous_step }}` for whole output

---

## Debugging Template Issues

### Symptom: HITL Shows Blank Message

**Cause:** Template resolved to empty string due to undefined variable.

**Debug steps:**

1. **Run with debug mode:**
   ```bash
   uv run flujo run pipeline.yaml --debug
   ```

2. **Check `hitl_history` in debug output:**
   ```json
   "hitl_history": [
     {
       "message_to_human": "",  // ← Empty! Template failed
       "human_response": "?",
       "timestamp": "..."
     }
   ]
   ```

3. **Enable template logging in `flujo.toml`:**
   ```toml
   [template]
   log_resolution = true
   ```

4. **Look for warnings in logs:**
   ```
   [TEMPLATE] WARNING: Template resolved to empty string!
   Original: '{{ context.question }}'
   Available variables: ['context', 'previous_step', 'steps']
   ```

5. **Enable strict mode to get clear errors:**
   ```toml
   [template]
   undefined_variables = "strict"
   ```
   
   Run again and see:
   ```
   TemplateResolutionError: Undefined template variable: 'context.question'
   Available variables: ['context', 'previous_step', 'steps']
   Suggestion: Use '{{ previous_step.question }}' or '{{ steps.agent.output.question }}'
   ```

---

### Symptom: Pipeline Fails with TemplateResolutionError

**Cause:** Strict mode is enabled and template references undefined variable.

**Solution:**

1. **Read the error message** - it shows available variables
2. **Use one of the correct patterns** above
3. **Test with `--debug` flag** to see template resolution

---

## Best Practices

### ✅ DO: Use `previous_step` for Simple Cases

```yaml
- kind: step
  name: agent
  uses: agents.my_agent

- kind: hitl
  message: "{{ previous_step.question }}"  # ✅ Simple and clear
```

### ✅ DO: Use Named References for Clarity

```yaml
- kind: step
  name: initialize
  ...

- kind: loop
  loop:
    body:
      - kind: hitl
        message: "Goal: {{ steps.initialize.output.goal }}"  # ✅ Explicit
```

### ✅ DO: Enable Strict Mode in Development

```toml
# flujo.toml (development)
[template]
undefined_variables = "strict"
log_resolution = true
```

### ✅ DO: Use Warn Mode in Production

```toml
# flujo.toml (production)
[template]
undefined_variables = "warn"  # Log warnings but don't break
log_resolution = false          # Reduce log noise
```

---

### ❌ DON'T: Assume Context Structure

```yaml
# ❌ WRONG - Assuming where data is stored
message: "{{ context.some_field }}"

# ✅ CORRECT - Explicit reference
message: "{{ previous_step.some_field }}"
```

### ❌ DON'T: Use Complex Expressions in Templates

```yaml
# ❌ WRONG - Complex logic in template
message: "{% if context.ready %}Ready{% else %}Not ready{% endif %}"

# ✅ CORRECT - Use conditional steps
- kind: conditional
  condition_expression: "context.ready"
  branches:
    true:
      - kind: hitl
        message: "Ready"
    false:
      - kind: hitl
        message: "Not ready"
```

---

## Examples

### Example 1: Conversational Loop

```yaml
- kind: loop
  name: conversation
  loop:
    body:
      # Agent asks questions or finishes
      - kind: step
        name: agent
        uses: agents.conversational_agent
        # Outputs: {"action": "ask"|"finish", "question": "..."}
      
      # Check if we need user input
      - kind: conditional
        condition_expression: "previous_step.action == 'ask'"
        branches:
          true:
            # ✅ Use previous_step to get question
            - kind: hitl
              name: ask_user
              message: "{{ previous_step.question }}"
              sink_to: "import_artifacts.user_response"
    
    exit_expression: "previous_step.action == 'finish'"
    max_loops: 10
```

### Example 2: Multi-Step Data Collection

```yaml
# Store initial goal explicitly
- kind: hitl
  name: get_goal
  message: "What is your goal?"
  sink_to: "import_artifacts.initial_goal"

# Use it in nested contexts
- kind: loop
  loop:
    body:
      - kind: step
        name: clarify
        uses: agents.clarifier
        # Template input references stored goal
        input: "Goal: {{ context.import_artifacts.initial_goal }}, ask clarification"
      
      - kind: conditional
        branches:
          true:
            - kind: hitl
              # ✅ Reference both stored context and current step
              message: |
                Goal: {{ context.import_artifacts.initial_goal }}
                Question: {{ previous_step.question }}
```

### Example 3: Error Recovery

```yaml
- kind: step
  name: agent
  uses: agents.my_agent
  fallback:
    - agent: { id: "flujo.builtins.passthrough" }
      input: '{"action": "error", "message": "Agent failed"}'

- kind: conditional
  condition_expression: "previous_step.action == 'error'"
  branches:
    true:
      # ✅ Handle fallback case
      - kind: hitl
        message: "Error: {{ previous_step.message }}"
```

---

## Related Documentation

- [Template System Reference](template_system_reference.md) - Full template syntax guide
- [Loop Step Scoping](loop_step_scoping.md) - Step references in loops
- [Configuration Guide](../guides/configuration.md) - How to configure Flujo
- [Troubleshooting HITL](../guides/troubleshooting_hitl.md) - HITL-specific debugging

---

## Summary

**Key Takeaways:**

1. ✅ **Use `{{ previous_step.field }}`** for simple cases
2. ✅ **Use `{{ steps.name.output.field }}`** for explicit references
3. ✅ **Enable strict mode in development** to catch errors early
4. ✅ **Check available variables** in error messages
5. ❌ **Don't assume `context.field` exists** without explicit storage

**Configuration:**
```toml
[template]
undefined_variables = "strict"  # Catch errors early
log_resolution = true            # See what's happening
```

**When in doubt:** Use `{{ previous_step }}` - it's the most reliable pattern in nested contexts.

---

**Questions or issues?** Check the [troubleshooting guide](../guides/troubleshooting_hitl.md) or enable strict mode to get clear error messages.
