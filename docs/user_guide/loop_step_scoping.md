# Loop Step Scoping Rules

When using steps inside a loop body, step references work differently than at the top level. This guide explains the scoping rules and common patterns.

---

## Scoping Rules

### Top-level (outside loops)

```yaml
steps:
  - kind: step
    name: first
    ...
  
  - kind: step
    name: second
    input: "{{ steps.first.output }}"  # ✅ Works
```

### Inside loop body

```yaml
- kind: loop
  name: my_loop
  loop:
    body:
      - kind: step
        name: process
        ...
      
      - kind: step
        name: check
        # ❌ WRONG - steps['process'] doesn't reference current iteration
        input: "{{ steps.process.output }}"
        
        # ✅ CORRECT - use previous_step for immediate previous step
        input: "{{ previous_step }}"
```

---

## Why This Happens

Loop iterations create **isolated execution contexts**. Each iteration:
1. Executes all body steps in sequence
2. Maintains its own step outputs
3. Does not pollute the parent `steps` dictionary

The `steps` dictionary inside a loop body only contains steps from **outside** the loop.

---

## Access Patterns

| What you want | How to access it | Example |
|---------------|------------------|---------|
| Previous step in loop body | `previous_step` | `{{ previous_step }}` |
| Named step outside loop | `steps['name']` | `{{ steps.init_data.output }}` |
| Step from previous iteration | ❌ Not possible | Use context to carry data |
| Step from current iteration | ❌ Not possible | Use `previous_step` or context |

---

## Common Mistake: Conditional on Loop Body Step

```yaml
# ❌ WRONG - doesn't work as expected
- kind: loop
  loop:
    body:
      - kind: step
        name: decide
        uses: agents.decision_maker
      
      - kind: conditional
        name: handle_decision
        condition_expression: "steps['decide'].output.action == 'continue'"  # ❌
        branches:
          true: [...]
```

**Why it fails:**
- `steps['decide']` tries to access a step in the current iteration
- Loop scoping prevents this reference from working
- Conditional never evaluates correctly
- Loop hits max_loops

**Fix:**
```yaml
# ✅ CORRECT - use previous_step
- kind: loop
  loop:
    body:
      - kind: step
        name: decide
        uses: agents.decision_maker
      
      - kind: conditional
        name: handle_decision
        condition_expression: "previous_step.action == 'continue'"  # ✅
        branches:
          true: [...]
```

---

## Carrying Data Between Iterations

If you need data from one iteration in the next, use **context**:

```yaml
- kind: loop
  name: accumulate
  loop:
    body:
      - kind: step
        name: process
        uses: agents.processor
        updates_context: true  # ← Stores output in context
      
      - kind: step
        name: next_step
        input: |
          Previous results: {{ context.accumulated_results }}
          Current result: {{ previous_step }}
    
    propagation:
      next_input: context  # ← Pass context to next iteration
```

---

## Exit Conditions

Use `previous_step` or `context` in exit expressions:

```yaml
loop:
  body:
    - kind: step
      name: check_done
      uses: agents.checker
  
  # ✅ Reference the step that just ran
  exit_expression: "previous_step.done == true"
  
  # ✅ Or use context
  exit_expression: "context.status == 'complete'"
```

---

## Debug Tips

If your loop isn't working:

1. **Check step references**
   - Inside loop body? Use `previous_step`
   - Outside loop? Use `steps['name']`

2. **Check exit condition**
   - Does it reference the correct step?
   - Is the step output what you expect?

3. **Use debug mode**
   ```bash
   flujo run --debug
   ```
   Look for which steps execute and their outputs.

4. **Check lens trace**
   ```bash
   flujo lens show <run_id> --verbose
   ```
   Verify conditional branches are being taken.

---

## Validation Warnings

### Validation Rule LOOP-001

Step reference in loop body (`steps['name']`) is invalid because loop iterations are isolated.

**Warning message:**
```
Warning [LOOP-001]: Step reference detected in condition_expression inside loop body 'my_loop'.
Loop body steps are scoped to the current iteration and may not be accessible via steps['name'].
```

**Fix:**
```yaml
# ❌ Inside loop body
condition_expression: "steps['process'].output.status == 'done'"

# ✅ Use previous_step instead
condition_expression: "previous_step.status == 'done'"
```

**When you see this warning:**
- The loop may hit max_loops without making progress
- Conditionals may never evaluate correctly
- Exit conditions may never trigger

**Action:** Change `steps['name']` to `previous_step` or use context.

---

## Complete Example

```yaml
version: "0.1"
name: "Loop Scoping Example"

agents:
  processor:
    model: "openai:gpt-4o"
    system_prompt: "Process the input and return status."
    output_schema:
      type: object
      properties:
        status: { type: string, enum: ["continue", "done"] }
        result: { type: string }

  analyzer:
    model: "openai:gpt-4o"
    system_prompt: "Analyze the result."
    output_schema:
      type: object
      properties:
        analysis: { type: string }

steps:
  # Initialize context
  - kind: step
    name: init
    agent:
      id: "flujo.builtins.passthrough"
    input: '{"accumulated": []}'
    updates_context: true
  
  # Loop with proper scoping
  - kind: loop
    name: processing_loop
    loop:
      body:
        # Process data
        - kind: step
          name: process
          uses: agents.processor
          input: "{{ context.current_input }}"
        
        # ✅ CORRECT: Use previous_step for conditional
        - kind: conditional
          name: check_status
          condition_expression: "previous_step.status == 'done'"
          branches:
            true:
              - kind: step
                name: finalize
                agent:
                  id: "flujo.builtins.passthrough"
                input: "Completed"
            
            false:
              # ✅ CORRECT: Use previous_step in template
              - kind: step
                name: continue_processing
                agent:
                  id: "flujo.builtins.passthrough"
                input: "{{ previous_step.result }}"
      
      propagation:
        next_input: context
      
      # ✅ CORRECT: Use previous_step in exit expression
      exit_expression: "previous_step.status == 'done'"
      max_loops: 5
```

---

## See Also

- [Template System Reference](./template_system_reference.md) - Template syntax and limitations
- [Loop Step Documentation](../advanced/loop_step.md) - Full loop step reference
- [Expression Language](../expression_language.md) - Expression syntax for conditions
