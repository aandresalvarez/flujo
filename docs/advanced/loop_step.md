# Loop Step Reference

**Status**: Under Development  
**Related**: [Loop Step Scoping Guide](../user_guide/loop_step_scoping.md)

---

## Overview

The `LoopStep` executes a sequence of steps repeatedly until an exit condition is met or maximum iterations reached.

## Basic Structure

```yaml
- kind: loop
  name: my_loop
  loop:
    body:
      - kind: step
        # ... steps to repeat
    
    exit_expression: "context.done == true"
    max_loops: 10
```

## Loop Configuration

### Body

List of steps to execute in each iteration.

**Scoping Rules**:
- Each iteration has isolated step history
- Use `previous_step` to reference immediate previous step
- `steps['name']` only accesses steps OUTSIDE the loop

See: [Loop Step Scoping Guide](../user_guide/loop_step_scoping.md)

### Exit Expression

Python expression evaluated after each iteration.

**Correct Patterns**:
```yaml
# ✅ Use previous_step
exit_expression: "previous_step.done == true"

# ✅ Use context
exit_expression: "context.status == 'complete'"

# ❌ Don't use steps['name'] from loop body
# exit_expression: "steps['check'].done == true"
```

### Max Loops

Maximum iterations before forcing exit.

```yaml
max_loops: 10  # Prevents infinite loops
```

**Default**: 100

---

## Context Propagation

### Carrying Data Between Iterations

Use `updates_context: true` to persist data:

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: accumulate
        uses: agents.processor
        updates_context: true  # ← Data available in next iteration
```

### Accessing Previous Iteration Data

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: process
        # Access data from previous iterations via context
        input: "Previous results: {{ context.results }}"
```

---

## Common Patterns

### Counter Loop

```yaml
- kind: loop
  name: counter
  loop:
    body:
      - kind: step
        name: increment
        agent: { id: "flujo.builtins.passthrough" }
        input: "{{ (context.count or 0) + 1 }}"
        updates_context: true
        sink_to: "count"
    
    exit_expression: "context.count >= 5"
    max_loops: 10
```

### Conditional Exit

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: check_status
        uses: agents.status_checker
        updates_context: true
    
    exit_expression: "previous_step.status == 'done'"
    max_loops: 100
```

### Accumulator Loop

```yaml
- kind: loop
  loop:
    body:
      - kind: step
        name: process_item
        uses: agents.processor
        updates_context: true
      
      - kind: step
        name: store_result
        agent: { id: "flujo.builtins.passthrough" }
        input: "{{ (context.results or []) + [previous_step] }}"
        updates_context: true
        sink_to: "results"
    
    exit_expression: "len(context.results or []) >= 10"
    max_loops: 20
```

---

## Troubleshooting

### Loop Never Exits

**Cause**: Exit expression never evaluates to `true`

**Debug**:
1. Check `exit_expression` syntax
2. Verify referenced variables exist
3. Add logging to see expression evaluation

```yaml
exit_expression: "previous_step.action == 'finish'"
# Check that previous_step actually has 'action' field
```

### Hit max_loops

**Cause**: Exit condition not met before max iterations

**Solutions**:
1. Review exit logic
2. Increase `max_loops` if legitimate
3. Add fallback logic in loop body

---

### Steps Not Accessible

**Symptom**: `steps['name']` doesn't work in loop

**Cause**: Loop scoping - steps inside loop aren't in `steps` dict

**Solution**: Use `previous_step` or `context`

See: [LOOP-001 Validation Rule](../user_guide/loop_step_scoping.md#validation-rule-loop-001)

---

## Best Practices

1. **Always set max_loops** - Prevent infinite loops
2. **Use clear exit conditions** - Make termination obvious
3. **Persist data via context** - Use `updates_context: true`
4. **Test edge cases** - What if loop exits immediately?
5. **Consider MapStep** - For parallel iteration over collections

---

## See Also

- [Loop Step Scoping Guide](../user_guide/loop_step_scoping.md)
- [MapStep Design](../map_step_design.md)
- [Pipeline Context](../user_guide/pipeline_context.md)
- [Template Variables](../user_guide/template_variables_nested_contexts.md)

