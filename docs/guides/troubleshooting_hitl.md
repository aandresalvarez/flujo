# HITL Troubleshooting Guide

**Status**: Under Development  
**Related**: [Template Variables in Nested Contexts](../user_guide/template_variables_nested_contexts.md)

---

## Common Issues

### Blank HITL Messages

**Symptom**: User sees empty prompt or blank message

**Cause**: Template resolved to empty string due to undefined variable

**Debug Steps**:

1. **Enable strict mode** in `flujo.toml`:
   ```toml
   [template]
   undefined_variables = "strict"
   log_resolution = true
   ```

2. **Run with debug flag**:
   ```bash
   flujo run pipeline.yaml --debug
   ```

3. **Check HITL history** in debug output:
   ```json
   "hitl_history": [{
     "message_to_human": "",  // Empty = template failure
     "human_response": "?",
     "timestamp": "..."
   }]
   ```

4. **Look for template warnings**:
   ```
   [TEMPLATE] WARNING: Template resolved to empty string!
   Original: '{{ context.question }}'
   Available variables: ['context', 'previous_step', 'steps']
   ```

**Solutions**:
- Use `{{ previous_step.field }}` for immediate previous step
- Use `{{ steps.step_name.output.field }}` for named steps
- Use explicit context storage with `sink_to`

See: [Template Variables Guide](../user_guide/template_variables_nested_contexts.md)

---

### HITL in Nested Contexts

**Symptom**: HITL step behaves unexpectedly in loops or conditionals

**Warning**: May see `WARN-HITL-001` during validation

**Context**: HITL in deeply nested structures can have complex pause/resume semantics

**Best Practices**:
1. Prefer top-level HITL steps when possible
2. If HITL must be nested, test pause/resume carefully
3. Consider restructuring to move HITL outside loops/conditionals
4. Use explicit context storage to pass data

**Example Restructuring**:

Instead of:
```yaml
- kind: loop
  body:
    - kind: conditional
      branches:
        true:
          - kind: hitl  # Nested HITL
            message: "Question?"
```

Consider:
```yaml
- kind: loop
  body:
    - kind: conditional
      condition_expression: "previous_step.needs_input"
      branches:
        true:
          - kind: step
            # Flag that input needed
            updates_context: true

- kind: hitl  # Top-level HITL
  message: "Question?"
```

---

### sink_to Not Working

**Symptom**: `sink_to` value not available in context

**Cause**: Usually template/scoping issue, not sink_to failure

**Verify sink_to Works**:
1. Check that step completed successfully
2. Verify value stored at correct path
3. Use correct template pattern to access

**Example**:
```yaml
- kind: hitl
  name: ask_user
  message: "Name?"
  sink_to: "import_artifacts.user_name"

- kind: step
  name: greet
  # ✅ Correct
  input: "Hello {{ context.import_artifacts.user_name }}"
  # ❌ Wrong
  # input: "Hello {{ context.user_name }}"
```

See: [HITL sink_to Documentation](../user_guide/template_variables_nested_contexts.md#sink-to-pattern)

---

### Pipeline Stuck After HITL

**Symptom**: Pipeline paused but won't resume

**Causes**:
1. No call to `resume_async()`
2. Wrong pipeline result passed to resume
3. HITL step ID mismatch

**Debug**:
```python
# Check pipeline status
result = await flujo.run_async(input)
print(f"Status: {result.final_pipeline_context.status}")

# Should show 'paused'
assert result.final_pipeline_context.status == 'paused'

# Resume with user input
resumed = await flujo.resume_async(result, user_input)
```

---

## Enable Debug Logging

For detailed HITL execution logs:

```toml
[template]
log_resolution = true
```

```bash
flujo run pipeline.yaml --debug
```

---

## See Also

- [Template Variables in Nested Contexts](../user_guide/template_variables_nested_contexts.md)
- [Loop Step Scoping](../user_guide/loop_step_scoping.md)
- [Configuration Guide](configuration.md)
