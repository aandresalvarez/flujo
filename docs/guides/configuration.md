# Configuration Guide

**Status**: Under Development  
**Related**: [Template Variables in Nested Contexts](../user_guide/template_variables_nested_contexts.md)

---

## Overview

This guide covers Flujo configuration options, including the new template resolution settings.

## Template Configuration

Configure template behavior in `flujo.toml`:

```toml
[template]
# Control undefined variable handling
undefined_variables = "strict"  # Options: "strict", "warn", "ignore"

# Enable template resolution logging
log_resolution = true  # Options: true, false
```

### Undefined Variables Setting

**`strict`** (Recommended for Development):
- Raises `TemplateResolutionError` on undefined variables
- Shows available variables in error message
- Provides suggestions for correct patterns
- Fails fast with clear feedback

**`warn`** (Default - Production Safe):
- Logs warning when undefined variables encountered
- Returns empty string (backward compatible)
- Pipeline continues execution
- Warnings visible in debug logs

**`ignore`** (Not Recommended):
- Silent fallback to empty string
- No warnings or errors
- Can lead to silent failures
- Use only if you have a specific reason

### Log Resolution Setting

**`true`**:
- Logs template rendering process
- Shows available variables at render time
- Warns when templates resolve to empty strings
- Useful for debugging template issues

**`false`** (Default):
- Minimal logging
- Better performance
- Recommended for production

## Environment Variables

Override configuration via environment:

```bash
export FLUJO_TEMPLATE_UNDEFINED="strict"
export FLUJO_TEMPLATE_LOG_RESOLUTION="true"
```

## Example Configurations

### Development Setup
```toml
[template]
undefined_variables = "strict"
log_resolution = true
```

### Production Setup
```toml
[template]
undefined_variables = "warn"
log_resolution = false
```

## See Also

- [Template Variables in Nested Contexts](../user_guide/template_variables_nested_contexts.md)
- [Template System Reference](../user_guide/template_system_reference.md)
- [Loop Step Scoping](../user_guide/loop_step_scoping.md)

