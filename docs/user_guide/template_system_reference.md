# Flujo Template System Reference

Flujo uses a **simplified subset of Jinja2** for templates. Only expressions and filters are supported.

---

## Supported Syntax

### ✅ Expressions
Access variables using `{{ }}`:

```yaml
input: "{{ context.user_name }}"
input: "{{ previous_step }}"
input: "{{ steps.analyze.output }}"
```

### ✅ Filters
Transform values using `|`:

```yaml
input: "{{ context.items | join(', ') }}"
input: "{{ previous_step | upper }}"
input: "{{ context.data | tojson }}"
```

**Available filters:**
- `join(separator)` - Join array elements
- `upper` - Convert to uppercase
- `lower` - Convert to lowercase
- `tojson` - Serialize as JSON
- `length` - Get length of array/string

### ✅ Nested Access
Access nested fields:

```yaml
input: "{{ context.user.profile.name }}"
input: "{{ steps.analyze.output.results[0] }}"
```

### ✅ Boolean Logic (in expressions only)
Use in `condition_expression` and `exit_expression`:

```yaml
condition_expression: "context.status == 'ready' and context.count > 0"
exit_expression: "previous_step.action == 'finish' or context.done"
```

---

## NOT Supported

### ❌ Control Structures
Jinja2 control flow syntax is **not supported**:

```yaml
# ❌ DOES NOT WORK
input: |
  {% for item in context.items %}
  - {{ item }}
  {% endfor %}

# ❌ DOES NOT WORK
input: |
  {% if context.ready %}
  Ready!
  {% else %}
  Not ready
  {% endif %}
```

**Why?** Control structures add complexity, security risks, and performance overhead. Flujo keeps templates simple and predictable.

**Validation:** Flujo will show an **ERROR** (rule `TEMPLATE-001`) if you use `{% %}` in templates.

### ❌ Macros, Includes, Imports

```yaml
# ❌ DOES NOT WORK
{% macro render_item(item) %}
  ...
{% endmacro %}

# ❌ DOES NOT WORK
{% include "template.jinja2" %}
```

---

## Alternative Patterns

### For Loops → Use Filters or Custom Skills

**Option 1: Use `join` filter**
```yaml
# Instead of {% for %}
input: "{{ context.items | join('\n- ') }}"
```

**Option 2: Use custom skill**
```python
# skills/formatters.py
async def format_list(items: list) -> str:
    return "\n".join(f"- {item}" for item in items)
```

```yaml
- kind: step
  name: format_items
  uses: "skills.formatters:format_list"
  input: "{{ context.items }}"
```

### For Conditionals → Use conditional steps

```yaml
# Instead of {% if %} in template
- kind: conditional
  name: check_status
  condition_expression: "context.ready"
  branches:
    true:
      - kind: step
        uses: agents.ready_handler
        input: "Ready!"
    false:
      - kind: step
        uses: agents.not_ready_handler
        input: "Not ready"
```

---

## Common Mistakes

### Mistake 1: Copying Jinja2 Examples
```yaml
# ❌ Found in Jinja2 docs, doesn't work in Flujo
input: |
  {% set items = context.data %}
  {% for item in items %}
  {{ item }}
  {% endfor %}

# ✅ Use filters instead
input: "{{ context.data | join('\n') }}"
```

### Mistake 2: Complex Logic in Templates
```yaml
# ❌ Too complex for templates
input: |
  {% if context.count > 10 %}
    Many items
  {% else %}
    Few items
  {% endif %}

# ✅ Use conditional step
- kind: conditional
  condition_expression: "context.count > 10"
  branches:
    true:
      - kind: step
        input: "Many items"
    false:
      - kind: step
        input: "Few items"
```

---

## Summary

| Feature | Supported | Alternative |
|---------|-----------|-------------|
| `{{ expression }}` | ✅ Yes | - |
| `{{ value \| filter }}` | ✅ Yes | - |
| `{% for %}` | ❌ No | Use filters or custom skills |
| `{% if %}` | ❌ No | Use conditional steps |
| `{% macro %}` | ❌ No | Use custom skills |
| `{% set %}` | ❌ No | Use step output |

**Rule of thumb:** If it uses `{% %}`, it won't work in Flujo templates.

---

## Validation Errors

### TEMPLATE-001: Unsupported Jinja2 Control Structure

**Error message:**
```text
Error [TEMPLATE-001]: Unsupported Jinja2 control structure '{%for%}' detected in input.
Flujo templates support expressions {{ }} and filters |, but NOT control structures {%for%}.
```

**Fix:**
- Use template filters like `{{ items | join }}` 
- Use custom skills for complex formatting
- Use conditional steps for if/else logic
- Pre-format data in a previous step

See alternatives above for specific patterns.

---

## See Also

- [Expression Language](../expression_language.md) - Detailed expression syntax reference
- [Loop Step Scoping](./loop_step_scoping.md) - Understanding step references in loops
- [Creating YAML](../creating_yaml.md) - General YAML pipeline syntax

