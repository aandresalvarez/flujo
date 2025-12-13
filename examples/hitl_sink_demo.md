# HITL Sink Demo - Automatic Context Storage

This demo shows how the new `sink_to` feature eliminates boilerplate passthrough steps.

## 🎯 What It Does

Automatically stores HITL responses to context without manual steps.

## 📝 Before (Old Way - 6 steps)

```yaml
steps:
  - kind: hitl
    name: get_user_name
    message: "What's your name?"
  
  # ❌ Manual storage step needed
  - kind: step
    name: store_name
    agent: { id: "flujo.builtins.passthrough" }
    input: "{{ previous_step }}"
    updates_context: true
  
  - kind: hitl
    name: get_preferences
    message: "Preferences?"
  
  # ❌ Another manual storage step
  - kind: step
    name: store_prefs
    agent: { id: "flujo.builtins.passthrough" }
    input: "{{ previous_step }}"
    updates_context: true
  
  - kind: step
    name: use_data
    input: "Name: {{ context.import_artifacts.store_name }}"
```

## ✨ After (New Way - 3 steps)

```yaml
steps:
  - kind: hitl
    name: get_user_name
    message: "What's your name?"
    sink_to: "import_artifacts.user_name"  # ✅ Automatic!
  
  - kind: hitl
    name: get_preferences
    message: "Preferences?"
    sink_to: "import_artifacts.user_preferences"  # ✅ Automatic!
  
  - kind: step
    name: use_data
    input: "Name: {{ context.import_artifacts.user_name }}"
```

**Result**: 50% fewer steps, clearer intent! 🚀

## 🧪 Running the Demo

```bash
# Run with HITL interaction
uv run flujo run examples/hitl_sink_demo.yaml

# Example interaction:
# Q: 👋 Welcome! What's your name?
# A: Alice
#
# Q: What are your preferences?
# A: {"theme": "dark", "notifications": true}
#
# Output: ✅ Greeting created: Hello Alice! I love that you prefer dark mode...
```

## 🎓 Key Features

1. **Nested paths**: `sink_to: "import_artifacts.settings.user.name"`
2. **Structured input**: Works with `input_schema`
3. **Graceful errors**: Invalid paths warn but don't crash
4. **Backward compatible**: Existing pipelines unaffected

## 📚 See Also

- Full docs: `docs/hitl.md`
- Integration tests: `tests/integration/test_hitl_sink_to.py`
- Implementation: Task 2.1 in `FSD.md`
