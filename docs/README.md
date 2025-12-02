# Flujo Documentation

This directory contains comprehensive documentation for the Flujo framework.

## Documentation Structure

### **For Framework Architecture Understanding**
- **[flujo.md](https://github.com/aandresalvarez/flujo/blob/main/flujo.md)**: Complete architectural overview of Flujo's design
  - Policy-driven execution core
  - Declarative DSL
  - Architectural patterns for common problems
  - System components and their interactions

### **For End Users (Building with Flujo)**
- **[DEVELOPER_GUIDE.md](https://github.com/aandresalvarez/flujo/blob/main/DEVELOPER_GUIDE.md)**: Practical guide for building applications
  - Best practices and anti-patterns
  - Context management
  - Error handling patterns
  - Systematic debugging methodology
  - Quick reference patterns
- **Debugging & Inspection**
  - **[Lens Quick Start](./guides/lens_quickstart.md)** - Quick reference for debugging with lens CLI ⭐ NEW
  - [Lens Improvements](./guides/lens_improvements.md) - Complete guide to lens v2.0 features
  - [Tracing Guide](./guides/tracing_guide.md) - Comprehensive tracing and debugging
  - [Debugging with Replay](./guides/debugging_with_replay.md) - Deterministic replay for debugging
- **Guides**
  - [Database Backends: SQLite and PostgreSQL](./guides/databases.md) ⭐ NEW
    - When to use SQLite vs PostgreSQL
    - Configuration and setup
    - Migration system and best practices
    - Performance optimization
  - [AROS: Adaptive Reasoning & Output System](./guides/aros.md)
    - Structured outputs via pydantic-ai (wrapper-based)
    - AOP (extraction/repair/coercion) with schema-aware options
    - Reasoning precheck (checklist, validator, consensus, feedback injection)
    - Health-check CLI and telemetry

### **For Core Team (Building Flujo Itself)**
- **[FLUJO_TEAM_GUIDE.md](https://github.com/aandresalvarez/flujo/blob/main/FLUJO_TEAM_GUIDE.md)**: Guide for framework contributors
  - Policy-driven architecture compliance
  - Critical exception handling patterns (PausedException, etc.)
  - Adding new step types and policies
  - Testing patterns for core development
  - Code review checklist
  - Performance optimization using internal systems

## Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| Understand Flujo's architecture | [flujo.md](https://github.com/aandresalvarez/flujo/blob/main/flujo.md) |
| Build an application with Flujo | [DEVELOPER_GUIDE.md](https://github.com/aandresalvarez/flujo/blob/main/DEVELOPER_GUIDE.md) |
| **Configure database backends** | **[Database Backends Guide](./guides/databases.md)** ⭐ NEW |
| **Debug and inspect pipeline runs** | **[Lens Quick Start](./guides/lens_quickstart.md)** ⭐ NEW |
| View execution traces | [Tracing Guide](./guides/tracing_guide.md) |
| Replay a failed run | [Debugging with Replay](./guides/debugging_with_replay.md) |
| Contribute to Flujo framework | [FLUJO_TEAM_GUIDE.md](https://github.com/aandresalvarez/flujo/blob/main/FLUJO_TEAM_GUIDE.md) |
| Debug a control flow issue | [FLUJO_TEAM_GUIDE.md#2-exception-handling-the-architectural-way](https://github.com/aandresalvarez/flujo/blob/main/FLUJO_TEAM_GUIDE.md#2-exception-handling-the-architectural-way) |
| Add a new step type | [FLUJO_TEAM_GUIDE.md#4-adding-new-step-types-the-complete-pattern](https://github.com/aandresalvarez/flujo/blob/main/FLUJO_TEAM_GUIDE.md#4-adding-new-step-types-the-complete-pattern) |

## Key Patterns

### **Control Flow Exception Handling**
The most critical pattern for both users and contributors:

**❌ Never do this:**
```python
try:
    result = await step_execution()
except PausedException as e:
    return StepResult(success=False, error=str(e))  # ❌ Breaks workflow control
```

**✅ Always do this:**
```python
try:
    result = await step_execution()
except PausedException as e:
    # Use Flujo's error classification system
    error_context = ErrorContext.from_exception(e)
    classifier.classify_error(error_context)

    if error_context.category == ErrorCategory.CONTROL_FLOW:
        raise e  # ✅ Re-raise for proper workflow control
```

This pattern ensures that control flow exceptions (like `PausedException` for HITL workflows) properly pause the entire pipeline instead of being converted to failed results.
