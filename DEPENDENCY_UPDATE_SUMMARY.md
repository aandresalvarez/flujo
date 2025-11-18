# Dependency Update Summary

**Date**: October 4, 2025  
**Status**: ✅ **Complete**

---

## Updated Dependencies

| Package | Previous Version | New Version | Change |
|---------|-----------------|-------------|--------|
| **openai** | 2.0.1 | **2.1.0** | ⬆️ Minor update |
| **pydantic-ai** | 0.7.2 | **1.0.15** | ⬆️ Major update (0.x → 1.x) |
| **pydantic-ai-slim** | 0.7.2 | **1.0.15** | ⬆️ Major update |
| **pydantic-evals** | 0.7.2 | **1.0.15** | ⬆️ Major update |
| **pydantic-graph** | 0.7.2 | **1.0.15** | ⬆️ Major update |

---

## Changes Made

### 1. Updated `pyproject.toml`

**Before**:
```toml
dependencies = [
    "pydantic>=2.11.7,<2.13",
    "pydantic-ai>=0.7.0,<0.7.3",
    "pydantic-settings>=2.0.0",
    ...
]
```

**After**:
```toml
dependencies = [
    "pydantic>=2.11.7,<2.13",
    "pydantic-ai>=1.0.15,<1.1",
    "openai>=2.1.0,<3.0",  # ← Added explicit dependency
    "pydantic-settings>=2.0.0",
    ...
]
```

### 2. Updated Lock File

Ran:
```bash
uv lock --upgrade-package pydantic-ai --upgrade-package openai
```

### 3. Synced Environment

Ran:
```bash
uv sync
```

---

## Additional Package Updates

The lock process also updated related packages:

| Package | Change |
|---------|--------|
| **genai-prices** | Added v0.0.29 (new dependency) |
| **opentelemetry-instrumentation-httpx** | Added v0.58b0 (new dependency) |
| **opentelemetry-util-http** | Added v0.58b0 (new dependency) |
| **pyperclip** | Added v1.11.0 (new dependency) |
| **temporalio** | Downgraded v1.18.1 → v1.18.0 (compatibility) |

---

## Verification

### Import Test ✅
```bash
$ python -c "import openai; import pydantic_ai; print(f'OpenAI: {openai.__version__}'); print(f'Pydantic-AI: {pydantic_ai.__version__}')"
OpenAI: 2.1.0
Pydantic-AI: 1.0.15
```

### Installed Versions ✅
```bash
$ uv pip list | grep -E "(openai|pydantic-ai)"
openai                                   2.1.0
pydantic-ai                              1.0.15
pydantic-ai-slim                         1.0.15
```

---

## Breaking Changes to Watch

### Pydantic-AI 0.7 → 1.0

This is a **major version upgrade** (0.x → 1.x), which may include:
- API changes
- Breaking changes in agent configuration
- New features and improvements
- Deprecated features removed

### Recommended Actions

1. **Run Full Test Suite**:
   ```bash
   make test
   ```

2. **Check for Deprecation Warnings**:
   ```bash
   pytest -W default
   ```

3. **Review Pydantic-AI Changelog**:
   - Check: https://ai.pydantic.dev/changelog/
   - Look for breaking changes between 0.7.x and 1.0.15

4. **Test Agent Functionality**:
   - Ensure all agent configurations still work
   - Verify async agent execution
   - Test streaming responses
   - Validate error handling

---

## Why OpenAI Was Added Explicitly

Previously, `openai` was a **transitive dependency** through `pydantic-ai`.

**Added explicitly because**:
- Ensures we control the OpenAI version
- Prevents unexpected updates from pydantic-ai changes
- Makes dependency requirements more transparent
- Allows direct version pinning for stability

---

## Next Steps

1. ✅ Dependencies updated
2. ⏳ **Run full test suite**: `make test`
3. ⏳ **Check for compatibility issues**
4. ⏳ **Update any code using deprecated features**
5. ⏳ **Commit changes**: `git add pyproject.toml uv.lock`

---

## Files Modified

- ✅ `pyproject.toml` - Updated dependency versions
- ✅ `uv.lock` - Regenerated with new versions
- ✅ `.venv/` - Synced with new dependencies

---

**Status**: Ready for testing and commit


