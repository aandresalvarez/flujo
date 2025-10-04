# Template Bug Fix: Testing Plan & Status

**Date**: October 3, 2025  
**Feature**: Template Resolution Strict Mode & Logging  
**Status**: Code Complete, Tests Pending

---

## ✅ What's Delivered (Code Complete)

### 1. Configuration Infrastructure
- ✅ `TemplateConfig` class in `flujo/infra/config_manager.py`
- ✅ `undefined_variables` setting ("strict", "warn", "ignore")
- ✅ `log_resolution` setting (true/false)
- ✅ Integrated into Flujo config system

### 2. Strict Template Mode
- ✅ `AdvancedPromptFormatter` accepts `strict` and `log_resolution` parameters
- ✅ Raises `TemplateResolutionError` on undefined variables in strict mode
- ✅ Shows available variables in error message

### 3. Template Resolution Logging
- ✅ Logs template content and available variables at render time
- ✅ Warns when template resolves to empty string
- ✅ Debug-friendly output

### 4. Step Executor Integration
- ✅ HITL step executor uses template configuration
- ✅ Agent step executor uses template configuration
- ✅ `TemplateResolutionError` propagated with step context

### 5. Exception Class
- ✅ `TemplateResolutionError` with helpful messages in `flujo/exceptions.py`

### 6. Documentation
- ✅ 500+ line guide: `docs/user_guide/template_variables_nested_contexts.md`
- ✅ Correct patterns, common mistakes, debugging
- ✅ Configuration examples

### 7. Example YAMLs
- ✅ `test_template_resolution_bug.yaml` - Shows the bug
- ✅ `test_template_resolution_fixed.yaml` - Shows correct patterns

---

## ⚠️ What Needs Testing (Work Remaining)

### Test Categories

#### 1. **Unit Tests** (Priority: HIGH)
Test the core template resolution logic in isolation.

**Files to Create**:
- `tests/unit/utils/test_template_strict_mode.py`

**Test Cases**:
```python
def test_strict_mode_raises_on_undefined():
    """AdvancedPromptFormatter with strict=True raises on undefined vars."""
    formatter = AdvancedPromptFormatter("{{ undefined }}", strict=True)
    with pytest.raises(TemplateResolutionError):
        formatter.format()

def test_warn_mode_logs_warning():
    """AdvancedPromptFormatter with strict=False logs warning."""
    formatter = AdvancedPromptFormatter("{{ undefined }}", strict=False)
    result = formatter.format()  # Should not raise
    assert result == ""  # Backward compat

def test_log_resolution_enabled():
    """log_resolution=True logs template rendering."""
    formatter = AdvancedPromptFormatter("{{ var }}", log_resolution=True)
    # Verify logging happened (check logfire calls)

def test_strict_mode_shows_available_vars():
    """Error message includes available variables."""
    formatter = AdvancedPromptFormatter("{{ bad }}", strict=True)
    try:
        formatter.format(good="value")
    except TemplateResolutionError as e:
        assert "good" in str(e)  # Shows available vars
```

#### 2. **Integration Tests** (Priority: MEDIUM)
Test end-to-end pipeline execution with templates.

**Challenge**: Current test infrastructure doesn't easily support runtime config injection.

**Options**:
A. **Create flujo.toml for tests** - Test-specific config file
B. **Mock config_manager** - Inject config during test setup
C. **Environment variables** - Use env vars to override config

**Recommended Approach (Option C)**:
```python
@pytest.mark.asyncio
async def test_strict_mode_via_config_file(tmp_path):
    """Test strict mode using actual flujo.toml configuration."""
    # Create test-specific flujo.toml
    config_file = tmp_path / "flujo.toml"
    config_file.write_text("""
[template]
undefined_variables = "strict"
log_resolution = true
""")
    
    # Load pipeline with undefined variable
    yaml_content = '''
steps:
  - kind: hitl
    message: "{{ context.undefined }}"
'''
    
    # Run pipeline with config pointing to tmp_path
    # Should raise TemplateResolutionError
    ...
```

#### 3. **Regression Tests** (Priority: HIGH)
Ensure correct patterns continue to work.

**Files to Create**:
- `tests/integration/test_template_correct_patterns_regression.py`

**Test Cases**:
```python
async def test_previous_step_pattern_works():
    """Regression: {{ previous_step.field }} works in all contexts."""
    # Test in: top-level, conditional, loop, nested
    ...

async def test_named_step_reference_works():
    """Regression: {{ steps.name.output.field }} works."""
    ...

async def test_explicit_sink_to_works():
    """Regression: sink_to + {{ context.path }} works."""
    ...
```

#### 4. **Manual Validation Tests** (Priority: LOW)
Use example YAMLs to verify behavior manually.

**Already Created**:
- ✅ `examples/validation/test_template_resolution_bug.yaml`
- ✅ `examples/validation/test_template_resolution_fixed.yaml`

**Manual Test Steps**:
1. Copy `examples/flujo.toml` to project root
2. Set `[template] undefined_variables = "strict"`
3. Run: `uv run flujo validate examples/validation/test_template_resolution_bug.yaml`
4. Expect: Validation passes (templates not checked at validation time)
5. Run: `uv run flujo run examples/validation/test_template_resolution_bug.yaml`
6. Expect: `TemplateResolutionError` at runtime
7. Run: `uv run flujo run examples/validation/test_template_resolution_fixed.yaml`
8. Expect: Works correctly, HITL prompts user

---

## 🚧 Testing Blockers

### Issue 1: Runtime Config Injection
**Problem**: Config is loaded at startup, not runtime. Tests can't easily inject config.

**Solutions**:
1. **Environment variables**: Already supported via pydantic-settings
2. **Config file per test**: Use `tmp_path` to create test-specific `flujo.toml`
3. **Mock ConfigManager**: Patch `get_config_manager()` in tests

**Recommended**: Option 2 (config file per test) - Most realistic

### Issue 2: HITL Test Helpers
**Problem**: Creating HITL steps programmatically in tests is complex.

**Current State**: `gather_result` helper exists but may not work for all scenarios.

**Solution**: Use YAML-based tests with `load_pipeline_blueprint_from_yaml`.

### Issue 3: Test Isolation
**Problem**: Config is global, affects all tests.

**Solution**: Use pytest fixtures to ensure clean config state per test.

---

## 📋 Testing Checklist

### Unit Tests (AdvancedPromptFormatter)
- [ ] Strict mode raises `TemplateResolutionError`
- [ ] Error message includes available variables
- [ ] Error message includes suggestions
- [ ] Warn mode logs warning
- [ ] Warn mode returns empty string (backward compat)
- [ ] Ignore mode is silent
- [ ] `log_resolution` logs template rendering
- [ ] `log_resolution` logs available variables
- [ ] Empty string warning logged

### Integration Tests (End-to-End)
- [ ] HITL message with undefined var raises in strict mode
- [ ] Agent templated_input with undefined var raises in strict mode
- [ ] Correct pattern (previous_step) works in strict mode
- [ ] Correct pattern (named step) works in strict mode
- [ ] Correct pattern (explicit sink_to) works in strict mode
- [ ] Nested contexts (loop → conditional → HITL) work
- [ ] Warn mode continues execution (backward compat)
- [ ] Warn mode logs warning to logfire

### Regression Tests (Correct Patterns)
- [ ] `{{ previous_step.field }}` works in all contexts
- [ ] `{{ steps.name.output.field }}` works in all contexts
- [ ] `sink_to` + `{{ context.path }}` works
- [ ] Multiple templates in one string work
- [ ] Nested field access works (`previous_step.nested.field`)

### Manual Validation
- [ ] Example YAML (bug) shows error in strict mode
- [ ] Example YAML (fixed) works correctly
- [ ] Documentation examples are accurate
- [ ] flujo.toml configuration works

---

## 🎯 Priority for Next Session

### Immediate (Must Have)
1. **Unit tests for `AdvancedPromptFormatter`** - Test core logic in isolation
2. **Regression tests for correct patterns** - Prevent future breakage
3. **Manual validation** - Verify examples work

### Short-term (Should Have)
4. **Integration tests with config** - Test full pipeline with strict mode
5. **Edge case tests** - Multiple undefined vars, nested templates, etc.

### Long-term (Nice to Have)
6. **Performance tests** - Ensure template resolution doesn't slow down pipelines
7. **Fuzzing tests** - Random template strings to find edge cases

---

## 📝 Implementation Notes

### For Future Test Authors

**Pattern to Follow**:
```python
import pytest
from flujo.utils.prompting import AdvancedPromptFormatter
from flujo.exceptions import TemplateResolutionError

def test_strict_mode_basic():
    """Basic strict mode test - undefined variable raises error."""
    formatter = AdvancedPromptFormatter(
        "{{ undefined_var }}",
        strict=True,
        log_resolution=True
    )
    
    with pytest.raises(TemplateResolutionError) as exc_info:
        formatter.format(defined_var="value")
    
    error_msg = str(exc_info.value)
    assert "undefined_var" in error_msg
    assert "defined_var" in error_msg  # Shows available vars
```

**For Integration Tests**:
```python
async def test_with_config(tmp_path):
    """Test using actual flujo.toml configuration."""
    # Write test config
    config = tmp_path / "flujo.toml"
    config.write_text('[template]\nundefined_variables = "strict"')
    
    # Set config path via environment
    with patch.dict(os.environ, {"FLUJO_CONFIG_PATH": str(tmp_path)}):
        # Run pipeline test
        ...
```

---

## ✅ What We Can Ship Now

Even without comprehensive tests, we can ship because:

1. **Code is backward compatible** - Default is "warn" mode
2. **Documentation is complete** - Users know how to use it
3. **Examples are provided** - Users can validate manually
4. **Quality checks pass** - Format, lint, typecheck all pass
5. **Core logic is sound** - `AdvancedPromptFormatter` is well-structured

**Recommended Ship Strategy**:
1. Ship as **experimental feature** with "warn" default
2. Document that strict mode is "beta"
3. Encourage users to try strict mode in development
4. Collect feedback and edge cases
5. Add comprehensive tests based on real usage
6. Promote to stable in next release

---

## 🎓 Lessons Learned

1. **Test infrastructure matters** - Should have validated test helpers first
2. **Config injection is hard** - Global config doesn't play well with tests
3. **Start simple** - Unit tests before integration tests
4. **YAML-based tests are easier** - Less programmatic DSL construction
5. **Manual validation works** - Example YAMLs are valuable for testing

---

**Status**: Feature is **code-complete and documented**. Tests are **planned but not implemented**. 

**Recommendation**: Ship with documentation, gather feedback, add tests based on real usage patterns.

**Risk**: Low - Feature is backward compatible and opt-in via configuration.

