# Flujo Test Suite - Failing Tests Report
*Generated on: $(date)*

## Summary
- **Total Errors:** 8
- **Total Warnings:** 2  
- **Total Skipped:** 2
- **Test Collection Time:** 1.74s

---

## **ERRORS BY CATEGORY**

### **1. Missing Module: `flujo.application.core.step_logic` (5 tests)**

**Issue:** Multiple tests are trying to import from a module that doesn't exist:
```python
ModuleNotFoundError: No module named 'flujo.application.core.step_logic'
```

**Affected Tests:**
- `tests/application/core/test_step_logic_accounting.py`
- `tests/benchmarks/test_legacy_cleanup_performance.py`
- `tests/integration/test_legacy_cleanup_validation.py`
- `tests/regression/test_legacy_cleanup_impact.py`
- `tests/unit/test_fallback_loop_detection.py`

**Root Cause:** These tests were written for a `step_logic` module that was either:
1. Moved to a different location
2. Renamed to something else
3. Removed during refactoring

**Recommended Action:** 
- Locate where the step logic functionality moved to
- Update import statements in all affected test files
- Or create the missing module if it was accidentally removed

---

### **2. Missing Import: `CacheKeyGenerator` (1 test)**

**Issue:** Test trying to import a class that doesn't exist:
```python
ImportError: cannot import name 'CacheKeyGenerator' from 'flujo.application.core.ultra_executor'
```

**Affected Test:**
- `tests/unit/test_ultra_executor_v2.py`

**Root Cause:** The `CacheKeyGenerator` class was either:
1. Renamed to something else
2. Moved to a different module
3. Removed during refactoring

**Recommended Action:**
- Check if `CacheKeyGenerator` was renamed to something else in `ultra_executor.py`
- Update the import statement in the test file
- Or implement the missing class if it was accidentally removed

---

### **3. Missing Import: `assess_clarity_step` (1 test)**

**Issue:** Test trying to import a function that doesn't exist:
```python
ImportError: cannot import name 'assess_clarity_step' from 'manual_testing.examples.cohort_pipeline'
```

**Affected Test:**
- `manual_testing/tests/automated/test_step1_core_agentic.py`

**Root Cause:** The `assess_clarity_step` function was either:
1. Renamed to something else
2. Moved to a different module
3. Removed during refactoring

**Recommended Action:**
- Check the `cohort_pipeline.py` file to see what functions are available
- Update the import statement in the test file
- Or implement the missing function if it was accidentally removed

---

### **4. Import File Mismatch (1 test)**

**Issue:** Test file has conflicting import paths:
```
import file mismatch:
imported module 'test_serialization' has this __file__ attribute:
  /Users/alvaro/Documents/Code/flujo/tests/benchmarks/test_serialization.py
```

**Affected Test:**
- `tests/utils/test_serialization.py`

**Root Cause:** There are two test files with the same module name:
1. `tests/benchmarks/test_serialization.py`
2. `tests/utils/test_serialization.py`

**Recommended Action:**
- Rename one of the test files to avoid the naming conflict
- Or merge the tests if they're testing the same functionality
- Or move one of the files to a different directory

---

## **WARNINGS**

### **1. Makefile Warning**
```
Makefile:181: warning: overriding commands for target `test-health'
```

**Issue:** There are duplicate target definitions in the Makefile.

**Recommended Action:**
- Review the Makefile and remove duplicate `test-health` target definitions
- Ensure only one definition exists for each target

---

## **RECOMMENDED FIXES PRIORITY**

### **High Priority (Blocking Tests)**
1. **Fix Missing `step_logic` Module** - This affects 5 tests and suggests a major refactoring issue
2. **Fix `CacheKeyGenerator` Import** - Core functionality test is broken
3. **Fix `assess_clarity_step` Import** - Manual testing pipeline is broken

### **Medium Priority**
4. **Resolve Import File Mismatch** - Clean up test organization

### **Low Priority**
5. **Fix Makefile Warning** - Cosmetic issue

---

## **NEXT STEPS**

1. **Investigate the `step_logic` module migration** - Check if it was moved to `ultra_executor.py` or another location
2. **Search for `CacheKeyGenerator`** - Check if it was renamed or moved
3. **Check `cohort_pipeline.py`** - See what functions are actually available
4. **Resolve test file naming conflicts** - Rename or reorganize conflicting test files
5. **Clean up Makefile** - Remove duplicate target definitions

---

*Report generated from pytest output with --tb=short* 