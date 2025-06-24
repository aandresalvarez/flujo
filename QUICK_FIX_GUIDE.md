# Quick Fix Guide for Flujo Issues

## 🚨 Critical Issues Fixed

### Issue 1: Parameter Passing Mismatch
**Problem**: Flujo passes `pipeline_context` but step functions expect `context`
**Fix**: Apply `flujo_parameter_fix.patch`

### Issue 2: TypeAdapter Schema Generation
**Problem**: `make_agent_async` fails with `TypeAdapter(str)` 
**Fix**: Apply `flujo_pydantic_fix_v3.patch`

## ⚡ Quick Fix Commands

```bash
# 1. Apply parameter passing fix
patch -p1 < flujo_parameter_fix.patch

# 2. Apply Pydantic schema fix  
patch -p1 < flujo_pydantic_fix_v3.patch

# 3. Verify fixes work
python test_flujo_fixes.py
```

## 🔧 Manual Fixes (if patches don't work)

### Fix 1: Parameter Passing
**File**: `flujo/application/flujo_engine.py`
**Lines**: 467, 485, 520, 538

Change:
```python
agent_kwargs["pipeline_context"] = pipeline_context
```

To:
```python
agent_kwargs["context"] = pipeline_context
```

### Fix 2: TypeAdapter Handling
**File**: `flujo/infra/agents.py`
**Lines**: 135-197

Add before Agent creation:
```python
# Handle TypeAdapter and complex type patterns
actual_type = output_type
if hasattr(output_type, '_type'):
    # Handle TypeAdapter instances - extract the underlying type
    actual_type = output_type._type
elif hasattr(output_type, '__origin__') and output_type.__origin__ is not None:
    # Handle generic types like TypeAdapter[str]
    if hasattr(output_type, '__args__') and output_type.__args__:
        if output_type.__origin__.__name__ == 'TypeAdapter':
            actual_type = output_type.__args__[0]

# Use actual_type instead of output_type in Agent constructor
```

## ✅ Verification

Run the test suite:
```bash
python test_flujo_fixes.py
```

Expected output:
```
✓ Parameter passing fix tests passed
✓ TypeAdapter handling tests passed  
✓ make_agent fix tests passed
All tests passed! The fixes appear to work correctly.
```

## 📋 Migration Checklist

- [ ] Apply parameter passing fix
- [ ] Apply Pydantic schema fix
- [ ] Run verification tests
- [ ] Update step functions to use `context` parameter
- [ ] Test with TypeAdapter types
- [ ] Run Flujo's own test suite

## 🆘 Troubleshooting

### Patch fails to apply?
- Check file paths match your Flujo installation
- Apply manual fixes instead

### Tests still fail?
- Verify you're using the correct patch files
- Check Flujo version compatibility
- Review error messages for specific issues

### Need more details?
- See `FLUJO_BUG_REPORT.md` for comprehensive analysis
- See `FLUJO_FIXES_SUMMARY.md` for implementation details
- Run `python test_flujo_fixes.py` for debugging info

## 📞 Support

For issues with these fixes:
1. Check the comprehensive bug report
2. Verify your Flujo version
3. Test with the provided test suite
4. Consider contributing fixes back to Flujo 