# 🐛 Flujo Bug Reports

This folder contains organized bug reports and issues found during systematic testing of Flujo.

## 📁 **Bug Reports**

### **✅ Template Resolution Bug (RESOLVED)**
- **Location**: `template_resolution_bug/`
- **Issue**: `{{ steps.step_name }}` template resolution completely broken
- **Severity**: **CRITICAL** - Affects all multi-step pipelines
- **Status**: **✅ RESOLVED** - Fixed by Flujo team
- **Workaround**: **✅ FIXED** - `{{ steps.step_name }}` now works perfectly
- **Debug Tools**: 5 debug scripts for investigation and testing

**Quick Access**: [View Template Resolution Bug Report](template_resolution_bug/README_BUG_REPORT.md) | [View Resolution Summary](template_resolution_bug/RESOLUTION_SUMMARY.md)

### **✅ Template Fallbacks Bug (RESOLVED)**
- **Location**: `template_fallbacks_bug/`
- **Issue**: Conditional template syntax `{{ a or b }}` fails silently
- **Severity**: **HIGH** - Breaks data flow and template fallback logic
- **Status**: **✅ RESOLVED** - Fixed by Flujo team
- **Workaround**: **✅ NO LONGER NEEDED** - Conditional templates now work natively
- **Investigation Tools**: 2 investigation scripts and comprehensive testing

**Quick Access**: [View Template Fallbacks Bug Report](template_fallbacks_bug/README_BUG_REPORT.md) | [View Resolution Summary](template_fallbacks_bug/RESOLUTION_SUMMARY.md)

### **🔍 Input Adaptation Bug (ACTIVE INVESTIGATION)**
- **Location**: `input_adaptation_bug/`
- **Issue**: Piped input not captured in pipeline context
- **Severity**: **HIGH** - Blocks core CLI functionality
- **Status**: **🔍 ACTIVE** - Under investigation and reporting
- **Workaround**: **LIMITED** - Use input files or here-strings instead
- **Investigation Tools**: 2 investigation scripts and comprehensive testing

**Quick Access**: [View Input Adaptation Bug Report](input_adaptation_bug/README_BUG_REPORT.md) | [Run Quick Test](input_adaptation_bug/quick_test.sh)

## 🎯 **Bug Status Overview**

### **✅ Resolved Bugs**
- **Template Resolution** - `{{ steps.step_name }}` now works perfectly
- **Template Fallbacks** - `{{ a or b }}` conditional syntax now works perfectly

### **🔍 Active Investigation**
- **Input Adaptation** - Piped input not captured in pipeline context

### **📊 Resolution Statistics**
- **Total Bugs Reported**: 3
- **Bugs Resolved**: 2 (67%) 🎉
- **Resolution Time**: ~1-2 weeks for template issues
- **Framework Quality**: **MOSTLY READY** - Only Input Adaptation issue remains

## 🔍 **Bug Report Structure**

Each bug report folder contains:

- **`README_BUG_REPORT.md`** - Complete package documentation
- **`CRITICAL_BUG_REPORT.md`** - Comprehensive bug report with evidence
- **`CRITICAL_FINDINGS_SUMMARY.md`** - Executive summary for stakeholders
- **`RESOLUTION_SUMMARY.md`** - ✅ **NEW** - Complete resolution documentation (for resolved bugs)
- **Reproduction scripts** - Multiple ways to reproduce the issue
- **Test pipelines** - YAML files demonstrating the bug
- **Working examples** - Pipelines using workarounds
- **Investigation tools** - Investigation and testing tools

## 🚀 **How to Use**

### **For Flujo Development Team**
1. **Navigate to specific bug folder** (e.g., `input_adaptation_bug/`)
2. **Read `README_BUG_REPORT.md`** for complete overview
3. **Use reproduction scripts** to verify the issue
4. **Review evidence** in `CRITICAL_BUG_REPORT.md`
5. **Implement fixes** based on findings

### **For Users/Contributors**
1. **Browse bug reports** to understand current limitations
2. **Use workarounds** documented in each report
3. **Report new issues** following the established format
4. **Test fixes** when they become available

## 🎯 **Reporting New Bugs**

When reporting new bugs, follow this structure:

1. **Create new folder** in `bug_reports/` with descriptive name
2. **Include reproduction steps** with clear examples
3. **Provide test cases** that demonstrate the issue
4. **Document workarounds** if available
5. **Assess impact** on user experience
6. **Suggest solutions** if possible

## 📝 **Template for New Bug Reports**

```markdown
# 🐛 [Bug Name]

## 📋 **Summary**
- **Issue**: Brief description
- **Severity**: [LOW/MEDIUM/HIGH/CRITICAL]
- **Status**: [REPRODUCIBLE/INVESTIGATING/FIXED]
- **Workaround**: [Available/None/Limited]

## 🔍 **Description**
Detailed description of the issue...

## 🧪 **Reproduction Steps**
1. Step 1
2. Step 2
3. Step 3

## 📊 **Impact Assessment**
- **Scope**: [Limited/Moderate/Universal]
- **User Experience**: [Minor/Major/Critical]
- **Development**: [Blocked/Limited/Unaffected]

## 🔧 **Workarounds**
Available workarounds...

## 🚀 **Next Steps**
Recommended actions...
```

## 🌟 **Quality Standards**

All bug reports must meet these standards:

- ✅ **Reproducible** - Clear steps to reproduce
- ✅ **Evidence-based** - Systematic testing results
- ✅ **Actionable** - Clear next steps for resolution
- ✅ **Professional** - Well-structured and documented
- ✅ **Complete** - All necessary information included

---

**Last Updated**: August 2024
**Maintainer**: Alvaro (Flujo user/contributor)
**Status**: Active bug tracking and reporting
**Progress**: 🎉 **67% Complete** - 2/3 critical issues resolved!
