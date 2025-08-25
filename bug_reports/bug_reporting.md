# 🐛 Flujo Bug Reporting Guide

## 📋 **Overview**

This guide provides comprehensive instructions for reporting bugs to the Flujo development team. It follows the architectural principles outlined in `FLUJO_TEAM_GUIDE.md` and incorporates the systematic approach that successfully identified and resolved critical framework issues.

**Purpose**: Enable users and contributors to report bugs in a way that accelerates resolution and maintains Flujo's quality standards.

**Audience**: Flujo users, contributors, and development team members.

---

## 🎯 **Core Principles (Following FLUJO_TEAM_GUIDE.md)**

### **1. Policy-Driven Architecture Respect**
- **Never bypass the policy system** - All execution logic belongs in dedicated policy classes
- **Report issues at the policy level** - Don't suggest changes to `ExecutorCore` dispatcher
- **Respect separation of concerns** - Identify whether the issue is in policies, domain logic, or infrastructure

### **2. Control Flow Exception Safety**
- **Distinguish control flow vs. data failures** - Control flow exceptions orchestrate workflows
- **Never convert control flow to data failures** - This breaks the entire orchestration system
- **Report exception handling issues** - Especially if `PausedException`, `PipelineAbortSignal`, or `InfiniteRedirectError` are mishandled

### **3. Context Idempotency**
- **Report context corruption issues** - Steps must be idempotent with respect to pipeline context
- **Identify isolation failures** - Complex steps must use `ContextManager.isolate()` properly
- **Report retry poisoning** - Failed attempts must not "poison" context for subsequent retries

### **4. Proactive Quota System**
- **Report reactive budget enforcement** - Resource limits must be enforced proactively
- **Identify governor patterns** - Legacy "governor" patterns are disallowed
- **Report quota splitting issues** - Parallel branches must use `Quota.split()` properly

### **5. Centralized Configuration**
- **Report direct config access** - All configuration must go through `flujo.infra.config_manager`
- **Identify environment variable bypasses** - Don't read `flujo.toml` or env vars directly
- **Report decentralized config patterns** - Configuration must be centralized

---

## 🚨 **Bug Classification System**

### **Critical (P0) - Immediate Attention Required**
- **Framework crashes** or infinite loops
- **Data corruption** or loss
- **Security vulnerabilities**
- **Complete functionality failure** (e.g., all templates broken)

### **High (P1) - High Priority**
- **Core functionality broken** (e.g., template resolution, piped input)
- **Performance regressions** >50%
- **User experience severely degraded**
- **Integration failures** with external systems

### **Medium (P2) - Normal Priority**
- **Feature limitations** or edge cases
- **Performance issues** <50%
- **Documentation gaps** or inconsistencies
- **Minor UI/UX issues**

### **Low (P3) - Low Priority**
- **Cosmetic issues**
- **Documentation typos**
- **Minor performance optimizations**
- **Feature requests**

---

## 📝 **Bug Report Structure**

### **Required Sections (All Bugs)**

#### **1. Executive Summary**
```markdown
## 🎯 **Executive Summary**

- **Bug ID**: `UNIQUE_ID_001`
- **Title**: Clear, concise description
- **Severity**: P0/P1/P2/P3
- **Priority**: Immediate/High/Normal/Low
- **Status**: New/Reproducible/In Progress/Resolved
- **Report Date**: YYYY-MM-DD
- **Reporter**: Your name/username
```

#### **2. Issue Description**
```markdown
## 📋 **Issue Description**

**What is broken?**
Clear description of the expected vs. actual behavior.

**When does it happen?**
Specific conditions, steps, or triggers.

**What should happen instead?**
Expected correct behavior.
```

#### **3. Reproduction Steps**
```markdown
## 🔄 **Reproduction Steps**

1. **Prerequisites**: Required setup, dependencies, configuration
2. **Steps**: Numbered, specific actions to reproduce
3. **Expected Result**: What should happen
4. **Actual Result**: What actually happens
5. **Reproducibility**: Always/Sometimes/Never (with percentages)
```

#### **4. Evidence & Logs**
```markdown
## 📊 **Evidence & Logs**

**Error Messages**: Complete error output
**Logs**: Relevant log files or console output
**Screenshots**: If applicable (UI issues)
**Stack Traces**: Full Python tracebacks
```

#### **5. Environment Information**
```markdown
## 🌍 **Environment Information**

- **Flujo Version**: `flujo --version`
- **Python Version**: `python --version`
- **Operating System**: OS name and version
- **Dependencies**: Relevant package versions
- **Configuration**: `flujo.toml` settings (if relevant)
```

#### **6. Impact Assessment**
```markdown
## 💥 **Impact Assessment**

**User Impact**: How many users are affected?
**Business Impact**: What functionality is blocked?
**Development Impact**: How does this affect development?
**Workarounds**: Are there temporary solutions?
```

#### **7. Proposed Solutions**
```markdown
## 🛠️ **Proposed Solutions**

**Immediate Workarounds**: Temporary fixes users can apply
**Long-term Fixes**: Architectural solutions
**Alternative Approaches**: Different ways to achieve the goal
**Implementation Priority**: Suggested development order
```

### **Optional Sections (Complex Bugs)**

#### **8. Technical Analysis**
```markdown
## 🔍 **Technical Analysis**

**Root Cause**: Technical explanation of why this happens
**Affected Components**: Which parts of the framework are involved
**Related Issues**: Similar bugs or related problems
**Architectural Implications**: How this affects the overall design
```

#### **9. Testing & Validation**
```markdown
## 🧪 **Testing & Validation**

**Test Cases**: Specific scenarios to verify the fix
**Edge Cases**: Unusual conditions that might break
**Performance Impact**: Any performance considerations
**Backward Compatibility**: Impact on existing pipelines
```

---

## 🚀 **Bug Report Creation Process**

### **Step 1: Issue Investigation**
1. **Reproduce the issue** - Ensure it's not user error
2. **Check existing issues** - Avoid duplicates
3. **Gather evidence** - Logs, error messages, screenshots
4. **Test workarounds** - Identify temporary solutions

### **Step 2: Report Creation**
1. **Use the template** - Follow the structure above
2. **Be specific** - Vague reports are harder to fix
3. **Include examples** - Code snippets, YAML files, error messages
4. **Prioritize correctly** - Don't overstate or understate severity

### **Step 3: Submission & Follow-up**
1. **Submit the report** - Use the appropriate channel
2. **Respond to questions** - Help the team understand the issue
3. **Test fixes** - Verify when solutions are provided
4. **Update status** - Mark as resolved when fixed

---

## 📁 **Bug Report Package Structure**

### **Standard Package Layout**
```
bug_reports/
├── your_bug_name/
│   ├── README_BUG_REPORT.md          # Main bug report
│   ├── CRITICAL_BUG_REPORT.md        # Detailed technical analysis
│   ├── CRITICAL_FINDINGS_SUMMARY.md  # Executive summary
│   ├── minimal_reproduction.py       # Python reproduction script
│   ├── quick_test.sh                 # Shell script for quick verification
│   ├── test_bug.yaml                 # Pipeline demonstrating the bug
│   ├── test_workaround.yaml          # Pipeline showing workarounds
│   └── BUG_REPORT_SUMMARY.md         # Package overview
```

### **File Naming Conventions**
- **README files**: `README_*.md` for main documentation
- **Bug reports**: `*_BUG_REPORT.md` for detailed reports
- **Summaries**: `*_SUMMARY.md` for executive summaries
- **Scripts**: `*.py` for Python, `*.sh` for shell scripts
- **Pipelines**: `test_*.yaml` for test pipelines

---

## 🧪 **Reproduction Scripts**

### **Python Reproduction Script**
```python
#!/usr/bin/env python3
"""
Minimal Reproduction Script for [Bug Name]

This script demonstrates the bug in a minimal, reproducible way.
"""

import subprocess
import tempfile
import os
import sys

def create_test_pipeline():
    """Create a minimal test pipeline that demonstrates the bug."""
    return """version: "0.1"
name: "bug_reproduction_test"

steps:
  - kind: step
    name: test_step
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.test_value or 'Default value' }}"
"""

def test_bug():
    """Test the bug and report results."""
    # Implementation here
    pass

def demonstrate_workarounds():
    """Show available workarounds."""
    # Implementation here
    pass

if __name__ == "__main__":
    print("Testing bug reproduction...")
    test_bug()
    print("Demonstrating workarounds...")
    demonstrate_workarounds()
```

### **Shell Quick Test Script**
```bash
#!/bin/bash
# Quick Test Script for [Bug Name]
# This script verifies the bug in under 2 minutes

set -e  # Exit on any error

echo "🚨 [BUG_NAME] - QUICK TEST"
echo "=========================="

# Check Flujo availability
if ! command -v flujo &> /dev/null; then
    echo "❌ Flujo not found. Please install Flujo first."
    exit 1
fi

echo "✅ Flujo found: $(flujo --version)"

# Create test pipeline
cat > test_bug.yaml << 'EOF'
version: "0.1"
name: "bug_reproduction_test"

steps:
  - kind: step
    name: test_step
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.test_value or 'Default value' }}"
EOF

echo "✅ Test pipeline created"

# Test 1: Demonstrate the bug
echo "🧪 Test 1: Demonstrating the bug..."
# Test implementation here

# Test 2: Show workaround
echo "🔧 Test 2: Testing workaround..."
# Workaround test here

echo "✅ Quick test completed"
```

---

## 🔍 **Investigation Tools & Techniques**

### **Built-in Flujo Tools**
```bash
# Check Flujo version and configuration
flujo --version
flujo --help

# Run with verbose output
flujo run --verbose pipeline.yaml

# Check configuration
flujo config --show

# Validate pipeline
flujo validate pipeline.yaml
```

### **Debugging Techniques**
1. **Minimal reproduction** - Strip down to the smallest case
2. **Progressive complexity** - Add complexity until bug appears
3. **Environment isolation** - Test in clean environment
4. **Version comparison** - Test with different Flujo versions
5. **Configuration variations** - Test with different settings

### **Logging & Telemetry**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use Flujo's telemetry
from flujo.telemetry import logfire
logfire.info("Debug information")
logfire.error("Error details")
```

---

## 📊 **Quality Standards**

### **Report Quality Checklist**
- [ ] **Reproducible**: Can the team reproduce the issue?
- [ ] **Specific**: Is the problem clearly defined?
- [ ] **Evidence-based**: Is there concrete proof of the issue?
- [ ] **Actionable**: Can the team act on this report?
- [ ] **Complete**: Are all required sections filled out?
- [ ] **Accurate**: Is the severity and priority correct?

### **Evidence Quality Standards**
- **Error messages**: Complete, unedited output
- **Logs**: Relevant sections with context
- **Code examples**: Minimal, working examples
- **Screenshots**: Clear, annotated images
- **Reproduction steps**: Specific, numbered instructions

---

## 🚫 **Common Mistakes to Avoid**

### **❌ Don't Do This**
- **Vague descriptions**: "It doesn't work" or "It's broken"
- **Missing reproduction steps**: Team can't verify the issue
- **Incomplete environment info**: Missing version numbers or OS details
- **Overstated severity**: Marking cosmetic issues as critical
- **Missing evidence**: No logs, errors, or examples
- **Duplicate reports**: Not checking existing issues first

### **✅ Do This Instead**
- **Specific descriptions**: "Template resolution fails for nested context access"
- **Complete steps**: Numbered, specific reproduction instructions
- **Full environment**: Complete version and configuration details
- **Accurate severity**: Honest assessment of impact
- **Rich evidence**: Logs, errors, screenshots, examples
- **Unique issues**: Check for existing reports before creating new ones

---

## 🔄 **Bug Lifecycle Management**

### **Status Tracking**
1. **New**: Initial report submitted
2. **Reproducible**: Team can reproduce the issue
3. **In Progress**: Team is working on a fix
4. **Testing**: Fix is being tested
5. **Resolved**: Issue is fixed and verified
6. **Closed**: Issue is fully resolved

### **Update Responsibilities**
- **Reporter**: Update when new information is discovered
- **Team**: Update status as work progresses
- **Both**: Communicate about testing and verification

---

## 📞 **Submission Channels**

### **Primary Channels**
1. **GitHub Issues**: For most bug reports
2. **Discord/Slack**: For urgent issues or discussions
3. **Email**: For security issues or sensitive information

### **Channel Selection Guide**
- **GitHub Issues**: Standard bug reports, feature requests
- **Discord/Slack**: Quick questions, urgent issues, discussions
- **Email**: Security vulnerabilities, private information

---

## 🎯 **Success Metrics**

### **Report Quality Metrics**
- **Resolution time**: How quickly bugs are fixed
- **Resolution rate**: Percentage of bugs resolved
- **User satisfaction**: Feedback on bug handling
- **Team efficiency**: Time spent on investigation vs. fixing

### **Continuous Improvement**
- **Template updates**: Refine based on team feedback
- **Process optimization**: Streamline reporting workflow
- **Training**: Help users create better reports
- **Automation**: Tools to improve report quality

---

## 📚 **Examples & Templates**

### **Example: Template Resolution Bug**
See `bug_reports/template_resolution_bug/` for a complete example of:
- Professional bug report structure
- Comprehensive evidence collection
- Systematic investigation approach
- Clear impact assessment
- Effective workarounds

### **Example: Input Adaptation Bug**
See `bug_reports/input_adaptation_bug/` for an example of:
- CLI functionality testing
- Pipeline automation verification
- Workaround documentation
- User experience impact analysis

### **Example: Template Fallbacks Bug**
See `bug_reports/template_fallbacks_bug/` for an example of:
- Conditional template testing
- Fallback logic verification
- Edge case identification
- Resolution tracking

---

## 🏆 **Best Practices Summary**

### **Before Reporting**
1. **Investigate thoroughly** - Ensure it's a real bug
2. **Check existing issues** - Avoid duplicates
3. **Gather evidence** - Logs, errors, examples
4. **Test workarounds** - Identify temporary solutions

### **During Reporting**
1. **Follow the template** - Use the structured format
2. **Be specific** - Clear, actionable descriptions
3. **Include examples** - Code, YAML, error messages
4. **Assess impact** - Honest severity and priority

### **After Reporting**
1. **Respond promptly** - Help with team questions
2. **Test fixes** - Verify when solutions are provided
3. **Update status** - Mark as resolved when fixed
4. **Provide feedback** - Help improve the process

---

## 🔗 **Additional Resources**

### **Related Documentation**
- [FLUJO_TEAM_GUIDE.md](../FLUJO_TEAM_GUIDE.md) - Core development principles
- [Contributing Guidelines](../CONTRIBUTING.md) - General contribution process
- [Development Setup](../docs/development/) - Local development environment

### **Community Resources**
- **Discord**: Join for real-time discussions
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Comprehensive guides and examples

### **Support Channels**
- **Bug Reports**: Use this guide and submit via GitHub Issues
- **Feature Requests**: Submit via GitHub Issues with enhancement label
- **Questions**: Use Discord or GitHub Discussions
- **Security Issues**: Email the security team directly

---

## 📝 **Quick Reference**

### **Essential Commands**
```bash
# Check Flujo version
flujo --version

# Validate pipeline
flujo validate pipeline.yaml

# Run with verbose output
flujo run --verbose pipeline.yaml

# Check configuration
flujo config --show
```

### **Required Report Sections**
1. Executive Summary
2. Issue Description
3. Reproduction Steps
4. Evidence & Logs
5. Environment Information
6. Impact Assessment
7. Proposed Solutions

### **Quality Checklist**
- [ ] Reproducible
- [ ] Specific
- [ ] Evidence-based
- [ ] Actionable
- [ ] Complete
- [ ] Accurate

---

**Remember**: Good bug reports save time for everyone. Take the time to create a comprehensive, well-structured report that the Flujo team can act on quickly and effectively.

**Goal**: Transform bug reports from time-consuming investigations into clear, actionable development tasks that accelerate Flujo's improvement and maintain its quality standards.
