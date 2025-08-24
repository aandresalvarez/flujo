#!/bin/bash

# Quick Test Script for Template Fallbacks Bug
# This script verifies the critical bug in under 2 minutes

set -e  # Exit on any error

echo "🚨 TEMPLATE FALLBACKS BUG - QUICK TEST"
echo "======================================="
echo ""
echo "This script verifies the critical bug where conditional template"
echo "syntax like {{ a or b }} fails silently."
echo ""
echo "Expected: {{ context.value or 'fallback' }} should work"
echo "Actual: Template resolution fails or always uses empty value"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "FAILURE")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Check if flujo is available
echo "🔍 Checking Flujo availability..."
if command -v flujo &> /dev/null; then
    print_status "SUCCESS" "Flujo command found"
    FLUJO_CMD="flujo"
elif [ -f ".venv/bin/flujo" ]; then
    print_status "SUCCESS" "Flujo found in .venv/bin/flujo"
    FLUJO_CMD=".venv/bin/flujo"
elif [ -f "../.venv/bin/flujo" ]; then
    print_status "SUCCESS" "Flujo found in ../.venv/bin/flujo"
    FLUJO_CMD="../.venv/bin/flujo"
else
    print_status "FAILURE" "Flujo command not found. Please ensure it's installed and in PATH"
    exit 1
fi

# Test flujo command
echo ""
echo "🧪 Testing Flujo command..."
if $FLUJO_CMD --help &> /dev/null; then
    print_status "SUCCESS" "Flujo command working"
else
    print_status "FAILURE" "Flujo command not working"
    exit 1
fi

# Create test pipeline (temporary file to avoid deleting repo fixtures)
echo ""
echo "📝 Creating test pipeline..."
BUG_SPEC="test_bug_tmp.yaml"
cat > "$BUG_SPEC" << 'EOF'
version: "0.1"
name: "template_fallbacks_bug_test"

steps:
  - kind: step
    name: test_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt or 'Fallback: No prompt provided' }}"
    
  - kind: step
    name: show_result
    agent:
      id: "flujo.builtins.stringify"
    input: "Template result: {{ steps.test_fallback }}"
EOF

print_status "SUCCESS" "Test pipeline created: $BUG_SPEC"

# Test 1: Conditional template (should work but doesn't)
echo ""
echo "🔍 Test 1: Conditional Template (This Should Work But Doesn't)"
echo "---------------------------------------------------------------"
echo "Command: $FLUJO_CMD run $BUG_SPEC"
echo "Expected: Should output fallback text when context.initial_prompt is empty"
echo "Actual: Template resolution fails or outputs empty string"
echo ""

# Helper: run with timeout if available (timeout/gtimeout), else run directly
run_with_timeout() {
    local seconds=$1; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$seconds" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$seconds" "$@"
    else
        "$@"
    fi
}

# Test with timeout to avoid hanging
echo "Running test with 10 second timeout (if available)..."
if run_with_timeout 10 bash -c "$FLUJO_CMD run $BUG_SPEC" 2>&1 | tee test_output.txt; then
    # Check if the output contains the expected fallback
    if grep -q "Fallback: No prompt provided" test_output.txt; then
        print_status "SUCCESS" "Conditional template worked (Bug is FIXED!)"
        BUG_FIXED=true
    elif grep -q "Template result:" test_output.txt && ! grep -q "Fallback: No prompt provided" test_output.txt; then
        print_status "FAILURE" "Template resolved but no fallback used (Bug CONFIRMED)"
        BUG_FIXED=false
    else
        print_status "WARNING" "Unexpected behavior - need manual inspection"
        BUG_FIXED=false
    fi
else
    print_status "FAILURE" "Command timed out or failed (Bug CONFIRMED)"
    BUG_FIXED=false
fi

# Test 2: Workaround - Explicit logic method
echo ""
echo "🔍 Test 2: Workaround Method (Should Work)"
echo "-------------------------------------------"
echo "Testing explicit conditional logic as workaround..."
echo ""

# Create workaround pipeline (temporary file)
WORKAROUND_SPEC="test_workaround_tmp.yaml"
cat > "$WORKAROUND_SPEC" << 'EOF'
version: "0.1"
name: "template_fallbacks_workaround"

steps:
  - kind: step
    name: check_prompt
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ context.initial_prompt }}"
    
  - kind: step
    name: use_fallback
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ steps.check_prompt or 'Fallback: No prompt provided' }}"
EOF

# Test with workaround
if run_with_timeout 10 bash -c "$FLUJO_CMD run $WORKAROUND_SPEC" 2>&1 | tee workaround_output.txt; then
    if grep -q "Fallback: No prompt provided" workaround_output.txt; then
        print_status "SUCCESS" "Workaround method works"
        WORKAROUND_WORKS=true
    else
        print_status "FAILURE" "Workaround method doesn't work"
        WORKAROUND_WORKS=false
    fi
else
    print_status "FAILURE" "Workaround method failed"
    WORKAROUND_WORKS=false
fi

# Clean up test files
echo ""
echo "🧹 Cleaning up test files..."
rm -f "$BUG_SPEC" "$WORKAROUND_SPEC" test_output.txt workaround_output.txt
print_status "SUCCESS" "Test files cleaned up"

# Summary
echo ""
echo "======================================="
echo "🎯 QUICK TEST RESULTS SUMMARY"
echo "======================================="
echo ""

if [ "$BUG_FIXED" = true ]; then
    print_status "SUCCESS" "BUG STATUS: FIXED - Conditional templates now working!"
    echo ""
    echo "🎉 The template fallbacks issue has been resolved!"
    echo "✅ Users can now use {{ a or b }} syntax in templates"
    echo "✅ Fallback values work correctly"
else
    print_status "FAILURE" "BUG STATUS: CONFIRMED - Conditional templates not working"
    echo ""
    echo "🚨 This is a critical issue that breaks data flow and template fallbacks"
    echo "❌ Users cannot use {{ a or b }} syntax in templates"
    echo "❌ No fallback values for missing context"
    echo ""
    
    if [ "$WORKAROUND_WORKS" = true ]; then
        print_status "SUCCESS" "Workarounds are available for immediate development"
        echo ""
        echo "🔧 Available workarounds:"
        echo "   1. Explicit conditional logic in separate steps"
        echo "   2. Default values in context"
        echo "   3. Separate steps with explicit logic"
    else
        print_status "WARNING" "Some workarounds may not be working properly"
    fi
fi

echo ""
echo "📋 Next Steps:"
if [ "$BUG_FIXED" = true ]; then
    echo "✅ Continue using Flujo with conditional templates"
    echo "✅ Test in your specific use cases"
    echo "✅ Remove any workarounds from your workflows"
else
    echo "🔧 Use workarounds documented above for immediate development"
    echo "📋 Report this issue to Flujo development team"
    echo "🧪 Test fixes when they become available"
fi

echo ""
echo "📊 Test Details:"
echo "   - Conditional Templates: $([ "$BUG_FIXED" = true ] && echo "✅ WORKING" || echo "❌ BROKEN")"
echo "   - Workarounds: $([ "$WORKAROUND_WORKS" = true ] && echo "✅ WORKING" || echo "❌ BROKEN")"

echo ""
echo "======================================="
echo "Quick test completed in under 2 minutes"
echo "For detailed analysis, run: python3 minimal_reproduction.py"
echo "======================================="
