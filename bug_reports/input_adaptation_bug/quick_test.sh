#!/bin/bash

# Quick Test Script for Input Adaptation Bug
# This script verifies the critical bug in under 2 minutes

set -e  # Exit on any error

echo "🚨 INPUT ADAPTATION BUG - QUICK TEST"
echo "====================================="
echo ""
echo "This script verifies the critical bug where piped input"
echo "is not captured in pipeline context."
echo ""
echo "Expected: echo 'goal' | flujo run pipeline.yaml should work"
echo "Actual: Pipeline ignores piped input and prompts interactively"
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

# Create test pipeline
echo ""
echo "📝 Creating test pipeline..."
cat > test_bug.yaml << 'PIPELINE_EOF'
version: "0.1"
name: "input_adaptation_bug_test"

steps:
  - kind: step
    name: get_input
    agent:
      id: "flujo.builtins.ask_user"
    input: "{{ context.initial_prompt or 'What do you want to do today?' }}"
    
  - kind: step
    name: process_input
    agent:
      id: "flujo.builtins.stringify"
    input: "Processing: {{ steps.get_input }}"
PIPELINE_EOF

print_status "SUCCESS" "Test pipeline created: test_bug.yaml"

# Test 1: Piped input (should work but doesn't)
echo ""
echo "🔍 Test 1: Piped Input (This Should Work But Doesn't)"
echo "------------------------------------------------------"
echo "Command: echo 'Test Goal' | $FLUJO_CMD run test_bug.yaml"
echo "Expected: Should use 'Test Goal' as input without prompting"
echo "Actual: Still prompts for input interactively"
echo ""

# Test with timeout to avoid hanging
echo "Running test with 10 second timeout..."
if timeout 10 bash -c "echo 'Test Goal' | $FLUJO_CMD run test_bug.yaml" 2>&1 | tee test_output.txt; then
    # Check if the output contains the expected input
    if grep -q "Test Goal" test_output.txt; then
        print_status "SUCCESS" "Piped input was used (Bug is FIXED!)"
        BUG_FIXED=true
    elif grep -q "What do you want to do today?" test_output.txt; then
        print_status "FAILURE" "Still prompting despite piped input (Bug CONFIRMED)"
        BUG_FIXED=false
    else
        print_status "WARNING" "Unexpected behavior - need manual inspection"
        BUG_FIXED=false
    fi
else
    print_status "FAILURE" "Command timed out or failed (Bug CONFIRMED)"
    BUG_FIXED=false
fi

# Clean up test files
echo ""
echo "🧹 Cleaning up test files..."
rm -f test_bug.yaml test_output.txt
print_status "SUCCESS" "Test files cleaned up"

# Summary
echo ""
echo "====================================="
echo "🎯 QUICK TEST RESULTS SUMMARY"
echo "====================================="
echo ""

if [ "$BUG_FIXED" = true ]; then
    print_status "SUCCESS" "BUG STATUS: FIXED - Piped input now working!"
    echo ""
    echo "🎉 The input adaptation issue has been resolved!"
    echo "✅ Users can now use standard Unix piping with Flujo"
    echo "✅ CLI automation and scripting is now supported"
else
    print_status "FAILURE" "BUG STATUS: CONFIRMED - Piped input not working"
    echo ""
    echo "🚨 This is a critical issue that blocks CLI automation"
    echo "❌ Users cannot use standard Unix piping with Flujo"
    echo "❌ Scripts and automation workflows will fail"
    echo ""
    print_status "SUCCESS" "Workarounds are available for immediate development"
    echo ""
    echo "🔧 Available workarounds:"
    echo "   1. Input files: echo 'goal' > input.txt && flujo run pipeline.yaml < input.txt"
    echo "   2. Here-strings: flujo run pipeline.yaml <<< 'goal'"
    echo "   3. Environment variables: FLUJO_INPUT='goal' flujo run pipeline.yaml"
fi

echo ""
echo "📋 Next Steps:"
if [ "$BUG_FIXED" = true ]; then
    echo "✅ Continue using Flujo with standard Unix piping"
    echo "✅ Test in your specific use cases"
    echo "✅ Remove any workarounds from your workflows"
else
    echo "🔧 Use workarounds for immediate development"
    echo "📋 Report this issue to Flujo development team"
    echo "🧪 Test fixes when they become available"
fi

echo ""
echo "====================================="
echo "Quick test completed in under 2 minutes"
echo "For detailed analysis, run: python3 minimal_reproduction.py"
echo "====================================="
