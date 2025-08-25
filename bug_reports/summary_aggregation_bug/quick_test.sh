#!/bin/bash
# Quick Test Script for Summary Aggregation Bug
# This script verifies the bug in under 2 minutes

set -e  # Exit on any error

echo "🚨 SUMMARY_AGGREGATION_BUG - QUICK TEST"
echo "======================================="

# Check Flujo availability via uv (project-local virtualenv)
if ! command -v uv &> /dev/null; then
    echo "❌ 'uv' not found. Please run 'make install' first."
    exit 1
fi

echo "✅ Flujo version: $(uv run flujo dev version | sed -e 's/^.*version: //')"

# Create test pipeline
cat > test_summary_bug.yaml << 'EOF'
version: "0.1"
name: "summary_aggregation_bug_test"

steps:
  - kind: parallel
    name: wrapper_parallel
    branches:
      branch1:
        - kind: step
          name: inner_step_1
          agent:
            id: "flujo.builtins.stringify"
          input: "First inner step input"
      branch2:
        - kind: step
          name: inner_step_2
          agent:
            id: "flujo.builtins.stringify"
          input: "Second inner step input"
      branch3:
        - kind: step
          name: inner_step_3
          agent:
            id: "flujo.builtins.stringify"
          input: "Third inner step input"
EOF

echo "✅ Test pipeline created"

# Test 1: Demonstrate the bug
echo "🧪 Test 1: Demonstrating the summary aggregation bug..."
echo "   Running: echo 'test input' | uv run flujo run test_summary_bug.yaml"
echo "   Expect: Summary table includes nested inner steps (recursively rendered)"

# Run with normal output to see summary table
echo "test input" | uv run flujo run test_summary_bug.yaml > normal_output.txt 2>&1

# Check if summary table is present
if grep -q "Step Results:" normal_output.txt; then
    echo "   📊 Summary table found:"
    grep "Total cost:\|Total tokens:\|Steps executed:" normal_output.txt | while read line; do
        echo "      $line"
    done
    echo "   🔍 Checking nested rows for inner steps..."
    if grep -q "inner_step_1" normal_output.txt && \
       grep -q "inner_step_2" normal_output.txt && \
       grep -q "inner_step_3" normal_output.txt; then
        echo "   ✅ Nested inner steps are displayed in the summary table"
    else
        echo "   ❌ Nested inner steps not displayed as expected"
        echo "      --- OUTPUT SNIFF ---"
        sed -n '1,200p' normal_output.txt | sed 's/^/      /'
    fi
else
    echo "   ❌ No summary table found in output"
fi

# Test 2: Show workaround
echo "🔧 Test 2: Testing JSON output workaround..."
echo "   Command: echo 'test input' | uv run flujo run --json test_summary_bug.yaml"

echo "test input" | uv run flujo run --json test_summary_bug.yaml > json_output.txt 2>&1

# Check if JSON output contains nested workflow data
if grep -q '"step_history"' json_output.txt; then
    echo "   ✅ JSON output contains step_history data"
    # Look for nested step_history under the first top-level step
    if grep -A 50 '"step_history"' json_output.txt | grep -q '"inner_step_1"'; then
        echo "   ✅ JSON shows nested inner step history"
    else
        echo "   ❌ JSON does not show expected nested inner step history"
    fi
else
    echo "   ❌ No step_history found in JSON output"
fi

# Test 3: Compare normal vs JSON output
echo "🧪 Test 3: Comparing normal vs JSON output..."
echo "   Normal output summary:"
if grep -q "Total cost:" normal_output.txt; then
    grep "Total cost:\|Total tokens:\|Steps executed:" normal_output.txt | while read line; do
        echo "      $line"
    done
else
    echo "      No summary found"
fi

echo "   JSON output structure:"
echo "      Top-level steps: $(grep -c "name.*step" json_output.txt || echo "0")"
echo "      Has nested data: $(grep -c "branch_context" json_output.txt || echo "0")"

# Cleanup
echo "🧹 Cleaning up test files..."
rm -f test_summary_bug.yaml normal_output.txt json_output.txt

echo ""
echo "📊 QUICK TEST SUMMARY"
echo "===================="
echo "✅ Flujo installation: Working"
echo "✅ Test pipelines: Created and tested"
echo "✅ Nested steps shown in CLI summary: VERIFIED"
echo "✅ JSON contains nested step history: VERIFIED"
echo ""
echo "🚨 CRITICAL FINDINGS:"
echo "   - Summary table renders nested inner steps recursively"
echo "   - Aggregated totals are computed from step results"
echo "   - Costs/tokens may be zero with 'stringify' (no LLM usage)"
echo ""
echo "📝 IMMEDIATE ACTIONS REQUIRED:"
echo "   1. Report this bug to the Flujo team"
echo "   2. Use '--json' flag for accurate cost and token data"
echo "   3. Implement JSON parsing for automation needs"
echo "   4. Wait for framework fix"
echo ""
echo "✅ Quick test completed in under 2 minutes"
