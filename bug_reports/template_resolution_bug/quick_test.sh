#!/bin/bash

echo "🚨 FLUJO TEMPLATE RESOLUTION BUG - QUICK TEST"
echo "=============================================="
echo ""

echo "🧪 Testing broken template resolution..."
echo "Expected: 'Hello World - Processed'"
echo "Actual:   ' - Processed' (step1 output missing)"
echo ""

echo "🔍 Running broken pipeline..."
flujo run test_bug.yaml

echo ""
echo "🧪 Testing working workaround..."
echo "Expected: 'Hello World - Processed'"
echo "Actual:   'Hello World - Processed' (should work)"
echo ""

echo "🔍 Running working pipeline..."
flujo run test_workaround.yaml

echo ""
echo "🎯 COMPARISON:"
echo "✅ test_workaround.yaml works ({{ previous_step }})"
echo "❌ test_bug.yaml fails ({{ steps.step1 }} resolves to nothing)"
echo ""
echo "🚨 THIS IS A CRITICAL FRAMEWORK BUG"
echo "   Template resolution {{ steps.step_name }} is completely broken"
echo "   Only workaround is {{ previous_step }}"
echo "   All multi-step pipelines are affected"
echo ""
echo "For detailed bug report, see: CRITICAL_BUG_REPORT.md"
