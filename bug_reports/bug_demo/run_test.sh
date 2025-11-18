#!/bin/bash

# Test script to validate nested loop bug
# Automatically provides responses to HITL prompts

cd "$(dirname "$0")"

echo "=== Testing HITL in Loop Behavior ==="
echo ""
echo "This test will:"
echo "1. Answer 3 questions (agent should finish after 3)"
echo "2. Check if loop exits cleanly or creates nested loops"
echo ""
echo "Expected: 3 iterations, then 'SUCCESS!' message"
echo "If nested loops: Will keep asking questions after 'Done!'"
echo ""
echo "Starting test..."
echo ""

# Run with automatic responses
# Answer "yes" to 5 questions (agent should finish at count=3)
{
  sleep 2
  echo "yes"
  sleep 2
  echo "yes"
  sleep 2
  echo "yes"
  sleep 2
  echo ""
  sleep 2
  echo ""
} | timeout 30 uv run flujo run --debug test_hitl_loop.yaml 2>&1 | tee test_output.log

echo ""
echo "=== Analyzing Results ==="
echo ""

# Check if we see nested loops in the output
if grep -q "test_loop.*test_loop" test_output.log; then
  echo "❌ NESTED LOOPS DETECTED!"
  echo "The bug is still present."
else
  echo "✅ No obvious nested loops in output"
fi

# Check if we reached the success step
if grep -q "SUCCESS! Loop exited cleanly" test_output.log; then
  echo "✅ Loop exited cleanly - reached final step"
else
  echo "❌ Did not reach final step - loop may not have exited"
fi

echo ""
echo "Full debug trace saved to: debug/[timestamp]_run.json"
echo "Test output saved to: test_output.log"

