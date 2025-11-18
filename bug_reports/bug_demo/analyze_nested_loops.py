#!/usr/bin/env python3
"""
Analyze debug trace JSON to detect nested loop structures.
"""

import json
import sys
from pathlib import Path

def count_nesting_depth(trace, step_name="clarification_loop", depth=0, max_depth=0):
    """
    Recursively count the maximum nesting depth of a specific step.
    """
    if not isinstance(trace, dict):
        return max_depth
    
    # Check if this is the step we're looking for
    if trace.get("name") == step_name:
        current_depth = depth + 1
        max_depth = max(max_depth, current_depth)
        print(f"  {'  ' * depth}└── {step_name} (depth {current_depth})")
    
    # Recurse into children
    for key, value in trace.items():
        if key == "children" and isinstance(value, list):
            for child in value:
                max_depth = count_nesting_depth(child, step_name, 
                                                depth + 1 if trace.get("name") == step_name else depth, 
                                                max_depth)
        elif isinstance(value, dict):
            max_depth = count_nesting_depth(value, step_name, 
                                            depth + 1 if trace.get("name") == step_name else depth, 
                                            max_depth)
        elif isinstance(value, list):
            for item in value:
                max_depth = count_nesting_depth(item, step_name, 
                                                depth + 1 if trace.get("name") == step_name else depth, 
                                                max_depth)
    
    return max_depth

def analyze_debug_log(filepath):
    """Analyze a debug log file for nested loops."""
    print(f"Analyzing: {filepath}")
    print("=" * 60)
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Find the trace data
    trace = data.get("trace_tree", {})
    
    print("\nSearching for nested 'clarification_loop' instances...")
    print()
    max_depth = count_nesting_depth(trace, "clarification_loop")
    
    print()
    print("=" * 60)
    print(f"Maximum nesting depth: {max_depth}")
    
    if max_depth > 1:
        print("❌ NESTED LOOPS DETECTED!")
        print(f"   Loop nested {max_depth} levels deep")
        print("   This indicates the bug is STILL PRESENT")
    elif max_depth == 1:
        print("✅ NO NESTED LOOPS")
        print("   Loop executed at single level (expected behavior)")
    else:
        print("⚠️  No clarification_loop found in trace")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Use most recent debug file
        debug_dir = Path(__file__).parent / "debug"
        files = sorted(debug_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print("No debug files found")
            sys.exit(1)
        filepath = files[0]
    
    analyze_debug_log(filepath)

