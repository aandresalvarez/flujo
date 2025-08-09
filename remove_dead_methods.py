#!/usr/bin/env python3
"""
Task 8: Final Cleanup - Remove dead code from ExecutorCore after policy migration

This script removes the old private _execute_... and _handle_... methods from ultra_executor.py
that are no longer used after the policy migration is complete.

Methods to remove (confirmed as dead code):
- _execute_agent_step (1418-1444)
- _handle_loop_step (1445-1465) 
- _handle_conditional_step (1622-1849)
- _handle_dynamic_router_step (1850-1948)
- _handle_hitl_step (1950-2053)
- _execute_complex_step (2055-2074)
- _handle_cache_step (2076-2166)
- _execute_simple_loop_body (3107-end)
"""

import re

def remove_dead_methods():
    """Remove dead methods from ultra_executor.py"""
    
    # Read the original file
    with open("flujo/application/core/ultra_executor.py", "r") as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Define the line ranges of methods to remove (1-based to 0-based)
    methods_to_remove = [
        (1417, 1443),  # _execute_agent_step (1418-1444 in 1-based)
        (1444, 1464),  # _handle_loop_step (1445-1465 in 1-based)  
        (1621, 1848),  # _handle_conditional_step (1622-1849 in 1-based)
        (1849, 1947),  # _handle_dynamic_router_step (1850-1948 in 1-based)
        (1949, 2052),  # _handle_hitl_step (1950-2053 in 1-based)
        (2054, 2073),  # _execute_complex_step (2055-2074 in 1-based)
        (2075, 2165),  # _handle_cache_step (2076-2166 in 1-based)
        (3106, None),  # _execute_simple_loop_body (3107-end in 1-based)
    ]
    
    # Sort ranges by start line in reverse order so we can remove from end to beginning
    methods_to_remove.sort(key=lambda x: x[0], reverse=True)
    
    # Remove each method
    for start_line, end_line in methods_to_remove:
        if end_line is None:
            # Remove from start_line to end of file
            lines = lines[:start_line]
        else:
            # Remove the specific range
            lines = lines[:start_line] + lines[end_line+1:]
    
    # Write the modified content back
    with open("flujo/application/core/ultra_executor.py", "w") as f:
        f.write('\n'.join(lines))
    
    print("✅ Removed dead methods from ultra_executor.py")
    print("Methods removed:")
    print("- _execute_agent_step")
    print("- _handle_loop_step") 
    print("- _handle_conditional_step")
    print("- _handle_dynamic_router_step")
    print("- _handle_hitl_step")
    print("- _execute_complex_step")
    print("- _handle_cache_step")
    print("- _execute_simple_loop_body")

if __name__ == "__main__":
    remove_dead_methods()
