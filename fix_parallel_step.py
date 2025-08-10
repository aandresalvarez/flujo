#!/usr/bin/env python3
"""Fix the run_branch function to use step_executor properly."""

import re

def fix_run_branch_function():
    """Fix the run_branch function to use step_executor instead of direct run()."""

    # Read the file
    with open('flujo/application/core/ultra_executor.py', 'r') as f:
        content = f.read()

    # Find the problematic section and replace it
    old_pattern = r'                # Execute the branch using step_executor\n                if branch_context is not None:\n                    branch_result = await branch_pipe\.run\(current_data, context=branch_context\)\n                else:\n                    branch_result = await branch_pipe\.run\(current_data\)\n                # After execution, capture the final state of branch_context\n                final_branch_context = copy\.deepcopy\(branch_context\) if branch_context is not None else None\n                return key, StepResult\(\n                    name=branch_pipe\.name,\n                    output=branch_result,\n                    success=True,\n                    branch_context=final_branch_context\n                \)'

    new_replacement = '''                # Execute the branch using step_executor
                step_result = await step_executor(
                    branch_pipe,
                    data,
                    branch_context,
                    resources,
                    breach_event
                )

                # Capture the final state of branch_context
                final_branch_context = copy.deepcopy(branch_context) if branch_context is not None else None

                # Create a new StepResult with branch_context
                return key, StepResult(
                    name=step_result.name,
                    output=step_result.output,
                    success=step_result.success,
                    attempts=step_result.attempts,
                    latency_s=step_result.latency_s,
                    token_counts=step_result.token_counts,
                    cost_usd=step_result.cost_usd,
                    feedback=step_result.feedback,
                    branch_context=final_branch_context,
                    metadata_=step_result.metadata_
                )'''

    # Replace the pattern
    new_content = re.sub(old_pattern, new_replacement, content, flags=re.DOTALL)

    # Write the updated content
    with open('flujo/application/core/ultra_executor.py', 'w') as f:
        f.write(new_content)

    print("Fixed run_branch function to use step_executor")

if __name__ == "__main__":
    fix_run_branch_function()
