 

### **Bug Report & Feature Proposal for Flujo Core Team**

**Ticket ID:** FSD-026 (Flujo System Design Follow-up)

**Title:** `ConditionalStep` does not natively support boolean results from `condition_expression`, requiring a boilerplate adapter step.

**User Story:**
As a Flujo developer, I want to use a boolean expression in the `condition_expression` field of a `ConditionalStep` and have the step automatically route to branches named `"true"` and `"false"`, so that I can create simple conditional logic in my YAML without writing any custom Python helper functions.

**Current Behavior (Bug):**
The `condition_expression` feature correctly evaluates a boolean expression (e.g., `{{ steps.my_step.output.value > 10 }}`) to a Python `bool` (`True` or `False`).

However, the `ConditionalStep`'s `branches` dictionary expects its keys to be **strings**. When the boolean result is used as a key, it causes a Pydantic validation error during pipeline compilation because the key is a `bool`, not a `str`.

**Error Message:**
```
ValidationError: ... Input should be a valid string [type=string_type, input_value=True, input_type=bool]
```

**Example of Failing (but intuitive) YAML:**
```yaml
- kind: conditional
  name: check_value
  condition_expression: "{{ steps.my_step.output.value > 10 }}"
  branches:
    # This fails because the keys are booleans, not strings.
    true:
      - ...
    false:
      - ...
```

**Current Workaround:**
The user must create a multi-step workaround:
1.  An adapter step to evaluate the expression.
2.  A Python helper function (`boolean_to_branch_key`) to convert the boolean result to a string.
3.  A second adapter step to call that helper.
4.  The `ConditionalStep` then branches on the resulting string ("true" or "false").

This works but adds unnecessary complexity and boilerplate for a very common use case, undermining the goal of a clean, declarative YAML experience.

**Proposed Solution (Feature Enhancement):**

The `ConditionalStep`'s execution policy should be enhanced to intelligently handle the output of a `condition_expression`.

1.  **Detect Boolean Results:** When a `condition_expression` is present, the `ConditionalStepExecutor` policy should check if the evaluated result is a boolean.
2.  **Automatic Key Coercion:** If the result is a boolean, it should be automatically converted to its string representation (`"true"` or `"false"`) before being used to look up a branch in the `branches` dictionary.
3.  **Update Validation:** The `BlueprintStepModel`'s validation for `ConditionalStep` should be updated to allow `true` and `false` (as unquoted YAML booleans) as valid keys within the `branches` dictionary, coercing them to strings during parsing.

**Benefits of this Change:**

*   **Reduces Boilerplate:** Eliminates the need for the `boolean_to_branch_key` helper and the extra adapter steps, cleaning up the YAML significantly.
*   **Improves User Experience:** Makes the framework behave as the user would intuitively expect. Writing a boolean expression should naturally lead to branching on `true` and `false`.
*   **Maximizes Declarative Power:** Fulfills the promise of the `condition_expression` feature by making simple conditional logic purely declarative.
*   **Maintains Consistency:** The `exit_expression` in `LoopStep` already works with booleans directly. This change would make the behavior of `ConditionalStep` consistent with it.

**Implementation Guidance (for the Flujo Team):**

1.  **`flujo/domain/blueprint/loader.py`:** In `_make_step_from_blueprint` for `kind: conditional`, modify the `branches` dictionary validation. When parsing, check if keys are booleans and, if so, convert them to their string equivalents (`"true"`, `"false"`).
2.  **`flujo/application/core/step_policies.py`:** In `DefaultConditionalStepExecutor`, before looking up the `branch_key`, add a check: `if isinstance(branch_key, bool): branch_key = str(branch_key).lower()`.
3.  **`flujo/utils/expressions.py`:** The `_SafeEvaluator` is already correctly producing boolean values. No changes are likely needed here.

**Acceptance Criteria:**

*   The following YAML snippet must pass validation and execute correctly, branching to the `true` branch.
    ```yaml
    steps:
      - kind: step
        name: get_bool
        agent: "lambda data: True" # A simple agent that returns True
      
      - kind: conditional
        name: check_bool
        condition_expression: "{{ previous_step.output }}"
        branches:
          true:
            - kind: step
              name: true_branch_step
              agent: "lambda data: 'It was true'"
          false:
            - kind: step
              name: false_branch_step
              agent: "lambda data: 'It was false'"
    ```
*   The `boolean_to_branch_key` helper function is no longer required for this pattern.
*   Existing pipelines that use string-based keys or Python function callables for `condition` must continue to work without regression.
 