# Functional Specification Document: FSD-11

**Title:** Signature-Aware Context Injection for Agent Execution
**Author:** Alvaro Alvarez (as per user feedback)
**Status:** Approved for Implementation
**Priority:** P1 - Critical
**Date:** July 23, 2025
**Version:** 1.1

---

## 1. Overview

This document specifies a critical bug fix for the Flujo framework's core execution logic. Currently, the `Flujo` runner attempts to inject a `pipeline_context` object into every agent's `run` method, regardless of its signature. This causes a `TypeError` for any simple, stateless agent created with the standard `make_agent_async` factory, as the underlying Pydantic-AI agent does not accept a `context` keyword argument. This bug makes the most basic and common use case of the framework fail.

The proposed solution is to make the context injection mechanism **signature-aware**. The runner will inspect the target agent's `run` method before execution. It will only inject the `context` object if the method signature explicitly includes a `context` parameter or a variable keyword argument (`**kwargs`). This change will make simple, stateless agents work out-of-the-box, significantly improving the new user experience while maintaining full functionality for advanced, context-aware agents.

Additionally, error reporting for step failures will be enhanced to expose the original underlying exception, dramatically improving debuggability.

## 2. Problem Statement

A developer attempting to run a minimal pipeline with a standard, stateless Flujo agent encounters an immediate and non-obvious failure.

**Scenario:** A developer defines a simple agent using the recommended `flujo.infra.agents.make_agent_async` factory. This agent is intended to be stateless and therefore does not have a `context` parameter in its logic.

```python
# The CORRECT way to create a stateless agent in Flujo
from flujo.infra.agents import make_agent_async

stateless_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are a helpful assistant.",
    output_type=str
)

# This agent is an AsyncAgentWrapper. Its `run` method passes arguments
# down to an internal Pydantic-AI agent, which does NOT accept a 'context' argument.
```

When this standard, stateless agent is used in a Flujo pipeline, the `Flujo` runner attempts to call the wrapper's `run` method with a `context` argument (`agent.run(data, context=...)`). The wrapper then passes this unexpected `context` argument down to the internal Pydantic-AI agent, which correctly rejects it. This results in a `TypeError`, which is then hidden by Flujo's current error handling.

**Current Erroneous Behavior:**

1. User creates a simple, stateless agent using the standard `make_agent_async` factory.
2. The `Flujo` runner unconditionally injects the `context` object when it calls the agent wrapper.
3. The agent call fails with `TypeError: run() got an unexpected keyword argument 'context'`.
4. Flujo's runner reports a vague failure like `Step 'MyStep' failed.` without exposing the underlying `TypeError`.

This behavior is incorrect because it prevents the most fundamental use case from working as expected and makes debugging unnecessarily difficult. It forces new users to immediately learn about and implement context-aware patterns for even the simplest tasks.

## 3. Functional Requirements (FR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| **FR-35** | The Flujo step executor **SHALL** inspect the signature of an agent's `run` method before invoking it. | To determine if the agent can accept a `context` argument. |
| **FR-35a** | The step executor **SHALL** pass the `context` object as a keyword argument **only if** the agent's `run` method signature contains either a parameter named `context` or a variable keyword parameter (`**kwargs`). | Corrects the core bug by preventing `TypeError` for stateless agents. |
| **FR-35b** | The step executor **SHALL NOT** pass the `context` object if the agent's `run` method signature does not meet the criteria in FR-35a. | Ensures simple, stateless agents can be executed without modification. |
| **FR-36** | When an agent execution fails within a step, the resulting `StepResult` object **SHALL** have its `feedback` attribute populated with the type and message of the original underlying exception. | Improves debuggability by exposing the root cause of failures, such as the `TypeError` described in the problem statement. |

## 4. Non-Functional Requirements (NFR)

| ID | Requirement | Justification |
| :--- | :--- | :--- |
| **NFR-13** | The signature inspection mechanism **MUST NOT** introduce a noticeable performance degradation. The overhead for a single step execution should be less than 1ms. | `inspect.signature()` can be slow. The implementation must include caching to ensure performance is maintained in high-throughput pipelines. |
| **NFR-14** | Existing pipelines with correctly defined context-aware agents (e.g., those implementing `ContextAwareAgentProtocol`) **MUST** continue to function without any changes. | Ensures backward compatibility for users who have already adopted advanced patterns. |

## 5. Technical Design & Specification

The fix will be implemented by introducing a cached signature inspection utility and modifying the runner's invocation logic.

### 5.1. Signature Inspection Helper

A cached helper function will be created in `flujo/application/context_manager.py`.

```python
# In flujo/application/context_manager.py

import inspect
import weakref
from typing import Callable, Any, Dict

# Weak-keyed cache for hashable callables, ID-based for unhashable ones
_accepts_param_cache_weak: weakref.WeakKeyDictionary[Callable, Dict[str, bool]] = weakref.WeakKeyDictionary()
_accepts_param_cache_id: weakref.WeakValueDictionary[int, Dict[str, bool]] = weakref.WeakValueDictionary()

def _accepts_param(func: Callable[..., Any], param: str) -> bool:
    """
    Check if a callable's signature includes `param` or `**kwargs`.
    Results are cached for performance.
    """
    # ... (Implementation of cache lookup) ...

    try:
        sig = inspect.signature(func)
        if param in sig.parameters:
            result = True
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            result = True
        else:
            result = False
    except (TypeError, ValueError):
        # If signature cannot be determined, assume it doesn't accept the param
        result = False

    # ... (Implementation of cache storage) ...
    return result
```

### 5.2. Runner Invocation Logic Modification

The core change will be in `flujo/application/core/ultra_executor.py`, within the `_execute_step_logic` function (or a helper it calls).

**Current (Simplified) Logic:**
```python
agent_kwargs = {}
if context is not None:
    agent_kwargs["context"] = context # BUG: This is always added
raw_output = await current_agent.run(data, **agent_kwargs)
```

**Proposed (Simplified) Logic:**
```python
from flujo.application.context_manager import _accepts_param

agent_kwargs = {}
# FR-35a & FR-35b: Use the helper to check the agent's signature
if context is not None and _accepts_param(current_agent.run, "context"):
    agent_kwargs["context"] = context

# ... (add other kwargs like resources, temperature similarly) ...
raw_output = await current_agent.run(data, **agent_kwargs)
```

### 5.3. Enhanced Error Reporting

The `try...except` block surrounding the agent call in `_execute_step_logic` will be modified.

**Current (Simplified) Logic:**
```python
try:
    # agent call
except Exception as e:
    result.feedback = "Step failed." # Hides the real error
```

**Proposed (Simplified) Logic:**
```python
try:
    # agent call
except Exception as e:
    # FR-36: Populate feedback with the actual error type and message
    result.feedback = f"Agent execution failed with {type(e).__name__}: {e}"
```

## 6. API Changes

None. This is a behavioral bug fix that makes the existing API function as users would intuitively expect. The user-facing API of `make_agent_async`, `Step`, `Pipeline`, and `Flujo` remains unchanged.

## 7. Testing Plan

A new integration test file (`tests/integration/test_context_injection.py`) will be created to validate the fix and prevent regressions.

* **Test Case 1: Stateless Agent (`make_agent_async`)**
  * **Given:** A pipeline with a `Step` using an agent created via `make_agent_async` that is stateless.
  * **When:** The pipeline is executed by the `Flujo` runner with a context model configured.
  * **Then:** The pipeline completes successfully, and no `TypeError` is raised. The `context` is **not** passed to the agent.

* **Test Case 2: Context-Aware Agent (Explicit `context` Param)**
  * **Given:** A pipeline with a `Step` using a custom agent class with `def run(self, data: str, *, context: MyContext)`.
  * **When:** The pipeline is executed with a `context_model`.
  * **Then:** The pipeline completes successfully, and the agent correctly receives and can modify the context.

* **Test Case 3: Context-Aware Agent (`**kwargs`)**
  * **Given:** A pipeline with a `Step` using an agent with `def run(self, data: str, **kwargs)`.
  * **When:** The pipeline is executed with a `context_model`.
  * **Then:** The pipeline completes successfully, and the agent correctly receives the context within its `kwargs`.

* **Test Case 4: Error Propagation**
  * **Given:** A pipeline with an agent that is designed to fail for a reason *other than* a signature mismatch (e.g., raises `ValueError("Internal error")`).
  * **When:** The pipeline is executed.
  * **Then:** The final `StepResult.feedback` string **must** contain the substring `"ValueError: Internal error"`.

## 8. Risks and Mitigation

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Performance degradation from repeated signature inspection. | Medium | This is mitigated by implementing a robust caching mechanism for the signature check results (NFR-13). |
| Some users may have worked around the bug by adding unused `**kwargs` to their agents. | Low | This fix will not break their code. Their agents will continue to receive the context. This is considered an improvement in correctness and is non-breaking. |

## 9. Backward Compatibility

This change is a bug fix that improves backward compatibility. Code that previously failed due to this issue will now work as expected. Code that was written to be context-aware (by correctly including a `context` or `**kwargs` parameter) will continue to work correctly (NFR-14). There are no breaking changes for correctly implemented pipelines.
