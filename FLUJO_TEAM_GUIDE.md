Of course. Here is a new iteration of the Developer Guide, incorporating the minor suggestions and refining the language to be even clearer and more directive. This version is polished and ready to be adopted as the official v2.0 guide for the core team.

---

### **Flujo Team Developer Guide: Architecture, Patterns & Contributions (v2.0)**

This guide is for the Flujo core team—developers building and maintaining the framework itself. It covers architectural principles, contribution patterns, and the critical anti-patterns that ensure Flujo remains robust and maintainable.

## **1. The Golden Rule: Respect the Policy-Driven Architecture**

Flujo's core strength is the **separation between the DSL and the Execution Core via Policies**. This is the most important principle to understand.

### **✅ Core Principle: Everything Goes Through Policies**
*   **When adding new step types**: Create a dedicated policy class in `flujo/application/core/step_policies.py`.
*   **When modifying execution logic**: Update the relevant policy, never the `ExecutorCore` dispatcher.
*   **When handling new exception types**: Use the `ErrorClassifier` and recovery strategy system.

### **❌ Critical Anti-Pattern: Monolithic Execution Logic**
*   **Never** put step-specific business logic directly in `ExecutorCore.execute()`. The `ExecutorCore` is a dispatcher, not an implementer.
*   **Never** create "special case" handling (e.g., `if isinstance(step, MySpecialStep): ...`) that bypasses the policy system.
*   **Never** duplicate execution logic across multiple policies. If logic is shared, extract it into a common utility.

---

## **2. Exception Handling: The Architectural Way**

**This is the most critical pattern for Flujo team members to understand.** Incorrectly handling exceptions can break the entire workflow orchestration system.

### **✅ Control Flow Exception Pattern**

When implementing exceptions that control workflow execution (like `PausedException`):

1. **Register in Error Classification System**:
   ```python
   # In flujo/application/core/optimized_error_handler.py
   self._category_mappings = {
       "YourCustomException": ErrorCategory.CONTROL_FLOW,
   }
   ```

2. **Create a Non-Retryable Recovery Strategy**:
   ```python
   # In _register_default_strategies()
   control_flow_strategy = RecoveryStrategy(
       name="your_control_flow",
       error_types={YourCustomException},
       max_retries=0,  # Never retry control flow exceptions
       primary_action=RecoveryAction.ESCALATE # Escalate immediately to the runner
   )
   ```

3. **Use in Step Policies**:
   ```python
   # In a step policy's main try/except block
   except Exception as e:
       # 1. First, immediately check for and re-raise control flow exceptions.
       if isinstance(e, (PausedException, PipelineAbortSignal, InfiniteRedirectError)):
           raise e  # Let the runner orchestrate.
           
       # 2. For all other exceptions, create a failure StepResult.
       # The runner's retry/fallback logic will handle this result.
       return StepResult(success=False, feedback=f"Step failed: {e}")
   ```

### **❌ The Fatal Anti-Pattern**

**NEVER do this in step policies**:
```python
try:
    result = await core._agent_runner.run(...)
except PausedException as e:
    # ❌ This breaks pause/resume workflows!
    return StepResult(success=False, feedback=str(e))
```
**This converts a control flow exception into a data-level failure, breaking the entire workflow orchestration system.**

---

## **3. Policy Implementation Patterns**

### **✅ Policy Class Structure**

Every policy should follow this pattern:

```python
class DefaultYourStepExecutor:
    async def execute(
        self,
        core: ExecutorCore,
        step: YourStepType,
        data: Any,
        context: Optional[PipelineContext],
        # ... other parameters from ExecutionFrame ...
    ) -> StepResult:
        # 1. Validate inputs
        # 2. Handle caching (if applicable)
        # 3. Execute core logic with proper exception handling
        # 4. Handle context updates/merging
        # 5. Return results
```

> **Note on `ExecutionFrame`**: The `core.execute` method and many policies now accept a single `ExecutionFrame` object instead of a long list of parameters. This is the preferred way to pass execution state. Your policy's `execute` method should be prepared to accept this frame and unpack its contents when calling `core.execute` recursively.

### **✅ Context Management in Policies**

Always use the centralized utilities for safety and consistency.

```python
# For isolation (parallel branches, loop iterations)
from flujo.application.core.context_manager import ContextManager
isolated_context = ContextManager.isolate(parent_context)

# For merging a single branch's context back into the main context
ContextManager.merge(main_context, successful_branch_context)

# For applying a dictionary of updates to an existing context
from flujo.utils.context import safe_merge_context_updates
safe_merge_context_updates(target_context, source_with_updates)
```

### **❌ Anti-Pattern: Direct Context Manipulation**
```python
# ❌ Don't do this
context.some_field = new_value  # This bypasses validation and won't persist correctly
```

---

## **4. Agent and Configuration Management**

### **✅ Agent Creation and Usage**

*   **Factory:** Always use `flujo.agents.factory.make_agent` for creating low-level `pydantic-ai` agents. This centralizes API key handling.
*   **Wrapper:** Use `flujo.agents.wrapper.make_agent_async` to get a production-ready agent with retries, timeouts, and auto-repair.
*   **Recipes:** For common use cases (review, solution), use the factory functions in `flujo.agents.recipes`.

### **✅ Configuration Access Pattern**

*   **Canonical Source:** The `flujo.infra.config_manager.ConfigManager` is the single source of truth for all configuration.
*   **Accessing Settings:** Use the global `get_settings()` function from `flujo.infra.settings`, which now delegates to the `ConfigManager`.
*   **Accessing CLI/TOML Values:** Use the helper functions `get_cli_defaults()` and `get_state_uri()` from `flujo.infra.config_manager`.

### **❌ Anti-Pattern: Decentralized Configuration**
*   **Never** read `flujo.toml` directly from any module other than `ConfigManager`.
*   **Never** check environment variables directly for settings; rely on the `pydantic-settings` behavior within the `Settings` class, which is then managed by `ConfigManager`.

---

## **5. Adding New Step Types: The Complete Pattern**

When adding a new step type to Flujo:

### **Step 1: Define the Step Class**
```python
# In a new module like flujo/domain/dsl/your_step.py
from flujo.domain.dsl import Step

class YourCustomStep(Step[InputType, OutputType]):
    # Step-specific configuration
    your_config: YourConfigType

    @property
    def is_complex(self) -> bool:
        # Mark as complex to ensure it's routed by the dispatcher
        # to its dedicated policy, not the simple step handler.
        return True
```

### **Step 2: Create the Policy**
```python
# In flujo/application/core/step_policies.py
class DefaultYourCustomStepExecutor:
    async def execute(self, core, step, data, context, ...):
        # Implementation following the policy pattern
```

### **Step 3: Register in ExecutorCore**
```python
# In flujo/application/core/ultra_executor.py

# 1. Update the constructor to accept the new policy
class ExecutorCore:
    def __init__(self, ..., your_custom_step_executor=None):
        # ...
        self.your_custom_step_executor = your_custom_step_executor or DefaultYourCustomStepExecutor()

# 2. Update the dispatcher to use the injected policy
async def execute(self, frame: ExecutionFrame, ...):
    step = frame.step
    if isinstance(step, YourCustomStep):
        return await self.your_custom_step_executor.execute(self, frame)
```

### **Step 4: Add Tests**
- Unit tests for the policy class in isolation.
- Integration tests for the complete step execution within a pipeline.
- Regression tests for any identified edge cases.

---

*(Sections on Error Recovery, Testing Patterns, Performance, Debugging, and the Code Review Checklist remain highly relevant and are omitted here for brevity. Their principles have been reinforced by the recent refactors.)*

---

## **10. Common Pitfalls and Their Solutions (Updated)**

### **❌ Pitfall: Context Mutation Instead of Updates**
```python
# ❌ Wrong
# In a policy, you might be tempted to directly change the context.
# This bypasses validation and can lead to inconsistent state.
context.some_field = new_value

# ✅ Correct Architectural Pattern
# Use `safe_merge_context_updates` from `flujo.utils.context` to apply changes.
# This ensures validation is respected and is the canonical way to merge state.
from flujo.utils.context import safe_merge_context_updates

# Create a source context or dict with the updates
updates_to_apply = YourContextModel(some_field=new_value)

# Safely merge the changes into the main context
safe_merge_context_updates(main_context, updates_to_apply)
```

### **❌ Pitfall: Exception Swallowing**
```python
# ❌ Wrong
try:
    result = await operation()
except Exception:
    return StepResult(success=False)  # Lost the original exception!

# ✅ Correct
try:
    result = await operation()
except Exception as e:
    # Use the error classification system to determine the nature of the error
    error_context = ErrorContext.from_exception(e)
    classifier.classify_error(error_context)
    
    # Re-raise control flow exceptions to let the runner handle them
    if error_context.category == ErrorCategory.CONTROL_FLOW:
        raise e
    
    # For other exceptions, create a rich failure result
    return StepResult(success=False, feedback=f"Step failed: {e}")
```

### **❌ Pitfall: Policy Logic in ExecutorCore**
```python
# ❌ Wrong - in ExecutorCore.execute()
if isinstance(step, LoopStep):
    # Complex loop logic here...

# ✅ Correct - in ExecutorCore.execute()
if isinstance(step, LoopStep):
    # Delegate to the injected policy
    return await self.loop_step_executor.execute(self, frame)
```

---

## **11. Engineering Excellence: Critical Lessons from Production (NEW)**

*These lessons were learned during FSD-008 completion and represent critical engineering principles that every team member must internalize.*

### **🎯 LESSON 1: Never Adjust Test Expectations to Make Tests Pass**

**❌ The Dangerous Anti-Pattern:**
```python
# ❌ NEVER DO THIS - Masking real issues
def test_fallback_behavior():
    result = await execute_step_with_failing_processor()
    # Originally expected: assert result.output == "fallback success"
    # Changed to mask issue: assert result.output == "primary success"  # ❌ WRONG!
```

**✅ The Engineering Excellence Pattern:**
```python
# ✅ ALWAYS DO THIS - Investigate and fix root causes
def test_fallback_behavior():
    result = await execute_step_with_failing_processor()
    # Test fails? Investigate WHY the fallback isn't triggering
    # Fix the underlying processor error handling logic
    # Keep the original expectation: assert result.output == "fallback success"
```

**Critical Principle:** Test failures are symptoms pointing to real bugs. Changing test expectations to make them pass masks real regressions and creates technical debt.

### **🎯 LESSON 2: Never Modify Performance Thresholds to Hide Problems**

**❌ The Dangerous Anti-Pattern:**
```python
# ❌ NEVER DO THIS - Hiding performance regressions
def get_performance_threshold():
    # Originally: return 1700.0  # 17x overhead limit
    # Changed to hide issue: return 2100.0  # ❌ WRONG! This hides real performance problems
```

**✅ The Engineering Excellence Pattern:**
```python
# ✅ ALWAYS DO THIS - Fix the underlying performance issues
def test_performance_overhead():
    # Performance test failing? Investigate WHY
    # Common causes: resource contention, parallel test interference, memory leaks
    # Fix: Proper test isolation, resource cleanup, worker-specific resources
    # Keep original thresholds to maintain performance standards
```

**Critical Principle:** Performance thresholds exist to catch regressions. Raising them to make tests pass allows real performance problems to slip into production.

### **🎯 LESSON 3: Root Cause Analysis Over Symptom Treatment**

**Real Example from FSD-008:**

**❌ Symptom Treatment (What We Initially Did Wrong):**
- Test shows processor failure doesn't trigger fallback → Change test to expect different behavior
- Performance test exceeds threshold → Increase threshold
- Test expects step failure but step succeeds → Change test to expect success

**✅ Root Cause Analysis (What We Did Right):**
```python
# Investigation revealed the real issue:
class DefaultProcessorPipeline:
    async def apply_prompt(self, processors, data, context):
        for proc in processors:
            try:
                processed_data = await proc.process(data)
            except Exception as e:
                # ❌ BUG: Silently swallowing exceptions!
                telemetry.logfire.error(f"Processor failed: {e}")
                processed_data = data  # Continue with original data
        return processed_data

# ✅ PROPER FIX: Re-raise exceptions to fail the step as expected
            except Exception as e:
                telemetry.logfire.error(f"Processor failed: {e}")
                raise e  # Let the step fail properly
```

### **🎯 LESSON 4: First Principles Debugging Methodology**

When tests fail, follow this methodology:

1. **Question the Change, Not the Test:**
   - "What did our changes break?" (not "Why is this test wrong?")
   - "What behavior regressed?" (not "How can we make this pass?")

2. **Trace the Execution Path:**
   - Follow the actual code execution from test input to output
   - Identify where the expected behavior diverges from actual behavior

3. **Find the Code Change That Caused the Regression:**
   - Use git blame, commit history, and systematic code review
   - Focus on changes made during the current work session

4. **Fix the Code, Not the Test:**
   - Restore the expected behavior in the implementation
   - Only change tests if the original behavior was genuinely incorrect

### **🎯 LESSON 5: Test Integrity Is Sacred**

**Core Principles:**
- **Tests are contracts** - They define the expected behavior of the system
- **Test failures are valuable signals** - They catch regressions before they reach users
- **Changing tests to pass is technical debt** - It weakens the safety net for future changes

**When Is It OK to Change a Test?**
✅ **Acceptable reasons:**
- Adding new test cases for uncovered scenarios
- Fixing tests that were testing the wrong behavior originally
- Updating tests when requirements legitimately change

**❌ **Never acceptable:**
- Making tests pass after your changes broke them
- Adjusting thresholds because performance regressed
- Changing expectations because the system behavior changed unexpectedly

### **🎯 LESSON 6: Performance Vigilance**

**Performance Test Philosophy:**
- Performance tests are **regression detectors**, not flexibility buffers
- Thresholds should be **realistic but strict** - based on actual production requirements
- **Investigate every performance regression** - don't mask them with higher thresholds

**When Performance Tests Fail:**
1. **Investigate first** - What changed? Resource contention? Algorithm regression?
2. **Fix the root cause** - Optimize the code, fix resource leaks, improve test isolation
3. **Only adjust thresholds** if investigation proves the original threshold was unrealistic

---

## **12. Engineering Excellence Checklist**

Before committing any code changes, ask yourself:

- [ ] **Did I change any test expectations?** If yes, can I justify why the old expectation was wrong?
- [ ] **Did I adjust any performance thresholds?** If yes, did I investigate and fix the underlying performance issue first?
- [ ] **Are all test failures caused by legitimate bugs in my code?** Have I fixed the root causes?
- [ ] **Did I follow first principles debugging?** Did I trace execution paths and find the real issue?
- [ ] **Will future developers understand why this change was made?** Is it documented in commit messages?

---

This guide ensures that all Flujo core team members build features that are architecturally consistent, performant, and maintainable. The patterns here have been battle-tested and should be followed religiously to maintain Flujo's quality and reliability.