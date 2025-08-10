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
   from typing import Dict, Type
   from flujo.application.core.error_categories import ErrorCategory

   class ErrorClassifier:
       def __init__(self) -> None:
           # Type-safe category mappings with explicit typing
           self._category_mappings: Dict[str, ErrorCategory] = {
               "YourCustomException": ErrorCategory.CONTROL_FLOW,
           }
   ```

2. **Create a Non-Retryable Recovery Strategy**:
   ```python
   # In _register_default_strategies()
   from typing import Set
   from flujo.application.core.recovery_strategies import RecoveryStrategy, RecoveryAction

   def _register_default_strategies(self) -> None:
       """Register default recovery strategies with proper typing."""
       # Type-safe error type sets
       control_flow_exceptions: Set[Type[Exception]] = {YourCustomException}

       control_flow_strategy = RecoveryStrategy(
           name="your_control_flow",
           error_types=control_flow_exceptions,
           max_retries=0,  # Never retry control flow exceptions
           primary_action=RecoveryAction.ESCALATE  # Escalate immediately to the runner
       )
   ```

3. **Use in Step Policies**:
   ```python
   # In a step policy's main try/except block
   from typing import Union
   from flujo.domain.models import StepResult
   from flujo.application.core.exceptions import PausedException, PipelineAbortSignal, InfiniteRedirectError

   async def execute(self, core: ExecutorCore, step: Step, data: Any, context: Optional[PipelineContext]) -> StepResult:
       try:
           # Your step logic here
           result: Any = await self._execute_step_logic(step, data, context)
           return StepResult(success=True, output=result)

       except Exception as e:
           # 1. First, immediately check for and re-raise control flow exceptions.
           if isinstance(e, (PausedException, PipelineAbortSignal, InfiniteRedirectError)):
               raise e  # Let the runner orchestrate.

           # 2. For all other exceptions, create a failure StepResult.
           # The runner's retry/fallback logic will handle this result.
           error_feedback: str = f"Step failed: {e}"
           return StepResult(success=False, feedback=error_feedback)
   ```

### **❌ The Fatal Anti-Pattern**

**NEVER do this in step policies**:
```python
from typing import Any, Optional
from flujo.domain.models import StepResult, PipelineContext
from flujo.application.core.exceptions import PausedException

async def execute(self, core: ExecutorCore, step: Step, data: Any, context: Optional[PipelineContext]) -> StepResult:
    try:
        # Execute agent runner with proper typing
        result: Any = await core._agent_runner.run(step, data, context)
        return StepResult(success=True, output=result)

    except PausedException as e:
        # ❌ This breaks pause/resume workflows!
        # Converting control flow exception to data failure breaks orchestration
        error_feedback: str = str(e)
        return StepResult(success=False, feedback=error_feedback)
```
**This converts a control flow exception into a data-level failure, breaking the entire workflow orchestration system.**

---

## **3. Policy Implementation Patterns**

### **✅ Policy Class Structure**

Every policy should follow this pattern with complete type safety:

```python
from typing import Any, Optional, Dict, List
from flujo.application.core import ExecutorCore
from flujo.domain.dsl import Step
from flujo.domain.models import StepResult, PipelineContext
from flujo.application.core.execution_frame import ExecutionFrame

class DefaultYourStepExecutor:
    """Executor policy for YourStep with complete type safety."""

    async def execute(
        self,
        core: ExecutorCore,
        step: "YourStepType",  # Use string for forward references
        data: Any,
        context: Optional[PipelineContext],
        execution_id: str,
        step_id: str,
        # ... other parameters from ExecutionFrame ...
    ) -> StepResult:
        """Execute the step with complete type safety.

        Args:
            core: The executor core for recursive execution
            step: The step instance to execute
            data: Input data for the step
            context: Optional pipeline context
            execution_id: Unique execution identifier
            step_id: Unique step identifier

        Returns:
            StepResult: Execution result with success/failure status
        """
        # 1. Validate inputs with proper typing
        validation_errors: List[str] = []

        # 2. Handle caching (if applicable)
        cache_key: str = f"{execution_id}:{step_id}"

        # 3. Execute core logic with proper exception handling
        try:
            result: StepResult = await self._execute_core_logic(step, data, context)
            return result
        except Exception as e:
            error_feedback: str = f"Step execution failed: {e}"
            return StepResult(success=False, feedback=error_feedback)

        # 4. Handle context updates/merging
        # 5. Return results
```

> **Note on `ExecutionFrame`**: The `core.execute` method and many policies now accept a single `ExecutionFrame` object instead of a long list of parameters. This is the preferred way to pass execution state. Your policy's `execute` method should be prepared to accept this frame and unpack its contents when calling `core.execute` recursively.

### **✅ Context Management in Policies**

Always use the centralized utilities for safety and consistency with proper typing:

```python
from typing import Dict, Any, Optional
from flujo.application.core.context_manager import ContextManager
from flujo.domain.models import PipelineContext
from flujo.utils.context import safe_merge_context_updates

async def execute_with_context_isolation(
    self,
    parent_context: PipelineContext,
    branch_data: Dict[str, Any]
) -> PipelineContext:
    """Execute with proper context isolation and typing."""

    # For isolation (parallel branches, loop iterations)
    isolated_context: PipelineContext = ContextManager.isolate(parent_context)

    # For merging a single branch's context back into the main context
    ContextManager.merge(parent_context, isolated_context)

    # For applying a dictionary of updates to an existing context
    # Use explicit typing for the updates dictionary
    updates_dict: Dict[str, Any] = {"branch_result": branch_data}
    safe_merge_context_updates(parent_context, updates_dict)

    return parent_context
```

### **❌ Anti-Pattern: Direct Context Manipulation**
```python
# ❌ Don't do this
context.some_field = new_value  # This bypasses validation and won't persist correctly
```

---

## **3.5 Idempotency in Step Policies (NEW)**

**This is a critical architectural principle that emerged during the refactor and is essential for robust workflow execution.**

### **✅ Core Principle: Step Execution Must Be Idempotent**

Step execution, especially within retries, must be **idempotent with respect to the pipeline context**. A failed attempt should not "poison" the context for the next attempt.

### **✅ The Correct Pattern: Context Isolation for Complex Steps**

The `ExecutorCore` automatically handles context isolation for retries in `_execute_simple_step`. For complex steps like `LoopStep` and `ParallelStep`, you **must do this manually**.

```python
# In a policy for a complex step (e.g., LoopStep)
from flujo.application.core.context_manager import ContextManager
from typing import Any, Dict, Optional, List
from flujo.domain.models import PipelineContext, StepResult
from flujo.domain.dsl import Step
from flujo.application.core.execution_frame import ExecutionFrame

class DefaultLoopStepExecutor:
    """Executor policy for LoopStep with proper context isolation."""

    async def execute(
        self,
        core: ExecutorCore,
        step: "LoopStep",  # Use string for forward references
        data: Any,
        context: Optional[PipelineContext],
        execution_id: str,
        step_id: str
    ) -> StepResult:
        """Execute loop with proper context isolation for idempotency."""
        
        # Store the original context for merging results
        current_context: PipelineContext = context or PipelineContext()
        loop_results: List[Any] = []
        
        # ... inside the loop ...
        for iteration in range(max_loops):
            # ✅ Create a pristine, isolated context for this iteration
            iteration_context: PipelineContext = ContextManager.isolate(current_context)
            
            # Execute the loop body with the isolated context
            iteration_result: StepResult = await core.execute(
                frame=ExecutionFrame(
                    step=step.body,
                    data=data,
                    context=iteration_context,
                    execution_id=execution_id,
                    step_id=f"{step_id}_iteration_{iteration}"
                )
            )
            
            # Only merge the context back if the iteration was successful
            if iteration_result.success:
                ContextManager.merge(current_context, iteration_context)
                loop_results.append(iteration_result.output)
            else:
                # Failed iteration doesn't affect the main context
                # This ensures idempotency - retry won't see "poisoned" state
                continue
        
        return StepResult(success=True, output=loop_results)
```

### **❌ Anti-Pattern: Mutating Shared Context**

```python
# ❌ NEVER pass the same context object into multiple parallel branches
# or successive loop iterations without isolating it first.

class WrongLoopStepExecutor:
    """❌ WRONG - This breaks idempotency and can cause context corruption."""
    
    async def execute(self, core: ExecutorCore, step: "LoopStep", data: Any, context: PipelineContext) -> StepResult:
        # ❌ This is a bug! All iterations share and mutate the same context object.
        for item in items:
            # ❌ Shared context gets corrupted across iterations
            await core.execute(..., context=context)  # BUG: Same context object!
        
        return StepResult(success=True, output=result)
```

**Why This Matters:**
- **Retry Safety**: If a step fails and retries, it must see the same initial state
- **Parallel Execution**: Multiple branches must not interfere with each other's context
- **Loop Consistency**: Each iteration must start with a clean, predictable state
- **Debugging**: Isolated contexts make it easier to trace execution flow

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
from typing import Any, Dict, Optional
from flujo.domain.dsl import Step
from pydantic import BaseModel, Field

class YourStepConfig(BaseModel):
    """Configuration for YourCustomStep with proper typing."""
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: float = Field(default=30.0, description="Operation timeout")
    custom_option: Optional[str] = Field(default=None, description="Optional custom setting")

class YourCustomStep(Step[Dict[str, Any], Dict[str, Any]]):
    """Custom step with complete type safety and configuration."""

    # Step-specific configuration with explicit typing
    your_config: YourStepConfig

    # Additional step properties with proper types
    step_name: str
    description: Optional[str] = None

    @property
    def is_complex(self) -> bool:
        """Mark as complex to ensure it's routed by the dispatcher
        to its dedicated policy, not the simple step handler."""
        return True

    @property
    def input_type(self) -> type:
        """Return the expected input type for type checking."""
        return Dict[str, Any]

    @property
    def output_type(self) -> type:
        """Return the expected output type for type checking."""
        return Dict[str, Any]
```

### **Step 2: Create the Policy**
```python
# In flujo/application/core/step_policies.py
from typing import Any, Dict, Optional, List
from flujo.application.core import ExecutorCore
from flujo.domain.dsl import Step
from flujo.domain.models import StepResult, PipelineContext
from flujo.domain.dsl.your_step import YourCustomStep, YourStepConfig

class DefaultYourCustomStepExecutor:
    """Executor policy for YourCustomStep with complete type safety."""

    async def execute(
        self,
        core: ExecutorCore,
        step: YourCustomStep,
        data: Dict[str, Any],
        context: Optional[PipelineContext],
        execution_id: str,
        step_id: str
    ) -> StepResult:
        """Execute YourCustomStep with proper error handling and typing.

        Args:
            core: Executor core for recursive execution
            step: YourCustomStep instance with configuration
            data: Input data dictionary
            context: Optional pipeline context
            execution_id: Unique execution identifier
            step_id: Unique step identifier

        Returns:
            StepResult: Success/failure result with output/feedback
        """
        # Extract configuration with proper typing
        config: YourStepConfig = step.your_config

        # Implementation following the policy pattern with type safety
        try:
            # Your step logic here with explicit typing
            result: Dict[str, Any] = await self._process_data(data, config, context)
            return StepResult(success=True, output=result)

        except Exception as e:
            error_feedback: str = f"YourCustomStep failed: {e}"
            return StepResult(success=False, feedback=error_feedback)

    async def _process_data(
        self,
        data: Dict[str, Any],
        config: YourStepConfig,
        context: Optional[PipelineContext]
    ) -> Dict[str, Any]:
        """Process data according to step configuration."""
        # Implementation here
        pass
```

### **Step 3: Register in ExecutorCore**
```python
# In flujo/application/core/ultra_executor.py
from typing import Optional, Any
from flujo.application.core.step_policies import DefaultYourCustomStepExecutor
from flujo.domain.dsl.your_step import YourCustomStep
from flujo.application.core.execution_frame import ExecutionFrame

# 1. Update the constructor to accept the new policy with proper typing
class ExecutorCore:
    def __init__(
        self,
        agent_runner: Any,
        your_custom_step_executor: Optional[DefaultYourCustomStepExecutor] = None,
        # ... other parameters ...
    ) -> None:
        """Initialize ExecutorCore with injected policies."""
        # ... other initialization ...

        # Type-safe policy injection with fallback to default
        self.your_custom_step_executor: DefaultYourCustomStepExecutor = (
            your_custom_step_executor or DefaultYourCustomStepExecutor()
        )

# 2. Update the dispatcher to use the injected policy with type safety
async def execute(self, frame: ExecutionFrame) -> Any:
    """Execute step with proper type checking and policy routing."""
    step: Any = frame.step

    # Type-safe step routing with explicit policy delegation
    if isinstance(step, YourCustomStep):
        # Delegate to the injected policy with complete frame
        return await self.your_custom_step_executor.execute(
            core=self,
            step=step,
            data=frame.data,
            context=frame.context,
            execution_id=frame.execution_id,
            step_id=frame.step_id
        )

    # ... handle other step types ...
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
# ❌ Wrong - Direct context mutation bypasses validation
# In a policy, you might be tempted to directly change the context.
# This bypasses validation and can lead to inconsistent state.
from typing import Any, Dict
from flujo.domain.models import PipelineContext

def wrong_context_update(context: PipelineContext, new_value: str) -> None:
    # ❌ NEVER do this - bypasses validation and type safety
    context.some_field = new_value  # mypy error: no such attribute

# ✅ Correct Architectural Pattern with Type Safety
# Use `safe_merge_context_updates` from `flujo.utils.context` to apply changes.
# This ensures validation is respected and is the canonical way to merge state.
from flujo.utils.context import safe_merge_context_updates
from flujo.domain.models import YourContextModel

def correct_context_update(
    main_context: PipelineContext,
    new_value: str
) -> None:
    """Update context safely with proper typing and validation."""

    # Create a source context or dict with the updates
    updates_to_apply: YourContextModel = YourContextModel(some_field=new_value)

    # Safely merge the changes into the main context
    # This respects validation and maintains type safety
    safe_merge_context_updates(main_context, updates_to_apply)
```

### **❌ Pitfall: Exception Swallowing**
```python
# ❌ Wrong - Exception swallowing loses critical error information
from typing import Any, Optional
from flujo.domain.models import StepResult

async def wrong_exception_handling() -> StepResult:
    try:
        result: Any = await operation()
        return StepResult(success=True, output=result)
    except Exception:
        # ❌ NEVER do this - lost the original exception!
        return StepResult(success=False)  # No feedback about what went wrong

# ✅ Correct - Proper exception handling with type safety
from flujo.application.core.error_categories import ErrorCategory
from flujo.application.core.error_context import ErrorContext
from flujo.application.core.error_classifier import ErrorClassifier

async def correct_exception_handling() -> StepResult:
    """Handle exceptions properly with error classification and typing."""
    try:
        result: Any = await operation()
        return StepResult(success=True, output=result)

    except Exception as e:
        # Use the error classification system to determine the nature of the error
        error_context: ErrorContext = ErrorContext.from_exception(e)
        classifier: ErrorClassifier = ErrorClassifier()
        classifier.classify_error(error_context)

        # Re-raise control flow exceptions to let the runner handle them
        if error_context.category == ErrorCategory.CONTROL_FLOW:
            raise e  # Let the runner orchestrate

        # For other exceptions, create a rich failure result with proper typing
        error_feedback: str = f"Step failed: {e}"
        return StepResult(success=False, feedback=error_feedback)
```

### **❌ Pitfall: Policy Logic in ExecutorCore**
```python
# ❌ Wrong - Putting policy logic directly in ExecutorCore
from typing import Any
from flujo.domain.dsl.loop import LoopStep
from flujo.application.core.execution_frame import ExecutionFrame

class ExecutorCore:
    async def execute(self, frame: ExecutionFrame) -> Any:
        step: Any = frame.step

        # ❌ NEVER do this - ExecutorCore becomes a monolithic implementer
        if isinstance(step, LoopStep):
            # Complex loop logic here...
            # This violates the policy-driven architecture
            loop_result: Any = await self._handle_loop_logic(step, frame)
            return loop_result

# ✅ Correct - Delegate to injected policies with type safety
from flujo.application.core.step_policies import DefaultLoopStepExecutor

class ExecutorCore:
    def __init__(self, loop_step_executor: Optional[DefaultLoopStepExecutor] = None) -> None:
        """Initialize with injected policies."""
        self.loop_step_executor: DefaultLoopStepExecutor = (
            loop_step_executor or DefaultLoopStepExecutor()
        )

    async def execute(self, frame: ExecutionFrame) -> Any:
        """Execute step by delegating to appropriate policy."""
        step: Any = frame.step

        # ✅ ALWAYS do this - delegate to the injected policy
        if isinstance(step, LoopStep):
            # Delegate to the injected policy with complete frame
            return await self.loop_step_executor.execute(
                core=self,
                step=step,
                data=frame.data,
                context=frame.context,
                execution_id=frame.execution_id,
                step_id=frame.step_id
            )
```

---

## **11. Engineering Excellence: Critical Lessons from Production (NEW)**

*These lessons were learned during FSD-008 completion and represent critical engineering principles that every team member must internalize.*

### **🎯 LESSON 1: Never Adjust Test Expectations to Make Tests Pass**

**❌ The Dangerous Anti-Pattern:**
```python
# ❌ NEVER DO THIS - Masking real issues
from typing import Any
from flujo.domain.models import StepResult

async def test_fallback_behavior() -> None:
    """Test fallback behavior with proper typing."""
    result: StepResult = await execute_step_with_failing_processor()

    # Originally expected: assert result.output == "fallback success"
    # Changed to mask issue: assert result.output == "primary success"  # ❌ WRONG!
    # This hides the real bug and creates technical debt
```

**✅ The Engineering Excellence Pattern:**
```python
# ✅ ALWAYS DO THIS - Investigate and fix root causes
from typing import Any
from flujo.domain.models import StepResult

async def test_fallback_behavior() -> None:
    """Test fallback behavior with proper typing and expectations."""
    result: StepResult = await execute_step_with_failing_processor()

    # Test fails? Investigate WHY the fallback isn't triggering
    # Fix the underlying processor error handling logic
    # Keep the original expectation: assert result.output == "fallback success"

    # Verify the fallback mechanism works as expected
    assert result.output == "fallback success", (
        f"Expected fallback success, got: {result.output}. "
        "Investigate why fallback didn't trigger."
    )
```

**Critical Principle:** Test failures are symptoms pointing to real bugs. Changing test expectations to make them pass masks real regressions and creates technical debt.

### **🎯 LESSON 2: Never Modify Performance Thresholds to Hide Problems**

**❌ The Dangerous Anti-Pattern:**
```python
# ❌ NEVER DO THIS - Hiding performance regressions
from typing import float

def get_performance_threshold() -> float:
    """Get performance threshold for overhead tests."""
    # Originally: return 1700.0  # 17x overhead limit
    # Changed to hide issue: return 2100.0  # ❌ WRONG! This hides real performance problems
    # This allows performance regressions to slip into production
    return 2100.0

# ❌ NEVER DO THIS - Adjusting thresholds to mask problems
def test_performance_overhead() -> None:
    """Test performance overhead with adjusted threshold."""
    threshold: float = get_performance_threshold()
    # Test passes with higher threshold, but performance is actually worse
    assert measured_overhead <= threshold  # This hides the real issue
```

**✅ The Engineering Excellence Pattern:**
```python
# ✅ ALWAYS DO THIS - Fix the underlying performance issues
from typing import float

def get_performance_threshold() -> float:
    """Get performance threshold for overhead tests."""
    # Keep original threshold to catch real performance regressions
    return 1700.0  # 17x overhead limit - don't change this!

def test_performance_overhead() -> None:
    """Test performance overhead with proper investigation."""
    # Performance test failing? Investigate WHY
    # Common causes: resource contention, parallel test interference, memory leaks

    # Fix: Proper test isolation, resource cleanup, worker-specific resources
    # Keep original thresholds to maintain performance standards

    threshold: float = get_performance_threshold()
    measured_overhead: float = measure_actual_overhead()

    # Use descriptive assertion messages for debugging
    assert measured_overhead <= threshold, (
        f"Performance overhead {measured_overhead} exceeds threshold {threshold}. "
        "Investigate resource contention, memory leaks, or algorithm regressions."
    )
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
from typing import Any, List, Optional
from flujo.domain.models import PipelineContext
from flujo.application.core.processors import Processor

class DefaultProcessorPipeline:
    """Processor pipeline with proper error handling and typing."""

    async def apply_prompt(
        self,
        processors: List[Processor],
        data: Any,
        context: Optional[PipelineContext]
    ) -> Any:
        """Apply processors to data with proper error handling."""
        processed_data: Any = data

        for proc in processors:
            try:
                processed_data = await proc.process(data)
            except Exception as e:
                # ❌ BUG: Silently swallowing exceptions!
                # This prevents proper error handling and fallback logic
                telemetry.logfire.error(f"Processor failed: {e}")
                processed_data = data  # Continue with original data

        return processed_data

# ✅ PROPER FIX: Re-raise exceptions to fail the step as expected
class FixedProcessorPipeline:
    """Fixed processor pipeline that properly handles errors."""

    async def apply_prompt(
        self,
        processors: List[Processor],
        data: Any,
        context: Optional[PipelineContext]
    ) -> Any:
        """Apply processors to data with proper error propagation."""
        processed_data: Any = data

        for proc in processors:
            try:
                processed_data = await proc.process(data)
            except Exception as e:
                # ✅ FIXED: Log the error and re-raise to fail the step properly
                telemetry.logfire.error(f"Processor failed: {e}")
                raise e  # Let the step fail properly, triggering fallback logic
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

### **🎯 LESSON 7: Be Mindful of the Hot Path (NEW)**

**The policy-driven architecture introduces layers of abstraction for correctness and maintainability. While generally fast, be mindful of performance in code that runs thousands of times per second (e.g., inside a tight loop).**

#### **✅ Best Practices for Performance-Sensitive Code:**

**1. Cache Results:**
```python
from typing import Dict, Any, Optional
from functools import lru_cache

class OptimizedStepExecutor:
    """Step executor with performance optimizations for hot paths."""
    
    def __init__(self) -> None:
        # Cache expensive calculations that are repeated
        self._validation_cache: Dict[str, bool] = {}
        self._config_cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=128)
    def _expensive_validation(self, data_hash: str) -> bool:
        """Cache validation results for repeated data patterns."""
        # Expensive validation logic here
        return validation_result
    
    def _get_cached_config(self, config_key: str) -> Any:
        """Cache configuration lookups for frequently accessed settings."""
        if config_key not in self._config_cache:
            self._config_cache[config_key] = self._load_config(config_key)
        return self._config_cache[config_key]
```

**2. Avoid Object Creation in Loops:**
```python
from typing import List, Any
from flujo.domain.models import StepResult

class OptimizedLoopExecutor:
    """Loop executor that minimizes object creation in hot paths."""
    
    def __init__(self) -> None:
        # Pre-allocate frequently used objects
        self._success_template: StepResult = StepResult(success=True, output=None)
        self._failure_template: StepResult = StepResult(success=False, feedback="")
    
    async def execute_loop(self, items: List[Any]) -> List[StepResult]:
        """Execute loop with minimal object creation."""
        results: List[StepResult] = []
        
        for item in items:
            # ✅ Reuse template objects instead of creating new ones
            if self._process_item(item):
                result: StepResult = StepResult(
                    success=True,
                    output=item
                )
            else:
                result: StepResult = StepResult(
                    success=False,
                    feedback=f"Failed to process {item}"
                )
            
            results.append(result)
        
        return results
```

**3. Profile Before Optimizing:**
```python
import cProfile
import pstats
from typing import Any, Callable

def profile_performance(func: Callable, *args, **kwargs) -> None:
    """Profile function performance to identify real bottlenecks."""
    profiler: cProfile.Profile = cProfile.Profile()
    
    try:
        # Profile the function execution
        profiler.enable()
        result: Any = func(*args, **kwargs)
        profiler.disable()
        
        # Analyze results
        stats: pstats.Stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 time-consuming operations
        
        return result
        
    except Exception as e:
        profiler.disable()
        raise e

# Usage: Profile before optimizing
async def optimize_hot_path() -> None:
    """Profile and optimize performance-critical code."""
    
    # Profile the current implementation
    result: Any = await profile_performance(
        current_implementation, 
        test_data
    )
    
    # Only optimize what the profiler shows as slow
    # Don't guess - let the data guide your optimization
```

#### **❌ Performance Anti-Patterns:**

**1. Premature Optimization:**
```python
# ❌ Don't optimize without profiling
def over_optimized_function(data: List[str]) -> List[str]:
    # ❌ This complexity is unnecessary if the function isn't actually slow
    return [item.upper() for item in data if item and len(item) > 0]

# ✅ Simple and clear - optimize only if profiling shows it's needed
def simple_function(data: List[str]) -> List[str]:
    return [item.upper() for item in data if item]
```

**2. Ignoring the Hot Path:**
```python
# ❌ Don't ignore performance in frequently executed code
async def frequently_called_function() -> None:
    # ❌ This runs thousands of times - every millisecond matters
    expensive_operation()  # Could be cached
    file_io_operation()    # Could be batched
    network_call()         # Could be async

# ✅ Profile and optimize the hot path
async def optimized_frequently_called_function() -> None:
    # ✅ Optimized for the hot path
    cached_result = await self._get_cached_result()
    batched_operations = await self._batch_operations()
    # Network calls only when necessary
```

#### **Performance Monitoring in Production:**

```python
from typing import Dict, Any
import time
from flujo.telemetry import metrics

class PerformanceAwareExecutor:
    """Executor that monitors performance in production."""
    
    async def execute_with_monitoring(self, step: Step, data: Any) -> StepResult:
        """Execute step with performance monitoring."""
        start_time: float = time.time()
        
        try:
            result: StepResult = await self._execute_step(step, data)
            
            # Record performance metrics
            execution_time: float = time.time() - start_time
            metrics.record_step_execution_time(step.step_id, execution_time)
            
            return result
            
        except Exception as e:
            # Record error metrics
            execution_time: float = time.time() - start_time
            metrics.record_step_error(step.step_id, execution_time, str(e))
            raise e
```

**Key Principles:**
- **Profile First**: Use `cProfile` or similar tools to identify real bottlenecks
- **Cache Wisely**: Cache expensive operations that are repeated
- **Minimize Allocation**: Avoid creating objects in tight loops
- **Monitor Production**: Track performance metrics to catch regressions
- **Optimize Incrementally**: Make small, measured improvements rather than big changes

---

## **12. Type Safety and Code Quality Maintenance (NEW)**

*These lessons were learned during the systematic fixing of 161 mypy errors and represent critical engineering principles that every team member must internalize to prevent type annotation debt accumulation.*

### **🎯 LESSON 7: Never Let Type Errors Accumulate**

**❌ The Dangerous Anti-Pattern:**
- Adding new code without type annotations
- Ignoring mypy errors during development
- Running `make all` only before releases
- Allowing type annotation debt to accumulate (like the 161 errors we just fixed)

**✅ The Engineering Excellence Pattern:**
- **Every commit must pass `make all`** - no exceptions
- **Fix type errors immediately** when they appear
- **Run type checking during development**, not just before commits
- **Treat mypy errors as blocking issues**, not technical debt

### **🎯 LESSON 8: Type Annotation Standards**

**Required Type Annotations - ALWAYS include these:**

```python
# ✅ ALWAYS annotate these with explicit types:
from typing import Any, Dict, List, Optional, Union
from flujo.domain.models import StepResult, PipelineContext

def process_data(
    input_data: str,
    config: Dict[str, Any],
    context: Optional[PipelineContext] = None
) -> StepResult:
    """Process input data according to configuration.

    Args:
        input_data: The string data to process
        config: Configuration dictionary with processing options
        context: Optional pipeline context for state management

    Returns:
        StepResult: Success/failure result with feedback
    """
    # Local variables must have explicit types
    processed_items: List[str] = []
    metadata: Dict[str, Any] = {}

    # Dictionary/list comprehensions need explicit types
    filtered_data: List[str] = [item for item in input_data.split() if item]

    # Return type must match the declared return type
    return StepResult(success=True, feedback="Processing completed")

# ❌ NEVER leave these untyped - this creates technical debt:
def process_data(input_data, config, context=None):  # Missing types
    result = []  # Missing type annotation
    return result  # Missing return type
```

**Type Annotation Checklist - Verify every function:**
- [ ] All function parameters have explicit type annotations
- [ ] All function return types are specified with `-> ReturnType`
- [ ] All local variables are typed when initialized: `var: Type = value`
- [ ] All dictionary/list comprehensions have explicit types: `result: List[Type] = [...]`
- [ ] All `Any` types are justified and documented with `# type: ignore` comments explaining why
- [ ] All generic types use proper syntax: `Dict[str, Any]`, `List[StepResult]`

### **🎯 LESSON 9: Daily Code Quality Practices**

**Before Every Commit - This is MANDATORY:**
1. **Run `make all`** - This must pass with 0 errors (no exceptions)
2. **Fix any mypy errors** - Don't commit with type errors
3. **Fix any linting issues** - Unused variables, imports, etc.
4. **Run relevant tests** - Ensure no regressions

**During Development - Don't wait until the end:**
1. **Run `make all` frequently** - Every 10-15 minutes of coding
2. **Use IDE type checking** - Enable real-time mypy integration
3. **Fix type errors as you go** - Don't let them accumulate
4. **Use type stubs** - Create `.pyi` files for external libraries if needed

### **🎯 LESSON 10: Preventing Type Error Accumulation**

**Team Practices - These are non-negotiable:**
- **Type Safety Gate**: No PR can be merged if `make all` fails
- **Regular Type Audits**: Weekly runs of `make all` on the entire codebase
- **Type Error Budget**: Maximum of 5 mypy errors allowed at any time
- **Immediate Fix Policy**: Type errors must be fixed within 24 hours

**Code Review Requirements - Every reviewer must check:**
- [ ] Does this code have complete type annotations?
- [ ] Does `make all` pass with this change?
- [ ] Are there any `Any` types that could be more specific?
- [ ] Does this change introduce new type errors?
- [ ] Are all local variables properly typed?

### **🎯 LESSON 11: Type Error Recovery Strategy**

**When Type Errors Accumulate (Current Situation - 161 errors):**
1. **Stop new feature development** - Focus on type safety first
2. **Systematic fixing approach**:
   - Start with linting errors (easier to fix)
   - Then fix mypy errors file by file
   - Re-run `make all` after each file to track progress
3. **Set daily targets** - Aim to reduce errors by 20-30 per day
4. **Document patterns** - Create templates for common type fixes

**Prevention for Future - Implement these safeguards:**
- **Automated Type Checking**: Integrate mypy into CI/CD pipeline
- **Pre-commit Hooks**: Run `make all` before allowing commits
- **Type Safety Metrics**: Track type annotation coverage over time
- **IDE Integration**: Enforce mypy checking in all development environments

### **🎯 LESSON 12: Common Type Error Patterns and Solutions**

**Pattern 1: Missing Type Annotations**
```python
# ❌ Problem: Missing type annotations cause mypy errors
def format_cost(value) -> str:
    return f"${value:.2f}"

# ✅ Solution: Add explicit type annotations
def format_cost(value: Union[float, int]) -> str:
    """Format cost value as currency string.

    Args:
        value: Numeric cost value (float or int)

    Returns:
        Formatted currency string
    """
    return f"${value:.2f}"
```

**Pattern 2: Dictionary Access on Potentially Non-Dictionary Types**
```python
# ❌ Problem: Mypy can't guarantee result["metadata"] is a dict
result: Dict[str, Any] = {"content": "data", "metadata": {}}
result["metadata"]["usage"] = {"tokens": 100}  # mypy error

# ✅ Solution: Use explicit typing and intermediate variables
result: Dict[str, Any] = {"content": "data", "metadata": {}}
metadata: Dict[str, Any] = result["metadata"]
metadata["usage"] = {"tokens": 100}  # mypy happy
```

**Pattern 3: Attribute Access on Potentially None Types**
```python
# ❌ Problem: Accessing attributes without null checks
def process_response(response: Optional[Response]) -> str:
    return response.content  # mypy error: response might be None

# ✅ Solution: Add explicit null checks
def process_response(response: Optional[Response]) -> str:
    if response is None:
        raise ValueError("Response cannot be None")
    return response.content  # mypy happy
```

**Pattern 4: List Operations with Incompatible Types**
```python
# ❌ Problem: Appending potentially None values to typed lists
feedback_list: List[str] = []
feedback_list.append(result.feedback)  # mypy error if feedback can be None

# ✅ Solution: Add null checks before appending
feedback_list: List[str] = []
if result.feedback is not None:
    feedback_list.append(result.feedback)  # mypy happy
```

### **🎯 LESSON 13: Type Safety in Policy Classes**

**Policy Class Type Safety - Always follow this pattern:**

```python
from typing import Any, Dict, List, Optional, Union
from flujo.application.core import ExecutorCore
from flujo.domain.dsl import Step
from flujo.domain.models import StepResult, PipelineContext

class DefaultYourStepExecutor:
    """Executor policy for YourStep with complete type safety."""

    async def execute(
        self,
        core: ExecutorCore,
        step: "YourStepType",  # Use string for forward references
        data: Any,
        context: Optional[PipelineContext],
        execution_id: str,
        step_id: str
    ) -> StepResult:
        """Execute the step with complete type safety.

        Args:
            core: The executor core for recursive execution
            step: The step instance to execute
            data: Input data for the step
            context: Optional pipeline context
            execution_id: Unique execution identifier
            step_id: Unique step identifier

        Returns:
            StepResult: Execution result with success/failure status
        """
        # Local variables with explicit types
        step_history: List[Any] = []
        validation_errors: List[str] = []

        try:
            # Your step logic here
            result: StepResult = await self._execute_core_logic(step, data, context)
            return result

        except Exception as e:
            # Handle exceptions with proper typing
            error_msg: str = f"Step execution failed: {e}"
            return StepResult(success=False, feedback=error_msg)

    async def _execute_core_logic(
        self,
        step: "YourStepType",
        data: Any,
        context: Optional[PipelineContext]
    ) -> StepResult:
        """Execute the core logic with proper error handling."""
        # Implementation here
        pass
```

---

## **14. Engineering Excellence Checklist (UPDATED)**

Before committing any code changes, ask yourself:

- [ ] **Did I change any test expectations?** If yes, can I justify why the old expectation was wrong?
- [ ] **Did I adjust any performance thresholds?** If yes, did I investigate and fix the underlying performance issue first?
- [ ] **Are all test failures caused by legitimate bugs in my code?** Have I fixed the root causes?
- [ ] **Did I follow first principles debugging?** Did I trace execution paths and find the real issue?
- [ ] **Will future developers understand why this change was made?** Is it documented in commit messages?
- [ ] **Does `make all` pass with 0 errors?** Have I fixed all type and linting issues?
- [ ] **Are all my functions and variables properly typed?** Have I added complete type annotations?
- [ ] **Did I run type checking during development?** Or did I wait until the end?
- [ ] **Are there any `Any` types that could be more specific?** Have I minimized type ambiguity?
- [ ] **Does this change introduce new type errors?** Have I verified mypy is happy?

---

This guide ensures that all Flujo core team members build features that are architecturally consistent, performant, maintainable, and **type-safe**. The patterns here have been battle-tested and should be followed religiously to maintain Flujo's quality, reliability, and prevent the accumulation of technical debt like the 161 mypy errors we just systematically resolved.
