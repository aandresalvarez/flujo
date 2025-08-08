# **Flujo Team Developer Guide: Architecture, Patterns & Contributions**

This guide is for the Flujo core team - developers building and maintaining the framework itself. It covers architectural principles, contribution patterns, and the critical anti-patterns that ensure Flujo remains robust and maintainable.

## **1. The Golden Rule: Respect the Policy-Driven Architecture**

Flujo's core strength is the **separation between DSL and Execution Core via Policies**.

### **✅ Core Principle: Everything Goes Through Policies**
*   **When adding new step types**: Create a dedicated policy class in `flujo/application/core/step_policies.py`
*   **When modifying execution logic**: Update the relevant policy, never the `ExecutorCore` dispatcher
*   **When handling new exception types**: Use the `ErrorClassifier` and recovery strategy system

### **❌ Critical Anti-Pattern: Monolithic Execution Logic**
*   **Never** put step-specific business logic directly in `ExecutorCore.execute()`
*   **Never** create "special case" handling that bypasses the policy system
*   **Never** duplicate execution logic across multiple policies

---

## **2. Exception Handling: The Architectural Way**

**This is the most critical pattern for Flujo team members to understand.**

### **✅ Control Flow Exception Pattern**

When implementing exceptions that control workflow execution (like `PausedException`):

1. **Register in Error Classification System**:
   ```python
   # In flujo/application/core/optimized_error_handler.py
   self._category_mappings = {
       "YourCustomException": ErrorCategory.CONTROL_FLOW,
   }
   ```

2. **Create Recovery Strategy**:
   ```python
   # In _register_default_strategies()
   control_flow_strategy = RecoveryStrategy(
       name="your_control_flow",
       error_types={YourCustomException},
       max_retries=0,  # Never retry control flow exceptions
       primary_action=RecoveryAction.ESCALATE
   )
   ```

3. **Use in Step Policies**:
   ```python
   # In step policies
   try:
       result = await core._agent_runner.run(...)
   except Exception as e:
       error_context = ErrorContext.from_exception(e, step_name=step.name)
       classifier.classify_error(error_context)
       
       if error_context.category == ErrorCategory.CONTROL_FLOW:
           # Re-raise immediately - never convert to StepResult
           raise e
   ```

### **❌ The Fatal Anti-Pattern**

**NEVER do this in step policies**:
```python
try:
    result = await core._agent_runner.run(...)
except PausedException as e:
    # ❌ This breaks pause/resume workflows!
    return StepResult(success=False, error=str(e))
```

**This converts control flow exceptions to failed results, breaking the entire workflow control system.**

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
        resources: Optional[AppResources],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable],
        cache_key: Optional[str],
        breach_event: Optional[Any],
    ) -> Union[StepResult, AsyncIterator[StepResult]]:
        # 1. Validate inputs
        # 2. Handle caching (if applicable)
        # 3. Execute core logic with proper exception handling
        # 4. Handle context updates/merging
        # 5. Return results
```

### **✅ Context Management in Policies**

Always use the centralized `ContextManager`:

```python
# For isolation (parallel branches, loop iterations)
isolated_context = ContextManager.isolate(parent_context)

# For merging (after parallel execution)
merged_context = ContextManager.merge_contexts([ctx1, ctx2, ctx3])
```

### **❌ Anti-Pattern: Direct Context Manipulation**
```python
# ❌ Don't do this
context.some_field = new_value  # This won't persist correctly
```

---

## **4. Adding New Step Types: The Complete Pattern**

When adding a new step type to Flujo:

### **Step 1: Define the Step Class**
```python
# In flujo/domain/dsl/
@dataclass
class YourCustomStep(Step[InputType, OutputType]):
    # Step-specific configuration
    your_config: YourConfigType
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
async def execute(self, step: Step, ...):
    if isinstance(step, YourCustomStep):
        policy = DefaultYourCustomStepExecutor()
        return await policy.execute(self, step, ...)
```

### **Step 4: Add Tests**
- Unit tests for the policy class
- Integration tests for the complete step execution
- Regression tests for edge cases

---

## **5. Error Recovery Strategy Development**

### **✅ Strategy Registration Pattern**

When creating new recovery strategies:

```python
# In flujo/application/core/optimized_error_handler.py
def _register_default_strategies(self):
    # Existing strategies...
    
    your_strategy = RecoveryStrategy(
        name="your_strategy",
        error_types={YourExceptionType},
        error_patterns=["pattern1", "pattern2"],
        max_retries=3,
        retry_delay_seconds=1.0,
        primary_action=RecoveryAction.RETRY,
        fallback_actions=[RecoveryAction.FALLBACK],
        applies_to_categories={ErrorCategory.YOUR_CATEGORY},
    )
    self.register_strategy(your_strategy)
```

---

## **6. Testing Patterns for Core Contributors**

### **✅ Policy Testing Pattern**

```python
# Test the policy directly
async def test_your_step_executor():
    policy = DefaultYourStepExecutor()
    mock_core = MockExecutorCore()
    step = YourCustomStep(...)
    
    result = await policy.execute(mock_core, step, data, context, ...)
    
    assert result.success
    assert result.output == expected_output
```

### **✅ Exception Propagation Testing**

```python
async def test_control_flow_exception_propagation():
    # Test that control flow exceptions are NOT converted to StepResult
    with pytest.raises(PausedException):
        await policy.execute(core, step, data, context, ...)
```

### **✅ Context Isolation Testing**

```python
async def test_context_isolation():
    # Ensure context changes in one branch don't affect others
    original_context = create_test_context()
    isolated = ContextManager.isolate(original_context)
    
    # Modify isolated context
    isolated.some_field = "changed"
    
    # Original should be unchanged
    assert original_context.some_field != "changed"
```

---

## **7. Performance and Memory Management**

### **✅ Use Flujo's Optimization Systems**

```python
# Leverage the object pool for common objects
from flujo.application.core.optimization.memory import get_global_object_pool
pool = get_global_object_pool()

# Use optimized context operations
from flujo.application.core.optimization.memory import OptimizedContextManager
context_manager = OptimizedContextManager()
```

### **✅ Telemetry Integration**

Always add telemetry to new components:

```python
from flujo.telemetry import get_global_telemetry
telemetry = get_global_telemetry()

# Log important events
telemetry.logfire.info(f"YourStep '{step.name}' starting execution")

# Record metrics
telemetry.increment_counter("flujo.your_step.executions")
```

---

## **8. Debugging Core Framework Issues**

### **Step 1: Use the ExecutorCore Diagnostics**
```python
# Enable verbose logging
telemetry.logfire.set_level("DEBUG")

# Use the tracer for execution flow
from flujo.console_tracer import ConsoleTracer
tracer = ConsoleTracer(log_inputs=True, log_outputs=True)
```

### **Step 2: Policy-Level Debugging**
```python
# Add debugging to policy execute methods
async def execute(self, core, step, data, context, ...):
    telemetry.logfire.debug(f"Policy {self.__class__.__name__} executing {step.name}")
    telemetry.logfire.debug(f"Input data: {data}")
    telemetry.logfire.debug(f"Context state: {context.model_dump() if context else None}")
```

### **Step 3: Exception Flow Tracing**
```python
# Trace exception propagation
try:
    result = await some_operation()
except Exception as e:
    telemetry.logfire.error(f"Exception in {self.__class__.__name__}: {type(e).__name__}: {str(e)}")
    telemetry.logfire.error(f"Exception category: {error_context.category}")
    raise  # Always re-raise after logging
```

---

## **9. Code Review Checklist for Core Team**

### **✅ Architecture Compliance**
- [ ] Does this follow the policy-driven architecture?
- [ ] Are control flow exceptions properly classified and handled?
- [ ] Is the `ContextManager` used for all context operations?
- [ ] Are error recovery strategies leveraged instead of ad-hoc handling?

### **✅ Performance**
- [ ] Are object pools used for common allocations?
- [ ] Is telemetry properly integrated?
- [ ] Are there any unnecessary deep copies of context?
- [ ] Is caching leveraged where appropriate?

### **✅ Testing**
- [ ] Are policy classes tested in isolation?
- [ ] Is exception propagation tested?
- [ ] Are context isolation/merging scenarios covered?
- [ ] Are regression tests included for bug fixes?

---

## **10. Common Pitfalls and Their Solutions**

### **❌ Pitfall: Context Mutation Instead of Updates**
```python
# ❌ Wrong
context.some_field = new_value

# ✅ Correct  
updated_context = context.model_copy(update={"some_field": new_value})
```

### **❌ Pitfall: Exception Swallowing**
```python
# ❌ Wrong
try:
    result = await operation()
except Exception:
    return StepResult(success=False)  # Lost the exception!

# ✅ Correct
try:
    result = await operation()
except Exception as e:
    # Use the error classification system
    error_context = ErrorContext.from_exception(e)
    classifier.classify_error(error_context)
    
    if error_context.category == ErrorCategory.CONTROL_FLOW:
        raise e  # Re-raise control flow exceptions
    
    # Handle other exceptions appropriately
```

### **❌ Pitfall: Policy Logic in ExecutorCore**
```python
# ❌ Wrong - in ExecutorCore.execute()
if isinstance(step, LoopStep):
    # Complex loop logic here...

# ✅ Correct - in ExecutorCore.execute()
if isinstance(step, LoopStep):
    policy = DefaultLoopStepExecutor()
    return await policy.execute(self, step, ...)
```

---

This guide ensures that all Flujo core team members build features that are architecturally consistent, performant, and maintainable. The patterns here have been battle-tested and should be followed religiously to maintain Flujo's quality and reliability.
