# **Flujo Developer's Guide: Best Practices and Anti-Patterns**

This guide provides a set of core principles and patterns for developing within the Flujo framework. Adhering to these will lead to more robust, maintainable, and architecturally-consistent code.

## **1. The Golden Rule: Respect the Separation of Concerns**

Flujo's primary strength is its separation of the **Declarative Shell (DSL)** and the **Execution Core (Policies)**.

*   **✅ Do:** Define *what* your pipeline does using the DSL (`Step`, `Pipeline`, `>>`).
*   **❌ Don't:** Put complex, imperative business logic inside your DSL definitions. A pipeline definition should read like a clear, high-level plan.
*   **✅ Do:** Encapsulate complex logic within a dedicated `Step`, either via `@step` on a function or by providing a custom agent.
*   **❌ Don't:** Create a single, giant `Step.from_callable` that contains an entire application's logic. Break it down into smaller, reusable steps.

---

## **2. Context Management: The Source of Truth**

The `PipelineContext` is the shared state of your workflow. Treat it with care.

*   **✅ Do:** Define a custom, strongly-typed `PipelineContext` class that inherits from `flujo.domain.models.PipelineContext` for your application. This enables static analysis and autocompletion.
*   **❌ Don't:** Pass state between steps by returning large, complex tuples or dictionaries. The context is the proper channel for shared state.
*   **✅ Do:** Use `updates_context=True` on steps that are *designed* to modify the context. The step's output should be a `BaseModel` or `dict` containing the fields to be updated.
*   **❌ Don't:** Mutate the context object directly from within a step's agent and expect those changes to persist globally, especially in parallel or loop steps. The `ContextManager` uses isolation; only the *returned* context from a branch or the *merged* context is propagated.

---

## **3. Agents and Steps: Keep Them Focused**

A `Step` is a configuration object; an `agent` is the thing that does the work.

*   **✅ Do:** Design agents to be stateless and reusable. An agent should perform one well-defined task (e.g., "summarize text," "validate SQL").
*   **❌ Don't:** Create stateful agents that hold data from previous runs. This breaks the immutability and re-runnability of pipelines. Use the `PipelineContext` for state.
*   **✅ Do:** Use `Step.from_callable` or the `@step` decorator for simple, stateless data transformations (e.g., extracting a field from a dictionary).
*   **❌ Don't:** Use `Step.from_callable` for logic that involves external I/O, complex state changes, or requires retries. This kind of logic belongs in a dedicated agent class where its dependencies can be managed cleanly.

---

## **4. Error and Control Flow Handling: Use the Framework**

Flujo has sophisticated systems for managing failures and non-linear execution. Use them instead of inventing your own.

*   **✅ Do:** Let exceptions happen. If an agent fails, let it raise an exception. The `ExecutorCore`'s policies are designed to catch this, apply retries, and trigger fallbacks.
*   **❌ Don't:** Write `try...except Exception: return None` inside your agents. This swallows errors, hides bugs, and prevents the framework's resilience mechanisms from working.
*   **✅ Do:** Use the `step.fallback(fallback_step)` method in the DSL to define a declarative recovery path for a failing step.
*   **❌ Don't (The Critical Anti-Pattern):** Catch a `PausedException` and return a `StepResult`. This breaks the pause/resume functionality. **Control flow exceptions must always be allowed to propagate.**
*   **✅ Do:** Use `Plugins` with `redirect_to` or `new_solution` for dynamic, data-driven changes to a step's execution path.

### **Control Flow Exception Handling - The Architectural Way**

When implementing custom control flow exceptions (like `PausedException`), follow the architectural pattern:

1. **Register in Error Classifier**: Add your exception to `ErrorClassifier` with `ErrorCategory.CONTROL_FLOW`
2. **Create Recovery Strategy**: Register a strategy with `max_retries=0` and `primary_action=RecoveryAction.ESCALATE`
3. **Use in Policies**: Leverage `ErrorClassifier.classify_error()` in step policies to detect and re-raise control flow exceptions
4. **Fire Events**: Use the hook system (`OnPauseRequestedPayload`) for observability
5. **State Persistence**: Leverage `StateManager` for durable pause/resume workflows

**Never implement ad-hoc exception catching** - use Flujo's sophisticated error handling architecture.

---

## **5. Debugging Flujo Pipelines: A Methodical Approach**

Debugging a distributed, asynchronous system can be hard. Follow this process.

*   **Step 1: Is it a DSL problem? Use `pipeline.validate_graph()`**
    *   Before you even run the code, run the validator. It will catch common issues like type mismatches between steps or steps without agents.

*   **Step 2: Is it a runtime problem? Use the `ConsoleTracer`**
    *   The `ConsoleTracer` is your best friend. Add it to your `Flujo` runner (`local_tracer="default"`). It will give you a beautiful, color-coded printout of:
        *   Which step is running.
        *   What data is flowing into it (`log_inputs=True`).
        *   What data is coming out of it (`log_outputs=True`).
        *   Exactly where it failed.

*   **Step 3: Is it a state problem? Use the `flujo lens` CLI**
    *   If your pipeline is behaving strangely across multiple runs or during resumption, the problem is likely in the state.
    *   Use `flujo lens list` to find your `run_id`.
    *   Use `flujo lens show <run_id> --verbose` to inspect the full input, output, and error for every single step.
    *   Use `flujo lens trace <run_id>` to see the exact execution hierarchy, which is invaluable for debugging complex nested steps.

*   **Step 4: Is it an agent logic problem? Test the agent in isolation.**
    *   If the tracer shows that a step is receiving the correct input but producing the wrong output, the problem is inside your agent.
    *   Write a simple unit test that calls your agent's `.run()` method directly, completely outside of a Flujo pipeline, and verify its logic.

By following this hierarchy, you can quickly narrow down the source of any problem without getting lost.

---

## **6. Performance Best Practices**

*   **✅ Do:** Use streaming (`stream=True`) for steps that process large data or take a long time to complete.
*   **✅ Do:** Leverage caching for expensive, deterministic operations.
*   **✅ Do:** Use `ParallelStep` for independent operations that can run concurrently.
*   **❌ Don't:** Create deeply nested pipeline structures unnecessarily - they add overhead.
*   **✅ Do:** Profile your pipelines using the built-in telemetry and `flujo lens` to identify bottlenecks.

---

## **7. Testing Strategies**

*   **Unit Tests**: Test agents in isolation using direct `.run()` calls
*   **Integration Tests**: Test complete pipelines with real data
*   **Regression Tests**: Use the evaluation framework (`evaluate_and_improve`) for systematic testing
*   **State Tests**: Test pause/resume functionality with different state backends

---

## **Quick Reference: Common Patterns**

### **Simple Data Transformation**
```python
@step
def extract_field(data: dict) -> str:
    return data.get("field", "default")
```

### **Custom Agent Step**
```python
class MyAgent:
    async def run(self, input_data: str, context: MyContext) -> str:
        # Stateless logic here
        return processed_data

my_step = Step(agent=MyAgent(), name="my_step")
```

### **Error Handling with Fallback**
```python
main_step = Step(agent=MainAgent(), name="main")
fallback_step = Step(agent=FallbackAgent(), name="fallback")
robust_step = main_step.fallback(fallback_step)
```

### **Context Updates**
```python
@step(updates_context=True)
def update_status(data: str) -> dict:
    return {"status": "processed", "result": data}
```

### **Conditional Logic**
```python
condition_step = ConditionalStep(
    condition=lambda ctx: ctx.should_process,
    if_step=process_step,
    else_step=skip_step
)
```

This guide should be your go-to reference for building robust, maintainable Flujo applications.
