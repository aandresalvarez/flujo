This implementation plan details the steps to add **LiteLLM** as a first-class alternative to **Pydantic AI** while simultaneously paying down the remaining high-priority technical debt identified in the **Team Developer Guide v2.0**.

By following this plan, we bridge the "Agentic Gap" (the fact that LiteLLM is a completion tool, not an agent) by leveraging Flujo’s existing policy-driven architecture.

---

### **Phase 1: Structural Excellence (The Prerequisite Debt)**
*Goal: Ensure the codebase is clean and secure before adding new provider logic.*

#### **1.1 Hardened Sandboxing (Security)**
*   **File:** `flujo/infra/settings.py`
    *   Add `docker_mem_limit`, `docker_pids_limit`, and `docker_network_mode` (default to `"none"`) to `SandboxSettings`.
*   **File:** `flujo/infra/sandbox/docker_sandbox.py`
    *   Modify `exec_code` to pass these settings as arguments to `self._client.containers.run(...)`.

#### **1.2 Purge `unittest` from Production (Hygiene)**
*   **Files:** `flujo/application/core/context_manager.py`, `flujo/utils/context.py`.
    *   Remove all `import unittest.mock` and `is_mock_like` checks.
*   **File:** `tests/test_types/fixtures.py`
    *   Define a `TestPipelineContext(PipelineContext)` and update all tests to use this instead of `Mock()`.

#### **1.3 Solve Circular DSL Imports (Structural Integrity)**
*   **Files:** `flujo/domain/dsl/pipeline.py`, `flujo/domain/dsl/step.py`.
    *   Move cross-module imports to the top level using `if TYPE_CHECKING:`.
    *   Use string forward references (e.g., `steps: list["Step"]`).

---

### **Phase 2: The LiteLLM Adapter (Bridging the Gap)**
*Goal: Create a normalized interface that makes LiteLLM look like an Agent to Flujo.*

#### **2.1 Define the `LiteLLMAdapter`**
*   **New File:** `flujo/agents/adapters/litellm_adapter.py`
*   **Implementation Details:**
    *   **History Translation:** Convert Flujo `ConversationTurn` objects into standard OpenAI-style dicts (`{"role": "user", "content": "..."}`).
    *   **Structured Output Parsing:** LiteLLM returns strings. The adapter must use the `output_type` (passed from the `Step`) to perform `TypeAdapter(output_type).validate_json(response_text)`.
    *   **Tool Call Handling:** LiteLLM returns tool requests. The adapter must:
        1.  Catch the tool call request.
        2.  Look up the tool in Flujo's `SkillRegistry`.
        3.  Execute the tool and append the result to the history.
        4.  Recursively call `litellm.acompletion` until a terminal response is reached (mimicking Pydantic AI's internal agent loop).

#### **2.2 Implement `LiteLLMUsageAdapter`**
*   **Work:** Map LiteLLM's `usage` dictionary (which uses `prompt_tokens` and `completion_tokens`) to the `FlujoAgentUsage` protocol.
*   **Costing:** Use LiteLLM’s built-in `response_cost` field to populate `FlujoAgentResult.cost_usd` directly, ensuring accurate cross-provider billing.

---

### **Phase 3: Integration & Factory Logic**
*Goal: Allow users to switch providers via a simple string prefix.*

#### **3.1 Update the Agent Factory**
*   **File:** `flujo/agents/factory.py`
*   **Logic:**
    *   Introduce a provider detection strategy. 
    *   **Default:** Any model string containing a `:` but NOT starting with `litellm:` or `ollama:` continues to use `PydanticAIAdapter`.
    *   **LiteLLM Path:** Any model starting with `litellm:` (e.g., `litellm:gpt-4o`) or `ollama:` (e.g., `ollama:llama3`) routes to `LiteLLMAdapter`.
    ```python
    if model.startswith(("litellm:", "ollama:", "vllm:")):
        return LiteLLMAdapter(model=model, ...), final_processors
    ```

#### **3.2 Update Async Agent Wrapper**
*   **File:** `flujo/agents/wrapper.py`
    *   Ensure the wrapper handles `AgentIOValidationError` consistently, regardless of whether the underlying error came from Pydantic AI's internal validator or LiteLLM's manual JSON parse step.

---

### **Phase 4: Granular Step Compatibility**
*Goal: Ensure LiteLLM works with crash-safe resumption.*

#### **4.1 Normalized History Storage**
*   **Requirement:** Since `GranularStep` stores history in the DB, we must ensure the format is consistent.
*   **Solution:** Both adapters (`PydanticAI` and `LiteLLM`) must accept a list of `ConversationTurn` objects.
    *   `PydanticAIAdapter` converts these to `ModelRequest/Response`.
    *   `LiteLLMAdapter` converts these to `dict`.
*   **Impact:** This ensures a `GranularStep` can be defined once and swapped between providers without changing the database schema.

---

### **Phase 5: Observability & Validation**
*Goal: Fulfill the "Excellent" DX requirement.*

#### **5.1 Local State Span Exporter**
*   **File:** `flujo/telemetry/otel_hook.py`
    *   Finish the implementation of `StateBackendSpanExporter`.
    *   Ensure it flushes to the local SQLite `spans` table.
    *   Verify that `flujo lens trace <run_id>` can render traces from a LiteLLM-backed run.

---

### **Summary of Results**

| Feature | Solution |
| :--- | :--- |
| **Agent Loops** | Managed by `LiteLLMAdapter` internal tool-execution loop. |
| **Granular Resumption** | Guaranteed by normalized history translation in the Adapter. |
| **Security** | Hardened via `ConfigManager`-driven Docker limits. |
| **Type Safety** | Enforced via `JSONObject` and `TypeAdapter` on completion results. |
| **Execution Path** | Unified. `ExecutorCore` remains untouched. |

**Complexity Level:** 🟡 **Medium**  
**Estimated Timeline:** 3–4 days of focused development. 

This plan adheres to the **"Golden Rule"** of Flujo: Everything goes through policies, and the `ExecutorCore` remains a pure dispatcher. Adding LiteLLM becomes an additive exercise in creating a new Adapter, rather than a risky refactor of the core engine.