# Flujo Strategic Roadmap: Enterprise Architecture Evolution

This document outlines a strategic roadmap to evolve Flujo from a robust execution framework into an enterprise-grade Agentic AI Platform. This analysis is based on a comparison with industry standards  and Flujo's current architecture.

The features are prioritized by a **High Impact / Low Effort** ratio to maximize immediate value.

---

## 1. Governance Policy Layer (The "Security Guard")
**Priority:** 🚨 Critical (Immediate)
**Impact:** High | **Effort:** Low/Medium

### Concept
Currently, Flujo manages *Resource Limits* (Quotas) but lacks *Behavioral Guardrails*. A Governance Policy Layer intercepts inputs and tool usage to enforce security and compliance rules before the LLM executes.

### Why it is Valuable
*   **Enterprise Requirement:** Regulated industries (Healthcare, Finance) cannot use agents without guarantees that PII won't leak or that specific tools (e.g., `delete_db`) won't be called in wrong contexts.
*   **Separation of Concerns:** Keeps prompt engineering separate from security logic.

### Implementation Strategy
*   **Location:** `flujo/application/core/step_coordinator.py`
*   **Mechanism:** Introduce a `PolicyHook` protocol that runs immediately before `_agent_runner.run`.
*   **Action:**
    1.  Define a `GovernancePolicy` protocol (Input -> Allow/Deny/Redact).
    2.  Create a default `PIIScrubbingPolicy` using simple regex/presidio.
    3.  Inject policies into `StepCoordinator.execute_step`.

```python
# Pseudo-code for flujo/domain/policy.py
class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REDACT = "redact"

class GovernancePolicy(Protocol):
    def check_input(self, context: PipelineContext, input_data: Any) -> PolicyDecision: ...
    def check_tool(self, tool_name: str) -> PolicyDecision: ...
```

---

## 2. OpenAPI Skill Generator (The "Integration Gateway")
**Priority:** 🚀 High (Short-term)
**Impact:** High | **Effort:** Low

### Concept
AgentCore creates tools automatically from API specs. Flujo currently requires manual Python coding for every skill in `flujo/infra/skill_registry.py`. This feature would generate Flujo-compatible skill wrappers automatically from a Swagger/OpenAPI JSON file.

### Why it is Valuable
*   **Developer Velocity:** Reduces the time to integrate internal microservices from hours to seconds.
*   **Reliability:** Auto-generated Pydantic models ensure type safety matches the API spec perfectly.

### Implementation Strategy
*   **Location:** `flujo/cli/dev_commands_experimental.py`
*   **Mechanism:** A CLI command `flujo dev import-openapi <url>`.
*   **Action:**
    1.  Use `datamodel-code-generator` to parse OpenAPI specs into Pydantic models.
    2.  Generate a Python file in `skills/generated/` that wraps `httpx` calls to the API.
    3.  Auto-register these functions in `SkillRegistry`.

---

## 3. Episodic Memory & RAG (The "Second Brain")
**Priority:** 🧠 Strategic (Medium-term)
**Impact:** Very High | **Effort:** Medium/High

### Concept
Flujo currently relies on `HistoryManager` (`flujo/application/conversation/history_manager.py`), which truncates session history. Episodic memory allows the agent to recall relevant facts from *previous, unrelated sessions* or long-past turns within the current session.

### Why it is Valuable
*   **Context Window Efficiency:** Reduces token costs by injecting only relevant memories rather than the whole history.
*   **Learning:** Agents stop making the same mistakes. If an agent solves a hard coding error, it "remembers" the fix for next time.

### Implementation Strategy
*   **Location:** `flujo/state/backends/postgres.py` (leverage `pgvector`).
*   **Mechanism:**
    1.  Update `PipelineContext` to include a `retrieve_memory(query)` method.
    2.  Create a `MemoryStep` (DSL) that runs a semantic search against the vector store and injects results into `context.scratchpad`.
    3.  Update `StateManager` to embed and persist successful `StepResult` outputs into a `memories` table.

---

## 4. Sandboxed Code Execution (The "Safety Net")
**Priority:** 🛡️ Strategic (Medium-term)
**Impact:** High | **Effort:** High

### Concept
Flujo executes tools in the main process. If an LLM generates malicious Python code (e.g., `os.system('rm -rf /')`), it runs on the host. AgentCore uses secure sandboxes.

### Why it is Valuable
*   **Safety:** Allows safe execution of arbitrary code generation (data analysis, plotting).
*   **Dependency Isolation:** Agents can install libraries (e.g., `pandas`, `numpy`) dynamically without bloating the main application environment.

### Implementation Strategy
*   **Location:** `flujo/builtins_extras.py`
*   **Mechanism:**
    1.  Create a new builtin skill: `flujo.builtins.docker_code_interpreter`.
    2.  Use the Docker SDK to spin up an ephemeral container (e.g., `python:3.11-slim`).
    3.  Mount a shared volume for file I/O.
    4.  Execute the code, capture stdout/stderr, destroy container.
    5.  *Alternative (Lower Effort):* Use `e2b` or specialized sandbox APIs if cloud execution is acceptable.

---

## 5. Shadow Evaluations (The "Quality Control")
**Priority:** 📊 Operational (Long-term)
**Impact:** Medium | **Effort:** Medium

### Concept
Evaluating agents usually happens in a separate "Bench" phase. Shadow Evaluations use Flujo's `BackgroundTaskManager` to score production runs asynchronously without slowing down the user experience.

### Why it is Valuable
*   **Production Visibility:** Detect drift in model quality (e.g., "Is GPT-4o getting lazy?") in real-time.
*   **Dataset Generation:** Automatically flags "bad" interactions for fine-tuning or debugging.

### Implementation Strategy
*   **Location:** `flujo/application/core/execution_manager.py`
*   **Mechanism:**
    1.  Define an `Evaluator` (LLM-as-Judge) in `flujo.toml`.
    2.  In `_persist_and_finalize`, trigger a background task using `self._background_task_manager`.
    3.  The task passes the completed `StepHistory` to the Evaluator agent.
    4.  Score is written to a new `evaluations` table in `flujo_ops.db`.

---

## Summary Matrix

| Feature | Impact | Effort | Value Proposition |
| :--- | :--- | :--- | :--- |
| **1. Governance Policies** | ⭐⭐⭐⭐⭐ | 🟢 Low | Enterprise security & compliance blocking (PII/Auth). |
| **2. OpenAPI Importer** | ⭐⭐⭐⭐ | 🟢 Low | accelerated development speed & integration. |
| **3. Episodic Memory** | ⭐⭐⭐⭐⭐ | 🟡 Med | Long-term intelligence & cost reduction via RAG. |
| **4. Shadow Evals** | ⭐⭐⭐ | 🟡 Med | Observability & quality assurance in production. |
| **5. Sandboxing** | ⭐⭐⭐⭐ | 🔴 High | Safety for "Code Interpreter" use cases. |