This is a detailed proposal to implement **Dual Execution Modes** for Flujo agents.

This design gives developers explicit control over the **Reliability vs. Latency** trade-off.

---

# Proposal: Dual Execution Modes (Atomic vs. Granular)

## 1. The Concept

We will introduce a configuration option that allows developers to choose how an Agent executes within a pipeline step.

### Mode A: Atomic Execution (Default)
*   **Behavior:** The Agent runs to completion (thinking, tool calling, final answer) within a single Flujo Step.
*   **Persistence:** State is saved only **after** the agent finishes everything.
*   **Crash Recovery:** If the process dies mid-execution, the **entire step restarts** from the beginning.
*   **Best For:** Fast queries, read-only tool usage, low-latency requirements.

### Mode B: Granular Execution (Resumable)
*   **Behavior:** The Agent's internal loop is "unrolled" into a Flujo `LoopStep`. Each iteration (Thought $\to$ Tool Call $\to$ Result) is treated as a distinct state transition.
*   **Persistence:** State is saved **after every tool execution**.
*   **Crash Recovery:** If the process dies, Flujo resumes **exactly after the last successful tool call**.
*   **Best For:** Long-running research, expensive API calls, side-effects (sending emails, booking tickets) where repetition is dangerous.

---

## 2. Developer Experience (API Design)

We will expose this via a clean factory interface on the `Step` class.

### 2.1 Atomic Usage (Current Standard)
```python
# Standard behavior: fast, simple, retries the whole block on failure
step = Step(
    name="research_atomic",
    agent=my_agent,
    input="Research quantum computing"
)
```

### 2.2 Granular Usage (New)
```python
# New behavior: persistent, safe, resumes mid-stream
step = Step.granular(
    name="research_granular",
    agent=my_agent,
    input="Research quantum computing",
    max_turns=20 # Safety limit for the internal loop
)
```

---

## 3. Technical Implementation

To achieve this without rewriting the core engine, we will use a **Compiler Pattern**. `Step.granular` will not return a simple `Step`, but rather a pre-configured `Pipeline` containing a `LoopStep` and a specialized `TurnTaker`.

### 3.1 The `PydanticTurnTaker` (The Engine)

We need a specialized wrapper that knows how to run *just one turn* of a Pydantic AI agent and manage the message history manually.

**Location:** `flujo/agents/granular.py`

```python
from pydantic_ai.messages import ModelMessage
from flujo.domain.models import PipelineContext

class PydanticTurnTaker:
    def __init__(self, agent):
        self.agent = agent

    async def run_turn(self, input_data: str, *, context: PipelineContext) -> dict:
        # 1. Load History from DB (via Context)
        # We store the raw Pydantic messages in the scratchpad
        history_data = context.scratchpad.get("granular_history", [])
        
        # Deserialize (pseudo-code, requires Pydantic adapter)
        messages = [ModelMessage.model_validate(m) for m in history_data]

        # 2. Execute ONE Turn
        # We use the agent to generate the next response based on history
        # Note: This relies on Pydantic AI allowing injection of history
        if not messages:
            # First run
            result = await self.agent.run(input_data)
        else:
            # Resume run
            result = await self.agent.run(input_data, message_history=messages)

        # 3. Update History
        # Pydantic AI returns the full updated history in the result
        new_history = [m.model_dump() for m in result.all_messages()]
        
        # 4. Check for Completion
        # If the last message is a final text response (not a tool call), we are done
        is_complete = not result.new_messages()[-1].parts[0].tool_calls

        return {
            "history": new_history,
            "final_output": result.data if is_complete else None,
            "is_complete": is_complete
        }
```

### 3.2 The `Step.granular` Factory

This factory compiles the logic into a Flujo Loop.

**Location:** `flujo/domain/dsl/step.py`

```python
    @classmethod
    def granular(
        cls, 
        name: str, 
        agent: Any, 
        input: str, 
        max_turns: int = 10
    ) -> "Pipeline":
        from .loop import LoopStep
        from .pipeline import Pipeline
        from ...agents.granular import PydanticTurnTaker

        # 1. Create the Turn Taker
        turn_taker = PydanticTurnTaker(agent)

        # 2. Define the Body Step (Runs one turn)
        body_step = Step.from_callable(
            turn_taker.run_turn,
            name=f"{name}_turn",
            updates_context=True, # Merges 'history' back to scratchpad
            input=input # Pass the original goal every time (agent ignores it if history exists)
        )

        # 3. Define the Loop (Manages persistence)
        loop = LoopStep(
            name=name,
            loop_body_pipeline=Pipeline.from_step(body_step),
            # Exit when the turn taker says "is_complete"
            exit_condition_callable=lambda out, ctx: out.get("is_complete", False),
            max_loops=max_turns,
            # Extract the final answer when done
            loop_output_mapper=lambda out, ctx: out.get("final_output")
        )

        # Return a Pipeline wrapping this logic
        return Pipeline.from_step(loop)
```

---

## 4. Serialization Strategy (The Hard Part)

For Granular mode to work, we must be able to serialize Pydantic AI's `ModelMessage` objects to JSON (for SQLite/Postgres) and reconstruct them perfectly.

*   **Challenge:** `ModelMessage` objects can contain complex types.
*   **Solution:** Use Pydantic V2's `model_dump(mode='json')` for serialization and `model_validate()` for deserialization.
*   **Action:** Ensure `flujo/utils/serialization.py` has a custom serializer registered for `ModelMessage` if it doesn't handle it automatically via the generic Pydantic support.

---

## 5. Implementation Roadmap

1.  **Serializer Verification:** Write a test case that serializes a Pydantic AI `ModelMessage` (containing a Tool Call) to JSON, saves it to SQLite, loads it back, and verifies `agent.run(history=loaded_messages)` works.
2.  **Implement `PydanticTurnTaker`:** Create the logic to run a single turn. *Note: We may need to use Pydantic AI's lower-level `Model` API if the high-level `Agent.run` doesn't support "run one step and stop" natively.*
3.  **Implement `Step.granular`:** Add the factory method to the DSL.
4.  **Integration Test:**
    *   Create a granular step.
    *   Run it.
    *   Kill the process mid-loop (simulate crash).
    *   Resume.
    *   Verify it skips the already-executed tool calls.

## 6. Decision Matrix for Developers

We will add this to the documentation to help users choose.

| Requirement | Use `Step(agent=...)` (Atomic) | Use `Step.granular(...)` |
| :--- | :---: | :---: |
| **Speed** | ✅ Faster (No DB writes between tools) | ❌ Slower (DB write every turn) |
| **Safety** | ❌ Low (Restarts step on crash) | ✅ High (Resumes exactly where left off) |
| **Cost** | ❌ Higher on failure (Re-spends tokens) | ✅ Optimized (Cached history) |
| **Complexity** | ✅ Simple | 🟡 Moderate (Creates a LoopStep) |
| **Side Effects** | ⚠️ Dangerous (May repeat API calls) | ✅ Safe (Idempotent via history) |