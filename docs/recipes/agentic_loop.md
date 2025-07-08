# Recipe: AgenticLoop (Factory Pattern)

> **Note:** The class-based `AgenticLoop` is deprecated. Use the new `make_agentic_loop_pipeline` factory function for full transparency, composability, and future YAML/AI support.

`make_agentic_loop_pipeline` provides a convenient and transparent way to build explorative agent workflows. A planner agent decides which command to run next and the pipeline executes it, recording every turn for traceability.

```python
from flujo.recipes.factories import make_agentic_loop_pipeline
from flujo.testing.utils import StubAgent
from flujo.domain.commands import RunAgentCommand, FinishCommand

planner = StubAgent([
    RunAgentCommand(agent_name="helper", input_data="hi"),
    FinishCommand(final_answer="done"),
])
agent_registry = {"helper": StubAgent(["ok"])}
pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry=agent_registry,
    max_loops=10,
)

# To run the pipeline:
from flujo.recipes.factories import run_agentic_loop_pipeline
import asyncio
result = asyncio.run(run_agentic_loop_pipeline(pipeline, "initial goal"))
print(result)
```

## Human-in-the-Loop (Pausing)

If the planner issues an `AskHumanCommand`, the loop will pause. You can handle this by checking the pipeline result and resuming as needed (see advanced examples in the main documentation).

## AgentCommand Models

Your planner agent must emit one of the following commands on each turn:

- `RunAgentCommand(agent_name, input_data)` – delegate work to a registered sub-agent.
- `AskHumanCommand(question)` – pause the loop and wait for human input.
- `FinishCommand(final_answer)` – end the loop with a final answer.

The previously supported `RunPythonCodeCommand` has been removed due to security concerns.

---

**Why use the factory?**
- Returns a standard Pipeline object (inspectable, composable, serializable)
- Enables YAML/AI-driven modification in the future
- Makes the workflow structure explicit and debuggable
- Fully aligned with the Flujo roadmap
