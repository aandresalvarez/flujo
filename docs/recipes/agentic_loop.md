# Recipe: AgenticLoop

`AgenticLoop` provides a convenient way to build explorative agent workflows. A planner agent decides which command to run next and the recipe executes it, recording every turn for traceability.

```python
from flujo import AgenticLoop, StubAgent
from flujo.domain.commands import RunAgentCommand, FinishCommand

planner = StubAgent([
    RunAgentCommand(agent_name="helper", input_data="hi"),
    FinishCommand(final_answer="done"),
])
loop = AgenticLoop(planner_agent=planner, agent_registry={"helper": StubAgent(["ok"])})
result = loop.run("initial goal")
print(result.final_pipeline_context.command_log)
```

## Security Note

`RunPythonCodeCommand` executes arbitrary code. Only use it with a secure sandbox.
