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

If the planner issues an `AskHumanCommand`, the loop pauses. Use `resume` or
`resume_async` to continue after receiving human input:

```python
paused = loop.run("goal")
if paused.status == "paused":
    resumed = loop.resume(paused, "human answer")
    print(resumed.final_pipeline_context.command_log)
```

## AgentCommand Models

Your planner agent must emit one of the following commands on each turn:

- `RunAgentCommand(agent_name, input_data)` – delegate work to a registered sub-agent.
- `RunPythonCodeCommand(code)` – run a Python snippet. The result should be stored in a variable named `result`.
- `AskHumanCommand(question)` – pause the loop and wait for human input.
- `FinishCommand(final_answer)` – end the loop with a final answer.

For example, you can run Python code safely:

```python
planner = StubAgent([
    RunPythonCodeCommand(code="result = 1 + 1"),
    FinishCommand(final_answer="done"),
])
loop = AgenticLoop(planner, {})
result = loop.run("goal")
print(result.final_pipeline_context.command_log[0].execution_result)
```

## Security Note

`RunPythonCodeCommand` executes Python code with built-ins disabled and will
reject any `import` statements. It still runs in-process, so you must ensure a
safe sandbox for untrusted input.
