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

## Security Note

`RunPythonCodeCommand` executes Python code with built-ins disabled and will
reject any `import` statements. It still runs in-process, so you must ensure a
safe sandbox for untrusted input.
