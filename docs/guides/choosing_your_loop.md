# Guide: Choosing Your Looping Strategy

Flujo offers two looping patterns for different kinds of tasks:

- **`Step.loop_until`** for deterministic iteration of a fixed pipeline body.
- **`make_agentic_loop_pipeline`** for explorative workflows where a planner decides what to do next.

This guide explains when to use each and shows how to migrate from one to the other.

## When to Use `Step.loop_until` (The State Machine)

`Step.loop_until` is ideal when you repeatedly run the same sub-pipeline until a
condition is met. Think of it like a `while` loop. The loop body is constant and
you decide when to exit based on the last output (and optionally the pipeline
context).

Typical use cases include iterative refinement of a single artifact or polling
for a result.

```python
loop_step = Step.loop_until(
    name="IterativeRefinement",
    loop_body_pipeline=refine_pipeline,
    exit_condition_callable=lambda last, ctx: last.is_done,
)
```

You have full control over how iteration inputs are mapped and can inspect or
update the shared context on each turn.

## When to Use `make_agentic_loop_pipeline` (The Explorer)

`make_agentic_loop_pipeline` shines when your workflow requires dynamic decision making or tool
use. A planner agent determines which command to run next, such as calling a
helper agent, running Python code, asking a human, or finishing the loop.

This pattern is great for research or data gathering tasks where the exact steps
aren't known ahead of time.

```python
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline

pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry={"tool": tool},
)
result = run_agentic_loop_pipeline(pipeline, "Find information about Python")
```

The loop continues until the planner issues a `FinishCommand`. Every command is
logged to `PipelineContext.command_log` for traceability.

## Migration Guide: Refactoring a `LoopStep` to an `AgenticLoop`

The snippet below shows a simplified before/after comparison.

### Before: Manual `LoopStep`

```python
# --- Build the loop body ---
body = Step("Decide", planner_agent) >> Step("Execute", execute_command)

# --- Manual exit condition ---
loop_step = Step.loop_until(
    name="ExplorationLoop",
    loop_body_pipeline=body,
    exit_condition_callable=lambda last, ctx: isinstance(last, FinishCommand),
    iteration_input_mapper=lambda result, ctx, i: {
        "last_command_result": result,
        "goal": ctx.initial_prompt if ctx else "",
    },
)
```

### After: `make_agentic_loop_pipeline`

```python
from flujo.recipes.factories import make_agentic_loop_pipeline

pipeline = make_agentic_loop_pipeline(
    planner_agent=planner_agent,
    agent_registry={"execute": execute_command},
)

runner = Flujo(pipeline)
result = runner.run("Explore this topic")
```

`make_agentic_loop_pipeline` removes boilerplate and automatically handles the planner/executor
interaction. The context keeps a command log so you can inspect the agent's
decisions.
