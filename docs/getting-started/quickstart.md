# Quickstart Guide

Get up and running with `flujo` in minutes. Choose a path:

- Project-based (no-code CLI) — recommended for most users
- Programmatic (Python DSL) — for full control

## 1. Install the Package

```bash
pip install flujo
```

## 2. Set Up Your API Keys

Create a `.env` file in your project directory:

```bash
cp .env.example .env
```

Add your API keys to `.env`:
```env
OPENAI_API_KEY=your_key_here
```

## 3. Project-Based Quickstart (Recommended)

Initialize a new project and create a pipeline via conversation.

```bash
mkdir my-bot && cd my-bot
flujo init
```

Start the architect to generate `pipeline.yaml` and budget in `flujo.toml`:

```bash
flujo create
# Answer prompts: goal, pipeline name, budget per-run
```

Run the pipeline (project-aware; no file paths needed):

```bash
flujo run --input "Hello, world"
```

Inspect and replay runs using the built-in lens tooling (uses `.flujo/state.db` by default):

```bash
flujo lens list
flujo lens trace <run_id>
flujo lens replay <run_id>
```

## 4. Programmatic Quickstart (Python DSL)

Create a new file `hello_agentic.py`:

```python
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.infra.agents import make_agent_async
from flujo.infra import init_telemetry
from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand

init_telemetry()

async def search_agent(query: str) -> str:
    print(f"   -> Tool Agent searching for '{query}'...")
    return "Python is a high-level, general-purpose programming language." if "python" in query.lower() else "No information found."

PLANNER_PROMPT = """
You are a research assistant. Use the `search_agent` to gather information.
When you have an answer, respond with `FinishCommand`.
"""
planner_agent = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    AgentCommand,
)

# Create the pipeline using the factory
pipeline = make_agentic_loop_pipeline(
    planner_agent=planner_agent,
    agent_registry={"search_agent": search_agent}
)

# Run the pipeline
result = await run_agentic_loop_pipeline(pipeline, "What is Python?")
print(result)

# Access trace information (FSD-12 feature)
if result.trace_tree:
    print(f"\n🌳 Execution Trace:")
    print(f"   Root span: {result.trace_tree.name}")
    print(f"   Status: {result.trace_tree.status}")
    print(f"   Duration: {result.trace_tree.end_time - result.trace_tree.start_time:.3f}s")
    print(f"   Steps: {len(result.step_history)}")
```

## 5. Run Your First Loop

```bash
python hello_agentic.py
```

You should see a short transcript of the planner running the search tool and finishing with an answer, plus trace information.

## 6. Debug with Tracing

After running your pipeline, you can inspect the execution trace:

```bash
# List recent runs
flujo lens list

# View trace for the most recent run (replace with actual run_id)
flujo lens trace <run_id>

# Show step details
flujo lens spans <run_id>
```

## 7. Next Steps

Now that you've seen the basics, explore:

- [Type-Safe Patterns](type_safe_patterns.md) — typed contexts and adapter best practices
- [Tutorial](tutorial.md) — deeper dive into pipeline composition
- [Concepts](../user_guide/concepts.md) — understand the Flujo architecture
