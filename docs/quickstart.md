# Quickstart Guide

Get up and running with `flujo` in 5 minutes!

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

## 3. Your First AgenticLoop

Create a new file `hello_agentic.py`:

```python
from flujo.recipes import AgenticLoop
from flujo import make_agent_async, init_telemetry
from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand
from pydantic import TypeAdapter

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
    TypeAdapter(AgentCommand),
)

loop = AgenticLoop(planner_agent=planner_agent, agent_registry={"search_agent": search_agent})
result = loop.run("What is Python?")
```

When the loop finishes, inspect `result.final_pipeline_context.command_log` to see each decision and tool invocation.

## 4. Run Your First Loop

```bash
python hello_agentic.py
```

You should see a short transcript of the planner running the search tool and finishing with an answer.

## 5. Next Steps

Now that you've seen the basics, explore the [Tutorial](tutorial.md) and [Concepts](concepts.md) pages for a deeper dive.
