# Tutorial: From Simple Orchestration to Custom AI Pipelines

Welcome! This tutorial will guide you through using the `flujo` library, from your very first request to building advanced, custom AI workflows. We'll start with the basics and progressively build up to more complex examples.

**Before You Begin:**
*   You should have a basic understanding of Python.
*   Make sure you have set up your API keys (e.g., `OPENAI_API_KEY`) in a `.env` file in your project directory. The orchestrator will automatically find and use them.

---

## Key Concepts: The Building Blocks

Before we write any code, let's understand the main components you'll be working with. Think of it like a chef learning about their ingredients before cooking.

*   **AgenticLoop (Recipe):** A dynamic planner-loop pattern. A planner agent decides which `AgentCommand` to execute next (run a tool agent, ask a human, run Python, or finish). This is the recommended starting point for explorative workflows.

*   **Flujo, Pipeline, Step:** When you need more control than the standard `Default` workflow, you'll use the **Pipeline DSL**. A `Pipeline` is a sequence of `Step` objects executed by `Flujo` to build fully custom multi-agent workflows.

*   **A Candidate:** This is the final result produced by the Default recipe. It contains the solution itself and the checklist used to grade it.

Now that we know the players, let's see them in action!

---

## 1. Your First AI Task: AgenticLoop

We'll begin with the `make_agentic_loop_pipeline` pattern. A planner agent decides which tool agent to run and when to finish.

```python
# 📂 step_1_agentic_loop.py
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.infra.agents import make_agent_async
from flujo.infra import init_telemetry
from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand

init_telemetry()

async def search_agent(query: str) -> str:
    return "Rainy days are peaceful" if "rain" in query else "No info"

PLANNER_PROMPT = """
You are a poet assistant. Use `search_agent` to gather inspiration.
When ready, reply with `FinishCommand` containing the final haiku.
"""
planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    AgentCommand,
)

pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry={"search_agent": search_agent},
)
result = run_agentic_loop_pipeline(pipeline, "Write a short, optimistic haiku about a rainy day.")

for entry in result.final_pipeline_context.command_log:
    print(entry)
```
This loop lets the planner decide when to call the tool and when to finish. The command log shows each decision.

## 2. Building Your First Custom Step with `@step`

The easiest way to extend a pipeline is to decorate an async function with `@step`.

```python
from flujo import step

@step
async def shout(text: str) -> str:
    return text.upper()
```

`shout` is now a typed `Step[str, str]` that you can compose with others.

## 3. Composing Your First Custom Pipeline

```python
from flujo import Flujo, step, PipelineResult, PipelineRegistry

@step
async def to_upper(text: str) -> str:
    return text.upper()

@step
async def add_excitement(text: str) -> str:
    return f"{text}!"

pipeline = to_upper >> add_excitement
registry = PipelineRegistry()
registry.register(pipeline, "demo", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="demo",
    pipeline_version="1.0.0",
)
result: PipelineResult[str] = runner.run(
    "hello", initial_context_data={"run_id": "example"}
)
print(result.step_history[-1].output)  # HELLO!
```

## 4. The Budget-Aware Workflow: Customizing Agents for Default Pipeline

The default pipeline factory is handy for simple, fixed workflows. You can customize its agents to mix models for cost and quality.

Professional AI workflows often involve a mix of models to balance cost, speed, and quality. Here, we'll use a **cheaper, faster model** for the initial draft (solution agent) but retain the **smarter models** for the critical thinking roles (planning, quality control, and strategy).

```python
# 📂 step_3_mixing_models.py
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.infra.agents import make_agent_async
from flujo.infra import init_telemetry
from flujo.models import Task
from flujo.infra.agents import make_review_agent, make_validator_agent
init_telemetry()
print("🚀 Building a workflow with a custom Solution Agent for the default pipeline...")
FAST_SOLUTION_PROMPT = "You are a creative but junior marketing copywriter. Write a catchy and concise slogan. Be quick and creative."
fast_copywriter_agent = make_agent_async("openai:gpt-4o-mini", FAST_SOLUTION_PROMPT, str)
pipeline = make_default_pipeline(
    review_agent=make_review_agent(),
    solution_agent=fast_copywriter_agent,
    validator_agent=make_validator_agent(),
)
task = Task(prompt="Write a slogan for a new brand of ultra-durable luxury coffee mugs.")
result = run_default_pipeline(pipeline, task)
# ... (printing logic)
# ... (printing logic)
```
This "cheap drafter, smart reviewer" pattern is a powerful way to get high-quality results efficiently. The fast agent produces drafts, and the smart agents ensure the final output is excellent.

---

## 5. Outputting Structured Data with a Custom Pipeline

So far, our agents have only outputted simple strings. What if we need structured data, like JSON? The underlying `pydantic-ai` library excels at this. You can specify a Pydantic `BaseModel` as the `output_type` for an agent.

Let's build a workflow that extracts information from a block of text into a structured `ContactCard` model.

```python
# 📂 step_4_structured_output.py
from pydantic import BaseModel, Field
from flujo import Step, Flujo
from flujo.infra.agents import make_agent_async
from flujo.infra import init_telemetry
from flujo.models import Checklist

init_telemetry()

# 1. Define our desired output structure using a Pydantic model
class ContactCard(BaseModel):
    name: str = Field(..., description="The full name of the person.")
    email: str | None = Field(None, description="The person's email address.")
    company: str | None = Field(None, description="The company they work for.")

# 2. Define Agents for our custom pipeline
print("🛠️ Creating a data-extraction agent...")
EXTRACTION_PROMPT = "You are a data-entry expert. Extract contact information from the user's text and format it precisely according to the ContactCard schema. If a field is not present, omit it."
extraction_agent = make_agent_async("openai:gpt-4o", EXTRACTION_PROMPT, ContactCard)

REVIEW_PROMPT_FOR_EXTRACTION = "Generate a checklist to verify the extracted contact details. Check for name correctness, email validity, and company presence."
review_agent_for_extraction = make_agent_async("openai:gpt-4o", REVIEW_PROMPT_FOR_EXTRACTION, Checklist)

VALIDATE_PROMPT_FOR_EXTRACTION = "You are a QA for data extraction. Use the checklist to verify the ContactCard."
validator_agent_for_extraction = make_agent_async("openai:gpt-4o", VALIDATE_PROMPT_FOR_EXTRACTION, Checklist)

# 3. Define the custom pipeline
data_extraction_pipeline = (
    Step.review(review_agent_for_extraction, name="PlanExtraction")
    >> Step.solution(extraction_agent, name="ExtractContactInfo")
    >> Step.validate_step(validator_agent_for_extraction, name="ValidateCard")
)

registry.register(data_extraction_pipeline, "extract", "1.0.0")
pipeline_runner = Flujo(
    registry=registry,
    pipeline_name="extract",
    pipeline_version="1.0.0",
)
unstructured_text = "Reach out to Jane Doe. She works at Innovate Corp and her email is jane.doe@example.com."

print(f"📄 Running custom pipeline to extract from: '{unstructured_text}'")
pipeline_result = pipeline_runner.run(unstructured_text)

if pipeline_result.step_history and pipeline_result.step_history[1].success:
    contact_card_solution = pipeline_result.step_history[1].output
    if isinstance(contact_card_solution, ContactCard):
        print("\n✅ Successfully extracted structured data (ContactCard object):")
        print(contact_card_solution.model_dump_json(indent=2))
    else:
        print(f"\n⚠️ Expected ContactCard, got: {type(contact_card_solution)}")
else:
    print("\n❌ Custom pipeline failed to extract contact info.")
```

#### **Expected Output:**

```
✅ Successfully extracted structured data:
{
  "name": "Jane Doe",
  "email": "jane.doe@example.com",
  "company": "Innovate Corp"
}
```

> **💡 Pro Tip: Beyond Basic Types**
> An agent's `output_type` can be `str`, `int`, `float`, or any Pydantic `BaseModel`. This is incredibly powerful for forcing the LLM to return clean, validated JSON that you can immediately use in your application.

---

## 6. The Grand Finale: A Fully Custom Pipeline with Tools

Now for the ultimate challenge. Let's build a workflow where **every agent is customized**, and our `solution_agent` can use **external tools** to get information it doesn't have.

**Scenario:** We need to write a factual report on a public company's stock price. The LLM doesn't know real-time stock prices, so it will need a tool.

1.  **Custom Planner:** A `review_agent` that knows what a good financial report looks like.
2.  **Tool-Using Doer:** A `solution_agent` that can call a `get_stock_price` function.
3.  **Custom Quality Analyst:** A `validator_agent` that is hyper-critical about financial data.

```python
# 📂 step_5_advanced_tools.py
import random
from pydantic import BaseModel
from pydantic_ai import Tool
from flujo import Flujo, Step, Pipeline
from flujo.testing import StubAgent

# --- 1. Define the Tool ---
# This is a fake stock price function for our example.
def get_stock_price(symbol: str) -> float:
    """Gets the current stock price for a given ticker symbol."""
    print(f"TOOL USED: Getting stock price for {symbol}...")
    # In a real app, this would make an API call. We'll fake it.
    if symbol.upper() == "AAPL":
        return round(random.uniform(150, 250), 2)
    return round(random.uniform(50, 500), 2)

# --- 2. Create the Fully Custom Agent Team ---
print("👑 Assembling a fully custom, tool-using agent team...")

# The Planner: Focused on financial report quality
review_agent = make_agent_async("openai:gpt-4o",
    "You are a financial analyst. Create a checklist for a brief, factual company report. Key items must include the company name, its stock symbol, the current price, and a concluding sentence.",
    Checklist)

# The Doer: Equipped with the stock price tool
class Report(BaseModel):
    company: str
    symbol: str
    current_price: float
    summary: str

# To use tools, we wrap them in a Tool object. The name of the tool
# must match the function name.
stock_tool = Tool(get_stock_price)

solution_agent = make_agent_async("openai:gpt-4o-mini", # Cheaper model for this
    "You are a junior analyst. Write a one-paragraph report on the requested company. Use the provided tools to get live data. Your final output must be a structured Report.",
    Report,
    # The magic happens here: we give the agent its tools.
    tools=[stock_tool])

# The Quality Analyst: Hyper-critical of data
validator_agent = make_agent_async("openai:gpt-4o",
    "You are a senior auditor. Meticulously check the report against the checklist. Be extremely strict about factual data. If the price is a placeholder, fail it.",
    Checklist)


# --- 3. Assemble and Run the Default Recipe ---
pipeline = make_default_pipeline(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
)
task = Task(prompt="Generate a stock report for Apple Inc. (AAPL).")

print("🧠 Running advanced tool-based workflow...")
best_candidate = run_default_pipeline(pipeline, task)

if best_candidate:
    print("\n🎉 Advanced workflow complete!")
    print(best_candidate.solution.model_dump_json(indent=2))
```

#### **What You'll See:**

During the execution, you will see a message from our tool function:
`TOOL USED: Getting stock price for AAPL...`

This confirms that the `solution_agent` recognized it needed information, called the function you provided, and used the result in its answer. The final output will be a perfectly structured report with the "live" data.

---

This concludes our tour! You've journeyed from a simple prompt to a sophisticated, tool-using AI system. You've learned to:
-   Understand the core concepts of **default pipeline factories and Agents**.
-   Run a basic multi-agent task and interpret its self-correction process.
-   Control the definition of quality using **weighted scoring**.
-   Optimize workflows by **mixing different AI models**.
-   Generate clean, **structured JSON** using Pydantic models.
-   Empower agents with **external tools** to overcome their knowledge limitations.

## 7. Building Custom Pipelines

The new Pipeline DSL lets you compose your own workflow using `Step` objects. Execute the pipeline with `Flujo`:

```python
from flujo import Step, Flujo, PipelineRegistry
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

registry = PipelineRegistry()
sql_step = Step.solution(StubAgent(["SELECT FROM"]))
check_step = Step.validate_step(StubAgent([None]), plugins=[SQLSyntaxValidator()])
pipeline_example = sql_step >> check_step
registry.register(pipeline_example, "sql", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="sql",
    pipeline_version="1.0.0",
)
result = runner.run("SELECT FROM")
print(result.step_history[-1].feedback)
```

### Using a Shared Typed Context

`Flujo` can share a Pydantic model instance across steps. This lets you
accumulate data or pass configuration during a run. See
[Typed Pipeline Context](../user_guide/pipeline_context.md) for more details.

```python
from flujo.domain.models import PipelineContext

class Stats(PipelineContext):
    calls: int = 0

@step
async def record(data: str, *, context: Stats | None = None) -> str:
    if context:
        context.calls += 1
    return data

pipeline = record >> record
registry = PipelineRegistry()
registry.register(pipeline, "ctx", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="ctx",
    pipeline_version="1.0.0",
    context_model=Stats,
)
final = runner.run("hi", initial_context_data={"run_id": "ctx-run"})
print(final.final_pipeline_context.calls)  # 2
```

### Iterative Loops with `LoopStep`

Some workflows require repeating a set of steps until a condition is met. `LoopStep`
lets you express this directly in the DSL.

```python
from flujo import Step, Flujo, Pipeline, step

@step
async def fixer(data: str) -> str:
    return data + "!"

body = Pipeline.from_step(fixer)

loop = Step.loop_until(
    name="add_exclamation",
    loop_body_pipeline=body,
    exit_condition_callable=lambda out, ctx: out.endswith("!!!"),
    max_loops=3,
)
registry = PipelineRegistry()
registry.register(loop, "loop", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="loop",
    pipeline_version="1.0.0",
)
result = runner.run("hi", initial_context_data={"run_id": "loop-run"})
print(result.step_history[-1].output)  # 'hi!!!'
```

### Conditional Branching with `ConditionalStep`

Sometimes a pipeline should take different actions depending on earlier results. `ConditionalStep` lets you define that logic declaratively.

```python
def choose(out, ctx):
    return "positive" if "!" in out else "neutral"

branches = {
    "positive": Pipeline.from_step(
        step(name="yay")(lambda x: x + " 😊")
    ),
    "neutral": Pipeline.from_step(
        step(name="meh")(lambda x: x)
    ),
}

branch = Step.branch_on(
    name="sentiment_router",
    condition_callable=choose,
    branches=branches,
)

pipeline = fixer >> branch
registry = PipelineRegistry()
registry.register(pipeline, "branch", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="branch",
    pipeline_version="1.0.0",
)
print(
    runner.run("ok", initial_context_data={"run_id": "branch-run"}).step_history[-1].output
)
```

You're now ready to build powerful and intelligent AI applications. Happy orchestrating
