 # A Guided Tour of the AI Orchestrator

Welcome! This tutorial will guide you through using the `pydantic-ai-orchestrator` library, from your very first request to building advanced, custom AI workflows. We'll start with the basics and progressively build up to more complex examples.

**Before You Begin:**
*   You should have a basic understanding of Python.
*   Make sure you have set up your API keys (e.g., `OPENAI_API_KEY`) in a `.env` file in your project directory. The orchestrator will automatically find and use them.

---

## Key Concepts: The Building Blocks

Before we write any code, let's understand the main components you'll be working with. Think of it like a chef learning about their ingredients before cooking.

*   **The Orchestrator:** This is the "project manager" or "conductor." Its job is to manage a team of specialized AI **Agents** to accomplish a goal. It directs the workflow, handles retries, and ensures the final result is high-quality.

*   **An Agent:** An **Agent** is a specialized AI model given a specific role and instructions (a "system prompt"). In our library, we have a team of four default agents:
    1.  **`review_agent` (The Planner):** Looks at your request and creates a detailed checklist of what a "good" solution should look like.
    2.  **`solution_agent` (The Doer):** The main worker. It takes your request and tries to produce a solution (e.g., write code, a poem, or an email).
    3.  **`validator_agent` (The Quality Analyst):** Takes the `solution_agent`'s work and grades it against the `review_agent`'s checklist.
    4.  **`reflection_agent` (The Strategist):** If the work isn't perfect, this agent analyzes the failures and provides high-level feedback to guide the next attempt.

*   **A Task:** This is a simple object that holds your request. It's how you tell the **Orchestrator** what you want to achieve.

*   **A Candidate:** This is the final result produced by the Orchestrator. It contains the solution itself, its final score, and the checklist used to grade it.

Now that we know the players, let's see them in action!

---

## 1. Your First AI Task: Simple Orchestration

Let's start with the most straightforward use case: give the orchestrator a prompt and let it manage its default team of smart agents to get the job done. This shows the entire self-correction loop in action.

```python
# 📂 step_1_basic_usage.py
from pydantic_ai_orchestrator import (
    Orchestrator, Task, review_agent, solution_agent, validator_agent, reflection_agent
)

print("🤖 Assembling the AI agent team...")
orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent)

print("📝 Defining task: 'Write a short, optimistic haiku about a rainy day.'")
task = Task(prompt="Write a short, optimistic haiku about a rainy day.")

print("🧠 Running the orchestrator...")
best_candidate = orch.run_sync(task)

if best_candidate:
    print("\n🎉 Orchestrator finished! Here is the best result:")
    print("-" * 50)
    print(f"Final Score: {best_candidate.score:.2f} / 1.00")
    print(f"\nSolution (the haiku):\n'{best_candidate.solution}'")
    if best_candidate.checklist:
        print("\nSelf-Correction Checklist:")
        for item in best_candidate.checklist.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            print(f"  - {item.description:<45} {status}")
```
You've successfully orchestrated a team of AI agents to complete a creative task with built-in quality control.

---

## 2. Taking Control: Weighted Scoring

Sometimes, not all requirements are created equal. The orchestrator allows you to provide a **weighted checklist** to guide the scoring, giving you precise control over what "good" means. For this, we'll create our own custom `review_agent`.

```python
# 📂 step_2_weighted_scoring.py
from pydantic_ai_orchestrator import (
    Orchestrator, Task, settings, make_agent_async, solution_agent,
    validator_agent, reflection_agent, Checklist
)

settings.scorer = "weighted"
our_weights = [
    {"item": "The function must have a docstring.", "weight": 0.7},
    {"item": "The function must use type hints.", "weight": 0.3},
]
CUSTOM_REVIEW_PROMPT = """You are an expert software engineer. Your task is to generate a checklist with EXACTLY these two items: "The function must have a docstring." and "The function must use type hints.". Return JSON only."""
print("📋 Creating a custom 'Planner' agent with weighted criteria...")
custom_review_agent = make_agent_async(settings.default_review_model, CUSTOM_REVIEW_PROMPT, Checklist)
orch = Orchestrator(custom_review_agent, solution_agent, validator_agent, reflection_agent)
task = Task("Write a simple Python function that adds two numbers.", metadata={"weights": our_weights})
best_candidate = orch.run_sync(task)
# ... (printing logic is the same as the previous tutorial)
```
By setting `settings.scorer = "weighted"` and passing weights in the `Task`'s metadata, you have successfully influenced the AI's definition of quality. A solution that only includes a docstring will now score `0.70`.

> **💡 Pro Tip: Scoring Strategies**
> The orchestrator supports three scoring methods, which you can set via `settings.scorer`:
> - **`"ratio"` (Default):** The score is the simple percentage of passed checklist items.
> - **`"weighted"`:** The score is calculated based on the weights you provide.
> - **`"reward"`:** An advanced mode where another AI model acts as a judge to give a holistic quality score from 0.0 to 1.0, ignoring the checklist.

---

## 3. The Budget-Aware Workflow: Mixing Agent Models

Professional AI workflows often involve a mix of models to balance cost, speed, and quality. Here, we'll use a **cheaper, faster model** for the initial draft (`solution_agent`) but retain the **smarter models** for the critical thinking roles (planning, quality control, and strategy).

```python
# 📂 step_3_mixing_models.py
from pydantic_ai_orchestrator import (
    Orchestrator, Task, review_agent, validator_agent, reflection_agent, make_agent_async
)
print("🚀 Building a budget-aware workflow by mixing models...")
FAST_SOLUTION_PROMPT = "You are a creative but junior marketing copywriter. Write a catchy and concise slogan. Be quick and creative."
fast_copywriter_agent = make_agent_async("openai:gpt-4.1-nano", FAST_SOLUTION_PROMPT, str)
orch = Orchestrator(review_agent, fast_copywriter_agent, validator_agent, reflection_agent)
task = Task(prompt="Write a slogan for a new brand of ultra-durable luxury coffee mugs.")
best_candidate = orch.run_sync(task)
# ... (printing logic)
```
This "cheap drafter, smart reviewer" pattern is a powerful way to get high-quality results efficiently. The fast agent produces drafts, and the smart agents ensure the final output is excellent.

---

## 4. Outputting Structured Data: Pydantic Models

So far, our agents have only outputted simple strings. What if we need structured data, like JSON? The underlying `pydantic-ai` library excels at this. You can specify a Pydantic `BaseModel` as the `output_type` for an agent.

Let's build a workflow that extracts information from a block of text into a structured `ContactCard` model.

```python
# 📂 step_4_structured_output.py
from pydantic import BaseModel, Field
from pydantic_ai_orchestrator import (
    Orchestrator, Task, make_agent_async, validator_agent, reflection_agent
)

# 1. Define our desired output structure using a Pydantic model
class ContactCard(BaseModel):
    name: str = Field(..., description="The full name of the person.")
    email: str | None = Field(None, description="The person's email address.")
    company: str | None = Field(None, description="The company they work for.")

# 2. Create a custom Solution Agent to extract information
print("🛠️ Creating a data-extraction agent...")
EXTRACTION_PROMPT = "You are a data-entry expert. Extract contact information from the user's text and format it precisely according to the ContactCard schema. If a field is not present, omit it."
extraction_agent = make_agent_async("openai:gpt-4o", EXTRACTION_PROMPT, ContactCard)

# 3. Create a custom Review Agent that understands structured data
REVIEW_PROMPT = "Generate a checklist to verify the extracted contact details. Check for name correctness, email validity, and company presence."
custom_review_agent = make_agent_async("openai:gpt-4o", REVIEW_PROMPT, Checklist)

# 4. Assemble and run the orchestrator
orch = Orchestrator(custom_review_agent, extraction_agent, validator_agent, reflection_agent)
unstructured_text = "Reach out to Jane Doe. She works at Innovate Corp and her email is jane.doe@example.com. She is the lead."
task = Task(prompt=unstructured_text)
best_candidate = orch.run_sync(task)

if best_candidate:
    # The solution is now a Pydantic model object, not just a string!
    contact_card_solution = best_candidate.solution
    print("\n✅ Successfully extracted structured data:")
    print(contact_card_solution.model_dump_json(indent=2))
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

## 5. The Grand Finale: A Fully Custom Team with Tools

Now for the ultimate challenge. Let's build a workflow where **every agent is customized**, and our `solution_agent` can use **external tools** to get information it doesn't have.

**Scenario:** We need to write a factual report on a public company's stock price. The LLM doesn't know real-time stock prices, so it will need a tool.

1.  **Custom Planner:** A `review_agent` that knows what a good financial report looks like.
2.  **Tool-Using Doer:** A `solution_agent` that can call a `get_stock_price` function.
3.  **Custom Quality Analyst:** A `validator_agent` that is hyper-critical about financial data.
4.  **Custom Strategist:** A `reflection_agent` focused on factual accuracy.

```python
# 📂 step_5_advanced_tools.py
import random
from pydantic import BaseModel
from pydantic_ai import Tool
from pydantic_ai_orchestrator import * # Import all for convenience

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

# The Strategist: Focused on facts
reflection_agent = get_reflection_agent("openai:gpt-4o-mini")

# --- 3. Assemble and Run the Orchestrator ---
orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent)
task = Task(prompt="Generate a stock report for Apple Inc. (AAPL).")

print("🧠 Running advanced tool-based workflow...")
best_candidate = orch.run_sync(task)

if best_candidate:
    print("\n🎉 Advanced workflow complete!")
    print(best_candidate.solution.model_dump_json(indent=2))
    print("\nFinal Score:", best_candidate.score)
```

#### **What You'll See:**

During the execution, you will see a message from our tool function:
`TOOL USED: Getting stock price for AAPL...`

This confirms that the `solution_agent` recognized it needed information, called the function you provided, and used the result in its answer. The final output will be a perfectly structured report with the "live" data.

---

This concludes our tour! You've journeyed from a simple prompt to a sophisticated, tool-using AI system. You've learned to:
-   Understand the core concepts of **Orchestrators and Agents**.
-   Run a basic multi-agent task and interpret its self-correction process.
-   Control the definition of quality using **weighted scoring**.
-   Optimize workflows by **mixing different AI models**.
-   Generate clean, **structured JSON** using Pydantic models.
-   Empower agents with **external tools** to overcome their knowledge limitations.

## 6. Building Custom Pipelines

The new Pipeline DSL lets you compose your own workflow using `Step` objects. Execute the pipeline with `PipelineRunner`:

```python
from pydantic_ai_orchestrator import Step, PipelineRunner
from pydantic_ai_orchestrator.plugins.sql_validator import SQLSyntaxValidator
from pydantic_ai_orchestrator.testing.utils import StubAgent

sql_step = Step.solution(StubAgent(["SELECT FROM"]))
check_step = Step.validate(StubAgent([None]), plugins=[SQLSyntaxValidator()])
runner = PipelineRunner(sql_step >> check_step)
result = runner.run("SELECT FROM")
print(result.step_history[-1].feedback)
```

You're now ready to build powerful and intelligent AI applications. Happy orchestrating
