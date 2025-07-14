<div align="center">
  <img src="assets/flujo.png" alt="Flujo Logo" width="180"/>
</div>

> **TL;DR**
> Stop babysitting brittle prompt chains. **Flujo** runs your LLM agents **durably**, **within budget**, and helps them **learn from every run**.

> **⚠️ Pre-v1 Status**
> Flujo is actively evolving towards v1.0. While we're committed to stability, breaking changes may occur as we refine the API. We recommend pinning your version and checking our [changelog](CHANGELOG.md) for updates. The core concepts are stable, but some interfaces are still being optimized.

---

# Flujo: Building AI Systems That Learn

**Prototype chains break in production** – they forget context on restart, blow through token budgets, and their prompts go stale without constant human intervention.

**Flujo is designed to fix this.** It's a Python-native framework for building AI systems that **observe, learn, and improve themselves over time.** While many tools can connect LLM calls, Flujo solves the Day-2 problems of running AI in the real world: How do you ensure reliability? How do you govern costs? And how do you create a feedback loop to make your agents smarter?

## Why Flujo?

Today, any developer can prototype an LLM agent in minutes. But shipping it to production reveals three hard problems that simple frameworks don't solve:

| The Pain | The Flujo Solution |
| :--- | :--- |
| **Durability:** A server restarts and your long-running agent loses all its progress and memory. | **Durable State:** Flujo has built-in SQLite & File backends to automatically save state after every step, enabling workflows that survive crashes and can be resumed. |
| **Governance:** An agent gets stuck in a loop and you get a surprise $300 bill from your LLM provider. | **Cost & Token Guardrails:** Flujo's `UsageGovernor` lets you set a hard budget on any run. The pipeline halts automatically if it exceeds its limit. |
| **Improvement:** Your prompts go stale, and you have no data-driven way to know if your changes are actually making the agent better or worse. | **AI-Driven Evals:** With `flujo improve`, you can run evaluations against a dataset, feed the results to a meta-agent, and get concrete, AI-generated patches for your prompts and configs. |

## The Flujo Vision: Self-Optimizing Workflows

Our architecture is built for a future where AI systems improve themselves. A workflow's source of truth is a **declarative YAML file**, which is parsed at runtime into a **type-safe Pydantic graph**. This enables our core vision: **AI-driven self-optimization.** The Flujo engine uses telemetry to identify opportunities for improvement—like high-cost steps or common failure patterns—and generates suggested changes as auditable **JSON Patches**.

This creates a powerful feedback loop where your systems learn from their own performance.

## Key Features: The Pillars of Reliable AI

Flujo's current features are the essential foundation for building intelligent, adaptive systems.

-   🧠 **Stateful & Durable by Default:** True learning requires memory. Flujo's built-in **SQLite backend** provides durable state persistence, so workflows can survive restarts and their history can be analyzed for improvement patterns.
-   ⛓️ **Explicit & Structured Control Flow:** Adaptive logic needs a predictable canvas. Flujo's Pythonic DSL provides first-class **loops (`Step.loop_until`)**, **conditionals (`Step.branch_on`)**, and **parallel fan-outs (`Step.parallel`)**, giving you full control over the agent's execution paths.
-   🔭 **Deep Observability as a Data Source:** You can't optimize what you can't see. One-line `init_telemetry()` enables **OpenTelemetry (OTLP)** tracing. This detailed data isn't just for debugging; it's the raw material for the AI meta-agent to analyze and improve the system.
-   💸 **Production Guardrails & Economic Policies:** Safety is paramount. The **Usage Governor** enforces hard cost and token limits. Our roadmap extends this to an **Economic Policy DSL**, allowing you to define sophisticated, ROI-aware rules for your agents' resource consumption.
-   🧩 **Composable Python-Native Design:** Build complex systems from simple, testable Python functions and classes. Flujo's design avoids configuration magic, keeping your logic clean, maintainable, and easy to integrate.

---

## Get Started in 60 Seconds

### 1. Installation

```bash
pip install flujo
```

### 2. Set Up Your API Key

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY="sk-..."
```

### 3. Run Your First Agentic Loop

This simple example creates an agent that decides which tool to use.

```python
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo import make_agent_async, init_telemetry
from flujo.domain.commands import AgentCommand
from pydantic import TypeAdapter

# Enable telemetry (optional but recommended)
init_telemetry()

async def search_agent(query: str) -> str:
    """A simple tool agent that returns information."""
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language."
    return "No information found."

PLANNER_PROMPT = """
You are a research assistant. Use the `search_agent` tool to gather facts.
When you know the answer, issue a `FinishCommand` with the final result.
"""
planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    TypeAdapter(AgentCommand),
)

# Create the pipeline using the factory
pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry={"search_agent": search_agent}
)

# Run the pipeline
# Note: In a real script, you'd use `await run_agentic_loop_pipeline(...)`
# For simplicity, this is a conceptual example.
# result = await run_agentic_loop_pipeline(pipeline, "What is Python?")
# print(result)
```

---

## Showcase: The Stateful, Budget-Aware Financial Analyst

This example highlights Flujo's unique strengths: orchestrating a stateful, multi-step process that operates under a strict budget and persists its state for auditing.

```python
# financial_analyst.py
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field

from flujo import Flujo, Step, step, init_telemetry, UsageLimits
from flujo.state import SQLiteBackend

# --- 1. Define the Shared State (The "Memory" for this Run) ---
class MarketAnalysisContext(BaseModel):
    companies: list[str] = Field(default_factory=list)
    findings: dict[str, str] = Field(default_factory=dict)
    final_report: str | None = None

# --- 2. Define the Specialized Agents & Steps ---
@step
async def fetch_financials(company: str) -> dict:
    print(f"   🔎 Fetching financials for: {company}...")
    class FinancialData(BaseModel):
        data: str; cost_usd: float = 0.02
    return FinancialData(data=f"Q3 revenue for {company} was $5B.")

@step
async def summarize_data(data: dict, *, context: MarketAnalysisContext) -> dict:
    print(f"   ✍️  Summarizing data...")
    company = data['data'].split(" ")[3]
    summary = f"Summary for {company}: Strong performance with revenue of $5B."
    context.findings[company] = summary # Update shared memory
    class SummaryOutput(BaseModel):
        summary: str; cost_usd: float = 0.05
    return SummaryOutput(summary=summary)

@step
async def generate_final_report(*, context: MarketAnalysisContext) -> str:
    print("   📈 Generating final market report...")
    report_lines = ["**Quarterly Market Report**"]
    for company, summary in context.findings.items():
        report_lines.append(f"- {summary}")
    return "\n".join(report_lines)

# --- 3. Assemble the Workflow with Flujo's DSL ---
company_analysis_pipeline = fetch_financials >> summarize_data
full_pipeline = (
    Step.map_over(name="AnalyzeAllCompanies", pipeline_to_run=company_analysis_pipeline, iterable_input="companies")
    >> generate_final_report
)

# --- 4. Orchestrate the Workflow with Production Guardrails ---
async def main():
    init_telemetry()
    backend = SQLiteBackend(Path("financial_reports.db"))
    limits = UsageLimits(total_cost_usd_limit=0.15)

    runner = Flujo(full_pipeline, context_model=MarketAnalysisContext, state_backend=backend, usage_limits=limits, delete_on_completion=False)
    run_id = "q3-market-analysis-2025"
    initial_data = { "companies": ["AlphaCorp", "BetaInc", "GammaLLC", "DeltaCo"] }

    print(f"🚀 Starting Financial Analysis workflow (run_id: {run_id})")
    print(f"   (Budget: ${limits.total_cost_usd_limit:.2f})\n")

    try:
        result = await runner.arun(None, initial_context_data=initial_data, run_id=run_id)
        print("\n🎉 Workflow complete!")
        print(result.step_history[-1].output)
    except Exception as e:
        print(f"\n⚠️  Workflow halted: {e}")

    final_state = await backend.load_state(run_id)
    if final_state:
        print("\n--- Final Workflow State (from DB) ---")
        final_ctx = MarketAnalysisContext.model_validate(final_state['pipeline_context'])
        print(f"Companies processed: {list(final_ctx.findings.keys())}")
        print(f"Final status: {final_state['status']}")
        print(f"Total cost incurred: ${final_state.get('cost_usd', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```
**What makes this different:** This stateful, budget-aware process would require significant custom code to implement reliably in simpler frameworks. Flujo handles the durability and governance automatically.

## Documentation & Community

- **[Full Documentation & Guides](docs/index.md):** The best place to start.
- **[Examples Directory](examples/):** See more patterns in action.
- **[Contributing Guide](CONTRIBUTING.md):** Join us in building the future of reliable AI systems.

## License

This project is dual-licensed under AGPL-3.0 and a Commercial License. See [LICENSE](LICENSE) for more information.
