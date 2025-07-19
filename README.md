<div align="center">

<img src="assets/flujo.png" alt="Flujo logo" width="180"/>

**Beyond Agents: The AI Employee.**

</div>

---

> **TL;DR**
> Flujo turns AI **agents** into production‑grade **digital employees**—with persistent memory, proactive budgeting, and continuous learning.

> **Approaching v1.0**  Core APIs (`@step`, `>>`, `Flujo`, state back‑ends) are stable. Pin your version and watch the changelog.

> **Flexible Licensing**  Dual‑licensed **AGPL‑3.0 / Commercial**. See [`LICENSE`](LICENSE) for details.

---

# Meet Flujo — The AI Employee That Delivers

Moving AI from prototype to production shouldn’t mean chaos. Flujo gives you AI workflows as reliable, accountable, and effective as your best employees.

## Traits of a Perfect AI Employee

### ✅ Never Forgets (**Durability**)

\* **Problem:** AI workflows often crash and lose progress.
\* **Flujo:** Automatically saves state with built‑in **SQLite**, resuming exactly where tasks left off.

### ✅ Keeps Spending in Check (**Governance**)

\* **Problem:** AI processes can overspend unpredictably.
\* **Flujo:** Set strict usage limits (e.g., `$0.50` per run); proactive cost guards halt execution before you overspend.

### ✅ Learns from Mistakes (**Improvement Loop**)

\* **Problem:** Debugging AI is slow and manual.
\* **Flujo:** `flujo improve` analyzes failures and auto‑generates concrete prompt and config suggestions.

### ✅ Knows When to Escalate (**Safety Rails**)

\* **Problem:** AI can’t handle every edge‑case alone.
\* **Flujo:** Route edge cases to human approval with `Step.branch_on` and `Step.human_in_the_loop`.

### ✅ Communicates Clearly (**Observability**)

\* **Problem:** AI tasks are a black box until they fail.
\* **Flujo:** Get real‑time updates via event hooks and full run histories with the `flujo lens` CLI.

---

## Simple Python Workflow, Powerful Results

```python
from flujo import step, Flujo, Step, make_agent_async

@step
async def validate_input(text: str) -> str:
    if not text:
        raise ValueError("Input required.")
    return text

summariser = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt="You are an expert summariser.",
    output_type=str,
)

pipeline = validate_input >> Step("Summarise", summariser)

print(Flujo(pipeline).run("Flujo is...").step_history[-1].output)
```

---

## The Flujo Advantage

* ✅ **Build Autonomously:** Compose agents that handle routine work and escalate edge cases with `Step.human_in_the_loop`.
* ✅ **Run Efficiently:** Execute tasks concurrently with `Step.parallel`, eliminate redundant work with `Step.cached`, and rely on a high‑performance runtime.
* ✅ **Audit Everything:** Get a complete, persistent history of every run. Use the `flujo lens` CLI to trace decisions and debug failures.
* ✅ **Integrate Anywhere:** `@step` turns any `async` Python code into a durable workflow component. Event hooks connect Flujo to your existing monitoring and notification tools.

---

## Showcase: A Stateful, Budget‑Aware AI Financial Analyst

This example builds a multi‑agent workflow that analyzes a list of companies, persists its state to SQLite, and halts if the total cost exceeds a 15¢ budget.

```python
# examples/financial_analyst.py
from flujo import (
    Flujo, Step, step, UsageLimits, make_agent_async
)
from flujo.state import SQLiteBackend
from flujo.domain.models import PipelineContext
from flujo.domain.dsl.step import adapter_step
from pydantic import Field

# 1️⃣ Define a shared context ("memory") for the run
class MarketCtx(PipelineContext):
    companies: list[str] = Field(default_factory=list)

# 2️⃣ Define steps using Python code and AI agents
@step
async def fetch_data(company: str) -> dict:
    return {"company": company, "data": f"Financial info for {company}"}

@adapter_step
async def select_data(fetched: dict) -> str:
    return fetched.get("data", "")

summariser = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt="Summarise the financial data point.",
    output_type=str,
)

# 3️⃣ Compose steps into a high‑level workflow
analysis_flow = fetch_data >> select_data >> Step.model_validate({
    "name": "Summarise",
    "agent": summariser
})

pipeline = Step.map_over(
    "AnalyseAllCompanies",
    analysis_flow,
    iterable_input="companies",
)

# 4️⃣ Execute the workflow with durability and governance
runner = Flujo(
    pipeline,
    context_model=MarketCtx,
    state_backend=SQLiteBackend("financial_reports.db"),
    usage_limits=UsageLimits(total_cost_usd_limit=0.15),
)

result = runner.run(
    initial_input=None,
    initial_context_data={"companies": ["Alpha", "Beta", "Gamma"]},
)
print(result.step_history[-1].output)
```

> This example is fully runnable. Copy it to a file and run with your `OPENAI_API_KEY`.

---

## Flujo vs Alternatives

| Feature                   | **Flujo**         | LangChain / LangGraph | Crew AI     | n8n / Make     |
| ------------------------- | ----------------- | --------------------- | ----------- | -------------- |
| Built‑in Persistent State | ✅ SQLite          | ⚠️ External store     | ⚠️ External | 🔒 Hidden SaaS |
| Proactive Cost Governor   | ✅                 | ❌                     | ❌           | ❌              |
| Self‑Improvement Loop     | ✅ `flujo improve` | ⚠️ Observe‑only       | ❌           | ❌              |
| Self‑Hosting Friendly     | ✅                 | ⚠️ Needs infra        | ⚠️ Needs DB | ❌              |
| Licence                   | AGPL / Commercial | MIT                   | MIT         | Proprietary    |

---

## Roadmap — Reliability at Scale

| Capability                | Status        |
| ------------------------- | ------------- |
| ✅ Persistent & Durable    | **Done**      |
| ✅ Budget Controls         | **Done**      |
| ✅ Parallel & Caching      | **Done**      |
| ✅ Conditional Routing     | **Done**      |
| 🟡 Notifications & Hooks  | *In Progress* |
| 🗺️ Security & Compliance | *Upcoming*    |

---

## Quick Start (60 seconds)

```bash
pip install flujo

echo '
from flujo import step

@step
async def hello(name: str) -> str:
    return f"Hello, {name}!"

# The `flujo run` CLI looks for a top‑level variable named "pipeline"
pipeline = hello
' > hello_pipeline.py

flujo run hello_pipeline.py --input "Flujo"
```

> Expected output: `Hello, Flujo!`

---

## Get Involved

* 📖 **[Documentation](docs/index.md)** — Guides, tutorials, API reference
* 🤝 **[Contribute](CONTRIBUTING.md)** — Join the community and shape Flujo’s future

---

## Licensing

Flexible **AGPL‑3.0 / Commercial**. See the [`LICENSE`](LICENSE) file for details.
