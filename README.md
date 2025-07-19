<div align="center">

<img src="assets/flujo.png" alt="Flujo logo" width="180"/>

**Turn fragile LLM scripts into durable, budget‑safe services.**

</div>

---

> **TL;DR**
> **Flujo** is a Python framework for building AI workflows that
> **remember their progress (✅ state)**, **stick to a budget (✅ guardrails)**,
> and **help you make them smarter over time (✅ auto‑evals)**.

> **Pre‑v1 Notice**
> We're closing in on **v1.0**. The core APIs (`@step`, `>>`, `Flujo`, state back‑ends) are **stable**.
> Finishing touches—AI‑driven evals & improved tooling—may adjust secondary interfaces.
> **Pin your version** and check the changelog on upgrade.

> **Licensing**
> Dual‑licensed **AGPL‑3.0 / Commercial**. The AGPL keeps the core open;
> commercial terms (with priority support) are available for proprietary use‑cases.
> See [`LICENSE`](LICENSE) for details.

---

# Flujo — Production‑Grade Workflows for LLM Agents

Prototype chains wow in demos—then crash, loop, or overspend in prod.
Flujo solves these **"day‑2" headaches** without Redis, Kafka, or any external DB.

## 1. An Employee Who Remembers Their Work (Durability)

*Zero‑infra durability.* Every step is automatically persisted to a local **SQLite** (or file) back‑end. If the host restarts mid‑run, your workflow **resumes exactly where it left off**—no re‑queueing, no lost context.

## 2. An Employee Who Sticks to a Budget (Governance)

*Model‑agnostic guardrails.* Set a per‑run budget (`$0.50`, `2 M tokens`, etc.). Flujo's **`UsageGovernor`** halts the pipeline — and cancels in‑flight parallel tasks — **before** costs exceed the limit.

## 3. An Employee Who Helps You Improve (Actionable Evals)

*Observability shows what broke; Flujo shows how to fix it.* Run `flujo improve <dataset>` → an AI meta‑agent analyses failures and **auto‑generates JSON patches** for prompts & config, closing the feedback loop.

---

## Flujo vs. Alternatives — Day‑2 Snapshot

| Feature | **Flujo** | LangChain + LangGraph | Crew AI | n8n / Make |
| :--- | :--- | :--- | :--- | :--- |
| **Built‑in Persistent State** | ✅ **SQLite / file** | ⚠️ User‑supplied store | ⚠️ External | 🔒 SaaS hidden |
| **Hard Cost Governor (per‑run)** | ✅ **Proactive** | ❌ | ❌ | ❌ |
| **Self‑Healing Eval Loop** | ✅ **`flujo improve`** | ⚠️ LangSmith (observe) | ❌ | ❌ |
| **Self‑Host Friendly** | ✅ | ⚠️ needs DB & infra | ⚠️ needs DB | ❌ |
| **Licence** | AGPL / Commercial | MIT | MIT | Proprietary |

---

## Engineer's View — A Python "Algebra" for Workflows

### Core expression: `@step >>`

A **Step** can be a plain `async` function *or* an **AI agent**. The `>>` operator chains them into a verifiable Pipeline.

```python
from flujo import step, Flujo, Step, make_agent_async

@step
async def validate_input(text: str) -> str:
    if not text:
        raise ValueError("Input cannot be empty!")
    return text

summariser = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt="You are an expert summariser. Be concise.",
    output_type=str,
)

pipeline = validate_input >> Step.model_validate({"name": "Summarise", "agent": summariser})

print(
    Flujo(pipeline).run("Flujo is a Python framework...").step_history[-1].output
)
```

### Built for Production, from Day One

Flujo provides the primitives you need to build robust, observable, and efficient services.

* ✅ **Safe Autonomy:** Build agents that work independently but know when to escalate. Use `Step.loop_until` for autonomous work, and `Step.branch_on` to route edge cases to a `Step.human_in_the_loop` for approval.
* ✅ **High-Performance Execution:** Don't let your framework be the bottleneck. Flujo provides `Step.parallel` for concurrent fan-out, `Step.cached` to eliminate redundant work, and an optimized runtime to minimize overhead.
* ✅ **Full Traceability:** Know exactly what happened. The `flujo lens` CLI gives you a complete, auditable history of every step's execution, input, and output, persisted in a queryable database.
* ✅ **Extensible & Interoperable:** Flujo plays well with your existing tools. The `@step` decorator wraps any `async` Python code, and the event hook system lets you integrate with any notification or monitoring service you use.

---

## Showcase — Stateful, Budget‑Aware AI Financial Analyst

```python
# financial_analyst.py
import asyncio, os
from pathlib import Path
from pydantic import BaseModel, Field
from flujo import (
    Flujo, Step, step, UsageLimits, init_telemetry,
    make_agent_async,
)
from flujo.domain.models import PipelineContext
from flujo.state import SQLiteBackend

# 1️⃣ Shared run‑state ("memory")
class MarketCtx(PipelineContext):
    companies: list[str] = Field(default_factory=list)
    findings: dict[str, str] = Field(default_factory=dict)
    final_report: str | None = None

# 2️⃣ Steps — mix code & AI agents -------------------------------
class FinancialData(BaseModel):
    company: str
    text: str
    cost_usd: float = 0.0

@step
async def fetch_financials(company: str) -> FinancialData:
    print(f"🔎  Fetching {company} financials …")
    revenue = {"Alpha": 5, "Beta": 4, "Gamma": 6}.get(company, 3)
    return FinancialData(company=company, text=f"Q3 revenue was ${revenue} B")

summariser = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt="Summarise the data point in one sentence.",
    output_type=str
)
report_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="Write a professional quarterly market report in Markdown based on the findings.",
    output_type=str
)

# 3️⃣ Pipeline composition ---------------------------------------
analyse_one = fetch_financials >> Step.model_validate({
    "name": "Summarise",
    "agent": summariser
})
pipeline = Step.map_over("AnalyseAll", analyse_one, iterable_input="companies") \
           >> Step.model_validate({
               "name": "FinalReport",
               "agent": report_agent
           })

# 4️⃣ Run with durability & budget -------------------------------
async def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY to run this example.")
    init_telemetry()
    runner = Flujo(
        pipeline, context_model=MarketCtx,
        state_backend=SQLiteBackend(Path("reports.db")),
        usage_limits=UsageLimits(total_cost_usd_limit=0.15),  # 15¢ cap
    )
    try:
        async for result in runner.run_async(
            initial_input=None,
            initial_context_data={"companies": ["Alpha", "Beta", "Gamma"]},
            run_id="q3-analysis-2025",
        ):
            pass  # Get the last result
        print("\n🎉  Done!\n\n" + result.step_history[-1].output)
    except Exception as err:
        print(f"\n⚠️  Halted: {err}")

if __name__ == "__main__":
    asyncio.run(main())
```

> Paste‑and‑run with an `OPENAI_API_KEY`. The workflow persists state to SQLite and halts if total spend exceeds $0.15.

---

## Roadmap — The Path to v1.0 and Beyond

Our goal is to make Flujo the most reliable AI "employee" you can hire. Here's where we are and where we're headed.

| Trait (The Goal) | Flujo Capability (The How) | Status |
| :--- | :--- | :--- |
| **Remembers Their Work** | SQLite / File-based durable state | ✅ **Done** |
| **Sticks to a Budget** | Proactive cost & token governors | ✅ **Done** |
| **Finishes Fast** | Parallel execution, caching, optimized runtime | ✅ **Done** |
| **Escalates Early** | Human-in-the-loop and conditional branching | ✅ **Done** |
| **Communicates Proactively** | Event hooks for Slack, email, webhooks | 🟡 **In Progress** |
| **Owns Their Results** | Enhanced `flujo lens` with version-pinning | 🚧 **Next Up** |
| **Scales With the Business** | Distributed state backends (Redis, etc.) | 🗺️ **On the Horizon** |
| **Guards Sensitive Info** | PII redaction, RBAC hooks, compliance features | 🗺️ **On the Horizon** |

---

## Quick‑start (60 seconds)

```bash
# 1 · Install
pip install flujo

# 2 · Create hello_pipeline.py
echo '
from flujo import step

@step
async def hello(name: str) -> str:
    return f"Hello, {name}!"

# The `flujo run` CLI looks for a top-level variable named "pipeline"
pipeline = hello
' > hello_pipeline.py

# 3 · Run it via the CLI
flujo run hello_pipeline.py --input "Flujo"
```

> Expected output → `Hello, Flujo!`
> For a deeper tour, see **[`docs/quickstart.md`](docs/quickstart.md)**.

---

## Documentation & Community

* **[Full Docs](docs/index.md)** – concepts, API reference, tutorials
* **Examples** – more patterns in [`examples/`](examples/)
* **Integrations** – OpenAI · Gemini · Anthropic · Ollama‑local · adapters for LangChain tools, Vertex AI, and more
* **Coming from LangChain?** – see `docs/migrate_from_langchain.md`
* **[Contributing Guide](CONTRIBUTING.md)** – help shape reliable AI!

---

## License

Flujo is **AGPL‑3.0** with a **Commercial** option.
Choose the model that meets your compliance and distribution needs — details in the [`LICENSE`](LICENSE) file.
