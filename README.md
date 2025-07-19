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

## 1 · **Memory & Resumability** (built‑in durability)

*Zero‑infra durability.* Every step is automatically persisted to a local **SQLite** (or file) back‑end.
If the host restarts mid‑run, your workflow **resumes exactly where it left off**—no re‑queueing, no lost context.

## 2 · **Automatic Budget Guardrails** (proactive governance)

*Model‑agnostic guardrails.* Set a per‑run budget (`$0.50`, `2 M tokens`, etc.).
Flujo's **`UsageGovernor`** halts the pipeline — and cancels in‑flight parallel tasks — **before** costs exceed the limit.

## 3 · **Actionable Evals** (self‑healing workflows)

Observability shows *what* broke; Flujo goes further.
Run `flujo improve <dataset>` → an AI meta‑agent analyses failures and **auto‑generates JSON patches** for prompts & config, closing the feedback loop.

---

## Flujo vs. Alternatives — Day‑2 Snapshot

| Feature                          | **Flujo**             | LangChain + LangGraph  | Crew AI     | n8n / Make     |
| :------------------------------- | :-------------------- | :--------------------- | :---------- | :------------- |
| **Built‑in Persistent State**    | ✅ **SQLite / file**   | ⚠️ User‑supplied store | ⚠️ External | 🔒 SaaS hidden |
| **Hard Cost Governor (per‑run)** | ✅ **Proactive**       | ❌                      | ❌           | ❌              |
| **Self‑Healing Eval Loop**       | ✅ **`flujo improve`** | ⚠️ LangSmith (observe) | ❌           | ❌              |
| **Self‑Host Friendly**           | ✅                     | ⚠️ needs DB & infra    | ⚠️ needs DB | ❌              |
| **Licence**                      | AGPL / Commercial     | MIT                    | MIT         | Proprietary    |

---

## Engineer's View — A Python "Algebra" for Workflows

### Core expression `@step >>`

```python
from flujo import step, Flujo

@step
async def greet(name: str) -> str:
    return f"Hello, {name}!"

@step
async def shout(text: str) -> str:
    return text.upper() + "!!!"

pipeline = greet >> shout
print(Flujo(pipeline).run("world").step_history[-1].output)
# → "HELLO, WORLD!!!"
```

### Control‑flow primitives & encapsulation

* 🔄 `Step.loop_until(...)` — iterative refinement
* 🔀 `Step.branch_on(...)` — typed conditionals
* ⚡ `Step.map_over(...)` / `Step.parallel(...)` — fan‑out concurrency
* 👤 `Step.human_in_the_loop(...)` — pause for approval

Wrap any pipeline into **one reusable `Step`** via `runner.as_step()`, enabling hierarchical, testable systems.

---

## Showcase — Stateful, Budget‑Aware *Financial Analyst*

```python
# financial_analyst.py
import asyncio, random
from pathlib import Path
from pydantic import BaseModel, Field
from flujo import Flujo, Step, step, UsageLimits, init_telemetry
from flujo.state import SQLiteBackend

# 1️⃣  Shared run‑state ("memory")
class MarketCtx(BaseModel):
    companies: list[str] = Field(default_factory=list)
    findings: dict[str, str] = Field(default_factory=dict)
    final_report: str | None = None

# 2️⃣  Steps
@step
async def fetch_financials(company: str) -> dict:
    await asyncio.sleep(0.3)                           # simulate network latency
    rev = random.randint(4, 6)
    return {"company": company, "text": f"Q3 revenue ${rev} B", "cost_usd": 0.02}

@step
async def summarise(data: dict) -> dict:
    summary = f"{data['company']}: strong performance ({data['text']})."
    return {"summary": summary, "cost_usd": 0.05, "company": data["company"]}

@step
async def final_report(summaries: Any) -> str:
    lines = ["**Quarterly Market Report**"]
    for summary_data in summaries:
        lines.append(f"- {summary_data['summary']}")
    return "\n".join(lines)

# 3️⃣  Compose workflow
analyse_one = fetch_financials >> summarise
pipeline = Step.map_over("AnalyseAll", analyse_one, iterable_input="companies") >> final_report

# 4️⃣  Run with durability & budget
async def main():
    init_telemetry()
    runner = Flujo(
        pipeline,
        context_model=MarketCtx,
        state_backend=SQLiteBackend(Path("reports.db")),
        usage_limits=UsageLimits(total_cost_usd_limit=0.15),   # 15 ¢ cap
        delete_on_completion=False,
    )

    run_id = "q3‑analysis‑2025"
    try:
        async for result in runner.run_async(None, initial_context_data={"companies": ["Alpha", "Beta", "Gamma"]}, run_id=run_id):
            pass  # Get the last result
        print("\n🎉 Done!\n", result.step_history[-1].output)
    except Exception as err:
        print(f"\n⚠️  Halted: {err}")

if __name__ == "__main__":
    asyncio.run(main())
```

> Runs out‑of‑the‑box: no DB setup, and the workflow halts if total spend exceeds $0.15.

---

## Quick‑start

```bash
pip install flujo
```

See **[`docs/quickstart.md`](docs/quickstart.md)** for a 60‑second hello‑world.

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
