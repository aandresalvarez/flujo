 
<div align="center">
  <img src="assets/flujo.png" alt="Flujo logo" width="180"/>
  <h1>Flujo — The Application Server for AI Workflows</h1>
  <p>
    <b>From brittle prototypes to governed, reliable, and observable production systems.</b>
  </p>
  
  <p>
    <a href="https://github.com/flujo-ai/flujo/actions/workflows/ci.yml">
      <img src="https://github.com/flujo-ai/flujo/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status (main)"/>
    </a>
    <a href="https://pypi.org/project/flujo/"><img src="https://img.shields.io/pypi/v/flujo.svg" alt="PyPI version"></a>
    <a href="https://github.com/flujo-ai/flujo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/flujo.svg" alt="License"></a>
  </p>
</div>

---

## Why Flujo?

Building production-grade AI is more than just calling an API.  
Teams moving from a demo to a deployed service face **three major challenges**:

1. **Runaway Costs** — Without governance, usage spikes can break budgets.  
2. **Operational Brittleness** — Ad-hoc scripts and loose prompt chains are fragile.  
3. **Governance & Compliance Risk** — Lack of structured logging, auditing, and oversight slows adoption.

Flujo solves these problems with:

- **Real-time Cost Attribution & Guardrails** — Track spend per step, enforce usage limits, prevent runaway costs.  
- **Structured, Type-Safe Workflows** — Built on Pydantic to ensure every output is validated and consistent.  
- **Full Observability & Auditability** — Durable run history, traces, and integration with OpenTelemetry for production visibility.

---

## Platform Capabilities

**Build**  
- Python-native DSL for declarative pipelines.  
- Parallelism, conditional logic, and loops made first-class.  
- Type-safe agents with automatic validation and repair.

**Govern**  
- Cost caps and usage quotas built into the runtime.  
- Pricing configuration per model (`flujo.toml`).  
- Policy-driven execution with safe sandboxes.

**Operate**  
- Real-time tracing and inspection via CLI (`flujo lens`).  
- Human-in-the-Loop (HITL) pauses and resumptions.  
- State persistence (SQLite, file, or in-memory backends).

---

## Self-Improving Flywheel

Flujo isn’t just static orchestration — it learns.  
Pipelines can run automated evaluations, detect weak points, and produce an **Improvement Report** with actionable changes to prompts, agents, and settings.

---

## Core Features

* **🧱 Declarative & Composable Workflows** — Define logic as `Step` objects and compose with the `>>` operator.  
* **🛡️ Structured Outputs** — Auto-validation via Pydantic models.  
* **🔧 Automatic Error Recovery** — Built-in repair agents, retries, timeouts, and fallbacks.  
* **💰 Built-in Cost Governance** — Hard usage limits, per-step cost tracking.  
* **🔭 Production-Grade Observability** — Tracing, OpenTelemetry integration, CLI inspection.  
* **🔄 Resumable Pipelines** — HITL-ready with persistent state backends.

---

## Quick Start

```bash
pip install flujo openai pydantic
export OPENAI_API_KEY="sk-..."
```

**Example: A simple translation pipeline**

```python
import asyncio
from pydantic import BaseModel
from flujo import Step, make_agent_async

class Translation(BaseModel):
    original_text: str
    translated_text: str
    language: str

translator_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are an expert translator. Translate the user's text into French.",
    output_type=Translation,
)

translation_pipeline = Step.solution(
    name="TranslateToFrench",
    agent=translator_agent
)

async def main():
    result = await translation_pipeline.run("Hello, world!")
    if result.success:
        translation = result.output
        print(f"Original: {translation.original_text}")
        print(f"Language: {translation.language}")
        print(f"Translated: {translation.translated_text}")
        print(f"Cost: ${result.cost_usd:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python main.py
```

---

## CLI Examples

```bash
# Run a pipeline
flujo run my_pipeline.py --input "Initial data"

# List runs
flujo lens list

# Inspect a specific run trace
flujo lens trace <run_id>

# Validate pipeline
flujo validate my_pipeline.py
```

---

## Installation

```bash
pip install flujo
```

For extras:
```bash
pip install flujo[openai,anthropic,prometheus]
```

---

## License

Flexible **AGPL-3.0 / Commercial**. See the [`LICENSE`](LICENSE) file for details.
 
