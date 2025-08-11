<div align="center">
  <img src="https://raw.githubusercontent.com/flujo-ai/flujo/main/docs/assets/flujo-logo.png" alt="Flujo Logo" width="400">
  <h1>Flujo: The Production Framework for Reliable AI Applications</h1>
  <p>
    <b>Build, test, and operate structured AI workflows that are reliable, observable, and self-improving.</b>
  </p>
  
  <p>
    <a href="https://github.com/flujo-ai/flujo/actions/workflows/ci.yml"><img src="https://github.com/flujo-ai/flujo/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
    <a href="https://pypi.org/project/flujo/"><img src="https://img.shields.io/pypi/v/flujo.svg" alt="PyPI version"></a>
    <a href="https://github.com/flujo-ai/flujo/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/flujo.svg" alt="License"></a>
  </p>
</div>

---

Building production-grade AI is more than just calling an API. It requires a solid foundation for chaining agents, handling unreliable outputs, managing costs, and ensuring the entire system is observable and resilient.

Flujo is an async-first Python framework that provides this foundation. It allows you to define complex AI workflows as **declarative pipelines of steps**, turning chaotic scripts into structured, testable, and production-ready applications.

## Core Features

Flujo is built on a set of principles directly reflected in its architecture:

*   **🧱 Declarative & Composable Workflows:** Define your logic as a series of `Step` objects and compose them into a `Pipeline` with the `>>` operator. The DSL supports complex patterns like **parallel execution**, **conditional branching**, and **loops**.
    *   *Evidence in code: `flujo/domain/dsl/step.py`, `pipeline.py`, `parallel.py`, `conditional.py`, `loop.py`*

*   **🛡️ Structured & Type-Safe Outputs:** Say goodbye to parsing messy strings. Flujo is built on Pydantic, ensuring that agent outputs are automatically validated against your defined data models.
    *   *Evidence in code: `flujo/agents/factory.py`'s use of `output_type`, `flujo/domain/models.py`*

*   **🔧 Automatic Error Recovery & Repair:** Flujo is designed for resilience. If an LLM returns malformed JSON, a built-in **repair agent** attempts to fix it. The execution engine includes configurable **retries with exponential backoff**, **timeouts**, and **fallback steps**.
    *   *Evidence in code: `flujo/agents/repair.py`, `flujo/application/core/error_recovery_strategies.py`, `Step(fallback_step=...)`*

*   **💰 Built-in Cost Governance:** Track the cost and token usage of every step in your pipeline. Set hard **usage limits** to prevent runaway spending and gain clear insight into your AI operational costs.
    *   *Evidence in code: `flujo/cost.py`, `flujo/domain/models/UsageLimits.py`, `flujo.toml` pricing configuration*

*   **🔭 Production-Grade Observability:** Flujo provides a first-class tracing system that generates a detailed, hierarchical view of every run. It integrates seamlessly with **OpenTelemetry** and includes a powerful command-line inspection tool, **`flujo lens`**, for debugging and analysis.
    *   *Evidence in code: `flujo/tracing/`, `flujo/telemetry/otel_hook.py`, `flujo/cli/lens.py`*

*   **🔄 Stateful, Resumable Pipelines:** With support for SQLite, file, and in-memory backends, Flujo can persist the state of long-running workflows. This enables pipelines to be paused for **Human-in-the-Loop (HITL)** interaction and resumed later.
    *   *Evidence in code: `flujo/state/backends/`, `flujo/domain/dsl/step.py::HumanInTheLoopStep`*

## Quick Start

Get up and running with a simple pipeline in minutes.

```bash
pip install flujo openai pydantic
export OPENAI_API_KEY="sk-..."
```

Create a file `main.py`:

```python
import asyncio
from pydantic import BaseModel
from flujo import Step, make_agent_async

# 1. Define your desired output structure
class Translation(BaseModel):
    original_text: str
    translated_text: str
    language: str

# 2. Create a type-safe agent
translator_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are an expert translator. Translate the user's text into French.",
    output_type=Translation,
)

# 3. Define your workflow as a pipeline of steps
translation_pipeline = Step.solution(
    name="TranslateToFrench",
    agent=translator_agent
)

# 4. Run the pipeline
async def main():
    result = await translation_pipeline.run("Hello, world!")
    
    # The output is a clean, validated Pydantic object
    if result.success:
        translation = result.output
        print(f"Original: '{translation.original_text}'")
        print(f"Language: {translation.language}")
        print(f"Translated: '{translation.translated_text}'")
        print(f"---")
        print(f"Cost: ${result.cost_usd:.6f}")
        print(f"Tokens: {result.token_counts}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it from your terminal:
```bash
python main.py
```

## Key Concepts

*   **`Step`**: The fundamental building block. A `Step` is a declarative configuration object that holds an **agent** (an LLM call or any async function), configuration (retries, timeouts), and optional plugins or validators.
*   **`Pipeline`**: An ordered sequence of `Step`s. You compose them using the `>>` operator (`pipeline = step1 >> step2`). The output of one step becomes the input to the next.
*   **`Context`**: A Pydantic model that acts as a shared state object for the entire pipeline run. Steps can read from and write to the context, allowing for complex state management.

## Building a More Complex Pipeline

Flujo's power comes from composing steps into sophisticated workflows. Here's a multi-step pipeline that uses a shared `Context`.

```python
import asyncio
from pydantic import Field
from flujo import Pipeline, Step, make_agent_async, step
from flujo.domain.models import PipelineContext, Checklist

# 1. Define a custom context to hold state across steps
class WritingContext(PipelineContext):
    topic: str
    requirements: Checklist = Field(default_factory=Checklist)
    draft: str = ""

# 2. Create agents for each task
planner_agent = make_agent_async(
    "openai:gpt-4o-mini",
    "Generate a checklist of key points to cover for the user's topic.",
    Checklist,
)
writer_agent = make_agent_async(
    "openai:gpt-4o",
    "Write a blog post on the given topic, following the provided checklist.",
    str,
)

# 3. Define steps as functions that interact with the context
@step(updates_context=True)
async def plan(topic: str, *, context: WritingContext) -> dict:
    """Generates a checklist and stores it in the context."""
    checklist = await planner_agent.run(topic)
    return {"requirements": checklist}

@step(updates_context=True)
async def write(topic: str, *, context: WritingContext) -> dict:
    """Writes a draft using the topic and checklist from the context."""
    prompt = f"Topic: {context.topic}\nChecklist: {context.requirements.model_dump_json()}"
    draft_text = await writer_agent.run(prompt)
    return {"draft": draft_text}

# 4. Compose the pipeline
blog_pipeline = plan >> write

# 5. Run it with an initial context
async def main():
    # Flujo runner manages the pipeline and its context
    from flujo import Flujo
    
    runner = Flujo(
        pipeline=blog_pipeline,
        context_model=WritingContext,
    )
    
    result = await runner.run_async("The future of AI in software development").__anext__()
    final_context = result.final_pipeline_context

    print("--- FINAL DRAFT ---")
    print(final_context.draft)
    print("\n--- METRICS ---")
    print(f"Total Cost: ${result.total_cost_usd:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## What Makes Flujo Different?

1.  **Self-Healing Outputs:** Flujo's `DeterministicRepairProcessor` and LLM-based repair agent automatically fix malformed JSON from models, saving you from writing endless parsing and validation boilerplate.

2.  **From Cost Center to Governed Resource:** Flujo treats AI costs as a first-class citizen. By defining model pricing in `flujo.toml` and setting `UsageLimits`, you can monitor, control, and govern AI spending across your organization.

3.  **Self-Improving Pipelines:** Flujo can run evaluations on your pipelines, analyze the root causes of failures, and generate an `ImprovementReport` with concrete suggestions for refining your prompts, agents, and configurations.

4.  **Operate with Confidence:** With built-in tracing, OpenTelemetry integration, and the `flujo lens` CLI, you get deep visibility into every step of your AI workflows, making debugging and production monitoring trivial.

## Installation

```bash
pip install flujo
```

For specific integrations, you can install extras:
```bash
pip install flujo[openai,anthropic,prometheus]
```

## Command-Line Interface (CLI)

Flujo includes a powerful CLI for running and inspecting pipelines.

```bash
# Run a pipeline from a file
flujo run my_pipeline.py --input "My initial data"

# Inspect past runs
flujo lens list

# Show a detailed trace of a specific run
flujo lens trace <run_id>

# Validate a pipeline's structure and types
flujo validate my_pipeline.py
```

 
 