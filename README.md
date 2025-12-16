 
<div align="center">
  <a href="https://github.com/aandresalvarez/flujo">
    <img src="https://raw.githubusercontent.com/aandresalvarez/flujo/main/assets/flujo.png" alt="Flujo logo" width="180"/>
  </a> 
  
  <h1>Flujo — The Type-Safe AI Workflow Server</h1>
  
  <p>
    <b>Orchestrate AI Agents with Confidence. From local script to production cluster.</b>
  </p>

| CI/CD | PyPI | Docs | License |
| :---: | :---: | :---: | :---: |
| [![CI status](https://github.com/aandresalvarez/flujo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aandresalvarez/flujo/actions/workflows/ci.yml) | [![PyPI version](https://img.shields.io/pypi/v/flujo.svg)](https://pypi.org/project/flujo/) | [![Documentation Status](https://readthedocs.org/projects/flujo/badge/?version=latest)](https://flujo.readthedocs.io/en/latest/?badge=latest) | [![License](https://img.shields.io/pypi/l/flujo.svg)](https://github.com/aandresalvarez/flujo/blob/main/LICENSE) |

</div>
---

Flujo is a framework for building, observing, and deploying AI agent workflows. It bridges the gap between simple Python scripts and complex enterprise orchestration, giving you **retries**, **durable state persistence**, and **human-in-the-loop** capabilities out of the box. See `docs/context_strict_mode.md` for strict context rules, `docs/adapter_allowlist.md` for the adapter allowlist required in strict DSL mode, and `docs/type_safety_observability.md` for the CI gates/metrics that keep type-safety strict by default.

Designed for high-stakes, restricted environments (like healthcare), Flujo delivers **Temporal-like resilience and auditability** within a single, deployable Python process, eliminating the need for complex external infrastructure.

## ✨ Key Features

*   **🧠 The Architect:** A built-in AI agent that generates complete, valid pipeline code from natural language goals.

*   **💾 Durable & Secure State:** Every step is persisted to SQLite (local) or Postgres (production) with byte-level serialization for cryptographic audit trails. Pause, resume, and replay workflows across server restarts with total determinism.

*   **🔀 Advanced Control Flow:** Native support for Loops, Conditionals (If/Else), Parallel execution, and Map/Reduce.

*   **👤 Human-in-the-Loop:** Pause execution for user approval or input, then resume exactly where you left off.

*   **🔍 Flujo Lens:** A powerful CLI TUI to debug traces, inspect step history, and analyze costs.

*   **⚡ Serverless Ready:** Lightweight architecture optimized for Google Cloud Run and AWS Lambda.

---

## The Flujo Experience: Idea to Production in 3 Commands

Imagine you need to automate a task: "Summarize a web article, translate it to Spanish, and post it to our company's Slack." With traditional tools, this is hours of coding, testing, and deploying.

**With Flujo, it's a conversation.**

#### **Step 1: Initialize a Project**

Create and enter a new project directory, then scaffold it:

```bash
mkdir weekly-bot && cd weekly-bot
flujo init
```

Re-initialize an existing project (overwriting templates):

```bash
# Prompt for confirmation
flujo init --force

# Non-interactive (CI/scripts):
flujo init --force --yes
```

#### **Step 2: Create Your Pipeline**

Start a conversation with the Flujo Architect from inside your project:

```bash
flujo create --goal "Summarize a web article, translate it to Spanish, and post to Slack."
```
> **Flujo Architect:** `Understood. To post to Slack, I have a 'post_to_slack' tool. Which channel should I use?`

Provide the missing details. The Architect confirms the plan.

```bash
> #daily_news_es
```
> **Flujo Architect:** `Great. I've designed a 3-step pipeline: FetchArticle → SummarizeAndTranslate → PostToSlack. I've generated pipeline.yaml for you. It is ready to run.`

In seconds, Flujo has generated a complete, secure, and production-ready YAML blueprint. No code written. No complex configuration.

For details on the new programmatic Architect that powers `flujo create`, see:
- `flujo/architect/README.md` (usage, states, extension points)

#### **Step 3: Run and Inspect**

Execute your new pipeline. Flujo handles the orchestration, cost tracking, and logging automatically.

```bash
flujo run --input "https://flujo-ai.dev/blog/some-article"
```
Every run is saved. If something goes wrong, you have a complete, replayable trace.
```bash
# Get a visual trace of the last run to see exactly what happened
flujo lens trace <run_id>

# Replay a failed production run locally for perfect debugging
flujo lens replay <run_id>
```

**This is the core of Flujo:** a framework that uses AI to build AI, guided by you, and governed by production-ready safety rails.

---

## What Makes This Possible?

Flujo is not just a scripting library; it's a complete application server for AI workflows, built on a few core principles:

| Principle | How Flujo Delivers |
| :--- | :--- |
| **Declarative Blueprints** | Your entire workflow—agents, prompts, tools, and logic (`parallel`, `loops`)—is defined in a single, human-readable **YAML file**. This is the source of truth that the Architect Agent generates and the Runner executes. |
| **Safety by Design** | The framework is built around **proactive Quotas** and **centralized Budgets**. A pipeline cannot start if it might exceed its budget, and parallel steps can't create race conditions that lead to overspending. |
| **Auditability as a Contract** | Every execution produces a **formal, structured trace**. This uses **byte-level serialization** (Blake3/Orjson) to create a deterministic ledger that enables 100% faithful replay, making bugs transparent and easy to fix, critical for compliance (HIPAA/GDPR). |
| **Embedded Resilience** | Flujo uses **custom memory pooling** to ensure predictable memory usage and prevent data bleed between runs, making it safe for long-running processes in restricted environments. |
| **Extensibility via Skills** | Add new capabilities (Python functions, API clients) to a central **Skill Registry**. The Architect Agent can discover and intelligently wire these skills into the pipelines it generates, allowing you to safely grant AI new powers. |

---

## 🛠️ Python API

For developers who prefer code over configuration, Flujo offers a fluent, type-safe Python DSL.

```python
import asyncio
from pydantic import BaseModel
from flujo import Step, Pipeline, Flujo
from flujo.agents import make_agent_async

# 1. Define Type-Safe Outputs
class Analysis(BaseModel):
    topic: str
    summary: str
    sentiment_score: float

# 2. Create Agents
researcher = make_agent_async("openai:gpt-4o", "You are a researcher.", str)
analyst = make_agent_async("openai:gpt-4o", "Analyze the text.", Analysis)

# 3. Define Steps
step_1 = Step(name="research", agent=researcher)
step_2 = Step(name="analyze", agent=analyst, input="{{ previous_step }}")

# 4. Compose Pipeline
pipeline = step_1 >> step_2

# 5. Run with State Persistence
async def main():
    runner = Flujo(pipeline)
    result = await runner.run_async("The future of Quantum Computing")
    print(result.output)  # Returns a validated Analysis object

if __name__ == "__main__":
    asyncio.run(main())
```

Your Python-defined pipelines get all the same benefits: automatic CLI generation, budget enforcement, and full traceability.

### 🔄 Granular Execution (Resumable Agents)

For long-running, multi-turn agent conversations that need crash-safe persistence:

```python
from flujo import Step, Flujo
from flujo.agents import make_agent_async

# Create a research agent
agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are a research assistant. Respond with 'COMPLETE' when done.",
    output_type=str,
)

# Wrap in granular execution - survives crashes!
pipeline = Step.granular("research_agent", agent, max_turns=20)

async def main():
    runner = Flujo(pipeline)
    async for result in runner.run_async("Research quantum computing"):
        print(result.output)
```

**Key Benefits:**
- ✅ **Crash-safe**: Resume after server restart without losing progress
- ✅ **No double-billing**: Completed turns are skipped on resume
- ✅ **Fingerprint validation**: Detects config changes between runs
- ✅ **Idempotency keys**: Safe external API retries

See the full guide: [docs/guides/granular_execution.md](docs/guides/granular_execution.md)

---

## 🧩 The Blueprint (YAML)

Pipelines can also be defined in YAML, making them language-agnostic, version-controllable, and editable by the Architect agent.

```yaml
version: "0.1"
name: "code_review_pipeline"

steps:
  - kind: step
    name: review_code
    agent: { id: "agents.senior_dev" }
    input: "{{ initial_prompt }}"
  
  - kind: conditional
    name: check_severity
    condition: "{{ previous_step.severity == 'high' }}"
    branches:
      true:
        - kind: hitl
          message: "High severity issue detected. Approve fix?"
          sink_to: "user_approval"
      false:
        - kind: step
          name: auto_merge
          agent: { id: "flujo.builtins.stringify" }
```

---

## Installation & Getting Started

**Install Flujo:**
```bash
pip install flujo
```

**Install with Extras (e.g., for specific LLM providers):**
```bash
pip install flujo[openai,anthropic,prometheus,postgres]
```

**Configure your API Keys:**
```bash
export OPENAI_API_KEY="sk-..."
```

For full guides, tutorials, and API references, please see our **[Official Documentation](https://flujo.readthedocs.io/)**.

Looking to use GPT‑5 with the Architect? See the guide: `docs/guides/gpt5_architect.md`.

---

## CLI Overview

- `init`: ✨ Initialize a new Flujo workflow project in this directory.
- `create`: 🤖 Start a conversation with the AI Architect to build your workflow.
- `run`: 🚀 Run the workflow in the current project.
- `lens`: 🔍 Inspect, debug, and trace past workflow runs.
  - `lens trace <run_id>` now shows prompt injection events per step (redacted preview). Use this to inspect how conversational history was rendered.

### 🔍 Observability with Lens

Flujo records every execution step, token usage, and cost. Inspect it all via the CLI.

```bash
# List recent runs
flujo lens list

# Visualize the execution tree of a specific run
flujo lens trace <run_id>

# View detailed inputs/outputs for debugging
flujo lens show <run_id> --verbose

# Replay a failed production run locally for perfect debugging
flujo lens replay <run_id>
```
- `dev`: 🛠️ Access advanced developer and diagnostic tools.
  - `validate`, `explain`, `visualize`, `compile-yaml`, `show-config`, `version`

## 🤝 Middleware & Observability API

Need to integrate Flujo with review dashboards or connector services? Use the `TaskClient` facade to interact with running workflows programmatically.

```python
from flujo.client import TaskClient

client = TaskClient()

# Resume a workflow waiting for Human Input
await client.resume_task(
    run_id="run_12345", 
    input_data="Approved"
)

# List paused runs
paused_tasks = await client.list_tasks(status="paused")

# Inspect HITL prompts
task_detail = await client.get_task("run_12345")
print(task_detail.hitl_prompt)
```

The `TaskClient` (`flujo.client.TaskClient`) lets you list paused runs, inspect HITL prompts, resume workflows, or store global watermarks without touching the database schema.  
See [docs/guides/building_middleware.md](docs/guides/building_middleware.md) for more examples.

### CLI Flags & Exit Codes (Quick Reference)

- Global flags:
  - `--project PATH`: Set project root and inject into `PYTHONPATH` (imports like `skills.*`).
  - `-v/--verbose`, `--trace`: Show full tracebacks.
- `validate`:
  - Strict-by-default (`--no-strict` to relax), `--format=json` for CI parsers.
- `run`:
  - `--dry-run` validates without executing (with `--json`, prints steps).
- Stable exit codes: `0` OK, `1` runtime, `2` config, `3` import, `4` validation failed, `130` SIGINT.

See the detailed reference: `docs/reference/cli.md`.

---

## CLI Input Piping (Non‑Interactive Usage)

Flujo supports standard Unix piping and env-based input for `flujo run`.

Input resolution precedence:
1) `--input VALUE` (if `VALUE` is `-`, read from stdin)
2) `FLUJO_INPUT` environment variable
3) Piped stdin (non‑TTY)
4) Empty string fallback

Examples:
```bash
# Pipe goal via stdin
echo "Summarize this" | uv run flujo run

# Read stdin explicitly via '-'
uv run flujo run --input - < input.txt

# Use environment variable
FLUJO_INPUT='Translate this to Spanish' uv run flujo run

# Run a specific pipeline file
printf 'hello' | uv run flujo run path/to/pipeline.yaml
```

---

## Conversational Loops (Zero‑Boilerplate)

Enable iterative, state‑aware conversations in loops using an opt‑in flag. Flujo automatically captures turns, injects conversation history into prompts, and surfaces a sanitized preview in `lens trace`.

Quick start:
```yaml
- kind: loop
  name: clarify
  loop:
    conversation: true
    history_management:
      strategy: truncate_tokens
      max_tokens: 4096
    body:
      - kind: step
        name: clarify
```

Advanced controls:
- `ai_turn_source`: `last` (default) | `all_agents` | `named_steps`
- `user_turn_sources`: include `'hitl'` and/or step names (e.g., `['hitl','ask_user']`)
- `history_template`: custom rendering

Use the `--wizard` flags to scaffold conversational loops with presets:
```bash
uv run flujo create \
  --wizard \
  --wizard-pattern loop \
  --wizard-conversation \
  --wizard-ai-turn-source all_agents \
  --wizard-user-turn-sources hitl,clarify \
  --wizard-history-strategy truncate_tokens \
  --wizard-history-max-tokens 4096
```

See `docs/conversational_loops.md` for details.

These semantics are implemented in the CLI layer only; policies and domain logic must not read from stdin or environment directly.

---

## Architect Pipeline Toggles

Control how the Architect pipeline is built (state machine vs. minimal) using environment variables:

- FLUJO_ARCHITECT_STATE_MACHINE=1: Force the full state-machine Architect.
- FLUJO_ARCHITECT_IGNORE_CONFIG=1: Ignore project config and use the minimal single-step generator.
- FLUJO_TEST_MODE=1: Test mode; behaves like ignore-config to keep unit tests deterministic.

Precedence: FLUJO_ARCHITECT_STATE_MACHINE → FLUJO_ARCHITECT_IGNORE_CONFIG/FLUJO_TEST_MODE → flujo.toml ([architect].state_machine_default) → minimal default.

---

## State Backend Configuration

Flujo persists workflow state (for traceability, resume, and lens tooling) via a pluggable state backend.

- Templates (init/demo): default to `state_uri = "sqlite:///.flujo/state.db"` (relative to project root) for reliable pause/resume and history.
- Core default when not using a project template: SQLite at `sqlite:///flujo_ops.db` (created in CWD) or as configured in `flujo.toml`.
- Ephemeral (in-memory): set one of the following to avoid any persistent files (handy for demos or CI):
  - In `flujo.toml`: `state_uri = "memory://"`
  - Env var: `FLUJO_STATE_URI=memory://`
  - Env var: `FLUJO_STATE_MODE=memory` or `FLUJO_STATE_MODE=ephemeral`
  - Env var: `FLUJO_EPHEMERAL_STATE=1|true|yes|on`

Examples:
```bash
# One-off ephemeral run
FLUJO_STATE_URI=memory:// flujo create --goal "Build a pipeline"

# Project-wide (recommended for demos)
echo 'state_uri = "memory://"' >> flujo.toml
```

When using persistent SQLite, ensure the containing directory exists and is writable (see `flujo/cli/config.py` for path normalization and validation).

---

## 📦 Deployment & Scale

Flujo uses a **"Stateless Worker, External Brain"** architecture.

1.  **Local Dev:** Uses SQLite (`.flujo/state.db`) for zero-setup persistence.

2.  **Production:** Switch to Postgres by setting `state_uri` in `flujo.toml`.

3.  **Scale:** Deploy to **Google Cloud Run** or **AWS Lambda**. Since state is external, you can scale workers to zero or infinity instantly.

```toml
# flujo.toml
state_uri = "postgresql://user:pass@db-host:5432/flujo_db"

[settings]
test_mode = false
# Optional: enable Memory (RAG) indexing
memory_indexing_enabled = true
memory_embedding_model = "openai:text-embedding-3-small"

# Optional: governance policy (module path: pkg.mod:Class)
governance_policy_module = "my_project.policies:MyPolicy"

# Optional: sandboxed code execution provider
[settings.sandbox]
mode = "docker"  # "null" | "remote" | "docker"
docker_image = "python:3.13-slim"
docker_pull = true

# Optional: shadow evaluations (LLM-as-judge)
# Note: shadow eval is experimental and currently defaults to disabled unless enabled programmatically.

# Docker sandbox dependency:
# pip install "flujo[docker]"

# Example governance policy
# examples/governance_policy.py
# governance_policy_module = "examples.governance_policy:DenyIfContainsSecret"
```

This architecture ensures that:
- Workers are stateless and can be killed/restarted without losing progress
- State is centralized in a durable database (SQLite for dev, Postgres for prod)
- Multiple workers can process different runs concurrently
- Failed runs can be resumed from any worker

---

## License

Flujo is available under a dual-license model:

*   **AGPL-3.0:** For open-source projects and non-commercial use, Flujo is licensed under the AGPL-3.0. See the [`LICENSE`](LICENSE) file for details.
*   **Commercial License:** For commercial use in proprietary applications, a separate commercial license is required. Please contact [Your Contact Email/Website] for more information.
 
