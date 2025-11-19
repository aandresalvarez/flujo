<div align="center">
  <a href="https://github.com/aandresalvarez/flujo">
    <img src="https://raw.githubusercontent.com/aandresalvarez/flujo/main/assets/flujo.png" alt="Flujo logo" width="180"/>
  </a> 
  
  <h1>Flujo — Your Conversational AI Workflow Server</h1>
  
  <p>
    <b>Go from a simple idea to a production-grade, auditable AI pipeline in a single conversation.</b>
  </p>

| CI/CD | PyPI | Docs | License |
| :---: | :---: | :---: | :---: |
| [![CI status](https://github.com/aandresalvarez/flujo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aandresalvarez/flujo/actions/workflows/ci.yml) | [![PyPI version](https://img.shields.io/pypi/v/flujo.svg)](https://pypi.org/project/flujo/) | [![Documentation Status](https://readthedocs.org/projects/flujo/badge/?version=latest)](https://flujo.readthedocs.io/en/latest/?badge=latest) | [![License](https://img.shields.io/pypi/l/flujo.svg)](https://github.com/aandresalvarez/flujo/blob/main/LICENSE) |

</div>
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

#### **Step 2: Tell Flujo Your Goal**

Start a conversation with the Flujo Architect from inside your project:

```bash
flujo create --goal "Summarize a web article, translate it to Spanish, and post to Slack."
```
> **Flujo Architect:** `Understood. To post to Slack, I have a 'post_to_slack' tool. Which channel should I use?`

#### **Step 2: Clarify and Confirm**

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
| **Auditability as a Contract** | Every execution produces a **formal, structured trace**. This isn't just logging; it's a deterministic ledger that enables 100% faithful replay, making bugs transparent and easy to fix. |
| **Extensibility via Skills** | Add new capabilities (Python functions, API clients) to a central **Skill Registry**. The Architect Agent can discover and intelligently wire these skills into the pipelines it generates, allowing you to safely grant AI new powers. |

---

## For Developers: The Power Under the Hood

While the CLI provides a no-code experience, Flujo offers a powerful, type-safe Python DSL for developers who need full control.

**Example: A Simple Translation Agent & Pipeline**
```python
# translate_pipeline.py
from pydantic import BaseModel
from flujo import Step, Pipeline, make_agent_async

class Translation(BaseModel):
    original_text: str
    translated_text: str
    language: str

# 1. Define an agent with a structured, Pydantic-validated output
translator_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="Translate the user's text into French.",
    output_type=Translation,
)

# 2. Compose your pipeline with the `>>` operator
pipeline = Step(name="TranslateToFrench", agent=translator_agent)
```
Your Python-defined pipelines get all the same benefits: automatic CLI generation, budget enforcement, and full traceability.

---

## Installation & Getting Started

**Install Flujo:**
```bash
pip install flujo
```

**Install with Extras (e.g., for specific LLM providers):**
```bash
pip install flujo[openai,anthropic,prometheus]
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
- `dev`: 🛠️ Access advanced developer and diagnostic tools.
  - `validate`, `show-steps`, `visualize`, `compile-yaml`, `show-config`, `version`
  - `experimental`: advanced tools like `solve`, `bench`, `add-case`, `improve`

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

- Templates (init/demo): default to `state_uri = "memory://"` so projects don’t persist state unless you opt in.
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

## License

Flujo is available under a dual-license model:

*   **AGPL-3.0:** For open-source projects and non-commercial use, Flujo is licensed under the AGPL-3.0. See the [`LICENSE`](LICENSE) file for details.
*   **Commercial License:** For commercial use in proprietary applications, a separate commercial license is required. Please contact [Your Contact Email/Website] for more information.
# Test comment for pre-commit hook verification
