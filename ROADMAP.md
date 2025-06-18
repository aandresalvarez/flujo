## Flujo v.Next Roadmap: From Engine to Ecosystem —and Beyond

**Document Owner:** Alvaro A. Alvarez  
**Status:** DRAFT  
**Last Updated:** 2025‑06‑18

---

### Vision Statement

With the core orchestration engine complete, *Flujo* now differentiates on **confidence**, **compliance**, and **continuous improvement**. Our roadmap delivers an unrivalled developer experience, a thriving plugin ecosystem, enterprise‑grade scale, and—through a new innovation phase—capabilities no other AI‑workflow framework offers: compile‑time graph safety, time‑travel debugging, built‑in policy enforcement, self‑healing pipelines, multimodal support, a live graph UI, and zero‑to‑prod DevOps.

---

### Owner Legend

| Tag      |  Role                         |
| -------- | ----------------------------- |
| **DOC**  | Documentation Guild           |
| **DX**   | Developer Experience Guild    |
| **FE**   | Framework Engineering Guild   |
| **INF**  | Infrastructure & DevOps Guild |
| **LEAD** | Product Lead / PM             |

---

## Phase 1 — Developer Onboarding & Polish (*The 1.0 Release Cycle*)

**Goal:** Make Flujo exceptionally easy to learn and adopt. This phase culminates in the **`Flujo 1.0`** GA release.

### Epic 1.1 — Comprehensive Documentation Overhaul

|  #                                                                                                                       |  Story                 |  Description                                                                                                                   |  Owner      |
| ------------------------------------------------------------------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------- |
|  1.1‑1                                                                                                                   | Rewrite the Tutorial   | Craft a narrative tutorial that walks from `pip install flujo` to a wow‑moment **AgenticLoop** demo.                           | **DOC**     |
|  1.1‑2                                                                                                                   | Update Core Concepts   | Refresh **concepts.md** to clarify *Routing vs. Exploration* (`ConditionalStep` vs. `AgenticLoop`).                            | **DOC**     |
|  1.1‑3                                                                                                                   | Enhance API Reference  | Ensure every public class (e.g. `AgenticLoop`, `PipelineContext`, `ExecutionBackend`) has exhaustive, example‑rich docstrings. | **FE / DX** |
|  1.1‑4                                                                                                                   | Add “Patterns” Section | New top‑level docs section housing canonical patterns: Plan‑then‑Execute, Stateful HITL, etc.                                  | **DOC**     |
|  1.1‑5                                                                                                                   | Technical Review       | Framework engineer sweeps every page for accuracy against `main` branch.                                                       | **FE**      |
| **Exit Criteria:** A first‑time user can implement a multi‑turn *stateful* pipeline by following only the official docs. |                        |                                                                                                                                |             |

### Epic 1.2 — Overhaul & Enhance Examples

|  #                                                                                                                                 |  Story                      |  Description                                                                                     |  Owner  |
| ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------ | ------- |
|  1.2‑1                                                                                                                             | Refactor `00_quickstart.py` | Update quick‑start to showcase **AgenticLoop** immediately.                                      | **DX**  |
|  1.2‑2                                                                                                                             | Create New Examples         | Add `10_adaptive_routing.py` (Plan‑then‑Execute) and `11_stateful_hitl.py` (stateful HITL loop). | **DX**  |
|  1.2‑3                                                                                                                             | Jupyter Notebooks           | Convert high‑impact examples into annotated notebooks for Colab / Binder.                        | **DX**  |
|  1.2‑4                                                                                                                             | Improve Readability         | Add rich comments + clear `print()` checkpoints across all examples.                             | **DX**  |
| **Exit Criteria:** `examples/` is a gold‑standard gallery; every script runs with `python example.py` and explains itself clearly. |                             |                                                                                                  |         |

---

## Phase 2 — Ecosystem & Community Growth

**Goal:** Transform Flujo from a library into the hub of a vibrant ecosystem.

### Epic 2.1 — The “Flujo Toolkit” Official Plugin Suite

|  #                                                                                                   |  Story                    |  Description                                                                  |  Owner  |
| ---------------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------- | ------- |
|  2.1‑1                                                                                               | `flujo‑pii‑redactor`      | Plugin that redacts PII (emails, phones) via regex + optional LLM refinement. | **FE**  |
|  2.1‑2                                                                                               | `flujo‑vector‑db`         | Adapters for Weaviate & Pinecone, enabling RAG pipelines in 3 lines.          | **FE**  |
|  2.1‑3                                                                                               | Update `AWESOME‑FLUJO.md` | Curate official + community plugins; launch early adopter outreach.           | **DX**  |
| **Exit Criteria:** Devs can build a compliant RAG pipeline using only core Flujo & official plugins. |                           |                                                                               |         |

### Epic 2.2 — Launch & Community Engagement

|  #                                                                                            |  Story            |  Description                                                     |  Owner        |
| --------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------- | ------------- |
|  2.2‑1                                                                                        | 1.0 Launch Blog   | Publish “Announcing Flujo 1.0” across Medium, Dev.to, X/Twitter. | **DX / LEAD** |
|  2.2‑2                                                                                        | Video Tutorials   | Produce three 2‑5 min screencasts for key recipes.               | **DX**        |
|  2.2‑3                                                                                        | Community Channel | Stand‑up Discord server + GitHub Discussions; seed with FAQs.    | **DX**        |
| **Exit Criteria:** `1.0.0` published on PyPI, blog live, Discord active with ≥50 early users. |                   |                                                                  |               |

---

## Phase 3 — Scaling the Architecture & Enterprise Focus

**Goal:** Prove Flujo at cloud scale and meet enterprise governance needs.

### Epic 3.1 — Distributed Backend Reference Implementation

|  #                                                                                             |  Story                 |  Description                                                                                  |  Owner      |
| ---------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- | ----------- |
|  3.1‑1                                                                                         | Serialization Strategy | Design secure schema for `StepExecutionRequest` + results, including agent registry look‑ups. | **FE**      |
|  3.1‑2                                                                                         | `flujo‑lambda‑backend` | OSS repo + SAM/CDK templates for AWS Lambda execution.                                        | **INF**     |
|  3.1‑3                                                                                         | Distributed Demo       | Example `AgenticLoop` where each agent invocation is a separate Lambda.                       | **FE / DX** |
| **Exit Criteria:** Users can deploy a fully distributed pipeline to AWS Lambda in <15 minutes. |                        |                                                                                               |             |

### Epic 3.2 — Advanced Enterprise Governance

|  #                                                                                             |  Story               |  Description                                                                                        |  Owner  |
| ---------------------------------------------------------------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------- | ------- |
|  3.2‑1                                                                                         | RBAC for Agents      | Extend `agent_registry` to include permission scopes; unauthorised calls raise compile‑time errors. | **FE**  |
|  3.2‑2                                                                                         | Time‑Based Budgets   | Enforce cost ceilings per hour/day; auto‑pause pipelines on breach.                                 | **FE**  |
|  3.2‑3                                                                                         | Telemetry Dashboards | Grafana JSONs for latency, cost, token usage, and policy events.                                    | **INF** |
| **Exit Criteria:** Admins can restrict expensive agents and observe real‑time cost dashboards. |                      |                                                                                                     |         |

---

## Phase 4 — Differentiator & Innovation Leap

**Goal:** Deliver unique capabilities that put Flujo in a league of its own.

### Epic 4.1 — Static‑Analysis Type Checker

|  #                                                            |  Story              |  Description                                                            |  Owner  |
| ------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------- | ------- |
|  4.1‑1                                                        | Graph Type Spec     | Formal contract + error taxonomy for `Step[A,B]` edges.                 | **FE**  |
|  4.1‑2                                                        | MyPy/Pyright Plugin | Emit graph‑level errors (unreachable node, type mismatch) at lint time. | **FE**  |
|  4.1‑3                                                        | IDE Patches         | VS Code extension suggests auto‑fixes during refactors.                 | **DX**  |
| **Exit Criteria:** CI fails on invalid graphs before runtime. |                     |                                                                         |         |

### Epic 4.2 — Flight‑Recorder Time‑Travel Debugger

|  #                                                                                      |  Story               |  Description                                                               |  Owner  |
| --------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------- | ------- |
|  4.2‑1                                                                                  | Low‑Overhead Capture | Stream every token + state to a binary log.                                | **FE**  |
|  4.2‑2                                                                                  | Replay CLI           | `flujodebug replay <run‑id>` steps forward/backward with state inspection. | **DX**  |
|  4.2‑3                                                                                  | Hot‑Swap Step        | CLI flag to swap a Step impl and continue replay.                          | **DX**  |
| **Exit Criteria:** Developers can deterministically replay and edit any production run. |                      |                                                                            |         |

### Epic 4.3 — Policy & Compliance DSL

|  #                                                                               |  Story                |  Description                                          |  Owner  |
| -------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------- | ------- |
|  4.3‑1                                                                           | DSL Engine            | YAML/JSON schema for declaring PHI/PII rulesets.      | **FE**  |
|  4.3‑2                                                                           | Pre‑built Packs       | Ship HIPAA & GDPR policy bundles.                     | **FE**  |
|  4.3‑3                                                                           | Telemetry Integration | Violations surface as OTel spans and block execution. | **INF** |
| **Exit Criteria:** Pipelines fail fast on policy violations without manual code. |                       |                                                       |         |

### Epic 4.4 — Auto‑Eval → Auto‑Patch Loop

|  #                                                                       |  Story            |  Description                                                |  Owner  |
| ------------------------------------------------------------------------ | ----------------- | ----------------------------------------------------------- | ------- |
|  4.4‑1                                                                   | Critique → Diff   | Convert LLM critique into structured Git patches.           | **FE**  |
|  4.4‑2                                                                   | GitHub Bot        | Open PRs, assign reviewers, rerun CI on merge.              | **DX**  |
|  4.4‑3                                                                   | Self‑Healing Demo | Showcase prompt that self‑patches after hallucination eval. | **DX**  |
| **Exit Criteria:** A failing eval triggers a PR that passes once merged. |                   |                                                             |         |

### Epic 4.5 — Multimodal & Multi‑Runtime Support

|  #                                                                                                    |  Story              |  Description                                              |  Owner  |
| ----------------------------------------------------------------------------------------------------- | ------------------- | --------------------------------------------------------- | ------- |
|  4.5‑1                                                                                                | `MediaStep` Base    | Typed steps for images, audio, tables.                    | **FE**  |
|  4.5‑2                                                                                                | Runtime Abstraction | Transparent dispatch to CPU, GPU, or serverless.          | **INF** |
|  4.5‑3                                                                                                | Cross‑Modal Example | Vision‑language loop: caption image, answer follow‑up Qs. | **DX**  |
| **Exit Criteria:** Users mix text & vision models in one DAG and deploy to Lambda *or* K8s unchanged. |                     |                                                           |         |

### Epic 4.6 — Live Graph UI + IntelliSense

|  #                                                                                                                             |  Story                 |  Description                                                  |  Owner      |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------- | ----------- |
|  4.6‑1                                                                                                                         | Web Canvas MVP         | Visual pipeline editor; drag‑and‑drop nodes sync to DSL file. | **FE / DX** |
|  4.6‑2                                                                                                                         | Type‑Aware Suggestions | Palette shows only type‑compatible Steps.                     | **FE**      |
|  4.6‑3                                                                                                                         | Round‑Trip Fidelity    | Code ↔ UI updates remain perfectly in sync.                   | **DX**      |
| **Exit Criteria:** A valid pipeline can be authored end‑to‑end without writing code, yet generates clean DSL committed to git. |                        |                                                               |             |

### Epic 4.7 — Opinionated Ops Stack

|  #                                                                                                                        |  Story               |  Description                                                   |  Owner  |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------- | ------- |
|  4.7‑1                                                                                                                    | `flujodeploy init`   | Scaffold Helm/Kustomize charts, CI/CD, canary configs.         | **INF** |
|  4.7‑2                                                                                                                    | Autoscaling Policies | Step‑level concurrency knobs + cost alerts shipped by default. | **INF** |
|  4.7‑3                                                                                                                    | Grafana Dashboards   | Pre‑wired boards for latency, tokens, cost, policy events.     | **INF** |
| **Exit Criteria:** A new project goes from `git clone` to production deployment in <30 min on any CNCF‑compliant cluster. |                      |                                                                |         |

---

### Roadmap Summary

*Phases 1–3* make Flujo delightful, documented, scalable, and enterprise‑ready. *Phase 4* positions Flujo as the **only** framework that is statically safe, debuggable in time, policy‑aware, self‑healing, and multimodal—with a turnkey ops story to match.

| # | Story | Description | Owner |
\|---
