### Functional Specification Document (FSD) — **Completion Road-map v1.0**

**Project:** `pydantic-ai-orchestrator`  **Last code drop:** 11 Jun 2025
**Audience:** Core maintainers, QA, Docs, DevOps, Security reviewers.
**Goal of this FSD:** Identify remaining gaps in the current codebase and prescribe the exact functional, non-functional and procedural work needed to ship a *stable, production-ready* `v1.0.0`.

---

## 1 · Executive Summary

The project is now \~80 % feature-complete.
Major components (typed settings, async orchestrator, CLI, unit/integration/E2E tests, docs) exist but several **critical gaps** remain:

| Area                   | Gaps (high-level)                                                                                                                             |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Correctness**        | Type errors, dangling imports, stubbed reward scorer, missing validation ↔ prompt glue, reflections factory misuse, checklist I/O mismatches. |
| **Resilience**         | Tenacity retry not wired, graceful `KeyboardInterrupt`, timeout propagation, telemetry double-init guard.                                     |
| **Security & secrets** | Keys printed in tracebacks, `.env.example` incomplete, no secret redaction in logs.                                                           |
| **Packaging & CI**     | `pyproject.toml` incomplete, wheels not built, version mismatches in `__init__`, coverage badge, release workflow un-tested.                  |
| **Docs**               | API reference auto-generation missing; README still minimal; contribution guidelines omit branch naming + conventional commits.               |
| **Testing**            | No *happy-path* sync integration test; E2E cassette asserts wrong type; reward-model scorer un-mocked; coverage at 74 % (< goal 90 %).        |

This FSD turns those findings into granular work items.

---

## 2 · Scope

### 2.1 In-Scope

* Feature completion for `v1.0.0`.
* Python 3.11 & 3.12 official support.
* OpenAI and local-LLM runtime parity (mock agent interface for offline).

### 2.2 Out-of-Scope

* UI front-end (Gradio) beyond CLI.
* External graph orchestration (LangGraph etc.).
* Persistent DB/MQ back-end.

---

## 3 · Functional Requirements

### 3.1 Domain & Scoring

| Ref  | Requirement                                                                                                                                                            | Acceptance Test                                                                                                      |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| F-D1 | **Checklist ↔ Validator contract** — `validator_agent` must accept `{"solution": str, "checklist": Checklist}` and **return** a *Checklist* with every item annotated. | Integration test asserts that mocked validator receives dict keys and that orchestrator handles KeyError gracefully. |
| F-D2 | **Weighted scoring** — orchestrator must allow user-supplied weights via `Task.metadata["weights"]` *or* CLI flag `--weights path.json`.                               | Unit test: given weights file, score matches formula.                                                                |
| F-D3 | **Reward scorer** — supply working stub that calls mini model *or* raise clear `FeatureDisabled` when `ORCH_REWARD=0`.                                                 | Unit test toggles env var and checks for fallback behaviour.                                                         |

### 3.2 Application / Orchestrator

| Ref  | Requirement                                                                                                                       | Acceptance Test                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| F-A1 | **Tenacity retry** applied to *every* agent call, with exponential back-off and jitter; raise `OrchestratorRetryError` after max. | Unit test patches agent to raise once then succeed. |
| F-A2 | **Timeout propagation** — if one async task hits `timeout`, cancel siblings, retry round.                                         | Integration test with `asyncio.sleep` stub.         |
| F-A3 | **Reflection memory cap** configurable (`ORCH_REFLEXION_LIMIT`, default = 3).                                                     | Existing integration test updated.                  |
| F-A4 | **Sync public API** — add `Orchestrator.run_sync(prompt:str) -> Candidate` for parity.                                            | Happy-path sync integration test.                   |
| F-A5 | **Graceful SIGINT/KeyboardInterrupt** — CLI abort leaves partial spans closed.                                                    | Manual smoke test.                                  |

### 3.3 Infrastructure

| Ref  | Requirement                                                                                            | Acceptance Test                          |
| ---- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| F-I1 | **Secret redaction** — OpenAI/Logfire keys must never reach console/Logfire.                           | Grep build logs in CI for secret prefix. |
| F-I2 | **Agent factory** must expose `make_agent_async()` returning pre-wrapped async agents to cut overhead. | Profile: first call < 100 ms overhead.   |

### 3.4 CLI

| Ref  | Requirement                                                                          | Acceptance Test           |
| ---- | ------------------------------------------------------------------------------------ | ------------------------- |
| F-C1 | `orch solve` **–weights** param accepts JSON or YAML file for weighted scoring.      | CLI functional test.      |
| F-C2 | `orch bench` prints percentile latencies (p50/p95) and uses `rich` table formatting. | Visual check CI artifact. |
| F-C3 | Return code 2 on validation errors (bad env, missing key).                           | Shell test.               |

---

## 4 · Non-Functional Requirements

| Category          | Target                                                          |
| ----------------- | --------------------------------------------------------------- |
| **Coverage**      | ≥ 90 % lines, ≥ 85 % branches.                                  |
| **Latency**       | ≤ 400 ms orchestrator overhead for `k=1`.                       |
| **Thread-safety** | Multiple `Orchestrator()` instances safe in asyncio event loop. |
| **Security**      | No secrets in core dumps; dependencies `pip-audit` clean.       |
| **Docs**          | MkDocs site passes build with API auto-docs.                    |
| **Packaging**     | `pip install pydantic_ai_orchestrator` downloads wheel < 2 MB.  |

---

## 5 · Detailed Work Items

### 5.1 Code Corrections

1. **Fix import:** `from ..infra.agents import Agent` inside orchestrator — there is no `Agent` class in that module; change to `from pydantic_ai import Agent as _Agent` or remove.
2. **`Candidate` model** currently lacks `checklist` field referenced in orchestrator — add optional attribute.
3. **`weighted_score`** uses `check.metadata` but receives `Checklist`; add helper to orchestrator to pass weights explicitly.
4. **RewardScorer.run\_sync** returns `AgentResult`, not float — wrap `.output`.
5. Update all tests to import `Orchestrator.run_sync` (exposed alias).

### 5.2 Enhancements

* **`infra/telemetry.py`** — add OTLP exporter env toggle, catch duplicate init.
* **`infra/settings.py`** — include `ORCH_REFLEXION_LIMIT`, `ORCH_REWARD`.
* **`infra/agents.py`** — move retry logic outside `Agent` kwargs (PydanticAI already has built-in retries).
* **Add** `exceptions.py` in root: `SettingsError`, `OrchestratorRetryError`, `RewardModelUnavailable`.
* **Add** `utils/redact.py` for secret masking.

### 5.3 Docs & Examples

* Expand `docs/usage.md` with weighted scoring example.
* Auto-generate API docs via `mkdocstrings` for key modules.
* Provide *quickstart colab* in `/examples`.

### 5.4 Testing

* **New unit tests:** `test_reward_scorer_score`, `test_ratio_vs_weighted_parametrised`.
* **Sync happy-path integration test** (`test_orchestrator_sync.py`).
* **Fault-injection test** for retry and timeout.
* Update `e2e/test_golden_transcript.py` to expect `Candidate` json, not raw string.

### 5.5 CI/CD

* Finish `.github/workflows/ci.yml`:

  * Steps: cache pip → install → lint (ruff) → mypy → tests → build wheel.
* Create `release.yml` for PyPI; sign wheels with trusted-publisher key.
* Publish coverage badge to README via Codecov.

---

## 6 · Timeline & Milestones

| Week    | Deliverable                                       |
| ------- | ------------------------------------------------- |
| **W-0** | Merge this FSD ➜ `docs/FSD_v1.md`.                |
| **W-1** | Code corrections (§5.1) + unit tests pass.        |
| **W-2** | Enhancements & CLI flags (§5.2, §5.3).            |
| **W-3** | All new tests green, coverage ≥ 90 %.             |
| **W-4** | Docs site built, CI pipeline green.               |
| **W-5** | Tag `v1.0.0-rc1`, PyPI dry-run, internal dogfood. |
| **W-6** | GA release `v1.0.0`, publish blog post.           |

---

## 7 · Risk Register

| Risk                       | Impact | Mitigation                                          |
| -------------------------- | ------ | --------------------------------------------------- |
| OpenAI model version drift | High   | Pin `gpt-4o-mini-2025-04-09`; nightly canary.       |
| Test token cost explosion  | Med    | Use `PydanticAI-dry-run=true` env; cassette re-use. |
| Secret leakage in logs     | High   | implement `utils/redact` + CI grep.                 |

---

## 8 · Glossary

* **Agent** – `pydantic_ai.Agent` object.
* **Checklist** – Pydantic model describing evaluation criteria.
* **Candidate** – Proposed solution plus evaluation metadata.
* **Reflection** – LLM-generated introspection string.

---

### ✅ Approval & Next Steps

*Review this FSD in the next project meeting; upon approval create GitHub issues labelled `v1.0` matching §5 work items.  Once all issues close and CI is green, cut the `v1.0.0` release.* 