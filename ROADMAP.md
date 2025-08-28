 
### **The Guiding Principle: The User-Value Flywheel**

Instead of building all roadmap features top-down, we will use a flywheel approach. Each phase validates a core user journey. The feedback and friction from that phase directly inform what we build next, ensuring every new feature is pulled by a real need, not pushed by a roadmap.

**`Generate -> Run & Debug -> Refine & Govern -> Repeat`**

---

### **Validation Plan: From "Can it work?" to "I can't work without it."**

This plan is broken into three phases, each with a clear objective, target user persona, key scenarios to validate, and success metrics.

#### **Phase 1: Validate the Core "Aha!" Moment: The Generative Workflow**

**Objective:** Prove that `flujo create` can take a developer's high-level goal and generate a useful, working `pipeline.yaml` faster and more easily than they could write it themselves. This is the single most important value proposition.

**Target User:** A single Python developer, familiar with LLM APIs but not necessarily an expert in orchestration frameworks.

**Key Scenarios to Validate:**

1.  **The Researcher:**
    *   **Goal:** `"Search the web for the latest news on AI hardware, summarize the top 3 articles, and save the summary to a file named 'ai_hardware_news.md'."`
    *   **Tests:** Use of `flujo.builtins.web_search`, chaining to a summarizer agent, and a side-effect with `flujo.builtins.fs_write_file`. This is the classic tool-chaining use case.

2.  **The Data Extractor:**
    *   **Goal:** `"I have a block of unstructured text. Extract the person's name, company, and email address into a structured format, then write it as JSON to 'contact.json'."`
    *   **Tests:** Use of `flujo.builtins.extract_from_text`, demonstrating structured data output, and file I/O.

3.  **The Content Remixer:**
    *   **Goal:** `"Take the content from this URL, create a 5-point bulleted list of the key takeaways, and then translate that list into Spanish."`
    *   **Tests:** Chaining `http_get` to two different LLM-powered steps, showcasing transformation and multi-step reasoning.

**Success Metrics (How we know it's valuable):**

*   **Time-to-First-Run (TTFR):** Can a developer get a working pipeline for these scenarios in **under 2 minutes** using `flujo create`? (Measure this against manual coding).
*   **Correction Overhead:** What percentage of generated YAMLs are **100% valid and runnable on the first try**? For those that aren't, how many lines does the user have to edit to make them work? (Goal: ≥ 80% first-try success, < 3 lines of edits for the rest).
*   **Qualitative Feedback:** "Did this feel like magic?" "Was the generated YAML easy to understand?"

**Implementation Focus for this Phase:**

*   **Refine `architect_pipeline.yaml`:** The prompts and logic within this pipeline are now the most critical part of the product. Test it relentlessly with the scenarios above and harden it.
*   **Strengthen `flujo.builtins`:** Ensure the core skills (`web_search`, `http_get`, `fs_write_file`, `extract_from_text`) are robust and well-documented.
*   **Improve the Repair Loop:** The `ValidateAndRepair` loop in the architect is key. It must be resilient. The `sanitize_blueprint_yaml` helper is a great start; make it even more robust.
*   **Prebuilt test-time compute recipes:** Add high-level helpers in `flujo/recipes/factories.py` for (a) K-variant generation + aggregator ("mixture-of-agents" atop `ParallelStep`), and (b) `refine_until` iterative loops with safe defaults.
*   **Docs and examples:** Provide runnable examples under `examples/` and short guides in `docs/` showing how to enable K-variants, aggregation, and refine-until patterns from `flujo create`.

---

#### **Phase 2: Validate the "Day 2" Experience: The Debug & Refine Loop**

**Objective:** Prove that once a pipeline exists, Flujo provides a superior experience for debugging, observing, and improving it. This validates the "learning" aspect of the roadmap.

**Target User:** The same developer from Phase 1, who now has a `pipeline.yaml` checked into their project and needs to maintain it.

**Key Scenarios to Validate:**

1.  **The Debugger:**
    *   **Scenario:** A run of a generated pipeline fails. The developer runs `flujo lens trace <run_id>`.
    *   **Tests:** Does the trace clearly show which step failed? Is the error message and input/output data for that step easily accessible and understandable?

2.  **The Refiner (Manual):**
    *   **Scenario:** The developer wants to improve the summary prompt in "The Researcher" pipeline. They edit `pipeline.yaml` and re-run it.
    *   **Tests:** Is the feedback loop fast and intuitive? Does the CLI provide clear output on the new results?

3.  **The Refiner (AI-Assisted):**
    *   **Scenario:** The "Data Extractor" pipeline occasionally fails to extract an email. The developer runs `flujo improve`.
    *   **Tests:** Does the output from `self_improvement.py` provide a concrete, actionable suggestion (e.g., "Strengthen the system prompt for the extractor agent by adding an example of the desired JSON format.")?

**Success Metrics:**

*   **Mean Time to Resolution (MTTR):** Can a developer use `flujo lens trace` to identify the root cause of a failure in **under 5 minutes**?
*   **Actionability of Suggestions:** What percentage of `flujo improve` suggestions are rated as "helpful" or "correct" by the developer?
*   **Observability Clarity:** On a scale of 1-5, how confident does a developer feel about what their pipeline is doing after reviewing a trace from `flujo lens`?

**Implementation Focus for this Phase:**

*   **Supercharge `flujo lens`:** This is your primary debugging tool. Enhance `lens_show.py` and `lens_trace.py` to be incredibly clear. The foundation in `flujo/tracing/manager.py` is solid; now focus on the presentation.
*   **Refine `self_improvement.py`:** Use the real-world failures from Phase 1 as the training ground for this agent. Its value is directly proportional to the quality of its suggestions.
*   **Bridge `improve` and `create`:** Instead of `JSON Patch` (complex), create a simpler bridge. `flujo improve` suggests a change. The user can then copy-paste that suggestion into a new `flujo create` goal: `flujo create --goal "Rebuild my data extractor, but this time, strengthen the prompt to handle missing emails."`

---

#### **Phase 2.5: Parallel UX & Monitor**

**Objective:** Provide real-time progress while long-running branches execute, without blocking the main work, by introducing a concurrent monitor pattern and streaming lens output.

**Target User:** The same developer from Phase 2, plus product teams that value responsiveness and progressive feedback during long compute steps.

**Key Scenarios to Validate:**

1.  **The Live Progress Monitor:**
    *   **Scenario:** A compute-heavy branch runs for minutes. A monitor pipeline runs concurrently and prints periodic status to the console.
    *   **Tests:** Monitor branch runs alongside main branches via `ParallelStep`; emits non-intrusive updates; does not affect outcome.

2.  **Asynchronous User Nudges:**
    *   **Scenario:** While branches execute, user feedback (e.g., "focus on source B") is accepted and routed to active agents.
    *   **Tests:** Monitor can accept simple signals and safely forward them without breaking idempotency.

3.  **Streaming Lens:**
    *   **Scenario:** Developer runs `flujo lens stream <run_id>` to see progressive branch updates.
    *   **Tests:** Traces reflect branch start/end, partial results, and merges in near real time using `flujo/tracing/manager.py`.

**Success Metrics:**

*   **Perceived Responsiveness:** Qualitative rating ≥ 4/5 for responsiveness during long runs.
*   **No Outcome Regressions:** Monitor branch must not change results of compute branches.
*   **Low Overhead:** Monitor adds negligible cost and latency (<2%).

**Implementation Focus for this Phase:**

*   **Monitor pattern:** Provide a reusable `monitor_pipeline` pattern that runs concurrently in a `ParallelStep`. Keep all execution logic in policies (`flujo/application/core/step_policies.py`).
*   **Streaming lens mode:** Add a `lens stream` CLI that tails trace events from `flujo/tracing/manager.py` and prints compact updates via `flujo/infra/console_tracer.py`.
*   **Agent monitoring hooks:** Use `flujo/agents/monitoring.py` to emit consistent telemetry around long-running agents; avoid any executor-core branching.

#### **Phase 3: Validate for Teams: Collaboration & Governance**

**Objective:** Prove that Flujo is not just a tool for solo developers but a platform for teams to build and manage workflows reliably and safely.

**Target User:** A small team (2-3 developers) and a team lead/platform owner.

**Key Scenarios to Validate:**

1.  **The Shared Tool Builder:**
    *   **Scenario:** Developer A creates a new, reusable skill in `skills/custom_tools.py` and checks it in. Developer B runs `flujo create`.
    *   **Tests:** Does Developer B's Architect agent immediately discover and offer to use the new skill? (Validates `skill_registry.py` and `skills_catalog.py`).

2.  **The Guardian of the Budget:**
    *   **Scenario:** The team lead sets a cost limit for a specific pipeline in `flujo.toml` using a new, user-friendly command. A developer runs the pipeline in a way that would exceed the budget.
    *   **Tests:** Does the run stop *exactly* when the budget is breached? Is the error message clear? (Validates `budget_resolver.py` and `UsageGovernor`).

3.  **The Code Reviewer:**
    *   **Scenario:** Developer A's `flujo create` session generates a `pipeline.yaml` that uses a side-effect skill (e.g., writing to a production database). They open a Pull Request.
    *   **Tests:** Is the YAML diff clear enough for Developer B to review and approve confidently?

**Success Metrics:**

*   **Budget Adherence:** 100% of runs must halt upon budget breach. This is a critical trust and safety feature.
*   **Zero-Config Skill Discovery:** Skills added to the project's `skills` directory must be available to `flujo create` without any manual registration steps.
*   **PR Review Time:** Can a reviewer understand the intent and impact of a change to a `pipeline.yaml` in under 60 seconds?

**Implementation Focus for this Phase:**

*   **First-Class Budget Commands:** Create `flujo budget set <pipeline_name> --cost-limit 5.0`. This makes the governance features visible and accessible, building on the solid foundation you already have.
*   **Implement a `flujo diff` Command:** This is the most important feature from Roadmap Phase 4a. A command that shows a human-readable summary of changes between two YAML files ("Added step 'X', removed validator from step 'Y'") would be invaluable for PRs.
*   **Side-Effect Warnings:** When `flujo create` generates a pipeline using a skill marked with `side_effects=True`, it should print a bold warning to the console, making the user aware of the potential impact. The logic for this is already in `flujo/cli/helpers.py`.
*   **Parallel budgets & lens:** Ensure `Quota.split()` per branch is surfaced in traces, and `lens trace` visualizes branch-level success, cost/tokens, and merge strategy (`CONTEXT_UPDATE` vs. explicit `field_mapping`).

By following this validation-driven plan, you ensure that you are building on the strongest parts of your current system and prioritizing the features that deliver the most immediate and tangible value to your users.

---

#### **Phase 4: Architect-Driven Decomposition & Parallel Coordination**

**Objective:** Enable the Architect to detect parallelizable sub-goals and emit `ParallelStep` plans with safe merge semantics and aggregation, aligning with mixture-of-agents and scalable test-time compute.

**Target User:** Developers who state compound goals (e.g., research + implementation) and teams seeking faster wall-clock results via parallel work.

**Key Scenarios to Validate:**

1.  **Automatic Parallel Plan:**
    *   **Scenario:** Goal includes clearly disjoint tasks ("Research X and also implement Y").
    *   **Tests:** Architect emits a `ParallelStep` containing sub-pipelines; uses `MergeStrategy.CONTEXT_UPDATE` with explicit `field_mapping` to avoid conflicts.

2.  **Mixture-of-Agents Aggregation:**
    *   **Scenario:** K different agents/models produce candidate outputs; an aggregator agent synthesizes the final result.
    *   **Tests:** Recipes use `ParallelStep` branches for candidates and a deterministic aggregation step; no executor-specific conditionals.

3.  **Refine-Until for Hard Tasks:**
    *   **Scenario:** Architect proposes a loop for iterative improvement when results fail validation.
    *   **Tests:** Uses `LoopStep` with `ContextManager.isolate()` semantics; merges back only on success.

**Success Metrics:**

*   **Decomposition Quality:** ≥ 60% of suitable goals produce correct `ParallelStep` plans on first try; < 3 edits for the rest.
*   **Speedup:** For N branches, wall-clock approaches max(branch) vs. sum(branch).
*   **Safety:** No context poisoning across retries/branches; zero control-flow demotions.

**Implementation Focus for this Phase:**

*   **Architect planner heuristics:** Extend `flujo/architect/builder.py` to detect disjoint sub-goals and emit `ParallelStep` with explicit `field_mapping`. Keep logic in policies; Architect only describes the plan.
*   **High-level recipes:** Provide `mixture-of-agents` and `refine_until` presets in `flujo/recipes/factories.py` with examples/docs.
*   **Policy-driven execution:** Reuse `DefaultParallelStepExecutor` and loop policies in `flujo/application/core/step_policies.py`; never branch in `ExecutorCore`.
*   **Centralized config & quotas:** Continue to use `flujo.infra.config_manager` for settings and `Quota.split()` for branch limits; no reactive governors.
