Of course. Here is a comprehensive, updated roadmap for `flujo` that integrates the unified AI-first strategy with your existing plans.

First, let's analyze the status of your current roadmap items based on the provided project files.

### Analysis of Current Roadmap and Project Status

Based on the file contents, here's what has already been implemented:

*   **`ParallelStep`:** ✅ **Implemented.** The `flujo/domain/pipeline_dsl.py` file contains the `ParallelStep` class and the `Step.parallel` factory. The `flujo_engine.py` also contains the `_execute_parallel_step_logic` function. The core logic for concurrent execution exists.
*   **Context Safety (`Isolate-and-Merge`):** ✅ **Partially Implemented.** The `DummyRemoteBackend` test hints at serialization/deserialization, and `_execute_parallel_step_logic` creates a `deepcopy` of the context for each branch, which aligns with the "Isolate" part of the strategy. The "Merge" part (handling conflicts) does not appear to be implemented yet.
*   **`fallback` Logic:** ❌ **Not Implemented.** The `Step` class in `pipeline_dsl.py` does not have a `fallback` method or a `fallback_step` attribute. The engine logic for handling this is also absent. This remains a valid future item.

Now, let's build the new, detailed roadmap.

---

# Flujo Project Roadmap: The Path to an AI-First Framework

**Last Updated:** June 25, 2025
**Status:** In Progress

## Vision Statement

To evolve `flujo` from a developer-centric orchestration library into a truly **AI-native framework**. The primary user persona we are designing for is a generative AI model, which requires an API that is explicit, discoverable, structured, and provides actionable feedback. This roadmap outlines the features and architectural changes required to achieve this vision, enabling both humans and AI agents to build, modify, and reason about complex workflows with ease.

---

## Theme 1: Production-Grade Control Flow (Resilience & Performance)

*Goal: Enhance the core DSL with robust patterns for handling real-world complexity like failures and I/O latency.*

| Epic | Status | Description & Key Tasks | Target Release |
| :--- | :--- | :--- | :--- |
| **Parallel Execution** | 🚧 **In Progress** | Implement concurrent execution of pipeline branches. | v0.5.0 |
| | ✅ Implemented | `Step.parallel` factory and `ParallelStep` class. | |
| | ✅ Implemented | `asyncio.TaskGroup`-based concurrency for structured execution. | |
| | ✅ Implemented | Basic context isolation via `deepcopy`. | |
| | ⏳ **To Do** | Implement **conflict-aware context merging**. Add an optional `on_context_conflict` handler to `Step.parallel` for custom merge logic. | |
| | ⏳ **To Do** | Enhance `StepResult` from `ParallelStep` to include a dictionary of individual branch results for better traceability. | |
| | ⏳ **To Do** | Write comprehensive documentation and a cookbook recipe for `Step.parallel`. | |
| **Fallback Logic** | 📝 **Proposed** | Introduce a declarative `fallback` mechanism for steps to handle transient errors gracefully. | v0.6.0 |
| | ⏳ **To Do** | Add `.fallback(other_step: Step)` method to the `Step` class. | |
| | ⏳ **To Do** | Modify the `flujo_engine` to catch step failures and execute the `fallback_step` with the original input. | |
| | ⏳ **To Do** | Enhance `StepResult.metadata_` to include `fallback_triggered` and the results of both the primary and fallback attempts. | |
| | ⏳ **To Do** | Add unit and integration tests for various fallback scenarios (e.g., fallback succeeds, fallback also fails). | |

---

## Theme 2: AI-Native API & Pipeline Construction

*Goal: Make pipeline construction explicit, structured, and intuitive for a generative AI model, reducing reliance on custom syntax and implicit magic.*

| Epic | Status | Description & Key Tasks | Target Release |
| :--- | :--- | :--- | :--- |
| **Declarative Pipeline Builder** | 📝 **Proposed** | Introduce a primary, AI-friendly way to build pipelines from structured data. | v0.7.0 |
| | ⏳ **To Do** | Design and implement a stable JSON/YAML schema for pipeline definitions. | |
| | ⏳ **To Do** | Create `Pipeline.from_definition(dict)` and `pipeline.to_dict()` methods for serialization and deserialization. | |
| | ⏳ **To Do** | Create an initial version of `docs/ai_patterns.md` demonstrating this new builder pattern as the canonical approach. | |
| **Fluent Builder API** | 📝 **Proposed** | Add a secondary, fluent API for programmatic pipeline construction. | v0.7.0 |
| | ⏳ **To Do** | Implement `Pipeline.builder().add_step(step).add_pipeline(pipeline)` methods. | |
| | ⏳ **To Do** | Ensure the builder correctly handles type flow and context propagation. | |
| **API Simplification & Consistency** | 🚧 **In Progress** | Deprecate confusing or redundant patterns and enforce consistent naming. | v0.5.0 |
| | ✅ Implemented | The library already favors a single `Flujo` engine over the `Default` recipe, aligning with simplification. | |
| | ⏳ **To Do** | Formally deprecate the `Default` recipe in documentation, guiding users to the more flexible `Flujo` engine and builders. | |
| | ⏳ **To Do** | Add `Step.validate` as an alias for `Step.validate_step` to improve naming consistency. Mark the old name for deprecation. | |
| | ⏳ **To Do** | Introduce a single, explicit `flujo.configure()` function for setting global configurations (API keys, models). | |

---

## Theme 3: Introspection and Safe Modification

*Goal: Empower AI agents to understand, analyze, and safely modify pipelines programmatically.*

| Epic | Status | Description & Key Tasks | Target Release |
| :--- | :--- | :--- | :--- |
| **Step Registry & Introspection** | 📝 **Proposed** | Create a central registry for discoverability and a public API for introspection. | v0.8.0 |
| | ⏳ **To Do** | Modify the `@step` decorator and `Step.from_callable` to optionally register steps in a global `STEP_REGISTRY`. | |
| | ⏳ **To Do** | Create the `flujo.introspect` module with `list_steps()` and `get_step_schema(step_name)` functions. | |
| | ⏳ **To Do** | Implement `flujo.introspect.explain_pipeline(pipeline)` to return a structured or natural language description. | |
| **Programmatic Pipeline Editing**| 📝 **Proposed** | Provide safe, high-level methods for an AI to modify a pipeline's structure. | v0.9.0 |
| | ⏳ **To Do** | Implement `pipeline.insert_after(step_name, new_step)`. | |
| | ⏳ **To Do** | Implement `pipeline.replace(step_name, new_step)`. | |
| | ⏳ **To Do** | Implement `pipeline.remove(step_name)`. | |
| | ⏳ **To Do** | Ensure all modification methods trigger a re-validation of the pipeline's type flow. | |

---

## Theme 4: AI-Centric Feedback and Extensibility

*Goal: Make the framework's feedback loop and extension patterns explicit and easy for an AI to learn from.*

| Epic | Status | Description & Key Tasks | Target Release |
| :--- | :--- | :--- | :--- |
| **Actionable Error System** | 📝 **Proposed** | Rework exceptions to provide structured, helpful suggestions for AI agents. | v0.8.0 |
| | ⏳ **To Do** | Create the `flujo.ai_errors` module with `AIFriendlyError` base class containing a `suggestion` field. | |
| | ⏳ **To Do** | Refactor the `flujo_engine` to catch internal errors (e.g., `TypeMismatchError`) and re-raise them as `AITypeMismatchError` with a concrete suggestion. | |
| | ⏳ **To Do** | Audit all user-facing exceptions and augment them with AI-friendly guidance. | |
| **Formalized Extension APIs** | 📝 **Proposed** | Simplify the process of creating new, reusable `Step` types. | v0.9.0 |
| | ⏳ **To Do** | Create an `AbstractStep` base class or mixins to encapsulate Pydantic initialization logic. | |
| | ⏳ **To Do** | Document a clear, simple pattern for creating new step types like `RetryStep` or `CacheStep`. | |
| | ⏳ **To Do** | Add a new "Extending Flujo" guide in the documentation focused on these new patterns. | |

---

## Theme 5: The Vision - Natural Language Orchestration

*Goal: Achieve the ultimate AI-first experience where the AI describes the workflow and `flujo` builds it.*

| Epic | Status | Description & Key Tasks | Target Release |
| :--- | :--- | :--- | :--- |
| **Natural Language Builder** | 📝 **Proposed** | Create a high-level function that uses an LLM to translate a natural language description into a `flujo` pipeline. | v1.0.0 |
| | ⏳ **To Do** | Create an internal "Builder Agent" that uses the `flujo.introspect` module as its toolset. | |
| | ⏳ **To Do** | The Builder Agent's goal is to output a valid pipeline definition dictionary. | |
| | ⏳ **To Do** | Implement the public-facing `flujo.build_from_description(description: str)` function, which orchestrates the Builder Agent and uses its output to call `Pipeline.from_definition()`. | |
| | ⏳ **To Do** | Create "AI dogfooding" tests where an LLM is prompted to build and run a pipeline using this new function. | |