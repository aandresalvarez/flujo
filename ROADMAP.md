# Flujo Roadmap: From Developer Framework to Enterprise Platform

## Vision

Flujo is evolving with a dual purpose in mind. We aim to deliver an unparalleled developer framework for individual AI workflow creators, and to grow it into an enterprise-ready platform for production-scale workloads. In other words, Flujo will make the single-developer "inner loop" experience as smooth and powerful as possible, while also providing a clear path to scale those workflows in distributed, governed, enterprise environments.

We will achieve this by keeping a consistent, declarative core (the pipeline DSL) and making execution, state management, and policy enforcement components pluggable for different needs.

**Our guiding principle is the user-value flywheel: Generate → Run & Debug → Refine & Govern → Repeat.** Each phase of the roadmap strengthens one or more parts of this cycle.

## Roadmap Overview (Phased Approach)

### Phase 1 – Perfecting the Developer Experience
Focus on core stability, speed, and ease of use for a single developer. Address technical debt and make the CLI/authoring experience robust and delightful.

**Current status (completed milestones):**
- Policy execution uses `ExecutionFrame` across all default executors (frame-only migration).
- Governance layer shipped with settings-driven allow/deny and telemetry counts.
- Shadow evaluations shipped with judge agent + telemetry sink (optional DB sink pending).
- Sandboxed code execution shipped (RemoteSandbox, Python-only DockerSandbox) plus `code_interpreter` skill.
- Memory interface shipped (VectorStore protocol, in-memory/Null stores, MemoryManager indexing, `PipelineContext.retrieve`).
- OpenAPI generator extended to emit typed agent/tool wrappers.
- Context typing and step I/O validation strengthened (branch/parallel/import-aware V-CTX1/V-CTX2).

### Phase 2 – Building the Bridge to Enterprise
Introduce a pluggable architecture. Define clear interfaces and add optional backends so Flujo can plug into enterprise infrastructure (databases, servers) without compromising the developer experience.

### Phase 3 – Scaling Execution for Massive Workloads
Enable distributed execution. Allow a pipeline to run across many machines and handle thousands or millions of parallel steps. Introduce a worker fleet and scheduling system to scale out.

### Phase 4 – Enterprise-Grade Governance & Management
Add governance, security, and management tools. Provide policy enforcement, advanced monitoring, and administrative controls so organizations can safely manage AI workflows at scale.

---

## Phase 1: Perfecting the Developer Experience (The Framework)

### Objective
Solidify Flujo as a best-in-class tool for a single developer to go from idea to a reliable, production-ready AI workflow on their local machine. This phase is all about the "inner loop" of development – making generation, running, and debugging of pipelines fast, intuitive, and robust. It focuses on eliminating pain points (like brittle code or confusing errors) and shoring up the foundation of the system before scaling up.

### Key Improvements

#### 1. Fortify the Foundation (Address Technical Debt)
- **Refactor monolithic or hard-to-maintain components** into smaller, clear units
- Break down the large pipeline builder module (e.g. `flujo/architect/builder.py`) and the core executor logic into single-responsibility pieces
- Finalize and integrate the new, modular UltraStepExecutor design as the single source of truth for execution logic
- **Addresses weakness**: Tangled codebase making the system unstable and hard to extend

#### 2. Enhance the Generative Workflow (`flujo create`)
- **Continuously refine the pipeline generation process** so that out-of-the-box pipelines work better with less tweaking
- Improve the template (`architect_pipeline.yaml`) that the AI uses to scaffold new pipelines
- Strengthen the built-in skills library (`flujo.builtins`) to ensure generated steps are reliable and production-ready
- Improve the YAML "repair loop" to handle a wider range of mistakes gracefully
- **Addresses weakness**: Generated pipelines often require manual fixes or produce errors

#### 3. Supercharge Debugging and Refinement (`flujo lens` & CLI)
- **Enhance the `flujo lens` suite** to provide clearer, more actionable diagnostics
- Implement real-time progress feedback for running pipelines (e.g. `flujo run --progress`)
- Refine the `flujo improve` command to give more concrete and accurate suggestions
- **Addresses weakness**: Lack of transparency during execution and vague improvement suggestions

#### 4. Introduce Foundational Resilience Patterns
- **Implement circuit breaker mechanism** for agents that call external APIs
- Prevent single flaky API from hanging or crashing the entire pipeline
- **Addresses weakness**: One bad API response can cascade and break a whole run

> **By the end of Phase 1**, Flujo's core will be clean, stable, and developer-friendly. We prioritize fixing current weaknesses in code structure and user experience before adding complexity. A single developer should feel that Flujo is intuitive and reliable for building AI workflows on their machine.

---

## Phase 2: Building the Bridge to Enterprise (Pluggable Architecture)

### Objective
Prepare Flujo's architecture for enterprise use-cases without sacrificing its developer-first ethos. In this phase, we make all the major runtime components swappable and introduce scalable alternatives for each. The idea is that a developer can prototype locally with a simple setup, then an organization can switch Flujo to "enterprise mode" by plugging in more robust backends (databases, servers, etc.) as needed.

### Key Improvements

#### 1. Formalize Backend Interfaces
- **Define clear interface contracts** (Python protocols or abstract base classes) for every major component
- Includes: ExecutionBackend, StateBackend, PolicyBackend, RegistryBackend, QuotaBackend
- Refine existing ExecutionBackend and StateBackend interfaces
- Create new interfaces for policy, registry, and quota management
- **Addresses weakness**: Tight-coupling makes it hard to plug in alternative implementations

#### 2. Implement Enterprise-Ready Backends
- **Develop optional, enterprise-grade backend implementations** as plugins or extras
- PostgreSQL-based StateBackend for robust, concurrent state storage
- Database-backed RegistryBackend for shared skills and pipelines
- **Addresses weakness**: Default backends are lightweight but not suitable for concurrent or multi-user scenarios

#### 3. Introduce the Flujo API Server (Control Plane)
- **Develop a lightweight FastAPI server** that wraps the Flujo engine
- Expose fundamental endpoints:
  - `POST /runs` to start a new pipeline run
  - `GET /runs/{run_id}` to fetch status/results
  - `POST /runs/{run_id}/resume` to supply input to paused HITL steps
- **Addresses weakness**: Flujo runs are currently initiated via CLI on a single machine only

#### 4. Make Backends Configurable (Seamless Switching)
- **Extend the configuration system** (`flujo.toml` and ConfigManager) to support backend selection
- Example: `execution_backend = "flujo_enterprise.backends.CeleryBackend"`
- Default settings continue to point to simple local backends
- **Addresses weakness**: Runtime assumptions are hard-coded (e.g. always using local SQLite)

> **By the end of Phase 2**, Flujo will still feel the same to a single developer using it on their laptop, but under the hood it will be capable of swapping in heavy-duty components. We create the option for scalability and multi-user operation.

---

## Phase 3: Scaling Execution for Massive Workloads (Distributed Platform)

### Objective
Enable Flujo to reliably execute pipelines that have massive parallelism or heavy workloads, by distributing work across many machines. In this phase we turn Flujo from a single-machine runner into a distributed execution platform. A given pipeline should be able to fan out to hundreds or thousands of parallel tasks (agents) and handle many pipelines running at once.

### Key Improvements

#### 1. Distributed Execution Backend
- **Implement a new ExecutionBackend** using a distributed task queue (Celery with RabbitMQ/Redis)
- Package each step or parallel agent as a task and push to queue
- Worker processes on other machines pick up tasks and execute them
- **Addresses weakness**: Flujo runs only on one machine and can exhaust resources on large workflows

#### 2. Stateless Worker Fleet
- **Develop a standard, containerized Flujo worker application** that can run anywhere
- Provide Kubernetes Helm charts and documentation for auto-scaling fleet deployment
- **Addresses weakness**: Parallel steps are limited by one machine's CPU/threads

#### 3. Enhanced Concurrency Control for Parallelism
- **Re-introduce and improve global concurrency limit mechanism** for parallel steps
- Add semaphore-like control configurable in `flujo.toml` or per-step in YAML
- Celery-based ExecutionBackend will respect this limit by throttling task pulls
- **Addresses weakness**: Without concurrency limits, pipelines could overload the system or APIs

#### 4. Distributed Quota Management
- **Replace in-memory Quota mechanism** with centralized quota service
- Implement Redis-backed QuotaBackend using atomic operations or Lua scripts
- All workers check and update usage quotas against single source of truth
- **Addresses weakness**: In-memory budget tracker cannot coordinate across multiple processes

> **By the end of Phase 3**, Flujo can run at cloud scale. A pipeline author won't need to change their YAML to benefit from this – the heavy lifting is done by the new ExecutionBackend and worker infrastructure.

---

## Phase 4: Enterprise-Grade Governance & Management

### Objective
Provide the tools and controls for organizations to manage, govern, and secure their AI workflows at scale. In this final phase, we focus on operational governance – making sure that as Flujo runs in production, it can meet enterprise requirements for security, compliance, monitoring, and ease of management.

### Key Improvements

#### 1. Centralized Policy Service & Enforcement
- **Create a Policy Registry Service** (centralized service with database) to store organization-wide rules
- Policies include: resilience rules, security rules, resource use budgets
- Integrate Policy Enforcement Point (PEP) into Flujo worker process
- **Addresses weakness**: No easy way to enforce rules across all pipelines

#### 2. Improved Parallel Context Merging
- **Enhance results merging from parallel branches** with powerful MergeConfig for ParallelStep
- Allow configurable merge strategies: deep-merging dictionaries, appending/deduplicating lists, accumulating numeric results
- **Addresses weakness**: Merging outputs from parallel steps is rigid and requires custom code

#### 3. Comprehensive Observability (Tracing & Logging)
- **Integrate distributed tracing** (OpenTelemetry) across API server, orchestrator, and worker nodes
- Provide configurations for centralized logging (ELK Stack, Grafana Loki)
- **Addresses weakness**: Troubleshooting distributed systems is hard with current tools

#### 4. Administrative Tools and UI
- **Develop simple administration interface** (extended CLI and/or minimal web UI)
- Features: pipeline run history, monitoring running pipelines, budget/quota management, policy editing
- Commands like `flujo admin policy list`, `flujo admin set-budget <pipeline> <limit>`
- **Addresses weakness**: Everything must be managed by editing config files or digging into databases

> **By the end of Phase 4**, Flujo will not only scale technically but also come with the controls and interfaces needed for safe enterprise operation. Organizations will be able to enforce rules, monitor activity, and manage the platform effectively.

---

## Conclusion

This roadmap outlines a clear evolution of Flujo from a powerful single-developer tool to a scalable, enterprise-grade platform. Each phase builds upon the previous one's successes:

1. **Perfect the core** and fix current weaknesses first
2. **Gradually add flexibility** to swap components
3. **Enable scaling out** with distributed execution
4. **Add governance and management** features for production use

Throughout this journey, we maintain Flujo's core philosophy of a declarative, user-friendly pipeline framework. By following this phased plan, we ensure that Flujo grows in capability without losing sight of developer experience – delivering a product that individual developers love to use, and that enterprises trust to run their most demanding AI workflows.