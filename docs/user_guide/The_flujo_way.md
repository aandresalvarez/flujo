# The Flujo Way: A Developer's Guide to Production-Ready AI Workflows

Welcome to the **official guide for developers using flujo**—a modern framework for orchestrating AI-powered pipelines built with explicit control flow, modular design, and production-grade resilience. This guide teaches you how to build delightful, powerful workflows using **the Flujo way**.

---

## 🌟 Core Philosophy: Explicit, Composable, Robust

**flujo** is built around three principles:

* **Explicit over implicit** – Control flow, logic, and data movement are *visible* in your pipeline definition.
* **Composable** – Workflows are made from modular, reusable agents and steps.
* **Robust by default** – Retry policies, validation, human-in-the-loop, and cost limits are all native features.

---

## 🧠 1. Production Steps with Real Logic

### A. Basic Agents with AsyncAgentProtocol

Use `AsyncAgentProtocol` for simple agents that don't need typed context:

```python
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.resources import AppResources
from flujo.domain.models import PipelineContext

class TriageContext(PipelineContext):
    post_content: str
    author_id: int
    author_reputation: float

class TriageAgent(AsyncAgentProtocol[dict, TriageContext]):
    async def run(self, data: dict, *, resources: AppResources, **kwargs) -> TriageContext:
        reputation = await resources.db.fetch_reputation(data["author_id"]) or 0.5
        return TriageContext(
            post_content=data["content"],
            author_id=data["author_id"],
            author_reputation=reputation,
        )
```

### B. Type-Safe Context with ContextAwareAgentProtocol

For agents that need typed pipeline context, use `ContextAwareAgentProtocol`:

```python
from flujo.domain.agent_protocol import ContextAwareAgentProtocol
from flujo.domain.models import PipelineContext

class ResearchContext(PipelineContext):
    research_topic: str = "Unknown"
    sources_found: int = 0
    summary: str | None = None

class PlanResearchAgent(ContextAwareAgentProtocol[str, str, ResearchContext]):
    async def run(
        self,
        data: str,
        *,
        context: ResearchContext,
        **kwargs: Any,
    ) -> str:
        """Identify the core topic and store it in the context."""
        topic = "The History of the Python Programming Language"
        context.research_topic = topic
        return f"Research plan for {topic}"
```

Declare the step cleanly:

```python
from flujo import Step
triage_step = Step("TriagePost", TriageAgent())
research_step = Step("PlanResearch", PlanResearchAgent())
```

✅ Encapsulation
✅ Testability
✅ Clear dependency injection
✅ Type-safe context handling

---

## 🔀 2. Control Flow as Code

### 🪢 A. Branching: ConditionalStep

```python
from flujo import Step, Pipeline
from flujo.models import BaseModel

def route(ctx: TriageContext, _):
    if ctx.author_reputation < 0.2: return "high_risk"
    if ctx.author_reputation < 0.6: return "standard_review"
    return "auto_approve"

router_step = Step.branch_on(
    name="RouteContent",
    condition_callable=route,
    branches={
        "high_risk": Pipeline.from_step(Step("Escalate", high_risk_agent)),
        "standard_review": Pipeline.from_step(Step("Review", make_review_agent())),
    },
    default_branch_pipeline=Pipeline.from_step(Step("AutoApprove", logging_agent)),
)
```

### 🔁 B. Looping: LoopStep

```python
def is_confident(result, _): return result.get("confidence", 0) > 0.95

refine_step = Step.loop_until(
    name="RefineSarcasm",
    loop_body_pipeline=Pipeline.from_step(Step("Refine", sarcasm_agent)),
    exit_condition_callable=is_confident,
    max_loops=3,
)
```

### 🤝 C. Human in the Loop

```python
from flujo import Step

hitl_step = Step.human_in_the_loop(
    name="ManualReview",
    message_for_user="Please label this post as sarcastic or not."
)
```

### 🔄 D. Refinement Loops

```python
refine_step = Step.refine_until(
    name="RefineCode",
    generator_pipeline=Pipeline.from_step(Step("Generate", code_agent)),
    critic_pipeline=Pipeline.from_step(Step("Critique", make_review_agent())),
    max_refinements=5,
)
```

---

## 📦 3. Shared State: PipelineContext

### A. Basic Context

```python
from flujo.domain.models import PipelineContext

class ModerationContext(PipelineContext):
    post_id: int
    triage_decision: str | None = None
    refinement_attempts: int = 0
    final_disposition: str | None = None
```

### B. Extended Context with Built-in Features

```python
from flujo.domain.models import PipelineContext
from pydantic import Field

class ResearchContext(PipelineContext):
    research_topic: str = "Unknown"
    sources_found: int = 0
    summary: str | None = None
    # Inherits: initial_prompt, import_artifacts, step_outputs, hitl_history, command_log
```

Use in any agent:

```python
async def run(self, decision: str, *, context: ModerationContext, **kwargs):
    context.final_disposition = decision
```

Attach it to the runner:

```python
from flujo import Flujo
runner = Flujo(pipeline, context_model=ModerationContext)
```

---

## ✅ 4. Validations & Quality Gates

```python
from flujo.domain.validation import BaseValidator, ValidationResult

class NoPII(BaseValidator):
    name = "NoPII"
    async def validate(self, text: str, **_) -> ValidationResult:
        if "ssn:" in text.lower():
            return ValidationResult(is_valid=False, feedback="PII detected", validator_name=self.name)
        return ValidationResult(is_valid=True, validator_name=self.name)
```

Use it in a validation step:

```python
from flujo import Step
quality_gate = Step.validate_step(
    name="CheckJustification",
    agent=make_validator_agent(),
    validators=[NoPII()],
    plugins=[StyleGuidePlugin()],
    strict=True,  # Step fails if validation fails
)
```

**Strict vs Non-Strict Validation:**
- **`strict=True` (default)**: If any validation fails, the entire step fails and the pipeline stops or retries.
- **`strict=False`**: Step passes but records validation failure in metadata for auditing.

---

## 💸 5. Cost Limits & Tracing

### 🔒 Cost Control

```python
from flujo import Flujo
from flujo.models import UsageLimits

runner = Flujo(pipeline, usage_limits=UsageLimits(total_cost_usd_limit=0.50))
```

> [!TIP]
> **Advanced Cost Control:** For complex scenarios involving loops, parallel execution, and nested workflows, see the [Cost Control cookbook](../cookbook/cost_control.md) guide.

### 🪄 Real-time Logs

```python
from flujo.infra.console_tracer import ConsoleTracer

# Quick enablement with defaults
runner = Flujo(pipeline, local_tracer="default")

# Or configure it yourself
custom_tracer = ConsoleTracer(level="debug", log_inputs=True)
runner = Flujo(pipeline, local_tracer=custom_tracer)
```

---

## 🧩 6. Composition: Pipelines of Pipelines

### A. Step Composition

```python
from flujo import Step, Pipeline

analysis = triage_step >> router_step >> refine_step
notify = Step("Format", format_agent) >> Step("Notify", send_agent)

main_pipeline = analysis >> notify
```

### B. Pipeline Composition (v2.1+)

```python
from flujo import Step, Pipeline

# Build independent pipelines
data_processing = Step("Extract", extract_agent) >> Step("Transform", transform_agent)
analysis = Step("Analyze", analyze_agent) >> Step("Validate", validate_agent)
reporting = Step("Format", format_agent) >> Step("Send", send_agent)

# Chain entire pipelines together
workflow = data_processing >> analysis >> reporting
```

---

## 🎛️ 7. Tuning Agents in Flujo

### A. Global model config via `.env`

```bash
DEFAULT_SOLUTION_MODEL=openai:gpt-3.5-turbo
DEFAULT_REVIEW_MODEL=openai:gpt-4
```

### B. Per-agent model + settings

```python
from flujo.infra.agents import make_agent_async

agent = make_agent_async(
    model="openai:gpt-4",
    system_prompt="You are an expert...",
    output_type=str,
    temperature=0.7,
    max_tokens=800,
    top_p=0.9
)
```

### C. Per-step overrides

```python
from flujo import Step

pipeline = (
    Step.review(agent, timeout=30)
    >> Step.solution(agent, retries=3, temperature=0.5)
)
```

### D. Retry logic

* **Step-level**: `Step(..., retries=3)`
* **Pipeline-level**:

```python
Flujo(pipeline, max_retries=2, retry_on_error=True)
```

---

## 🔧 8. Advanced Features

### A. Pipeline Validation

```python
from flujo import Pipeline

# Validate pipeline before running
validation_report = pipeline.validate_graph()
if not validation_report.is_valid:
    print("Pipeline validation failed:")
    for finding in validation_report.findings:
        print(f"  - {finding.message}")
```

### B. CLI Validation

```bash
flujo validate my_pipeline.py
```

### C. Streaming Output

```python
async for chunk in runner.stream_async("hello"):
    if isinstance(chunk, str):
        print(chunk, end="")
    else:
        result = chunk  # Final PipelineResult
```

### D. Pipeline as Step

```python
# Convert a pipeline into a reusable step
sub_pipeline = Step("A", agent_a) >> Step("B", agent_b)
pipeline_step = runner.as_step(name="SubWorkflow")

# Use in another pipeline
main_pipeline = Step("Start", start_agent) >> pipeline_step >> Step("End", end_agent)
```

> [!TIP]
> **Advanced Composition Patterns:**
> For comprehensive examples of the `as_step` pattern, including context propagation, resource sharing, and crash recovery, see the [Pipeline as a Step](../cookbook/pipeline_as_step.md) cookbook guide.

---

## 📚 9. Import Structure

### A. Core Components (Top Level)

```python
from flujo import (
    Flujo,           # Main pipeline runner
    Step,            # Pipeline step builder
    step,            # Step decorator
    Pipeline,        # Pipeline composition
    Task,            # Task model
    Candidate,       # Candidate model
    make_agent_async, # Agent factory
    settings,        # Global settings
    init_telemetry,  # Telemetry initializer
)
```

### B. Domain-Specific Imports

```python
# Agent protocols
from flujo.domain.agent_protocol import AsyncAgentProtocol, ContextAwareAgentProtocol

# Models and types
from flujo.models import BaseModel, UsageLimits, PipelineResult

# Resources
from flujo.domain.resources import AppResources

# Validation
from flujo.domain.validation import BaseValidator, ValidationResult

# Tracing
from flujo.infra.console_tracer import ConsoleTracer

# Testing utilities
from flujo.testing import StubAgent, gather_result
```

---

## ✅ Summary

| Feature         | How to Use                                                    |
| --------------- | ------------------------------------------------------------- |
| 🧱 Basic Agents | `AsyncAgentProtocol`, clean encapsulation                     |
| 🧠 Context Agents | `ContextAwareAgentProtocol`, type-safe context handling      |
| 🔁 Control Flow | `Step.branch_on`, `Step.loop_until`, `Step.human_in_the_loop` |
| 🔄 Refinement   | `Step.refine_until` for generator-critic loops                |
| 🧠 Context      | `context: MyContext` shared across steps                      |
| ✅ Validation    | `Step.validate_step(..., validators=[...], plugins=[...])`    |
| 💵 Cost Limits  | `UsageLimits(total_cost_usd_limit=...)`                       |
| 📜 Logs         | `ConsoleTracer` for debug visibility                          |
| 🔧 Tuning       | Use `make_agent_async(...)` and `Step(..., temperature=...)`  |
| 🔍 Validation   | `pipeline.validate_graph()` and `flujo validate` CLI         |
| 📦 Composition  | `pipeline1 >> pipeline2` for modular workflows                |

---

This is the **Flujo Way**: empowering developers to build resilient, maintainable, and intelligent AI workflows with clarity and joy.
