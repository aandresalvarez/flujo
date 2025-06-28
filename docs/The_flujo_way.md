 

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

Use `AsyncAgentProtocol` to encapsulate logic in production-ready agents:

```python
from flujo import AppResources, AsyncAgentProtocol
from pydantic import BaseModel

class TriageContext(BaseModel):
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

Declare the step cleanly:

```python
from flujo import Step
triage_step = Step("TriagePost", TriageAgent())
```

✅ Encapsulation
✅ Testability
✅ Clear dependency injection

---

## 🔀 2. Control Flow as Code

### 🪢 A. Branching: ConditionalStep

```python
from flujo import Step, Pipeline

def route(ctx: TriageContext, _): 
    if ctx.author_reputation < 0.2: return "high_risk"
    if ctx.author_reputation < 0.6: return "standard_review"
    return "auto_approve"

router_step = Step.branch_on(
    name="RouteContent",
    condition_callable=route,
    branches={
        "high_risk": Pipeline.from_step(Step("Escalate", high_risk_agent)),
        "standard_review": Pipeline.from_step(Step("Review", llm_review_agent)),
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

---

## 📦 3. Shared State: PipelineContext

```python
from pydantic import BaseModel

class ModerationContext(BaseModel):
    post_id: int
    triage_decision: str | None = None
    refinement_attempts: int = 0
    final_disposition: str | None = None
```

Use in any agent:

```python
async def run(self, decision: str, *, pipeline_context: ModerationContext, **kwargs):
    pipeline_context.final_disposition = decision
```

Attach it to the runner:

```python
from flujo import Flujo
runner = Flujo(pipeline, context_model=ModerationContext)
```

---

## ✅ 4. Validations & Quality Gates

```python
from flujo.validation import BaseValidator, ValidationResult

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
    agent=style_validator_agent,
    validators=[NoPII()],
    plugins=[StyleGuidePlugin()]
)
```

---

## 💸 5. Cost Limits & Tracing

### 🔒 Cost Control

```python
from flujo import Flujo, UsageLimits

runner = Flujo(pipeline, usage_limits=UsageLimits(total_cost_usd_limit=0.50))
```

### 🪄 Real-time Logs

```python
from flujo.tracing import ConsoleTracer

runner = Flujo(pipeline, local_tracer="default")
```

---

## 🧩 6. Composition: Pipelines of Pipelines

```python
from flujo import Step, Pipeline

analysis = triage_step >> router_step >> refine_step
notify = Step("Format", format_agent) >> Step("Notify", send_agent)

main_pipeline = analysis >> notify
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
from flujo import make_agent_async

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

## ✅ Summary

| Feature         | How to Use                                                    |
| --------------- | ------------------------------------------------------------- |
| 🧱 Agents       | `AsyncAgentProtocol`, clean encapsulation                     |
| 🔁 Control Flow | `Step.branch_on`, `Step.loop_until`, `Step.human_in_the_loop` |
| 🧠 Context      | `pipeline_context: MyContext` shared across steps             |
| ✅ Validation    | `Step.validate_step(..., validators=[...], plugins=[...])`    |
| 💵 Cost Limits  | `UsageLimits(total_cost_usd_limit=...)`                       |
| 📜 Logs         | `ConsoleTracer` for debug visibility                          |
| 🔧 Tuning       | Use `make_agent_async(...)` and `Step(..., temperature=...)`  |

---

This is the **Flujo Way**: empowering developers to build resilient, maintainable, and intelligent AI workflows with clarity and joy.
