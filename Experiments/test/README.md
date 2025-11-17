# Flujo Evidence-First Reasoning -- Builder's Guide (Score-Free Search)

## Abstract

When decisions are subjective (tone, trade-offs, conflicting evidence), a single numeric heuristic is brittle. This guide shows how to implement **evidence-first, score-free search** in **Flujo**. Instead of optimizing one score, your pipeline compares options **pairwise** using **badges** (rules passed, corroboration, source quality, recency, cost), advances the winner, and escalates to **human-in-the-loop (HITL)** only when **Risk x Uncertainty** is high. Everything here is **framework-native**: YAML pipelines, small Python hooks, portable across model vendors.

## Motivation

* **Explainability**: badges and pairwise comparisons are easy to justify.
* **Robustness**: avoids fragile "magic numbers."
* **Practicality**: works with verifiers, tools, and HITL you already use in Flujo.
* **Portability**: keep your controller logic while swapping model providers.

---

## What you build in Flujo

* **Pipelines (YAML)**: plan -> retrieve/tools -> act -> verify -> save.
* **Controllers**: choose how the frontier advances (here: `ordinal`).
* **Evidence & Comparator hooks (Python)**: compute badges and decide winners.
* **HITL router**: gate (blocking), nudge (non-blocking), or review (post-hoc).
* **Traces**: every decision is logged for replay and audit.

---

## Quick start (project layout)

_This experimental package lives entirely under `Experiments/test` and is not wired into core Flujo services._

> **Setup tip:** export `OPENAI_API_KEY` (or update `providers.yaml`) before running the controller pipeline. Without real credentials, the planning agent will raise a configuration error.

```
Experiments/test/
|-- app/
|   |-- evidence.py        # build badges per state
|   |-- comparators.py     # pairwise comparator
|   |-- controllers.py     # experimental ordinal controller
|   |-- models.py          # pydantic models shared by agents
|   |-- verifiers.py       # domain checks (tests/rules)
|   \-- tools/             # optional tools (search, code, db)
|-- pipelines/
|   |-- score_free.py               # runnable in-process pipeline
|   |-- score_free_controller.py    # LLM-driven pipeline with ordinal controller
|   |-- research_synth.concept.yaml
|   \-- code_fix.concept.yaml
|-- providers.yaml         # model engines (OpenAI/Anthropic...)
\-- run.py                 # tiny runner (optional; CLI also works)
```

Concept blueprints with ordinal controller metadata still live as `.concept.yaml` files until the YAML loader supports those knobs.

`score_free_controller.py` wires everything together with real components:

* Planning agent via `make_agent_async` (requires an API key, e.g., `OPENAI_API_KEY`)
* Local documentation search tool that enriches badge support/evidence
* An ordinal controller that performs pairwise comparisons and exposes trace logs

### Providers (swap vendors without touching logic)

```yaml
# providers.yaml
providers:
  openai_default:    { engine: "openai:gpt-*",     temperature: 0.2 }
  anthropic_safe:    { engine: "anthropic:claude-*", temperature: 0.1 }
```

---

## Evidence badges (build signals, not a single score)

```python
# Experiments/test/app/evidence.py
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, MutableMapping

_ACCUMULATING_FIELDS = {
    "support",
    "contradict",
    "verify_hits",
    "cost_tokens",
    "tool_cost_usd",
}


@dataclass
class Badges:
    constraints_ok: bool = True
    support: int = 0
    contradict: int = 0
    quality_avg: float = 0.5
    recency: float = 0.5
    directness: float = 0.5
    verify_hits: int = 0
    cost_tokens: int = 0
    tool_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    summary = property(to_dict)

    def merged(self, other: Mapping[str, Any]) -> "Badges":
        merged_badges = replace(self)
        for key, value in other.items():
            if not hasattr(merged_badges, key):
                continue
            setattr(merged_badges, key, value)
        return merged_badges


def copy_badges(source: Badges | Mapping[str, Any] | None) -> Badges:
    if isinstance(source, Badges):
        return replace(source)
    if isinstance(source, Mapping):
        known: Dict[str, Any] = {}
        for key in Badges().__dict__.keys():
            if key in source:
                known[key] = source[key]
        return Badges(**known)
    return Badges()


def update_badges(badges: Badges, deltas: Mapping[str, Any]) -> Badges:
    updated = replace(badges)
    for key, value in deltas.items():
        if not hasattr(updated, key):
            continue
        current = getattr(updated, key)
        if isinstance(current, bool):
            setattr(updated, key, bool(value))
            continue
        if (
            isinstance(current, (int, float))
            and isinstance(value, (int, float))
            and key in _ACCUMULATING_FIELDS
        ):
            setattr(updated, key, current + value)
            continue
        setattr(updated, key, value)
    return updated


def badge_diff(base: Badges, other: Badges) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in base.__dict__.keys():
        left = getattr(base, key)
        right = getattr(other, key)
        if left != right:
            diff[key] = right
    return diff


def ensure_badges(container: MutableMapping[str, Any], key: str = "badges") -> Badges:
    existing = container.get(key)
    badges = copy_badges(existing)
    container[key] = badges
    return badges
```

## Comparator (pairwise; lexicographic, human-readable)

```python
# Experiments/test/app/comparators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from Experiments.test.app.evidence import Badges, copy_badges


@dataclass
class ComparisonRecord:
    winner: int
    reason: str
    details: Dict[str, Any]


def compare(a: Badges | Dict[str, Any], b: Badges | Dict[str, Any]) -> int:
    lhs = copy_badges(a)
    rhs = copy_badges(b)

    hard_rules = _evaluate_hard_rules(lhs, rhs)
    if hard_rules != 0:
        return hard_rules

    corroboration = _compare_support(lhs, rhs)
    if corroboration != 0:
        return corroboration

    quality = _compare_quality(lhs, rhs)
    if quality != 0:
        return quality

    verification = _compare_verify_hits(lhs, rhs)
    if verification != 0:
        return verification

    return _compare_cost(lhs, rhs)


def compare_with_reason(a: Badges | Dict[str, Any], b: Badges | Dict[str, Any]) -> ComparisonRecord:
    lhs = copy_badges(a)
    rhs = copy_badges(b)

    hard = _evaluate_hard_rules(lhs, rhs)
    if hard != 0:
        return ComparisonRecord(
            winner=hard,
            reason="constraints_ok priority",
            details={
                "a_constraints_ok": lhs.constraints_ok,
                "b_constraints_ok": rhs.constraints_ok,
            },
        )

    corroboration = _compare_support(lhs, rhs)
    if corroboration != 0:
        return ComparisonRecord(
            winner=corroboration,
            reason="support minus contradiction",
            details={
                "a_support": lhs.support,
                "a_contradict": lhs.contradict,
                "b_support": rhs.support,
                "b_contradict": rhs.contradict,
            },
        )

    quality = _compare_quality(lhs, rhs)
    if quality != 0:
        return ComparisonRecord(
            winner=quality,
            reason="quality plus recency plus directness",
            details={
                "a_quality": lhs.quality_avg,
                "a_recency": lhs.recency,
                "a_directness": lhs.directness,
                "b_quality": rhs.quality_avg,
                "b_recency": rhs.recency,
                "b_directness": rhs.directness,
            },
        )

    verification = _compare_verify_hits(lhs, rhs)
    if verification != 0:
        return ComparisonRecord(
            winner=verification,
            reason="verification hits",
            details={
                "a_verify_hits": lhs.verify_hits,
                "b_verify_hits": rhs.verify_hits,
            },
        )

    cost = _compare_cost(lhs, rhs)
    return ComparisonRecord(
        winner=cost,
        reason="total cost",
        details={
            "a_cost_tokens": lhs.cost_tokens,
            "a_tool_cost_usd": lhs.tool_cost_usd,
            "b_cost_tokens": rhs.cost_tokens,
            "b_tool_cost_usd": rhs.tool_cost_usd,
        },
    )


def _evaluate_hard_rules(lhs: Badges, rhs: Badges) -> int:
    if lhs.constraints_ok and not rhs.constraints_ok:
        return 1
    if rhs.constraints_ok and not lhs.constraints_ok:
        return -1
    return 0


def _compare_support(lhs: Badges, rhs: Badges) -> int:
    score_a = lhs.support - lhs.contradict
    score_b = rhs.support - rhs.contradict
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def _compare_quality(lhs: Badges, rhs: Badges) -> int:
    score_a = lhs.quality_avg + lhs.recency + lhs.directness
    score_b = rhs.quality_avg + rhs.recency + rhs.directness
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    return 0


def _compare_verify_hits(lhs: Badges, rhs: Badges) -> int:
    if lhs.verify_hits > rhs.verify_hits:
        return 1
    if rhs.verify_hits > lhs.verify_hits:
        return -1
    return 0


def _compare_cost(lhs: Badges, rhs: Badges) -> int:
    cost_a = lhs.cost_tokens + lhs.tool_cost_usd
    cost_b = rhs.cost_tokens + rhs.tool_cost_usd
    if cost_a < cost_b:
        return 1
    if cost_b < cost_a:
        return -1
    return 0
```

---

## Pipeline (YAML) -- score-free controller + HITL

```python
# Experiments/test/pipelines/score_free.py
from flujo.domain.dsl import Pipeline, Step
from Experiments.test.app.comparators import compare_with_reason
from Experiments.test.app.evidence import Badges, update_badges

async def plan(goal: str) -> dict[str, object]:
    base = Badges()
    options = [
        {"id": "exploratory", "summary": f"Explore framings for '{goal}'.", "badges": update_badges(base, {"support": 1}).to_dict()},
        {"id": "precise", "summary": f"Deliver focused answer on '{goal}'.", "badges": update_badges(base, {"support": 2, "quality_avg": 0.7}).to_dict()},
        {"id": "lean", "summary": f"Baseline answer for '{goal}'.", "badges": update_badges(base, {"support": 1, "cost_tokens": -20}).to_dict()},
    ]
    return {"goal": goal, "options": options, "log": []}

async def duel_options(payload: dict[str, object]) -> dict[str, object]:
    options = payload["options"]
    current = options[0]
    for contender in options[1:]:
        record = compare_with_reason(contender["badges"], current["badges"])
        payload["log"].append(
            f"compare({contender['id']} vs {current['id']}): winner="
            f"{contender['id'] if record.winner > 0 else current['id']} via {record.reason}"
        )
        if record.winner > 0:
            current = contender
    payload["selected"] = current
    payload["selected_badges"] = current["badges"]
    return payload

async def verify_selection(payload: dict[str, object]) -> dict[str, object]:
    badges = Badges(**payload["selected_badges"])
    payload["selected_badges"] = update_badges(badges, {"verify_hits": 1, "support": 1}).to_dict()
    payload["log"].append("verify: +support (>=2 domains), +verification hit")
    return payload

async def finalize(payload: dict[str, object]) -> dict[str, object]:
    return {
        "goal": payload["goal"],
        "selection": payload["selected"]["summary"],
        "badge_snapshot": payload["selected_badges"],
        "duel_log": payload["log"],
    }

pipeline = Pipeline(
    steps=[
        Step.from_callable(plan, name="plan"),
        Step.from_callable(duel_options, name="duel_options"),
        Step.from_callable(verify_selection, name="verify_selection"),
        Step.from_callable(finalize, name="finalize"),
    ]
)

__all__ = ["pipeline"]
```

### Controller pipeline (Python DSL with tools + LLM)

```python
# Experiments/test/pipelines/score_free_controller.py
from flujo.agents import make_agent_async
from flujo.domain.dsl import Pipeline, Step
from Experiments.test.app.controllers import OrdinalController, OrdinalControllerConfig
from Experiments.test.app.models import PlanResponse
from Experiments.test.app.tools import search_local_docs

plan_agent = make_agent_async("openai:gpt-4o-mini", PLAN_PROMPT, PlanResponse, temperature=0.2)

async def gather_evidence(plan: PlanResponse) -> dict[str, object]:
    # run doc search, update badges, attach evidence per candidate
    ...

async def run_ordinal_controller(state: dict[str, object]) -> dict[str, object]:
    controller = OrdinalController(config=OrdinalControllerConfig(min_support=3))
    result = controller.run(state["candidates"])
    return {"goal": state["goal"], "winner": result.winner, "controller_log": result.log}

pipeline = Pipeline([
    Step.model_validate({"name": "Plan", "agent": plan_agent}),
    Step.from_callable(gather_evidence, name="GatherEvidence"),
    Step.from_callable(run_ordinal_controller, name="OrdinalController"),
    Step.from_callable(finalize_output, name="Finalize"),
])
```

### Verifier and HITL trigger (sketch)

```python
# Experiments/test/app/verifiers.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping

from Experiments.test.app.evidence import Badges, copy_badges, ensure_badges, update_badges

JsonDict = Dict[str, Any]
_MISSING = object()


class VerifyResult(dict):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


def generic(context: Any) -> VerifyResult:
    badges = _consume_badges(context)
    domains = set(_iterable_from_context(context, "domains"))
    extra_support = 1 if len(domains) >= 2 else 0
    checklist_ok = bool(_resolve_path(context, ["output", "checklist_ok"], False))
    citations_ok = bool(_resolve_path(context, ["output", "has_required_citations"], True))
    updated = update_badges(
        badges,
        {
            "support": extra_support,
            "verify_hits": int(checklist_ok),
            "constraints_ok": citations_ok,
        },
    )
    _store_badges(context, updated)
    return VerifyResult(badges=updated.to_dict())


def hitl_gate_or_nudge(context: Any) -> JsonDict:
    badges = _consume_badges(context)
    risk = float(_resolve_path(context, ["meta", "action_risk"], 0.4))
    uncertainty = float(_resolve_path(context, ["meta", "uncertainty"], 0.6))
    product = risk * uncertainty
    _append_trace(context, {"hitl_score": product, "risk": risk, "uncertainty": uncertainty})

    if _context_policy(context, "hard_gate"):
        decision = {"hitl": "gate"}
    elif product >= _hitl_threshold(context, default=0.35):
        decision = {"hitl": "nudge", "options": ["Option A", "Option B", "Default"]}
    else:
        decision = {"hitl": "none"}

    _store_badges(context, badges)
    return decision


def _consume_badges(context: Any) -> Badges:
    if isinstance(context, MutableMapping):
        return ensure_badges(context)
    existing = getattr(context, "badges", None)
    return copy_badges(existing)


def _store_badges(context: Any, badges: Badges) -> None:
    if isinstance(context, MutableMapping):
        context["badges"] = badges
    else:
        setattr(context, "badges", badges)


def _iterable_from_context(context: Any, key: str) -> Iterable[Any]:
    if isinstance(context, Mapping):
        value = context.get(key, [])
    else:
        value = getattr(context, key, [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return value
    return []


def _resolve_path(context: Any, path: Iterable[Any], default: Any) -> Any:
    current: Any = context
    for part in path:
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _append_trace(context: Any, entry: JsonDict) -> None:
    trace = None
    if isinstance(context, Mapping):
        trace = context.get("trace")
    else:
        trace = getattr(context, "trace", None)
    if hasattr(trace, "add"):
        trace.add(entry)
    elif hasattr(trace, "append"):
        trace.append(entry)


def _context_policy(context: Any, policy_name: str) -> bool:
    policy_attr = None
    if hasattr(context, "policy"):
        policy_attr = context.policy
    elif isinstance(context, Mapping):
        policy_attr = context.get("policy")
    if callable(policy_attr):
        return bool(policy_attr(policy_name))
    return False


def _hitl_threshold(context: Any, default: float) -> float:
    cfg = None
    if isinstance(context, Mapping):
        cfg = context.get("cfg")
    else:
        cfg = getattr(context, "cfg", None)
    if cfg is None:
        return default
    threshold = _resolve_path(cfg, ["hitl", "threshold", "value"], default)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return default
```

---

## CLI usage (developer flow)

```bash
# run with OpenAI provider (Python DSL pipeline)
flujo run Experiments/test/pipelines/score_free.py --goal "Write a balanced brief with citations on X"

# inspect decisions & HITL
flujo lens trace <run_id>

# swap vendor (concept YAML, requires controller support in loader)
flujo run Experiments/test/pipelines/score_free.py --override provider=anthropic_safe

# tune search (bigger frontier)
flujo run Experiments/test/pipelines/score_free.py --override controller.ordinal.max_frontier=40
```

The override examples target future YAML-first controllers; the current Python pipeline ignores those flags and simply demonstrates the badge comparator + verifier flow.

If you want the fully wired experiment (LLM planner + doc search tool + ordinal controller), run:

```bash
# requires an API key (e.g., export OPENAI_API_KEY=sk-...)
python Experiments/test/run.py --pipeline score_free_controller.py --goal "Design a tutorial for Flujo evidence-first reasoning"
```

---

## Two ready-to-adapt recipes

### 1) Research Synthesis (concept; nudge, non-blocking)

```yaml
# Experiments/test/pipelines/research_synth.concept.yaml
pipeline:
  name: research-synth
  provider: anthropic_safe
  controller: { type: ordinal, ordinal: { tournament_k: 4, max_frontier: 16, stop_rules: { min_support: 3 }}}
  hitl: { mode: nudge, threshold: { rule: product, value: 0.3 }, timeout_s: 20, default_action: proceed }
  steps:
    - plan:     { prompt: "Subquestions + source types." }
    - retrieve: { tools: [web.search], with: { domain_whitelist: [".gov", ".edu", ".org"] } }
    - act:      { tool: writer.compose, args: { include_citations: true } }
    - verify:   { fn: Experiments.test.app.verifiers.citations }    # sets badges.support / constraints_ok
    - decide:   { fn: Experiments.test.app.verifiers.hitl_gate_or_nudge }
    - save:     { what: markdown, evidence }
```

### 2) Code Repair (concept; gate before merge)

```yaml
# Experiments/test/pipelines/code_fix.concept.yaml
pipeline:
  name: code-fix
  provider: openai_default
  controller: { type: ordinal, ordinal: { tournament_k: 5, max_frontier: 12, stop_rules: { min_support: 2 }}}
  hitl: { mode: gate }
  steps:
    - plan:   { prompt: "Summarize failing tests; propose 3 patch paths." }
    - retrieve:{ tools: [code.read_repo], with: { paths: ["src/", "tests/"] } }
    - act:    { tool: code.apply_patch, args_from: plan }
    - verify: { fn: Experiments.test.app.verifiers.run_pytests }      # fills verify_hits, constraints_ok
    - hitl:   { when: { step: verify }, show: ["patch_diff", "pytest_report"], options: ["Apply", "Revise", "Cancel"] }
    - save:   { what: patch, test_report }
```

---

## Non-blocking HITL in Flujo (how it feels)

* Pipeline sets `hitl.mode: nudge` + `timeout_s` + `default_action`.
* Flujo sends a compact card (summary, evidence, 2-4 choices).
* If no reply in time, it proceeds with the **default**, and records that fact in the trace.

---

## Metrics & tracing

Track these per run in Flujo's ledger:

* **Escalation precision**: % human prompts that changed the outcome
* **Verifier pass-rate**: before vs after nudge/gate
* **Cost per accepted result**: tokens + tools + human minutes
* **Time-to-acceptable**: latency to goal met

---

## Variants you can toggle

* **Minimax-regret** comparator (multiple stakeholder weights)
* **LCB** (mean - beta * stdev) if you estimate uncertainty per option
* **Beam->A*** escalation if frontier stalls (config in `controller:`)
* **Case-based planning**: retrieve & edit past winning traces before exploring

---

## Takeaway

In Flujo, evidence-first reasoning is just:

1. **Badges** (small, auditable signals),
2. an **ordinal controller** (pairwise compare), and
3. a **HITL router** (nudge/gate when Risk x Uncertainty is high).

You keep the same steps (plan / retrieve / act / verify / save); you only change **how** the next candidate is chosen and **when** to involve a human. The result is robust, explainable decisions without pretending a single number is truth.
