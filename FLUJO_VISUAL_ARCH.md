 
 # Flujo Visual Grammar (FVG) — Specification v1.2

**Status:** Draft for Implementation Review
**Date:** Aug 10, 2025
**Owner:** Flujo Development Team
**Applies to:** Flujo YAML/DSL → UI rendering pipeline

---

## 0. Conformance & Terminology

This document uses **MUST/SHOULD/MAY** as defined in RFC 2119. “Renderer” refers to the UI component (e.g., React Flow) that consumes the transport graph. “Engine” refers to Flujo’s runtime. “Transport” refers to the JSON payload produced by serializing a Python `Pipeline` (the *plan*) and optionally a `Run` (the *run*time state).

**Single Source of Truth:** The **YAML DSL** defines pipelines. The in‑memory DSL graph is canonical. FVG is a **rendering target** (transport for visualization), not a definition or authoring format.

---

## 1. Conceptual Model

FVG represents a **Hierarchical Dataflow + Control Graph (H‑DCG)**:

* **Dataflow** (left→right) carries typed values between **ports** on **nodes**.
* **Control** (dashed) determines execution paths, policies, approvals, retries.
* **Hierarchy** allows any subgraph (pipeline) to be treated as a node (algebraic closure). Nodes can **expand/collapse**.

---

## 2. Core Principles

1. **Data Flow Primacy (MUST):** Layout emphasizes left→right data movement.
2. **Explicit Control (MUST):** Control edges are distinct, labeled, and never carry data.
3. **Typed Ports (MUST):** All data edges connect **output\:Type** → **input\:Type**; types must be compatible.
4. **Hierarchy & Composability (MUST):** Composite nodes render nested graphs; collapsed view renders as an atomic node.
5. **Progressive Disclosure (SHOULD):** Default view is uncluttered; details revealed by interaction.
6. **Semantic Shapes & Icons (SHOULD):** Shape encodes semantics; color is supplemental.

---

## 3. Visual Grammar

### 3.1 Node Kinds

| Kind            | Shape                                | Icon | Semantics                                   |
| --------------- | ------------------------------------ | ---- | ------------------------------------------- |
| `step`          | Rounded rectangle                    | —    | Atomic function/transform.                  |
| `agent`         | Rounded rectangle                    | 🤖   | LLM/tool call (model in attrs).             |
| `pipeline`      | Rounded rectangle, **dashed border** | 📁   | Composite subgraph (expandable).            |
| `parallel`      | Thick vertical bar (split)           | ⚡    | Fan‑out; pairs with `parallel-join`.        |
| `parallel-join` | Thick vertical bar (join)            | ⚡    | Fan‑in; aggregation semantics.              |
| `conditional`   | Diamond                              | 🔀   | Predicate‑based routing.                    |
| `loop`          | Container w/ loop badge              | 🔄   | Repeated subgraph; termination rules.       |
| `policy`        | Hexagon                              | 🛡️  | Budget/rate/guardrail evaluation; outcomes. |
| `artifact`      | Document/cylinder                    | 💾   | Persisted output; versionable.              |
| `human`         | Parallelogram                        | 👤   | Human‑in‑the‑loop interaction.              |
| `repair`        | Rounded rectangle, red accent        | 🛠️  | Automated repair/recovery path.             |

**Attributes (per node) — required unless noted:**

* `name` (MUST), `description` (MAY)
* `ports.in[]`, `ports.out[]` (MUST for dataflow nodes); each port: `{id, type, label?}`
* `attrs` (MAY) — kind‑specific (see §6)
* `a11y.label` (SHOULD)
* `run` (only in **run** view; see §5)

### 3.2 Edge Kinds

| Kind       | Style       | Carries        | Notes                                                       |         |            |
| ---------- | ----------- | -------------- | ----------------------------------------------------------- | ------- | ---------- |
| `data`     | Solid line  | Values (typed) | Label MAY show `Type` or field alias.                       |         |            |
| `control`  | Dashed line | Signals        | Label MUST indicate outcome (`ok`, `else`, `breach`, etc.). |         |            |
| `fallback` | Dotted line | Signals        | From a node’s *bottom*; reason=\`error                      | timeout | invalid\`. |

**Edge Decorators (MAY):** `policy` (micro‑budget on an edge), `checkpoint` markers, `deadline`.

### 3.3 Annotations (Run View)

* **Status ring:** `running|success|failure|warning|skipped`.
* **Metric chips:** `tokens`, `$`, `lat(ms)`, `retries`.
* **OTel correlation:** `trace_id`, `span_id` (SHOULD if telemetry enabled).
* **Context update badge:** `C↑` when `updates_context=True`.

---

## 4. Layout Rules

* **Direction:** Left→Right for data; control edges MAY route top→bottom.
* **Alignment:** Fork (`parallel`) and `parallel-join` MUST be vertically aligned.
* **Crossings:** Renderer SHOULD minimize crossings (ELK/Dagre).
* **Placement:** Place `policy` nodes immediately upstream of expensive `agent` nodes.
* **Nesting:** Expanded composites visually contain children; collapsed composites render a summary header (ports MUST remain visible).

---

## 5. Views: Plan vs Run

**Plan View** presents definition only.

* Excludes `run` blocks.
* MAY include `layout_hints` and `schemas` refs.

**Run View** enriches with runtime state to enable time‑scrub/replay.

* Each node MAY include `run = {status, latency_ms, cost_usd, retries, error_category?, timestamp_in?, timestamp_out?, telemetry_id?}`.
* Each edge MAY include `{events:[{ts, kind}...]}`.
* **Time Scrubber (SHOULD):** Renderer supports replay by timestamps.

---

## 6. Control Semantics (Normative)

### 6.1 Parallel

* `parallel` MUST declare `join_mode`:

  * `barrier` (default): all branches complete.
  * `any`: first completed branch short‑circuits others (cancel policy: `graceful|immediate`).
  * `quorum(n)`: any `n` branches complete.
* **Join:** A matching `parallel-join` MUST exist (explicit or implied). If omitted, renderer MUST insert an implied join.
* **Aggregation (SHOULD):** `aggregation = concat|reduce(function)|first|best(metric)`.
* **Error handling:** `on_error = continue|fail|quorum`.

### 6.2 Conditional

* Outgoing control edges MUST have **unique** `label` predicates.
* If no predicate matches and `else` exists, follow `else`; otherwise node outcome is `skipped`.

### 6.3 Loop

* MUST declare `termination`:

  * `{kind: "max_iters", value: <int>}` (MUST be finite), or
  * `{kind: "predicate"}`, or `{kind: "convergence", metric, threshold}`.
* MAY declare `carry_state: bool` and `accumulator` semantics.
* **Loop‑body** is a composite subgraph.

### 6.4 Policy Gate

* MUST expose outcomes `ok` and `breach` (MAY expose `throttle`).
* On `breach`, control MUST route to a recovery path (`repair` or `human`).
* Policy attrs MAY include `budget_usd`, `token_limit`, `rate_limit`, `circuit_breaker`.

### 6.5 Fallback

* Any node MAY declare a `fallback` target edge with `reason`.
* **Type Compatibility (MUST):** Fallback path output type MUST be compatible with the failed node’s declared output type(s).

### 6.6 Human‑in‑the‑Loop

* MUST declare `interaction.kind = approve|edit|route` and `blocking = true|false`.
* MAY declare `sla_ms`, `escalate_to`, `default_action`.

---

## 7. Transport Schema (Backend → UI)

The transport is a **render‑only** JSON payload. Major version changes are **breaking**.

```jsonc
{
  "graph_version": "1.2",
  "view": "plan",                    // or "run"
  "pipeline_id": "pipe_9f3c...",
  "title": "Customer Triage",
  "schemas": {                        // optional JSON Schemas for port types
    "#/types/Query": {"type":"object","properties":{"text":{"type":"string"}}}
  },
  "layout_hints": {                   // optional
    "rankdir": "LR",
    "groups": [{"id":"expensive","nodes":["g1"],"label":"High Cost"}]
  },
  "nodes": [
    {
      "id": "s1",
      "kind": "step",
      "name": "Parse Query",
      "ports": {"in":[{"id":"in","type":"#/types/Query"}], "out":[{"id":"query","type":"#/types/Query"}]},
      "attrs": {"updates_context": false}
    },
    {
      "id": "g1",
      "kind": "agent",
      "name": "Retrieve",
      "ports": {"in":[{"id":"q","type":"#/types/Query"}], "out":[{"id":"docs","type":"Doc[]"}]},
      "attrs": {"model":"openai:o4-mini"},
      "run": {                         // present only when view=run
        "status":"success",
        "latency_ms":247,
        "cost_usd":0.0034,
        "retries":1,
        "telemetry_id":"trace_abc"
      }
    },
    {"id":"pol1","kind":"policy","name":"Budget Gate","ports":{"in":[{"id":"in","type":"Doc[]"}],"out":[{"id":"ok","type":"Doc[]"}]},
      "attrs":{"budget_usd":5,"token_limit":100000}},
    {"id":"fork1","kind":"parallel","name":"Fan Out","attrs":{"join_mode":"barrier","aggregation":"concat"}},
    {"id":"join1","kind":"parallel-join","name":"Join"},
    {"id":"loop1","kind":"loop","name":"Refine","attrs":{"termination":{"kind":"max_iters","value":5}}},
    {"id":"out1","kind":"artifact","name":"Report","attrs":{"format":"markdown","version":"v3","preview_url":"/artifacts/123"}}
  ],
  "edges": [
    {"id":"e1","from":"s1:query","to":"g1:q","kind":"data","type":"#/types/Query"},
    {"id":"e2","from":"g1:docs","to":"pol1:in","kind":"data","type":"Doc[]"},
    {"id":"e3","from":"pol1:ok","to":"fork1","kind":"control","label":"ok"},
    {"id":"e4","from":"fork1","to":"join1","kind":"data","fanout":2},
    {"id":"e5","from":"join1","to":"loop1","kind":"data"},
    {"id":"e6","from":"loop1","to":"out1","kind":"data"},
    {"id":"e7","from":"g1","to":"repair1","kind":"fallback","reason":"timeout"}
  ],
  "checkpoints": [
    {"id":"ck_12","edge":"e5","seq":4,"ts":"2025-08-10T18:01:23Z"}
  ],
  "a11y": {"labels": {"s1":"Parses raw query into structured form"}}
}
```

**Transport Requirements:**

* **IDs (MUST):** Stable per pipeline version. Recommended: `id = base32(blake3(pipeline_id + node_path))`.
* **Port Declaration (MUST):** All data edges reference existing port IDs.
* **Types (SHOULD):** Use `$ref` to shared schemas where possible.
* **Unknown Major Versions (MUST):** Renderer refuses to render.

---

## 8. Identity, Paths & Stability

* Each node carries an implicit `node_path` (root→…→node) in the DSL. The serializer SHOULD embed `node_path` for debugging.
* Renames MUST preserve IDs across versions unless structure changes materially (documented in release notes).

---

## 9. Observability, Telemetry & Traces

* When OpenTelemetry is enabled, nodes and edges SHOULD include correlation IDs (`trace_id`, `span_id`).
* Renderer MAY offer one‑click deep‑link to span in tracing UI.
* Metrics overlays SHOULD allow toggling `tokens/$/latency/retries`.

---

## 10. Checkpoints & Resume Semantics

* `checkpoints[]` MAY annotate edges with resume points.
* Engine MUST resume from the **nearest upstream** checkpoint when instructed by the user.
* Renderer SHOULD provide a context menu to “Resume from here”.

---

## 11. Security, Privacy & Compliance Overlays

* Nodes/ports MAY include `data_classification = public|internal|confidential|restricted`.
* Renderer SHOULD display PII badges and prevent export of restricted artifacts without confirmation.

---

## 12. Accessibility & Theming

* Shapes encode semantics independent of color (MUST).
* Minimum contrast 4.5:1 (SHOULD). Provide high‑contrast & color‑blind palettes.
* ARIA roles/labels for nodes and edges (SHOULD).

---

## 13. Export & Interchange

* Renderer SHOULD support export to **SVG/PNG** and to **Mermaid** (via `to_mermaid()` parity).
* Back‑export to canonical `.flujo.yml` MUST preserve semantics (no lossy transforms).

---

## 14. Extensibility

* Custom node kinds MAY be introduced under a `x-` namespace (e.g., `x-widget`).
* Reserved keywords: `step, agent, pipeline, parallel, parallel-join, conditional, loop, policy, artifact, human, repair`.

---

## 15. Example Mermaid (Docs)

```mermaid
flowchart LR
  subgraph Pipeline[Customer Triage]
    S1[Step: Parse] -->|Query| A1([Agent: Retrieve])
    A1 --> P1{{Policy Gate}}
    P1 -- ok --> ||Fork||
    ||Fork|| --> ||Join||
    ||Join|| --> L1[[Loop: Refine]]
    L1 --> OUT[(Artifact: Report)]
    P1 -. breach .-> R1[Repair]
  end
```

---

## 16. Implementation Checklists

### Backend

* [ ] `pipeline.to_fvg(view="plan|run")` serializer (stable IDs, ports, edges, schemas).
* [ ] `/pipelines/:id/fvg?view=plan` and `/runs/:id/fvg?view=run` endpoints.
* [ ] Optional `/pipelines/:id/mermaid` for documentation.
* [ ] Release notes for breaking graph\_version changes; migration guide.

### UI/Renderer

* [ ] React Flow custom nodes per **kind**; port rendering with type tooltips.
* [ ] ELK/Dagre layout (LR), fork/join alignment, crossing minimization.
* [ ] Plan/Run toggle; time scrubber for replay; OTel deep‑linking.
* [ ] Overlays: metrics chips, status rings, policy outcomes; hide/show toggles.
* [ ] Export SVG/PNG; back‑export `.flujo.yml`.
* [ ] A11y: ARIA roles, keyboard navigation, high‑contrast theme.

---

## 17. Change Log

* **v1.2 (2025‑08‑10):** Added normative control semantics; plan/run split; first‑class ports/edges; identity/versioning; telemetry, security, a11y; full transport schema.
* **v1.1 (2023‑10‑27):** Established FVG as rendering target of YAML/DSL; initial grammar components.

---

## 18. Future Extensions (Non‑Normative)

* **Edge‑level policies** (micro‑budgets) with enforcement stats.
* **Reduction nodes** with library of aggregators.
* **Type unions & generics** across ports (e.g., `Result<T>`).
* **Schema‑aware diffing** of pipelines between versions.
