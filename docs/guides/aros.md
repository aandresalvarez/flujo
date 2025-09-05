# AROS: Adaptive Reasoning & Output System

AROS adds robust, opt‑in guardrails around pydantic‑ai agents in Flujo:

- Structured outputs (via pydantic‑ai wrapper)
- Adaptive Output Processing (AOP) for JSON extraction/repair/coercion
- Reasoning precheck (local checklist, validator agent, consensus gate)
- Health‑check CLI and trace events

By default, AROS is OFF (no behavior change). Enable it per step via `processing` keys in your YAML.

## Structured Outputs (pydantic‑ai)

Use `processing.structured_output` to opt‑in. The policy delegates to the wrapper, which attaches a best‑effort `response_format` hint when supported (e.g., OpenAI JSON Schema).

```yaml
processing:
  structured_output: openai_json  # off | auto | openai_json
  schema:
    type: object
    properties:
      ok: { type: boolean }
```

Notes:
- Wrapper passes a hint; pydantic‑ai/provider may ignore it when unsupported.
- Experimental `outlines|xgrammar` modes are telemetry‑only stubs (no runtime effect).

## Adaptive Output Processing (AOP)

Enable AOP to handle stringified/near‑valid JSON and safe coercions:

```yaml
processing:
  aop: full         # off | minimal | full
  coercion:
    tolerant_level: 0              # 0=off, 1=json5, 2=json-repair
    allow:
      integer: ["str->int"]
      number:  ["str->float"]
      boolean: ["str->bool"]
  schema:
    type: object
    required: ["count"]
    properties:
      count: { type: integer }
```

Stages:
- Stage 0: Extract largest balanced JSON/array; bounded unescape for double‑encoded JSON
- Stage 1: Tolerant decode (json/json5/json‑repair by config)
- Stage 2: Deterministic repair
- Stage 3: Schema‑aware coercion (anyOf/oneOf; allow‑listed safe conversions)

## Reasoning Precheck

Telemetry‑first checks before or around generation (opt‑in):

```yaml
processing:
  reasoning_precheck:
    enabled: true
    required_context_keys: ["initial_input"]
    validator_agent: agents.plan_checker
    score_threshold: 0.7
    inject_feedback: prepend           # prepend | context_key | (unset)
    retry_guidance_prefix: "Guidance: "
    context_feedback_key: "_aros_retry_guidance"
    # Optional consensus gate
    consensus_agent: agents.plan_sampler
    consensus_samples: 2
    consensus_threshold: 0.7
```

Events:
- `aros.reasoning.precheck.pass|fail`
- `reasoning.validation {result, score?}` for validator/consensus gates

## Health‑check CLI

Analyze AROS telemetry from recent runs:

```bash
uv run flujo dev health-check --since-hours 24 --limit 100 \
  --trend-buckets 6 --step stepA --model openai:gpt-4o \
  --export json --output aros_report.json

Note: Use `--output -` to print JSON to stdout instead of writing to a file.
```

Outputs totals, top steps/models, top transforms, and recommendations (including targeted hints for top offending step/model). With `--since-hours`, add `--trend-buckets N` to split the window into equal time buckets:

- Prints a compact bucketed coercion series and a top-stage summary across buckets.
- JSON export includes `trend.buckets` with:
  - `coercions`: total coercions in the bucket
  - `stages`: per-stage coercion counts in the bucket
  - `step_stages`: per-step stage distributions for the bucket
  - `model_stages`: per-model stage distributions for the bucket
  - The CLI also emits trend-based hints (e.g., top rising step/model between first and last buckets) and stage-specific trend guidance (e.g., tolerant/semantic/extract rising) in the recommendations.

## Grammar Enforcement (experimental)

Enable regex-based grammar checks for outlines/xgrammar to fail fast on malformed outputs:

```yaml
processing:
  structured_output: outlines   # or xgrammar
  enforce_grammar: true         # opt-in, off by default
  schema:
    type: object
```

Notes:
- Uses schema-derived regex (placeholders in current adapters) and validates the entire output string.
- Emits `grammar.enforce.pass|fail|skipped` events; skips on non-string outputs, missing schema, or compile errors.
- Keep `enforce_grammar` off unless outputs are consistently schema-shaped; combine with SOE where supported.

## AOP Extras

Install tolerant decoders as extras to enable `coercion.tolerant_level` tiers:

```bash
pip install .[aop-extras]
```

Then configure:

```yaml
processing:
  aop: full
  coercion:
    tolerant_level: 2   # 1=json5, 2=json-repair
```

## Good Defaults

- Keep AROS OFF globally; turn on per step where helpful.
- Start with `aop: minimal`; advance to `full` only when safe (strict validation remains).
- Use `structured_output: openai_json` with a schema when the model supports it.
- Add `required_context_keys` to catch obvious missing inputs early.
