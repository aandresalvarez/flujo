# Validation Examples (Good/Bad)

This folder contains minimal YAML blueprints that intentionally trigger (bad) or avoid (good) specific validation rules. Use these to learn patterns and to verify fixer behavior.

Templates

- V‑T1: previous_step.output misuse
  - templates/vt1_bad.yaml
  - templates/vt1_good.yaml
- V‑T3: common filter typos
  - templates/vt3_bad.yaml
  - templates/vt3_good.yaml

Schema

- V‑S2: structured output stringified downstream
  - schema/vs2_bad.yaml
  - schema/vs2_good.yaml

Context

- V‑C2: legacy mapping to removed scratchpad root (use import_artifacts.<key> instead)
  - context/vc2_bad.yaml
  - context/vc2_good.yaml

Orchestration

- V‑P2: explicit outputs conflict across parallel branches
  - orchestration/vp2_bad.yaml
  - orchestration/vp2_good.yaml
- V‑P3: heterogeneous branch input types
  - orchestration/vp3_bad.yaml
  - orchestration/vp3_good.yaml
- V‑L1: loop exit coverage
  - orchestration/vl1_bad.yaml
  - orchestration/vl1_good.yaml
- V‑CF1: unconditional infinite loop
  - orchestration/vcf1_bad.yaml
  - orchestration/vcf1_good.yaml

Agents

- V‑A8: structured output with non‑JSON response mode
  - agents/va8_bad.yaml
  - agents/va8_good.yaml

Imports

- V‑I2: mapping root sanity (unknown root)
  - imports/vi2_bad.yaml
  - imports/vi2_good.yaml

Schema

- V‑S3: string schema awareness
  - schema/vs3_bad.yaml
  - schema/vs3_good.yaml

Usage tips

- Run `flujo validate <file>.yaml --format text` to see warnings and suggestions.
- Try the fixers:
  - `--fix` to preview + apply; `--yes` to skip prompts.
  - `--fix-rules V-T1,V-T3,V-C2` to target specific fixers.
  - `--fix-dry-run` to preview the unified diff without writing.
