# High-Assurance Reasoning with Flujo

This guide demonstrates how to build **high-assurance reasoning systems** using Flujo's B-Method inspired invariant system. You'll learn how to prevent hallucinations, contamination, and logical errors using formal constraints that are checked at runtime.

## Table of Contents

1. [Quick Start](#quick-start)
2. [What Are Invariants?](#what-are-invariants)
3. [Static Invariants](#static-invariants)
4. [Discovery Agents](#discovery-agents)
5. [Cost Analysis](#cost-analysis)
6. [Advanced Patterns](#advanced-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

Run the comprehensive example:

```bash
cd examples
python high_assurance_med13_extractor.py
```

This demonstrates extracting gene-disease relationships while **preventing contamination** from paralog genes using invariants.

---

## What Are Invariants?

**Invariants** are formal rules that must **always** be true during execution. In Flujo's `TreeSearchStep`, invariants act as **guardrails** that:

1. **Pre-emptively prune** bad candidates before expensive LLM evaluation
2. **Backtrack** when a node violates a constraint
3. **Save costs** by avoiding doomed search branches
4. **Ensure correctness** through formal verification

### The Problem: Contamination in Knowledge Extraction

When extracting facts about the **MED13 gene**, an LLM might accidentally include facts about **MED13L** (a paralog gene with different functions). This contamination can:

- Corrupt your knowledge base
- Lead to incorrect medical conclusions
- Waste tokens evaluating fundamentally wrong answers

### The Solution: Invariants

```python
# Static invariant: "MED13L must never appear in output"
invariant = "'MED13L' not in str(output).upper()"
```

Flujo checks this invariant **before** calling the evaluator LLM. If violated, the branch is killed instantly (0ms, $0.00).

---

## Static Invariants

Static invariants are **pre-defined rules** that you write once and apply to all search nodes.

### Expression-Based Invariants

Use Python expressions as strings:

```python
from flujo.domain.dsl.tree_search import TreeSearchStep

step = TreeSearchStep(
    name="gene_extractor",
    proposer=my_proposer,
    evaluator=my_evaluator,
    static_invariants=[
        "'MED13L' not in str(output).upper()",  # No paralog contamination
        "len(output) > 0",                       # No empty results
        "'mutation' in str(output).lower()",     # Must mention mutations
    ],
)
```

**How it works:**
- Flujo compiles the expression into a callable
- The callable receives `(output, context)` as arguments
- Returns `True` if invariant holds, `False` if violated

### Callable-Based Invariants

For complex logic, use Python functions:

```python
def no_paralog_contamination(output: object, context: BaseModel | None = None) -> bool:
    """Invariant: Output must not mention any paralog genes."""
    output_str = str(output).upper()
    forbidden = ["MED13L", "MED12", "MED14"]
    return not any(gene in output_str for gene in forbidden)

step = TreeSearchStep(
    name="gene_extractor",
    proposer=my_proposer,
    evaluator=my_evaluator,
    static_invariants=[
        no_paralog_contamination,
        lambda out, ctx: len(str(out)) > 10,  # Lambdas work too
    ],
)
```

**Signature options:**
- `(output, context) -> bool` - Full signature
- `(output) -> bool` - Output only
- `(context) -> bool` - Context only
- `() -> bool` - No arguments

Flujo tries each signature until one works.

### When Are Invariants Checked?

Invariants are checked at **two points**:

1. **Before evaluation** (line 612-637 in `tree_search_policy.py`):
   - After proposer generates candidates
   - Before expensive evaluator LLM call
   - **Saves money** by pruning early

2. **After expansion** (line 435-467 in `tree_search_policy.py`):
   - When a node is popped from the open set
   - Before expanding to children
   - **Ensures consistency** as search progresses

---

## Discovery Agents

**Discovery agents** are LLMs that analyze your goal and **deduce invariants at runtime**. This allows the system to adapt to different tasks automatically.

### Basic Usage

```python
from flujo import make_agent_async

# Define a discovery agent
discovery_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="""
    Analyze the user's goal and extract hard invariants.
    Return a JSON array of Python expressions that must always be true.
    
    Example: ["'paralog' not in str(output).lower()", "len(output) > 0"]
    """,
    output_type=list[str],
)

# Use it in TreeSearchStep
step = TreeSearchStep(
    name="adaptive_extractor",
    proposer=my_proposer,
    evaluator=my_evaluator,
    discovery_agent=discovery_agent,  # Deduces rules at runtime
    static_invariants=[...],           # Baseline rules
)
```

### How It Works

1. **Before search begins**, Flujo calls the discovery agent with:
   ```
   Analyze the goal and extract hard invariants that must never be violated.
   Return a JSON array of invariant expressions or one rule per line.
   Primary Objective: Extract MED13 gene triplets, avoiding paralog MED13L
   ```

2. **Discovery agent returns** rules like:
   ```json
   [
     "'MED13L' not in str(output).upper()",
     "'13L' not in str(output)",
     "len(str(output)) > 10"
   ]
   ```

3. **Flujo stores** these in `context.tree_search_state.deduced_invariants`

4. **During search**, both static and deduced invariants are checked

### Benefits

- **Adaptability**: Different goals → different rules
- **Explainability**: See what rules the system deduced
- **Flexibility**: Combine with static invariants for defense-in-depth

### Example Output

```python
# After search completes
state = context.tree_search_state
print(f"Discovered invariants: {state.deduced_invariants}")
# Output: ["'MED13L' not in str(output).upper()", "'13L' not in str(output)", ...]
```

---

## Cost Analysis

Invariants save money by **avoiding expensive LLM calls** for doomed branches.

### Without Invariants

```
1. Proposer generates 3 candidates (including 1 with MED13L)
2. Evaluator scores all 3 candidates → 3 LLM calls
3. Evaluator detects MED13L contamination → score=0.0
4. Branch is pruned, but tokens already spent
```

**Cost**: 3 evaluator calls × ~500 tokens × $0.01/1K = **$0.015**

### With Static Invariants

```
1. Proposer generates 3 candidates (including 1 with MED13L)
2. Invariant check catches MED13L → branch killed (0ms)
3. Evaluator scores only 2 valid candidates → 2 LLM calls
```

**Cost**: 2 evaluator calls × ~500 tokens × $0.01/1K = **$0.010**

**Savings**: 33% reduction in evaluation costs

### Real-World Impact

On a 100-page biomedical paper:
- **Without invariants**: ~500 contaminated branches × $0.005/evaluation = **$2.50 wasted**
- **With invariants**: Contaminated branches caught instantly = **$0.00 wasted**
- **Total savings**: **$2.50 per paper**

At scale (1000 papers): **$2,500 saved**

---

## Advanced Patterns

### Combining Invariants with Custom Cost Functions

```python
def evidence_based_cost(candidate, parent, depth, evaluation):
    """Lower cost for candidates with strong evidence."""
    if "mutation" in str(candidate).lower():
        return depth * 0.5  # Prefer mutation evidence
    return depth * 1.0

step = TreeSearchStep(
    name="evidence_weighted_search",
    proposer=my_proposer,
    evaluator=my_evaluator,
    cost_function=evidence_based_cost,
    static_invariants=["'MED13L' not in str(output)"],
)
```

### Context-Aware Invariants

```python
def no_duplicate_facts(output: object, context: PipelineContext | None = None) -> bool:
    """Invariant: Don't extract the same fact twice."""
    if context is None:
        return True
    
    existing_facts = getattr(context, 'extracted_facts', [])
    new_fact = str(output)
    
    return new_fact not in existing_facts
```

### Invariant Violation Feedback

Violations are stored in `state.metadata['invariant_violations']` and passed to the proposer:

```python
# In your proposer prompt
"""
Previous attempts violated these invariants:
- 'MED13L' not in str(output).upper()
- len(output) > 0

Please propose candidates that satisfy all invariants.
"""
```

This creates a **feedback loop** where the proposer learns from violations.

---

## Troubleshooting

### Invariant Never Triggers

**Problem**: Your invariant is never violated, even when you expect it to be.

**Solutions**:
1. Check the invariant expression syntax:
   ```python
   # ❌ Wrong: Uses undefined variable
   "'MED13L' not in output"
   
   # ✅ Correct: Converts to string first
   "'MED13L' not in str(output)"
   ```

2. Verify the invariant is added to the step:
   ```python
   print(step.static_invariants)  # Should show your invariants
   ```

3. Check if discovery is disabled:
   ```python
   from flujo.infra.settings import get_settings
   settings = get_settings()
   print(settings.tree_search_discovery_enabled)  # Should be True
   ```

### Invariant Triggers Too Often

**Problem**: Invariant rejects all candidates, search finds nothing.

**Solutions**:
1. Make the invariant less strict:
   ```python
   # ❌ Too strict: Rejects partial matches
   "output.subject == 'MED13'"
   
   # ✅ Better: Allows flexibility
   "'MED13' in str(output)"
   ```

2. Use `candidate_validator` for lightweight checks:
   ```python
   step = TreeSearchStep(
       proposer=my_proposer,
       evaluator=my_evaluator,
       candidate_validator=lambda c: isinstance(c, dict) and 'subject' in c,
       static_invariants=[...],  # For semantic checks
   )
   ```

### Viewing Invariant Violations

```python
# After search completes
state = context.tree_search_state

# Get all violation events
violations = [e for e in state.trace if e['event'] == 'invariant_violation']

for v in violations:
    print(f"Node {v['node_id']} violated:")
    for violation in v['violations']:
        print(f"  - Rule: {violation['rule']}")
        print(f"  - Reason: {violation.get('reason', 'N/A')}")
```

### Performance Issues

**Problem**: Invariant checking is slow.

**Solutions**:
1. Use expression-based invariants (faster than callables):
   ```python
   # ✅ Fast: Compiled once
   "'MED13L' not in str(output)"
   
   # ❌ Slower: Function call overhead
   lambda out, ctx: 'MED13L' not in str(out)
   ```

2. Avoid expensive operations in invariants:
   ```python
   # ❌ Slow: Regex on every check
   "re.search(r'MED13[^L]', str(output)) is not None"
   
   # ✅ Fast: Simple string check
   "'MED13' in str(output) and 'MED13L' not in str(output)"
   ```

---

## Further Reading

- [TreeSearchStep API Reference](../docs/user_guide/pipeline_dsl.md#treesearchstep)
- [A* Search Cookbook](../docs/cookbook/reasoning_with_astar.md)
- [Biomedical A* Demo](biomedical_astar_agent_demo.py)
- [Game of 24 Example](../tests/integration/test_tree_search_game_of_24.py)

---

## Summary

| Feature | Purpose | Cost | When to Use |
|---------|---------|------|-------------|
| **Static Invariants** | Pre-defined rules | 0ms, $0.00 | Known constraints, domain rules |
| **Discovery Agents** | Runtime-deduced rules | 1 LLM call | Adaptive tasks, varying goals |
| **Candidate Validator** | Lightweight pre-filter | 0ms, $0.00 | Type checks, basic validation |
| **Evaluator `hard_fail`** | Semantic rejection | 1 LLM call | Complex quality checks |

**Best Practice**: Use all four layers for defense-in-depth:
1. Candidate validator (type safety)
2. Static invariants (domain rules)
3. Discovery agent (adaptive rules)
4. Evaluator (semantic quality)

This creates a **high-assurance reasoning system** that prevents errors at multiple levels while minimizing cost.
