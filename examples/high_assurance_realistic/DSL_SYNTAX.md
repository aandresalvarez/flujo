# Flujo DSL Syntax Examples

This document demonstrates ALL Flujo DSL syntax patterns used in this example.

## 1. @step Decorator

Define custom steps with automatic context injection:

```python
from flujo import step
from flujo.domain.models import PipelineContext

@step(name="load_abstracts")
async def load_abstracts_step(goal: str, *, context: PipelineContext) -> dict:
    """
    Load data for extraction.
    
    Args:
        goal: Input from previous step
        context: Automatically injected pipeline context
        
    Returns:
        Output for next step
    """
    # Store data in context
    context.total_abstracts = 6
    
    return {"text": "...", "pmid": "12345"}
```

**Key Features:**
- `@step(name="...")` decorator
- `*, context: PipelineContext` for automatic injection
- Return value becomes input to next step
- Can mutate context for cross-step communication

## 2. Pipeline >> Composition

Chain steps together with the >> operator:

```python
from flujo import Pipeline

# Simple pipeline
pipeline = step1 >> step2 >> step3

# Complex pipeline with TreeSearchStep
pipeline = (
    load_abstracts_step
    >> TreeSearchStep(...)
    >> format_results_step
)
```

**Key Features:**
- `>>` operator chains steps left-to-right
- Output of each step becomes input to next
- Context flows through entire pipeline
- Can mix different step types

## 3. Step.from_callable

Create steps from regular functions:

```python
from flujo import Step

async def preprocess_text(text: str, *, context: PipelineContext) -> str:
    """Clean text."""
    context.preprocessing_done = True
    return text.strip()

# Wrap in a Step
preprocess_step = Step.from_callable(
    preprocess_text,
    name="preprocess_text",
    updates_context=False,
)
```

**Key Features:**
- Wraps any async function
- `name` parameter for identification
- `updates_context` flag for optimization
- Useful for simple transformations

## 4. TreeSearchStep

Complex search step with invariants:

```python
from flujo.domain.dsl.tree_search import TreeSearchStep

search_step = TreeSearchStep(
    name="extraction",
    proposer=proposer_agent,           # LLM that generates candidates
    evaluator=evaluator_agent,         # LLM that scores candidates
    discovery_agent=discovery_agent,   # LLM that deduces invariants
    static_invariants=[...],           # Pre-defined rules
    branching_factor=3,
    beam_width=3,
    max_depth=4,
    goal_score_threshold=0.9,
)
```

**Key Features:**
- Specialized step for tree search
- Supports invariants (static + discovered)
- Built-in cost tracking
- Configurable search parameters

## 5. make_agent_async

Create LLM agents:

```python
from flujo import make_agent_async

agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="Extract triplets...",
    output_type=list[Triplet],
)
```

**Key Features:**
- `model` specifies provider and model
- `system_prompt` for instructions
- `output_type` for structured output
- Returns async callable

## 6. Context Injection

Access shared state across steps:

```python
@step(name="step1")
async def step1(input: str, *, context: PipelineContext) -> dict:
    # Write to context
    context.my_data = "value"
    return {"output": "..."}

@step(name="step2")
async def step2(input: dict, *, context: PipelineContext) -> str:
    # Read from context
    data = context.my_data  # "value"
    return f"Processed: {data}"
```

**Key Features:**
- `*, context: PipelineContext` parameter
- Automatically injected by Flujo
- Shared across all steps in pipeline
- Use for cross-step communication

## 7. Complete Example

Putting it all together:

```python
from flujo import step, Step, Pipeline, make_agent_async
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.models import PipelineContext

# 1. Define agents
proposer = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="...",
    output_type=list[Triplet],
)

# 2. Define custom steps
@step(name="load_data")
async def load_data(goal: str, *, context: PipelineContext) -> dict:
    context.loaded = True
    return {"data": "..."}

@step(name="format_output")
async def format_output(result: object, *, context: PipelineContext) -> dict:
    return {"formatted": result, "loaded": context.loaded}

# 3. Create preprocessing step
def create_preprocess() -> Step:
    async def preprocess(data: dict, *, context: PipelineContext) -> str:
        return data["data"].strip()
    
    return Step.from_callable(preprocess, name="preprocess")

# 4. Compose pipeline
pipeline = (
    load_data                          # @step decorator
    >> create_preprocess()             # Step.from_callable
    >> TreeSearchStep(                 # Complex step
        name="search",
        proposer=proposer,
        evaluator=evaluator,
        static_invariants=[...],
    )
    >> format_output                   # @step decorator
)

# 5. Execute
from flujo.application.runner import Flujo

runner = Flujo(pipeline, context_model=PipelineContext)
result = await runner.run_async("initial input")
```

## Summary Table

| Syntax | Use Case | Example |
|--------|----------|---------|
| `@step` | Custom processing steps | `@step(name="load")` |
| `>>` | Chain steps together | `step1 >> step2` |
| `Step.from_callable` | Wrap functions | `Step.from_callable(fn)` |
| `TreeSearchStep` | Tree search with invariants | `TreeSearchStep(...)` |
| `make_agent_async` | Create LLM agents | `make_agent_async(model=...)` |
| `*, context: PipelineContext` | Access shared state | `async def f(..., *, context)` |

## See Also

- `agents.py` - Complete implementation
- `main.py` - Usage examples
- Flujo documentation - Full API reference
