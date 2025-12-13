# Granular Execution Mode (Resumable Agents)

> **Crash-safe, multi-turn agent execution with automatic resume and progress persistence.**

Granular Execution Mode enables long-running agent conversations that can survive crashes, restarts, and interruptions. Each turn is atomically persisted, allowing exact resume without re-execution or double-billing.

## Quick Start

```python
from flujo import Step, Flujo
from flujo.agents import make_agent_async

# Create an agent
agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are a research assistant. Gather information iteratively.",
    output_type=str,
)

# Wrap in granular execution
pipeline = Step.granular("research_agent", agent, max_turns=20)

# Run - survives crashes!
runner = Flujo(pipeline)
async for result in runner.run_async("Research quantum computing breakthroughs"):
    print(result.output)
```

## How It Works

```text
┌─────────────────────────────────────────────────────────────┐
│                    Step.granular(...)                        │
│                           │                                  │
│                    compiles to                               │
│                           ▼                                  │
│              Pipeline(LoopStep(GranularStep))                │
│                           │                                  │
│    ┌──────────────────────┼──────────────────────┐          │
│    │                      │                      │          │
│    ▼                      ▼                      ▼          │
│  Turn 1 ───persist───► Turn 2 ───persist───► Turn 3 ...    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Each turn:
1. **Checks state** - Resume from last persisted turn if restarting
2. **Validates fingerprint** - Ensures same model/prompt/tools configuration
3. **Executes agent** - Runs one LLM call
4. **Persists atomically** - Saves state before returning
5. **Checks completion** - Agent output determines if done

## Key Features

### 🔒 Crash-Safe Resume
If your server crashes mid-conversation, restart and the pipeline resumes exactly where it left off:

```python
# First run - crashes after turn 5
result = await runner.run_async("complex task")  # Crash!

# After restart - resumes from turn 6
result = await runner.run_async("complex task")  # Continues!
```

### 🚫 No Double-Execution
CAS (Compare-And-Set) guards prevent running the same turn twice:
- On resume, completed turns are skipped
- No duplicate LLM calls = no double billing

### 🔑 Idempotency Keys
Each turn gets a deterministic idempotency key for safe tool retries:

```python
key = f"sha256({run_id}:{step_name}:{turn_index})"
# e.g., "a3f2c1..." - use this in your external API calls
```

### 📊 Fingerprint Validation
Resuming with different configuration fails fast:

```python
# Original run
pipeline = Step.granular("agent", agent_v1, ...)

# Later, trying to resume with different agent
pipeline = Step.granular("agent", agent_v2, ...)  
# Raises ResumeError: fingerprint mismatch
```

## Configuration Options

```python
pipeline = Step.granular(
    name="my_agent",
    agent=agent,
    
    # Maximum turns before forcing exit
    max_turns=20,
    
    # Token limit for conversation history
    history_max_tokens=128_000,
    
    # Offload payloads larger than this to blob storage
    blob_threshold_bytes=20_000,
    
    # Require idempotency keys for tool calls
    enforce_idempotency=False,
)
```

## Completion Detection

The agent signals completion by returning specific patterns:

```python
# Agent says "done" -> completion detected
agent = make_agent_async(
    system_prompt="When finished, respond with 'TASK COMPLETE'",
    ...
)
```

Detected completion patterns:
- `is_complete: true` in structured output
- `done: true` in output dict
- `"COMPLETE"` or `"DONE"` in string output

## Error Handling

### ResumeError
Raised when resume state is inconsistent:

```python
from flujo.domain.dsl import ResumeError

try:
    await runner.run_async(input_data)
except ResumeError as e:
    if e.irrecoverable:
        # Fingerprint mismatch - cannot resume
        # Must start fresh with new run_id
        pass
    else:
        # Temporary issue - retry
        pass
```

### Abort on Failure
Granular steps use `on_failure="abort"` - any error stops the loop immediately without partial state corruption.

## Best Practices

### 1. Use Structured Output
```python
from pydantic import BaseModel

class AgentResponse(BaseModel):
    content: str
    is_complete: bool
    
agent = make_agent_async(..., output_type=AgentResponse)
```

### 2. Design for Resumability
- Make prompts deterministic (no random seeds, timestamps)
- Use idempotency keys for external calls
- Keep tool responses JSON-serializable

### 3. Monitor Turn Count
```python
# Check progress in granular_state
(ctx.granular_state or {}).get("turn_index", 0)
```

## State Schema

The granular state is stored in `context.granular_state`:

```python
{
    "turn_index": 5,              # Completed turns
    "history": [...],             # Message history
    "is_complete": False,         # Completion flag
    "final_output": None,         # Result when complete
    "fingerprint": "abc123..."    # Config hash
}
```

## Blob Storage

Large payloads (>20KB by default) are offloaded to durable storage:

```text
Original: {"large_data": "...50KB of content..."}
Stored:   {"large_data": "<<FL_BLOB_REF:abc123:size=50000>>"}
```

On resume, blob references are automatically hydrated.

## See Also

- [Conversational Loops](../conversational_loops.md) - Alternative for simpler loops
- [Human-in-the-Loop](../hitl.md) - Pause for user input
- [Blueprints](../blueprints.md) - YAML configuration
