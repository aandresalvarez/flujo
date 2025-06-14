# API Reference

This guide provides detailed documentation for all public interfaces in `pydantic-ai-orchestrator`.

## Core Components

### Orchestrator

The `Orchestrator` class is a high-level facade for running a **standard, fixed
multi-agent pipeline**: Review -> Solution -> Validate. It uses the agents you
provide for these roles. For custom pipelines with different logic, see
`PipelineRunner` and the `Step` DSL.

```python
from pydantic_ai_orchestrator import Orchestrator

orchestrator = Orchestrator(
    review_agent: AsyncAgentProtocol[Any, Any],
    solution_agent: AsyncAgentProtocol[Any, Any],
    validator_agent: AsyncAgentProtocol[Any, Any],
    reflection_agent: Optional[AsyncAgentProtocol[Any, Any]] = None,
    max_iters: Optional[int] = None,
    k_variants: Optional[int] = None,
    reflection_limit: Optional[int] = None,
)
```

#### Methods

```python
# Run a task synchronously
result = orchestrator.run_sync(Task(prompt="Generate a poem"))

# Run a task asynchronously
candidate = await orchestrator.run_async(Task(prompt="Generate a poem"))
```

### Pipeline DSL & `PipelineRunner`

The Pipeline DSL lets you create flexible, custom workflows and execute them
with `PipelineRunner`.

```python
from pydantic_ai_orchestrator import (
    Step, PipelineRunner, Task,
    review_agent, solution_agent, validator_agent,
)
from pydantic import BaseModel

class MyContext(BaseModel):
    counter: int = 0

# Create a pipeline
custom_pipeline = (
    Step.review(review_agent)      # Review step
    >> Step.solution(              # Solution step
        solution_agent,
        tools=[tool1, tool2]       # Optional tools
    )
    >> Step.validate(              # Validation step
        validator_agent,
        criteria=["quality", "correctness"]
    )
)

runner = PipelineRunner(custom_pipeline)
# With a shared typed context
runner_with_ctx = PipelineRunner(
    custom_pipeline,
    context_model=MyContext,
    initial_context_data={"counter": 0},
)
```

#### Methods

```python
# Run the pipeline
pipeline_result = runner.run(
    "Your initial prompt"  # Input for the first step
)  # Returns PipelineResult

for step_res in pipeline_result.step_history:
    print(step_res.name, step_res.success)

# Inspect pipeline structure
structure = custom_pipeline.structure()

# Access runner config
config = runner.get_config()

# Access final pipeline context
final_ctx = pipeline_result.final_pipeline_context
```

### Agents

Agent creation and configuration.

```python
from pydantic_ai_orchestrator import make_agent_async

# Create an agent
agent = make_agent_async(
    model: str,                    # Model identifier
    system_prompt: str,            # System prompt
    output_type: type,             # Output type
    tools: list[Tool] = None,      # Optional tools
    temperature: float = 0.7,      # Model temperature
    max_tokens: int = 1000,        # Max tokens per response
    timeout: int = 30              # Operation timeout
)
```

#### Methods

```python
# Run the agent
result = await agent.run(
    prompt: str,                   # Task prompt
    metadata: dict = None,         # Optional metadata
    constraints: dict = None       # Optional constraints
) -> Any

# Get agent configuration
config = agent.get_config() -> dict

# Update configuration
agent.update_config(
    temperature: float = None,
    max_tokens: int = None,
    timeout: int = None
) -> None
```

### Tools

Tool creation and configuration.

```python
from pydantic_ai import Tool, ToolConfig

# Create a tool
tool = Tool(
    function: Callable,            # Tool function
    timeout: int = 10,             # Tool timeout
    retries: int = 2,              # Number of retries
    backoff_factor: float = 1.5,   # Backoff between retries
    rate_limit: int = None,        # Calls per minute
    cache_ttl: int = None,         # Cache TTL in seconds
    validate_input: bool = True,   # Validate input types
    validate_output: bool = True,  # Validate output type
    debug: bool = False            # Enable debugging
)

# Or with advanced configuration
config = ToolConfig(
    timeout=10,
    retries=2,
    backoff_factor=1.5,
    rate_limit=100,
    cache_ttl=300,
    validate_input=True,
    validate_output=True
)
tool = Tool(function, config=config)
```

#### Methods

```python
# Run the tool
result = tool.run(
    *args,                         # Positional arguments
    **kwargs                       # Keyword arguments
) -> Any

# Run asynchronously
result = await tool.run_async(
    *args,                         # Positional arguments
    **kwargs                       # Keyword arguments
) -> Any

# Get tool configuration
config = tool.get_config() -> ToolConfig

# Update configuration
tool.update_config(
    timeout: int = None,
    retries: int = None,
    backoff_factor: float = None,
    rate_limit: int = None,
    cache_ttl: int = None,
    validate_input: bool = None,
    validate_output: bool = None,
    debug: bool = None
) -> None
```

## Data Models

### Candidate

Represents a solution produced by the orchestrator.

```python
from pydantic_ai_orchestrator import Candidate

candidate = Candidate(
    solution: Any,                 # The solution
    quality_checklist: dict,       # Quality assessment
    metadata: dict,                # Additional metadata
    score: float,                  # Quality score
    model: str,                    # Model used
    tokens: int,                   # Tokens used
    duration: float                # Duration in seconds
)
```

#### Methods

```python
# Get candidate as dict
data = candidate.dict() -> dict

# Get candidate as JSON
json = candidate.json() -> str

# Get quality score
score = candidate.get_score() -> float

# Get quality checklist
checklist = candidate.get_checklist() -> dict
```

### Task

Represents a task to be processed by the orchestrator.

```python
from pydantic_ai_orchestrator import Task

task = Task(
    prompt: str,                   # Task prompt
    metadata: dict = None,         # Optional metadata
    constraints: dict = None       # Optional constraints
)
```

#### Methods

```python
# Get task as dict
data = task.dict() -> dict

# Get task as JSON
json = task.json() -> str

# Get task constraints
constraints = task.get_constraints() -> dict
```

## Telemetry

### Initialization

```python
from pydantic_ai_orchestrator import init_telemetry

init_telemetry(
    enable_export: bool = True,    # Enable metric export
    export_endpoint: str = None,   # Export endpoint
    service_name: str = "orchestrator",  # Service name
    environment: str = "development"      # Environment
)
```

### Metrics

```python
from pydantic_ai_orchestrator import get_metrics

# Get all metrics
metrics = get_metrics() -> dict

# Get specific metric
value = get_metrics("task_duration") -> float
```

### Tracing

```python
from pydantic_ai_orchestrator import enable_tracing, get_traces

# Enable tracing
with enable_tracing():
    result = orchestrator.run("prompt")

# Get traces
traces = get_traces() -> list[dict]
```

## Utilities

### Model Management

```python
from pydantic_ai_orchestrator import list_available_models

# List available models
models = list_available_models() -> list[str]

# Check model availability
is_available = is_model_available("openai:gpt-4") -> bool
```

### Error Handling

```python
from pydantic_ai_orchestrator import get_error_details

# Get detailed error info
details = get_error_details(error) -> dict
```

### Profiling

```python
from pydantic_ai_orchestrator import enable_profiling

# Enable profiling
with enable_profiling():
    result = orchestrator.run("prompt")
```

## Constants

### Model Identifiers

```python
from pydantic_ai_orchestrator import (
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    GOOGLE_MODELS
)

# Available models
print(OPENAI_MODELS)    # ["gpt-4", "gpt-3.5-turbo", ...]
print(ANTHROPIC_MODELS) # ["claude-3-opus", "claude-3-sonnet", ...]
print(GOOGLE_MODELS)    # ["gemini-pro", ...]
```

### Configuration

```python
from pydantic_ai_orchestrator import (
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_RETRIES
)

# Default values
print(DEFAULT_TIMEOUT)      # 30
print(DEFAULT_MAX_TOKENS)   # 1000
print(DEFAULT_TEMPERATURE)  # 0.7
print(DEFAULT_RETRIES)      # 2
```

## Type Definitions

### Common Types

```python
from typing import (
    Any,                # Any type
    Dict,              # Dictionary type
    List,              # List type
    Optional,          # Optional type
    Union,             # Union type
    Callable           # Callable type
)

# Common type aliases
ModelIdentifier = str
QualityScore = float
TokenCount = int
Duration = float
```

### Custom Types

```python
from pydantic_ai_orchestrator import (
    Candidate,         # Solution candidate
    Task,             # Task definition
    Tool,             # Tool definition
    ToolConfig,       # Tool configuration
    Orchestrator,     # Main orchestrator
    PipelineRunner    # Pipeline runner
)
```

## Next Steps

- Read the [Usage Guide](usage.md)
- Check [Advanced Topics](extending.md)
- Explore [Use Cases](use_cases.md) 