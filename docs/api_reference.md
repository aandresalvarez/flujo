# API Reference

This guide provides detailed documentation for all public interfaces in `flujo`.

## Core Components

### Default Recipe

`flujo.recipes.Default` is a high-level facade for running a **standard, fixed
multi-agent pipeline**: Review -> Solution -> Validate -> Reflection. It uses the agents you
provide for these roles. For custom pipelines with different logic, see
`Flujo` and the `Step` DSL.

```python
from flujo.recipes import Default

orchestrator = Default(
    review_agent: AsyncAgentProtocol[Any, Checklist],
    solution_agent: AsyncAgentProtocol[Any, str],
    validator_agent: AsyncAgentProtocol[Any, Checklist],
    reflection_agent: Optional[AsyncAgentProtocol[Any, str]] = None,
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

### Pipeline DSL & `Flujo`

The Pipeline DSL lets you create flexible, custom workflows and execute them
with `Flujo`.

```python
from flujo import (
    Step, Flujo, Task,
    review_agent, solution_agent, validator_agent,
)
from pydantic import BaseModel
from typing import Any
from flujo import AppResources

class MyResources(AppResources):
    db_pool: Any

my_resources = MyResources(db_pool=make_pool())

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
        plugins=[plugin1]          # Optional validation plugins
    )
)

runner = Flujo(custom_pipeline)
# With a shared typed context
runner_with_ctx = Flujo(
    custom_pipeline,
    context_model=MyContext,
    initial_context_data={"counter": 0},
    resources=my_resources,
)

# Advanced constructs
looping_step = Step.loop_until(
    name="refinement_loop",
    loop_body_pipeline=Pipeline.from_step(Step.solution(solution_agent)),
    exit_condition_callable=lambda out, ctx: "done" in out.lower(),
)

# Conditional branching
router = Step.branch_on(
    name="router",
    condition_callable=lambda out, ctx: "code" if "function" in out else "text",
    branches={
        "code": Pipeline.from_step(Step.solution(solution_agent)),
        "text": Pipeline.from_step(Step.validate(validator_agent)),
    },
)
```

#### Methods

```python
# Run the pipeline
pipeline_result = runner.run(
    "Your initial prompt"  # Input for the first step
)  # Returns PipelineResult

# Access step results
for step_res in pipeline_result.step_history:
    print(f"Step: {step_res.name}, Success: {step_res.success}")

# Get total cost
total_cost = pipeline_result.total_cost_usd

# Access final pipeline context (if using typed context)
final_ctx = pipeline_result.final_pipeline_context
```

### Agents

Agent creation and configuration utilities.

```python
from flujo import make_agent_async

# Create a custom agent
agent = make_agent_async(
    model: str,                    # Model identifier (e.g., "openai:gpt-4")
    system_prompt: str,            # System prompt
    output_type: type,             # Output type (str, Pydantic model, etc.)
    tools: Optional[List[Tool]] = None,  # Optional tools
)

# Pre-built agents
from flujo import (
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
    get_reflection_agent,
)
```

#### Default Agents

- **`review_agent`**: Creates quality checklists (outputs `Checklist`)
- **`solution_agent`**: Generates solutions (outputs `str`)
- **`validator_agent`**: Validates solutions (outputs `Checklist`)
- **`reflection_agent`**: Provides reflection and improvement suggestions (outputs `str`)

## Data Models

### AppResources

Container for long-lived resources shared across pipeline steps.

```python
from flujo import AppResources

class MyResources(AppResources):
    db_pool: Any
```

### Task

Represents a task to be solved by the orchestrator.

```python
from flujo import Task

task = Task(
    prompt: str,                   # The task prompt
    metadata: Dict[str, Any] = {}  # Optional metadata
)
```

### Candidate

Represents a solution produced by the orchestrator.

```python
from flujo import Candidate

candidate = Candidate(
    solution: str,                 # The solution
    score: float,                  # Quality score (0.0 to 1.0)
    checklist: Optional[Checklist] = None,  # Quality assessment
)
```

### Checklist & ChecklistItem

Quality evaluation structures.

```python
from flujo import Checklist, ChecklistItem

item = ChecklistItem(
    description: str,              # What is being checked
    passed: Optional[bool] = None, # Whether it passed
    feedback: Optional[str] = None # Feedback if failed
)

checklist = Checklist(
    items: List[ChecklistItem]     # List of checklist items
)
```

### Pipeline Results

Results from pipeline execution.

```python
from flujo import PipelineResult, StepResult

step_result = StepResult(
    name: str,                     # Step name
    output: Any = None,            # Step output
    success: bool = True,          # Whether step succeeded
    attempts: int = 0,             # Number of attempts
    latency_s: float = 0.0,        # Execution time
    token_counts: int = 0,         # Token usage
    cost_usd: float = 0.0,         # Cost in USD
    feedback: Optional[str] = None, # Error feedback
    metadata_: Optional[Dict[str, Any]] = None,  # Additional metadata
)

pipeline_result = PipelineResult(
    step_history: List[StepResult] = [],  # All step results
    total_cost_usd: float = 0.0,          # Total cost
    final_pipeline_context: Optional[BaseModel] = None,  # Final context
)
```

## Self-Improvement & Evaluation

### Evaluation Functions

```python
from flujo import run_pipeline_async, evaluate_and_improve

# Run pipeline evaluation
result = await run_pipeline_async(
    inputs: str,                   # Input prompt
    runner: Flujo,                 # Pipeline runner
    **kwargs                       # Additional arguments
)

# Generate improvement suggestions
report = await evaluate_and_improve(
    task_fn: Callable,             # Task function
    dataset: Any,                  # Evaluation dataset
    agent: SelfImprovementAgent,   # Improvement agent
    pipeline_definition: Optional[Pipeline] = None,  # Pipeline definition
)
```

### Improvement Models

```python
from flujo import (
    SelfImprovementAgent,
    ImprovementReport,
    ImprovementSuggestion,
)

# Create improvement agent
improvement_agent = SelfImprovementAgent(
    agent: AsyncAgentProtocol[Any, str]  # Underlying agent
)

# Improvement suggestion structure
suggestion = ImprovementSuggestion(
    target_step_name: Optional[str],     # Target step
    suggestion_type: SuggestionType,     # Type of suggestion
    failure_pattern_summary: str,        # What failed
    detailed_explanation: str,           # Detailed explanation
    estimated_impact: Optional[str],     # Impact estimate
    estimated_effort_to_implement: Optional[str],  # Effort estimate
)

# Improvement report
report = ImprovementReport(
    suggestions: List[ImprovementSuggestion]  # All suggestions
)
```

## Configuration & Settings

### Settings

```python
from flujo import settings

# Access current settings
current_settings = settings

# Key settings properties:
# - default_solution_model: str
# - default_review_model: str  
# - default_validator_model: str
# - default_reflection_model: str
# - reflection_enabled: bool
# - scorer: str
# - agent_timeout: int
# - telemetry_export_enabled: bool
```

### Telemetry

```python
from flujo import init_telemetry

# Initialize telemetry
init_telemetry()

# Telemetry is automatically enabled for all operations
# Use environment variables to configure:
# - TELEMETRY_EXPORT_ENABLED=true
# - OTLP_EXPORT_ENABLED=true
# - OTLP_ENDPOINT=https://your-endpoint
```

## Plugins & Extensions

### Validation Plugins

```python
from flujo import ValidationPlugin, PluginOutcome
from flujo.plugins import SQLSyntaxValidator

# Use built-in SQL validator
sql_validator = SQLSyntaxValidator()

# Create custom validation plugin
class MyPlugin(ValidationPlugin):
    def validate(self, output: Any, context: Any) -> PluginOutcome:
        # Custom validation logic
        if self.is_valid(output):
            return PluginOutcome(passed=True)
        return PluginOutcome(passed=False, feedback="Validation failed")
```

### Testing Utilities

```python
from flujo.testing import StubAgent, DummyPlugin

# Create stub agent for testing
stub_agent = StubAgent(
    return_value="test response",
    output_type=str
)

# Create dummy plugin for testing
dummy_plugin = DummyPlugin(should_pass=True)
```

## Exceptions

```python
from flujo import (
    OrchestratorError,
    ConfigurationError,
    SettingsError,
)

# Base exception for all orchestrator errors
try:
    result = orchestrator.run_sync(task)
except OrchestratorError as e:
    print(f"Orchestrator error: {e}")

# Configuration-specific errors
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Settings-specific errors  
except SettingsError as e:
    print(f"Settings error: {e}")
```

## Command Line Interface

The package provides a comprehensive CLI:

```bash
# Main commands
flujo solve "prompt"              # Solve a task
flujo bench "prompt" --rounds 5   # Benchmark performance
flujo show-config                 # Show configuration
flujo version-cmd                 # Show version

# Advanced commands
flujo improve pipeline.py dataset.py     # Generate improvements
flujo explain pipeline.py                # Explain pipeline structure
flujo add-eval-case --dataset dataset.py # Add evaluation case
```

## Best Practices

1. **Always use async contexts** when possible for better performance
2. **Implement proper error handling** using the provided exception types
3. **Use typed pipeline contexts** for complex workflows
4. **Enable telemetry** for production deployments
5. **Implement custom validation plugins** for domain-specific requirements
6. **Use the CLI** for quick testing and benchmarking

## Next Steps

- Explore [Pipeline DSL Guide](pipeline_dsl.md) for advanced workflows
- Read [Intelligent Evals](intelligent_evals.md) for evaluation strategies
- Check [Telemetry Guide](telemetry.md) for monitoring setup
- Review [Extending Guide](extending.md) for custom components 