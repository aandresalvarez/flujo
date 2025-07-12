# API Reference

This guide provides detailed documentation for all public interfaces in `flujo`.

## Core Components

### Default Recipe

`flujo.recipes.Default` is a high-level facade for running a **standard, fixed
multi-agent pipeline**: Review -> Solution -> Validate -> Reflection. It uses the agents you
provide for these roles. For custom pipelines with different logic, see
`Flujo` and the `Step` DSL.

```python
from flujo.recipes.factories import make_default_pipeline

pipeline = make_default_pipeline(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
    max_iters=max_iters,
    k_variants=k_variants,
    reflection_limit=reflection_limit,
)
```

#### Methods

```python
# Run a task synchronously
result = orchestrator.run_sync(Task(prompt="Generate a poem"))

# Run a task asynchronously and gather the result
from flujo.testing import gather_result
candidate = await gather_result(orchestrator.run_async(Task(prompt="Generate a poem")))

# Stream the response chunk by chunk
async for piece in orchestrator.stream_async(Task(prompt="Generate a poem")):
    ...
```
`run_async` is an async generator that yields chunks from the final step. Use
`stream_async` as a clearer alias.

### Pipeline DSL & `Flujo`

The Pipeline DSL lets you create flexible, custom workflows and execute them
with `Flujo`.

```python
from flujo import Step, Flujo, Task, PipelineRegistry
from flujo.state import SQLiteBackend
from flujo.infra.agents import make_review_agent, make_solution_agent, make_validator_agent
from flujo.infra.backends import LocalBackend
from pydantic import BaseModel
from typing import Any
from flujo.domain.resources import AppResources
from flujo.models import UsageLimits

class MyResources(AppResources):
    db_pool: Any

my_resources = MyResources(db_pool=make_pool())

from flujo.domain.models import PipelineContext

class MyContext(PipelineContext):
    counter: int = 0

# Create a pipeline
custom_pipeline = (
    Step.review(make_review_agent())      # Review step
    >> Step.solution(              # Solution step
        make_solution_agent(),
        tools=[tool1, tool2],      # Optional tools
        processors=my_processors   # Optional processors
    )
    >> Step.validate_step(              # Validation step
        make_validator_agent(),
        plugins=[plugin1],         # Optional validation plugins
        validators=[validator1],   # Optional programmatic validators
        processors=my_processors
    )
)

# The `processors` argument lets you run custom pre- and post-processing
# logic for a step. See [Using Processors](cookbook/using_processors.md).

# You can also build steps from async functions using
# `Step.from_mapper` or the `mapper` alias:
async def to_upper(text: str) -> str:
    return text.upper()

upper_step = Step.from_mapper(to_upper)

registry = PipelineRegistry()
registry.register(custom_pipeline, "custom", "1.0.0")
runner = Flujo(
    registry=registry,
    pipeline_name="custom",
    pipeline_version="1.0.0",
)
# With a shared typed context
runner_with_ctx = Flujo(
    registry=registry,
    pipeline_name="custom",
    pipeline_version="1.0.0",
    context_model=MyContext,
    initial_context_data={"counter": 0},
    resources=my_resources,
    usage_limits=UsageLimits(total_cost_usd_limit=10.0),
    hooks=[my_hook],
    backend=LocalBackend(),
    state_backend=SQLiteBackend("state.db"),
)

# Advanced constructs
looping_step = Step.loop_until(
    name="refinement_loop",
loop_body_pipeline=Pipeline.from_step(Step.solution(make_solution_agent())),
    exit_condition_callable=lambda out, ctx: "done" in out.lower(),
)

# Pause for human input
approval_step = Step.human_in_the_loop(
    name="approval",
    message_for_user="Is the draft acceptable?",
)

# Conditional branching
router = Step.branch_on(
    name="router",
    condition_callable=lambda out, ctx: "code" if "function" in out else "text",
    branches={
        "code": Pipeline.from_step(Step.solution(solution_agent)),
        "text": Pipeline.from_step(Step.validate_step(validator_agent)),
    },
)
```

#### Methods

```python
# Run the pipeline
pipeline_result = runner.run(
    "Your initial prompt"  # Input for the first step
)  # Returns PipelineResult

# Stream results asynchronously
async for part in runner.stream_async("Your initial prompt"):
    ...

# Access step results
for step_res in pipeline_result.step_history:
    print(f"Step: {step_res.name}, Success: {step_res.success}")

# Get total cost
total_cost = pipeline_result.total_cost_usd

# Access final pipeline context (if using typed context)
final_ctx = pipeline_result.final_pipeline_context
```

### Agents

`flujo` defines several protocols that agents can implement to interact with the orchestrator. These protocols ensure type safety and define the expected behavior of agents.

#### `AsyncAgentProtocol`

This is the generic asynchronous agent interface. Agents implementing this protocol can be used in `Step`s that do not require access to the pipeline's `context` object.

```python
from flujo.domain.agent_protocol import AsyncAgentProtocol
from typing import Any

class MySimpleAgent(AsyncAgentProtocol[str, str]):
    async def run(self, data: str, **kwargs: Any) -> str:
        return f"Processed: {data.upper()}"
```

#### `StreamingAgentProtocol`

This protocol is for agents that can stream their output asynchronously, yielding chunks of data as they become available. This is particularly useful for real-time applications or when dealing with large outputs.

```python
from flujo.domain.streaming_protocol import StreamingAgentProtocol
from typing import AsyncIterator, Any

class MyStreamingAgent(StreamingAgentProtocol):
    async def stream(self, data: Any, **kwargs: Any) -> AsyncIterator[str]:
        for char in data:
            yield char
        yield "\n"
```

#### `ContextAwareAgentProtocol`

This protocol is for agents that need to access or modify the pipeline's shared `context` object. Agents implementing this protocol must define a `context` parameter in their `run` method, type-hinted with the specific `PipelineContext` subclass they expect.

```

```python
from flujo.domain.agent_protocol import ContextAwareAgentProtocol
from flujo.domain.models import PipelineContext
from typing import Any

class MyCustomContext(PipelineContext):
    counter: int = 0

class MyContextAwareAgent(ContextAwareAgentProtocol[str, str, MyCustomContext]):
    async def run(self, data: str, *, context: MyCustomContext, **kwargs: Any) -> str:
        context.counter += 1
        return f"Processed (context updated): {data.lower()}"
```

#### Agent Creation and Configuration

`flujo` provides utilities for creating and configuring agents.

```python
from flujo import make_agent_async

# Create a custom agent
agent = make_agent_async(
    model: str,                    # Model identifier (e.g., "openai:gpt-4")
    system_prompt: str,            # System prompt
    output_type: type,             # Output type (str, Pydantic model, etc.)
    tools: Optional[List[Tool]] = None,  # Optional tools
)

# Implement a streaming agent
class MyStreamer(StreamingAgentProtocol):
    async def stream(self, data: str) -> AsyncIterator[str]:
        yield data
```

#### Pre-built Agents

`flujo` includes several pre-built agents for common tasks:

```python
from flujo.infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
    get_reflection_agent,
)
```

- **`review_agent`**: Creates quality checklists (outputs `Checklist`)
- **`solution_agent`**: Generates solutions (outputs `str`)
- **`validator_agent`**: Validates solutions (outputs `Checklist`)
- **`reflection_agent`**: Provides reflection and improvement suggestions (outputs `str`)

## Data Models

### AppResources

Container for long-lived resources shared across pipeline steps.

```python
from flujo.domain.resources import AppResources

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
from flujo.models import Checklist, ChecklistItem

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
from flujo.models import PipelineResult, StepResult

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
    final_pipeline_context: Optional[ContextT] = None,  # Final context
)
```

### PipelineContext

Each run gets a `PipelineContext` with:

- `run_id`: unique identifier
- `initial_prompt`: the first input
- `scratchpad`: a mutable dictionary for agents
- `hitl_history`: list of `HumanInteraction` records

### Type Aliases

`flujo` uses several type aliases to improve readability and maintainability. These aliases are primarily used for type hinting throughout the library.

#### `ContextT`

A generic type variable bound to `pydantic.BaseModel`, used to represent the type of the pipeline context. This allows for type-safe definition of custom context models.

```python
from flujo.domain.types import ContextT
from pydantic import BaseModel

class MyCustomContext(BaseModel):
    data: str

def process_with_context(context: ContextT):
    # context will be type-checked as MyCustomContext if used with it
    pass
```

#### `HookCallable`

A type alias for an asynchronous callable that represents a lifecycle hook. Hooks receive a `HookPayload` object containing event-specific data. Raise `PipelineAbortSignal` from a hook to stop the run gracefully.

```python
from flujo.domain.types import HookCallable
from flujo.domain.events import HookPayload

async def my_hook_function(payload: HookPayload) -> None:
    print(f"Hook triggered: {payload.event_name}")

# This function can be used as a HookCallable
hook: HookCallable = my_hook_function
```

### Orchestrator Commands

These models represent the commands that the `flujo` orchestrator can generate and execute internally. Understanding these commands is crucial for advanced customization and debugging.

#### `RunAgentCommand`

Instructs the orchestrator to run a registered sub-agent.

```python
from flujo.domain.commands import RunAgentCommand

command = RunAgentCommand(
    agent_name="my_custom_agent",
    input_data={"query": "What is the weather like?"}
)
```


#### `AskHumanCommand`

Pauses the pipeline execution and prompts a human user for input.

```python
from flujo.domain.commands import AskHumanCommand

command = AskHumanCommand(
    question="Is this solution acceptable?"
)
```

#### `FinishCommand`

Signals the completion of the task and provides the final answer or summary.

```python
from flujo.domain.commands import FinishCommand

command = FinishCommand(
    final_answer="The final answer is 42."
)
```

#### `ExecutedCommandLog`

A structured log entry representing a command that was executed in the loop, including the generated command and its execution result.

```python
from flujo.domain.commands import ExecutedCommandLog, RunAgentCommand
from datetime import datetime, timezone

log_entry = ExecutedCommandLog(
    turn=1,
    generated_command=RunAgentCommand(agent_name="test_agent", input_data="test"),
    execution_result={"output": "agent response"},
    timestamp=datetime.now(timezone.utc)
)
```

### Resuming a Paused Pipeline

Use `Flujo.resume_async(paused_result, human_input)` to continue after a `HumanInTheLoopStep`.
```

### UsageLimits

Define cost or token ceilings for a run.

```python
from flujo.models import UsageLimits

limits = UsageLimits(
    total_cost_usd_limit: Optional[float] = None,
    total_tokens_limit: Optional[int] = None,
)
```

### Execution Backends

`Flujo` delegates step execution to an `ExecutionBackend`. The built-in `LocalBackend` runs steps in the current process and is the default backend if none is explicitly provided. It's suitable for most use cases where pipeline steps can be executed within the same Python process.

#### `LocalBackend`

This backend executes pipeline steps directly within the current process. It handles the execution of agents and the propagation of context and resources between steps.

```python
from flujo.domain.backends import ExecutionBackend, StepExecutionRequest
from flujo.infra.backends import LocalBackend
from flujo.domain.agent_protocol import AsyncAgentProtocol
from typing import Any

# You can optionally provide an agent_registry if your steps refer to agents by name
local_backend = LocalBackend(agent_registry={
    "my_agent": MyAgentImplementation() # MyAgentImplementation should be an AsyncAgentProtocol
})

# When a step is executed, a StepExecutionRequest is sent to the backend:
request = StepExecutionRequest(
    step=Step(...),
    input_data=...,
    context=PipelineContext(initial_prompt=""),
    resources=None,
    stream=False,  # Set to True to enable streaming
    on_chunk=None,  # Optional callback for streamed chunks
)

# The execute_step method handles the actual running of the step's logic
# result = await local_backend.execute_step(request)
```

Custom backends can be implemented by inheriting from `ExecutionBackend` and overriding the `execute_step(request) -> StepResult` method. This allows for advanced scenarios like distributed execution or custom resource management.

### Lifecycle Hooks

You can register callbacks to observe or control pipeline execution. Hooks receive a typed payload object describing the event. Raise `PipelineAbortSignal` from a hook to stop the run gracefully.

```python
from flujo.domain import HookCallable
from flujo.exceptions import PipelineAbortSignal
from flujo.domain.events import HookPayload

async def my_hook(payload: HookPayload) -> None:
    print("event:", payload.event_name)

runner = Flujo(pipeline, hooks=[my_hook])
```

#### Hook Payloads

These models define the structure of the data passed to your lifecycle hooks, allowing you to access relevant information at different stages of pipeline execution.

##### `PreRunPayload`

Payload for hooks executed before the pipeline starts. Contains the initial input, context object, and resources that will be used during pipeline execution.

```python
from flujo.domain.events import PreRunPayload

# Example usage in a hook:
async def log_pre_run(payload: PreRunPayload):
    print(f"Pipeline starting with input: {payload.initial_input}")
```

##### `PostRunPayload`

Payload for hooks executed after the pipeline completes (successfully or with errors). Contains the final pipeline result, context object, and resources used during pipeline execution.

```python
from flujo.domain.events import PostRunPayload

# Example usage in a hook:
async def log_post_run(payload: PostRunPayload):
    print(f"Pipeline finished. Total cost: {payload.pipeline_result.total_cost_usd}")
```

##### `PreStepPayload`

Payload for hooks executed before an individual step starts. Contains the step that is about to be executed, its input data, context object, and resources available for the step.

```python
from flujo.domain.events import PreStepPayload

# Example usage in a hook:
async def log_pre_step(payload: PreStepPayload):
    print(f"Step '{payload.step.name}' starting with input: {payload.step_input}")
```

##### `PostStepPayload`

Payload for hooks executed after an individual step completes. Contains the step result, context object, and resources that were used during step execution.

```python
from flujo.domain.events import PostStepPayload

# Example usage in a hook:
async def log_post_step(payload: PostStepPayload):
    print(f"Step '{payload.step_result.name}' finished. Success: {payload.step_result.success}")
```

##### `OnStepFailurePayload`

Payload for hooks executed when an individual step fails. Contains the step result (with failure details), context object, and resources that were used during the failed step execution.

```python
from flujo.domain.events import OnStepFailurePayload

# Example usage in a hook:
async def log_step_failure(payload: OnStepFailurePayload):
    print(f"Step '{payload.step_result.name}' failed with feedback: {payload.step_result.feedback}")
```

## Self-Improvement & Evaluation

### Evaluation Functions

```python
from flujo.application import run_pipeline_async, evaluate_and_improve

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
from flujo.application import SelfImprovementAgent
from flujo.models import ImprovementReport, ImprovementSuggestion

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
from flujo.infra import settings

# Access current settings
current_settings = settings

# Key settings properties:
# - default_solution_model: str
# - default_review_model: str
# - default_validator_model: str
# - default_reflection_model: str
# - default_repair_model: str
# - reflection_enabled: bool
# - scorer: str
# - agent_timeout: int
# - telemetry_export_enabled: bool
```

### Telemetry

```python
from flujo.infra import init_telemetry

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
from flujo.domain import ValidationPlugin, PluginOutcome
from flujo.plugins import SQLSyntaxValidator

# Use built-in SQL validator
sql_validator = SQLSyntaxValidator()

# Create custom validation plugin
class MyPlugin(ValidationPlugin):
    def validate(self, output: Any, context: Any) -> PluginOutcome:
        # Custom validation logic
        if self.is_valid(output):
            return PluginOutcome(passed=True)
        return PluginOutcome(passed(False, feedback="Validation failed")
```

#### `SQLSyntaxValidator`

This plugin checks the syntax of SQL queries. It can be used in a `validate` step to ensure that any generated SQL is syntactically correct.

```python
from flujo import Step, Flujo
from flujo.plugins import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

# Example of a step that might generate SQL
sql_agent = StubAgent(["SELECT * FROM users WHERE id = 1;"])

# Create a pipeline with the SQLSyntaxValidator
pipeline = (
    Step.solution(sql_agent)
    >> Step.validate(plugins=[SQLSyntaxValidator()])
)

# Run the pipeline
runner = Flujo(pipeline)
result = runner.run("Generate a SQL query")

# The validation result will indicate if the SQL is valid
print(result.step_history[-1].success)
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

`flujo` defines a hierarchy of custom exceptions to provide more granular error handling and to signal specific conditions within the orchestrator. All custom exceptions inherit from `OrchestratorError`.

*   `OrchestratorError`: Base exception for all application-specific errors.
*   `SettingsError`: Raised for errors related to application settings and configuration.
*   `ConfigurationError`: A subclass of `SettingsError`, specifically raised when a required configuration for a provider (e.g., API key) is missing.
*   `OrchestratorRetryError`: Raised when an agent operation fails after all configured retries have been exhausted.
*   `RewardModelUnavailable`: Raised when a reward model is required for scoring but is not available or configured.
*   `FeatureDisabled`: Raised when an attempt is made to invoke a feature that has been explicitly disabled in the settings.
*   `InfiniteRedirectError`: Raised when a redirect loop is detected during pipeline execution, preventing infinite processing.
*   `PipelineContextInitializationError`: Raised when there is an issue initializing a typed pipeline context.
*   `ContextInheritanceError`: Raised when a nested pipeline fails to inherit required fields from its parent context.
*   `UsageLimitExceededError`: Raised when a pipeline run exceeds its defined usage limits (e.g., total cost, token usage).
*   `PipelineAbortSignal`: A special exception that can be raised by lifecycle hooks to gracefully stop a pipeline's execution.
*   `PausedException`: An internal exception used to signal that a pipeline has been paused, typically for human input.
*   `ImproperStepInvocationError`: Raised when a `Step` object is invoked directly instead of being run through a `Flujo` runner.
*   `MissingAgentError`: A subclass of `ConfigurationError`, raised when a pipeline step is configured without a required agent.
*   `TypeMismatchError`: A subclass of `ConfigurationError`, raised when consecutive steps in a pipeline have incompatible input/output types.

```python
from flujo.exceptions import (
    OrchestratorError,
    ConfigurationError,
    SettingsError,
    UsageLimitExceededError,
    PipelineAbortSignal,
    ImproperStepInvocationError,
    MissingAgentError,
    TypeMismatchError,
    RewardModelUnavailable,
    FeatureDisabled,
    InfiniteRedirectError,
    PipelineContextInitializationError,
    ContextInheritanceError,
    PausedException,
    OrchestratorRetryError,
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

# Usage governor errors
except UsageLimitExceededError as e:
    print(f"Usage limits hit: {e}")
```

## Command Line Interface

The `flujo` package provides a comprehensive command-line interface (CLI) for interacting with the orchestrator and performing various tasks. You can access the CLI by running `flujo` in your terminal.

### Global Options

*   `--profile`: Enable Logfire STDOUT span viewer for profiling. This will print detailed tracing information to your console during execution.

### Commands

#### `flujo solve`

Solves a task using the multi-agent orchestrator.

```bash
flujo solve "Your task prompt here" \
  --max-iters 3 \
  --k 2 \
  --reflection \
  --scorer weighted \
  --weights-path ./weights.json \
  --solution-model openai:gpt-4 \
  --review-model openai:gpt-3.5-turbo \
  --validator-model openai:gpt-3.5-turbo \
  --reflection-model openai:gpt-3.5-turbo
```

**Arguments:**

*   `prompt`: The task prompt to solve (required).

**Options:**

*   `--max-iters <int>`: Maximum number of iterations for the orchestrator (default: `3`).
*   `--k <int>`: Number of solution variants to generate per iteration (default: `2`).
*   `--reflection`: Enable or disable the reflection agent (default: `true` from settings).
*   `--scorer <ratio|weighted|reward>`: Scoring strategy to use (default: `ratio` from settings).
*   `--weights-path <path>`: Path to a JSON or YAML file containing weights for the `weighted` scorer.
*   `--solution-model <model_name>`: Override the default model for the Solution agent.
*   `--review-model <model_name>`: Override the default model for the Review agent.
*   `--validator-model <model_name>`: Override the default model for the Validator agent.
*   `--reflection-model <model_name>`: Override the default model for the Reflection agent.

#### `flujo version-cmd`

Prints the installed `flujo` package version.

```bash
flujo version-cmd
```

#### `flujo show-config`

Prints the effective `flujo` settings, with sensitive information (like API keys) masked.

```bash
flujo show-config
```

#### `flujo bench`

Runs a quick micro-benchmark of generation latency and score for a given prompt.

```bash
flujo bench "Generate a short poem" --rounds 10
```

**Arguments:**

*   `prompt`: The prompt to benchmark (required).

**Options:**

*   `--rounds <int>`: Number of benchmark rounds to run (default: `10`).

#### `flujo add-eval-case`

Prints a new `Case(...)` definition that you can manually add to a `pydantic-evals` dataset file.

```bash
flujo add-eval-case \
  --dataset path/to/my_evals.py \
  --name "my_new_test_case" \
  --inputs "User input for the case" \
  --expected "Expected output for the case" \
  --metadata '{"difficulty": "easy"}'
```

**Options:**

*   `--dataset, -d <path>`: Path to the Python file containing the `Dataset` object (required).
*   `--name, -n <string>`: A unique name for the new evaluation case (required, prompted if not provided).
*   `--inputs, -i <string>`: The primary input for this case (required, prompted if not provided).
*   `--expected, -e <string>`: The expected output for this case (optional, prompted if not provided).
*   `--metadata, -m <json_string>`: JSON string for case metadata (optional).
*   `--dataset-var <string>`: Name of the `Dataset` variable in the file (default: `dataset`).

#### `flujo improve`

Runs evaluation on a pipeline and dataset, then generates improvement suggestions using a self-improvement agent.

```bash
flujo improve my_pipeline.py my_dataset.py \
  --improvement-model openai:gpt-4 \
  --json
```

**Arguments:**

*   `pipeline_path`: Path to the pipeline definition file (required).
*   `dataset_path`: Path to the dataset definition file (required).

**Options:**

*   `--improvement-model <model_name>`: LLM model to use for the `SelfImprovementAgent`.
*   `--json`: Output raw JSON instead of a formatted table.

#### `flujo explain`

Prints a summary of a pipeline defined in a file, listing its step names.

```bash
flujo explain my_pipeline.py
```

**Arguments:**

*   `path`: Path to the pipeline definition file (required).

#### `flujo validate`

Validates a pipeline defined in a file, checking for configuration errors and adherence to rules.

```bash
flujo validate my_pipeline.py --strict
```

**Arguments:**

*   `path`: Path to the pipeline definition file (required).

**Options:**

*   `--strict`: Exit with a non-zero status if validation errors are found.

#### `flujo pipeline-mermaid`

Outputs a pipeline's Mermaid diagram at the chosen detail level, either to stdout or a file.

```bash
flujo pipeline-mermaid \
  --file my_pipeline.py \
  --object pipeline \
  --detail-level medium \
  --output diagram.md
```

**Options:**

*   `--file, -f <path>`: Path to the Python file containing the pipeline object (required).
*   `--object, -o <string>`: Name of the pipeline variable in the file (default: `pipeline`).
*   `--detail-level <auto|high|medium|low>`: Detail level for the Mermaid diagram (default: `auto`).
*   `--output, -O <path>`: Output file path (if omitted, prints to stdout).

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

## Pipeline Visualization

### Pipeline.to_mermaid_with_detail_level(detail_level)

Generate a Mermaid diagram of the pipeline with configurable detail levels.

**Signature:**

```python
Pipeline.to_mermaid_with_detail_level(detail_level: str = "auto") -> str
```

- `detail_level`: One of `"high"`, `"medium"`, `"low"`, or `"auto"`.
    - `"high"`: Full detail, all subgraphs, annotations, and control flow.
    - `"medium"`: Simplified, uses emojis, no subgraphs.
    - `"low"`: Minimal, groups steps, high-level overview.
    - `"auto"`: Uses an AI agent to select the best level based on pipeline complexity.

**Returns:**
A valid Mermaid graph definition string.

**Example:**

```python
from flujo import Step, Pipeline

pipeline = Step("Extract", agent) >> Step("Transform", agent) >> Step("Load", agent)
mermaid_code = pipeline.to_mermaid_with_detail_level("medium")
print(mermaid_code)
```

See also: [Visualizing Pipelines](cookbook/visualizing_pipelines.md)

## Serialization Utilities

Flujo provides a comprehensive set of utilities for handling custom serialization scenarios.

### Global Registry Functions

#### `register_custom_serializer`

Register a custom serializer for a specific type globally.

```python
from flujo.utils import register_custom_serializer

def serialize_my_type(obj: MyType) -> Any:
    return obj.to_dict()

register_custom_serializer(MyType, serialize_my_type)
```

**Parameters:**
- `obj_type: type` - The type to register a serializer for
- `serializer_func: Callable[[Any], Any]` - Function that converts the type to a serializable format

**Returns:** `None`

**Example:**
```python
from datetime import datetime
from flujo.utils import register_custom_serializer

def serialize_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

register_custom_serializer(datetime, serialize_datetime)
```

#### `lookup_custom_serializer`

Look up a registered serializer for a value's type.

```python
from flujo.utils import lookup_custom_serializer

serializer = lookup_custom_serializer(value)
if serializer:
    result = serializer(value)
```

**Parameters:**
- `value: Any` - The value to find a serializer for

**Returns:** `Callable[[Any], Any] | None` - The registered serializer function, or None if not found

#### `safe_serialize`

Safely serialize an object with intelligent fallback handling.

```python
from flujo.utils import safe_serialize

result = safe_serialize(obj, default_serializer=str)
```

**Parameters:**
- `obj: Any` - The object to serialize
- `default_serializer: Callable[[Any], Any]` - Function to use as fallback (default: `str`)

**Returns:** `Any` - Serialized representation of the object

**Example:**
```python
from flujo.utils import safe_serialize

# Try Pydantic serialization first, then JSON, then fallback
result = safe_serialize(complex_object)
```

#### `serializable_field` (Deprecated)

Decorator to mark a field as serializable with a custom serializer. **This function is deprecated due to fundamental design issues with Pydantic v2.**

```python
# DEPRECATED - Use register_custom_serializer or manual field_serializer instead
from flujo.utils import serializable_field
from flujo.domain.models import BaseModel

class MyModel(BaseModel):
    @serializable_field(lambda x: x.to_dict())
    complex_object: ComplexType
```

**Recommended alternatives:**

1. **Global Registry:**
```python
from flujo.utils import register_custom_serializer

register_custom_serializer(ComplexType, lambda x: x.to_dict())

class MyModel(BaseModel):
    complex_object: ComplexType  # Uses global serializer
```

2. **Manual field_serializer:**
```python
from pydantic import field_serializer

class MyModel(BaseModel):
    complex_object: ComplexType

    @field_serializer('complex_object', when_used='json')
    def serialize_complex_object(self, value: ComplexType) -> dict:
        return value.to_dict()
```

**Parameters:**
- `serializer_func: Callable[[Any], Any]` - Function to serialize the field

**Returns:** `Callable[[T], T]` - Decorator function (deprecated)

#### `create_serializer_for_type`

Create a serializer function that handles a specific type.

```python
from flujo.utils import create_serializer_for_type

def serialize_my_type(obj: MyType) -> dict:
    return {"id": obj.id, "name": obj.name}

MyTypeSerializer = create_serializer_for_type(MyType, serialize_my_type)
```

**Parameters:**
- `obj_type: Type[Any]` - The type to create a serializer for
- `serializer_func: Callable[[Any], Any]` - Function that serializes the type

**Returns:** `Callable[[Any], Any]` - A serializer function that handles the specific type

#### `create_field_serializer`

Create a field_serializer method for a specific field.

```python
from flujo.utils import create_field_serializer
from pydantic import field_serializer

class MyModel(BaseModel):
    complex_field: ComplexType

    @field_serializer('complex_field', when_used='json')
    def serialize_complex_field(self, value: ComplexType) -> dict:
        return create_field_serializer('complex_field', lambda x: x.to_dict())(value)
```

**Parameters:**
- `field_name: str` - Name of the field to serialize
- `serializer_func: Callable[[Any], Any]` - Function that serializes the field value

**Returns:** `Callable[[Any], Any]` - A serializer function that can be used within field_serializer methods

### Enhanced BaseModel

The `BaseModel` in Flujo includes intelligent fallback serialization for common types:

#### Automatic Type Handling

The following types are automatically handled by the enhanced `BaseModel`:

- **datetime objects**: Converted to ISO format strings
- **Enum values**: Converted to the enum's value
- **Complex numbers**: Converted to `{"real": x, "imag": y}` format
- **Sets and frozensets**: Converted to lists
- **Bytes and memoryview**: UTF-8 decoded to strings
- **Callable objects**: Converted to module-qualified names
- **Custom objects**: Dictionary representation or string representation

#### Usage

```python
from flujo.domain.models import BaseModel
from datetime import datetime
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"

class MyModel(BaseModel):
    timestamp: datetime
    status: Status
    custom_data: Any

# Automatic serialization
model = MyModel(
    timestamp=datetime.now(),
    status=Status.PENDING,
    custom_data={"complex": "data"}
)

# Use JSON mode to trigger custom serialization
serialized = model.model_dump(mode="json")
```

### Field Serializer Decorator

For model-specific serialization, use Pydantic's `@field_serializer`:

```python
from pydantic import field_serializer
from flujo.domain.models import BaseModel

class MyModel(BaseModel):
    custom_field: MyCustomType

    @field_serializer('custom_field', when_used='json')
    def serialize_custom_field(self, value: MyCustomType) -> dict:
        return {
            "data": value.data,
            "metadata": value.metadata
        }
```

**Parameters:**
- `field_name: str` - Name of the field to serialize
- `when_used: str` - When to use the serializer (e.g., 'json')

**Returns:** `Callable[[Any], Any]` - The serialized value

## Integration Examples

### State Backend Integration

Custom types are automatically handled in state backends:

```python
from flujo.state.backends.file import FileBackend
from flujo.utils import register_custom_serializer

class DatabaseConnection:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

def serialize_db_connection(conn: DatabaseConnection) -> dict:
    return {"host": conn.host, "port": conn.port}

register_custom_serializer(DatabaseConnection, serialize_db_connection)

# State backends automatically use the global registry
backend = FileBackend("/tmp/state")
context = PipelineContext(db_connection=DatabaseConnection("localhost", 5432))
# Custom types are automatically serialized when saving state
```

### Cache Integration

Custom types in cache keys are automatically serialized:

```python
from flujo.steps.cache_step import CacheStep
from flujo.utils import register_custom_serializer

class ComplexKey:
    def __init__(self, value: str):
        self.value = value

def serialize_complex_key(key: ComplexKey) -> str:
    return f"key_{key.value}"

register_custom_serializer(ComplexKey, serialize_complex_key)

# Cache keys automatically use the global registry
step = CacheStep.cached(some_step)
# Custom types in cache keys are automatically serialized
```

### Context Integration

Custom types in pipeline context are automatically handled:

```python
from flujo.domain.models import PipelineContext
from flujo.utils import register_custom_serializer

class CustomContext(PipelineContext):
    custom_data: MyCustomType
    model_config = {"arbitrary_types_allowed": True}

def serialize_my_custom_type(obj: MyCustomType) -> dict:
    return obj.to_dict()

register_custom_serializer(MyCustomType, serialize_my_custom_type)

# Context updates automatically use the global registry
context = CustomContext(custom_data=MyCustomType("value"))
# Custom types are automatically serialized in context updates
```

## Best Practices

### When to Use Global Registry

- **Application-wide custom types**: Register once, use everywhere
- **Third-party types**: Handle types from external libraries
- **Consistent serialization**: Ensure the same type is always serialized the same way
- **Performance**: Avoid repeating serialization logic

### When to Use Field Serializers

- **Model-specific serialization**: Different serialization for the same type in different models
- **Context-dependent serialization**: Serialization that depends on model state
- **Temporary customizations**: One-off serialization needs

### When to Use Arbitrary Types

- **Simple custom types**: When you just need the object to be serializable
- **Quick prototypes**: For rapid development and testing
- **Legacy code**: When migrating existing code

### Performance Considerations

- **Global registry lookups are fast**: Simple dictionary lookups
- **Avoid complex serialization in hot paths**: Consider pre-serializing data
- **Use `model_dump(mode="json")` sparingly**: It's slower than `model_dump()`
- **Cache serialized results**: For frequently serialized objects

## Error Handling

### Common Errors and Solutions

**"Unable to generate pydantic-core schema"**

```python
# Solution: Add arbitrary_types_allowed
class MyModel(BaseModel):
    custom_field: MyCustomType
    model_config = {"arbitrary_types_allowed": True}
```

**"Object of type X is not JSON serializable"**

```python
# Solution: Register a custom serializer
from flujo.utils import register_custom_serializer

def serialize_my_type(obj):
    return obj.to_dict()

register_custom_serializer(MyType, serialize_my_type)
```

**Global registry not working**

```python
# Solution: Use JSON mode
# This uses the global registry
serialized = model.model_dump(mode="json")

# This does NOT use the global registry
serialized = model.model_dump()
```

## Migration from Previous Versions

### From v0.6.0 and Earlier

```python
# Before
class MyModel(BaseModel):
    custom_field: MyCustomType

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data['custom_field'] = self.custom_field.to_dict()
        return data

# After
from flujo.utils import register_custom_serializer

register_custom_serializer(MyCustomType, lambda x: x.to_dict())

class MyModel(BaseModel):
    custom_field: MyCustomType  # No changes needed
```

### From Custom State Backends

```python
# Before
class CustomStateBackend(StateBackend):
    def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        serialized = self._custom_serialize(state)
        # ... save logic

# After
class CustomStateBackend(StateBackend):
    def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        # Custom types are automatically handled
        serialized = json.dumps(state)
        # ... save logic
```

For more detailed examples and advanced usage patterns, see the [Advanced Serialization Guide](cookbook/advanced_serialization.md).
