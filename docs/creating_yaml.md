 
# Writing Flujo Pipelines in YAML

This comprehensive guide covers all aspects of creating Flujo pipelines using YAML syntax, including available options, use cases, and best practices.

## Table of Contents

1. [Basic Structure](#basic-structure)
2. [Step Types](#step-types)
3. [Agent Definitions](#agent-definitions)
4. [Pipeline Imports](#pipeline-imports)
5. [Configuration Options](#configuration-options)
6. [Advanced Features](#advanced-features)
7. [Use Cases and Examples](#use-cases-and-examples)
8. [Best Practices](#best-practices)

## Basic Structure

Every Flujo YAML pipeline follows this basic structure:

```yaml
version: "0.1"
name: "my_pipeline" # Optional but recommended
agents:
  # Agent definitions (optional)
imports:
  # Pipeline imports (optional)
steps:
  # Step definitions
```

### Required Fields

- **`version`**: Must be `"0.1"` (current supported version).
- **`steps`**: A list of step definitions.

### Optional Sections

- **`name`**: A unique name for the pipeline, useful for logging and identification.
- **`agents`**: Define inline agents for reuse within this file.
- **`imports`**: Import other pipeline files as reusable components.

## Step Types

Flujo supports several built-in step types, plus first-class custom primitives via the framework registry. The built-ins are listed below; custom primitives can be used by setting `kind` to the registered name (see “Extensible Step Primitives”).

### 1. Basic Step (`kind: step`)

The fundamental building block for executing a single operation.

```yaml
- kind: step
  name: process_data
  uses: agents.data_processor
  input: "{{ context.raw_data }}"
  config:
    max_retries: 3
    timeout_s: 60
  updates_context: true
```

**Use Cases:**
- Data processing, API calls, LLM interactions, file operations, or any single, focused task.

### 2. Parallel Step (`kind: parallel`)

Execute multiple operations concurrently and merge their results.

```yaml
- kind: parallel
  name: parallel_analysis
  merge_strategy: context_update
  on_branch_failure: ignore
  context_include_keys: ["user_id", "document_id"]
  branches:
    sentiment:
      - kind: step
        name: analyze_sentiment
        uses: agents.sentiment_analyzer
    keywords:
      - kind: step
        name: extract_keywords
        uses: agents.keyword_extractor
    summary:
      - kind: step
        name: generate_summary
        uses: agents.summarizer
```

**Use Cases:**
- Independent data processing, multiple API calls, parallel analysis, performance optimization, and fan-out/fan-in patterns.

**Merge Strategies:**
- `context_update`: (Default) Safely merges fields from each branch's context into the main context. Fails if two branches try to update the same field.
- `overwrite`: Merges all branch contexts, with later branches overwriting earlier ones in case of conflicts.
- `no_merge`: No context merging is performed.
- `error_on_conflict`: Fails the entire step if any merge conflict is detected.
- `merge_scratchpad`: Merges only the `scratchpad` dictionary from each branch.

**Branch Failure Handling:**
- `propagate`: (Default) Fail the entire step if any branch fails.
- `ignore`: Continue with successful branches; the step will succeed if at least one branch succeeds.

### 3. Conditional Step (`kind: conditional`)

Route execution to different branches based on a runtime condition.

```yaml
- kind: conditional
  name: route_by_type
  condition: "flujo.utils.routing:route_by_content_type" # A callable that returns a branch key
  default_branch: general # Optional branch to run if no key matches
  branches:
    code:
      - kind: step
        name: process_code
        uses: agents.code_processor
    text:
      - kind: step
        name: process_text
        uses: agents.text_processor
    general:
      - kind: step
        name: general_processing
        uses: agents.general_processor
```

**Use Cases:**
- Content type routing, user role-based workflows, error handling branches, and dynamic workflow selection.

### 4. Loop Step (`kind: loop`)

Execute a pipeline repeatedly until a condition is met.

```yaml
- kind: loop
  name: refinement_loop
  loop:
    body:
      - kind: step
        name: refine_content
        uses: agents.content_refiner
    max_loops: 5
    exit_condition: "flujo.utils.looping:quality_threshold_met"
```

**Use Cases:**
- Content refinement, iterative problem solving, and quality improvement loops.

**Loop Configuration:**
- `body`: The pipeline to execute in each iteration.
- `max_loops`: The maximum number of iterations.
- `exit_condition`: A callable that returns `True` to stop the loop.

### 5. Map Step (`kind: map`)

Apply a pipeline to each item in a collection from the context.

```yaml
- kind: map
  name: process_items
  map:
    iterable_input: "context.items_to_process"
    body:
      - kind: step
        name: process_single_item
        uses: agents.item_processor
        input: "{{ this }}" # 'this' refers to the current item in the iteration
```

**Use Cases:**
- Batch processing, collection transformation, and parallel item processing.

### 6. Dynamic Router (`kind: dynamic_router`)

Let an agent decide which branches to execute at runtime.

```yaml
- kind: dynamic_router
  name: smart_router
  router:
    router_agent: agents.workflow_router # This agent returns a list of branch names
    branches:
      billing:
        - kind: step
          name: handle_billing
          uses: agents.billing_handler
      support:
        - kind: step
          name: handle_support
          uses: agents.support_handler
```

**Use Cases:**
- AI-driven workflow selection, dynamic content routing, and intelligent task delegation.

### 7. Human-in-the-Loop Step (`kind: hitl`)

Pause pipeline execution to wait for human input or approval (supported in YAML).

```yaml
- kind: hitl
  name: user_approval
  message: "Please review and approve the generated content before proceeding"
  input_schema:
    type: object
    properties:
      confirmation: { type: string, enum: ["yes", "no"] }
      reasoning: { type: string }
    required: [confirmation]
```

**Use Cases:**
- Content approval workflows, human validation steps, manual quality checks, and interactive decision points.

**HITL Configuration:**
- `message`: (Optional) Message to display when the pipeline pauses.
- `input_schema`: (Optional) JSON Schema object used to validate the human response. Flujo compiles this to a Pydantic model internally.

When a `hitl` step runs, the pipeline pauses and records a message. Resume execution with the validated payload to continue.

### 8. State Machine Step (`kind: StateMachine`)

Drive execution across named states; each state maps to its own Pipeline.

```yaml
- kind: StateMachine
  name: sm
  start_state: analyze
  end_states: [refine]
  states:
    analyze:
      - kind: step
        name: Analyze
    refine:
      - kind: step
        name: Refine
```

For full details, see “State Machine Step”.

## Custom Step Primitives

Flujo supports first-class custom primitives via `flujo.framework.registry`. Once registered, you can reference them by `kind` in YAML, and they are executed by their corresponding policies.

See “Extensible Step Primitives” for registration and best practices.


## Agent Definitions

Define reusable agents inline within your YAML file.

### Basic Agent

```yaml
agents:
  text_processor:
    model: "openai:gpt-4o"
    system_prompt: "You are a text processing expert."
    output_schema:
      type: object
      properties:
        processed_text: { type: string }
        confidence: { type: number }
      required: [processed_text, confidence]
```

### Advanced Agent Configuration

```yaml
agents:
  advanced_processor:
    model: "openai:gpt-5"
    model_settings:
      reasoning: { effort: "high" }
      text: { verbosity: "medium" }
    system_prompt: |
      You are an advanced data processor.
      Process the input according to the specified schema.
    output_schema:
      type: object
      properties:
        result: { type: string }
        metadata: { type: object }
      required: [result]
    timeout: 180
    max_retries: 2
```

**Agent Properties:**
- `model`: LLM model identifier (e.g., `openai:gpt-4o`).
- `system_prompt`: Instructions for the agent.
- `output_schema`: The expected output structure (JSON Schema).
- `model_settings`: Model-specific configuration (e.g., for GPT-5).
- `timeout`: Execution timeout in seconds.
- `max_retries`: Retry attempts on failure.

## Pipeline Imports

Import other pipeline files to create modular, reusable workflows.

### Basic Import

```yaml
imports:
  data_processing: "./data_processing.yaml"
  validation: "./validation.yaml"

steps:
  - kind: step
    name: process_data
    uses: imports.data_processing
  - kind: step
    name: validate_results
    uses: imports.validation
```

**Note:** Imported pipelines are automatically wrapped as steps using `pipeline.as_step(name=...)`, providing the same functionality as the Python `as_step()` method.

### Relative Path Resolution

Paths are resolved relative to the current YAML file's location.

```yaml
imports:
  local: "./local_workflow.yaml"
  sibling: "../sibling_workflow.yaml"
  absolute: "/path/to/workflow.yaml" # Use with caution for portability
```

### Import Security

YAML imports of Python objects are restricted by an allow-list in `flujo.toml`:

```toml
# flujo.toml
blueprint_allowed_imports = ["my_safe_pkg", "my_safe_pkg.agents"]
```

## Configuration Options

### Step Configuration

```yaml
- kind: step
  name: configured_step
  config:
    max_retries: 3          # Retry attempts on failure
    timeout_s: 120          # Timeout in seconds
    temperature: 0.7        # LLM temperature (0.0-1.0)
    top_k: 50               # Top-k sampling
    top_p: 0.9              # Nucleus sampling
    preserve_fallback_diagnostics: true # Keep failure details even if fallback succeeds
```

### Step Flags

```yaml
- kind: step
  name: flagged_step
  updates_context: true     # Merge step output into the pipeline context
  validate_fields: true     # Validate that output fields match context fields
```

### Usage Limits

```yaml
- kind: step
  name: limited_step
  usage_limits:
    total_cost_usd_limit: 0.10
    total_tokens_limit: 10000
```

### Fallback Steps

Define a step to run if the primary step fails.

```yaml
- kind: step
  name: primary_step
  uses: agents.primary_agent
  fallback:
    kind: step
    name: fallback_step
    uses: agents.backup_agent
```

## Advanced Features

### Templated Input

Use Jinja-like templates for dynamic input values.

```yaml
steps:
  - kind: step
    name: process_user_input
    input: "{{ context.user_input }}"
  - kind: step
    name: follow_up
    input: "Based on: {{ previous_step }}"
  - kind: map
    name: process_items
    map:
      iterable_input: "context.items"
      body:
        - name: process_item
          input: "{{ this }}" # 'this' refers to the current item
```

**Template Variables:**
- `context`: The current pipeline context.
- `previous_step`: The output from the immediately preceding step.
- `this`: (Inside a `map` step) The current item from the iterable.

### Context Management

Control how step results and feedback are stored in the context.

```yaml
- kind: step
  name: context_aware_step
  updates_context: true
  validate_fields: true
  persist_feedback_to_context: "feedback_history" # Appends failure feedback to a context list
  persist_validation_results_to: "validation_results" # Appends validator results to a context list
```

### Plugins and Validators

Enhance steps with custom validation logic.

```yaml
- kind: step
  name: validated_step
  plugins:
    - "flujo.plugins.sql_validator"
    - path: "custom.plugin:CustomValidator"
      priority: 10
  validators:
    - "flujo.validators.json_schema"
    - "custom.validators:CustomValidator"
```

## Best Practices

1.  **Naming Conventions:** Use descriptive, action-oriented names for steps and agents.
2.  **Modular Design:** Break complex workflows into smaller, reusable pipelines and use `imports`.
3.  **Error Handling:** Use `fallback` steps and configure `max_retries` for resilience.
4.  **Context Management:** Use `context_include_keys` in parallel steps to limit data copying and `updates_context` judiciously.
5.  **Performance:** Use `parallel` and `map` steps for concurrent operations where appropriate.
6.  **Security:** Restrict Python object imports via `blueprint_allowed_imports` in `flujo.toml`.
7.  **Human-in-the-Loop:** Use HITL steps for approval workflows and manual validation. Prefer small, explicit `input_schema` objects so resumes are predictable and validatable.

### 8. Cache Step (`kind: cache`)

Wrap a step to cache its result for identical inputs.

```yaml
- kind: cache
  name: cached_stringify
  wrapped_step:
    kind: step
    name: stringify
    agent: { id: "flujo.builtins.stringify" }
```

**Use Cases:**
- Expensive or rate-limited operations (LLM calls, remote APIs) where identical inputs recur.

**Notes:**
- The default cache backend is used unless configured otherwise.
- Ensure the wrapped step is deterministic with respect to its input; avoid time-/random-dependent side effects.

## Running YAML Pipelines

### Command Line

```bash
# Basic execution
flujo run pipeline.yaml --input "Hello World"

# With context data
flujo run pipeline.yaml --input "Hello" --context-data '{"user_id": "123"}'

# With context file
flujo run pipeline.yaml --input "Hello" --context-file context.json
```

### Programmatic Execution

```python
from flujo import Pipeline, Flujo

# Load from YAML
pipeline = Pipeline.from_yaml_file("pipeline.yaml")

# Run the pipeline
runner = Flujo(pipeline)
result = runner.run("input_data")
```

## Troubleshooting

### Common Issues

1.  **Validation Errors**: Check that all required fields are present and correctly spelled.
2.  **Import Failures**: Verify import paths and the `blueprint_allowed_imports` list in `flujo.toml`.
3.  **Agent Resolution**: Ensure `uses` references are correct (e.g., `agents.my_agent` or `imports.my_alias`).
4.  **Template Errors**: Check syntax (`{{ ... }}`) and variable names (`context`, `previous_step`, `this`).
5.  **HITL Steps**: Human-in-the-Loop steps (`kind: hitl`) are not yet supported in YAML format. Use the Python DSL for HITL functionality.

### Validation Command

Run this command to check your pipeline for errors before execution.

```bash
flujo validate pipeline.yaml
```

## Conclusion

Flujo's YAML syntax provides a powerful, declarative way to define complex AI workflows. By understanding the available step types, configuration options, and best practices, you can create maintainable, scalable pipelines that leverage the full power of the Flujo framework.

The `as_step` functionality is available through the `imports` section, where imported pipelines are automatically wrapped as steps, providing the same modular composition capabilities as the Python API.

**Note on Step Type Support:** While most step types are fully supported in YAML, Human-in-the-Loop (HITL) steps currently require the Python DSL. This limitation will be addressed in future releases to provide complete YAML coverage for all Flujo step types.
 
