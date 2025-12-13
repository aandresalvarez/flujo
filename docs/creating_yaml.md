 
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
- `merge_scratchpad` has been removed; scratchpad no longer exists. Use `context_update` or explicit typed fields instead.

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

You can also use inline expressions instead of importing a Python function:

```yaml
- kind: conditional
  name: route_by_kind
  condition_expression: "previous_step.kind"  # Returns a branch key
  branches:
    text: [...]
    code: [...]
```

Security note
-------------
The `condition` field accepts only importable callables (e.g., `pkg.mod:func`).
Inline Python (such as `lambda ...`) in YAML is intentionally not supported for
security reasons. For inline logic, use `condition_expression`, or reference a
safe callable like `flujo.builtins.passthrough` when you already have a boolean
from a previous step.

Boolean condition expressions
----------------------------
Many condition expressions evaluate to a boolean. Flujo now treats these as a
first‑class routing pattern for YAML:

```yaml
- kind: conditional
  name: check_ok
  # Any expression that returns True/False
  condition_expression: "{{ previous_step.ok }}"
  branches:
    true:
      - kind: step
        name: on_true
        uses: agents.handle_success
    false:
      - kind: step
        name: on_false
        uses: agents.handle_failure
```

Notes:
- YAML parses unquoted `true`/`false` as booleans. The loader automatically coerces
  these branch keys to strings ("true"/"false") so validation succeeds.
- The execution policy also resolves boolean results by preferring exact boolean keys
  (for programmatic DSL usage) and falling back to string keys (for YAML). This keeps
  both styles working consistently.
- Telemetry stores the evaluated key in `executed_branch_key` and, when a fallback
  mapping occurs (e.g., `True` -> `"true"`), the `resolved_branch_key` metadata is added.

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

**Basic Loop Configuration:**
- `body`: The pipeline to execute in each iteration.
- `max_loops`: The maximum number of iterations.
- `exit_condition`: A callable that returns `True` to stop the loop.

Alternatively, use an inline expression for exit logic:

```yaml
- kind: loop
  name: conversational
  loop:
    body: [...]
    max_loops: 10
    exit_expression: "previous_step.action == 'finish'"
```

Declarative per-iteration state updates can be specified under `loop.state`:

```yaml
- kind: loop
  name: stateful
  loop:
    body: [...]
    max_loops: 5
    state:
      append:
        - target: "context.import_artifacts.history"
          value: "OUT: {{ previous_step }}"
      set:
        - target: "context.summary"
          value: "{{ previous_step }}"
      merge:
        - target: "context.metrics"
          value: '{"count": 1}'  # JSON object required for merge
```

Notes:
- Targets must begin with `context.` and may use dotted paths.
- Values are rendered using the same template syntax (including filters).
- Merge values must render to a JSON object; non-object values are ignored.

Declarative init/propagation/output (no Python helpers required):

```yaml
- kind: loop
  name: conversational_minimal
  loop:
    # 1) Init runs once before the first iteration (on isolated context)
    init:
      - append: { target: "context.import_artifacts.history", value: "User: {{ steps.get_goal.output }}" }

    # 2) Loop body (steps omitted)
    body:
      - kind: step
        name: clarify
        # ...
        updates_context: true

    # 3) Propagation chooses next iteration input
    propagation:
      next_input: context   # presets: context | previous_output | auto | template string

    # 4) Exit condition and final output mapping
    exit_expression: "context.import_artifacts.last_agent_command.action == 'finish'"
    output_template: "{{ context.import_artifacts.history | join('\\n') }}"
```

Friendly presets (natural-language aliases):

```yaml
- kind: loop
  name: conversational_friendly
  loop:
    conversation: true          # ensures history, propagates context, exit when agent finished
    init:
      history:
        start_with:
          from_step: get_goal
          prefix: "User: "
    body:
      - kind: step
        name: clarify
        updates_context: true
    stop_when: agent_finished   # alias for the common exit expression
    output:
      text: conversation_history  # newline-joined history as the final text

Auto propagation preset:

```yaml
- kind: loop
  name: conversational_auto
  loop:
    body:
      - kind: step
        name: clarify
        updates_context: true
    propagation: auto   # chooses 'context' because a body step updates the context
    exit_expression: "context.import_artifacts.counter >= 2"
```
```

Guidance:
- `init` is executed once and safely on the iteration clone; it never mutates the shared context directly.
- `propagation.next_input` supports `context`, `previous_output` (default), or a template string.
- `output_template` formats a single final string; `output: { ... }` builds an object from templates.
- Control-flow exceptions (pause/abort/redirect) are never swallowed; they propagate per the team guide.

**Enhanced Loop Configuration:**
For sophisticated agentic workflows, you can also specify input/output mappers:

```yaml
- kind: loop
  name: conversational_loop
  loop:
    body:
      - kind: step
        name: planner
        uses: agents.conversation_planner
      - kind: step
        name: executor
        uses: agents.command_executor
    initial_input_mapper: "skills.helpers:map_initial_goal"
    iteration_input_mapper: "skills.helpers:map_conversation_state"
    exit_condition: "skills.helpers:is_conversation_complete"
    loop_output_mapper: "skills.helpers:map_final_result"
    max_loops: 10
```

**Enhanced Loop Configuration Keys:**
- `body`: The pipeline to execute in each iteration
- `max_loops`: Maximum number of iterations (prevents infinite loops)
- `exit_condition`: Callable that returns `True` to stop the loop
- `initial_input_mapper`: **NEW** - Maps LoopStep input to first iteration's body input
- `iteration_input_mapper`: Maps previous iteration output to next iteration input
- `loop_output_mapper`: Maps final successful output to LoopStep output

**Use Cases:**
- Content refinement, iterative problem solving, and quality improvement loops
- **Conversational AI workflows** with structured data transformation
- **Agentic planning and execution cycles** with state management
- **Multi-step refinement processes** with context preservation

**Benefits of Enhanced Loops:**
- **Clean Data Transformation**: No more adapter steps or complex agent logic
- **Consistent State Management**: Maintains conversation context across iterations
- **Rich Output**: Comprehensive results with metadata and conversation history
- **Declarative Configuration**: Intuitive YAML syntax for complex workflows
- **Backward Compatibility**: Existing loop configurations continue to work unchanged

**Practical Example - Conversational Clarification Loop:**
```yaml
- kind: loop
  name: clarification_loop
  loop:
    body:
      - kind: step
        name: analyze_goal
        uses: agents.goal_analyzer
      - kind: step
        name: ask_clarification
        agent:
          id: "flujo.builtins.ask_user"
        input: "{{ previous_step }}"
    initial_input_mapper: "skills.helpers:map_initial_goal"
    iteration_input_mapper: "skills.helpers:map_conversation_state"
    exit_condition: "skills.helpers:is_goal_complete"
    loop_output_mapper: "skills.helpers:map_final_goal"
    max_loops: 5
```

This pattern enables sophisticated conversational workflows where:
- The `initial_input_mapper` transforms the user's initial goal into a structured format
- The `iteration_input_mapper` maintains conversation context and history
- The `exit_condition` determines when the goal is sufficiently clarified
- The `loop_output_mapper` provides a comprehensive final result

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

Imports inside StateMachine states
---------------------------------

You can use `uses: imports.<alias>` within `states` to compose precompiled child pipelines. The loader compiles these into `ImportStep`s, so you can leverage `inherit_context`, `input_to`, and `outputs` mapping just like a top‑level import.

Example:

```yaml
version: "0.1"
imports:
  clar: "./clarification/pipeline.yaml"
steps:
  - kind: StateMachine
    name: sm
    start_state: s1
    end_states: [done]
    states:
      s1:
        - name: Clarification
          uses: imports.clar
          updates_context: true
          config:
            inherit_context: true
            outputs:
              - { child: "import_artifacts.foo", parent: "import_artifacts.foo" }
      done:
        - kind: step
          name: Done
```

Tips:
- Set `updates_context: true` on the import step and use `config.outputs` to map explicit fields from the child back to the parent context.
- Use `config.input_to` (`initial_prompt` | `import_artifacts` | `both`) to route the parent input into the child. For scalar routing into import_artifacts, `config.input_scratchpad_key` can help.

### 9. Agentic Loop (`kind: agentic_loop`)

YAML sugar to define a conversational loop using the built-in factory.

```yaml
- kind: agentic_loop
  name: clarification_loop
  planner: "agents.clarifier"          # or import path
  registry: "agents.registry"          # dict mapping tool-name -> agent
  # Optional: classic loop knobs
  config:
    max_retries: 5                      # per-step retries inside the loop
  # Optional: render final result via template (applies after internal mapping)
  output_template: "FINAL: {{ previous_step.execution_result }}"
```

This compiles to the existing LoopStep powered by `recipes.make_agentic_loop_pipeline`. It does not change executor behavior.

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

### Template Filters

You can apply deterministic, side-effect-free filters to template placeholders to transform values inline. Filters operate on the resolved placeholder value and can be chained.

- join(delim): Join iterable values into a string with the given delimiter.
  - Example: `"{{ steps.items | join(', ') }}"` → `"a, b, c"`
  - Works on `list`/`tuple`; other types are stringified.
- upper: Convert to uppercase. Example: `"{{ previous_step | upper }}"`
- lower: Convert to lowercase. Example: `"{{ previous_step | lower }}"`
- length: Length of strings, lists, or dicts. Example: `"{{ context.tags | length }}"`
- tojson: JSON-safe serialization. Example: `"{{ context.payload | tojson }}"`

Filters compose after fallback expressions. For example:

```yaml
input: "{{ context.maybe_value or 'default' | upper }}"  # → DEFAULT if maybe_value is empty
```

Notes:
- Unknown filters raise an explicit error during template rendering.
- Filters only operate on already-resolved values; they do not evaluate code or access the environment.
- Allow-list: configure enabled filters in `flujo.toml` under `[settings]` using `enabled_template_filters`. Example:
  ```toml
  [settings]
  enabled_template_filters = ["upper", "join", "tojson"]
  ```

### Enhanced Loop Mappers

Flujo now supports comprehensive input/output mapping for sophisticated agentic loops directly in YAML. This eliminates the need for adapter steps and complex agent logic when building conversational workflows.

**Key Features:**
- **`initial_input_mapper`**: Transform LoopStep input for the first iteration
- **`iteration_input_mapper`**: Map previous iteration output to next iteration input
- **`loop_output_mapper`**: Transform final successful output to LoopStep result
- **Full YAML Support**: All mapper functionality available declaratively
- **Backward Compatibility**: Existing loops continue to work unchanged

**Common Use Cases:**
- Conversational AI with context preservation
- Iterative refinement with state management
- Agentic planning with conversation history
- Quality improvement loops with progressive enhancement

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
- `steps`: Map of prior steps by name (values are proxies exposing `.output/.result/.value`).
- `this`: (Inside a `map` step) The current item from the iterable.

### Previous Step vs Steps Map

Flujo exposes the last step’s value directly as `previous_step` and also exposes a map of prior steps under `steps.<name>`.

- `previous_step` is the raw output value of the last step. It does not have `.output`/`.result`/`.value` attributes.
- `steps.<name>` entries are proxies that do expose `.output`/`.result`/`.value` for convenience.
- Use the `tojson` filter when you want a JSON string for a structured value.

Examples:

```yaml
# ✅ Correct: serialize the raw previous value
input: "{{ previous_step | tojson }}"

# ✅ Correct: access a named prior step via proxy
input: "{{ steps.generate_greeting.output | tojson }}"

# ❌ Incorrect: previous_step is a raw value (no .output); this becomes null
input: "{{ previous_step.output | tojson }}"
```

Validation Tips

- Prefer `tojson` when interpolating structured values; avoid `previous_step.output`.
- Use `steps.<name>.output` to reference a prior step’s output through the proxy.
- Suppressing specific rules:
  - Inline comment on a step or list item: `# flujo: ignore V-T1 V-P3` (supports globs, e.g., `V-*`).
  - Programmatic (Python): `step.meta['suppress_rules'] = ["V-T*"]`.

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

### Built-in Data Transforms

Reduce Python glue in YAML by using side-effect-free, CPU-only transforms registered under `flujo.builtins.*`. These do not consume tokens or cost.

- to_csv: Convert `list[dict]` to a CSV string.
  - Parameters: `headers` (optional `list[str]`). If omitted, headers are the sorted union of keys across rows. Extra keys are ignored when headers are provided.
  - Example:
    ```yaml
    - kind: step
      name: to_csv
      agent:
        id: "flujo.builtins.to_csv"
        params: { headers: ["id", "price"] }
      input: "{{ previous_step }}"  # rows (list[dict])
    ```

- aggregate: Aggregate numeric values in a `list[dict]`.
  - Parameters: `operation` ("sum" | "avg" | "count"), `field` (optional for count of all rows; required for sum/avg).
  - Non-numeric values are ignored in `sum/avg`.
  - Example:
    ```yaml
    - kind: step
      name: total_price
      agent: { id: "flujo.builtins.aggregate", params: { operation: "sum", field: "price" } }
      input: "{{ steps.fetch_prices }}"  # list[dict]
    ```

- select_fields: Project and/or rename fields on a dict or list of dicts.
  - Parameters: `include` (optional `list[str]`), `rename` (optional `map[str,str]`).
  - If only `rename` is provided, other keys are preserved; renamed keys are rewritten.
  - Example:
    ```yaml
    - kind: step
      name: view
      agent:
        id: "flujo.builtins.select_fields"
        params:
          include: ["id", "name"]
          rename: { name: "display_name" }
      input: "{{ steps.fetch_users }}"  # dict or list[dict]
    ```

- flatten: Flatten one level of a nested list.
  - Parameters: none.
  - Non-list elements are passed through in the output sequence; non-list input produces an empty list.
  - Example:
    ```yaml
    - kind: step
      name: flattened
      agent: { id: "flujo.builtins.flatten" }
      input: "{{ steps.batch_chunks }}"  # e.g., [[1,2],[3],(4,5),6]
    ```

## Best Practices

1.  **Naming Conventions:** Use descriptive, action-oriented names for steps and agents.
2.  **Modular Design:** Break complex workflows into smaller, reusable pipelines and use `imports`.
3.  **Error Handling:** Use `fallback` steps and configure `max_retries` for resilience.
4.  **Context Management:** Use `context_include_keys` in parallel steps to limit data copying and `updates_context` judiciously.
5.  **Performance:** Use `parallel` and `map` steps for concurrent operations where appropriate.
6.  **Security:** Restrict Python object imports via `blueprint_allowed_imports` in `flujo.toml`.
7.  **Human-in-the-Loop:** Use HITL steps for approval workflows and manual validation. Prefer small, explicit `input_schema` objects so resumes are predictable and validatable.

See also: Guides → [Cost and Budget Governance](guides/cost_and_budget_governance.md) for centrally managed budgets and governance controls that apply to YAML pipelines.

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
5.  **HITL Steps**: Human-in-the-Loop steps (`kind: hitl`) are supported in YAML. See docs/hitl.md and the imports demo under `examples/imports_demo/main_with_hitl.yaml`.

### Validation Command

Run this command to check your pipeline for errors before execution.

```bash
flujo validate pipeline.yaml
```

## Real-World Development Insights

This section documents key learnings from building and debugging complex Flujo YAML pipelines, particularly insights gained from real-world development exercises.

### 1. **Built-in Skills: Your First Choice**

**Key Learning**: Flujo's built-in skills are more powerful than initially apparent and can handle surprisingly complex workflows without custom development.

```yaml
# ✅ Leverage built-in skills for common patterns
- kind: step
  name: get_user_input
  agent:
    id: "flujo.builtins.ask_user"
  input: "What would you like to accomplish?"

- kind: step
  name: stringify_response
  agent:
    id: "flujo.builtins.stringify"
  input: "{{ previous_step }}"

- kind: step
  name: web_search
  agent:
    id: "flujo.builtins.web_search"
    params:
      query: "{{ previous_step }}"
      max_results: 5
```

**Available Built-in Skills:**
- **Human Interaction**: `flujo.builtins.ask_user`, `flujo.builtins.check_user_confirmation`
- **Data Processing**: `flujo.builtins.stringify`, `flujo.builtins.extract_from_text`
- **Web Operations**: `flujo.builtins.web_search`, `flujo.builtins.http_get`
- **File Operations**: `flujo.builtins.fs_write_file`
- **AI Operations**: `flujo.builtins.extract_from_text`

**Guidelines:**
- **Start with built-ins**: Explore `flujo.builtins.*` namespace before building custom solutions
- **Combine multiple built-ins**: Chain simple skills to create complex behaviors
- **Use `flujo.builtins.discover_skills`**: Run this to see all available capabilities
- **Test built-in combinations**: Often you can achieve your goal without custom development

### 2. **Skill Discovery and Troubleshooting**

**Key Learning**: Understanding when and how skills are loaded is crucial for successful pipeline execution.

**Debugging Skill Issues:**
```bash
# ✅ Check what skills are available
flujo run debug_skills.yaml

# ✅ Verify skill registry contents
python3 -c "from flujo.infra.skill_registry import get_skill_registry; print(list(get_skill_registry()._entries.keys()))"

# ✅ Test minimal pipeline first
flujo run test_minimal.yaml
```

**Common Error Patterns:**
```bash
# ❌ "Unknown skill id" usually means:
# - Skill not registered in the registry
# - Skill loading happens after pipeline compilation
# - Import path issues in custom skills

# ✅ Debug with:
flujo --debug run pipeline.yaml
```

**Guidelines:**
- **Built-in skills load automatically**: No registration needed
- **Custom skills require explicit registration**: Use programmatic registration in `__init__.py` files
- **Test skill availability**: Use `flujo.builtins.discover_skills` to verify what's loaded
- **Understand loading timing**: Skills must be available before pipeline compilation

### 3. **Building Interactive Workflows with YAML**

**Key Learning**: Complex interactive workflows can be built using only built-in skills and proper YAML composition.

```yaml
# ✅ Example: Multi-step clarification workflow using built-ins
version: "0.1"
name: "clarification_workflow"

agents:
  clarification_agent:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      Analyze the user's goal and either:
      - Ask ONE clarifying question if more details are needed
      - Respond with "COMPLETE: " followed by a summary
    output_schema:
      type: string

steps:
  - kind: step
    name: get_initial_goal
    agent:
      id: "flujo.builtins.ask_user"
    input: "What would you like to accomplish?"

  - kind: step
    name: analyze_and_clarify
    uses: agents.clarification_agent
    input: "Initial goal: {{ previous_step }}"

  - kind: step
    name: stringify_response
    agent:
      id: "flujo.builtins.stringify"
    input: "{{ previous_step }}"

  - kind: step
    name: get_clarification
    agent:
      id: "flujo.builtins.ask_user"
    input: "{{ previous_step }}"
```

**Interactive Workflow Patterns:**
- **Use HITL steps strategically**: Combine `ask_user` with AI agents for intelligent interactions
- **Stringify complex outputs**: Use `flujo.builtins.stringify` to handle type conversions
- **Build incrementally**: Add complexity one step at a time
- **Test each addition**: Ensure the pipeline works before adding more steps

### 4. **When to Use Custom Skills vs. Built-ins**

**Decision Framework:**
```yaml
# ✅ Use Built-ins When:
# - You need human interaction (ask_user, check_user_confirmation)
# - You need data processing (stringify, web_search, http_get)
# - You need file operations (fs_write_file)
# - You need basic AI operations (extract_from_text)
# - You can combine multiple built-ins to achieve your goal

# 🔧 Consider Custom Skills When:
# - You need domain-specific business logic
# - You need to integrate with external APIs not covered by built-ins
# - You need complex state management beyond what YAML provides
# - You've exhausted built-in capabilities and combinations
```

**Custom Skill Implementation Pattern:**
```python
# skills/__init__.py - Programmatic registration
from flujo.infra.skill_registry import get_skill_registry
from .helpers import my_custom_function

registry = get_skill_registry()
registry.register(
    id="my_custom_skill",
    factory=my_custom_function,
    description="Custom business logic"
)
```

### 5. **Testing and Validation Best Practices**

**Key Learning**: Test skill availability and pipeline loading separately from execution.

```bash
# ✅ Test skill discovery
flujo run debug_skills.yaml

# ✅ Test minimal pipeline
flujo run test_minimal.yaml

# ✅ Test full pipeline
flujo run main_pipeline.yaml

# ✅ Validate pipeline structure
flujo validate main_pipeline.yaml
```

**Testing Checklist:**
- [ ] Skills are discoverable (`flujo.builtins.discover_skills`)
- [ ] Minimal pipeline loads without errors
- [ ] Each step can execute independently
- [ ] Full pipeline completes successfully
- [ ] Error handling works as expected
- [ ] Performance meets requirements

### 6. **Performance and Reliability Considerations**

**Key Learning**: Built-in skills are optimized and tested, while custom skills require additional validation.

```yaml
# ✅ Built-in skills have built-in optimization
- kind: step
  name: web_search
  agent:
    id: "flujo.builtins.web_search"
    params:
      query: "{{ context.search_term }}"
      max_results: 3

# 🔧 Custom skills need explicit configuration
- kind: step
  name: custom_operation
  agent:
    id: "my_custom_skill"
  config:
    timeout_s: 60
    max_retries: 2
```

**Guidelines:**
- **Built-ins are production-ready**: They include proper error handling and retries
- **Custom skills need testing**: Validate error handling and edge cases
- **Monitor performance**: Built-ins include cost tracking and performance metrics
- **Use appropriate timeouts**: Set realistic timeouts for custom operations

### 7. **Migration Path from Custom to Built-in Skills**

**Key Learning**: Many custom skill requirements can be satisfied by combining built-in capabilities.

```yaml
# ❌ Before: Custom command executor
# skills.yaml
command_executor:
  path: "skills:command_executor_factory"

# ✅ After: Built-in pattern
- kind: step
  name: get_user_input
  agent:
    id: "flujo.builtins.ask_user"
  input: "{{ context.question }}"

- kind: step
  name: process_response
  uses: agents.response_processor
  input: "{{ previous_step }}"
```

**Migration Strategy:**
1. **Inventory your custom skills**: List what each one does
2. **Map to built-ins**: Find equivalent or combinable built-in skills
3. **Refactor incrementally**: Replace one skill at a time
4. **Test thoroughly**: Ensure the new approach works as expected
5. **Document patterns**: Share successful built-in combinations with your team

### 8. **Enhanced Loop Mappers for Conversational AI**

**Key Learning**: Flujo's enhanced loop mappers provide a declarative solution for complex conversational workflows that previously required custom adapter steps.

**Before Enhanced Loops:**
```yaml
# ❌ Required complex workarounds
- kind: step
  name: adapter_step
  uses: agents.complex_adapter
  input: "{{ context.user_goal }}"

- kind: loop
  name: basic_loop
  loop:
    body:
      - kind: step
        name: process
        uses: agents.processor
    max_loops: 5
    exit_condition: "helpers:is_complete"
```

**After Enhanced Loops:**
```yaml
# ✅ Clean, declarative configuration
- kind: loop
  name: conversational_loop
  loop:
    body:
      - kind: step
        name: process
        uses: agents.processor
    initial_input_mapper: "skills.helpers:map_initial_goal"
    iteration_input_mapper: "skills.helpers:map_conversation_state"
    exit_condition: "skills.helpers:is_complete"
    loop_output_mapper: "skills.helpers:map_final_result"
    max_loops: 5
```

**Benefits:**
- **Eliminates Adapter Steps**: Direct mapping between input and loop body expectations
- **Maintains Context**: Conversation history and state preserved across iterations
- **Rich Output**: Comprehensive results with metadata and conversation flow
- **Declarative**: Intuitive YAML syntax for complex workflows
- **Production Ready**: Built-in error handling and validation

### 9. **Advanced YAML Composition Patterns**

**Key Learning**: Complex workflows can be built by combining multiple YAML features effectively.

```yaml
# ✅ Example: Conditional routing with built-in skills
- kind: conditional
  name: content_router
  condition: "flujo.builtins.has_yaml_key"
  branches:
    present:
      - kind: step
        name: process_yaml
        agent:
          id: "flujo.builtins.extract_yaml_text"
    absent:
      - kind: step
        name: create_yaml
        uses: agents.yaml_generator

# ✅ Example: Parallel processing with built-ins
- kind: parallel
  name: parallel_analysis
  merge_strategy: context_update
  branches:
    sentiment:
      - kind: step
        name: analyze_sentiment
        agent:
          id: "flujo.builtins.extract_from_text"
        input: |
          Extract sentiment from: {{ context.text }}
          Schema: {{ context.sentiment_schema }}
    keywords:
      - kind: step
        name: extract_keywords
        agent:
          id: "flujo.builtins.extract_from_text"
        input: |
          Extract keywords from: {{ context.text }}
          Schema: {{ context.keyword_schema }}
```

**Composition Guidelines:**
- **Combine step types**: Use `conditional`, `parallel`, and `loop` with built-in skills
- **Leverage templating**: Use `{{ context.* }}` and `{{ previous_step }}` effectively
- **Chain operations**: Build complex workflows from simple, focused steps
- **Test combinations**: Ensure each composition works before building further

---

## Summary of Key Insights

The real-world development experience revealed several critical insights for Flujo YAML development:

1. **Built-in skills are underestimated** - They can handle surprisingly complex workflows
2. **Skill discovery timing matters** - Skills must be available before pipeline compilation
3. **Systematic debugging is essential** - Question assumptions and test hypotheses methodically
4. **YAML-driven workflows are powerful** - When skills work, the declarative approach scales well
5. **Start simple, add complexity incrementally** - Test each addition before building further
6. **Enhanced loop mappers eliminate complexity** - Conversational AI workflows are now declarative and intuitive

**Remember**: Flujo's built-in capabilities are often sufficient for your needs. Explore them thoroughly before building custom solutions, and always test incrementally to catch issues early.

## Conclusion

Flujo's YAML syntax provides a powerful, declarative way to define complex AI workflows. By understanding the available step types, configuration options, and best practices, you can create maintainable, scalable pipelines that leverage the full power of the Flujo framework.

### Imports & Composition (ImportStep)

Imported blueprints are compiled relative to the parent YAML directory and wrapped as a first‑class ImportStep with policy‑driven execution. This enables predictable input propagation and context merging without ad‑hoc adapters.

Key options (under `steps[*].config` when `uses: imports.<alias>`):
- `input_to`: Where to project the parent step input for the child run. One of `initial_prompt`, `import_artifacts`, or `both`.
- `input_scratchpad_key`: Key name used when the input is a scalar and `input_to: import_artifacts` or `both` (default: `initial_input`).
- `outputs`: List of mappings `{ child: <path>, parent: <path> }` for deterministic merges when `updates_context: true`.
- `inherit_context`: Whether to inherit and deep‑copy the parent context into the child run (default: false).
- `inherit_conversation`: Whether HITL prompts from the child participate in the parent conversation (default: true).
- `propagate_hitl`: If `true`, a HITL pause inside the child import propagates to the parent so the runner surfaces the question and pauses the parent run (default: true).

Input precedence and conversation
- When `input_to` is set, the mapped value is used as the child’s actual initial input. For `initial_prompt`, dict/list inputs are JSON‑encoded. This explicit input takes precedence over any inherited conversation turns, preventing accidental “previous assistant message” bleed‑through.
- `inherit_conversation: true` still preserves conversation history for continuity and traceability, but it will not override the explicit input provided via `input_to`.

Example — three imported pipelines chained end‑to‑end without re‑prompting:

```yaml
version: "0.1"
imports:
  clarification: "clarification.yaml"
  concept_discovery: "concept_discovery.yaml"
  query_builder: "query_builder.yaml"

steps:
  - kind: step
  name: clarification
  uses: imports.clarification
  updates_context: true
  config:
    input_to: initial_prompt
    propagate_hitl: true  # Surface child HITL questions to the parent
    outputs:
      - { child: import_artifacts.cohort_definition, parent: import_artifacts.cohort_definition }

  - kind: step
    name: concept_discovery
    uses: imports.concept_discovery
    updates_context: true
    config:
      input_to: import_artifacts
      outputs:
        - { child: import_artifacts.concept_sets, parent: import_artifacts.concept_sets }

  - kind: step
    name: query_builder
    uses: imports.query_builder
    updates_context: true
    config:
      input_to: import_artifacts
      outputs:
        - { child: import_artifacts.final_sql, parent: import_artifacts.final_sql }
```

Notes:
- Child‑local skills resolve robustly: `skills.*` modules next to each child YAML are resolved relative to that child with per‑import namespace isolation. This avoids `sys.modules` collisions across multiple imported children with the same `skills` package name and works in both `dev validate` and `run` without PYTHONPATH hacks.
- For JSON inputs to child `initial_prompt`, dict/list inputs are JSON‑encoded; for `import_artifacts` inputs, dicts deep‑merge and scalars store under `input_scratchpad_key`.
- Control-flow propagation:
  - HITL pauses propagate to the parent only when `propagate_hitl: true`.
  - Aborts/redirects always propagate to the parent (independent of `propagate_hitl`).
  - Context merges occur only on successful child completion.
- When a downstream child needs to read fields written by an upstream child, set `inherit_context: true` on that import step so the child receives the parent context built so far.

**Enhanced Loop Support:** Flujo now provides comprehensive YAML support for sophisticated loop workflows through enhanced mappers (`initial_input_mapper`, `iteration_input_mapper`, `loop_output_mapper`). This enables declarative conversational AI patterns and complex iterative workflows without requiring custom adapter steps.

**Note on Step Type Support:** All core step types, including Human-in-the-Loop (`kind: hitl`), are supported in YAML. For a working example that pauses and then resumes within an imported child, see `examples/imports_demo/main_with_hitl.yaml`.
 
MapStep pre/post hooks (sugars):

```yaml
- kind: map
  name: summarize_values
  map:
    iterable_input: items
    body:
      - kind: step
        name: transform
        # ...
    init:
      - set: "context.import_artifacts.note"
        value: "mapping"
    finalize:
      output:
        results_str: "{{ previous_step }}"   # sees the aggregated list
```

Notes:
- `map.init` runs once before mapping begins (idempotent; isolated per run).
- `map.finalize` sees the aggregated results list as `previous_step` and can format
  a single string (`output_template`) or build an object (`output: { ... }`).
- Control‑flow exceptions propagate; quotas unchanged.
Parallel reduction (sugar):

```yaml
- kind: parallel
  name: gather
  branches:
    a:
      - kind: step
        name: step_a
    b:
      - kind: step
        name: step_b
  reduce: keys     # presets: keys | values | union | concat | first | last
```

Notes:
- `reduce: keys` returns branch names in declared order.
- `reduce: values` returns outputs in branch order.
- `reduce: union` merges dict outputs with last-wins (branch order).
- `reduce: concat` concatenates list outputs (non-list values are appended).
- `reduce: first|last` picks the first/last available branch output.
- Without `reduce`, the default output is a map: `{branch: output|StepResult}`.
### CLI Wizard and Explain

Generate a minimal, natural YAML via a friendly wizard:

```bash
uv run flujo create --wizard --non-interactive \
  --goal "Summarize latest news" \
  --name clarification_loop \
  --output-dir ./myproj
```

Explain a YAML file in plain language:

```bash
uv run flujo explain ./pipeline.yaml
```

The wizard uses the natural presets documented above (conversation, propagation: auto,
stop_when: agent_finished, and simple output) and writes a ready-to-run YAML.

Advanced wizard flags:

- `--wizard-pattern loop|map|parallel` (default: loop)
- For map: `--wizard-iterable-name <field>` (default: items)
- For parallel: `--wizard-reduce-mode keys|values|union|concat|first|last` (default: keys)
