# Flujo YAML Best Practices and Examples

This document provides comprehensive best practices, patterns, and examples for building robust and maintainable Flujo pipelines using YAML syntax.

## Table of Contents

1. [Core Principles](#core-principles)
2. [File Structure and Organization](#file-structure-and-organization)
3. [Agent Definition Best Practices](#agent-definition-best-practices)
4. [Step Design Patterns](#step-design-patterns)
5. [Pipeline Composition Strategies](#pipeline-composition-strategies)
6. [Error Handling and Resilience](#error-handling-and-resilience)
7. [Performance Optimization](#performance-optimization)
8. [Testing and Validation](#testing-and-validation)
9. [Real-World Examples](#real-world-examples)
10. [Common Anti-Patterns](#common-anti-patterns)

## Core Principles

### 1. **Single Responsibility Principle**
Each step should have one clear purpose. Break complex operations into multiple focused steps.

```yaml
# ❌ Bad: Single step doing multiple things
- kind: step
  name: process_and_validate_data
  uses: agents.super_agent

# ✅ Good: Separate concerns
- kind: step
  name: process_data
  uses: agents.data_processor
- kind: step
  name: validate_data
  uses: agents.validator
```

### 2. **Explicit Naming**
Use descriptive, action-oriented names that clearly indicate what each step does.

```yaml
# ❌ Bad: Generic names
- kind: step
  name: step1
  uses: agents.agent1

# ✅ Good: Descriptive names
- kind: step
  name: analyze_user_sentiment
  uses: agents.sentiment_analyzer
- kind: step
  name: extract_key_insights
  uses: agents.insight_extractor
```

### 3. **Progressive Enhancement**
Start simple and add complexity only when needed. Avoid over-engineering.

```yaml
# Start with a basic step
- kind: step
  name: process_input
  uses: agents.processor

# Add complexity incrementally as needed
- kind: step
  name: process_input
  uses: agents.processor
  config:
    max_retries: 3
    timeout_s: 60
  fallback:
    kind: step
    name: fallback_processor
    uses: agents.backup_processor
```

## File Structure and Organization

### Recommended Directory Structure

```
project/
├── pipelines/
│   ├── main.yaml              # Main pipeline
│   ├── data_processing.yaml   # Data processing sub-pipeline
│   └── validation.yaml        # Validation sub-pipeline
├── agents/
│   └── main_agents.yaml       # Agent definitions
├── skills/
│   ├── __init__.py
│   └── custom_tools.py        # Custom Python skills
├── skills.yaml                # Skills catalog (optional)
└── flujo.toml                 # Project configuration
```

### Pipeline Organization

```yaml
# pipelines/main.yaml
version: "0.1"
name: "Data Analysis Pipeline"

imports:
  data_processing: "./data_processing.yaml"
  validation: "./validation.yaml"

agents:
  orchestrator:
    model: "openai:gpt-4o"
    system_prompt: "You are a data pipeline orchestrator."

steps:
  - kind: step
    name: orchestrate_flow
    uses: agents.orchestrator
  - kind: step
    name: process_data
    uses: imports.data_processing
  - kind: step
    name: validate_results
    uses: imports.validation
```

## Agent Definition Best Practices

### 1. **Clear System Prompts**
Write specific, actionable system prompts that clearly define the agent's role, task, and constraints.

```yaml
agents:
  # ❌ Bad: Vague prompt
  processor:
    model: "openai:gpt-4o"
    system_prompt: "Process data well."

  # ✅ Good: Specific, actionable prompt
  data_processor:
    model: "openai:gpt-4o"
    system_prompt: |
      You are a data processing expert. Your task is to:
      1. Clean and normalize the input text.
      2. Remove duplicate words and invalid characters.
      3. Return a structured JSON object according to the schema.
      
      Always validate your output against the required schema.
```

### 2. **Structured Output Schemas**
Define clear, validated output schemas to ensure consistent and reliable results.

```yaml
agents:
  sentiment_analyzer:
    model: "openai:gpt-4o"
    system_prompt: "Analyze the sentiment of the given text."
    output_schema:
      type: object
      properties:
        sentiment: 
          type: string
          enum: ["positive", "negative", "neutral"]
        confidence: 
          type: number
          minimum: 0
          maximum: 1
        reasoning: 
          type: string
      required: [sentiment, confidence]
```

### 3. **Model-Specific Configuration**
Use `model_settings` for provider-specific controls (e.g., for GPT-5).

```yaml
agents:
  creative_writer:
    model: "openai:gpt-5"
    model_settings:
      reasoning: { effort: "medium" }
      text: { verbosity: "high" }
    system_prompt: "Write a creative story."
    output_schema: { type: "string" }
```

### 4. **Timeout and Retry Configuration**
Set appropriate timeouts and retries at the **step level**, not in the agent definition.

```yaml
# ✅ Good: Configuration is part of the step
- kind: step
  name: complex_analysis
  uses: agents.analyzer
  config:
    timeout_s: 180
    max_retries: 3
```

## Step Design Patterns


### 5. **Human-in-the-Loop (HITL) Gate**
Use a HITL step to pause execution and wait for human approval or structured input. Keep schemas minimal and explicit to ensure valid resumes.

```yaml
- kind: hitl
  name: get_user_approval
  message: "Approve the generated plan? (yes/no)"
  input_schema:
    type: object
    properties:
      confirmation: { type: string, enum: ["yes", "no"] }
      reason: { type: string }
    required: [confirmation]
- kind: conditional
  name: check_approval
  condition: "flujo.builtins.check_user_confirmation_sync"
  branches:
    approved:
      - kind: step
        name: proceed
        uses: agents.executor
    denied:
      - kind: step
        name: abort
        uses: agents.logger
```

**Guidelines:**
- Use `message` to provide short, actionable instructions.
- Prefer JSON Schema for `input_schema`; the engine compiles it to a Pydantic model.
- In conditionals, use the synchronous `check_user_confirmation_sync` for declarative YAML.

### 6. **Cache Wrapper**
Cache the result of expensive or rate-limited operations. Wrap only deterministic steps.

```yaml
- kind: cache
  name: cached_summarize
  wrapped_step:
    kind: step
    name: summarize
    uses: agents.summarizer
    input: "{{ context.article_text }}"
```

**Guidelines:**
- Ensure the wrapped step is deterministic with respect to input.
- Avoid caching steps that read external state without including it in the input.
- Start with the default cache backend; consider storage/backends later if needed.


### 1. **Sequential Processing Pattern**
For linear workflows where each step depends on the previous one.

```yaml
steps:
  - kind: step
    name: extract_data
    uses: agents.data_extractor
    updates_context: true
  
  - kind: step
    name: transform_data
    uses: agents.data_transformer
    input: "{{ context.extracted_data }}"
    updates_context: true
  
  - kind: step
    name: load_data
    uses: agents.data_loader
    input: "{{ context.transformed_data }}"
```

### 2. **Parallel Processing Pattern**
For independent operations that can run concurrently.

```yaml
- kind: parallel
  name: parallel_analysis
  merge_strategy: context_update
  branches:
    sentiment:
      - kind: step
        name: analyze_sentiment
        uses: agents.sentiment_analyzer
        input: "{{ context.text_content }}"
    keywords:
      - kind: step
        name: extract_keywords
        uses: agents.keyword_extractor
        input: "{{ context.text_content }}"
```

### 3. **Conditional Routing Pattern**
For dynamic workflow selection based on content or context.

```yaml
- kind: conditional
  name: content_router
  condition: "flujo.utils.routing:route_by_content_type"
  branches:
    code:
      - kind: step
        name: process_code
        uses: agents.code_processor
    text:
      - kind: step
        name: process_text
        uses: agents.text_processor
```

### 4. **Loop Pattern**
For iterative refinement or quality improvement.

```yaml
- kind: loop
  name: quality_improvement_loop
  loop:
    body:
      - kind: step
        name: improve_content
        uses: agents.content_improver
      - kind: step
        name: evaluate_quality
        uses: agents.quality_evaluator
    max_loops: 5
    exit_condition: "flujo.utils.looping:quality_threshold_met"
```

**Basic Loop Configuration:**
- `body`: The pipeline to execute in each iteration
- `max_loops`: Maximum number of iterations (prevents infinite loops)
- `exit_condition`: Callable that returns `True` to stop the loop

**Enhanced Loop Configuration:**
For sophisticated agentic workflows, you can now specify comprehensive input/output mappers directly in YAML:

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
- **Conversational AI loops** with structured data transformation
- **Iterative refinement workflows** with context preservation
- **Agentic planning and execution cycles** with state management
- **Quality improvement loops** with progressive enhancement
- **Multi-step clarification workflows** with conversation history

**Benefits of Enhanced Loops:**
- **Clean Data Transformation**: No more adapter steps or complex agent logic
- **Consistent State Management**: Maintains conversation context across iterations
- **Rich Output**: Comprehensive results with metadata and conversation history
- **Declarative Configuration**: Intuitive YAML syntax for complex workflows
- **Backward Compatibility**: Existing loop configurations continue to work unchanged

### 5. **Map Pattern**
For batch processing of collections.

```yaml
- kind: map
  name: batch_process_items
  map:
    iterable_input: "context.items"
    body:
      - kind: step
        name: process_single_item
        uses: agents.item_processor
        input: "{{ this }}" # Use 'this' to refer to the current item
```

## Pipeline Composition Strategies

### 1. **Modular Design with Imports**
Break complex pipelines into focused, reusable components.

```yaml
# main.yaml
version: "0.1"
imports:
  ingestion: "./data_ingestion.yaml"
  processing: "./data_processing.yaml"

steps:
  - kind: step
    name: ingest_data
    uses: imports.ingestion
  - kind: step
    name: process_data
    uses: imports.processing
```

### 2. **Agent Reuse**
Define agents once and reuse them across multiple steps with different inputs.

```yaml
agents:
  llm_processor:
    model: "openai:gpt-4o"
    system_prompt: "Process and analyze the given input."
    output_schema: { type: "string" }

steps:
  - kind: step
    name: analyze_sentiment
    uses: agents.llm_processor
    input: "Analyze sentiment: {{ context.text }}"
  - kind: step
    name: extract_keywords
    uses: agents.llm_processor
    input: "Extract keywords: {{ context.text }}"
```

### 3. **Pipeline Composition with `as_step`**
The `as_step` functionality is automatically available through `imports`, where imported pipelines are wrapped as steps.

ImportStep tips:
- Use `updates_context: true` and an explicit `config.outputs` mapping to make merges predictable and reviewable in PRs.
- Prefer `config.input_to: import_artifacts` for structured handoff between parent and child; reserve `initial_prompt` for purely conversational handoffs.
- When a child can pause for HITL, set `config.propagate_hitl: true` (default) so the question surfaces to the parent runner and pause and resume work end to end.
- Explicit `config.input_to` always determines the child’s initial input even when `inherit_conversation: true`; this prevents the previous assistant/user message from becoming the child’s prompt accidentally.
- Keep child blueprints self‑contained and place any local skills (`skills.yaml`, Python modules) next to the child YAML; Flujo auto‑loads them relative to the child file.
- Avoid step‑local env reads inside children; use centralized config (`flujo.infra.config_manager`).

```yaml
# This automatically uses pipeline.as_step(name=...) under the hood
imports:
  sub_workflow: "./sub_workflow.yaml"

steps:
  - kind: step
    name: execute_sub_workflow
    uses: imports.sub_workflow  # Automatically wrapped as a step
    updates_context: true
    config:
      propagate_hitl: true
      input_to: import_artifacts
      outputs:
        - { child: import_artifacts.artifact, parent: import_artifacts.artifact }
```

## Error Handling and Resilience

### 1. **Fallback Strategies**
Provide backup options for critical steps.

```yaml
- kind: step
  name: primary_processor
  uses: agents.primary_agent
  config:
    max_retries: 3
  fallback:
    kind: step
    name: fallback_processor
    uses: agents.backup_agent
```

### 2. **Branch Failure Handling**
Configure how parallel branches handle failures.

```yaml
- kind: parallel
  name: resilient_processing
  on_branch_failure: ignore  # Continue with successful branches
  branches:
    critical:
      - name: critical_operation
        # ...
    optional:
      - name: optional_operation
        # ...
```

### 3. **Validation and Quality Gates**
Add validation steps to catch issues early.

```yaml
- kind: step
  name: process_data
  uses: agents.data_processor
  updates_context: true

- kind: step
  name: validate_output
  uses: agents.validator
  input: "{{ context.processed_data }}"
```

## Performance Optimization

### 1. **Context Optimization**
Only copy necessary context fields to parallel branches.

```yaml
- kind: parallel
  name: optimized_parallel
  context_include_keys: ["user_id", "session_id"]
  branches:
    # ...
```

### 2. **Efficient Merge Strategies**
Choose `no_merge` if branches don't need to update the main context.

```yaml
- kind: parallel
  name: analysis_tasks
  merge_strategy: no_merge
  branches:
    # ...
```

### 3. **Batch Processing**
Use `map` steps for efficient collection processing.

```yaml
- kind: map
  name: process_batch
  map:
    iterable_input: "context.items"
    body:
      - name: process_item
        input: "{{ this }}"
        # ...
```

## Testing and Validation

### 1. **Schema Validation**
Use `output_schema` in agent definitions to enforce structure.

```yaml
agents:
  validated_agent:
    model: "openai:gpt-4o"
    output_schema:
      type: object
      properties:
        result: { type: string }
      required: [result]
```

### 2. **Step Validation**
Enable `validate_fields: true` for context updates to catch schema mismatches.

```yaml
- kind: step
  name: validated_step
  updates_context: true
  validate_fields: true
```

### 3. **Pipeline Validation Command**
Use the `flujo validate` command to check your pipeline for errors before running.

```bash
flujo validate my_pipeline.yaml
```

## Common Anti-Patterns

### 1. **Monolithic Steps**
Avoid creating single steps that perform multiple distinct operations. Break them down for clarity and reusability.

### 2. **Over-Nested Conditionals**
Deeply nested `conditional` steps are hard to read and maintain. Prefer flattening logic, possibly by using a router agent to determine the correct path.

### 3. **Hardcoded Values**
Avoid hardcoding values like URLs or file paths directly in the YAML. Pass them in via the context for better flexibility.

```yaml
# ❌ Bad: Hardcoded URL
- kind: step
  name: fetch_data
  agent:
    id: "flujo.builtins.http_get"
    params:
      url: "https://api.example.com/data"

# ✅ Good: Configurable via context
- kind: step
  name: fetch_data
  agent:
    id: "flujo.builtins.http_get"
    params:
      url: "{{ context.api_endpoint }}"
```

### 4. **Ignoring Failures**
Always consider how a step might fail. Use `fallback` steps and configure `max_retries` for critical operations.

## Lessons Learned from Real-World Development

This section documents key insights gained from building and debugging complex Flujo workflows, particularly the clarification workflow exercise that revealed important patterns for success.

### 1. **Built-in Skills First, Custom Skills Last**

**Key Learning**: Flujo's built-in skills are more powerful than initially apparent and can handle surprisingly complex workflows without custom development.

```yaml
# ✅ Good: Leverage built-in skills for common patterns
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

**Guidelines:**
- **Start with built-ins**: Explore `flujo.builtins.*` namespace before building custom solutions
- **Combine multiple built-ins**: Chain simple skills to create complex behaviors
- **Use `flujo.builtins.discover_skills`**: Run this to see all available built-in capabilities
- **Test built-in combinations**: Often you can achieve your goal without custom development

### 2. **Skill Discovery and Registration Patterns**

**Key Learning**: Understanding when and how skills are loaded is crucial for successful pipeline execution.

```yaml
# ❌ Problematic: Custom skills that aren't discoverable
# This approach requires framework-level changes to work properly
skills:
  custom_executor:
    path: "my_skills:command_executor"
    description: "Custom command executor"

# ✅ Recommended: Use built-in skills or ensure proper registration
# Built-ins are automatically available and well-tested
```

**Guidelines:**
- **Built-in skills load automatically**: No registration needed
- **Custom skills require explicit registration**: Use programmatic registration in `__init__.py` files
- **Test skill availability**: Use `flujo.builtins.discover_skills` to verify what's loaded
- **Understand loading timing**: Skills must be available before pipeline compilation

### 3. **Debugging Complex Pipeline Issues**

**Key Learning**: Systematic debugging approaches are essential for resolving pipeline issues, especially skill-related problems.

**Debugging Checklist:**
1. **Verify skill availability**: Run `flujo.builtins.discover_skills` to see what's loaded
2. **Test minimal pipelines**: Start with simple workflows before adding complexity
3. **Check import paths**: Ensure custom skills can be imported independently
4. **Validate skill registration**: Verify skills appear in the registry after import
5. **Use debug mode**: Run with `flujo --debug` for detailed error information

**Common Error Patterns:**
```bash
# ❌ "Unknown skill id" usually means:
# - Skill not registered in the registry
# - Skill loading happens after pipeline compilation
# - Import path issues in custom skills

# ✅ Debug with:
flujo --debug run pipeline.yaml
python3 -c "from flujo.infra.skill_registry import get_skill_registry; print(list(get_skill_registry()._entries.keys()))"
```

### 4. **Architectural Patterns for Interactive Workflows**

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

**Guidelines:**
- **Use HITL steps strategically**: Combine `ask_user` with AI agents for intelligent interactions
- **Stringify complex outputs**: Use `flujo.builtins.stringify` to handle type conversions
- **Template clarity**: Use `{{ previous_step | tojson }}` for the last step’s structured output, or `{{ steps.<name>.output | tojson }}` for a named step. Avoid `{{ previous_step.output }}` since `previous_step` is the raw value, not a proxy.
- **Build incrementally**: Add complexity one step at a time
- **Test each addition**: Ensure the pipeline works before adding more steps

### 5. **When to Use Custom Skills vs. Built-ins**

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

### 8. **Testing and Validation Best Practices**

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

---

## Summary of Key Insights

The clarification workflow exercise revealed several critical insights for Flujo development:

1. **Built-in skills are underestimated** - They can handle surprisingly complex workflows
2. **Skill discovery timing matters** - Skills must be available before pipeline compilation
3. **Systematic debugging is essential** - Question assumptions and test hypotheses methodically
4. **YAML-driven workflows are powerful** - When skills work, the declarative approach scales well
5. **Start simple, add complexity incrementally** - Test each addition before building further

**Remember**: Flujo's built-in capabilities are often sufficient for your needs. Explore them thoroughly before building custom solutions, and always test incrementally to catch issues early.
 
