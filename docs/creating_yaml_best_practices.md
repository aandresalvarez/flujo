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

```yaml
# This automatically uses pipeline.as_step(name=...) under the hood
imports:
  sub_workflow: "./sub_workflow.yaml"

steps:
  - kind: step
    name: execute_sub_workflow
    uses: imports.sub_workflow  # Automatically wrapped as a step
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
 