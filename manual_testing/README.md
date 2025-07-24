# Manual Testing Pipeline

This directory contains a step-by-step implementation of Flujo features for learning purposes.

## Current Implementation: Step 1 - Core Agentic Step

This is the simplest possible pipeline that demonstrates:
- Creating a basic AI agent with `make_agent_async()` (FSD-11 fixed)
- Defining a `Step` with the agent
- Creating a `Pipeline` from the step
- Running the pipeline with `Flujo` using correct async patterns
- **FSD-12 Tracing**: Automatic observability with `flujo lens`

## How to Run

### Option 1: Using Environment Variables
1. Make sure you have your API key set in your environment:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Run the pipeline:
   ```bash
   # From the project root directory
   python -m manual_testing.main
   ```

### Option 2: Using Configuration File (Recommended)
1. The `manual_testing/` directory includes a `flujo.toml` configuration file that will be automatically recognized by Flujo.

2. The `.env` file in the `manual_testing/` directory already contains your API key (secure and gitignored).

3. Run the pipeline from the `manual_testing/` directory to use the local configuration:
   ```bash
   # From the manual_testing directory
   cd manual_testing
   python -m main
   ```

**Security Note**: The API key is loaded from the `.env` file (which is gitignored) and never hardcoded in the source code.

**Note**: The `flujo.toml` file provides local configuration for this testing environment, including:
- Local state storage (`flujo_ops.db` in the manual_testing directory)
- Model configurations
- Feature toggles
- CLI defaults

**Important**: The configuration file is discovered based on the current working directory. Running from the project root will use the root's configuration (if any), while running from `manual_testing/` will use the local configuration.

3. Enter a clinical cohort definition when prompted, for example:
   - "patients with diabetes" (unclear - will ask for clarification)
   - "adult patients with Type 2 diabetes diagnosed in the last 5 years" (clear - will confirm)

## Expected Behavior

- **Unclear definitions**: The agent will ask clarifying questions
- **Clear definitions**: The agent will restate the definition and add `[CLARITY_CONFIRMED]`
- **Tracing**: Each run is automatically traced and can be inspected with `flujo lens`

## Current Status

✅ **Step 1 Implementation Complete!**

The pipeline is now working correctly with the latest Flujo features:
- ✅ **FSD-11 Fix**: `make_agent_async()` works perfectly without context injection errors
- ✅ **FSD-12 Tracing**: Automatic observability with local `flujo_ops.db`
- ✅ **Correct Async Patterns**: Using `run_async()` for proper execution
- ✅ **Clean Architecture**: No mock agents or workarounds needed

## Technical Details

### FSD-11 Improvements:
- **Signature-Aware Context Injection**: The framework now intelligently detects whether an agent accepts a `context` parameter
- **No More Wrappers**: `make_agent_async()` works out-of-the-box for stateless agents
- **Backward Compatibility**: Context-aware agents continue to work as before

### FSD-12 Features:
- **Automatic Tracing**: Every pipeline run is traced by default
- **Local Storage**: Traces are saved to `manual_testing_ops.db` in the manual_testing directory
- **CLI Inspection**: Use `flujo lens trace <run_id>` to inspect execution details
- **Rich Metadata**: See step inputs, outputs, timing, and context evolution

### Architecture:
- **Agent**: Clinical research assistant that reviews cohort definitions
- **Step**: Single step that processes user input through the agent
- **Pipeline**: Simple pipeline with one step
- **Runner**: `Flujo` with automatic tracing and context management

## Observability Demo

After running the pipeline, you'll see output like:
```
✨ OBSERVABILITY (FSD-12) ✨
Pipeline run completed with ID: run_e962cec94e93410c9a211caf2758da8d
A trace has been saved to the local `manual_testing_ops.db` file.

To inspect the trace, run this command in your terminal:

  flujo lens trace run_e962cec94e93410c9a211caf2758da8d
```

Run the suggested command to see detailed execution traces, step-by-step execution, timing information, and context evolution.

## Test Results

### Test Case 1: Unclear Definition
**Input**: "patients with diabetes"
**Output**: "Is the cohort limited to a specific type of diabetes (e.g., Type 1, Type 2, gestational) or any diagnosis of diabetes?"
**Status**: ✅ Working correctly - agent asks for clarification

### Test Case 2: Clear Definition
**Input**: "adult patients with Type 2 diabetes diagnosed in the last 5 years"
**Output**: "Adult patients who have been diagnosed with Type 2 diabetes within the last 5 years. [CLARITY_CONFIRMED]"
**Status**: ✅ Working correctly - agent confirms clarity

## Next Steps

This is Step 1 of 5 planned steps:
1. ✅ **Core Agentic Step** (current - with FSD-11 & FSD-12 features)
2. 🔄 Clarification Loop (iteration)
3. 🔄 State Management (PipelineContext)
4. 🔄 Human Interaction (HITL)
5. 🔄 Professional Refinement (structured outputs)
