# Manual Testing Pipeline

This directory contains a step-by-step implementation of Flujo features for learning purposes.

## Current Implementation: Step 1 - Core Agentic Step

This is the simplest possible pipeline that demonstrates:
- Creating a basic AI agent with `make_agent()`
- Defining a `Step` with the agent
- Creating a `Pipeline` from the step
- Running the pipeline with `Flujo`

## How to Run

1. Make sure you have your API key set in your environment:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Run the pipeline:
   ```bash
   # From the project root directory
   python -m manual_testing.main
   ```

3. Enter a clinical cohort definition when prompted, for example:
   - "patients with diabetes" (unclear - will ask for clarification)
   - "adult patients with Type 2 diabetes diagnosed in the last 5 years" (clear - will confirm)

## Expected Behavior

- **Unclear definitions**: The agent will ask clarifying questions
- **Clear definitions**: The agent will restate the definition and add `[CLARITY_CONFIRMED]`

## Current Status

✅ **Step 1 Implementation Complete!**

The pipeline is now working correctly:
- ✅ Context parameter issue resolved with `SimpleAgentWrapper`
- ✅ Pipeline executes without errors
- ✅ Error handling and logging implemented
- ⚠️ API key authentication issue (needs valid OpenAI API key)

## Technical Details

### Issues Resolved:
1. **Context Parameter Issue**: The Flujo runner was trying to pass a `context` parameter to the pydantic-ai Agent, which doesn't accept it. Fixed by creating a `SimpleAgentWrapper` that filters out unwanted parameters.

2. **Async Execution Issue**: Initially tried to use `make_agent_async()` but switched to `make_agent()` for simpler implementation.

3. **Error Handling**: Added comprehensive error reporting to show step success/failure status.

### Architecture:
- **Agent**: Clinical research assistant that reviews cohort definitions
- **Step**: Single step that processes user input through the agent
- **Pipeline**: Simple pipeline with one step
- **Wrapper**: `SimpleAgentWrapper` handles parameter filtering

## Next Steps

This is Step 1 of 5 planned steps:
1. ✅ **Core Agentic Step** (current - API key issue only)
2. 🔄 Clarification Loop (iteration)
3. 🔄 State Management (PipelineContext)
4. 🔄 Human Interaction (HITL)
5. 🔄 Professional Refinement (structured outputs)
