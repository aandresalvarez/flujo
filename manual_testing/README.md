# Manual Testing Pipeline

This directory contains a step-by-step implementation of Flujo features for learning purposes.

## Current Implementation: Step 1 - Core Agentic Step

This is the simplest possible pipeline that demonstrates:
- Creating a basic AI agent with `make_agent_async()` (FSD-11 fixed)
- Defining a `Step` with the agent
- Creating a `Pipeline` from the step
- Running the pipeline with `Flujo` using correct async patterns
- **FSD-12 Tracing**: Automatic observability with `flujo lens`

## 📁 Organized Structure

The manual testing directory is now organized into clear folders:

```
manual_testing/
├── 📋 tests/
│   ├── 🤖 automated/     # Automated test suites
│   └── 🧪 manual/        # Manual tests with real API
├── 📚 docs/              # Documentation and summaries
├── 🔧 examples/          # Example implementations
├── 📄 README.md          # This file
├── ⚙️  flujo.toml        # Configuration
└── 🚀 run_tests.py       # Main test runner
```

## 🚀 Quick Start

### Option 1: Interactive Test Runner (Recommended)
```bash
cd manual_testing
python3 run_tests.py
```

This provides a menu-driven interface to run all tests and view documentation.

### Option 2: Direct Test Execution
```bash
cd manual_testing

# Automated tests
python3 -m tests.automated.run_step1_test
python3 -m tests.automated.test_bug_demonstration
python3 -m tests.automated.test_config

# Manual tests (Real API)
python3 -m tests.manual.manual_test_step1
python3 -m tests.manual.manual_test_step1_challenging
python3 -m tests.manual.interactive_test_step1

# Examples
python3 -m examples.main
```

### Option 3: Individual Test Files
```bash
cd manual_testing

# Automated tests
python3 tests/automated/run_step1_test.py
python3 tests/automated/test_bug_demonstration.py
python3 tests/automated/test_config.py

# Manual tests
python3 tests/manual/manual_test_step1.py
python3 tests/manual/manual_test_step1_challenging.py
python3 tests/manual/interactive_test_step1.py

# Examples
python3 examples/main.py
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
- ✅ **Comprehensive Testing**: Full test suite validates all core concepts

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

### Manual Testing Results

**Challenging Test Results:**

**Test Case 1: Very Vague Definition**
- **Input**: "sick people"
- **Output**: "Which specific illness or condition defines the 'sick people' cohort?"
- **Status**: ✅ Working correctly - agent asks for clarification

**Test Case 2: Incomplete Definition**
- **Input**: "cancer patients"
- **Output**: "The clinical cohort is defined as 'cancer patients.' Could you specify the type of cancer or any additional criteria such as stage, treatment status, or demographic information? This will help clarify the cohort definition."
- **Status**: ✅ Working correctly - agent asks for clarification

**Test Case 3: Ambiguous Definition**
- **Input**: "patients with heart problems"
- **Output**: "Could you clarify what specific types of heart problems are included in this cohort definition? For example, is it meant to include conditions such as heart failure, myocardial infarction, arrhythmia, or all heart-related conditions?"
- **Status**: ✅ Working correctly - agent asks for clarification

**Test Case 4: Complete Definition**
- **Input**: "adult patients aged 18-65 with confirmed Type 2 diabetes diagnosed between 2020-2024, currently prescribed metformin at a dose of 500-2000mg daily, with HbA1c levels between 7.0-10.0%"
- **Output**: "Adult patients aged 18-65 with confirmed Type 2 diabetes diagnosed between 2020-2024, currently prescribed metformin at a dose of 500-2000mg daily, with HbA1c levels between 7.0-10.0%. [CLARITY_CONFIRMED]"
- **Status**: ✅ Working correctly - agent confirms clarity

### Automated Testing
The comprehensive test suite (`test_step1_core_agentic.py`) validates:

**Core Functionality Tests:**
- ✅ Agent creation with `make_agent_async()`
- ✅ Step creation and configuration
- ✅ Pipeline structure and composition
- ✅ Pipeline execution with mock agents
- ✅ Error handling and edge cases

**FSD-11 Tests:**
- ✅ Signature-aware context injection
- ✅ Stateless agents work with context present
- ✅ Context-aware agents work correctly

**FSD-12 Tests:**
- ✅ Automatic tracing and observability
- ✅ Run ID generation and tracking
- ✅ Step history and metadata capture

**Integration Tests:**
- ✅ Real agent integration (when API key available)
- ✅ API key validation
- ✅ Configuration loading

## Next Steps

This is Step 1 of 5 planned steps:
1. ✅ **Core Agentic Step** (current - with FSD-11 & FSD-12 features)
2. 🔄 Clarification Loop (iteration)
3. 🔄 State Management (PipelineContext)
4. 🔄 Human Interaction (HITL)
5. 🔄 Professional Refinement (structured outputs)
