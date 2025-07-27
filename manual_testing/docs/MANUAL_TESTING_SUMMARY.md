# Manual Testing Summary

## Overview

This directory contains comprehensive manual testing scripts for **Step 1: The Core Agentic Step** of the Flujo framework. These scripts demonstrate real API interactions with actual cohort definitions.

## Available Test Scripts

### 1. **Basic Manual Test** (`manual_test_step1.py`)
- Tests 2 predefined cohort definitions
- Shows basic pipeline execution
- Good for quick validation

### 2. **Challenging Manual Test** (`manual_test_step1_challenging.py`)
- Tests 4 challenging cohort definitions
- Demonstrates both clarification requests and confirmations
- Shows real agent behavior with vague vs. complete definitions

### 3. **Interactive Manual Test** (`interactive_test_step1.py`)
- Allows you to input your own cohort definitions
- Real-time testing with custom inputs
- Perfect for exploring different scenarios

### 4. **Comprehensive Automated Test** (`run_step1_test.py`)
- 11 automated tests covering all aspects
- Mock agents for deterministic testing
- Full validation of FSD-11 and FSD-12 features

## Real API Test Results

### ✅ **Incomplete Definitions → Clarification Requests**

**Example 1: "sick people"**
```
🤖 Agent Response: Which specific illness or condition defines the "sick people" cohort?
❓ RESULT: Definition needs CLARIFICATION
```

**Example 2: "cancer patients"**
```
🤖 Agent Response: The clinical cohort is defined as "cancer patients." Could you specify the type of cancer or any additional criteria such as stage, treatment status, or demographic information? This will help clarify the cohort definition.
❓ RESULT: Definition needs CLARIFICATION
```

**Example 3: "patients with heart problems"**
```
🤖 Agent Response: Could you clarify what specific types of heart problems are included in this cohort definition? For example, is it meant to include conditions such as heart failure, myocardial infarction, arrhythmia, or all heart-related conditions?
❓ RESULT: Definition needs CLARIFICATION
```

### ✅ **Complete Definitions → Confirmation**

**Example: Detailed diabetes cohort**
```
🤖 Agent Response: Adult patients aged 18-65 with confirmed Type 2 diabetes diagnosed between 2020-2024, currently prescribed metformin at a dose of 500-2000mg daily, with HbA1c levels between 7.0-10.0%. [CLARITY_CONFIRMED]
🎉 RESULT: Definition is CLEAR
```

## Key Observations

### **Agent Behavior**
- **Vague definitions** → Agent asks for specific details
- **Incomplete definitions** → Agent requests missing criteria
- **Ambiguous definitions** → Agent asks for clarification
- **Complete definitions** → Agent confirms with `[CLARITY_CONFIRMED]`

### **Pipeline Features Demonstrated**
- ✅ **Real API Integration**: Actual OpenAI GPT-4o calls
- ✅ **Cost Tracking**: Automatic cost calculation and logging
- ✅ **Tracing**: Run IDs and execution history captured
- ✅ **Error Handling**: Graceful failure handling
- ✅ **Context Management**: Proper context injection (FSD-11)
- ✅ **Observability**: Full execution traces (FSD-12)

### **Cost Information**
Each test run shows detailed cost breakdown:
```
Cost calculation: prompt_cost=0.00034, completion_cost=0.00024, total=0.00058
Calculated cost for step 'AssessClarity': 0.00058 USD for model gpt-4o
```

## How to Run

### **Quick Start**
```bash
cd manual_testing

# Basic test with predefined examples
python3 manual_test_step1.py

# Challenging test with incomplete definitions
python3 manual_test_step1_challenging.py

# Interactive test - input your own definitions
python3 interactive_test_step1.py
```

### **Example Interactive Session**
```bash
$ python3 interactive_test_step1.py

============================================================
Enter a cohort definition (or 'quit' to exit):
============================================================
📝 Cohort Definition: elderly patients

📊 RESULTS:
============================================================
✅ Pipeline executed successfully!
🤖 Agent Response:
   Could you specify the age range that defines "elderly" patients and any specific health conditions or criteria for inclusion in this cohort?

❓ RESULT: Definition needs CLARIFICATION
   The agent is asking for more specific details.
============================================================
```

## Tracing and Debugging

### **View Execution Traces**
After each test run, you can inspect the detailed execution:

```bash
# Use the run ID from the test output
flujo lens trace run_44f484162a6e495c94dc257910f8ec59
```

### **Trace Information Includes**
- Step-by-step execution details
- Input/output for each step
- Timing information
- Cost calculations
- Context evolution
- Error details (if any)

## Benefits of Manual Testing

### **For Learning**
- **Real API Interaction**: See actual agent responses
- **Behavior Understanding**: Understand how agents handle different inputs
- **Cost Awareness**: See real costs of API calls
- **Tracing Practice**: Learn to use observability features

### **For Development**
- **Feature Validation**: Test new features with real data
- **Edge Case Discovery**: Find unexpected behaviors
- **Performance Monitoring**: Track response times and costs
- **Debugging Practice**: Use tracing for troubleshooting

### **For Production**
- **Quality Assurance**: Validate pipeline behavior
- **Cost Management**: Monitor API usage and costs
- **User Experience**: Understand how users will interact
- **Documentation**: Create examples for users

## Next Steps

This manual testing foundation can be extended for the remaining steps:

- **Step 2**: Clarification Loop (iteration)
- **Step 3**: State Management (PipelineContext)
- **Step 4**: Human Interaction (HITL)
- **Step 5**: Professional Refinement (structured outputs)

Each step can build upon these testing patterns to validate new concepts while ensuring existing functionality remains intact. 