# Quickstart Guide

Get up and running with **Flujo** in 5 minutes! This guide will walk you through installing Flujo, setting up your environment, and running your first AI workflows.

## Prerequisites

- Python 3.11 or higher
- An OpenAI API key (or other supported provider)
- Basic familiarity with Python

## 1. Installation

Install Flujo using pip:

```bash
pip install flujo
```

## 2. Set Up Your API Keys

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env
```

Add your API keys to the `.env` file:

```env
# Required: Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other providers
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

> **💡 Tip**: You can get an OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)

## 3. Your First Workflow: Default Recipe

The **Default Recipe** is the easiest way to get started. It runs a fixed workflow: Review → Solution → Validate → Reflection.

Create a file called `first_workflow.py`:

```python
from flujo.recipes import Default
from flujo import Task, init_telemetry

# Initialize telemetry for monitoring (optional but recommended)
init_telemetry()

# Create the default recipe with built-in agents
orch = Default()

# Define a task
task = Task(prompt="Write a Python function to calculate Fibonacci numbers")

# Run the workflow
result = orch.run_sync(task)

# Check the result
if result:
    print("✅ Success!")
    print(f"Solution:\n{result.solution}")
    print(f"Quality Score: {result.score:.2f}")
    
    # Show the quality checklist
    if result.checklist:
        print("\nQuality Checklist:")
        for item in result.checklist.items:
            status = "✅" if item.passed else "❌"
            print(f"  {status} {item.description}")
else:
    print("❌ Workflow failed")
```

Run your first workflow:

```bash
python first_workflow.py
```

### Expected Output

```
✅ Success!
Solution:
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

Quality Score: 0.87

Quality Checklist:
  ✅ Function is correctly implemented
  ✅ Includes proper docstring
  ✅ Handles edge cases (n <= 1)
  ✅ Uses recursive approach as requested
```

## 4. Your First AgenticLoop

**AgenticLoop** is a more flexible pattern where a planner agent decides what to do next. This is great for complex, multi-step tasks.

Create a file called `first_agentic.py`:

```python
from flujo.recipes import AgenticLoop
from flujo import make_agent_async, init_telemetry
from flujo.domain.commands import AgentCommand
from pydantic import TypeAdapter

init_telemetry()

# Define a simple tool agent
async def search_agent(query: str) -> str:
    """A simple search tool that returns information."""
    print(f"🔍 Searching for: {query}")
    
    # Simple mock search results
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language known for its simplicity and readability."
    elif "flujo" in query.lower():
        return "Flujo is a Python library for orchestrating AI workflows with type safety and production-ready features."
    else:
        return "No specific information found for that query."

# Create a planner agent
PLANNER_PROMPT = """
You are a helpful research assistant. You have access to a search tool to gather information.

Your process:
1. Use the search_agent to find relevant information
2. When you have enough information to answer the question, use FinishCommand
3. Be thorough but concise

Available commands:
- RunAgentCommand: Use search_agent to find information
- FinishCommand: Provide the final answer
"""

planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    TypeAdapter(AgentCommand),
)

# Create the AgenticLoop
loop = AgenticLoop(
    planner_agent=planner,
    agent_registry={"search_agent": search_agent}
)

# Run the loop
result = loop.run("What is Python and how does it relate to Flujo?")

# Show the results
print("\n🎯 Final Answer:")
print(result.final_pipeline_context.command_log[-1].execution_result)

print("\n📋 Command Log:")
for i, command in enumerate(result.final_pipeline_context.command_log, 1):
    print(f"{i}. {command.command_type}: {command.agent_name or 'N/A'}")
```

Run your first AgenticLoop:

```bash
python first_agentic.py
```

### Expected Output

```
🔍 Searching for: python
🔍 Searching for: flujo

🎯 Final Answer:
Python is a high-level, general-purpose programming language known for its simplicity and readability. Flujo is a Python library built on top of Python that provides tools for orchestrating AI workflows with type safety and production-ready features.

📋 Command Log:
1. RunAgentCommand: search_agent
2. RunAgentCommand: search_agent
3. FinishCommand: N/A
```

## 5. Command Line Interface

Flujo also provides a powerful CLI for quick tasks:

```bash
# Solve a task quickly
flujo solve "Write a function to sort a list"

# Run benchmarks
flujo bench "Generate a poem" --rounds 3

# Show your configuration
flujo show-config
```

## 6. Understanding the Results

### Default Recipe Results

The Default recipe returns a `Candidate` object with:

- **`solution`**: The generated solution (string)
- **`score`**: Quality score (0.0 to 1.0)
- **`checklist`**: Detailed quality assessment
- **`metadata`**: Cost, duration, and other metrics

### AgenticLoop Results

AgenticLoop returns a result with:

- **`final_pipeline_context`**: Complete execution context
- **`command_log`**: Step-by-step execution history
- **`execution_result`**: Final answer from the planner

## 7. Next Steps

Now that you've seen the basics, explore these resources:

### For Beginners
- **[Core Concepts](concepts.md)** - Understand the fundamental building blocks
- **[Configuration Guide](configuration.md)** - Learn about advanced configuration options
- **[Basic Usage](usage.md)** - Master the essential patterns

### For Developers
- **[Tutorial](tutorial.md)** - Comprehensive guided tour from simple to advanced
- **[Pipeline DSL Guide](pipeline_dsl.md)** - Build custom workflows
- **[API Reference](api_reference.md)** - Detailed API documentation

### For Advanced Users
- **[Architecture Overview](architecture.md)** - Understand the system design
- **[Extending Guide](extending.md)** - Create custom components
- **[Cookbook](cookbook/)** - Advanced patterns and examples

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Make sure your `.env` file exists and contains the correct API key
- Check that the key is valid and has sufficient credits

**"Module not found"**
- Ensure you've installed Flujo: `pip install flujo`
- Check your Python environment is activated

**"API rate limit exceeded"**
- Wait a moment and try again
- Consider using a different model or reducing request frequency

### Getting Help

- Check the **[Troubleshooting Guide](troubleshooting.md)** for detailed solutions
- Visit **[GitHub Issues](https://github.com/aandresalvarez/flujo/issues)** to report bugs
- Join **[GitHub Discussions](https://github.com/aandresalvarez/flujo/discussions)** for community support

## What You've Learned

✅ **Installed Flujo** and set up your environment  
✅ **Ran your first Default Recipe** workflow  
✅ **Created an AgenticLoop** with custom tools  
✅ **Used the CLI** for quick tasks  
✅ **Understood the results** and next steps  

You're now ready to build powerful AI workflows with Flujo! 🚀
