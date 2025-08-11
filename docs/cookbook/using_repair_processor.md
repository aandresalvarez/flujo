# Cookbook: Using the Repair Processor

The `DeterministicRepairProcessor` is a powerful tool for automatically fixing malformed JSON data. It can handle a variety of common errors, such as missing quotes, trailing commas, and incorrect literal values. This processor is particularly useful when working with language models that may not always produce perfectly valid JSON.

## How it Works

The `DeterministicRepairProcessor` applies a series of regular expressions and other techniques to clean up and repair malformed JSON strings. It can fix the following issues:

*   Code fences (e.g., ```json ... ```)
*   Line and block comments
*   Trailing commas
*   Single-quoted strings
*   Python literals (e.g., `None`, `True`, `False`)
*   Unquoted keys

## Usage

To use the `DeterministicRepairProcessor`, you can add it to the `output_processors` of a step in your pipeline. For example:

```python
from flujo import Step, AgentProcessors
from flujo.agents.repair import DeterministicRepairProcessor
from flujo.testing.utils import StubAgent

# This agent returns a malformed JSON string
agent = StubAgent(['{'name': "John", 'age': 30,}'])

# The repair processor will fix the JSON before it is passed to the next step
processors = AgentProcessors(output_processors=[DeterministicRepairProcessor()])
step = Step.solution(agent, processors=processors)

# Run the pipeline
runner = Flujo(step)
result = runner.run("Give me some JSON")

# The output will be a valid JSON string
print(result.step_history[0].output)
```

In this example, the `DeterministicRepairProcessor` will take the malformed JSON string `{'name': "John", 'age': 30,}` and convert it to the valid JSON string `{"name": "John", "age": 30}`.
