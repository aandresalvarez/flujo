# Cookbook: Serialize Pydantic Models

`SerializePydantic` converts models produced by one step into plain dictionaries before passing them along. This is handy when the next agent or tool doesn't understand Pydantic models.

```python
from pydantic import BaseModel
from flujo import Step, Flujo, AgentProcessors
from flujo.processors import SerializePydantic
from flujo.testing.utils import StubAgent

class User(BaseModel):
    name: str
    age: int

# First step emits a User model
producer = Step.solution(
    StubAgent([User(name="Ada", age=42)]),
    processors=AgentProcessors(output_processors=[SerializePydantic()]),
)

# Second step expects a dict
consumer_agent = StubAgent(["User data processed: Ada (42 years old)"])
consumer = Step.solution(consumer_agent)

pipeline = producer >> consumer
runner = Flujo(pipeline)
runner.run(None)
print(consumer_agent.inputs[0])
```

This prints `{"name": "Ada", "age": 42}` showing how the processor bridges a Pydantic model to plain data.
