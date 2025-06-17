# Cookbook: Real-Time Chatbot

Stream responses from a pipeline to build a snappy chatbot interface.

```python
import asyncio
from flujo import Step, Flujo

class EchoStreamer:
    async def stream(self, data: str):
        for c in data.upper():
            yield c
            await asyncio.sleep(0)

pipeline = Step.solution(EchoStreamer())
runner = Flujo(pipeline)

async def chat():
    async for part in runner.stream_async("hello"):
        if isinstance(part, str):
            print(part, end="")

asyncio.run(chat())
```
