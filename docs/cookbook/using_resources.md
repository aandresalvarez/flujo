# Using Managed Resources

This cookbook demonstrates how to share long-lived objects across your pipeline.

```python
from unittest.mock import MagicMock
from flujo import Flujo, Step, AppResources

class MyResources(AppResources):
    db: MagicMock

resources = MyResources(db=MagicMock())

class LookupAgent:
    async def run(self, user_id: int, *, resources: MyResources) -> str:
        return resources.db.get(user_id)

pipeline = Step("lookup", LookupAgent())
runner = Flujo(pipeline, resources=resources)
result = runner.run(123)
```

Any agent or plugin can declare a `resources` parameter to access the shared
container.
