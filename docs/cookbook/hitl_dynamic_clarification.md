# Cookbook: Dynamic Clarification

Let an earlier step produce the question for the human by omitting `message_for_user`.

```python
pipeline = Step("ask", StubAgent(["Is this ok?"])) >> Step.human_in_the_loop("clarify")
runner = Flujo(pipeline)
result = await runner.run_async("hello")
# display pause message from context and resume
result = await runner.resume_async(result, "sure")
```
