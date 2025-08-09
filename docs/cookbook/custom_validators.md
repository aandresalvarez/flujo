# Cookbook: Custom Validators

`flujo` allows you to create custom validators to check the output of a step against your own criteria. You can create a custom validator by either subclassing the `BaseValidator` class or by using the `validator` decorator.

## Using the `BaseValidator` Class

You can create a custom validator by subclassing the `BaseValidator` class and implementing the `validate` method.

```python
from flujo.domain.validation import BaseValidator
from flujo.domain.validation import ValidationResult

class MyValidator(BaseValidator):
    async def validate(self, output_to_check: any, *, context: any = None) -> ValidationResult:
        if "hello" in output_to_check.lower():
            return ValidationResult(is_valid=True, feedback="Contains 'hello'")
        else:
            return ValidationResult(is_valid=False, feedback="Does not contain 'hello'")
```

## Using the `validator` Decorator

You can also create a custom validator by decorating a function with the `validator` decorator. The function should take the output to check as an argument and return a tuple containing a boolean indicating whether the output is valid and an optional feedback string.

```python
from flujo.domain.validation import validator

@validator
def contains_world(output_to_check: any) -> tuple[bool, str | None]:
    if "world" in output_to_check.lower():
        return True, "Contains 'world'"
    else:
        return False, "Does not contain 'world'"
```

## Using Custom Validators in a Pipeline

Once you have created a custom validator, you can use it in a `validate` step in your pipeline.

```python
from flujo import Step, Flujo
from flujo.testing.utils import StubAgent

# Create a pipeline with a custom validator
pipeline = (
    Step.solution(StubAgent(["Hello, world!"]))
    >> Step.validate(validators=[MyValidator(), contains_world])
)

# Run the pipeline
runner = Flujo(pipeline)
result = runner.run("some input")

# The output will be valid
assert result.step_history[-1].success
```
