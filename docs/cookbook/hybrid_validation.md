# Hybrid Validation

Sometimes you want fast, deterministic checks in addition to an LLM based review. The `validators` parameter on `Step.validate_step` lets you run custom code validators alongside your agent.

```python
from flujo import Step
from flujo.domain.validation import BaseValidator, ValidationResult

class ShortAnswerValidator(BaseValidator):
    async def validate(self, output_to_check: str, *, context=None) -> ValidationResult:
        words = len(output_to_check.split())
        return ValidationResult(
            is_valid=words <= 5,
            feedback=f"Answer too long: {words} words" if words > 5 else None,
            validator_name=self.name,
        )

validation_step = Step.validate_step(
    make_validator_agent(),
    validators=[ShortAnswerValidator()]
)
```

You can combine many validators to create a robust quality gate before the next step runs.
