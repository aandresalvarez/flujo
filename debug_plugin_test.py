#!/usr/bin/env python3

import asyncio
from flujo.domain import Step
from flujo.domain.plugins import PluginOutcome
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.validation import BaseValidator
from flujo.domain.validation import ValidationResult
from tests.conftest import create_test_flujo
from pydantic import BaseModel


class FailValidator(BaseValidator):
    async def validate(
        self, output_to_check, *, context: BaseModel | None = None
    ) -> ValidationResult:
        return ValidationResult(is_valid=False, feedback="bad output", validator_name=self.name)


async def main():
    agent = StubAgent(["bad"])
    plugin = DummyPlugin(outcomes=[PluginOutcome(success=False, feedback="plugin fail")])
    step = Step.validate_step(agent, plugins=[(plugin, 0)], validators=[FailValidator()])
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")

    print(f"Step success: {result.step_history[0].success}")
    print(f"Step feedback: {result.step_history[0].feedback}")
    print(f"Expected 'plugin fail' in feedback: {'plugin fail' in (result.step_history[0].feedback or '')}")
    print(f"Expected 'FailValidator' in feedback: {'FailValidator' in (result.step_history[0].feedback or '')}")


if __name__ == "__main__":
    asyncio.run(main())
