"""Sequential pipeline execution utilities."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List

from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    """Configuration for an individual step."""

    max_retries: int = 0


class Step(BaseModel):
    """Represents a single pipeline step."""

    name: str
    func: Callable[[Any], Awaitable[Any]]
    config: StepConfig = Field(default_factory=StepConfig)

    class Config:
        arbitrary_types_allowed = True


class StepResult(BaseModel):
    """Outcome of a step execution."""

    name: str
    attempts: int
    output: Any | None = None
    success: bool = True


class PipelineResult(BaseModel):
    """Aggregated result of running a pipeline."""

    steps: List[StepResult] = Field(default_factory=list)
    output: Any | None = None


class PipelineRunner:
    """Executes steps sequentially with retry support."""

    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps

    async def run(self, data: Any) -> PipelineResult:
        """Run the pipeline returning a ``PipelineResult``."""
        history: List[StepResult] = []
        current = data
        for step in self.steps:
            attempts = 0
            while True:
                try:
                    attempts += 1
                    current = await step.func(current)
                    history.append(
                        StepResult(
                            name=step.name,
                            attempts=attempts,
                            output=current,
                            success=True,
                        )
                    )
                    break
                except Exception as exc:
                    if attempts > step.config.max_retries:
                        history.append(
                            StepResult(
                                name=step.name,
                                attempts=attempts,
                                output=None,
                                success=False,
                            )
                        )
                        raise exc
                    await asyncio.sleep(0)
                    continue
        return PipelineResult(steps=history, output=current)
