"""Example: Running a custom pipeline with the CLI (`flujo run`).

To try this example:
    flujo run examples/10_cli_run_demo.py --input "quickstart" --context-model DemoContext
    flujo run examples/10_cli_run_demo.py --input "with context" --context-model DemoContext --context-data '{"counter": 10}'
"""

from flujo import step
from flujo.domain.models import PipelineContext
from pydantic import Field


class DemoContext(PipelineContext):
    counter: int = Field(default=0)
    log: list[str] = Field(default_factory=list)


@step
async def greet(data: str, *, context: DemoContext | None = None) -> str:
    msg = f"Hello, {data}!"
    if context:
        context.counter += 1
        context.log.append(msg)
    return msg


@step
async def emphasize(data: str, *, context: DemoContext | None = None) -> str:
    msg = data.upper() + "!!!"
    if context:
        context.counter += 1
        context.log.append(msg)
    return msg


@step
async def summarize(data: str, *, context: DemoContext | None = None) -> str:
    summary = f"Summary: {data} (steps: {context.counter if context else 0})"
    if context:
        context.counter += 1
        context.log.append(summary)
    return summary


pipeline = greet >> emphasize >> summarize
