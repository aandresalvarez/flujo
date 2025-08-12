"""Demonstrates the make_state_machine_pipeline factory."""

import asyncio

from flujo import Flujo, step
from flujo.recipes import make_state_machine_pipeline
from flujo.domain.models import PipelineContext


class Ctx(PipelineContext):
    next_state: str = "start"
    is_complete: bool = False
    counter: int = 0


@step
async def start(data: str, *, context: Ctx) -> str:
    context.counter += 1
    context.next_state = "end" if context.counter > 1 else "start"
    return data


@step
async def end(data: str, *, context: Ctx) -> str:
    context.is_complete = True
    return f"completed after {context.counter} loops"


pipeline = make_state_machine_pipeline(
    nodes={"start": start, "end": end},
    context_model=Ctx,
)

runner = Flujo(pipeline, context_model=Ctx)


async def main() -> None:
    result = await runner.arun("go")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
