from typing import Any
from flujo.domain.models import PipelineContext


async def emit(_data: Any, *, context: PipelineContext | None = None) -> dict:
    return {"scratchpad": {"value": 1}}
