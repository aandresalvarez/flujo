from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ...domain.models import UsageLimits, StepResult
from ...domain.validation import ValidationResult
from flujo.type_definitions.common import JSONObject


# --- Core execution protocols ---
class IAgentRunner(Protocol):
    async def run(
        self,
        agent: object,
        payload: object,
        *,
        context: object,
        resources: object,
        options: JSONObject,
        stream: bool = False,
        on_chunk: Callable[[object], Awaitable[None]] | None = None,
    ) -> object: ...


class IProcessorPipeline(Protocol):
    async def apply_prompt(
        self, processors: object, data: object, *, context: object
    ) -> object: ...
    async def apply_output(
        self, processors: object, data: object, *, context: object
    ) -> object: ...


class IValidatorRunner(Protocol):
    async def validate(
        self, validators: list[object], data: object, *, context: object
    ) -> list[ValidationResult]: ...


class IPluginRunner(Protocol):
    async def run_plugins(
        self,
        plugins: list[tuple[object, int]],
        data: object,
        *,
        context: object,
        resources: object | None = None,
    ) -> object: ...


class IUsageMeter(Protocol):
    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None: ...
    async def guard(
        self, limits: UsageLimits, step_history: list[object] | None = None
    ) -> None: ...
    async def snapshot(self) -> tuple[float, int, int]: ...


class ITelemetry(Protocol):
    def trace(self, name: str) -> Callable[[Callable[..., object]], Callable[..., object]]: ...


class ISerializer(Protocol):
    def serialize(self, obj: object) -> bytes: ...
    def deserialize(self, blob: bytes) -> object: ...


class IHasher(Protocol):
    def digest(self, data: bytes) -> str: ...


class ICacheBackend(Protocol):
    async def get(self, key: str) -> StepResult | None: ...
    async def put(self, key: str, value: StepResult, ttl_s: int) -> None: ...
    async def clear(self) -> None: ...


__all__ = [
    "IAgentRunner",
    "IProcessorPipeline",
    "IValidatorRunner",
    "IPluginRunner",
    "IUsageMeter",
    "ITelemetry",
    "ISerializer",
    "IHasher",
    "ICacheBackend",
]
