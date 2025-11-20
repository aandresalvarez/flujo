from __future__ import annotations
from typing import Protocol, Any, Awaitable, Optional, List, Dict, Callable
from ...domain.models import UsageLimits, StepResult
from ...domain.validation import ValidationResult


# --- Core execution protocols ---
class IAgentRunner(Protocol):
    async def run(
        self,
        agent: Any,
        payload: Any,
        *,
        context: Any,
        resources: Any,
        options: Dict[str, Any],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Any: ...


class IProcessorPipeline(Protocol):
    async def apply_prompt(self, processors: Any, data: Any, *, context: Any) -> Any: ...
    async def apply_output(self, processors: Any, data: Any, *, context: Any) -> Any: ...


class IValidatorRunner(Protocol):
    async def validate(
        self, validators: List[Any], data: Any, *, context: Any
    ) -> List[ValidationResult]: ...


class IPluginRunner(Protocol):
    async def run_plugins(
        self,
        plugins: List[tuple[Any, int]],
        data: Any,
        *,
        context: Any,
        resources: Optional[Any] = None,
    ) -> Any: ...


class IUsageMeter(Protocol):
    async def add(self, cost_usd: float, prompt_tokens: int, completion_tokens: int) -> None: ...
    async def guard(
        self, limits: UsageLimits, step_history: Optional[List[Any]] = None
    ) -> None: ...
    async def snapshot(self) -> tuple[float, int, int]: ...


class ITelemetry(Protocol):
    def trace(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


class ISerializer(Protocol):
    def serialize(self, obj: Any) -> bytes: ...
    def deserialize(self, blob: bytes) -> Any: ...


class IHasher(Protocol):
    def digest(self, data: bytes) -> str: ...


class ICacheBackend(Protocol):
    async def get(self, key: str) -> Optional[StepResult]: ...
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
