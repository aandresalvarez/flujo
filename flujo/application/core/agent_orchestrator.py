"""Agent step orchestration (retries, validation, plugins, fallback)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ...domain.dsl.loop import LoopStep
from ...domain.models import StepOutcome, StepResult, Success
from ...exceptions import InfiniteFallbackError
from ...infra import telemetry
from .default_components import DefaultPluginRunner

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class AgentOrchestrator:
    """Encapsulates agent orchestration previously hosted on ExecutorCore."""

    def __init__(self, *, plugin_runner: Optional[Any] = None) -> None:
        self._plugin_runner = plugin_runner or DefaultPluginRunner()

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[Any],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        fallback_depth: int,
    ) -> StepOutcome[StepResult]:
        # Delegate to the existing core helper for now; further extraction can move logic here.
        return await core._execute_agent_with_orchestration(
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            fallback_depth,
        )

    async def cache_success_if_applicable(
        self,
        *,
        core: Any,
        step: Any,
        cache_key: Optional[str],
        outcome: StepOutcome[StepResult],
    ) -> None:
        """Persist successful agent outcomes via CacheManager."""
        try:
            if (
                isinstance(outcome, Success)
                and cache_key
                and core._enable_cache
                and not isinstance(step, LoopStep)
                and not (
                    isinstance(getattr(outcome.step_result, "metadata_", None), dict)
                    and outcome.step_result.metadata_.get("no_cache")
                )
            ):
                await core._cache_manager.persist_step_result(
                    cache_key, outcome.step_result, ttl_s=3600
                )
        except Exception:
            telemetry.logfire.debug(
                f"[AgentOrchestrator] cache persist failed for {getattr(step, 'name', '<unnamed>')}"
            )

    def reset_fallback_chain(self, core: Any, depth: int) -> None:
        """Reset fallback handler at top-level invocations."""
        try:
            if int(depth) == 0:
                core._fallback_handler.reset()
        except Exception:
            pass

    def guard_fallback_loop(self, core: Any, step: Any, depth: int) -> None:
        """Guard against fallback loops."""
        if depth > core._fallback_handler.MAX_CHAIN_LENGTH:
            raise InfiniteFallbackError(
                f"Fallback chain length exceeded maximum of {core._fallback_handler.MAX_CHAIN_LENGTH}"
            )
