"""Thin orchestrator facade delegating to AgentExecutionRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ...domain.models import StepOutcome, StepResult, Success
from ...infra import telemetry
from .agent_execution_runner import AgentExecutionRunner
from .agent_fallback_handler import AgentFallbackHandler
from .agent_plugin_runner import AgentPluginRunner

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class AgentOrchestrator:
    """Coordinates agent execution using injected runners."""

    def __init__(self, *, plugin_runner: Optional[Any] = None) -> None:
        plugin_helper = plugin_runner if isinstance(plugin_runner, AgentPluginRunner) else None
        self._execution_runner = AgentExecutionRunner(
            plugin_runner=plugin_helper or AgentPluginRunner(),
            fallback_handler=AgentFallbackHandler(),
        )

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
        """Orchestrate agent execution with retries, validation, plugins, fallback."""
        telemetry.logfire.debug(
            f"[AgentOrchestrator] Orchestrate simple agent step: {getattr(step, 'name', '<unnamed>')} depth={fallback_depth}"
        )
        self.reset_fallback_chain(core, fallback_depth)
        self.guard_fallback_loop(core, step, fallback_depth)
        return await self._execution_runner.execute(
            core=core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=fallback_depth,
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
            if isinstance(outcome, Success) and cache_key and core._enable_cache:
                await core._cache_manager.maybe_persist_step_result(
                    step, outcome.step_result, cache_key, ttl_s=3600
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
        handler = getattr(core, "_fallback_handler", None)
        if handler is None:
            return

        # Check chain length first
        if depth > handler.MAX_CHAIN_LENGTH:
            from ...exceptions import InfiniteFallbackError

            telemetry.logfire.warning(
                f"Fallback chain length exceeded maximum of {handler.MAX_CHAIN_LENGTH}"
            )
            raise InfiniteFallbackError(
                f"Fallback chain exceeded maximum length ({handler.MAX_CHAIN_LENGTH})"
            )

        # Check for cycles
        if depth > 0 and handler.is_step_in_chain(step):
            from ...exceptions import InfiniteFallbackError

            telemetry.logfire.warning(
                f"Infinite fallback loop detected for step '{getattr(step, 'name', '<unnamed>')}'"
            )
            raise InfiniteFallbackError(
                f"Infinite fallback loop detected: step '{getattr(step, 'name', '<unnamed>')}' is already in the chain"
            )

        # Add to chain if safe
        if depth > 0:
            handler.push_to_chain(step)
