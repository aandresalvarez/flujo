"""Thin orchestrator facade delegating to AgentExecutionRunner."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ....domain.models import BaseModel, StepOutcome, StepResult, Success
from ....infra import telemetry
from .agent_execution_runner import AgentExecutionRunner
from .agent_fallback_handler import AgentFallbackHandler
from .agent_plugin_runner import AgentPluginRunner

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


class AgentOrchestrator:
    """Coordinates agent execution using injected runners."""

    def __init__(self, *, plugin_runner: object | None = None) -> None:
        plugin_helper = plugin_runner if isinstance(plugin_runner, AgentPluginRunner) else None
        self._execution_runner = AgentExecutionRunner(
            plugin_runner=plugin_helper or AgentPluginRunner(),
            fallback_handler=AgentFallbackHandler(),
        )

    async def execute(
        self,
        *,
        core: "ExecutorCore[BaseModel]",
        step: object,
        data: object,
        context: BaseModel | None,
        resources: object | None,
        limits: object | None,
        stream: bool,
        on_chunk: Callable[[object], Awaitable[None]] | None,
        cache_key: str | None,
        fallback_depth: int,
    ) -> StepOutcome[StepResult]:
        """Orchestrate agent execution with retries, validation, plugins, fallback."""
        telemetry.logfire.debug(
            f"[AgentOrchestrator] Orchestrate simple agent step: {getattr(step, 'name', '<unnamed>')} depth={fallback_depth}"
        )
        governance = getattr(core, "_governance_engine", None)
        effective_data = data
        if governance is not None:
            effective_data = await governance.enforce(
                core=core, step=step, data=data, context=context, resources=resources
            )
        self.reset_fallback_chain(core, fallback_depth)
        self.guard_fallback_loop(core, step, fallback_depth)
        return await self._execution_runner.execute(
            core=core,
            step=step,
            data=effective_data,
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
        core: object,
        step: object,
        cache_key: str | None,
        outcome: StepOutcome[StepResult],
    ) -> None:
        """Persist successful agent outcomes via CacheManager."""
        try:
            enable_cache = bool(getattr(core, "_enable_cache", False))
            cache_mgr = getattr(core, "_cache_manager", None)
            persist_fn = (
                getattr(cache_mgr, "maybe_persist_step_result", None)
                if cache_mgr is not None
                else None
            )
            if isinstance(outcome, Success) and cache_key and enable_cache and callable(persist_fn):
                res = persist_fn(step, outcome.step_result, cache_key, ttl_s=3600)
                if inspect.isawaitable(res):
                    await res
        except Exception:
            telemetry.logfire.debug(
                f"[AgentOrchestrator] cache persist failed for {getattr(step, 'name', '<unnamed>')}"
            )

    def reset_fallback_chain(self, core: object, depth: int) -> None:
        """Reset fallback handler at top-level invocations."""
        try:
            if int(depth) == 0:
                handler = getattr(core, "_fallback_handler", None)
                reset_fn = getattr(handler, "reset", None) if handler is not None else None
                if callable(reset_fn):
                    reset_fn()
        except Exception:
            pass

    def guard_fallback_loop(self, core: object, step: object, depth: int) -> None:
        """Guard against fallback loops."""
        handler = getattr(core, "_fallback_handler", None)
        if handler is None:
            return

        # Check chain length first
        max_len_obj = getattr(handler, "MAX_CHAIN_LENGTH", None)
        max_len = max_len_obj if isinstance(max_len_obj, int) else None
        if max_len is not None and depth > max_len:
            from ....exceptions import InfiniteFallbackError

            telemetry.logfire.warning(f"Fallback chain length exceeded maximum of {max_len}")
            raise InfiniteFallbackError(f"Fallback chain exceeded maximum length ({max_len})")

        # Check for cycles
        is_in_chain_fn = getattr(handler, "is_step_in_chain", None)
        if depth > 0 and callable(is_in_chain_fn):
            try:
                if bool(is_in_chain_fn(step)):
                    from ....exceptions import InfiniteFallbackError

                    telemetry.logfire.warning(
                        f"Infinite fallback loop detected for step '{getattr(step, 'name', '<unnamed>')}'"
                    )
                    raise InfiniteFallbackError(
                        f"Infinite fallback loop detected: step '{getattr(step, 'name', '<unnamed>')}' is already in the chain"
                    )
            except InfiniteFallbackError:
                raise
            except Exception:
                pass

        # Add to chain if safe
        if depth > 0:
            push_fn = getattr(handler, "push_to_chain", None)
            if callable(push_fn):
                push_fn(step)
