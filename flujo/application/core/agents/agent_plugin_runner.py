from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from ....domain.models import BaseModel, StepOutcome, StepResult, Success, Failure
from ....exceptions import InfiniteRedirectError
from ....infra import telemetry
from ....utils.performance import time_perf_ns, time_perf_ns_to_seconds


@dataclass
class PluginState:
    """Mutable plugin execution state shared across attempts."""

    primary_tokens_known: bool
    primary_tokens_total: int
    primary_latency_total: float
    best_primary_tokens: int
    best_primary_cost_usd: float
    last_plugin_failure_feedback: Optional[str] = None


@dataclass
class PluginHandlingResult:
    """Outcome of a plugin run within an agent attempt."""

    processed_output: object
    outcome: StepOutcome[StepResult] | None
    retry: bool
    resources_closed: bool
    state: PluginState
    attempt_exception: BaseException | None


def _in_loop_context(attempt_context: BaseModel | None, core: object, step: object) -> bool:
    try:
        return bool(
            getattr(attempt_context, "_last_loop_iterations", None) is not None
            or getattr(attempt_context, "_loop_iteration_active", False)
            or getattr(core, "_inside_loop_iteration", False)
            or getattr(step, "_force_loop_fallback", False)
        )
    except Exception:
        return False


class AgentPluginRunner:
    """Encapsulates plugin orchestration for agent steps."""

    async def handle_plugins(
        self,
        *,
        core: object,
        step: object,
        data: object,
        context: BaseModel | None,
        attempt_context: BaseModel | None,
        attempt_resources: object | None,
        limits: object | None,
        stream: bool,
        on_chunk: Optional[Callable[[object], Awaitable[None]]],
        fallback_depth: int,
        attempt: int,
        total_attempts: int,
        start_ns: int,
        result: StepResult,
        processed_output: object,
        pre_attempt_context: BaseModel | None,
        prompt_tokens_latest: int,
        close_resources: Callable[[BaseException | None], Awaitable[None]],
        state: PluginState,
    ) -> PluginHandlingResult:
        """Run plugins and handle retry/fallback paths."""
        if not getattr(step, "plugins", None):
            return PluginHandlingResult(
                processed_output=processed_output,
                outcome=None,
                retry=False,
                resources_closed=False,
                state=state,
                attempt_exception=None,
            )

        def _safe_step_name(step_obj: object) -> str:
            safe_fn = getattr(core, "_safe_step_name", None)
            if callable(safe_fn):
                try:
                    return str(safe_fn(step_obj))
                except Exception:
                    pass
            return str(getattr(step_obj, "name", "<unnamed>"))

        telemetry.logfire.info(
            f"[AgentExecutionRunner] Running plugins for step '{getattr(step, 'name', '<unnamed>')}'"
        )
        timeout_s = None
        try:
            cfg = getattr(step, "config", None)
            if cfg is not None and getattr(cfg, "timeout_s", None) is not None:
                timeout_s = float(cfg.timeout_s)
        except Exception:
            timeout_s = None

        try:
            plugin_redirector = getattr(core, "plugin_redirector", None)
            run_plugins = (
                getattr(plugin_redirector, "run", None) if plugin_redirector is not None else None
            )
            if not callable(run_plugins):
                raise TypeError("Executor core must provide plugin_redirector.run()")
            processed_output = await run_plugins(
                initial=processed_output,
                step=step,
                data=data,
                context=attempt_context,
                resources=attempt_resources,
                timeout_s=timeout_s,
            )
            telemetry.logfire.info(
                f"[AgentExecutionRunner] Plugins completed for step '{getattr(step, 'name', '<unnamed>')}'"
            )
            return PluginHandlingResult(
                processed_output=processed_output,
                outcome=None,
                retry=False,
                resources_closed=False,
                state=state,
                attempt_exception=None,
            )
        except Exception as e:
            if e.__class__.__name__ == "InfiniteRedirectError" or isinstance(
                e, InfiniteRedirectError
            ):
                raise
            if isinstance(e, asyncio.TimeoutError):
                raise
            state.last_plugin_failure_feedback = f"Plugin execution failed: {e!s}"
            result.success = False
            result.feedback = state.last_plugin_failure_feedback
            result.output = None
            result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
            attempt_exception: BaseException | None = e
            if getattr(step, "updates_context", False) and attempt_context is not None:
                result.branch_context = attempt_context
            try:
                if state.primary_tokens_known:
                    state.best_primary_tokens = max(
                        state.best_primary_tokens, int(result.token_counts or 0)
                    )
            except Exception:
                pass
            try:
                state.primary_latency_total += float(result.latency_s or 0.0)
            except Exception:
                pass

            is_loop_context = _in_loop_context(attempt_context, core, step)
            fb_step = getattr(step, "fallback_step", None)
            if hasattr(fb_step, "_mock_name") and not hasattr(fb_step, "agent"):
                fb_step = None
            if fb_step is None and attempt < total_attempts:
                telemetry.logfire.warning(
                    f"Step '{getattr(step, 'name', '<unnamed>')}' plugin attempt {attempt}/{total_attempts} failed: {e}"
                )
                return PluginHandlingResult(
                    processed_output=processed_output,
                    outcome=None,
                    retry=True,
                    resources_closed=False,
                    state=state,
                    attempt_exception=attempt_exception,
                )
            if fb_step is not None and attempt < total_attempts and not is_loop_context:
                telemetry.logfire.warning(
                    f"Step '{getattr(step, 'name', '<unnamed>')}' plugin attempt {attempt}/{total_attempts} failed; retrying primary before fallback"
                )
                return PluginHandlingResult(
                    processed_output=processed_output,
                    outcome=None,
                    retry=True,
                    resources_closed=False,
                    state=state,
                    attempt_exception=attempt_exception,
                )
            if fb_step is None:
                if getattr(step, "updates_context", False) and attempt_context is not None:
                    result.branch_context = attempt_context
                try:
                    if not result.token_counts and state.primary_tokens_known:
                        result.token_counts = state.best_primary_tokens
                    if not result.cost_usd:
                        result.cost_usd = state.best_primary_cost_usd
                except Exception:
                    pass
                try:
                    if (
                        context is not None
                        and pre_attempt_context is not None
                        and getattr(step, "updates_context", False)
                    ):
                        from ..context_manager import ContextManager as _CM

                        _CM.merge(context, pre_attempt_context)
                except Exception:
                    pass
                await close_resources(e)
                return PluginHandlingResult(
                    processed_output=processed_output,
                    outcome=Failure(error=e, feedback=result.feedback, step_result=result),
                    retry=False,
                    resources_closed=True,
                    state=state,
                    attempt_exception=attempt_exception,
                )

            primary_fb = result.feedback or "Plugin execution failed"
            try:
                telemetry.logfire.debug(
                    f"[AgentExecutionRunner] Invoking fallback step '{getattr(fb_step, 'name', '<unnamed>')}' after plugin failure for '{getattr(step, 'name', '<unnamed>')}' attempt={attempt}/{total_attempts}"
                )
            except Exception:
                pass
            try:
                execute_fn = getattr(core, "execute", None)
                if not callable(execute_fn):
                    raise TypeError("Executor core must provide execute()")
                fallback_result_sr = await execute_fn(
                    step=fb_step,
                    data=data,
                    context=attempt_context,
                    resources=attempt_resources,
                    limits=limits,
                    stream=stream,
                    on_chunk=on_chunk,
                    _fallback_depth=fallback_depth + 1,
                )
            except Exception as fb_exc:
                if getattr(fb_exc, "__class__", type(fb_exc)).__name__ == "InfiniteFallbackError":
                    fb_txt = (
                        f"Fallback loop detected for step '{getattr(fb_step, 'name', '<unnamed>')}'"
                    )
                    return PluginHandlingResult(
                        processed_output=processed_output,
                        outcome=Failure(
                            error=fb_exc,
                            feedback=fb_txt,
                            step_result=StepResult(
                                name=_safe_step_name(step),
                                output=None,
                                success=False,
                                attempts=result.attempts,
                                latency_s=time_perf_ns_to_seconds(time_perf_ns() - start_ns),
                                token_counts=result.token_counts,
                                cost_usd=result.cost_usd,
                                feedback=fb_txt,
                                branch_context=None,
                                metadata_={"fallback_triggered": True},
                                step_history=[],
                            ),
                        ),
                        retry=False,
                        resources_closed=False,
                        state=state,
                        attempt_exception=attempt_exception,
                    )
                raise

            unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
            if not callable(unwrap_fn):
                raise TypeError(
                    "Executor core must provide _unwrap_outcome_to_step_result()"
                ) from None
            fallback_result_sr = unwrap_fn(fallback_result_sr, _safe_step_name(fb_step))
            if fallback_result_sr.metadata_ is None:
                fallback_result_sr.metadata_ = {}
            fallback_result_sr.metadata_["fallback_triggered"] = True
            combo = (
                f"Original error: {result.feedback}; Fallback error: {fallback_result_sr.feedback}"
            )
            try:
                fb_tokens = int(fallback_result_sr.token_counts)
            except Exception:
                try:
                    fb_tokens = int(getattr(fallback_result_sr, "token_counts", 0))
                except Exception:
                    fb_tokens = 0
            try:
                if state.best_primary_tokens:
                    primary_tokens = state.best_primary_tokens * max(1, total_attempts)
                elif state.primary_tokens_total:
                    primary_tokens = state.primary_tokens_total * max(1, total_attempts)
                else:
                    primary_tokens = max(1, total_attempts)
                if not primary_tokens:
                    primary_tokens = int(result.token_counts or 0)
            except Exception:
                primary_tokens = int(result.token_counts or 0)

            if fallback_result_sr.success:
                fallback_result_sr.feedback = None
                fallback_result_sr.metadata_["original_error"] = primary_fb
                try:
                    fb_attempts = int(getattr(fallback_result_sr, "attempts", 1) or 1)
                    fallback_result_sr.attempts = int(total_attempts + fb_attempts)
                except Exception:
                    pass
                try:
                    unit_primary = int(state.best_primary_tokens)
                except Exception:
                    unit_primary = 0
                if unit_primary == 0:
                    try:
                        unit_primary = int(state.primary_tokens_total) // max(1, total_attempts)
                    except Exception:
                        pass
                if unit_primary == 0:
                    try:
                        unit_primary = int(result.token_counts or 0)
                    except Exception:
                        unit_primary = 0
                primary_total_tokens = (
                    state.primary_tokens_total
                    if state.primary_tokens_total not in (None, 0)
                    else unit_primary * max(1, total_attempts)
                )
                fb_unit = fb_tokens if fb_tokens is not None else unit_primary
                fallback_result_sr.token_counts = primary_total_tokens + fb_unit
                try:
                    fallback_result_sr.token_counts = int(fallback_result_sr.token_counts)
                except Exception:
                    pass
                try:
                    usage_meter = getattr(core, "_usage_meter", None)
                    add_fn = getattr(usage_meter, "add", None) if usage_meter is not None else None
                    if callable(add_fn):
                        await add_fn(
                            float(fallback_result_sr.cost_usd or 0.0),
                            prompt_tokens_latest + fb_tokens,
                            0,
                        )
                except Exception:
                    pass
                try:
                    fb_handler = getattr(core, "_fallback_handler", None)
                    reset_fn = (
                        getattr(fb_handler, "reset", None) if fb_handler is not None else None
                    )
                    if callable(reset_fn):
                        reset_fn()
                except Exception:
                    pass
                await close_resources(None)
                return PluginHandlingResult(
                    processed_output=fallback_result_sr.output,
                    outcome=Success(step_result=fallback_result_sr),
                    retry=False,
                    resources_closed=True,
                    state=state,
                    attempt_exception=attempt_exception,
                )

            await close_resources(e)
            return PluginHandlingResult(
                processed_output=processed_output,
                outcome=Failure(
                    error=Exception(combo),
                    feedback=combo,
                    step_result=StepResult(
                        name=_safe_step_name(step),
                        output=None,
                        success=False,
                        attempts=result.attempts,
                        latency_s=fallback_result_sr.latency_s,
                        token_counts=primary_tokens + fb_tokens,
                        cost_usd=fallback_result_sr.cost_usd,
                        feedback=combo,
                        branch_context=None,
                        metadata_=fallback_result_sr.metadata_,
                        step_history=[],
                    ),
                ),
                retry=False,
                resources_closed=True,
                state=state,
                attempt_exception=attempt_exception,
            )
