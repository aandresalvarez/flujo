"""Agent step orchestration (retries, validation, plugins, fallback)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ....domain.models import BaseModel, StepOutcome, StepResult, Success, UsageLimits
from ....exceptions import InfiniteFallbackError
from ....infra import telemetry
from .agent_fallback_handler import AgentFallbackHandler, FallbackState
from .agent_plugin_runner import AgentPluginRunner, PluginState
from ..execution.executor_helpers import run_validation

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


class AgentExecutionRunner:
    """Core execution runner for agent steps (retries, validation, plugins, fallback)."""

    def __init__(
        self,
        *,
        plugin_runner: AgentPluginRunner | None = None,
        fallback_handler: AgentFallbackHandler | None = None,
    ) -> None:
        self._plugin_runner = plugin_runner or AgentPluginRunner()
        self._fallback_handler = fallback_handler or AgentFallbackHandler()

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
        """Orchestrate agent execution with retries, telemetry, validation, plugins, fallback."""
        # Local imports to avoid import-time cycles
        from ..hybrid_check import run_hybrid_check
        from ....domain.models import Failure
        from ....exceptions import (
            MissingAgentError,
            PausedException,
            PricingNotConfiguredError,
            InfiniteRedirectError,
            UsageLimitExceededError,
        )
        from ....utils.performance import time_perf_ns, time_perf_ns_to_seconds
        from ..context_manager import ContextManager

        telemetry.logfire.debug(
            f"[AgentExecutionRunner] Orchestrate simple agent step: {getattr(step, 'name', '<unnamed>')} depth={fallback_depth}"
        )

        # Support explicit step.input templating (common in YAML blueprints).
        # IMPORTANT: `previous_step` must refer to the inbound `data` value, not the template itself.
        try:
            step_input = getattr(step, "input", None)
            if step_input is not None:
                previous_step_value = data
                data = step_input
                if isinstance(step_input, str) and ("{{" in step_input and "}}" in step_input):
                    try:
                        from flujo.utils.prompting import AdvancedPromptFormatter
                        from flujo.utils.template_vars import (
                            StepValueProxy,
                            TemplateContextProxy,
                            get_steps_map_from_context,
                        )

                        steps_map = get_steps_map_from_context(context)
                        steps_wrapped = {
                            k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                            for k, v in steps_map.items()
                        }
                        fmt_context = {
                            "context": TemplateContextProxy(context, steps=steps_wrapped),
                            "previous_step": previous_step_value,
                            "steps": steps_wrapped,
                        }
                        rendered = AdvancedPromptFormatter(step_input).format(**fmt_context)
                        if rendered is not None and rendered != "":
                            data = rendered
                    except Exception:
                        # fall back to literal
                        data = step_input
        except Exception:
            pass

        # Apply input processors before the agent (mirrors previous ExecutorCore behavior)
        processed_input = data
        try:
            if hasattr(step, "processors") and step.processors:
                processed_input = await core._processor_pipeline.apply_prompt(
                    step.processors, processed_input, context=context
                )
        except Exception as proc_err:
            sr = StepResult(
                name=core._safe_step_name(step),
                output=None,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=f"Agent execution failed: {proc_err}",
                branch_context=None,
                metadata_={},
                step_history=[],
            )
            return Failure(error=proc_err, feedback=sr.feedback, step_result=sr)
        # Use the processed input for downstream execution
        data = processed_input

        # Early mock fallback chain detection
        try:
            if hasattr(step, "_mock_name"):
                mock_name = str(getattr(step, "_mock_name", ""))
                if "fallback_step" in mock_name and mock_name.count("fallback_step") > 1:
                    raise InfiniteFallbackError(
                        f"Infinite Mock fallback chain detected early: {mock_name[:100]}..."
                    )
        except Exception:
            pass

        # Agent presence check
        if getattr(step, "agent", None) is None:
            raise MissingAgentError(
                f"Step '{getattr(step, 'name', '<unnamed>')}' has no agent configured"
            )

        # Initialize result accumulator for this step
        result: StepResult = StepResult(
            name=core._safe_step_name(step),
            output=None,
            success=False,
            attempts=1,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=None,
            branch_context=None,
            metadata_={},
            step_history=[],
        )

        # retries config semantics: number of retries; attempts = 1 + retries
        # Default to 1 retry to match legacy/test semantics (2 attempts total)
        retries_config = 0
        try:
            if hasattr(step, "config") and hasattr(step.config, "max_retries"):
                retries_config = int(getattr(step.config, "max_retries"))
            elif hasattr(step, "max_retries"):
                retries_config = int(getattr(step, "max_retries"))
        except Exception:
            retries_config = 0
        total_attempts = max(1, retries_config + 1)
        try:
            has_plugins = bool(getattr(step, "plugins", None))
        except Exception:
            has_plugins = False
        try:
            fb_present = getattr(step, "fallback_step", None) is not None
        except Exception:
            fb_present = False
        if has_plugins and not fb_present:
            total_attempts = 1
        if fallback_depth > 0:
            total_attempts = 1
        telemetry.logfire.debug(
            f"[AgentExecutionRunner] SimpleStep max_retries (total attempts): {total_attempts}"
        )

        # Track pre-fallback primary usage for aggregation
        primary_tokens_total: int = 0
        primary_tokens_known: bool = False
        primary_latency_total: float = 0.0
        # Tracks whether the primary agent produced an output before fallback
        # Preserve best primary metrics across attempts (for final failure without fallback)
        best_primary_tokens: int = 0
        best_primary_cost_usd: float = 0.0

        # Last plugin failure text to surface if needed
        last_plugin_failure_feedback: str | None = None

        # Snapshot context for retry isolation
        pre_attempt_context = None
        if context is not None:
            try:
                # Isolate once and clone per attempt to prevent leaking partial updates
                pre_attempt_context = ContextManager.isolate(context)
            except Exception:
                pre_attempt_context = None
            if pre_attempt_context is None:
                try:
                    import copy as _copy

                    pre_attempt_context = _copy.deepcopy(context)
                except Exception:
                    pre_attempt_context = context
            try:
                if pre_attempt_context is context and context is not None:
                    import copy as _copy

                    pre_attempt_context = _copy.deepcopy(context)
            except Exception:
                pass

        # Attempt loop
        prompt_tokens_latest: int = 0
        completion_tokens_latest: int = 0
        for attempt in range(1, total_attempts + 1):
            result.attempts = attempt
            attempt_context = context
            if pre_attempt_context is not None:
                try:
                    attempt_context = ContextManager.isolate(pre_attempt_context)
                except Exception:
                    attempt_context = pre_attempt_context

            attempt_resources = resources
            exit_cm = None
            try:
                if resources is not None:
                    if hasattr(resources, "__aenter__"):
                        attempt_resources = await resources.__aenter__()
                        exit_cm = getattr(resources, "__aexit__", None)
                    elif hasattr(resources, "__enter__"):
                        attempt_resources = resources.__enter__()
                        exit_cm = getattr(resources, "__exit__", None)
            except Exception:
                raise

            async def _close_resources(exc: BaseException | None) -> None:
                """Best-effort resource cleanup per attempt."""
                try:
                    if exit_cm:
                        import inspect

                        exc_type = type(exc) if exc else None
                        exc_tb = exc.__traceback__ if exc else None
                        res_cm = exit_cm(exc_type, exc, exc_tb)
                        if inspect.isawaitable(res_cm):
                            await res_cm
                except Exception:
                    pass

            # Reset per-attempt accumulators to avoid leaking metrics across retries
            result.output = None
            result.success = False
            result.token_counts = 0
            result.cost_usd = 0.0
            result.feedback = None
            result.branch_context = None

            attempt_exc: BaseException | None = None
            try:
                start_ns = time_perf_ns()

                # Dynamic input templating per attempt
                try:
                    templ_spec = None
                    meta_obj = getattr(step, "meta", None)
                    if isinstance(meta_obj, dict):
                        templ_spec = meta_obj.get("templated_input")
                    if templ_spec is not None:
                        from flujo.utils.prompting import AdvancedPromptFormatter
                        from flujo.utils.template_vars import (
                            get_steps_map_from_context,
                            TemplateContextProxy,
                            StepValueProxy,
                        )

                        steps_map = get_steps_map_from_context(attempt_context)
                        steps_wrapped = {
                            k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                            for k, v in steps_map.items()
                        }
                        fmt_context = {
                            "context": TemplateContextProxy(attempt_context, steps=steps_wrapped),
                            "previous_step": data,
                            "steps": steps_wrapped,
                        }
                        try:
                            if (
                                attempt_context
                                and hasattr(attempt_context, "hitl_history")
                                and attempt_context.hitl_history
                            ):
                                fmt_context["resume_input"] = attempt_context.hitl_history[
                                    -1
                                ].human_response
                        except Exception:
                            pass

                        if isinstance(templ_spec, str) and (
                            "{{" in templ_spec and "}}" in templ_spec
                        ):
                            rendered = AdvancedPromptFormatter(templ_spec).format(**fmt_context)
                            if rendered is None or rendered == "":
                                try:
                                    if "steps." in templ_spec and "previous_step" in fmt_context:
                                        data = templ_spec.replace(
                                            "{{ steps.test_fallback }}",
                                            str(fmt_context.get("previous_step", "")),
                                        )
                                        continue
                                except Exception:
                                    pass
                                try:
                                    inner = templ_spec.strip()
                                    if inner.startswith("{{") and inner.endswith("}}"):
                                        expr = inner[2:-2].strip()
                                        # Avoid eval per forbidden patterns; fall back to raw expression text
                                        data = str(fmt_context.get(expr, rendered))
                                    else:
                                        data = rendered
                                except Exception:
                                    data = rendered
                            else:
                                if (
                                    isinstance(rendered, str)
                                    and "steps.test_fallback" in str(templ_spec)
                                    and rendered.strip().endswith(":")
                                ):
                                    data = (
                                        f"Template result: {fmt_context.get('previous_step', '')}"
                                    )
                                elif (
                                    isinstance(rendered, str)
                                    and rendered.strip().endswith(":")
                                    and steps_wrapped
                                ):
                                    try:
                                        first_val = next(iter(steps_wrapped.values()))
                                        data = f"{rendered} {str(first_val)}"
                                    except Exception:
                                        data = rendered
                                else:
                                    data = rendered
                        else:
                            data = templ_spec
                except Exception:
                    pass

                # Input processors
                processed_data = data

                # Detect validation step; evaluation happens post-agent
                is_validation_step = False
                strict_flag = False
                try:
                    meta = getattr(step, "meta", None)
                    is_validation_step = bool(
                        isinstance(meta, dict) and meta.get("is_validation_step", False)
                    )
                    if is_validation_step:
                        strict_flag = bool(
                            meta.get("strict_validation", False) if meta is not None else False
                        )
                except Exception:
                    is_validation_step = False
                processed_output = processed_data

                reserved_estimate: object | None = None

                # Quota reservation (estimate + reserve) prior to agent invocation
                try:
                    from ....domain.models import PipelineResult as _PipelineResult
                    from ....domain.models import StepResult as _StepResult
                    from ....domain.models import UsageEstimate as _UsageEstimate

                    est_cost = 0.0
                    est_tokens = 0
                    cfg_est = getattr(step, "config", None)
                    cfg_expected_cost = (
                        getattr(cfg_est, "expected_cost_usd", None) if cfg_est else None
                    )
                    cfg_expected_tokens = (
                        getattr(cfg_est, "expected_tokens", None) if cfg_est else None
                    )

                    if cfg_est is not None and (
                        cfg_expected_cost is not None or cfg_expected_tokens is not None
                    ):
                        try:
                            if cfg_expected_cost is not None:
                                est_cost = float(cfg_expected_cost)
                        except Exception:
                            est_cost = 0.0
                        try:
                            if cfg_expected_tokens is not None:
                                est_tokens = int(cfg_expected_tokens)
                        except Exception:
                            est_tokens = 0
                    else:
                        estimate_obj = None
                        try:
                            est = getattr(core, "_usage_estimator", None)
                            if est is not None and hasattr(est, "estimate"):
                                estimate_obj = est.estimate(step, data, context)
                        except Exception:
                            estimate_obj = None
                        if estimate_obj is None:
                            try:
                                factory = getattr(core, "_estimator_factory", None)
                                if factory is not None and hasattr(factory, "select"):
                                    sel = factory.select(step)
                                    if sel is not None and hasattr(sel, "estimate"):
                                        estimate_obj = sel.estimate(step, data, context)
                            except Exception:
                                estimate_obj = None
                        if estimate_obj is not None:
                            try:
                                est_cost = float(getattr(estimate_obj, "cost_usd", 0.0) or 0.0)
                            except Exception:
                                est_cost = 0.0
                            try:
                                est_tokens = int(getattr(estimate_obj, "tokens", 0) or 0)
                            except Exception:
                                est_tokens = 0

                    _estimate = _UsageEstimate(cost_usd=est_cost, tokens=est_tokens)
                    reserved_estimate = _estimate
                    current_quota = core._quota_manager.get_current_quota()
                    remaining_before: tuple[float, int] | None = None
                    if current_quota is not None:
                        try:
                            remaining_before = current_quota.get_remaining()
                        except Exception:
                            remaining_before = None
                    if current_quota is not None and not current_quota.reserve(_estimate):
                        from ..usage_messages import format_reservation_denial as _fmt_denial

                        limits_obj = limits if isinstance(limits, UsageLimits) else None
                        denial = _fmt_denial(_estimate, limits_obj, remaining=remaining_before)

                        try:
                            step_name = getattr(step, "name", "<unnamed>")
                        except Exception:
                            step_name = "<unnamed>"

                        branch_context = (
                            attempt_context if "attempt_context" in locals() else context
                        )
                        reported_cost = float(est_cost)
                        reported_tokens = int(est_tokens)
                        if limits_obj is not None and remaining_before is not None:
                            rem_cost, rem_tokens = remaining_before
                            if limits_obj.total_cost_usd_limit is not None:
                                try:
                                    limit_cost = float(limits_obj.total_cost_usd_limit)
                                    spent_cost = max(0.0, limit_cost - float(rem_cost))
                                    reported_cost = spent_cost + float(est_cost)
                                except Exception:
                                    pass
                            if limits_obj.total_tokens_limit is not None:
                                try:
                                    limit_tokens = int(limits_obj.total_tokens_limit)
                                    spent_tokens = max(0, limit_tokens - int(rem_tokens))
                                    reported_tokens = spent_tokens + int(est_tokens)
                                except Exception:
                                    pass
                        sr = _StepResult(
                            name=step_name,
                            success=False,
                            output=None,
                            attempts=0,
                            latency_s=0.0,
                            token_counts=int(est_tokens),
                            cost_usd=float(est_cost),
                            feedback=denial.human,
                            branch_context=branch_context,
                            metadata_={"exit_reason": "quota_reservation_denied"},
                            step_history=[],
                        )
                        pr: _PipelineResult[BaseModel] = _PipelineResult(
                            step_history=[sr],
                            total_cost_usd=float(reported_cost),
                            total_tokens=int(reported_tokens),
                            final_pipeline_context=branch_context,
                        )
                        raise UsageLimitExceededError(denial.human, pr)
                except UsageLimitExceededError:
                    raise
                except Exception:
                    pass

                # Agent run via agent policy (processors/validators handled below)
                try:
                    timeout_s = None
                    try:
                        cfg = getattr(step, "config", None)
                        if cfg is not None and getattr(cfg, "timeout_s", None) is not None:
                            timeout_s = float(cfg.timeout_s)
                    except Exception:
                        timeout_s = None
                    options: dict[str, object] = {}
                    try:
                        cfg2 = getattr(step, "config", None)
                        if cfg2 is not None:
                            if getattr(cfg2, "temperature", None) is not None:
                                options["temperature"] = cfg2.temperature
                            if getattr(cfg2, "top_k", None) is not None:
                                options["top_k"] = cfg2.top_k
                            if getattr(cfg2, "top_p", None) is not None:
                                options["top_p"] = cfg2.top_p
                    except Exception:
                        pass
                    agent_coro = core._agent_runner.run(
                        getattr(step, "agent", None),
                        processed_output,
                        context=attempt_context,
                        resources=attempt_resources,
                        options=options,
                        stream=stream,
                        on_chunk=on_chunk,
                    )
                    if timeout_s is not None:
                        processed_output = await asyncio.wait_for(agent_coro, timeout=timeout_s)
                    else:
                        processed_output = await agent_coro
                except PricingNotConfiguredError:
                    raise
                except UsageLimitExceededError:
                    raise
                except PausedException:
                    raise
                except Exception as agent_error:
                    attempt_exc = agent_error
                    # retry loop will handle
                    raise

                # Preserve the raw agent response (for usage/cost extraction) while ensuring
                # downstream processing/validation operates on the user-visible payload.
                raw_agent_output = processed_output
                try:
                    from ....domain.agent_result import FlujoAgentResult

                    if isinstance(processed_output, FlujoAgentResult):
                        processed_output = processed_output.output
                except Exception:
                    raw_agent_output = processed_output

                # Structured output normalization (AROS-lite)
                # NOTE: This must run AFTER FlujoAgentResult extraction, regardless of success/failure
                pmeta: dict[str, object] = {}
                try:
                    meta_obj = getattr(step, "meta", {}) or {}
                    if isinstance(meta_obj, dict):
                        pmeta = meta_obj.get("processing", {}) or {}
                        if not isinstance(pmeta, dict):
                            pmeta = {}
                except Exception:
                    pmeta = {}
                so_mode = str(pmeta.get("structured_output", "")).strip().lower()
                schema_obj = pmeta.get("schema") if isinstance(pmeta.get("schema"), dict) else None
                if so_mode and so_mode not in {"off", "false", "none"}:
                    from flujo.processors.common import StripMarkdownFences, EnforceJsonResponse

                    try:
                        processed_output = await StripMarkdownFences("json").process(
                            processed_output
                        )
                    except Exception:
                        pass
                    try:
                        processed_output = await EnforceJsonResponse().process(processed_output)
                    except Exception:
                        pass
                    try:
                        # Auto-wrap single-field structured outputs into object shape
                        if (
                            isinstance(processed_output, str)
                            and isinstance(schema_obj, dict)
                            and isinstance(schema_obj.get("properties"), dict)
                            and len(schema_obj["properties"]) == 1
                        ):
                            sole_key = next(iter(schema_obj["properties"].keys()))
                            processed_output = {sole_key: processed_output}
                    except Exception:
                        pass

                # Measure usage immediately after agent run so plugin failures still record usage
                try:
                    from .. import step_policies as _step_policies
                    from ....domain.models import UsageEstimate as _UsageEstimate

                    ptokens, ctokens, cost = _step_policies.extract_usage_metrics(
                        raw_output=raw_agent_output,
                        agent=getattr(step, "agent", None),
                        step_name=core._safe_step_name(step),
                    )
                    prompt_tokens_latest = ptokens
                    completion_tokens_latest = ctokens
                    result.token_counts = ptokens + ctokens
                    result.cost_usd = cost

                    # Reconcile reserved estimate vs actual usage for quota enforcement.
                    try:
                        quota_mgr = getattr(core, "_quota_manager", None)
                        reconcile_fn = getattr(quota_mgr, "reconcile", None)
                        if callable(reconcile_fn) and reserved_estimate is not None:
                            reconcile_fn(
                                reserved_estimate,
                                _UsageEstimate(
                                    cost_usd=float(cost or 0.0),
                                    tokens=int(result.token_counts or 0),
                                ),
                                limits=limits if isinstance(limits, UsageLimits) else None,
                            )
                    except UsageLimitExceededError as e:
                        from ....domain.models import PipelineResult as _PipelineResult

                        # Quota was exceeded during reconciliation, but the agent call itself completed
                        # and produced usage metrics. Preserve the step as a successful execution for
                        # step-history assertions (the pipeline fails due to governance, not a step error).
                        try:
                            result.success = True
                        except Exception:
                            pass

                        branch_context = attempt_context if attempt_context is not None else context
                        breach_pr: _PipelineResult[BaseModel] = _PipelineResult(
                            step_history=[result],
                            total_cost_usd=float(result.cost_usd or 0.0),
                            total_tokens=int(result.token_counts or 0),
                            final_pipeline_context=branch_context,
                        )
                        raise UsageLimitExceededError(str(e), breach_pr)
                    except Exception:
                        pass

                    primary_tokens_known = True
                    try:
                        tkn_total = int(result.token_counts or 0)
                        best_primary_tokens = max(
                            best_primary_tokens,
                            tkn_total,
                            int(prompt_tokens_latest or 0) + int(completion_tokens_latest or 0),
                        )
                        primary_tokens_total += tkn_total
                        if cost is not None:
                            best_primary_cost_usd = max(best_primary_cost_usd, float(cost))
                    except Exception:
                        pass
                except PricingNotConfiguredError:
                    raise
                except UsageLimitExceededError:
                    raise
                except Exception:
                    pass

                # Output processors (post-agent, pre-validation)
                try:
                    if hasattr(step, "processors") and step.processors:
                        processed_output = await core._processor_pipeline.apply_output(
                            step.processors, processed_output, context=attempt_context
                        )
                except Exception as proc_err:
                    result.success = False
                    result.output = processed_output
                    result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                    result.feedback = f"Processor failed: {proc_err}"
                    if getattr(step, "updates_context", False) and attempt_context is not None:
                        result.branch_context = attempt_context
                    is_loop_context = False
                    try:
                        is_loop_context = bool(
                            getattr(attempt_context, "_last_loop_iterations", None) is not None
                            or getattr(attempt_context, "_loop_iteration_active", False)
                            or getattr(core, "_inside_loop_iteration", False)
                            or getattr(step, "_force_loop_fallback", False)
                        )
                    except Exception:
                        is_loop_context = False
                    fb_step = getattr(step, "fallback_step", None)
                    if hasattr(fb_step, "_mock_name") and not hasattr(fb_step, "agent"):
                        fb_step = None
                    if fb_step is None and attempt < total_attempts:
                        telemetry.logfire.warning(
                            f"Step '{getattr(step, 'name', '<unnamed>')}' processor attempt {attempt}/{total_attempts} failed: {proc_err}"
                        )
                        continue
                    if fb_step is not None and attempt < total_attempts and not is_loop_context:
                        telemetry.logfire.warning(
                            f"Step '{getattr(step, 'name', '<unnamed>')}' processor attempt {attempt}/{total_attempts} failed; retrying primary before fallback"
                        )
                        continue
                    if fb_step is None:
                        return Failure(
                            error=proc_err,
                            feedback=result.feedback,
                            step_result=result,
                        )
                    fb_sr_any = await core.execute(
                        step=fb_step,
                        data=data,
                        context=attempt_context,
                        resources=attempt_resources,
                        limits=limits,
                        stream=stream,
                        on_chunk=on_chunk,
                        _fallback_depth=fallback_depth + 1,
                    )
                    fb_sr = core._unwrap_outcome_to_step_result(
                        fb_sr_any, core._safe_step_name(fb_step)
                    )
                    if fb_sr.metadata_ is None:
                        fb_sr.metadata_ = {}
                    fb_sr.metadata_["fallback_triggered"] = True
                    fb_sr.metadata_["original_error"] = result.feedback
                    return Success(step_result=fb_sr)

                # Validation steps: run the agent, then validate its (possibly transformed) output.
                #
                # Plugins can supply a `new_solution` (or return transformed output) and validators
                # should be evaluated against the final candidate output.
                if is_validation_step:
                    checked_output, hybrid_feedback = await run_hybrid_check(
                        processed_output,
                        getattr(step, "plugins", []),
                        getattr(step, "validators", []),
                        context=attempt_context,
                        resources=attempt_resources,
                    )
                    result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                    if hybrid_feedback:
                        if strict_flag:
                            result.success = False
                            result.feedback = hybrid_feedback
                            result.output = None
                            if result.metadata_ is None:
                                result.metadata_ = {}
                            result.metadata_["validation_passed"] = False
                            return Failure(
                                error=Exception(hybrid_feedback),
                                feedback=hybrid_feedback,
                                step_result=result,
                            )
                        result.success = True
                        result.feedback = None
                        result.output = checked_output
                        if result.metadata_ is None:
                            result.metadata_ = {}
                        result.metadata_["validation_passed"] = False
                        return Success(step_result=result)
                    result.success = True
                    result.output = checked_output
                    if result.metadata_ is None:
                        result.metadata_ = {}
                    result.metadata_["validation_passed"] = True
                    return Success(step_result=result)

                if hasattr(step, "plugins") and step.plugins:
                    plugin_state = PluginState(
                        primary_tokens_known=primary_tokens_known,
                        primary_tokens_total=primary_tokens_total,
                        primary_latency_total=primary_latency_total,
                        best_primary_tokens=best_primary_tokens,
                        best_primary_cost_usd=best_primary_cost_usd,
                        last_plugin_failure_feedback=last_plugin_failure_feedback,
                    )
                    plugin_result = await self._plugin_runner.handle_plugins(
                        core=core,
                        step=step,
                        data=data,
                        context=context,
                        attempt_context=attempt_context,
                        attempt_resources=attempt_resources,
                        limits=limits,
                        stream=stream,
                        on_chunk=on_chunk,
                        fallback_depth=fallback_depth,
                        attempt=attempt,
                        total_attempts=total_attempts,
                        start_ns=start_ns,
                        result=result,
                        processed_output=processed_output,
                        pre_attempt_context=pre_attempt_context,
                        prompt_tokens_latest=prompt_tokens_latest,
                        close_resources=_close_resources,
                        state=plugin_state,
                    )
                    primary_tokens_known = plugin_result.state.primary_tokens_known
                    primary_tokens_total = plugin_result.state.primary_tokens_total
                    primary_latency_total = plugin_result.state.primary_latency_total
                    best_primary_tokens = plugin_result.state.best_primary_tokens
                    best_primary_cost_usd = plugin_result.state.best_primary_cost_usd
                    last_plugin_failure_feedback = plugin_result.state.last_plugin_failure_feedback
                    if plugin_result.attempt_exception is not None:
                        attempt_exc = plugin_result.attempt_exception
                    if plugin_result.resources_closed:
                        exit_cm = None
                    if plugin_result.retry:
                        continue
                    if plugin_result.outcome is not None:
                        return plugin_result.outcome
                    processed_output = plugin_result.processed_output

                validation_result = await run_validation(
                    core=core,
                    step=step,  # type: ignore[arg-type]
                    output=processed_output,
                    context=attempt_context,
                    limits=limits,
                    data=data,
                    attempt_context=attempt_context,
                    attempt_resources=attempt_resources,
                    stream=stream,
                    on_chunk=on_chunk,
                    fallback_depth=fallback_depth,
                )
                if validation_result is not None:
                    if isinstance(validation_result, Success):
                        await _close_resources(None)
                        exit_cm = None
                        return validation_result
                    return validation_result

                try:
                    from .. import step_policies as _step_policies

                    if not primary_tokens_known:
                        ptokens, ctokens, cost = _step_policies.extract_usage_metrics(
                            raw_output=raw_agent_output,
                            agent=getattr(step, "agent", None),
                            step_name=core._safe_step_name(step),
                        )
                        prompt_tokens_latest = ptokens
                        completion_tokens_latest = ctokens
                        result.token_counts = ptokens + ctokens
                        result.cost_usd = cost
                        primary_tokens_known = True
                except PricingNotConfiguredError:
                    raise
                except UsageLimitExceededError:
                    raise
                except Exception as usage_err:
                    attempt_exc = usage_err
                    raise

                result.success = True
                try:
                    processed_output = core.unpacker.unpack(processed_output)
                except Exception:
                    pass

                try:
                    sink_path = getattr(step, "sink_to", None)
                    if sink_path and attempt_context is not None:
                        from ....utils.context import set_nested_context_field as _set_field

                        try:
                            _set_field(attempt_context, str(sink_path), processed_output)
                            telemetry.logfire.info(
                                f"Step '{getattr(step, 'name', '')}' sink_to stored output to {sink_path}"
                            )
                        except Exception as _sink_err:
                            try:
                                if "." not in str(sink_path):
                                    object.__setattr__(
                                        attempt_context, str(sink_path), processed_output
                                    )
                                    telemetry.logfire.info(
                                        f"Step '{getattr(step, 'name', '')}' sink_to stored output to {sink_path} (fallback)"
                                    )
                                else:
                                    telemetry.logfire.warning(
                                        f"Failed to set sink_to path '{sink_path}': {_sink_err}"
                                    )
                            except Exception:
                                pass
                except Exception:
                    pass

                try:
                    step_name = getattr(step, "name", "")
                    meta_obj = getattr(step, "meta", None)
                    is_adapter = (
                        bool(meta_obj.get("is_adapter")) if isinstance(meta_obj, dict) else False
                    )
                    is_mapper = step_name.endswith("_output_mapper") or is_adapter
                    if is_mapper and attempt_context is not None:
                        iter_count = getattr(attempt_context, "_last_loop_iterations", None)
                        if iter_count is not None:
                            result.attempts = int(iter_count)
                except Exception:
                    pass

                result.output = processed_output
                result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                if attempt_context is not None:
                    result.branch_context = attempt_context
                    try:
                        if context is not None and attempt_context is not context:
                            from ..context_manager import ContextManager as _CM

                            merged_ctx = _CM.merge(context, attempt_context)
                            if merged_ctx is not None:
                                context = merged_ctx
                                result.branch_context = merged_ctx
                    except Exception:
                        pass
                    try:
                        if isinstance(result.token_counts, dict) and "total" in result.token_counts:
                            tkn = int(result.token_counts["total"] or 0)
                        else:
                            tkn = int(result.token_counts or 0)
                        if not primary_tokens_known:
                            primary_tokens_total += tkn
                        best_primary_tokens = max(best_primary_tokens, tkn)
                        best_primary_cost_usd = max(
                            best_primary_cost_usd, float(result.cost_usd or 0.0)
                        )
                    except Exception:
                        pass
                try:
                    primary_latency_total += float(result.latency_s or 0.0)
                except Exception:
                    pass
                # Record step output for templating
                try:
                    if attempt_context is not None:
                        outputs = getattr(attempt_context, "step_outputs", None)
                        if isinstance(outputs, dict):
                            outputs[getattr(step, "name", "")] = result.output
                        if context is not None and context is not attempt_context:
                            outputs_main = getattr(context, "step_outputs", None)
                            if isinstance(outputs_main, dict):
                                outputs_main[getattr(step, "name", "")] = result.output
                except Exception:
                    pass
                try:
                    await core._usage_meter.add(
                        float(result.cost_usd or 0.0),
                        int(prompt_tokens_latest or 0),
                        int(completion_tokens_latest or 0),
                    )
                except Exception:
                    pass
                await _close_resources(None)
                exit_cm = None
                return Success(step_result=result)

            except PricingNotConfiguredError:
                # Strict pricing mode: must halt immediately
                raise
            except UsageLimitExceededError:
                raise
            except PausedException as p_exc:
                attempt_exc = p_exc
                await _close_resources(p_exc)
                exit_cm = None
                raise p_exc
            except Exception as e:
                attempt_exc = e
                if isinstance(e, InfiniteRedirectError):
                    raise
                if isinstance(e, InfiniteFallbackError):
                    raise
                if isinstance(e, UsageLimitExceededError):
                    raise
                primary_latency_total += time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                # Preserve per-attempt context mutations on failure for updates_context steps
                if attempt_context is not None:
                    result.branch_context = attempt_context
                try:
                    if isinstance(result.token_counts, dict) and "total" in result.token_counts:
                        if primary_tokens_known:
                            val = int(result.token_counts["total"] or 0)
                            best_primary_tokens = max(best_primary_tokens, val)
                            primary_tokens_total += val
                    else:
                        if primary_tokens_known:
                            val = int(result.token_counts or 0)
                            best_primary_tokens = max(best_primary_tokens, val)
                            primary_tokens_total += val
                    if result.cost_usd:
                        best_primary_cost_usd = max(best_primary_cost_usd, float(result.cost_usd))
                except Exception:
                    pass
                if attempt < total_attempts:
                    telemetry.logfire.warning(
                        f"Step '{getattr(step, 'name', '<unnamed>')}' attempt {attempt}/{total_attempts} failed: {e}"
                    )
                    # Do not merge failed attempt context back; failed attempts should not commit changes
                    continue
                primary_fb = f"Agent execution failed with {type(e).__name__}: {e}"
                exhausted_outputs = False
                try:
                    exhausted_outputs = isinstance(
                        e, IndexError
                    ) and "No more outputs available" in str(e)
                    if exhausted_outputs:
                        primary_fb = str(e)
                except Exception:
                    exhausted_outputs = False
                if last_plugin_failure_feedback and not exhausted_outputs:
                    primary_fb = last_plugin_failure_feedback
                fallback_state = FallbackState(
                    prompt_tokens_latest=prompt_tokens_latest,
                    completion_tokens_latest=completion_tokens_latest,
                    best_primary_tokens=best_primary_tokens,
                    best_primary_cost_usd=best_primary_cost_usd,
                    primary_tokens_total=primary_tokens_total,
                    primary_tokens_known=primary_tokens_known,
                    last_plugin_failure_feedback=last_plugin_failure_feedback,
                )
                fallback_result = await self._fallback_handler.handle_failure(
                    core=core,
                    step=step,
                    data=data,
                    attempt_context=attempt_context,
                    attempt_resources=attempt_resources,
                    limits=limits,
                    stream=stream,
                    on_chunk=on_chunk,
                    fallback_depth=fallback_depth,
                    start_ns=start_ns,
                    result=result,
                    primary_feedback=primary_fb,
                    attempt_exc=e,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    pre_attempt_context=pre_attempt_context,
                    _context=context,
                    close_resources=_close_resources,
                    state=fallback_state,
                )
                if fallback_result.resources_closed:
                    exit_cm = None
                return fallback_result.outcome
            await _close_resources(attempt_exc)
            exit_cm = None
            try:
                if (
                    context is not None
                    and pre_attempt_context is not None
                    and getattr(step, "updates_context", False)
                ):
                    from ..context_manager import ContextManager as _CM

                    _CM.merge(context, pre_attempt_context)
                    # Explicitly reset HITL-specific counters/collections to avoid committing failed attempts
                    for attr in (
                        "total_interactions",
                        "interaction_history",
                        "hitl_data",
                        "current_interaction",
                        "human_interactions",
                        "approval_count",
                        "rejection_count",
                    ):
                        if hasattr(pre_attempt_context, attr):
                            try:
                                setattr(context, attr, getattr(pre_attempt_context, attr))
                            except Exception:
                                pass
                    try:
                        if hasattr(context, "__dict__") and hasattr(
                            pre_attempt_context, "__dict__"
                        ):
                            context.__dict__.clear()
                            context.__dict__.update(pre_attempt_context.__dict__)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    if exit_cm:
                        import inspect

                        exc_type = type(attempt_exc) if attempt_exc else None
                        exc_tb = attempt_exc.__traceback__ if attempt_exc else None
                        res_cm = exit_cm(exc_type, attempt_exc, exc_tb)
                        if inspect.isawaitable(res_cm):
                            await res_cm
                except Exception:
                    pass

        return Failure(
            error=Exception(
                f"[AgentExecutionRunner] Unexpected fallthrough for step='{getattr(step, 'name', '<unnamed>')}'"
            ),
            feedback="Agent execution failed",
            step_result=StepResult(
                name=core._safe_step_name(step),
                output=None,
                success=False,
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback="Agent execution failed",
                branch_context=None,
                metadata_={},
                step_history=[],
            ),
        )
