from __future__ import annotations
from flujo.type_definitions.common import JSONObject

import inspect

from ._shared import (  # noqa: F401
    Awaitable,
    Callable,
    Dict,
    AsyncMock,
    MagicMock,
    Mock,
    InfiniteFallbackError,
    InfiniteRedirectError,
    MissingAgentError,
    MockDetectionError,
    NonRetryableError,
    Optional,
    Paused,
    PausedException,
    Protocol,
    Quota,
    StepOutcome,
    StepResult,
    UsageEstimate,
    UsageLimitExceededError,
    UsageLimits,
    extract_usage_metrics,
    telemetry,
    time_perf_ns,
    time_perf_ns_to_seconds,
    to_outcome,
    _load_template_config,
    _normalize_plugin_feedback,
)
from ....domain.models import BaseModel as DomainBaseModel


async def run_agent_execution(
    executor: object,
    core: object,
    step: object,
    data: object,
    context: DomainBaseModel | None,
    resources: object | None,
    limits: UsageLimits | None,
    stream: bool,
    on_chunk: Callable[[object], Awaitable[None]] | None,
    cache_key: str | None,
    _fallback_depth: int = 0,
) -> StepOutcome[StepResult]:
    handler = getattr(core, "_agent_handler", None)
    execute_fn = getattr(handler, "execute", None)
    if not callable(execute_fn):
        raise TypeError("run_agent_execution requires core._agent_handler.execute(...)")
    try:
        outcome = await execute_fn(
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=_fallback_depth,
        )
    except PausedException as e:
        raise e
    if isinstance(outcome, StepOutcome):
        return outcome
    if isinstance(outcome, StepResult):
        return to_outcome(outcome)
    return to_outcome(
        StepResult(
            name=str(getattr(step, "name", "<unnamed>")),
            success=False,
            feedback=f"Unsupported agent handler outcome: {type(outcome).__name__}",
        )
    )
    if hasattr(step, "_mock_name"):
        mock_name = str(getattr(step, "_mock_name", ""))
        if "fallback_step" in mock_name and mock_name.count("fallback_step") > 1:
            raise InfiniteFallbackError(
                f"Infinite Mock fallback chain detected early: {mock_name[:100]}..."
            )
    from pydantic import BaseModel
    import time

    if getattr(step, "agent", None) is None:
        raise MissingAgentError(f"Step '{step.name}' has no agent configured")
    result = StepResult(
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

    def _unpack_agent_result(output: object) -> object:
        if isinstance(output, BaseModel):
            return output
        for attr in ("output", "content", "result", "data", "text", "message", "value"):
            if hasattr(output, attr):
                return getattr(output, attr)
        return output

    def _detect_mock_objects(obj: object) -> None:
        if isinstance(obj, (Mock, MagicMock, AsyncMock)):
            raise MockDetectionError(f"Step '{step.name}' returned a Mock object")

    overall_start_time = time.monotonic()
    last_processed_output: object | None = None
    last_exception: Optional[Exception] = None
    try:
        telemetry.logfire.info(
            f"[AgentPolicy] Enter execute for step='{getattr(step, 'name', '<unnamed>')}'"
        )
    except Exception:
        pass
    try:
        templ_spec = None
        if hasattr(step, "meta") and isinstance(step.meta, dict):
            templ_spec = step.meta.get("templated_input")
        if templ_spec is not None:
            from flujo.utils.prompting import AdvancedPromptFormatter
            from flujo.utils.template_vars import (
                get_steps_map_from_context,
                TemplateContextProxy,
                StepValueProxy,
            )
            from flujo.exceptions import TemplateResolutionError

            strict, log_resolution = _load_template_config()
            steps_map = get_steps_map_from_context(context)
            steps_wrapped: JSONObject = {
                k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                for k, v in steps_map.items()
            }
            fmt_context: JSONObject = {
                "context": TemplateContextProxy(context, steps=steps_wrapped),
                "previous_step": data,
                "steps": steps_wrapped,
            }
            try:
                if context and hasattr(context, "hitl_history") and context.hitl_history:
                    fmt_context["resume_input"] = context.hitl_history[-1].human_response
            except Exception:
                pass  # resume_input will be undefined if no HITL history
            if isinstance(templ_spec, str) and ("{{" in templ_spec and "}}" in templ_spec):
                formatter = AdvancedPromptFormatter(
                    templ_spec, strict=strict, log_resolution=log_resolution
                )
                data = formatter.format(**fmt_context)
            else:
                data = templ_spec
    except TemplateResolutionError as e:
        telemetry.logfire.error(
            f"[AgentStep] Template resolution failed in step '{step.name}': {e}"
        )
        raise
    except Exception as e:
        telemetry.logfire.debug(f"[AgentStep] Non-fatal template error in step '{step.name}': {e}")
        pass
    try:
        estimate = None
        strategy_name = "heuristic"
        estimator = getattr(core, "_usage_estimator", None)
        if estimator is not None:
            try:
                estimate = estimator.estimate(step, data, context)
                strategy_name = "injected"
            except Exception:
                estimate = None
        selector = getattr(core, "_estimator_factory", None)
        if estimate is None and selector is not None:
            try:
                est = selector.select(step)
                estimate = est.estimate(step, data, context)
                try:
                    cname = type(est).__name__.lower()
                    if "learnable" in cname:
                        strategy_name = "learnable"
                    elif "minimal" in cname:
                        strategy_name = "adapter_minimal"
                    else:
                        strategy_name = "heuristic"
                except Exception:
                    strategy_name = "heuristic"
            except Exception:
                estimate = None
        if estimate is None:
            estimate = executor._estimate_usage(step, data, context, core=core)
            strategy_name = "heuristic"
        try:
            telemetry.logfire.debug(
                f"[cost.estimator.selected] step='{getattr(step, 'name', '<unnamed>')}', strategy={strategy_name}, cost_usd={getattr(estimate, 'cost_usd', None)}, tokens={getattr(estimate, 'tokens', None)}"
            )
            try:
                from flujo.application.core.optimized_telemetry import increment_counter as _inc

                _inc("estimator.usage", 1, tags={"strategy": strategy_name})
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        estimate = executor._estimate_usage(step, data, context, core=core)

    def _get_quota() -> Optional[Quota]:
        try:
            if hasattr(core, "_get_current_quota"):
                return core._get_current_quota()
        except Exception:
            return None
        return None

    def _set_quota(quota: Optional[Quota]) -> Optional[object]:
        try:
            if hasattr(core, "_quota_manager"):
                return core._set_current_quota(quota)
        except Exception:
            return None
        return None

    def _reset_quota(token: Optional[object]) -> None:
        try:
            if token is not None and hasattr(core, "_reset_current_quota"):
                core._reset_current_quota(token)
        except Exception:
            pass

    current_quota: Optional[Quota] = _get_quota()
    if current_quota is not None:
        try:
            rem_cost, rem_tokens = current_quota.get_remaining()
            telemetry.logfire.debug(
                f"[quota.reserve.attempt] step='{getattr(step, 'name', '<unnamed>')}', est_cost={getattr(estimate, 'cost_usd', None)}, est_tokens={getattr(estimate, 'tokens', None)}, rem_cost={rem_cost}, rem_tokens={rem_tokens}"
            )
        except Exception:
            pass
        if not current_quota.reserve(estimate):
            try:
                from ..usage_messages import format_reservation_denial
            except Exception:
                from flujo.application.core.usage_messages import format_reservation_denial
            denial = format_reservation_denial(estimate, limits, remaining=(rem_cost, rem_tokens))
            failure_msg = denial.human
            try:
                telemetry.logfire.warning(
                    f"[quota.reserve.denied] step='{getattr(step, 'name', '<unnamed>')}', code={denial.code}, msg='{failure_msg}'"
                )
                try:
                    from flujo.application.core.optimized_telemetry import (
                        increment_counter as _inc,
                    )

                    _inc("quota.denials.total", 1, tags={"code": denial.code})
                except Exception:
                    pass
            except Exception:
                pass
            raise UsageLimitExceededError(failure_msg)

    def _ret(sr: StepResult) -> StepOutcome[StepResult]:
        return to_outcome(sr)

    retries_config = getattr(getattr(step, "config", None), "max_retries", 1)
    try:
        if hasattr(retries_config, "_mock_name") or isinstance(
            retries_config, (Mock, MagicMock, AsyncMock)
        ):
            retries_config = 2
        else:
            retries_config = int(retries_config) if retries_config is not None else 1
    except Exception:
        retries_config = 1
    total_attempts = max(1, 1 + max(0, retries_config))
    try:
        telemetry.logfire.info(
            f"[AgentPolicy] retries_config={retries_config} total_attempts={total_attempts} step='{getattr(step, 'name', '<unnamed>')}'"
        )
    except Exception:
        pass
    if stream:
        total_attempts = 1
    pre_attempt_context = None
    if context is not None and total_attempts > 1:
        from flujo.application.core.context_manager import ContextManager

        pre_attempt_context = ContextManager.isolate(context)
    for attempt in range(1, total_attempts + 1):
        result.attempts = attempt
        try:
            telemetry.logfire.info(
                f"[AgentPolicy] attempt {attempt}/{total_attempts} for step='{getattr(step, 'name', '<unnamed>')}'"
            )
        except Exception:
            pass
        if total_attempts > 1 and pre_attempt_context is not None:
            telemetry.logfire.debug(
                f"[AgentStep] Creating isolated context for attempt {attempt} (total_attempts={total_attempts})"
            )
            attempt_context = ContextManager.isolate(pre_attempt_context)
            telemetry.logfire.debug(
                f"[AgentStep] Isolated context for attempt {attempt}, original context preserved"
            )
            if (
                attempt_context is not None
                and pre_attempt_context is not None
                and hasattr(attempt_context, "branch_count")
                and hasattr(pre_attempt_context, "branch_count")
            ):
                telemetry.logfire.debug(
                    f"[AgentStep] Attempt {attempt}: attempt_context.branch_count={getattr(attempt_context, 'branch_count', 'N/A')}, pre_attempt_context.branch_count={getattr(pre_attempt_context, 'branch_count', 'N/A')}"
                )
        else:
            attempt_context = context
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
        attempt_exc: BaseException | None = None
        try:
            start_ns = time_perf_ns()
            try:
                from flujo.tracing.manager import get_active_trace_manager as _get_tm

                tm = _get_tm()
                rp_cfg: JSONObject = {}
                try:
                    meta_obj = getattr(step, "meta", {}) or {}
                    if isinstance(meta_obj, dict):
                        pmeta = meta_obj.get("processing", {}) or {}
                        rp_cfg = pmeta.get("reasoning_precheck", {}) or {}
                        if not isinstance(rp_cfg, dict):
                            rp_cfg = {}
                except Exception:
                    rp_cfg = {}
                if rp_cfg.get("enabled"):
                    missing: list[str] = []
                    req_keys = rp_cfg.get("required_context_keys") or []
                    if isinstance(req_keys, list):
                        for key in req_keys:
                            try:
                                has = (
                                    hasattr(attempt_context, key)
                                    and getattr(attempt_context, key) is not None
                                )
                            except Exception:
                                has = False
                            if not has:
                                missing.append(str(key))
                    if missing:
                        if tm is not None:
                            tm.add_event(
                                "aros.reasoning.precheck.fail",
                                {"reason": "missing_context_keys", "keys": missing},
                            )
                    else:
                        if tm is not None:
                            tm.add_event(
                                "aros.reasoning.precheck.pass", {"reason": "local_checklist"}
                            )
                        try:
                            inject_mode = str(rp_cfg.get("inject_feedback", "")).strip().lower()
                            guidance_text = None
                        except Exception:
                            inject_mode = ""
                            guidance_text = None
                        try:
                            validator_agent = rp_cfg.get("validator_agent") or rp_cfg.get("agent")
                            if validator_agent is not None:
                                delimiters = rp_cfg.get("delimiters") or [
                                    "<thinking>",
                                    "</thinking>",
                                ]
                                goal_key = rp_cfg.get("goal_context_key") or "initial_input"
                                score_threshold = rp_cfg.get("score_threshold")

                                def _extract_plan(
                                    text: str, delims: list[str] | tuple[str, str]
                                ) -> Optional[str]:
                                    try:
                                        if isinstance(delims, (list, tuple)) and len(delims) >= 2:
                                            d0, d1 = str(delims[0]), str(delims[1])
                                        else:
                                            d0, d1 = "<thinking>", "</thinking>"
                                        s = text.find(d0)
                                        e = text.find(d1, s + len(d0)) if s >= 0 else -1
                                        if s >= 0 and e > s:
                                            return text[s + len(d0) : e].strip()
                                    except Exception:
                                        return None
                                    return None

                                plan_text = _extract_plan(
                                    data if isinstance(data, str) else "",
                                    delimiters,
                                )
                                try:
                                    goal_val = getattr(attempt_context, goal_key, None)
                                except Exception:
                                    goal_val = None
                                payload = {"plan": plan_text, "goal": goal_val}
                                if not plan_text:
                                    if tm is not None:
                                        tm.add_event(
                                            "aros.reasoning.precheck.skipped",
                                            {"reason": "no_plan"},
                                        )
                                else:
                                    try:
                                        max_tok = None
                                        try:
                                            if rp_cfg.get("max_tokens") is not None:
                                                max_tok = int(rp_cfg.get("max_tokens"))
                                        except Exception:
                                            max_tok = None
                                        if hasattr(validator_agent, "run"):
                                            try:
                                                if max_tok is not None:
                                                    vres = await validator_agent.run(
                                                        payload, max_tokens=max_tok
                                                    )
                                                else:
                                                    vres = await validator_agent.run(payload)
                                            except Exception:
                                                vres = None
                                        else:
                                            vopts = {}
                                            if max_tok is not None:
                                                vopts["max_tokens"] = max_tok
                                            vres = await core._agent_runner.run(
                                                agent=validator_agent,
                                                payload=payload,
                                                context=attempt_context,
                                                resources=attempt_resources,
                                                options=vopts,
                                                stream=False,
                                                on_chunk=None,
                                            )
                                    except Exception:
                                        vres = None
                                    verdict = None
                                    vscore = None
                                    vdict = {}
                                    if vres is not None:
                                        try:
                                            if hasattr(vres, "model_dump"):
                                                vdict = vres.model_dump()
                                            elif isinstance(vres, dict):
                                                vdict = vres
                                            else:
                                                vdict = getattr(vres, "__dict__", {}) or {}
                                        except Exception:
                                            vdict = {}
                                    for k in ("is_correct", "is_valid", "ok"):
                                        if k in vdict:
                                            verdict = bool(vdict[k])
                                            break
                                    if "score" in vdict:
                                        try:
                                            vscore = float(vdict["score"])  # best-effort parse
                                        except Exception:
                                            vscore = None
                                    if vscore is not None and isinstance(
                                        score_threshold, (int, float)
                                    ):
                                        verdict = verdict if verdict is not None else True
                                        if vscore < float(score_threshold):
                                            verdict = False
                                    if tm is not None:
                                        tm.add_event(
                                            "reasoning.validation",
                                            {
                                                "result": "pass" if verdict else "fail",
                                                "score": vscore,
                                            },
                                        )
                                    guidance_text = (
                                        vdict.get("feedback") if isinstance(vdict, dict) else None
                                    )
                        except Exception:
                            pass
                        try:
                            if inject_mode == "prepend" and guidance_text and isinstance(data, str):
                                prefix = str(rp_cfg.get("retry_guidance_prefix", "Guidance: "))
                                data = f"{prefix}{guidance_text}\n\n{data}"
                            elif (
                                inject_mode == "context_key"
                                and guidance_text
                                and attempt_context is not None
                            ):
                                ckey = str(
                                    rp_cfg.get("context_feedback_key", "_aros_retry_guidance")
                                )
                                try:
                                    setattr(attempt_context, ckey, guidance_text)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            consensus_agent = rp_cfg.get("consensus_agent")
                            raw_samples = int(rp_cfg.get("consensus_samples", 0) or 0)
                            samples = min(max(raw_samples, 0), 5)
                            threshold = float(rp_cfg.get("consensus_threshold", 0.7) or 0.7)
                            if consensus_agent is not None and samples >= 2:
                                plans: list[str] = []
                                try:
                                    goal_val = getattr(attempt_context, goal_key, None)
                                except Exception:
                                    goal_val = None
                                for _ in range(samples):
                                    try:
                                        quota_token = None
                                        try:
                                            cq = _get_quota()
                                            if cq is not None:
                                                subq = cq.split(1)[0]
                                                quota_token = _set_quota(subq)
                                        except Exception:
                                            quota_token = None
                                        try:
                                            cres = await core._agent_runner.run(
                                                agent=consensus_agent,
                                                payload={"goal": goal_val},
                                                context=attempt_context,
                                                resources=attempt_resources,
                                                options={},
                                                stream=False,
                                                on_chunk=None,
                                            )
                                        finally:
                                            _reset_quota(quota_token)
                                        txt = None
                                        if isinstance(cres, dict):
                                            txt = cres.get("plan") or cres.get("output") or None
                                        if txt is None:
                                            txt = str(getattr(cres, "output", cres))
                                        if isinstance(txt, str) and txt.strip():
                                            plans.append(txt.strip())
                                    except Exception:
                                        continue

                                def _jaccard(a: str, b: str) -> float:
                                    ta = set(a.lower().split())
                                    tb = set(b.lower().split())
                                    if not ta and not tb:
                                        return 1.0
                                    inter = len(ta & tb)
                                    uni = len(ta | tb) or 1
                                    return inter / uni

                                score = None
                                if len(plans) >= 2:
                                    from itertools import combinations

                                    vals = [_jaccard(x, y) for x, y in combinations(plans, 2)]
                                    if vals:
                                        score = sum(vals) / len(vals)
                                if score is not None and tm is not None:
                                    tm.add_event(
                                        "reasoning.validation",
                                        {
                                            "result": "pass" if score >= threshold else "fail",
                                            "score": score,
                                        },
                                    )
                        except Exception:
                            pass
            except Exception:
                pass
            processed_data = data
            if hasattr(step, "processors") and getattr(step, "processors", None):
                processed_data = await core._processor_pipeline.apply_prompt(
                    step.processors, data, context=attempt_context
                )
            options: JSONObject = {}
            cfg = getattr(step, "config", None)
            if cfg:
                if getattr(cfg, "temperature", None) is not None:
                    options["temperature"] = cfg.temperature
                if getattr(cfg, "top_k", None) is not None:
                    options["top_k"] = cfg.top_k
                if getattr(cfg, "top_p", None) is not None:
                    options["top_p"] = cfg.top_p
            try:
                from flujo.infra.config_manager import get_aros_config as _get_aros
                from flujo.tracing.manager import get_active_trace_manager as _get_tm

                aros_cfg = _get_aros()
                tm = _get_tm()
                pmeta = {}
                try:
                    meta_obj = getattr(step, "meta", {}) or {}
                    if isinstance(meta_obj, dict):
                        pmeta = meta_obj.get("processing", {}) or {}
                        if not isinstance(pmeta, dict):
                            pmeta = {}
                except Exception:
                    pmeta = {}
                so_mode = (
                    str(pmeta.get("structured_output", aros_cfg.structured_output_default))
                    .strip()
                    .lower()
                )
                explicit = "structured_output" in pmeta
                provider = None
                try:
                    _mid = getattr(step.agent, "model_id", None)
                    if isinstance(_mid, str) and ":" in _mid:
                        provider = _mid.split(":", 1)[0].lower()
                except Exception:
                    provider = None
                if explicit and so_mode in {"auto", "openai_json"}:
                    try:
                        wrapper = getattr(step, "agent", None)
                        schema_obj = (
                            pmeta.get("schema") if isinstance(pmeta.get("schema"), dict) else None
                        )
                        if wrapper is not None and hasattr(wrapper, "enable_structured_output"):
                            name = str(getattr(step, "name", "step_output")) or "step_output"
                            wrapper.enable_structured_output(json_schema=schema_obj, name=name)
                            if tm is not None:
                                try:
                                    import json as _json
                                    import hashlib as _hash

                                    sh = None
                                    if isinstance(schema_obj, dict):
                                        s = _json.dumps(
                                            schema_obj, sort_keys=True, separators=(",", ":")
                                        ).encode()
                                        sh = _hash.sha256(s).hexdigest()
                                except Exception:
                                    sh = None
                                tm.add_event(
                                    "grammar.applied",
                                    {"mode": "openai_json", "schema_hash": sh},
                                )
                    except Exception:
                        pass
                elif explicit and so_mode in {"outlines", "xgrammar"}:
                    try:
                        schema_obj = (
                            pmeta.get("schema") if isinstance(pmeta.get("schema"), dict) else None
                        )
                        if schema_obj is not None:
                            try:
                                if so_mode == "outlines":
                                    from flujo.grammars.adapters import (
                                        compile_outlines_regex as _compile,
                                    )
                                else:
                                    from flujo.grammars.adapters import (
                                        compile_xgrammar as _compile,
                                    )
                                pat = _compile(schema_obj)
                                options.setdefault(
                                    "structured_grammar", {"mode": so_mode, "pattern": pat}
                                )
                            except Exception:
                                pass
                        if tm is not None:
                            try:
                                import json as _json
                                import hashlib as _hash

                                sh = None
                                if isinstance(schema_obj, dict):
                                    s = _json.dumps(
                                        schema_obj, sort_keys=True, separators=(",", ":")
                                    ).encode()
                                    sh = _hash.sha256(s).hexdigest()
                            except Exception:
                                sh = None
                            tm.add_event("grammar.applied", {"mode": so_mode, "schema_hash": sh})
                    except Exception:
                        pass
                elif explicit:
                    if tm is not None:
                        reason = (
                            "unsupported_provider" if provider not in {"openai"} else "disabled"
                        )
                        tm.add_event("aros.soe.skipped", {"reason": reason, "mode": so_mode})
            except Exception:
                pass
            try:
                agent_output = await core._agent_runner.run(
                    agent=step.agent,
                    payload=processed_data,
                    context=attempt_context,
                    resources=attempt_resources,
                    options=options,
                    stream=stream,
                    on_chunk=on_chunk,
                )
                try:
                    telemetry.logfire.info(
                        f"[AgentPolicy] agent_output acquired (method-local) type={type(agent_output).__name__}"
                    )
                except Exception:
                    pass
            except PausedException:
                raise
            if isinstance(agent_output, (Mock, MagicMock, AsyncMock)):
                raise MockDetectionError(f"Step '{step.name}' returned a Mock object")
            _detect_mock_objects(agent_output)
            try:
                prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                    raw_output=agent_output, agent=step.agent, step_name=step.name
                )
                result.cost_usd = cost_usd
                result.token_counts = prompt_tokens + completion_tokens
                try:
                    telemetry.logfire.info(
                        f"[AgentPolicy] usage extracted (method-local) tokens={result.token_counts} cost={result.cost_usd}"
                    )
                except Exception:
                    pass
            except Exception as e_usage:
                result.success = False
                result.feedback = str(e_usage)
                result.output = None
                result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                try:
                    if current_quota is not None:
                        current_quota.reclaim(
                            estimate,
                            UsageEstimate(
                                cost_usd=result.cost_usd or 0.0, tokens=result.token_counts or 0
                            ),
                        )
                except Exception:
                    pass
                try:
                    from flujo.tracing.manager import get_active_trace_manager as _get_tm

                    tm = _get_tm()
                    pmeta = {}
                    try:
                        meta_obj = getattr(step, "meta", {}) or {}
                        if isinstance(meta_obj, dict):
                            pmeta = meta_obj.get("processing", {}) or {}
                            if not isinstance(pmeta, dict):
                                pmeta = {}
                    except Exception:
                        pmeta = {}
                    so_mode = str(pmeta.get("structured_output", "off")).strip().lower()
                    schema_obj = (
                        pmeta.get("schema") if isinstance(pmeta.get("schema"), dict) else None
                    )
                    if tm is not None and so_mode not in {"", "off", "none", "false"}:
                        try:
                            import json as _json
                            import hashlib as _hash

                            sh = None
                            if isinstance(schema_obj, dict):
                                s = _json.dumps(
                                    schema_obj, sort_keys=True, separators=(",", ":")
                                ).encode()
                                sh = _hash.sha256(s).hexdigest()
                        except Exception:
                            sh = None
                        tm.add_event("grammar.applied", {"mode": so_mode, "schema_hash": sh})
                except Exception:
                    pass
                try:
                    telemetry.logfire.debug(
                        f"[quota.reconcile] step='{getattr(step, 'name', '<unnamed>')}', actual_cost={result.cost_usd}, actual_tokens={result.token_counts}, est_cost={getattr(estimate, 'cost_usd', None)}, est_tokens={getattr(estimate, 'tokens', None)}"
                    )
                    try:
                        from flujo.application.core.optimized_telemetry import (
                            increment_counter as _inc,
                        )

                        est_cost = float(getattr(estimate, "cost_usd", 0.0) or 0.0)
                        est_tok = int(getattr(estimate, "tokens", 0) or 0)
                        act_cost = float(result.cost_usd or 0.0)
                        act_tok = int(result.token_counts or 0)
                        d_cost = abs(act_cost - est_cost)
                        d_tok = abs(act_tok - est_tok)

                        def _bucket(v: float) -> str:
                            if v == 0:
                                return "0"
                            if v <= 1:
                                return "<=1"
                            if v <= 10:
                                return "<=10"
                            if v <= 100:
                                return "<=100"
                            return ">100"

                        _inc(
                            "estimation.variance.count",
                            1,
                            tags={"type": "cost", "bucket": _bucket(d_cost)},
                        )
                        _inc(
                            "estimation.variance.count",
                            1,
                            tags={"type": "tokens", "bucket": _bucket(float(d_tok))},
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
                return to_outcome(result)
            processed_output = agent_output
            last_processed_output = processed_output
            if hasattr(step, "processors") and step.processors:
                try:
                    processed_output = await core._processor_pipeline.apply_output(
                        step.processors, processed_output, context=attempt_context
                    )
                    last_processed_output = processed_output
                    try:
                        telemetry.logfire.info(
                            f"[AgentPolicy] output processors applied (method-local) for step='{getattr(step, 'name', '<unnamed>')}'"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    result.success = False
                    result.feedback = f"Processor failed: {str(e)}"
                    result.output = processed_output
                    result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                    telemetry.logfire.error(f"Step '{step.name}' processor failed: {e}")
                    return to_outcome(result)
                validation_result = await core._validation_orchestrator.validate(
                    core=core,
                    step=step,
                    output=processed_output,
                    context=attempt_context,
                    limits=limits,
                    data=data,
                    attempt_context=attempt_context,
                    attempt_resources=attempt_resources,
                    stream=stream,
                    on_chunk=on_chunk,
                    fallback_depth=_fallback_depth,
                )
                if validation_result is not None:
                    return (
                        to_outcome(validation_result)
                        if isinstance(validation_result, StepResult)
                        else validation_result
                    )
                try:
                    if hasattr(step, "plugins") and step.plugins:
                        timeout_s = None
                        try:
                            cfg = getattr(step, "config", None)
                            if cfg is not None and getattr(cfg, "timeout_s", None) is not None:
                                timeout_s = float(cfg.timeout_s)
                        except Exception:
                            timeout_s = None
                        processed_output = await core.plugin_redirector.run(
                            initial=processed_output,
                            step=step,
                            data=data,
                            context=context,
                            resources=attempt_resources,
                            timeout_s=timeout_s,
                        )
                        if isinstance(processed_output, dict) and "output" in processed_output:
                            processed_output = processed_output["output"]
                except Exception as e:
                    if isinstance(e, InfiniteRedirectError):
                        raise
                    result.success = False
                    result.feedback = f"Plugin failed: {str(e)}"
                    result.output = processed_output
                    result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                    telemetry.logfire.error(f"Step '{step.name}' plugin failed: {e}")
                    return to_outcome(result)
                result.output = _unpack_agent_result(processed_output)
                _detect_mock_objects(result.output)
                result.success = True
                result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_ns)
                result.feedback = None
                try:
                    telemetry.logfire.info(
                        f"[AgentPolicy] Success return for step='{getattr(step, 'name', '<unnamed>')}'"
                    )
                except Exception:
                    pass
                    if (
                        total_attempts > 1
                        and context is not None
                        and attempt_context is not None
                        and attempt_context is not context
                    ):
                        from flujo.application.core.context_manager import ContextManager

                        ContextManager.merge(context, attempt_context)
                        telemetry.logfire.debug(
                            f"Merged successful agent step attempt {attempt} context back to main context"
                        )
                    result.branch_context = context
                    try:
                        if context is not None:
                            outputs = getattr(context, "step_outputs", None)
                            if isinstance(outputs, dict):
                                outputs[getattr(step, "name", "")] = result.output
                    except Exception:
                        pass
                    try:
                        step_name = getattr(step, "name", "")
                        is_adapter = False
                        try:
                            is_adapter = isinstance(
                                getattr(step, "meta", None), dict
                            ) and step.meta.get("is_adapter")
                        except Exception:
                            is_adapter = False
                        if (
                            is_adapter or str(step_name).endswith("_output_mapper")
                        ) and context is not None:
                            if hasattr(context, "_last_loop_iterations"):
                                try:
                                    result.attempts = int(getattr(context, "_last_loop_iterations"))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if cache_key and getattr(core, "_enable_cache", False):
                        try:
                            await core._cache_backend.put(cache_key, result, ttl_s=3600)
                        except Exception:
                            pass
                    return to_outcome(result)
            try:
                pass
            except Exception as e:
                last_exception = e
                if isinstance(
                    e,
                    (
                        PausedException,
                        InfiniteFallbackError,
                        InfiniteRedirectError,
                        UsageLimitExceededError,
                        NonRetryableError,
                    ),
                ):
                    telemetry.logfire.info(
                        f"Step '{step.name}' encountered a non-retryable exception: {type(e).__name__} - re-raising"
                    )
                    raise e
                if attempt < total_attempts:
                    telemetry.logfire.warning(
                        f"Step '{step.name}' agent execution attempt {attempt} failed: {e}"
                    )
                    continue

                def _format_failure_feedback(err: Exception) -> str:
                    if isinstance(err, ValueError) and str(err).startswith("Plugin"):
                        msg = str(err)
                        return f"Plugin execution failed after max retries: {msg}"
                    return f"Agent execution failed with {type(err).__name__}: {str(err)}"

                primary_fb = _format_failure_feedback(e)
                fb_candidate = getattr(step, "fallback_step", None)
                try:
                    if hasattr(fb_candidate, "_mock_name") and not hasattr(fb_candidate, "agent"):
                        fb_candidate = None
                except Exception:
                    pass
                if fb_candidate is not None:
                    if hasattr(step.fallback_step, "_mock_name"):
                        mock_name = str(getattr(step.fallback_step, "_mock_name", ""))
                        if "fallback_step.fallback_step" in mock_name:
                            raise InfiniteFallbackError(
                                f"Infinite Mock fallback chain detected: {mock_name[:100]}..."
                            )
                    try:
                        fb_res = await core.execute(
                            step=fb_candidate,
                            data=data,
                            context=attempt_context,
                            resources=attempt_resources,
                            limits=limits,
                            stream=stream,
                            on_chunk=on_chunk,
                            cache_key=None,
                            _fallback_depth=_fallback_depth + 1,
                        )
                        result.cost_usd = (result.cost_usd or 0.0) + (fb_res.cost_usd or 0.0)
                        result.token_counts = (result.token_counts or 0) + (
                            fb_res.token_counts or 0
                        )
                        result.metadata_["fallback_triggered"] = True
                        result.metadata_["original_error"] = primary_fb
                        telemetry.logfire.error(
                            f"Step '{step.name}' agent failed after {result.attempts} attempts"
                        )
                        if fb_res.success:
                            fb_res.metadata_ = {**(fb_res.metadata_ or {}), **result.metadata_}
                            preserve_diagnostics = (
                                hasattr(step, "config")
                                and step.config is not None
                                and hasattr(step.config, "preserve_fallback_diagnostics")
                                and step.config.preserve_fallback_diagnostics is True
                            )
                            fb_res.feedback = (
                                f"Primary agent failed: {primary_fb}"
                                if preserve_diagnostics
                                else (fb_res.feedback or "Primary agent failed")
                            )
                            return to_outcome(fb_res)
                        else:
                            result.success = False
                            try:
                                norm_primary = _normalize_plugin_feedback(primary_fb or "")
                            except Exception:
                                norm_primary = primary_fb
                            try:
                                norm_fb = _normalize_plugin_feedback(fb_res.feedback or "")
                            except Exception:
                                norm_fb = fb_res.feedback
                            result.feedback = (
                                f"Original error: {norm_primary}; Fallback error: {norm_fb}"
                            )
                            result.output = None
                            result.latency_s = time.monotonic() - overall_start_time
                            return to_outcome(result)
                    except Exception as fb_e:
                        try:
                            from flujo.exceptions import UsageLimitExceededError as _ULE

                            if isinstance(fb_e, _ULE) and getattr(fb_e, "result", None) is not None:
                                res = fb_e.result
                                try:
                                    result.cost_usd = (result.cost_usd or 0.0) + float(
                                        getattr(res, "total_cost_usd", 0.0)
                                    )
                                except Exception:
                                    pass
                                try:
                                    result.token_counts = (result.token_counts or 0) + int(
                                        getattr(res, "total_tokens", 0)
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        result.success = False
                        try:
                            norm_primary = _normalize_plugin_feedback(primary_fb or "")
                        except Exception:
                            norm_primary = primary_fb
                        result.feedback = (
                            f"Original error: {norm_primary}; Fallback execution failed: {fb_e}"
                        )
                        result.output = None
                        result.latency_s = time.monotonic() - overall_start_time
                        telemetry.logfire.error(
                            f"Step '{step.name}' fallback execution raised: {fb_e}"
                        )
                        return to_outcome(result)
                result.success = False
                result.feedback = primary_fb
                result.output = None
                result.latency_s = time.monotonic() - overall_start_time
                telemetry.logfire.error(
                    f"Step '{step.name}' agent failed after {result.attempts} attempts"
                )
                return to_outcome(result)
        except BaseException as exc:
            attempt_exc = exc
            raise
        finally:
            if exit_cm is not None:
                try:
                    exc_for_exit = attempt_exc
                    try:
                        if exc_for_exit is None and result.success is False:
                            exc_for_exit = RuntimeError("step attempt failed")
                    except Exception:
                        pass
                    exit_result = exit_cm(
                        type(exc_for_exit) if exc_for_exit is not None else None,
                        exc_for_exit,
                        getattr(exc_for_exit, "__traceback__", None)
                        if exc_for_exit is not None
                        else None,
                    )
                    if inspect.isawaitable(exit_result):
                        await exit_result
                except Exception:
                    pass
    if last_exception is None:
        try:
            extract_usage_metrics(raw_output=None, agent=step.agent, step_name=step.name)
        except Exception as e_usage_fallback:
            result.success = False
            result.feedback = str(e_usage_fallback)
            result.latency_s = 0.0
            try:
                telemetry.logfire.error(
                    f"[AgentPolicy] Surfacing late usage/pricing error for step='{getattr(step, 'name', '<unnamed>')}'"
                )
            except Exception:
                pass
            return to_outcome(result)
    if last_exception is None and last_processed_output is not None:
        try:
            result.output = _unpack_agent_result(last_processed_output)
            _detect_mock_objects(result.output)
            result.success = True
            result.feedback = None
            return to_outcome(result)
        except Exception:
            pass
    result.success = False
    result.feedback = (
        str(last_exception) if last_exception is not None else "Unexpected execution path"
    )
    result.latency_s = 0.0
    try:
        telemetry.logfire.error(
            f"[AgentPolicy] Unexpected fallthrough for step='{getattr(step, 'name', '<unnamed>')}'"
        )
    except Exception:
        pass
    if current_quota is not None:
        try:
            actual = UsageEstimate(cost_usd=result.cost_usd or 0.0, tokens=result.token_counts or 0)
            current_quota.reclaim(estimate, actual)
        except Exception:
            pass
    return to_outcome(result)
