from __future__ import annotations
# mypy: ignore-errors

from ._shared import (  # noqa: F401
    Any,
    Awaitable,
    Callable,
    Dict,
    ImportStep,
    InfiniteRedirectError,
    NonRetryableError,
    Optional,
    Paused,
    PausedException,
    Pipeline,
    PipelineResult,
    PricingNotConfiguredError,
    Protocol,
    Success,
    Failure,
    StepOutcome,
    StepResult,
    UsageLimits,
    UsageLimitExceededError,
    telemetry,
    to_outcome,
)

# --- Import Step Executor policy ---


class ImportStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: ImportStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Callable[[PipelineResult[Any], Optional[Any]], None],
    ) -> StepOutcome[StepResult]: ...


class DefaultImportStepExecutor:
    async def execute(
        self,
        core: Any,
        step: ImportStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Callable[[PipelineResult[Any], Optional[Any]], None],
    ) -> StepOutcome[StepResult]:
        from ..context_manager import ContextManager
        import json
        import copy

        def _looks_like_status_string(text: str) -> bool:
            try:
                if not isinstance(text, str):
                    return False
                s = text.strip()
                if not s:
                    return False
                # Short, emoji/prefix-driven status messages commonly used in logs
                prefixes = (
                    "✅",
                    "✔",
                    "ℹ",
                    "Info:",
                    "Status:",
                    "Ready",
                    "Done",
                    "OK",
                    "Definition ready",
                    "[OK]",
                    "[Info]",
                )
                if any(s.startswith(p) for p in prefixes) and len(s) <= 120:
                    return True
                # Single-line very short confirmations
                return (len(s) <= 40) and s.lower() in {"ok", "done", "ready", "success"}
            except Exception:
                return False

        # Build child context based on inherit_context and inherit_conversation flags
        sub_context = None
        if step.inherit_context:
            # Isolate to avoid poisoning parent on failure/retries
            sub_context = ContextManager.isolate(context)
            if sub_context is None and context is not None:
                try:
                    sub_context = type(context).model_validate(context.model_dump())
                except Exception:
                    try:
                        sub_context = copy.deepcopy(context)
                    except Exception:
                        sub_context = context
        else:
            if context is not None:
                try:
                    sub_context = type(context).model_construct()
                except Exception:
                    try:
                        sub_context = type(context)()
                    except Exception:
                        sub_context = None

        # Copy conversation fields when requested but not inheriting full context
        if (
            step.inherit_conversation
            and sub_context is not None
            and context is not None
            and not step.inherit_context
        ):
            for conv_field in ("hitl_history", "conversation_history"):
                try:
                    if hasattr(context, conv_field):
                        setattr(
                            sub_context, conv_field, copy.deepcopy(getattr(context, conv_field))
                        )
                except Exception:
                    pass

        # Project input into child run and compute the child's initial_input explicitly,
        # honoring explicit inputs over inherited conversation or parent data.
        # Precedence:
        #   1) sub_context.scratchpad[input_scratchpad_key] when present (explicit artifact)
        #   2) provided data argument (parent current_data)
        #   3) empty string fallback
        resolved_origin = "parent_data"
        sub_initial_input = data
        try:
            if sub_context is not None and hasattr(sub_context, "scratchpad"):
                sp = getattr(sub_context, "scratchpad")
                if isinstance(sp, dict):
                    key = step.input_scratchpad_key or "initial_input"
                    if key in sp and sp.get(key) is not None:
                        sub_initial_input = sp.get(key)
                        resolved_origin = f"scratchpad:{key}"
        except Exception:
            pass
        try:
            # scratchpad projection (deep merge for dicts)
            if step.input_to in ("scratchpad", "both") and sub_context is not None:
                sp = getattr(sub_context, "scratchpad", None)
                if isinstance(sp, dict):
                    if isinstance(data, dict):

                        def _deep_merge_dict(a: dict, b: dict) -> dict:
                            res = dict(a)
                            for k, v in b.items():
                                if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                                    res[k] = _deep_merge_dict(res[k], v)
                                else:
                                    res[k] = v
                            return res

                        merged = _deep_merge_dict(sp, copy.deepcopy(data))
                        try:
                            setattr(sub_context, "scratchpad", merged)
                        except Exception:
                            sp.update(copy.deepcopy(data))
                    else:
                        key = step.input_scratchpad_key or "initial_input"
                        sp[key] = data

            # initial_prompt projection and precedence for child's initial_input
            if step.input_to in ("initial_prompt", "both"):
                # Recompute init_text from the resolved explicit input (not blindly from `data`)
                init_text = (
                    json.dumps(sub_initial_input, default=str)
                    if isinstance(sub_initial_input, (dict, list))
                    else str(sub_initial_input)
                )
                if sub_context is not None:
                    try:
                        object.__setattr__(sub_context, "initial_prompt", init_text)
                    except Exception:
                        setattr(sub_context, "initial_prompt", init_text)
                # Enforce explicit input precedence: child's effective initial_input is the resolved one
                sub_initial_input = init_text
        except Exception:
            # Non-fatal: continue with best-effort routing
            pass

        # Lightweight diagnostics for import input routing
        try:
            preview = None
            try:
                preview = (
                    str(sub_initial_input)[:200]
                    if not isinstance(sub_initial_input, (dict, list))
                    else json.dumps(sub_initial_input, default=str)[:200]
                )
            except Exception:
                preview = str(type(sub_initial_input))
            telemetry.logfire.info(
                f"[ImportStep] initial_input_resolved origin={resolved_origin} preview={preview}"
            )
        except Exception:
            pass

        # Execute the child pipeline directly via core orchestration to preserve control-flow semantics
        try:
            pipeline_result: PipelineResult[Any] = await core._execute_pipeline_via_policies(
                step.pipeline,
                sub_initial_input,
                sub_context,
                resources,
                limits,
                context_setter,
            )
        except PausedException as e:
            # Preserve child (imported) context state on pause and proxy to parent when requested
            # Rationale: The child pipeline (e.g., a LoopStep with HITL) may update
            # conversation/hitl state inside its isolated context. Without merging
            # that state back to the parent's context before propagating the pause,
            # resuming will re-enter with stale state and cause repeated questions.
            try:
                if context is not None and sub_context is not None:
                    try:
                        # Prefer robust merge that preserves lists/history and dicts
                        from flujo.utils.context import safe_merge_context_updates as _safe_merge

                        _safe_merge(context, sub_context)
                    except Exception:
                        # Fallback to model-level merge when available
                        try:
                            merged_ctx = ContextManager.merge(context, sub_context)
                            if merged_ctx is not None:
                                context = merged_ctx
                        except Exception:
                            pass
                # Mark parent context as paused only when propagation is enabled
                propagate = bool(getattr(step, "propagate_hitl", True))
                if propagate:
                    if context is not None and hasattr(context, "scratchpad"):
                        try:
                            sp = getattr(context, "scratchpad")
                            if isinstance(sp, dict):
                                sp["status"] = "paused"
                                msg = getattr(e, "message", None)
                                sp["pause_message"] = msg if isinstance(msg, str) else str(e)
                        except Exception:
                            pass
                else:
                    # Ensure status remains running when not propagating
                    if context is not None and hasattr(context, "scratchpad"):
                        try:
                            sp = getattr(context, "scratchpad")
                            if isinstance(sp, dict):
                                if sp.get("status") == "paused":
                                    sp["status"] = "running"
                                sp.pop("pause_message", None)
                        except Exception:
                            pass
            except Exception:
                # Non-fatal: propagate pause regardless
                pass

            # Proxy child HITL to parent when requested
            if propagate:
                return Paused(message=str(e))
            # Legacy/opt-out: do not pause parent; return empty success result
            parent_sr = StepResult(
                name=step.name,
                success=True,
                output={},
                attempts=1,
                latency_s=0.0,
                token_counts=0,
                cost_usd=0.0,
                feedback=None,
                branch_context=context,
                metadata_={"hitl_propagation": "suppressed"},
                step_history=[],
            )
            return Success(step_result=parent_sr)
        except (
            UsageLimitExceededError,
            InfiniteRedirectError,
            NonRetryableError,
            PricingNotConfiguredError,
        ):
            # Re-raise control-flow/config exceptions per policy
            raise
        except Exception as e:
            return Failure(
                error=e,
                feedback=f"Failed to execute imported pipeline: {e}",
                step_result=StepResult(
                    name=step.name,
                    success=False,
                    output=None,
                    attempts=0,
                    latency_s=0.0,
                    token_counts=0,
                    cost_usd=0.0,
                    feedback=f"Failed to execute imported pipeline: {e}",
                    branch_context=context,
                    metadata_={},
                    step_history=[],
                ),
            )

        # Normalize successful child outcome
        inner_sr = None
        try:
            # Prefer the last step result from the child pipeline when available
            if getattr(pipeline_result, "step_history", None):
                inner_sr = pipeline_result.step_history[-1]
        except Exception:
            inner_sr = None

        # Parent-facing result; core will merge according to updates_context
        # Aggregate child latency across steps
        try:
            _total_child_latency = sum(
                float(getattr(sr, "latency_s", 0.0) or 0.0)
                for sr in (getattr(pipeline_result, "step_history", []) or [])
            )
        except Exception:
            _total_child_latency = float(getattr(inner_sr, "latency_s", 0.0) or 0.0)
        parent_sr = StepResult(
            name=step.name,
            success=True,
            output=None,
            attempts=(getattr(inner_sr, "attempts", 1) if inner_sr is not None else 1),
            latency_s=_total_child_latency,
            token_counts=int(getattr(pipeline_result, "total_tokens", 0) or 0),
            cost_usd=float(getattr(pipeline_result, "total_cost_usd", 0.0) or 0.0),
            feedback=None,
            branch_context=context,
            metadata_={},
            step_history=([inner_sr] if inner_sr is not None else []),
        )

        # Attach traceable metadata for diagnostics and tests
        try:
            if parent_sr.metadata_ is None:
                parent_sr.metadata_ = {}
            md = parent_sr.metadata_
            # Track where the child's input came from and a short preview
            md["import.initial_input_resolved"] = {
                "origin": resolved_origin,
                "type": type(sub_initial_input).__name__,
                "length": (
                    len(sub_initial_input)
                    if isinstance(sub_initial_input, (str, list, dict))
                    else None
                ),
                "preview": (
                    str(sub_initial_input)[:200]
                    if not isinstance(sub_initial_input, (dict, list))
                    else json.dumps(sub_initial_input, default=str)[:200]
                ),
            }
            # Heuristic validator warning for status-only strings when structured content is expected
            try:
                if step.input_to in ("initial_prompt", "both") and _looks_like_status_string(
                    sub_initial_input if isinstance(sub_initial_input, str) else ""
                ):
                    warn_msg = (
                        "ImportStep received a status-like string as initial input; "
                        "if the child expects structured content, route an explicit artifact "
                        "via scratchpad or ensure the correct payload is provided."
                    )
                    telemetry.logfire.warn(warn_msg)
                    md["import.initial_input_warning"] = warn_msg
            except Exception:
                pass
        except Exception:
            pass

        # Determine child's final context for default-merge behavior
        child_final_ctx = getattr(pipeline_result, "final_pipeline_context", sub_context)

        if inner_sr is not None and not getattr(inner_sr, "success", True):
            # Honor on_failure behavior for explicit child failure
            # Honor on_failure behavior
            mode = getattr(step, "on_failure", "abort")
            if mode == "skip":
                parent_sr = StepResult(
                    name=step.name,
                    success=True,
                    output=None,
                    attempts=getattr(inner_sr, "attempts", 1),
                    latency_s=getattr(inner_sr, "latency_s", 0.0),
                    token_counts=int(getattr(pipeline_result, "total_tokens", 0) or 0),
                    cost_usd=float(getattr(pipeline_result, "total_cost_usd", 0.0) or 0.0),
                    feedback=None,
                    branch_context=context,
                    metadata_={},
                    step_history=([inner_sr] if inner_sr is not None else []),
                )
                return Success(step_result=parent_sr)
            if mode == "continue_with_default":
                parent_sr = StepResult(
                    name=step.name,
                    success=True,
                    output={},
                    attempts=getattr(inner_sr, "attempts", 1),
                    latency_s=getattr(inner_sr, "latency_s", 0.0),
                    token_counts=int(getattr(pipeline_result, "total_tokens", 0) or 0),
                    cost_usd=float(getattr(pipeline_result, "total_cost_usd", 0.0) or 0.0),
                    feedback=None,
                    branch_context=context,
                    metadata_={},
                    step_history=([inner_sr] if inner_sr is not None else []),
                )
                return Success(step_result=parent_sr)
            # Default abort behavior: bubble child's failure
            # Mark the synthesized parent result as failed
            parent_sr.success = False
            parent_sr.feedback = getattr(inner_sr, "feedback", None)
            return Failure(
                error=Exception(getattr(inner_sr, "feedback", "child failed")),
                feedback=getattr(inner_sr, "feedback", None),
                step_result=parent_sr,
            )

        if getattr(step, "updates_context", False) and step.outputs:
            # Build a minimal context update dict using outputs mapping
            update_data: Dict[str, Any] = {}

            def _get_child(path: str) -> Any:
                parts = [p for p in path.split(".") if p]
                # Prefer the child's final context produced by the imported pipeline
                cur: Any = child_final_ctx
                for part in parts:
                    if cur is None:
                        return None
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    elif isinstance(cur, dict):
                        cur = cur.get(part)
                    else:
                        return None
                return cur

            def _assign_parent(path: str, value: Any) -> None:
                parts = [p for p in path.split(".") if p]
                if not parts:
                    return
                tgt = update_data
                for part in parts[:-1]:
                    if part not in tgt or not isinstance(tgt[part], dict):
                        tgt[part] = {}
                    tgt = tgt[part]
                tgt[parts[-1]] = value

            try:
                for mapping in step.outputs:
                    try:
                        parent_path = mapping.parent
                        child_val = _get_child(mapping.child)
                        # Skip missing child paths
                        if child_val is None:
                            continue
                        _assign_parent(parent_path, child_val)
                    except Exception:
                        continue
                parent_sr.output = update_data
            except Exception:
                parent_sr.output = inner_sr.output
        elif getattr(step, "updates_context", False) and step.outputs == []:
            # Explicit empty mapping provided: do not merge anything back
            parent_sr.output = None
        elif (
            getattr(step, "updates_context", False)
            and getattr(step, "outputs", None) is None
            and child_final_ctx is not None
        ):
            # No mapping provided: merge entire child context back deterministically
            try:
                parent_sr.output = PipelineResult(final_pipeline_context=child_final_ctx)
            except Exception:
                parent_sr.output = inner_sr.output
        else:
            parent_sr.output = getattr(inner_sr, "output", None)

        return Success(step_result=parent_sr)


# --- End Import Step Executor policy ---
