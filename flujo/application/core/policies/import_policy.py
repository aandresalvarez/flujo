from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Type, TypeGuard, cast
from collections.abc import MutableMapping
from flujo.domain.models import ImportArtifacts

from ._shared import (
    Any,
    ImportStep,
    InfiniteRedirectError,
    NonRetryableError,
    PipelineAbortSignal,
    Paused,
    PausedException,
    PipelineResult,
    PricingNotConfiguredError,
    Protocol,
    Success,
    Failure,
    StepOutcome,
    StepResult,
    UsageLimitExceededError,
    telemetry,
)
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame

# --- Import Step Executor policy ---


class ImportStepExecutor(Protocol):
    async def execute(self, core: Any, frame: ExecutionFrame[Any]) -> StepOutcome[StepResult]: ...


class DefaultImportStepExecutor(StepPolicy[ImportStep]):
    @property
    def handles_type(self) -> Type[ImportStep]:
        return ImportStep

    async def execute(self, core: Any, frame: ExecutionFrame[Any]) -> StepOutcome[StepResult]:
        step = cast(ImportStep, frame.step)
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        context_setter = frame.context_setter
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

        # Seed child import artifacts from parent and mirror for legacy scratchpad readers.
        try:
            if context is not None and sub_context is not None:
                parent_artifacts = getattr(context, "import_artifacts", None)
                if isinstance(parent_artifacts, MutableMapping):
                    try:
                        child_artifacts = getattr(sub_context, "import_artifacts", None)
                        if isinstance(child_artifacts, MutableMapping):
                            child_artifacts.update(parent_artifacts)
                        else:
                            setattr(sub_context, "import_artifacts", parent_artifacts)
                    except Exception:
                        pass
                    try:
                        child_sp = getattr(sub_context, "scratchpad", None)
                        if isinstance(child_sp, dict):
                            for k, v in parent_artifacts.items():
                                if v is None:
                                    continue
                                child_sp.setdefault(k, v)
                            setattr(sub_context, "scratchpad", child_sp)
                    except Exception:
                        pass
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

                        def _deep_merge_dict(a: JSONObject, b: JSONObject) -> JSONObject:
                            res: JSONObject = dict(a)
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
        child_final_ctx = sub_context
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
            propagate = True
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
                # Ensure parent scratchpad reflects child pause markers
                try:
                    if (
                        context is not None
                        and hasattr(context, "scratchpad")
                        and isinstance(context.scratchpad, dict)
                        and sub_context is not None
                        and hasattr(sub_context, "scratchpad")
                        and isinstance(sub_context.scratchpad, dict)
                    ):
                        for key in (
                            "status",
                            "pause_message",
                            "hitl_message",
                            "hitl_data",
                            "paused_step_input",
                        ):
                            if key in sub_context.scratchpad:
                                context.scratchpad[key] = sub_context.scratchpad[key]
                except Exception:
                    pass
                # Default to propagating unless explicitly disabled on the step
                try:
                    propagate = bool(getattr(step, "propagate_hitl", True))
                except Exception:
                    propagate = True
                # Mark parent context as paused only when propagation is enabled
                if propagate:
                    if context is not None and hasattr(context, "scratchpad"):
                        try:
                            sp = getattr(context, "scratchpad")
                            if isinstance(sp, dict):
                                sp["status"] = "paused"
                                msg = getattr(e, "message", None)
                                # Use plain message for backward compatibility
                                sp["pause_message"] = (
                                    msg if isinstance(msg, str) else getattr(e, "message", "")
                                )
                                sp.setdefault("hitl_message", sp.get("pause_message"))
                        except Exception:
                            pass
                    # Also preserve assistant turn so resume later has both roles
                    try:
                        if context is not None:
                            from flujo.domain.models import ConversationTurn, ConversationRole

                            hist = getattr(context, "conversation_history", None)
                            if not isinstance(hist, list):
                                hist = []
                            msg = getattr(e, "message", None) or ""
                            if msg and (not hist or getattr(hist[-1], "content", None) != msg):
                                hist.append(
                                    ConversationTurn(role=ConversationRole.assistant, content=msg)
                                )
                            setattr(context, "conversation_history", hist)
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
                return Paused(message=getattr(e, "message", ""))
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
            PipelineAbortSignal,
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
            branch_context=(
                # When outputs is specified (non-None, non-empty), use parent context
                # to prevent child values from leaking; the outputs mapping handles the merge.
                # When outputs is None or empty [], inherit child context if inherit_context=True.
                context
                if step.outputs
                else (
                    child_final_ctx
                    if step.inherit_context and child_final_ctx is not None
                    else context
                )
            ),
            metadata_={},
            step_history=([inner_sr] if inner_sr is not None else []),
        )
        if getattr(step, "outputs", None) == []:
            parent_sr.branch_context = context

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
        try:
            if getattr(pipeline_result, "step_history", None):
                last_ctx = getattr(pipeline_result.step_history[-1], "branch_context", None)
                if last_ctx is not None:
                    child_final_ctx = last_ctx
        except Exception:
            pass
        # Proactively merge child scratchpad into parent context to avoid state leakage
        # when upstream merge strategies or exclusions skip scratchpad fields.
        # Skip this merge when outputs is specified, as the outputs mapping will handle it.
        try:
            outputs = getattr(step, "outputs", None)
            # Only do proactive merge when outputs is None (not when outputs is specified or empty list)
            if outputs is None:
                if (
                    context is not None
                    and child_final_ctx is not None
                    and hasattr(context, "scratchpad")
                    and hasattr(child_final_ctx, "scratchpad")
                    and isinstance(context.scratchpad, dict)
                    and isinstance(child_final_ctx.scratchpad, dict)
                ):
                    context.scratchpad.update(child_final_ctx.scratchpad)
                    # Keep branch_context aligned with parent after merge to simplify callers
                    child_final_ctx = context
        except Exception:
            pass

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
            update_data: JSONObject = {}

            # Sentinel to distinguish "path not found" from "path found with None value"
            _NOT_FOUND: Any = object()

            def _traverse_path(obj: Any, parts: list[str]) -> Any:
                """Traverse a path through an object (context or dict).

                Returns _NOT_FOUND if the path doesn't exist, otherwise returns
                the value at the path (which may be None).
                """
                cur = obj
                for part in parts:
                    if cur is None:
                        return _NOT_FOUND
                    if hasattr(cur, part):
                        cur = getattr(cur, part)
                    elif isinstance(cur, dict):
                        if part in cur:
                            cur = cur[part]
                        else:
                            return _NOT_FOUND
                    else:
                        return _NOT_FOUND
                return cur

            def _get_child(path: str) -> tuple[Any, str] | Any:
                """Get a value from child context or last step output.

                Returns _NOT_FOUND if the path doesn't exist in either location.
                Returns the actual value (which may be None) if found.
                """
                parts = [p for p in path.split(".") if p]

                def _is_import_artifacts(obj: object) -> TypeGuard[ImportArtifacts]:
                    return isinstance(obj, ImportArtifacts)

                def _get_from_artifacts(artifact_path: list[str]) -> Any:
                    try:
                        art = getattr(child_final_ctx, "import_artifacts", None)
                        if _is_import_artifacts(art) and len(artifact_path) == 1:
                            name = artifact_path[0]
                            value = getattr(art, name, None)
                            try:
                                field = getattr(art.__class__, "model_fields", {}).get(name)
                                default_val = (
                                    field.default_factory()
                                    if field is not None and field.default_factory is not None
                                    else (field.default if field is not None else None)
                                )
                            except Exception:
                                default_val = None
                            if value is None or value == default_val:
                                return _NOT_FOUND
                            return value
                        if _is_import_artifacts(art):
                            return _NOT_FOUND
                        if isinstance(art, MutableMapping):
                            cur_art: Any = art
                            for part in artifact_path:
                                if isinstance(cur_art, MutableMapping) and part in cur_art:
                                    cur_art = cur_art[part]
                                else:
                                    return _NOT_FOUND
                            return _NOT_FOUND if cur_art is None else cur_art
                    except Exception:
                        pass
                    return _NOT_FOUND

                inner_candidate = _NOT_FOUND
                if inner_sr is not None:
                    inner_output = getattr(inner_sr, "output", None)
                    if isinstance(inner_output, dict):
                        inner_candidate = _traverse_path(inner_output, parts)

                if parts and parts[0] == "scratchpad":
                    # Prefer redirected import artifacts when scratchpad keys were migrated.
                    artifact_value = _get_from_artifacts(parts[1:])
                    if artifact_value is not _NOT_FOUND:
                        if artifact_value is not None:
                            return artifact_value, "artifacts"
                        # If artifacts contain an explicit None, still fall back to scratchpad in case
                        # the scratchpad has a concrete value.
                        result_ctx = _traverse_path(child_final_ctx, parts)
                        if result_ctx is not _NOT_FOUND:
                            return result_ctx, "context"
                        return None, "artifacts"

                # First: try to get from child's final context (branch_context)
                result = _traverse_path(child_final_ctx, parts)
                if result is not _NOT_FOUND:
                    # If context path exists but is empty (not None), prefer richer inner step output.
                    if (
                        result in ({}, [])
                        and inner_candidate is not _NOT_FOUND
                        and inner_candidate not in ({}, None)
                    ):
                        return inner_candidate, "output"
                    return result, "context"  # Found in context (may be None, that's valid)
                # Second: check the last step's output if context didn't have the value
                # This handles tool steps that return {"scratchpad": {...}} as output
                # but haven't had that output merged into context yet.
                if inner_candidate is not _NOT_FOUND:
                    return inner_candidate, "output"  # Found in output (may be None, that's valid)
                return _NOT_FOUND  # Not found anywhere

            parent_ctx = context

            def _assign_parent(path: str, value: Any) -> None:
                parts = [p for p in path.split(".") if p]
                if not parts:
                    return

                def _assign_nested(
                    target: MutableMapping[str, Any], keys: list[str], val: Any
                ) -> None:
                    cur = target
                    for k in keys[:-1]:
                        nxt = cur.get(k)
                        if not isinstance(nxt, MutableMapping):
                            nxt = {}
                            cur[k] = nxt
                        cur = nxt
                    cur[keys[-1]] = val

                normalized = value
                if isinstance(value, dict):
                    try:
                        from flujo.state.backends.base import _serialize_for_json as _normalize_json

                        normalized = _normalize_json(value)
                    except Exception:
                        normalized = value

                # Preserve scratchpad mapping when explicitly requested
                scratch_keys = None
                if parts[0] == "scratchpad":
                    scratch_keys = parts[1:]
                    if scratch_keys:
                        sp_target = update_data.setdefault("scratchpad", {})
                        if isinstance(sp_target, MutableMapping):
                            _assign_nested(sp_target, scratch_keys, normalized)

                # Route mapped outputs into import_artifacts for deterministic propagation
                tgt_parts = parts[1:] if parts[0] == "scratchpad" else parts
                if not tgt_parts:
                    return

                tgt = update_data.setdefault("import_artifacts", {})
                if not tgt and parent_ctx is not None and hasattr(parent_ctx, "import_artifacts"):
                    pa = getattr(parent_ctx, "import_artifacts", None)
                    if isinstance(pa, MutableMapping):
                        try:
                            tgt.update(dict(pa))
                        except Exception:
                            pass
                if isinstance(tgt, MutableMapping):
                    _assign_nested(tgt, tgt_parts, normalized)

                if parent_ctx is not None and hasattr(parent_ctx, "import_artifacts"):
                    pc_artifacts = getattr(parent_ctx, "import_artifacts", None)
                    if isinstance(pc_artifacts, MutableMapping):
                        _assign_nested(pc_artifacts, tgt_parts, normalized)

            try:
                for mapping in step.outputs:
                    try:
                        parent_path = mapping.parent
                        child_val = _get_child(mapping.child)
                        # Skip only truly missing child paths (not found in context or output)
                        # Note: None is a valid value if the path exists
                        if child_val is _NOT_FOUND:
                            continue
                        if isinstance(child_val, tuple):
                            child_value, source = child_val
                        else:
                            child_value, source = child_val, "context"

                        # Preserve explicit None on parent artifacts over non-context outputs.
                        if (
                            source == "output"
                            and parent_ctx is not None
                            and hasattr(parent_ctx, "import_artifacts")
                        ):
                            try:
                                existing = getattr(parent_ctx.import_artifacts, parent_path, None)
                                if existing is None:
                                    _assign_parent(parent_path, None)
                                    continue
                            except Exception:
                                pass

                        _assign_parent(parent_path, child_value)
                    except Exception:
                        continue
                parent_sr.output = update_data
            except Exception:
                parent_sr.output = getattr(inner_sr, "output", None)
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
                parent_sr.output = getattr(inner_sr, "output", None)
        else:
            parent_sr.output = getattr(inner_sr, "output", None) if inner_sr is not None else None

        return Success(step_result=parent_sr)


# --- End Import Step Executor policy ---
