from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import TypeGuard
from collections.abc import MutableMapping
from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.domain.models import ImportArtifacts

from ._shared import (
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
    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultImportStepExecutor(StepPolicy[ImportStep]):
    @property
    def handles_type(self) -> type[ImportStep]:
        return ImportStep

    async def execute(
        self, core: object, frame: ExecutionFrame[DomainBaseModel]
    ) -> StepOutcome[StepResult]:
        step = frame.step
        if not isinstance(step, ImportStep):
            raise TypeError(
                f"DefaultImportStepExecutor received non-ImportStep: {type(step).__name__}"
            )
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

        # Seed child import artifacts from parent.
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
        except Exception:
            pass

        # Project input into child run and compute the child's initial_input explicitly,
        # honoring explicit inputs over inherited conversation or parent data.
        # Precedence:
        #   1) sub_context.import_artifacts[input_scratchpad_key] when present (explicit artifact)
        #   2) provided data argument (parent current_data)
        #   3) empty string fallback
        resolved_origin = "parent_data"
        sub_initial_input = data
        try:
            if sub_context is not None and hasattr(sub_context, "import_artifacts"):
                art = getattr(sub_context, "import_artifacts", None)
                if isinstance(art, MutableMapping):
                    key = step.input_scratchpad_key or "initial_input"
                    if key in art and art.get(key) is not None:
                        sub_initial_input = art.get(key)
                        resolved_origin = f"import_artifacts:{key}"
        except Exception:
            pass
        try:
            # import_artifacts projection
            if step.input_to in ("import_artifacts", "both") and sub_context is not None:
                art = getattr(sub_context, "import_artifacts", None)
                if isinstance(art, MutableMapping):
                    if isinstance(data, dict):
                        try:
                            art.update(copy.deepcopy(data))
                        except Exception:
                            for k, v in copy.deepcopy(data).items():
                                art[k] = v
                    else:
                        key = step.input_scratchpad_key or "initial_input"
                        art[key] = data

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
            exec_pipeline = getattr(core, "_execute_pipeline_via_policies", None)
            if not callable(exec_pipeline):
                raise AttributeError("_execute_pipeline_via_policies is not available on core")
            pipeline_result: PipelineResult[DomainBaseModel] = await exec_pipeline(
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
                            if merged_ctx is not None and isinstance(merged_ctx, DomainBaseModel):
                                context = merged_ctx
                        except Exception:
                            pass
                # No scratchpad propagation: prefer typed fields for pause markers and HITL data.
                # Context merges above (_safe_merge / ContextManager.merge) already handle typed fields.
                # Default to propagating unless explicitly disabled on the step
                try:
                    propagate = bool(getattr(step, "propagate_hitl", True))
                except Exception:
                    propagate = True
                # Mark parent context as paused only when propagation is enabled
                if propagate:
                    if context is not None:
                        try:
                            # Use typed field for status and pause_message
                            if hasattr(context, "status"):
                                context.status = "paused"
                            msg = getattr(e, "message", None)
                            if hasattr(context, "pause_message"):
                                context.pause_message = (
                                    msg if isinstance(msg, str) else getattr(e, "message", "")
                                )
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
                    if context is not None:
                        try:
                            # Use typed field for status
                            if hasattr(context, "status"):
                                current_status = getattr(context, "status", None)
                                if current_status == "paused":
                                    context.status = "running"
                            if hasattr(context, "pause_message"):
                                context.pause_message = None
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
                        "via import_artifacts or ensure the correct payload is provided."
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
        # Proactively merge child *typed* framework fields into parent context (no scratchpad writes).
        # Skip this merge when outputs is specified, as the outputs mapping will handle it.
        try:
            outputs = getattr(step, "outputs", None)
            # Only do proactive merge when outputs is None (not when outputs is specified or empty list)
            if outputs is None:
                if context is not None and child_final_ctx is not None:
                    for attr in (
                        "status",
                        "pause_message",
                        "current_state",
                        "next_state",
                        "paused_step_input",
                        "user_input",
                        "hitl_data",
                        "loop_iteration_index",
                        "loop_step_index",
                        "loop_last_output",
                        "loop_resume_requires_hitl_output",
                        "loop_paused_step_name",
                    ):
                        if hasattr(child_final_ctx, attr) and hasattr(context, attr):
                            try:
                                setattr(context, attr, getattr(child_final_ctx, attr))
                            except Exception:
                                pass
                    # Merge step_outputs if present
                    if hasattr(context, "step_outputs") and isinstance(
                        getattr(context, "step_outputs", None), dict
                    ):
                        if hasattr(child_final_ctx, "step_outputs") and isinstance(
                            getattr(child_final_ctx, "step_outputs", None), dict
                        ):
                            try:
                                context.step_outputs.update(child_final_ctx.step_outputs)
                            except Exception:
                                pass
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
            _NOT_FOUND: object = object()

            def _traverse_path(obj: object, parts: list[str]) -> object:
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

            def _get_child(path: str) -> tuple[object, str] | object:
                """Get a value from child context or last step output.

                Returns _NOT_FOUND if the path doesn't exist in either location.
                Returns the actual value (which may be None) if found.
                """
                parts = [p for p in path.split(".") if p]

                def _is_import_artifacts(obj: object) -> TypeGuard[ImportArtifacts]:
                    return isinstance(obj, ImportArtifacts)

                def _get_from_artifacts(artifact_path: list[str]) -> object:
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
                            cur_art: object = art
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
                # This handles tool steps that return dict-shaped updates
                # but haven't had that output merged into context yet.
                if inner_candidate is not _NOT_FOUND:
                    return inner_candidate, "output"  # Found in output (may be None, that's valid)
                return _NOT_FOUND  # Not found anywhere

            parent_ctx = context

            def _assign_parent(path: str, value: object) -> None:
                parts = [p for p in path.split(".") if p]
                if not parts:
                    return

                def _assign_nested(
                    target: MutableMapping[str, object], keys: list[str], val: object
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

                # Route mapped outputs into import_artifacts for deterministic propagation.
                # Parent paths may be bare (relative to import_artifacts) or already rooted
                # with "import_artifacts.*". Avoid double-nesting by stripping the root.
                tgt_parts = parts
                if tgt_parts and tgt_parts[0] == "import_artifacts":
                    tgt_parts = tgt_parts[1:]
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
                                existing_key = str(parent_path)
                                if existing_key.startswith("import_artifacts."):
                                    existing_key = existing_key.split(".", 1)[1]
                                if (
                                    "." not in existing_key
                                    and existing_key in parent_ctx.import_artifacts
                                    and parent_ctx.import_artifacts.get(existing_key) is None
                                ):
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
