from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Protocol


class _PipelineResultLike(Protocol):
    @property
    def step_history(self) -> Sequence[object]: ...

    @property
    def final_pipeline_context(self) -> object | None: ...


def seed_conversation_history(iteration_context: object, current_data: object) -> None:
    """Initialize conversation history for the current iteration."""
    from flujo.domain.models import ConversationRole, ConversationTurn

    hist_obj = getattr(iteration_context, "conversation_history", None)
    hist: list[object]
    if isinstance(hist_obj, list):
        hist = hist_obj
    else:
        hist = []
        try:
            setattr(iteration_context, "conversation_history", hist)
        except Exception:
            pass

    if isinstance(hist, list) and not hist:
        initial_text = str(current_data) if current_data is not None else ""
        if (initial_text or "").strip():
            hist.append(ConversationTurn(role=ConversationRole.user, content=initial_text))

    try:
        hitl_hist = getattr(iteration_context, "hitl_history", None)
        if isinstance(hitl_hist, list) and hitl_hist:
            last_resp = getattr(hitl_hist[-1], "human_response", None)
            text = str(last_resp) if last_resp is not None else ""
            if text.strip():
                last_content = getattr(hist[-1], "content", None) if hist else None
                if last_content != text:
                    hist.append(ConversationTurn(role=ConversationRole.user, content=text))
    except Exception:
        pass


def collect_step_name_sources(body_pipeline: object) -> tuple[set[str], set[str]]:
    """Collect HITL and agent step names for conversation sourcing."""
    from ._shared import HumanInTheLoopStep, Step

    hitl_step_names: set[str] = set()
    agent_step_names: set[str] = set()

    def _collect_names_recursive(p: object) -> None:
        try:
            steps_list = list(getattr(p, "steps", []) or [])
        except Exception:
            steps_list = []
        for _st in steps_list:
            try:
                if isinstance(_st, HumanInTheLoopStep):
                    hitl_step_names.add(getattr(_st, "name", ""))
                elif isinstance(_st, Step) and not getattr(_st, "is_complex", False):
                    agent_step_names.add(getattr(_st, "name", ""))
            except Exception:
                pass
            try:
                branches = getattr(_st, "branches", None)
                if isinstance(branches, dict):
                    for _bp in branches.values():
                        _collect_names_recursive(_bp)
            except Exception:
                pass
            try:
                def_branch = getattr(_st, "default_branch_pipeline", None)
                if def_branch is not None:
                    _collect_names_recursive(def_branch)
            except Exception:
                pass
            try:
                lbp = (
                    _st.get_loop_body_pipeline()
                    if hasattr(_st, "get_loop_body_pipeline")
                    else getattr(_st, "loop_body_pipeline", None)
                )
                if lbp is not None:
                    _collect_names_recursive(lbp)
            except Exception:
                pass

    try:
        if hasattr(body_pipeline, "steps"):
            _collect_names_recursive(body_pipeline)
    except Exception:
        pass
    return hitl_step_names, agent_step_names


def sync_conversation_history(
    *,
    current_context: object | None,
    iteration_context: object | None,
    pipeline_result: _PipelineResultLike,
    conv_enabled: bool,
    ai_src: str,
    user_src: list[str],
    named_steps_set: set[str],
    hitl_step_names: set[str],
) -> None:
    """Append conversation turns after an iteration completes."""
    from flujo.domain.models import ConversationRole, ConversationTurn

    if not conv_enabled:
        return
    try:
        ctx_target = (
            current_context
            if current_context is not None
            else (
                pipeline_result.final_pipeline_context
                if pipeline_result.final_pipeline_context is not None
                else iteration_context
            )
        )
        if ctx_target is None or not hasattr(ctx_target, "conversation_history"):
            return
        hist = getattr(ctx_target, "conversation_history", None)
        if not isinstance(hist, list):
            return

        def _flatten(results: Sequence[object]) -> list[object]:
            flat: list[object] = []

            def _rec(items: Sequence[object]) -> None:
                for _sr in items:
                    flat.append(_sr)
                    try:
                        children = getattr(_sr, "step_history", None)
                    except Exception:
                        children = None
                    if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                        _rec(children)

            _rec(results)
            return flat

        all_srs = _flatten(pipeline_result.step_history)
        try:
            sources_set = set(s for s in user_src if isinstance(s, str))
            for sr in all_srs:
                n = getattr(sr, "name", "")
                if not getattr(sr, "success", False):
                    continue
                if "hitl" in sources_set and n in hitl_step_names:
                    txt = str(getattr(sr, "output", "") or "")
                    if txt.strip():
                        last_content = getattr(hist[-1], "content", None) if hist else None
                        if last_content != txt:
                            hist.append(ConversationTurn(role=ConversationRole.user, content=txt))
                        continue
                if n in sources_set:
                    txt = str(getattr(sr, "output", "") or "")
                    if txt.strip():
                        last_content = getattr(hist[-1], "content", None) if hist else None
                        if last_content != txt:
                            hist.append(ConversationTurn(role=ConversationRole.user, content=txt))
        except Exception:
            pass

        def _extract_assistant_question(val: object) -> tuple[Optional[str], Optional[str]]:
            try:
                if isinstance(val, dict):
                    act = val.get("action")
                    action = str(act).lower() if act is not None else None
                    q = val.get("question")
                    qtxt = q if isinstance(q, str) and q.strip() else None
                    return qtxt, action
                action_attr = getattr(val, "action", None)
                question_attr = getattr(val, "question", None)
                if action_attr is not None or question_attr is not None:
                    action = str(action_attr).lower() if action_attr is not None else None
                    qtxt = (
                        str(question_attr).strip()
                        if isinstance(question_attr, str) and str(question_attr).strip()
                        else None
                    )
                    return qtxt, action
                if isinstance(val, str):
                    raw = val.strip()
                    if (raw.startswith("{") and raw.endswith("}")) or (
                        raw.startswith("[") and raw.endswith("]")
                    ):
                        import json as _json

                        try:
                            obj = _json.loads(raw)
                            if isinstance(obj, dict):
                                act = obj.get("action")
                                action = str(act).lower() if act is not None else None
                                q = obj.get("question")
                                qtxt = q if isinstance(q, str) and q.strip() else None
                                return qtxt, action
                        except Exception:
                            pass
                    low = raw.lower()
                    if low in {"finish", "done"}:
                        return None, "finish"
                    if '"action":"finish"' in low or "action='finish'" in low:
                        return None, "finish"
                    return (raw if raw else None), None
            except Exception:
                pass
            return None, None

        try:
            src = ai_src
            if src == "last":
                chosen: Optional[str] = None
                for _sr in reversed(all_srs):
                    if not getattr(_sr, "success", False):
                        continue
                    qtxt, action = _extract_assistant_question(getattr(_sr, "output", None))
                    if (action or "").lower() != "finish" and qtxt and qtxt.strip():
                        chosen = qtxt
                        break
                if chosen:
                    hist.append(ConversationTurn(role=ConversationRole.assistant, content=chosen))
            elif src == "all_agents":
                for sr in all_srs:
                    if not getattr(sr, "success", False):
                        continue
                    out_val = getattr(sr, "output", None)
                    if (
                        isinstance(out_val, dict)
                        and out_val
                        and not ("action" in out_val or "question" in out_val)
                    ):
                        for _v in out_val.values():
                            qtxt, action = _extract_assistant_question(_v)
                            if (action or "").lower() != "finish" and qtxt and qtxt.strip():
                                hist.append(
                                    ConversationTurn(role=ConversationRole.assistant, content=qtxt)
                                )
                        continue
                    qtxt, action = _extract_assistant_question(out_val)
                    if (action or "").lower() != "finish" and qtxt and qtxt.strip():
                        hist.append(ConversationTurn(role=ConversationRole.assistant, content=qtxt))
            elif src == "named_steps":
                for sr in all_srs:
                    n = getattr(sr, "name", "")
                    if not getattr(sr, "success", False):
                        continue
                    if n in named_steps_set:
                        qtxt, action = _extract_assistant_question(getattr(sr, "output", None))
                        if (action or "").lower() != "finish" and qtxt and qtxt.strip():
                            hist.append(
                                ConversationTurn(role=ConversationRole.assistant, content=qtxt)
                            )
        except Exception:
            pass
    except Exception:
        pass
