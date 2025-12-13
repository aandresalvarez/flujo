from __future__ import annotations


def _resolve_context_target(ctx: object, target: str) -> tuple[object | None, str | None]:
    """Resolve a context.* target path to (parent, key)."""
    try:
        if not isinstance(target, str) or not target.startswith("context."):
            return None, None
        parts = target.split(".")[1:]
        cur = ctx
        parent = None
        key = None
        for p in parts:
            parent = cur
            key = p
            try:
                if isinstance(cur, dict):
                    if p not in cur:
                        try:
                            cur[p] = {}
                        except Exception:
                            pass
                    cur = cur.get(p)
                    continue
            except Exception:
                pass
            nxt = getattr(cur, p, None)
            if nxt is None:
                try:
                    setattr(cur, p, {})
                    nxt = getattr(cur, p, None)
                except Exception:
                    nxt = None
            cur = nxt
        return parent, key
    except Exception:
        return None, None


def _render_template_value(prev_output: object, ctx: object, tpl: object) -> object:
    """Render a template string using context and previous output."""
    try:
        from ...utils.template_vars import (
            TemplateContextProxy as _TCP,
            get_steps_map_from_context as _get_steps,
            StepValueProxy as _SVP,
        )
        from ...utils.prompting import AdvancedPromptFormatter as _Fmt

        steps_map = _get_steps(ctx)
        steps_wrapped = {k: v if isinstance(v, _SVP) else _SVP(v) for k, v in steps_map.items()}
        fmt_ctx = {
            "context": _TCP(ctx, steps=steps_wrapped),
            "previous_step": prev_output,
            "steps": steps_wrapped,
        }

        try:
            if ctx is not None and hasattr(ctx, "hitl_history"):
                hitl_history = getattr(ctx, "hitl_history", None)
                if isinstance(hitl_history, list) and hitl_history:
                    fmt_ctx["resume_input"] = getattr(hitl_history[-1], "human_response", None)
        except Exception:
            pass

        return _Fmt(str(tpl)).format(**fmt_ctx)
    except Exception:
        try:
            return str(tpl)
        except Exception:
            return tpl


__all__ = ["_resolve_context_target", "_render_template_value"]
