"""Context update handling for steps marked with updates_context."""

from __future__ import annotations

from flujo.domain.models import BaseModel

from .context_adapter import _build_context_update, _inject_context_with_deep_merge


class ContextUpdateManager:
    """Applies step outputs into pipeline context with deep-merge semantics."""

    def apply_updates(
        self,
        *,
        step: object,
        output: object,
        context: BaseModel | None,
    ) -> str | None:
        """Apply updates_context semantics. Returns validation_error (if any)."""
        if context is None or not getattr(step, "updates_context", False):
            return None

        try:
            update_data = _build_context_update(output)
        except Exception:
            return None

        if not update_data:
            return None

        try:
            validation_error = _inject_context_with_deep_merge(context, update_data, type(context))

            if validation_error:
                # Best-effort merge from nested PipelineResult contexts when available
                try:
                    sub_ctx = None
                    if hasattr(output, "final_pipeline_context"):
                        sub_ctx = getattr(output, "final_pipeline_context", None)
                    if sub_ctx is not None:
                        cm = type(context)
                        for fname in getattr(cm, "model_fields", {}):
                            if not hasattr(sub_ctx, fname):
                                continue
                            new_val = getattr(sub_ctx, fname)
                            if new_val is None:
                                continue
                            cur_val = getattr(context, fname, None)
                            if isinstance(cur_val, dict) and isinstance(new_val, dict):
                                try:
                                    cur_val.update(new_val)
                                except Exception:
                                    setattr(context, fname, new_val)
                            else:
                                setattr(context, fname, new_val)
                        validation_error = None
                except Exception:
                    # Preserve original validation_error if fallback merge fails
                    pass

            return validation_error
        except Exception:
            # Never raise from context update handling
            return None
