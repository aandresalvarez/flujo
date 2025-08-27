from __future__ import annotations

from typing import Any, Optional

from .base import Processor
from ..application.conversation.history_manager import HistoryManager, HistoryStrategyConfig
from ..domain.models import BaseModel, PipelineContext
from ..utils.prompting import format_prompt
from ..utils.redact import summarize_and_redact_prompt
from ..tracing.manager import get_active_trace_manager


class ConversationHistoryPromptProcessor(Processor):
    """Injects conversational history into the agent input.

    This processor is attached dynamically by loop policy when `conversation: true`.
    It does not mutate the agent definition or global step configuration; instead,
    it prepares a rendered history block and augments the prompt at call time.
    """

    def __init__(
        self,
        *,
        history_manager: Optional[HistoryManager] = None,
        history_template: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._hm = history_manager or HistoryManager(HistoryStrategyConfig())
        self._tpl = history_template or (
            """--- Conversation History ---\n{{#each history}}{{ this.role }}: {{ this.content }}\n{{/each}}\n-----------------------------\n"""
        )
        self._model_id = model_id
        self.name = "conversation_history_injector"

    async def process(self, data: Any, context: Optional[BaseModel] = None) -> Any:
        # Only operate with a PipelineContext that has a non-empty history
        ctx = context if isinstance(context, PipelineContext) else None
        turns = list(getattr(ctx, "conversation_history", []) or []) if ctx else []
        if not turns:
            return data

        # Prepare bounded/summarized history slice
        filtered = self._hm.filter_natural_text(turns)
        bounded = self._hm.bound_history(filtered, model_id=self._model_id)

        # Pin the original initial user turn as the first element when it would
        # otherwise be truncated, so templates like `history.0` reliably refer
        # to the initial goal. Avoid duplication when already present.
        try:
            initial_turn = filtered[0] if filtered else None
            if initial_turn is not None:
                need_pin = True
                if bounded:
                    b0 = bounded[0]
                    need_pin = not (
                        getattr(b0, "role", None) == getattr(initial_turn, "role", None)
                        and getattr(b0, "content", None) == getattr(initial_turn, "content", None)
                    )
                if need_pin:
                    bounded = [initial_turn] + bounded
        except Exception:
            # Best-effort: if anything goes wrong, fall back to bounded
            pass

        # Render via template
        try:
            rendered = format_prompt(
                self._tpl,
                history=[{"role": t.role.value, "content": t.content} for t in bounded],
            )
        except Exception:
            # Fallback to a simple text join
            rendered = "\n".join(f"{t.role.value}: {t.content}" for t in bounded)

        # Redact and clamp logging size for safety when surfaced in tracing
        redacted_preview = summarize_and_redact_prompt(rendered, max_length=1000)
        try:
            tm = get_active_trace_manager()
            if tm is not None:
                tm.add_event("agent.prompt", {"rendered_history": redacted_preview})
        except Exception:
            pass

        # Injection strategy:
        # - If input is a string, prepend history block.
        # - If input is a dict with a 'prompt' key, prepend history to it.
        # - Otherwise, pass through unchanged (future: chat messages path).
        if isinstance(data, str):
            return f"{rendered}\n\n{data}"
        if isinstance(data, dict):
            if isinstance(data.get("prompt"), str):
                new = dict(data)
                new["prompt"] = f"{rendered}\n\n{data['prompt']}"
                return new
        return data
