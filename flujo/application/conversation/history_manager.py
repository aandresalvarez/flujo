from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ...domain.models import ConversationTurn, ConversationRole


@dataclass
class HistoryStrategyConfig:
    """Configuration for conversation history management.

    Fields intentionally mirror the DSL keys proposed in FSD-033.
    """

    strategy: str = "truncate_tokens"  # truncate_tokens | truncate_turns | summarize
    max_tokens: int = 4096
    max_turns: int = 20
    summarizer_agent: Optional[Any] = None  # Future: Agent callable or registry key
    summarize_ratio: float = 0.5  # Proportion of oldest turns to condense when summarizing


class HistoryManager:
    """Prepare a bounded/summarized history slice suitable for prompt injection.

    This utility does not read settings directly. Callers can pass model_id
    or other hints as needed. Token estimation uses best-effort heuristics
    with optional tiktoken support when available.
    """

    def __init__(self, cfg: Optional[HistoryStrategyConfig] = None) -> None:
        self.cfg = cfg or HistoryStrategyConfig()

    def bound_history(
        self,
        history: Sequence[ConversationTurn],
        *,
        model_id: Optional[str] = None,
    ) -> List[ConversationTurn]:
        if not history:
            return []

        strat = (self.cfg.strategy or "truncate_tokens").strip().lower()
        if strat == "truncate_turns":
            return self._by_turns(history)
        if strat == "summarize":
            return self._summarize(history, model_id=model_id)
        # default: truncate_tokens
        return self._by_tokens(history, model_id=model_id)

    # --------------------
    # Strategies
    # --------------------
    def _by_turns(self, history: Sequence[ConversationTurn]) -> List[ConversationTurn]:
        max_turns = max(1, int(self.cfg.max_turns or 0))
        if len(history) <= max_turns:
            return list(history)
        return list(history)[-max_turns:]

    def _by_tokens(
        self, history: Sequence[ConversationTurn], *, model_id: Optional[str]
    ) -> List[ConversationTurn]:
        max_tokens = max(256, int(self.cfg.max_tokens or 0))
        # Keep most recent turns within token budget
        kept: List[ConversationTurn] = []
        running = 0
        for turn in reversed(history):
            t = self._estimate_turn_tokens(turn, model_id=model_id)
            if running + t > max_tokens and kept:
                break
            kept.append(turn)
            running += t
        kept.reverse()
        return kept

    def _summarize(
        self, history: Sequence[ConversationTurn], *, model_id: Optional[str]
    ) -> List[ConversationTurn]:
        if not history:
            return []
        ratio = self.cfg.summarize_ratio
        if not (0.0 < ratio < 1.0):
            ratio = 0.5
        split_idx = max(1, int(len(history) * ratio))
        older = list(history[:split_idx])
        newer = list(history[split_idx:])

        # If a summarizer agent is provided, call it to produce a compact assistant turn.
        # Otherwise, fallback to a deterministic compact join of older content.
        summary_text = (
            self._summarize_with_agent(older)
            if self.cfg.summarizer_agent
            else self._simple_summarize(older)
        )

        compact = ConversationTurn(role=ConversationRole.assistant, content=summary_text)
        candidate = [compact] + newer
        # Enforce final token bound as a second pass
        return self._by_tokens(candidate, model_id=model_id)

    # --------------------
    # Helpers
    # --------------------
    def _estimate_turn_tokens(self, turn: ConversationTurn, *, model_id: Optional[str]) -> int:
        # Best-effort heuristic with optional tiktoken support
        txt = f"{turn.role.value}: {turn.content}"
        try:
            import importlib as _importlib

            _t = _importlib.import_module("tiktoken")
            enc = _t.get_encoding("cl100k_base")
            return max(1, len(enc.encode(txt)))
        except Exception:
            pass
        # Fallback heuristic: ~4 chars per token
        return max(1, len(txt) // 4)

    def _simple_summarize(self, turns: Sequence[ConversationTurn]) -> str:
        # Deterministic compact form: keep first/last user messages and note compression
        if not turns:
            return ""
        texts = [f"{t.role.value}: {t.content}" for t in turns if (t.content or "").strip()]
        if not texts:
            return ""
        if len(texts) <= 2:
            return " \n".join(texts)
        return texts[0] + "\n... (summarized) ...\n" + texts[-1]

    def _summarize_with_agent(self, turns: Sequence[ConversationTurn]) -> str:
        # Pluggable path: accept a callable(agent) with a simple signature or registry key in future.
        try:
            agent = self.cfg.summarizer_agent
            if agent is None:
                return self._simple_summarize(turns)
            # Accept a simple callable that returns str; avoid tight coupling to Agent protocol here
            payload = "\n".join(f"{t.role.value}: {t.content}" for t in turns)
            result = agent(payload)
            if isinstance(result, str) and result.strip():
                return result
        except Exception:
            pass
        return self._simple_summarize(turns)

    @staticmethod
    def filter_natural_text(turns: Sequence[ConversationTurn]) -> List[ConversationTurn]:
        # Future: strip tool-call artifacts; for now, pass through unchanged
        return list(turns)
