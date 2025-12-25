from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Callable

from ...domain.models import ConversationTurn, ConversationRole
from ...type_definitions.common import JSONObject

# Optional import exposed at module scope so tests can monkeypatch it.
# If the central config manager is unavailable, this name remains None
# and defaults will be used.
try:  # pragma: no cover - exercised indirectly via tests
    from ...infra.config_manager import get_config_manager as _cfg_getter

    get_config_manager: Callable[[bool], Any] = _cfg_getter
except Exception:  # pragma: no cover
    # Fallback: no config manager available (tests can monkeypatch)
    def get_config_manager(_force_reload: bool = False) -> Any:
        return None


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
        # Initialize from explicit cfg or centralized config defaults
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = self._load_defaults_from_config_manager() or HistoryStrategyConfig()

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

    def summarize(
        self,
        parts: Sequence[str],
        *,
        max_tokens: int = 2000,
        model_id: Optional[str] = None,
    ) -> str:
        """Summarize a list of path segments into a token-bounded string."""
        if not parts:
            return ""
        max_tokens = max(1, int(max_tokens or 0))
        kept: List[str] = []
        running = 0
        for text in reversed(parts):
            turn = ConversationTurn(role=ConversationRole.assistant, content=str(text))
            t = self._estimate_turn_tokens(turn, model_id=model_id)
            if running + t > max_tokens and kept:
                break
            kept.append(str(text))
            running += t
        kept.reverse()
        if len(kept) < len(parts):
            return "... (summarized) ...\n" + "\n".join(kept)
        return "\n".join(kept)

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
        # Honor the provided token budget; clamp only to a sane minimum of 1
        max_tokens = max(1, int(self.cfg.max_tokens or 0))
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
        # Best-effort heuristic with optional tiktoken support.
        # Use a conservative estimate to avoid undercounting tokens which
        # could lead to overshooting the limit in tests/CI environments.
        txt = f"{turn.role.value}: {turn.content}"
        # Conservative fallback: assume ~2.5 chars per token to avoid overflow on JSON/code.
        fallback = max(1, int(len(txt) / 2.5))
        # Fast path: when no model_id is provided, avoid tokenizer imports entirely
        # and use the conservative heuristic. This dramatically reduces overhead
        # in tight loops (e.g., CI benchmarks) while keeping bounds safe.
        if model_id is None:
            return fallback
        try:  # pragma: no cover - environment dependent
            import importlib as _importlib

            _t = _importlib.import_module("tiktoken")
            enc = _t.get_encoding("cl100k_base")
            measured = max(1, len(enc.encode(txt)))
            # Take the maximum of measured and fallback to be conservative.
            # This keeps behavior stable even if tokenization yields fewer tokens
            # for repetitive strings (e.g., "yyyy...").
            return max(measured, fallback)
        except Exception:
            # No tokenizer available or failed; use fallback heuristic
            return fallback

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
        # Strip obvious tool/function-call artifacts heuristically; keep natural text
        out: List[ConversationTurn] = []
        for t in turns:
            content = (t.content or "").strip()
            if not content:
                continue
            # Heuristics for tool/function-call artifacts
            lower = content.lower()
            if lower.startswith("tool:") or lower.startswith("[tool]") or "function_call" in lower:
                continue
            # Skip JSON-like tool call payloads with common keys
            if content.startswith("{") and content.endswith("}"):
                try:
                    import json as _json

                    obj: JSONObject = _json.loads(content)
                    keys = {str(k).lower() for k in obj.keys()}
                    if {"tool", "tool_call", "function", "arguments", "name"} & keys:
                        continue
                except Exception:
                    # Non-JSON or large text; keep it
                    pass
            out.append(t)
        return out

    def _load_defaults_from_config_manager(self) -> Optional[HistoryStrategyConfig]:
        """Load default strategy from centralized configuration, if available.

        Expected shape (flujo.toml → dict):
        {
          "conversation": {
            "history_management": {
              "strategy": "truncate_tokens|truncate_turns|summarize",
              "max_tokens": 4096,
              "max_turns": 20,
              "summarize_ratio": 0.5
            }
          }
        }
        """
        try:
            # Use module-scoped hookable symbol for easier testing/monkeypatching
            cfg_loader = get_config_manager
            if cfg_loader is None:
                return None
            # Support both signatures: get_config_manager() and get_config_manager(force_reload: bool)
            try:
                mgr = cfg_loader(False)
            except TypeError:
                mgr = cfg_loader()  # type: ignore[call-arg]
            cfg_obj: Any = mgr.load_config()
            # Coerce to dict to support FlujoConfig (pydantic model)
            if hasattr(cfg_obj, "model_dump") and callable(getattr(cfg_obj, "model_dump")):
                cfg: Any = cfg_obj.model_dump()
            else:
                cfg = cfg_obj or {}
            conv: Any = (cfg.get("conversation") or {}) if isinstance(cfg, dict) else {}
            hm: Any = (conv.get("history_management") or {}) if isinstance(conv, dict) else {}
            if not isinstance(hm, dict):
                return None
            strategy = str(hm.get("strategy") or "truncate_tokens")
            max_tokens = int(hm.get("max_tokens") or 4096)
            max_turns = int(hm.get("max_turns") or 20)
            summarize_ratio = float(hm.get("summarize_ratio") or 0.5)
            return HistoryStrategyConfig(
                strategy=strategy,
                max_tokens=max_tokens,
                max_turns=max_turns,
                summarizer_agent=None,
                summarize_ratio=summarize_ratio,
            )
        except Exception:
            return None
