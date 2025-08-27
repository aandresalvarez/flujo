from flujo.application.conversation.history_manager import (
    HistoryManager,
    HistoryStrategyConfig,
)
from flujo.domain.models import ConversationTurn, ConversationRole


def _mk_turn(role: str, text: str) -> ConversationTurn:
    return ConversationTurn(role=ConversationRole(role), content=text)


def test_truncate_turns_strategy() -> None:
    turns = [_mk_turn("user", f"u{i}") for i in range(5)]
    hm = HistoryManager(HistoryStrategyConfig(strategy="truncate_turns", max_turns=3))
    bounded = hm.bound_history(turns)
    assert [t.content for t in bounded] == ["u2", "u3", "u4"]


def test_truncate_tokens_strategy() -> None:
    # Each token ~4 chars; set max_tokens small to force truncation
    turns = [_mk_turn("assistant", "x" * 100), _mk_turn("user", "y" * 100)]
    hm = HistoryManager(HistoryStrategyConfig(strategy="truncate_tokens", max_tokens=30))
    bounded = hm.bound_history(turns)
    # Expect only last turn retained under budget
    assert len(bounded) == 1
    assert bounded[0].role == ConversationRole.user


def test_summarize_strategy_compacts_and_bounds() -> None:
    turns = [
        _mk_turn("user", "hello"),
        _mk_turn("assistant", "world"),
        _mk_turn("user", "what's up"),
        _mk_turn("assistant", "not much"),
    ]
    hm = HistoryManager(
        HistoryStrategyConfig(strategy="summarize", summarize_ratio=0.5, max_tokens=50)
    )
    bounded = hm.bound_history(turns)
    # First older half summarized into a single assistant turn
    assert len(bounded) >= 2
    assert bounded[0].role == ConversationRole.assistant


def test_config_manager_defaults(monkeypatch) -> None:
    class _FakeCfgMgr:
        def load_config(self):
            return {
                "conversation": {
                    "history_management": {"strategy": "truncate_turns", "max_turns": 1}
                }
            }

    def _fake_get_config_manager():
        return _FakeCfgMgr()

    import flujo.application.conversation.history_manager as hm_mod

    monkeypatch.setattr(hm_mod, "get_config_manager", _fake_get_config_manager, raising=False)
    hm = HistoryManager()
    assert hm.cfg.strategy == "truncate_turns"
    # Strategy should respect default max_turns=1
    turns = [_mk_turn("user", "a"), _mk_turn("assistant", "b")]
    bounded = hm.bound_history(turns)
    assert len(bounded) == 1


def test_filter_natural_text_skips_tool_payloads() -> None:
    turns = [
        _mk_turn("assistant", "TOOL: call something"),
        _mk_turn("assistant", '{"tool_call": "x", "arguments": {}}'),
        _mk_turn("user", "hello"),
    ]
    kept = HistoryManager.filter_natural_text(turns)
    assert [t.content for t in kept] == ["hello"]
