from flujo.processors.conversation import ConversationHistoryPromptProcessor
from flujo.application.conversation.history_manager import HistoryStrategyConfig, HistoryManager
from flujo.domain.models import PipelineContext, ConversationTurn, ConversationRole
import asyncio


def _mk_ctx(turns: list[tuple[str, str]]) -> PipelineContext:
    ctx = PipelineContext(initial_prompt="start")
    for role, text in turns:
        ctx.conversation_history.append(ConversationTurn(role=ConversationRole(role), content=text))
    return ctx


def test_processor_injects_into_string_prompt() -> None:
    ctx = _mk_ctx([("user", "Hello"), ("assistant", "Hi!")])
    proc = ConversationHistoryPromptProcessor(
        history_manager=HistoryManager(
            HistoryStrategyConfig(strategy="truncate_turns", max_turns=10)
        )
    )

    out = asyncio.run(proc.process("Task: do X", context=ctx))
    assert isinstance(out, str)
    # Compare in lowercase to avoid case-sensitivity issues
    assert "user: hello" in out.lower()
    assert "assistant: hi!" in out.lower()
    assert out.strip().endswith("Task: do X")


def test_processor_injects_into_dict_prompt() -> None:
    ctx = _mk_ctx([("user", "Question?"), ("assistant", "Answer.")])
    proc = ConversationHistoryPromptProcessor()
    data = {"prompt": "Original"}
    out = asyncio.run(proc.process(data, context=ctx))
    assert isinstance(out, dict)
    assert "question?" in out["prompt"].lower()
    assert out["prompt"].endswith("Original")


def test_processor_no_history_passthrough() -> None:
    ctx = _mk_ctx([])
    proc = ConversationHistoryPromptProcessor()
    s = "Hello"
    out = asyncio.run(proc.process(s, context=ctx))
    assert out == s


def test_processor_filters_tool_artifacts_injection() -> None:
    ctx = _mk_ctx(
        [
            ("assistant", '{"tool_call": "x", "arguments": {}}'),
            ("user", "plain message"),
        ]
    )
    proc = ConversationHistoryPromptProcessor()
    out = asyncio.run(proc.process("Do it", context=ctx))
    assert "tool_call" not in out
    assert "plain message" in out


def test_processor_calls_redaction(monkeypatch) -> None:
    called = {"v": False}

    import flujo.processors.conversation as mod

    def _fake_redact(text: str, max_length: int = 2000):  # type: ignore[override]
        called["v"] = True
        return text

    monkeypatch.setattr(mod, "summarize_and_redact_prompt", _fake_redact)
    ctx = _mk_ctx([("user", "hello")])
    proc = ConversationHistoryPromptProcessor()
    _ = asyncio.run(proc.process("Hi", context=ctx))
    assert called["v"] is True
