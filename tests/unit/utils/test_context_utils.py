"""Tests for contexto utils: get_context_field_safely."""

from flujo.utils.context import get_context_field_safely


class Dummy:
    def __init__(self) -> None:
        self.exists = 42


def test_get_context_field_safely_returns_default_when_context_none():
    assert get_context_field_safely(None, "anything", default="fallback") == "fallback"


def test_get_context_field_safely_returns_existing_field():
    d = Dummy()
    assert get_context_field_safely(d, "exists", default="fallback") == 42


def test_get_context_field_safely_returns_default_when_missing():
    d = Dummy()
    assert get_context_field_safely(d, "missing", default=0) == 0


def test_get_context_field_safely_handles_attribute_error_and_returns_default():
    # Create an object where hasattr says True but getattr raises AttributeError
    class Tricky:
        @property
        def bad(self):  # type: ignore[override]
            raise AttributeError("boom")

    t = Tricky()
    # hasattr(t, 'bad') is True for property
    assert get_context_field_safely(t, "bad", default="fallback") == "fallback"
