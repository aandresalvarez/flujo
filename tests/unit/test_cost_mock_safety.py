import types

from unittest.mock import Mock

from flujo.cost import extract_usage_metrics


class DummyAgent:
    pass


def test_extract_usage_metrics_with_mock_explicit_cost_is_zero() -> None:
    raw_output = Mock()
    # cost_usd and token_counts are mocks by default
    prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
        raw_output=raw_output, agent=DummyAgent(), step_name="test-step"
    )

    assert prompt_tokens == 0
    assert completion_tokens == 0
    assert cost_usd == 0.0


def test_extract_usage_metrics_with_numeric_explicit_cost() -> None:
    raw_output = types.SimpleNamespace(cost_usd=1.23, token_counts=42)
    prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
        raw_output=raw_output, agent=DummyAgent(), step_name="test-step"
    )

    assert prompt_tokens == 0
    assert completion_tokens == 42
    assert cost_usd == 1.23


def test_extract_usage_metrics_usage_with_mocks_yields_zero() -> None:
    usage_obj = Mock()
    usage_obj.request_tokens = Mock()  # will be treated as 0
    usage_obj.response_tokens = Mock()  # will be treated as 0

    raw_output = Mock()
    raw_output.usage = Mock(return_value=usage_obj)

    prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
        raw_output=raw_output, agent=DummyAgent(), step_name="test-step"
    )

    assert prompt_tokens == 0
    assert completion_tokens == 0
    assert cost_usd == 0.0
