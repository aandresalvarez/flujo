"""Contract tests for Flujo agent adapters.

Adapters listed in ADAPTER_CASES must pass these shared behaviors so usage
metrics and outputs are consistent across backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
from unittest.mock import AsyncMock, MagicMock

import pytest

from flujo.agents.adapters.pydantic_ai_adapter import PydanticAIAdapter
from flujo.agents.agent_like import AgentLike
from flujo.domain.agent_result import FlujoAgentResult


class AdapterUnderTest(Protocol):
    """Minimal adapter interface for contract tests."""

    async def run(self, *args: object, **kwargs: object) -> FlujoAgentResult: ...


AdapterFactory = Callable[[AgentLike], AdapterUnderTest]


@dataclass(frozen=True)
class AdapterCase:
    name: str
    factory: AdapterFactory


ADAPTER_CASES: tuple[AdapterCase, ...] = (
    AdapterCase(name="pydantic_ai", factory=PydanticAIAdapter),
)


class Usage:
    def __init__(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


class ResponseWithUsageMethod:
    def __init__(self, output: str, usage: Usage | None) -> None:
        self.output = output
        self._usage = usage

    def usage(self) -> Usage | None:
        return self._usage


class ResponseWithUsageAttribute:
    def __init__(self, output: str, usage: Usage | None) -> None:
        self.output = output
        self.usage = usage


class ResponseWithCostData:
    def __init__(self, output: str, cost_usd: float, token_counts: int) -> None:
        self.output = output
        self.cost_usd = cost_usd
        self.token_counts = token_counts


def _assert_usage_values(result: FlujoAgentResult, expected: Usage) -> None:
    usage = result.usage()
    assert usage is not None
    assert usage.input_tokens == expected.input_tokens
    assert usage.output_tokens == expected.output_tokens
    assert usage.cost_usd == expected.cost_usd


def _assert_usage_none(result: FlujoAgentResult) -> None:
    assert result.usage() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_usage_method(case: AdapterCase) -> None:
    """Adapters must adapt usage when the response exposes usage() method."""
    usage = Usage(input_tokens=12, output_tokens=7, cost_usd=0.004)
    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithUsageMethod("ok", usage))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_values(result, usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_usage_attribute(case: AdapterCase) -> None:
    """Adapters must adapt usage when the response exposes usage attribute."""
    usage = Usage(input_tokens=12, output_tokens=7, cost_usd=0.004)
    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithUsageAttribute("ok", usage))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_values(result, usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_usage_method_none(case: AdapterCase) -> None:
    """Adapters must return None when usage() yields None."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithUsageMethod("ok", None))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_none(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_usage_attribute_none(case: AdapterCase) -> None:
    """Adapters must return None when usage attribute is None."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithUsageAttribute("ok", None))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_none(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_missing_usage(case: AdapterCase) -> None:
    """Adapters must return None when no usage data is present."""

    class ResponseWithoutUsage:
        def __init__(self, output: str) -> None:
            self.output = output

    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithoutUsage("ok"))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_none(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ADAPTER_CASES, ids=lambda case: case.name)
async def test_adapter_contract_explicit_cost_data(case: AdapterCase) -> None:
    """Adapters must propagate explicit cost and token counts when provided."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=ResponseWithCostData("ok", 0.12, 42))

    adapter = case.factory(agent)
    result = await adapter.run("prompt")

    assert isinstance(result, FlujoAgentResult)
    assert result.output == "ok"
    _assert_usage_none(result)
    assert result.cost_usd == 0.12
    assert result.token_counts == 42
