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
    def __init__(self, output: str, usage: Usage) -> None:
        self.output = output
        self._usage = usage

    def usage(self) -> Usage:
        return self._usage


class ResponseWithUsageAttribute:
    def __init__(self, output: str, usage: Usage) -> None:
        self.output = output
        self.usage = usage


def _assert_usage_values(result: FlujoAgentResult, expected: Usage) -> None:
    usage = result.usage()
    assert usage is not None
    assert usage.input_tokens == expected.input_tokens
    assert usage.output_tokens == expected.output_tokens
    assert usage.cost_usd == expected.cost_usd


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
