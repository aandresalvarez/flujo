from __future__ import annotations

import json
from typing import Any

import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result


pytestmark = [pytest.mark.slow]


def _parse_candidate(prompt: str) -> dict[str, Any]:
    marker = "Candidate:"
    if marker not in prompt:
        return {}
    candidate_line = prompt.split(marker, 1)[1].strip().splitlines()[0]
    try:
        return json.loads(candidate_line)
    except Exception:
        return {}


def _make_root_state(numbers: list[int]) -> dict[str, Any]:
    return {"nums": [{"value": num, "expr": str(num)} for num in numbers]}


def _expand_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    nums = state.get("nums") or []
    entries = [item for item in nums if isinstance(item, dict)]
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a = entries[i]
            b = entries[j]
            rest = [entries[idx] for idx in range(len(entries)) if idx not in (i, j)]
            aval = int(a.get("value"))
            bval = int(b.get("value"))
            aexpr = str(a.get("expr"))
            bexpr = str(b.get("expr"))

            candidates: list[tuple[int, str]] = [
                (aval + bval, f"({aexpr}+{bexpr})"),
                (aval * bval, f"({aexpr}*{bexpr})"),
                (aval - bval, f"({aexpr}-{bexpr})"),
                (bval - aval, f"({bexpr}-{aexpr})"),
            ]
            if bval != 0 and aval % bval == 0:
                candidates.append((aval // bval, f"({aexpr}/{bexpr})"))
            if aval != 0 and bval % aval == 0:
                candidates.append((bval // aval, f"({bexpr}/{aexpr})"))

            for value, expr in candidates:
                new_nums = rest + [{"value": int(value), "expr": expr}]
                signature = tuple(sorted(item["value"] for item in new_nums))
                if signature in seen:
                    continue
                seen.add(signature)
                results.append({"nums": new_nums})
    return results


async def _proposer(prompt: str) -> list[dict[str, Any]]:
    state = _parse_candidate(prompt)
    if not state:
        return []
    return _expand_state(state)


async def _evaluator(prompt: str) -> float:
    state = _parse_candidate(prompt)
    nums = state.get("nums") or []
    if len(nums) == 1:
        try:
            value = int(nums[0].get("value"))
        except Exception:
            return 0.0
        return 1.0 if value == 24 else 0.0
    try:
        best = min(abs(int(item.get("value")) - 24) for item in nums)
    except Exception:
        return 0.0
    score = max(0.0, 1.0 - min(1.0, best / 24))
    if best == 0:
        score = min(score, 0.9)
    return score


@pytest.mark.asyncio
async def test_tree_search_game_of_24_end_to_end() -> None:
    step = TreeSearchStep(
        name="game_of_24",
        proposer=_proposer,
        evaluator=_evaluator,
        branching_factor=50,
        beam_width=50,
        max_depth=3,
        max_iterations=200,
        goal_score_threshold=1.0,
        require_goal=True,
    )
    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    start_state = _make_root_state([4, 4, 6, 8])
    result = await gather_result(
        runner,
        start_state,
        initial_context_data={"initial_prompt": "Use all numbers to make 24"},
    )
    output = getattr(result, "final_output", None)
    if output is None:
        output = getattr(result, "output", None)
    if output is None and hasattr(result, "step_result"):
        output = result.step_result.output

    assert isinstance(output, dict)
    nums = output.get("nums")
    assert isinstance(nums, list)
    assert len(nums) == 1
    assert nums[0].get("value") == 24
    assert isinstance(nums[0].get("expr"), str)
