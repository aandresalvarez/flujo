# Reasoning with A* (TreeSearchStep)

`TreeSearchStep` lets you explore multiple solution paths with durable, resumable A* search. This is useful for problems that benefit from backtracking (e.g., logic puzzles, planning tasks).

## Example: Game of 24

The goal: use all numbers to make 24. The proposer expands numeric states, and the evaluator scores how close a state is to 24 (only returning `1.0` when a full solution is found).

```python
import asyncio
import json
from typing import Any

from flujo.application.runner import Flujo
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.models import PipelineContext
from flujo.testing.utils import gather_result


def _parse_candidate(prompt: str) -> dict[str, Any]:
    marker = "Candidate:"
    if marker not in prompt:
        return {}
    candidate_line = prompt.split(marker, 1)[1].strip().splitlines()[0]
    return json.loads(candidate_line)


def _expand_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    nums = state.get("nums") or []
    entries = [item for item in nums if isinstance(item, dict)]
    results: list[dict[str, Any]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a = entries[i]
            b = entries[j]
            rest = [entries[idx] for idx in range(len(entries)) if idx not in (i, j)]
            aval = int(a["value"])
            bval = int(b["value"])
            aexpr = str(a["expr"])
            bexpr = str(b["expr"])
            candidates = [
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
                results.append({"nums": rest + [{"value": int(value), "expr": expr}]})
    return results


async def proposer(prompt: str) -> list[dict[str, Any]]:
    return _expand_state(_parse_candidate(prompt))


async def evaluator(prompt: str) -> float:
    state = _parse_candidate(prompt)
    nums = state.get("nums") or []
    if len(nums) == 1:
        return 1.0 if int(nums[0]["value"]) == 24 else 0.0
    best = min(abs(int(item["value"]) - 24) for item in nums)
    score = max(0.0, 1.0 - min(1.0, best / 24))
    return min(score, 0.9)


async def main() -> None:
    step = TreeSearchStep(
        name="game_of_24",
        proposer=proposer,
        evaluator=evaluator,
        branching_factor=50,
        beam_width=50,
        max_depth=3,
        goal_score_threshold=1.0,
        require_goal=True,
    )
    runner = Flujo(step, context_model=PipelineContext, persist_state=False)
    start_state = {"nums": [{"value": n, "expr": str(n)} for n in [4, 4, 6, 8]]}
    result = await gather_result(
        runner,
        start_state,
        initial_context_data={"initial_prompt": "Use all numbers to make 24"},
    )
    print("Solution:", result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

## Notes

- Keep candidate payloads JSON-serializable; they are persisted in `context.tree_search_state`.
- Use `candidate_validator` to prune invalid candidates cheaply.
- Set `goal_score_threshold=1.0` and `require_goal=True` when only exact matches should succeed.

## Heuristic Tuning & Distillation

After a run, you can analyze the search trace to tighten your evaluator rubric or distill the winning path into a reusable prompt. The tuning helper defaults to the `shadow_eval.judge_model` when configured.

```python
from flujo.application.tree_search_improvement import (
    build_tree_search_trace_report,
    distill_tree_search_path,
    tune_tree_search_evaluator,
)

# state is a SearchState captured from context.tree_search_state
report = build_tree_search_trace_report(state, objective="Use all numbers to make 24")
improvements = await tune_tree_search_evaluator(report)
distilled_prompt = await distill_tree_search_path(state, objective="Use all numbers to make 24")
```
