#!/usr/bin/env python3
"""
Consensus + Diff Auto-Fix Demo

Demonstrates:
- ParallelStep with majority-vote consensus
- DiffProcessor to compute JSON patch deltas
- Applying the patch to auto-correct missing schema fields
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from flujo.application.runner import Flujo
from flujo.domain.consensus import majority_vote
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineContext
from flujo.processors.diff import DiffProcessor
from flujo.testing.utils import gather_result


EXPECTED_SCHEMA = {"name": "", "email": "", "age": 0}


def _unescape_path(segment: str) -> str:
    # Replace ~1 before ~0 to avoid turning "~01" into "/" (RFC 6901).
    return segment.replace("~1", "/").replace("~0", "~")


def _apply_patch(doc: Any, patch_ops: list[dict[str, Any]]) -> Any:
    target = copy.deepcopy(doc)
    for op in patch_ops:
        path = str(op.get("path", "") or "")
        parts = [_unescape_path(p) for p in path.strip("/").split("/") if p]
        parent = target
        for key in parts[:-1]:
            if isinstance(parent, list):
                parent = parent[int(key)]
            else:
                parent = parent.setdefault(key, {})
        final_key = parts[-1] if parts else ""
        if op.get("op") == "remove":
            if isinstance(parent, list):
                parent.pop(int(final_key))
            elif final_key:
                parent.pop(final_key, None)
        elif op.get("op") in {"add", "replace"}:
            if isinstance(parent, list):
                index = int(final_key)
                value = op.get("value")
                if op.get("op") == "add":
                    if index == len(parent):
                        parent.append(value)
                    elif 0 <= index < len(parent):
                        parent.insert(index, value)
                    else:
                        raise IndexError(f"add index out of range: {index}")
                else:
                    parent[index] = value
            elif final_key:
                parent[final_key] = op.get("value")
            else:
                target = op.get("value")
    return target


async def _agent_alpha(_data: object) -> dict[str, Any]:
    return {"name": "Ada Lovelace", "email": "ada@example.com"}


async def _agent_beta(_data: object) -> dict[str, Any]:
    return {"name": "Ada Lovelace", "age": 36}


async def _agent_gamma(_data: object) -> dict[str, Any]:
    return {"name": "Ada Lovelace", "email": "ada@example.com", "age": 36}


async def _auto_fix(candidate: dict[str, Any]) -> dict[str, Any]:
    diff = await DiffProcessor().process({"before": candidate, "after": EXPECTED_SCHEMA})
    # Only apply add ops so schema defaults don't overwrite existing candidate values.
    patch = [op for op in (diff.get("patch") or []) if op.get("op") == "add"]
    fixed = _apply_patch(candidate, patch)
    return {"candidate": candidate, "patch": patch, "fixed": fixed}


async def main() -> None:
    branches = {
        "alpha": Step(name="alpha", agent=_agent_alpha),
        "beta": Step(name="beta", agent=_agent_beta),
        "gamma": Step(name="gamma", agent=_agent_gamma),
    }
    consensus = ParallelStep(name="panel", branches=branches, reduce=majority_vote)
    fix_step = Step(name="auto_fix", agent=_auto_fix)
    pipeline = Pipeline(name="consensus_auto_fix", steps=[consensus, fix_step])

    runner = Flujo(pipeline, context_model=PipelineContext, persist_state=False)
    result = await gather_result(runner, "Generate a contact card")
    output = getattr(result, "output", None)
    if not isinstance(output, dict):
        raise RuntimeError("Expected auto-fix output to be a dict")

    print("Consensus candidate:", output.get("candidate"))
    print("Patch:", output.get("patch"))
    print("Fixed output:", output.get("fixed"))


if __name__ == "__main__":
    asyncio.run(main())
