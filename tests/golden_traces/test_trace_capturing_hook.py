from __future__ import annotations


import pytest

from tests.golden_traces.utils import TraceCapturingHook, trees_equal


def _simple_payloads():
    class PreRun:
        def __init__(self) -> None:
            self.event_name = "pre_run"
            self.initial_input = {"x": 1}
            self.context = None
            self.resources = None
            self.run_id = None
            self.pipeline_name = None
            self.pipeline_version = None

    class StepObj:
        def __init__(self) -> None:
            self.name = "s"

    class PreStep:
        def __init__(self) -> None:
            self.event_name = "pre_step"
            self.step = StepObj()
            self.step_input = "x"
            self.context = None
            self.resources = None
            self.attempt_number = 1
            self.cache_hit = False

    class PostStep:
        def __init__(self) -> None:
            self.event_name = "post_step"

            class _SR:
                def __init__(self) -> None:
                    self.name = "s"
                    self.success = True
                    self.latency_s = 0.01
                    self.cost_usd = 0.0
                    self.token_counts = 0

            self.step_result = _SR()
            self.context = None
            self.resources = None

    return PreRun(), PreStep(), PostStep()


@pytest.mark.asyncio
async def test_trace_capturing_hook_builds_contract_tree() -> None:
    hook = TraceCapturingHook()
    pre_run, pre_step, post_step = _simple_payloads()
    await hook.hook(pre_run)
    await hook.hook(pre_step)
    await hook.hook(post_step)

    tree = hook.get_contract_tree()
    assert tree is not None
    # Minimal structural expectations:
    assert tree["name"] == "pipeline_run"
    assert any(a in tree["attributes"] for a in ("flujo.input",))
    assert len(tree["children"]) == 1
    child = tree["children"][0]
    assert child["name"] == "s"
    # Required step attributes per helper
    assert "flujo.step.type" in child["attributes"]
    assert "flujo.step.policy" in child["attributes"]
    # Round-trip normalize should be stable
    assert trees_equal(tree, tree)
