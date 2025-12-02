from __future__ import annotations

import textwrap
from typing import Any, Optional

from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml


def _get_yaml_loc(step: Any) -> Optional[dict[str, Any]]:
    try:
        meta = getattr(step, "meta", None)
        if isinstance(meta, dict):
            return meta.get("_yaml_loc")
    except Exception:
        return None
    return None


def test_nested_yaml_loc_indexing_state_parallel_map_loop_fallback() -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - kind: StateMachine
            name: SM
            start_state: s1
            end_states: [done]
            states:
              s1:
                - name: S1
                  agent: { id: "flujo.builtins.stringify" }
              done:
                - name: D1
                  agent: { id: "flujo.builtins.stringify" }
          - kind: parallel
            name: P
            branches:
              A:
                - name: A1
                  agent: { id: "flujo.builtins.stringify" }
              B:
                - name: B1
                  agent: { id: "flujo.builtins.stringify" }
          - kind: map
            name: M
            map:
              iterable_input: []
              body:
                - name: MB1
                  agent: { id: "flujo.builtins.stringify" }
          - kind: loop
            name: L
            loop:
              body:
                - name: LB1
                  agent: { id: "flujo.builtins.stringify" }
          - name: FMain
            agent: { id: "flujo.builtins.stringify" }
            fallback:
              name: FBack
              agent: { id: "flujo.builtins.stringify" }
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    steps = list(getattr(pipeline, "steps", []))
    # 0: StateMachine, 1: Parallel, 2: Map, 3: Loop, 4: Fallback main
    sm = steps[0]
    par = steps[1]
    mp = steps[2]
    lp = steps[3]
    fm = steps[4]

    # StateMachine: nested state step should have _yaml_loc
    # SM internal: states.s1.steps[0] (name S1)
    s_states = getattr(sm, "states", {})
    s1_pipe = s_states.get("s1")
    assert s1_pipe is not None
    s1_step = s1_pipe.steps[0]
    assert _get_yaml_loc(s1_step) is not None

    # Parallel branches: A.steps[0], B.steps[0]
    branches = getattr(par, "branches", {})
    a_pipe = branches.get("A")
    b_pipe = branches.get("B")
    assert a_pipe is not None and b_pipe is not None
    assert _get_yaml_loc(a_pipe.steps[0]) is not None
    assert _get_yaml_loc(b_pipe.steps[0]) is not None

    # Map body: map.body.steps[0]
    body = getattr(mp, "original_body_pipeline", None) or getattr(mp, "pipeline_to_run", None)
    assert body is not None and _get_yaml_loc(body.steps[0]) is not None

    # Loop body: loop.body.steps[0]
    lbody = getattr(lp, "loop_body_pipeline", None)
    assert lbody is not None and _get_yaml_loc(lbody.steps[0]) is not None

    # Fallback step: yaml_path ends with .fallback
    fback = getattr(fm, "fallback_step", None)
    assert fback is not None and _get_yaml_loc(fback) is not None


def test_yaml_loc_indexing_conditional_default_branch() -> None:
    yaml_text = textwrap.dedent(
        """
        version: "0.1"
        steps:
          - kind: conditional
            name: C
            branches:
              ok:
                - name: COK
                  agent: { id: "flujo.builtins.stringify" }
            default_branch:
              - name: CDEF
                agent: { id: "flujo.builtins.stringify" }
        """
    )
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    cond = pipeline.steps[0]
    # ok branch step
    bpipe = cond.branches.get("ok")
    assert bpipe is not None and _get_yaml_loc(bpipe.steps[0]) is not None
    # default branch step
    dpipe = cond.default_branch_pipeline
    assert dpipe is not None and _get_yaml_loc(dpipe.steps[0]) is not None
