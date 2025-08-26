from __future__ import annotations


from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml


def _make_loop_with_set(target: str) -> any:
    # Build YAML without f-strings to preserve {{ }} placeholders
    yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: L
    loop:
      body:
        - kind: step
          name: s
      max_loops: 1
      state:
        set:
          - target: "TARGET_HERE"
            value: "{{ previous_step }}"
"""
    yaml_text = yaml_text.replace("TARGET_HERE", target)
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    return pipeline.steps[0]


def test_resolve_target_on_plain_dict_creates_nested_levels() -> None:
    step = _make_loop_with_set("context.level1.level2")
    it = step.iteration_input_mapper
    ctx: dict[str, object] = {}
    assert it("X", ctx, 1) == "X"
    assert ctx == {"level1": {"level2": "X"}}


def test_attribute_fallback_on_plain_object_creates_dict_then_sets_key() -> None:
    class C:
        pass

    step = _make_loop_with_set("context.foo.bar")
    it = step.iteration_input_mapper
    ctx = C()
    assert it("Y", ctx, 1) == "Y"
    assert hasattr(ctx, "foo")
    assert isinstance(getattr(ctx, "foo"), dict)
    assert ctx.foo["bar"] == "Y"


def test_slots_object_is_unchanged() -> None:
    class S:
        __slots__ = ()

    step = _make_loop_with_set("context.a.b")
    it = step.iteration_input_mapper
    ctx = S()
    assert it("Z", ctx, 1) == "Z"
    assert not hasattr(ctx, "a")


def test_tuple_context_is_unchanged() -> None:
    ctx = tuple()
    step = _make_loop_with_set("context.a")
    it = step.iteration_input_mapper
    assert it("V", ctx, 1) == "V"
    # tuples cannot have attributes; just ensure we didn't error


def test_mixed_dict_then_attribute_path() -> None:
    class P:
        pass

    inner = P()
    ctx = {"obj": inner}
    step = _make_loop_with_set("context.obj.meta.prop")
    it = step.iteration_input_mapper
    assert it("W", ctx, 1) == "W"
    assert isinstance(ctx["obj"].__dict__.get("meta"), dict)
    assert ctx["obj"].__dict__["meta"]["prop"] == "W"


def test_dict_assignment_failure_is_ignored() -> None:
    class ReadOnly(dict):
        def __setitem__(self, key, value):  # type: ignore[override]
            raise RuntimeError("read-only")

    ctx = {"holder": ReadOnly()}
    step = _make_loop_with_set("context.holder.newkey")
    it = step.iteration_input_mapper
    # Implementation swallows dict assignment errors; ensure no exception and no mutation
    assert it("Q", ctx, 1) == "Q"
    assert "newkey" not in ctx["holder"]
