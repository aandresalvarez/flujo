from __future__ import annotations


from flujo.infra.budget_resolver import combine_limits
from flujo.domain.models import UsageLimits
from flujo.domain.dsl import Step, Pipeline, MergeStrategy


def _make_simple_step(name: str):
    async def fn(x: int) -> int:
        return x

    return Step.from_callable(fn, name=name)


def _make_str_step(name: str):
    async def fn(x: str) -> str:
        return x

    return Step.from_callable(fn, name=name)


def _make_object_consumer(name: str):
    async def fn(x: object) -> None:
        return None

    return Step.from_callable(fn, name=name)


def test_validate_graph_detects_merge_conflict() -> None:
    # Build two branches; context_include_keys hints 'shared' field is updated
    a = _make_simple_step("a")
    b = _make_simple_step("b")
    pipe_a = Pipeline.from_step(a)
    pipe_b = Pipeline.from_step(b)

    # Create a parallel step by using class directly to set fields
    from flujo.domain.dsl.parallel import ParallelStep

    parallel = ParallelStep(
        name="p",
        branches={"A": pipe_a, "B": pipe_b},
        context_include_keys=["shared"],
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=None,
    )
    pipeline = Pipeline(steps=[parallel])

    report = pipeline.validate_graph()
    assert any(f.rule_id == "V-P1" for f in report.errors)


def test_unbound_output_warning() -> None:
    # First step returns str; second step accepts generic object, not consuming output meaningfully
    s1 = _make_str_step("producer")
    s2 = _make_object_consumer("consumer")
    pipeline = Pipeline.from_step(s1) >> s2
    report = pipeline.validate_graph()
    assert any(f.rule_id == "V-A5" for f in report.warnings)


def test_fallback_incompatible_signature() -> None:
    # Primary expects str; fallback expects int
    primary = _make_str_step("primary")

    async def fb(x: int) -> int:
        return x

    fallback = Step.from_callable(fb, name="fallback")
    primary.fallback_step = fallback
    pipeline = Pipeline.from_step(primary)
    report = pipeline.validate_graph()
    assert any(f.rule_id == "V-F1" for f in report.errors)


def test_combine_limits_applies_min_semantics_grouped() -> None:
    a = UsageLimits(total_cost_usd_limit=10.0, total_tokens_limit=None)
    b = UsageLimits(total_cost_usd_limit=None, total_tokens_limit=500)
    c = combine_limits(a, b)
    assert c.total_cost_usd_limit == 10.0
    assert c.total_tokens_limit == 500

    d = UsageLimits(total_cost_usd_limit=2.0, total_tokens_limit=100)
    e = combine_limits(c, d)
    assert e.total_cost_usd_limit == 2.0
    assert e.total_tokens_limit == 100
