from __future__ import annotations

import pytest

from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.exceptions import ConfigurationError


def _build_nested_hitl_pipeline() -> Pipeline:
    hitl = HumanInTheLoopStep(name="ask_user", message_for_user="Confirm?")
    branch = Pipeline.from_step(hitl)
    conditional = ConditionalStep(
        name="gate",
        condition_callable=lambda *_: "yes",
        branches={"yes": branch},
    )
    loop_body = Pipeline.from_step(conditional)
    loop = LoopStep(
        name="outer_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda *_: True,
    )
    return Pipeline.from_step(loop)


def test_hitl_nested_in_conditional_inside_loop_raises() -> None:
    pipeline = _build_nested_hitl_pipeline()
    with pytest.raises(ConfigurationError) as excinfo:
        pipeline.validate_graph(raise_on_error=True)
    assert "HITL" in str(excinfo.value)


def test_hitl_nested_in_conditional_inside_loop_reports_validation_error() -> None:
    pipeline = _build_nested_hitl_pipeline()
    report = pipeline.validate_graph()
    assert not report.is_valid
    rule_ids = [finding.rule_id for finding in report.errors]
    assert "HITL-NESTED-001" in rule_ids
