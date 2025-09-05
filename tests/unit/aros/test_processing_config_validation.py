from __future__ import annotations

import pytest

from flujo.domain.blueprint.loader import (
    BlueprintPipelineModel,
    build_pipeline_from_blueprint,
    BlueprintError,
)


def _make_minimal_pipeline(step_overrides: dict) -> dict:
    base = {
        "version": "0.1",
        "steps": [
            {
                "kind": "step",
                "name": "s1",
                # Use a simple importable function for agent
                "agent": "builtins.print",
            }
        ],
    }
    base["steps"][0].update(step_overrides)
    return base


@pytest.mark.fast
def test_invalid_structured_output_value_raises():
    model = BlueprintPipelineModel.model_validate(
        _make_minimal_pipeline({"processing": {"structured_output": "invalid-mode"}})
    )
    with pytest.raises(BlueprintError):
        build_pipeline_from_blueprint(model)


@pytest.mark.fast
def test_invalid_coercion_allow_raises():
    bad = {"processing": {"aop": "full", "coercion": {"allow": {"boolean": ["str->int"]}}}}
    model = BlueprintPipelineModel.model_validate(_make_minimal_pipeline(bad))
    with pytest.raises(BlueprintError):
        build_pipeline_from_blueprint(model)


@pytest.mark.fast
def test_valid_reasoning_precheck_passes():
    good = {
        "processing": {
            "reasoning_precheck": {
                "enabled": True,
                "required_context_keys": ["initial_input"],
                "inject_feedback": "prepend",
                "consensus_samples": 2,
                "consensus_threshold": 0.7,
            }
        }
    }
    model = BlueprintPipelineModel.model_validate(_make_minimal_pipeline(good))
    # Should not raise
    build_pipeline_from_blueprint(model)


@pytest.mark.fast
def test_schema_alias_preserved_in_processing_meta():
    good = {
        "processing": {
            "structured_output": "openai_json",
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        }
    }
    model = BlueprintPipelineModel.model_validate(_make_minimal_pipeline(good))
    p = build_pipeline_from_blueprint(model)
    # Extract step and check meta
    step = p.steps[0]
    proc = getattr(step, "meta", {}).get("processing")
    assert isinstance(proc, dict)
    assert "schema" in proc and isinstance(proc["schema"], dict)
