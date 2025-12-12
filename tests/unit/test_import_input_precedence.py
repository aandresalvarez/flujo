from __future__ import annotations
from flujo.type_definitions.common import JSONObject

from typing import Any, Optional

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.import_step import ImportStep, OutputMapping
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext


def _echo_agent(payload: Any, *, context: Optional[Any] = None) -> Any:
    # Return payload as-is (used inside child pipeline)
    return payload


def _echo_to_import_artifacts(payload: Any, *, context: Optional[Any] = None) -> JSONObject:
    # Map the incoming payload into import_artifacts.echo via updates_context
    return {"import_artifacts": {"echo": payload}}


def test_import_explicit_input_precedence_uses_import_artifacts_key() -> None:
    # Child: single step that echoes its input
    child = Pipeline(
        steps=[
            Step(name="echo_child", agent=_echo_agent, updates_context=False),
        ]
    )

    # Parent: step 1 produces a status string; import step should ignore that
    status_step = Step(name="status", agent=lambda _in: "✅ Definition ready (sub-pipeline)")
    imp = ImportStep(
        name="run_child_import",
        pipeline=child,
        inherit_context=True,
        inherit_conversation=True,
        input_to="initial_prompt",
        # default scalar input key = "initial_input"
        updates_context=False,
    )
    parent = Pipeline(steps=[status_step, imp])

    # Seed explicit artifact in import_artifacts.initial_input; this must take precedence
    explicit_payload = {"cohort_definition": "Influenza A, ICD-10 J10"}
    runner = Flujo(
        pipeline=parent,
        context_model=PipelineContext,
        initial_context_data={"import_artifacts": {"initial_input": explicit_payload}},
        enable_tracing=False,
    )
    result = runner.run(initial_input="ignored")

    # Parent import step result should reflect that the child saw our explicit payload
    assert result.step_history[-1].name == "run_child_import"
    # When updates_context=False, parent_sr.output mirrors last child step output
    # input_to="initial_prompt" routes stringified JSON into the child
    assert result.step_history[-1].output == (
        "{" + '"cohort_definition": "Influenza A, ICD-10 J10"' + "}"
    )
    # Metadata exposes trace of where initial input was resolved from
    md = result.step_history[-1].metadata_
    assert isinstance(md, dict)
    assert (
        md.get("import.initial_input_resolved", {}).get("origin")
        == "import_artifacts:initial_input"
    )


def test_import_outputs_mapping_repeated_imports_with_both_projection() -> None:
    # Child: map input into import_artifacts.echo (so parent outputs mapping can pull it)
    child = Pipeline(
        steps=[
            Step(name="echo_to_ctx", agent=_echo_to_import_artifacts, updates_context=True),
        ]
    )

    # Two imports in sequence, with a setter step in between to change the explicit artifact
    first_imp = ImportStep(
        name="child_imp_1",
        pipeline=child,
        inherit_context=True,
        inherit_conversation=True,
        input_to="both",  # seed initial_prompt and import_artifacts
        outputs=[OutputMapping(child="import_artifacts.echo", parent="child_echo")],
        updates_context=True,
    )

    second_value = {"cohort_definition": "Influenza B, SNOMED 6142004"}

    def _set_second(_: Any, *, context: Optional[Any] = None) -> JSONObject:
        # Overwrite the explicit artifact key between imports
        return {"import_artifacts": {"initial_input": second_value}}

    set_second = Step(name="set_second_input", agent=_set_second, updates_context=True)

    second_imp = ImportStep(
        name="child_imp_2",
        pipeline=child,
        inherit_context=True,
        inherit_conversation=True,
        input_to="both",
        outputs=[OutputMapping(child="import_artifacts.echo", parent="child_echo")],
        updates_context=True,
    )

    parent = Pipeline(steps=[first_imp, set_second, second_imp])

    runner = Flujo(
        pipeline=parent,
        context_model=PipelineContext,
        initial_context_data={
            "import_artifacts": {"initial_input": {"cohort_definition": "Influenza, unspecified"}}
        },
        enable_tracing=False,
    )
    result = runner.run(initial_input="ignored")

    # Verify that outputs mapping captured the echo from the second import
    final_ctx = result.final_pipeline_context
    assert final_ctx is not None
    # With input_to="both", the child sees stringified JSON in its echo artifacts
    assert final_ctx.import_artifacts["child_echo"] == (
        "{" + '"cohort_definition": "Influenza B, SNOMED 6142004"' + "}"
    )
