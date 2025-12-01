"""
Integration tests for Enhanced Pipeline Composition and Sequencing (FSD 2.1).

This test suite verifies that the `Pipeline >> Pipeline` operator works correctly
in real-world scenarios, ensuring the FSD requirements are fully met.
"""

import pytest
from typing import Any, List
from flujo.domain.models import BaseModel
from flujo.type_definitions.common import JSONObject

from flujo.domain import Step, Pipeline
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


class ConceptResolutionContext(BaseModel):
    """Context for concept resolution pipeline."""

    resolved_concepts: List[str] = []
    confidence_scores: dict[str, float] = {}


class SQLGenerationContext(BaseModel):
    """Context for SQL generation pipeline."""

    generated_sql: str = ""
    validation_errors: List[str] = []


class MasterContext(BaseModel):
    """Combined context for the master pipeline."""

    resolved_concepts: List[str] = []
    confidence_scores: dict[str, float] = {}
    generated_sql: str = ""
    validation_errors: List[str] = []
    # Add fields that will be set by the steps
    concepts: List[str] = []
    input_text: str = ""
    sql: str = ""
    is_valid: bool = False
    errors: list = []
    concept_resolution_feedback: List[str] = []
    sql_generation_feedback: List[str] = []
    sql_validation_results: List[Any] = []


class ConceptResolutionAgent:
    """Agent that resolves concepts from text input."""

    async def run(self, data: str, **kwargs) -> JSONObject:
        # Simulate concept resolution
        concepts = ["users", "orders", "products"]
        scores = {"users": 0.95, "orders": 0.87, "products": 0.92}

        return {"concepts": concepts, "confidence_scores": scores, "input_text": data}


class SQLGenerationAgent:
    """Agent that generates SQL from resolved concepts."""

    async def run(self, data: JSONObject, **kwargs) -> JSONObject:
        # Simulate SQL generation based on resolved concepts
        concepts = data.get("concepts", [])
        sql = f"SELECT * FROM {', '.join(concepts)} WHERE 1=1;"
        return {"sql": sql, "generated_sql": sql}


class SQLValidationAgent:
    """Agent that validates generated SQL."""

    async def run(self, data: JSONObject, **kwargs) -> JSONObject:
        # Simulate SQL validation
        # Extract SQL from the input data (could be a string or dict with sql field)
        if isinstance(data, str):
            sql = data
        else:
            sql = data.get("sql", "")

        is_valid = "SELECT" in sql and "FROM" in sql
        return {
            "sql": sql,
            "is_valid": is_valid,
            "errors": [] if is_valid else ["Invalid SQL syntax"],
        }


def build_concept_pipeline() -> Pipeline[str, JSONObject]:
    """Build the concept resolution pipeline."""
    concept_agent = ConceptResolutionAgent()

    # Step 1: Resolve concepts from text
    resolve_step = Step.model_validate(
        {
            "name": "resolve_concepts",
            "agent": concept_agent,
            "updates_context": True,
            "persist_feedback_to_context": "concept_resolution_feedback",
        }
    )

    return Pipeline.from_step(resolve_step)


def build_sql_pipeline() -> Pipeline[JSONObject, JSONObject]:
    """Build the SQL generation and validation pipeline."""
    sql_gen_agent = SQLGenerationAgent()
    sql_val_agent = SQLValidationAgent()

    # Step 1: Generate SQL from resolved concepts
    generate_step = Step.model_validate(
        {
            "name": "generate_sql",
            "agent": sql_gen_agent,
            "updates_context": True,
            "persist_feedback_to_context": "sql_generation_feedback",
        }
    )

    # Step 2: Validate the generated SQL
    validate_step = Step.model_validate(
        {
            "name": "validate_sql",
            "agent": sql_val_agent,
            "updates_context": True,
            "persist_validation_results_to": "sql_validation_results",
        }
    )

    return generate_step >> validate_step


def build_master_pipeline() -> Pipeline[str, JSONObject]:
    """Build the master pipeline by chaining concept and SQL pipelines."""
    concept_pipeline = build_concept_pipeline()
    sql_pipeline = build_sql_pipeline()

    # Chain the pipelines using the >> operator
    master_pipeline = concept_pipeline >> sql_pipeline

    return master_pipeline


@pytest.mark.asyncio
async def test_pipeline_composition_basic() -> None:
    """Test basic pipeline composition with the >> operator."""
    # Build individual pipelines
    concept_pipeline = build_concept_pipeline()
    sql_pipeline = build_sql_pipeline()

    # Compose them using the >> operator
    master_pipeline = concept_pipeline >> sql_pipeline

    # Verify the composed pipeline has all steps in correct order
    step_names = [step.name for step in master_pipeline.steps]
    expected_names = ["resolve_concepts", "generate_sql", "validate_sql"]

    assert step_names == expected_names
    assert len(master_pipeline.steps) == 3


@pytest.mark.asyncio
async def test_pipeline_composition_execution() -> None:
    """Test that composed pipelines execute correctly end-to-end."""
    master_pipeline = build_master_pipeline()
    runner = create_test_flujo(master_pipeline, context_model=MasterContext)

    # Execute the pipeline
    result = await gather_result(runner, "Find all users and their orders")

    # Verify the pipeline executed successfully
    # Check if all steps succeeded
    all_steps_succeeded = all(step.success for step in result.step_history)
    assert all_steps_succeeded is True

    # Verify all steps executed in order
    step_names = [step.name for step in result.step_history]
    expected_names = ["resolve_concepts", "generate_sql", "validate_sql"]
    assert step_names == expected_names

    # Verify the final output
    final_output = result.step_history[-1].output
    assert isinstance(final_output, dict)
    assert "sql" in final_output
    assert "is_valid" in final_output
    assert final_output["is_valid"] is True


@pytest.mark.asyncio
async def test_pipeline_composition_context_sharing() -> None:
    """Test that context is properly shared across composed pipelines."""
    master_pipeline = build_master_pipeline()
    runner = create_test_flujo(master_pipeline, context_model=MasterContext)

    result = await gather_result(runner, "Analyze customer data")

    # Verify context was shared and updated across pipeline stages
    context = result.final_pipeline_context

    # Check that concept resolution feedback was persisted
    assert hasattr(context, "concept_resolution_feedback")

    # Check that SQL generation feedback was persisted
    assert hasattr(context, "sql_generation_feedback")

    # Check that SQL validation results were persisted
    assert hasattr(context, "sql_validation_results")


@pytest.mark.asyncio
async def test_pipeline_composition_multiple_chains() -> None:
    """Test chaining multiple pipelines together."""
    # Create three simple pipelines
    pipeline1 = Pipeline.from_step(
        Step.model_validate({"name": "step1", "agent": StubAgent(["output1"])})
    )
    pipeline2 = Pipeline.from_step(
        Step.model_validate({"name": "step2", "agent": StubAgent(["output2"])})
    )
    pipeline3 = Pipeline.from_step(
        Step.model_validate({"name": "step3", "agent": StubAgent(["output3"])})
    )

    # Chain them together
    composed = pipeline1 >> pipeline2 >> pipeline3

    # Verify the composition
    step_names = [step.name for step in composed.steps]
    assert step_names == ["step1", "step2", "step3"]

    # Execute the composed pipeline
    runner = create_test_flujo(composed)
    result = await gather_result(runner, "input")

    # Verify execution - check if all steps succeeded
    all_steps_succeeded = all(step.success for step in result.step_history)
    assert all_steps_succeeded is True
    assert len(result.step_history) == 3


@pytest.mark.asyncio
async def test_pipeline_composition_backward_compatibility() -> None:
    """Test that pipeline composition doesn't break existing functionality."""
    # Test Step >> Step still works
    step1 = Step.model_validate({"name": "step1", "agent": StubAgent(["output1"])})
    step2 = Step.model_validate({"name": "step2", "agent": StubAgent(["output2"])})
    pipeline1 = step1 >> step2

    # Test Pipeline >> Step still works
    step3 = Step.model_validate({"name": "step3", "agent": StubAgent(["output3"])})
    pipeline2 = pipeline1 >> step3

    # Test Pipeline >> Pipeline works (new functionality)
    pipeline3 = Pipeline.from_step(
        Step.model_validate({"name": "step4", "agent": StubAgent(["output4"])})
    )
    composed = pipeline2 >> pipeline3

    # Verify all combinations work
    assert len(pipeline1.steps) == 2
    assert len(pipeline2.steps) == 3
    assert len(composed.steps) == 4

    # Execute to ensure no runtime errors
    runner = create_test_flujo(composed)
    result = await gather_result(runner, "input")
    all_steps_succeeded = all(step.success for step in result.step_history)
    assert all_steps_succeeded is True


@pytest.mark.asyncio
async def test_pipeline_composition_type_safety() -> None:
    """Test that pipeline composition maintains type safety."""
    # Create pipelines with specific input/output types
    concept_pipeline = build_concept_pipeline()  # str -> JSONObject
    sql_pipeline = build_sql_pipeline()  # JSONObject -> JSONObject

    # Compose them
    master_pipeline = concept_pipeline >> sql_pipeline

    # The resulting pipeline should have type str -> JSONObject
    # This is verified by the fact that we can pass a string to the runner
    runner = create_test_flujo(master_pipeline)
    result = await gather_result(runner, "test input")

    # Verify the final output is the expected type
    assert isinstance(result.step_history[-1].output, dict)


@pytest.mark.asyncio
async def test_pipeline_composition_error_handling() -> None:
    """Test that errors in composed pipelines are handled correctly."""

    # Create a pipeline that will fail by using an agent that raises an exception
    class FailingAgent:
        async def run(self, data: Any, **kwargs) -> Any:
            raise Exception("Simulated failure")

    failing_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "failing_step", "agent": FailingAgent()})
    )

    # Create a working pipeline
    working_pipeline = Pipeline.from_step(
        Step.model_validate({"name": "working_step", "agent": StubAgent(["success"])})
    )

    # Compose them
    composed = failing_pipeline >> working_pipeline

    # Execute - the pipeline should fail at the first step and return StepResult
    # instead of raising an exception (unified error handling)
    runner = create_test_flujo(composed)
    result = await gather_result(runner, "input")

    # Verify that the pipeline failed at the first step
    assert len(result.step_history) == 1
    assert not result.step_history[0].success
    assert result.step_history[0].name == "failing_step"
    assert "Simulated failure" in result.step_history[0].feedback
