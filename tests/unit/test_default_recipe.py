"""Tests for flujo.recipes.default module."""

import pytest
import warnings
from typing import Any

from flujo.recipes.default import Default
from flujo.domain.models import Task, Candidate, Checklist, ChecklistItem
from flujo.testing.utils import StubAgent


class MockReviewAgent:
    """Mock review agent that returns a checklist."""

    def __init__(self, checklist: Checklist):
        self.checklist = checklist

    async def run(self, data: Any, **kwargs: Any) -> Any:
        """Return the checklist."""
        return type("MockResult", (), {"output": self.checklist})()


class MockSolutionAgent:
    """Mock solution agent that returns a solution."""

    def __init__(self, solution: str):
        self.solution = solution

    async def run(self, data: Any, **kwargs: Any) -> Any:
        """Return the solution."""
        return type("MockResult", (), {"output": self.solution})()


class MockValidatorAgent:
    """Mock validator agent that returns a validated checklist."""

    def __init__(self, checklist: Checklist):
        self.checklist = checklist

    async def run(self, data: Any, **kwargs: Any) -> Any:
        """Return the validated checklist."""
        return type("MockResult", (), {"output": self.checklist})()


class MockReflectionAgent:
    """Mock reflection agent that returns a reflection."""

    def __init__(self, reflection: str):
        self.reflection = reflection

    async def run(self, data: Any, **kwargs: Any) -> Any:
        """Return the reflection."""
        return type("MockResult", (), {"output": self.reflection})()


def test_default_recipe_deprecation_warning():
    """Test that Default recipe raises deprecation warning."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning, match="The Default class is deprecated"):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        Default(review, solution, validator)


def test_default_recipe_initialization():
    """Test Default recipe initialization."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        assert recipe.flujo_engine is not None


def test_default_recipe_initialization_with_reflection():
    """Test Default recipe initialization with reflection agent."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        reflection = MockReflectionAgent("test reflection")
        recipe = Default(review, solution, validator, reflection_agent=reflection)

        assert recipe.flujo_engine is not None


def test_default_recipe_initialization_with_optional_params():
    """Test Default recipe initialization with optional parameters."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(
            review, solution, validator, max_iters=5, k_variants=3, reflection_limit=10
        )

        assert recipe.flujo_engine is not None


@pytest.mark.asyncio
async def test_default_recipe_run_async():
    """Test Default recipe run_async method."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        task = Task(prompt="test task")
        result = await recipe.run_async(task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None
        assert result.score is not None


@pytest.mark.asyncio
async def test_default_recipe_run_async_with_reflection():
    """Test Default recipe run_async method with reflection."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        reflection = MockReflectionAgent("test reflection")
        recipe = Default(review, solution, validator, reflection_agent=reflection)

        task = Task(prompt="test task")
        result = await recipe.run_async(task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None
        assert result.score is not None


@pytest.mark.asyncio
async def test_default_recipe_run_async_no_solution():
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        # Solution agent that doesn't set solution in context
        solution = StubAgent(["no solution"])
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        task = Task(prompt="test task")
        result = await recipe.run_async(task)
        # Instead of asserting result is None, check the Candidate's solution
        assert isinstance(result, Candidate)
        assert result.solution == "no solution"


@pytest.mark.asyncio
async def test_default_recipe_run_async_no_checklist():
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = StubAgent(["no checklist"])
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        task = Task(prompt="test task")
        result = await recipe.run_async(task)
        # Instead of asserting result is None, check the Candidate's checklist
        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None


def test_default_recipe_run_sync():
    """Test Default recipe run_sync method."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        task = Task(prompt="test task")
        result = recipe.run_sync(task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None
        assert result.score is not None


def test_default_recipe_run_sync_with_reflection():
    """Test Default recipe run_sync method with reflection."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        reflection = MockReflectionAgent("test reflection")
        recipe = Default(review, solution, validator, reflection_agent=reflection)

        task = Task(prompt="test task")
        result = recipe.run_sync(task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None
        assert result.score is not None


@pytest.mark.asyncio
async def test_default_recipe_agent_wrappers():
    """Test that the agent wrappers work correctly."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):
        review = MockReviewAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        solution = MockSolutionAgent("test solution")
        validator = MockValidatorAgent(
            Checklist(
                items=[
                    ChecklistItem(description="item1", passed=True),
                    ChecklistItem(description="item2", passed=True),
                ]
            )
        )
        recipe = Default(review, solution, validator)

        # Test that the pipeline was created with wrapped agents
        assert recipe.flujo_engine is not None
        assert recipe.flujo_engine.pipeline is not None


@pytest.mark.asyncio
async def test_default_recipe_with_callable_agents():
    """Test Default recipe with callable agents instead of objects with run method."""
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning):

        async def review_callable(data: Any, **kwargs: Any) -> Any:
            return type(
                "MockResult",
                (),
                {
                    "output": Checklist(
                        items=[
                            ChecklistItem(description="item1", passed=True),
                            ChecklistItem(description="item2", passed=True),
                        ]
                    )
                },
            )()

        async def solution_callable(data: Any, **kwargs: Any) -> Any:
            return type("MockResult", (), {"output": "test solution"})()

        async def validator_callable(data: Any, **kwargs: Any) -> Any:
            return type(
                "MockResult",
                (),
                {
                    "output": Checklist(
                        items=[
                            ChecklistItem(description="item1", passed=True),
                            ChecklistItem(description="item2", passed=True),
                        ]
                    )
                },
            )()

        recipe = Default(review_callable, solution_callable, validator_callable)

        task = Task(prompt="test task")
        result = await recipe.run_async(task)

        assert isinstance(result, Candidate)
        assert result.solution == "test solution"
        assert result.checklist is not None
        assert result.score is not None
