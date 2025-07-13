import pytest
from typing import Any
from unittest.mock import Mock

from flujo.domain import Step, Pipeline
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.dsl.step import MergeStrategy, BranchFailureStrategy


class TestDynamicParallelRouterStep:
    """Test suite for DynamicParallelRouterStep."""

    def test_dynamic_parallel_router_step_init_validation_empty_branches(self) -> None:
        """Test that empty branches raises ValueError."""
        with pytest.raises(ValueError, match="'branches' dictionary cannot be empty"):
            DynamicParallelRouterStep.model_validate(
                {
                    "name": "router",
                    "router_agent": Mock(),
                    "branches": {},
                }
            )

    def test_dynamic_parallel_router_step_init_with_step_branches(self) -> None:
        """Test initialization with Step objects as branches."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert len(step.branches) == 1
        assert "branch1" in step.branches
        # Should convert Step to Pipeline
        assert isinstance(step.branches["branch1"], Pipeline)

    def test_dynamic_parallel_router_step_init_with_pipeline_branches(self) -> None:
        """Test initialization with Pipeline objects as branches."""
        mock_agent = Mock()
        pipeline = Pipeline.from_step(Step.model_validate({"name": "step1"}))
        branches = {"branch1": pipeline}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert len(step.branches) == 1
        assert "branch1" in step.branches
        assert step.branches["branch1"] == pipeline

    def test_dynamic_parallel_router_step_with_context_include_keys(self) -> None:
        """Test initialization with context_include_keys."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
                "context_include_keys": ["key1", "key2"],
            }
        )

        assert step.context_include_keys == ["key1", "key2"]

    def test_dynamic_parallel_router_step_with_merge_strategy(self) -> None:
        """Test initialization with custom merge strategy."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
                "merge_strategy": MergeStrategy.OVERWRITE,
            }
        )

        assert step.merge_strategy == MergeStrategy.OVERWRITE

    def test_dynamic_parallel_router_step_with_branch_failure_strategy(self) -> None:
        """Test initialization with branch failure strategy."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
                "on_branch_failure": BranchFailureStrategy.IGNORE,
            }
        )

        assert step.on_branch_failure == BranchFailureStrategy.IGNORE

    def test_dynamic_parallel_router_step_with_custom_merge_function(self) -> None:
        """Test initialization with custom merge function."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        def custom_merge(ctx1: Any, ctx2: Any) -> None:
            pass

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
                "merge_strategy": custom_merge,
            }
        )

        assert step.merge_strategy == custom_merge

    def test_dynamic_parallel_router_step_repr(self) -> None:
        """Test the __repr__ method."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        expected = "DynamicParallelRouterStep(name='router', branches=['branch1'])"
        assert repr(step) == expected

    def test_dynamic_parallel_router_step_factory_method(self) -> None:
        """Test the Step.dynamic_parallel_branch factory method."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = Step.dynamic_parallel_branch(
            name="router",
            router_agent=mock_agent,
            branches=branches,
        )

        assert isinstance(step, DynamicParallelRouterStep)
        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert len(step.branches) == 1
        assert "branch1" in step.branches

    def test_dynamic_parallel_router_step_factory_method_with_all_options(self) -> None:
        """Test the factory method with all optional parameters."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        def custom_merge(ctx1: Any, ctx2: Any) -> None:
            pass

        step = Step.dynamic_parallel_branch(
            name="router",
            router_agent=mock_agent,
            branches=branches,
            context_include_keys=["key1", "key2"],
            merge_strategy=custom_merge,
            on_branch_failure=BranchFailureStrategy.IGNORE,
        )

        assert isinstance(step, DynamicParallelRouterStep)
        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert step.context_include_keys == ["key1", "key2"]
        assert step.merge_strategy == custom_merge
        assert step.on_branch_failure == BranchFailureStrategy.IGNORE

    def test_dynamic_parallel_router_step_model_validate_with_dict_args(self) -> None:
        """Test model_validate with dictionary as first argument."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert len(step.branches) == 1
        assert "branch1" in step.branches

    def test_dynamic_parallel_router_step_model_validate_with_kwargs(self) -> None:
        """Test model_validate with keyword arguments."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        # model_validate doesn't accept kwargs directly, so we pass a dict
        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        assert step.name == "router"
        assert step.router_agent == mock_agent
        assert len(step.branches) == 1
        assert "branch1" in step.branches

    def test_dynamic_parallel_router_step_default_values(self) -> None:
        """Test that default values are set correctly."""
        mock_agent = Mock()
        branches = {"branch1": Step.model_validate({"name": "step1"})}

        step = DynamicParallelRouterStep.model_validate(
            {
                "name": "router",
                "router_agent": mock_agent,
                "branches": branches,
            }
        )

        assert step.context_include_keys is None
        assert step.merge_strategy == MergeStrategy.NO_MERGE
        assert step.on_branch_failure == BranchFailureStrategy.PROPAGATE
