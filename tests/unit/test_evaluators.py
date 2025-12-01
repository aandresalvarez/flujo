"""Tests for flujo.application.evaluators module."""

from flujo.application.evaluators import FinalSolutionEvaluator
from flujo.domain.models import PipelineResult
from tests.test_types.fixtures import create_test_step_result


class TestFinalSolutionEvaluator:
    """Test FinalSolutionEvaluator functionality."""

    def test_evaluate_with_matching_output(self):
        """Test evaluate method with matching expected output."""

        # Create a mock context with expected output
        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        # Create a pipeline result with step history
        step_result = create_test_step_result(
            name="test_step",
            output="expected_result",
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step_result]

        context = MockContext(pipeline_result, "expected_result")
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is True

    def test_evaluate_with_non_matching_output(self):
        """Test evaluate method with non-matching expected output."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        step_result = create_test_step_result(
            name="test_step",
            output="actual_result",
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step_result]

        context = MockContext(pipeline_result, "expected_result")
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is False

    def test_evaluate_with_empty_step_history(self):
        """Test evaluate method with empty step history."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        pipeline_result = PipelineResult()
        pipeline_result.step_history = []

        context = MockContext(pipeline_result, "expected_result")
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is False  # final_output is None, so it doesn't match

    def test_evaluate_with_none_expected_output(self):
        """Test evaluate method with None expected output."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        step_result = create_test_step_result(
            name="test_step",
            output=None,
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step_result]

        context = MockContext(pipeline_result, None)
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is True  # None matches None

    def test_evaluate_with_multiple_steps(self):
        """Test evaluate method with multiple steps in history."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        step1 = create_test_step_result(
            name="step1",
            output="first_result",
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        step2 = create_test_step_result(
            name="step2",
            output="final_result",
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step1, step2]

        context = MockContext(pipeline_result, "final_result")
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is True  # Should use the last step's output

    def test_evaluate_with_different_types(self):
        """Test evaluate method with different output types."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        # Test with integer output
        step_result = create_test_step_result(
            name="test_step",
            output=42,
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step_result]

        context = MockContext(pipeline_result, 42)
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is True

        # Test with list output
        step_result2 = create_test_step_result(
            name="test_step",
            output=[1, 2, 3],
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result2 = PipelineResult()
        pipeline_result2.step_history = [step_result2]

        context2 = MockContext(pipeline_result2, [1, 2, 3])
        evaluator2 = FinalSolutionEvaluator()

        result2 = evaluator2.evaluate(context2)
        assert result2 is True

    def test_evaluate_with_complex_objects(self):
        """Test evaluate method with complex object outputs."""

        class MockContext:
            def __init__(self, output, expected_output):
                self.output = output
                self.expected_output = expected_output

        class ComplexObject:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                if isinstance(other, ComplexObject):
                    return self.value == other.value
                return False

        complex_output = ComplexObject("test_value")
        expected_complex = ComplexObject("test_value")

        step_result = create_test_step_result(
            name="test_step",
            output=complex_output,
            success=True,
            attempts=1,
            latency_s=0.1,
            token_counts=0,
            cost_usd=0.0,
        )

        pipeline_result = PipelineResult()
        pipeline_result.step_history = [step_result]

        context = MockContext(pipeline_result, expected_complex)
        evaluator = FinalSolutionEvaluator()

        result = evaluator.evaluate(context)
        assert result is True
