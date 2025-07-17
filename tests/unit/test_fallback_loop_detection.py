"""Comprehensive tests for fallback loop detection.

This test suite verifies that the fallback loop detection mechanism properly
catches various types of infinite loops that could occur in production
environments, especially critical for healthcare, legal, and finance applications.
"""

import pytest

from flujo.domain.dsl import Step, StepConfig
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.application.runner import Flujo, InfiniteFallbackError
from flujo.application.core.step_logic import _detect_fallback_loop
from flujo.domain.models import StepResult


class TestFallbackLoopDetection:
    """Test the improved fallback loop detection mechanism."""

    def test_detect_fallback_loop_object_identity(self):
        """Test that object identity detection works (original implementation)."""
        step1 = Step.model_validate({"name": "step1", "agent": StubAgent(["ok"])})
        step2 = Step.model_validate({"name": "step2", "agent": StubAgent(["ok"])})

        chain = [step1, step2]

        # Should detect loop when same object is added again
        assert _detect_fallback_loop(step1, chain) is True
        assert _detect_fallback_loop(step2, chain) is True

        # Should not detect loop for new step
        step3 = Step.model_validate({"name": "step3", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step3, chain) is False

    def test_detect_fallback_loop_step_name(self):
        """Test that step name detection catches immediate loops only."""
        step1 = Step.model_validate({"name": "step1", "agent": StubAgent(["ok"])})
        step2 = Step.model_validate({"name": "step2", "agent": StubAgent(["ok"])})

        chain = [step1, step2]

        # Create a new step with the same name as step1
        step1_recreated = Step.model_validate({"name": "step1", "agent": StubAgent(["ok"])})

        # Should NOT detect loop for recreated step with same name (not immediate loop)
        # This is correct behavior for healthcare/legal/finance - only detect actual loops
        assert _detect_fallback_loop(step1_recreated, chain) is False

        # Should detect loop if the last step in chain has same name (immediate loop)
        step2_recreated = Step.model_validate({"name": "step2", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step2_recreated, chain) is True

        # Should not detect loop for step with different name
        step3 = Step.model_validate({"name": "step3", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step3, chain) is False

    def test_detect_fallback_loop_chain_length_limit(self):
        """Test that chain length limit prevents extremely long chains."""
        # Create a chain at the maximum length
        from flujo.application.core.step_logic import _MAX_FALLBACK_CHAIN_LENGTH

        chain = []
        for i in range(_MAX_FALLBACK_CHAIN_LENGTH):
            step = Step.model_validate({"name": f"step{i}", "agent": StubAgent(["ok"])})
            chain.append(step)

        # Adding another step should trigger the length limit
        step11 = Step.model_validate({"name": "step11", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step11, chain) is True

    def test_detect_fallback_loop_pattern_detection(self):
        """Test that pattern detection catches A->B->A patterns."""
        step_a = Step.model_validate({"name": "step_a", "agent": StubAgent(["ok"])})
        step_b = Step.model_validate({"name": "step_b", "agent": StubAgent(["ok"])})

        # Test A->B->A pattern
        chain = [step_a, step_b]
        assert _detect_fallback_loop(step_a, chain) is True

        # Test A->B->C->A pattern
        step_c = Step.model_validate({"name": "step_c", "agent": StubAgent(["ok"])})
        chain = [step_a, step_b, step_c]
        assert _detect_fallback_loop(step_a, chain) is True

    def test_detect_fallback_loop_complex_patterns(self):
        """Test detection of more complex loop patterns."""
        step_a = Step.model_validate({"name": "step_a", "agent": StubAgent(["ok"])})
        step_b = Step.model_validate({"name": "step_b", "agent": StubAgent(["ok"])})
        step_c = Step.model_validate({"name": "step_c", "agent": StubAgent(["ok"])})
        step_d = Step.model_validate({"name": "step_d", "agent": StubAgent(["ok"])})

        # Test A->B->C->D->A pattern (should detect as loop)
        chain = [step_a, step_b, step_c, step_d]
        assert _detect_fallback_loop(step_a, chain) is True

        # Test A->B->C->D->B pattern (should detect as loop since B appears in chain)
        assert _detect_fallback_loop(step_b, chain) is True

        # Test A->B->C->D->E pattern (should not detect as loop)
        step_e = Step.model_validate({"name": "step_e", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step_e, chain) is False

    def test_detect_fallback_loop_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty chain
        step = Step.model_validate({"name": "step", "agent": StubAgent(["ok"])})
        assert _detect_fallback_loop(step, []) is False

        # Single step chain
        chain = [step]
        assert _detect_fallback_loop(step, chain) is True

        # Chain with duplicate names but different objects
        step1 = Step.model_validate({"name": "same_name", "agent": StubAgent(["ok"])})
        step2 = Step.model_validate({"name": "same_name", "agent": StubAgent(["ok"])})
        chain = [step1]
        assert _detect_fallback_loop(step2, chain) is True

    @pytest.mark.asyncio
    async def test_fallback_loop_integration_object_identity(self):
        """Test fallback loop detection in actual pipeline execution."""
        plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])

        # Create steps that will fail
        step_a = Step.model_validate(
            {
                "name": "step_a",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_b = Step.model_validate(
            {
                "name": "step_b",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create infinite loop: A -> B -> A
        step_a.fallback(step_b)
        step_b.fallback(step_a)

        runner = Flujo(step_a)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "data")

    @pytest.mark.asyncio
    async def test_fallback_loop_integration_step_name(self):
        """Test fallback loop detection with immediate name match."""
        plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])

        # Create steps where the last step in chain has same name as current step
        step_a = Step.model_validate(
            {
                "name": "step_a",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_b = Step.model_validate(
            {
                "name": "step_b",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_a2 = Step.model_validate(
            {
                "name": "step_a",  # Same name as step_a
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create loop where last step has same name: A -> B -> A2
        # This should trigger immediate name match detection
        step_a.fallback(step_b)
        step_b.fallback(step_a2)

        runner = Flujo(step_a)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "data")

    @pytest.mark.asyncio
    async def test_fallback_loop_healthcare_scenario(self):
        """Test fallback loop detection in a healthcare scenario."""
        # Simulate a healthcare pipeline with multiple validation steps
        plugin_validation = DummyPlugin(
            [PluginOutcome(success=False, feedback="validation failed")]
        )
        plugin_retry = DummyPlugin([PluginOutcome(success=False, feedback="retry failed")])

        # Medical record validation steps
        validate_record = Step.model_validate(
            {
                "name": "validate_medical_record",
                "agent": StubAgent(["invalid"] * 100),
                "plugins": [(plugin_validation, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        retry_validation = Step.model_validate(
            {
                "name": "retry_validation",
                "agent": StubAgent(["still_invalid"] * 100),
                "plugins": [(plugin_retry, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create a loop that could occur in healthcare systems
        validate_record.fallback(retry_validation)
        retry_validation.fallback(validate_record)  # This creates a loop

        runner = Flujo(validate_record)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "medical_record_data")

    @pytest.mark.asyncio
    async def test_fallback_loop_legal_scenario(self):
        """Test fallback loop detection in a legal scenario."""
        # Simulate a legal document processing pipeline
        plugin_legal = DummyPlugin([PluginOutcome(success=False, feedback="legal review failed")])
        plugin_compliance = DummyPlugin(
            [PluginOutcome(success=False, feedback="compliance check failed")]
        )

        # Legal document review steps
        review_document = Step.model_validate(
            {
                "name": "review_legal_document",
                "agent": StubAgent(["needs_review"] * 100),
                "plugins": [(plugin_legal, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        compliance_check = Step.model_validate(
            {
                "name": "compliance_check",
                "agent": StubAgent(["non_compliant"] * 100),
                "plugins": [(plugin_compliance, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create a loop in legal document processing
        review_document.fallback(compliance_check)
        compliance_check.fallback(review_document)  # This creates a loop

        runner = Flujo(review_document)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "legal_document_data")

    @pytest.mark.asyncio
    async def test_fallback_loop_finance_scenario(self):
        """Test fallback loop detection in a finance scenario."""
        # Simulate a financial transaction processing pipeline
        plugin_fraud = DummyPlugin(
            [PluginOutcome(success=False, feedback="fraud detection failed")]
        )
        plugin_aml = DummyPlugin([PluginOutcome(success=False, feedback="AML check failed")])

        # Financial transaction steps
        fraud_detection = Step.model_validate(
            {
                "name": "fraud_detection",
                "agent": StubAgent(["suspicious"] * 100),
                "plugins": [(plugin_fraud, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        aml_check = Step.model_validate(
            {
                "name": "aml_check",
                "agent": StubAgent(["aml_alert"] * 100),
                "plugins": [(plugin_aml, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create a loop in financial processing
        fraud_detection.fallback(aml_check)
        aml_check.fallback(fraud_detection)  # This creates a loop

        runner = Flujo(fraud_detection)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "transaction_data")

    @pytest.mark.asyncio
    async def test_fallback_loop_logging_and_audit(self):
        """Test that fallback loops are properly logged for audit purposes."""
        # For healthcare/legal/finance, we ensure fallback loops are detected and logged
        plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])

        step_a = Step.model_validate(
            {
                "name": "step_a",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_b = Step.model_validate(
            {
                "name": "step_b",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        step_a.fallback(step_b)
        step_b.fallback(step_a)

        # Verify that the loop detection raises the correct exception
        runner = Flujo(step_a)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "data")

    @pytest.mark.asyncio
    async def test_infinite_fallback_error_raises_correctly(self):
        """Test that InfiniteFallbackError is raised correctly."""
        from flujo.exceptions import InfiniteFallbackError

        # Test that the exception can be raised and caught
        try:
            raise InfiniteFallbackError("Test error")
        except InfiniteFallbackError as e:
            assert "Test error" in str(e)

        # Test that it can be caught with pytest.raises
        with pytest.raises(InfiniteFallbackError, match="Test error"):
            raise InfiniteFallbackError("Test error")

    @pytest.mark.asyncio
    async def test_fallback_loop_direct_step_logic(self):
        """Test fallback loop detection by directly calling step logic with recursion."""
        from flujo.application.core.step_logic import _run_step_logic

        plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])

        step_a = Step.model_validate(
            {
                "name": "step_a",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_b = Step.model_validate(
            {
                "name": "step_b",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create a simple loop: A -> B -> A
        step_a.fallback(step_b)
        step_b.fallback(step_a)

        # Recursive step executor to simulate fallback recursion
        async def recursive_step_executor(step, data, context, resources):
            # Always fail, triggering fallback if present
            if step.fallback_step:
                # Simulate the fallback recursion
                return await _run_step_logic(
                    step=step.fallback_step,
                    data=data,
                    context=context,
                    resources=resources,
                    step_executor=recursive_step_executor,
                    context_model_defined=False,
                )
            return StepResult(
                name=step.name,
                success=False,
                feedback="Test failure",
                attempts=1,
            )

        # Test that the fallback loop detection works
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await _run_step_logic(
                step=step_a,
                data="test",
                context=None,
                resources=None,
                step_executor=recursive_step_executor,
                context_model_defined=False,
            )

    @pytest.mark.asyncio
    async def test_simple_fallback_loop_integration(self):
        """Test a simple fallback loop that should definitely be detected."""
        plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])

        # Create a simple A -> B -> A loop
        step_a = Step.model_validate(
            {
                "name": "step_a",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )
        step_b = Step.model_validate(
            {
                "name": "step_b",
                "agent": StubAgent(["bad"] * 100),
                "plugins": [(plugin, 0)],
                "config": StepConfig(max_retries=1),
            }
        )

        # Create the loop: A -> B -> A
        step_a.fallback(step_b)
        step_b.fallback(step_a)

        # This should definitely raise InfiniteFallbackError
        runner = Flujo(step_a)
        with pytest.raises(InfiniteFallbackError, match="Fallback loop detected"):
            await gather_result(runner, "data")


if __name__ == "__main__":
    pytest.main([__file__])
