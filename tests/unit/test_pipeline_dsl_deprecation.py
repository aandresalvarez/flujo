"""Test that pipeline_dsl imports trigger deprecation warnings."""

import warnings


def test_pipeline_dsl_deprecation_warning():
    """Test that importing from pipeline_dsl triggers a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This import should trigger a deprecation warning
        from flujo.domain.pipeline_dsl import (
            StepConfig,
            Step,
            Pipeline,
            LoopStep,
            ConditionalStep,
            ParallelStep,
            MapStep,
            MergeStrategy,
            BranchFailureStrategy,
            BranchKey,
            step,
            adapter_step,
            mapper,
            HumanInTheLoopStep,
        )

        # Use the imports to verify they work and trigger warnings
        assert StepConfig is not None
        assert Step is not None
        assert Pipeline is not None
        assert LoopStep is not None
        assert ConditionalStep is not None
        assert ParallelStep is not None
        assert MapStep is not None
        assert MergeStrategy is not None
        assert BranchFailureStrategy is not None
        assert BranchKey is not None
        assert step is not None
        assert adapter_step is not None
        assert mapper is not None
        assert HumanInTheLoopStep is not None

        # Check that we got at least one deprecation warning
        assert len(w) > 0
        assert any("deprecated" in str(warning.message).lower() for warning in w)
