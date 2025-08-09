from typing import Any, Optional, List, Tuple
from flujo.domain.plugins import PluginOutcome, ValidationPlugin
from flujo.domain.validation import Validator


async def run_hybrid_check(
    data: Any,
    plugins: List[Tuple[ValidationPlugin, int]],
    validators: List[Validator],
    context: Optional[Any] = None,
    resources: Optional[Any] = None,
) -> Tuple[Any, Optional[str]]:
    """
    Run plugins then validators in sequence and return a tuple:
      (possibly-transformed data, aggregated failure feedback or None).

    PluginOutcome failures are collected into feedback; ValidationResult failures
    are collected; combined with "; " between them.
    Plugin exceptions raise PluginError.
    """
    # 1. Plugins
    output = data
    plugin_feedbacks: List[str] = []
    for plugin, priority in sorted(plugins, key=lambda x: x[1], reverse=True):
        plugin_kwargs = {}
        # decide whether to pass context/resources (import helpers accordingly)
        try:
            result = await plugin.validate(output, **plugin_kwargs)
        except Exception as e:
            raise ValueError(str(e))
        if isinstance(result, PluginOutcome):
            if not result.success:
                plugin_feedbacks.append(result.feedback)
            else:
                if result.new_solution is not None:
                    output = result.new_solution
        else:
            output = result
    # 2. Validators
    if validators:
        from flujo.domain.validation import BaseValidator
        from flujo.domain.validation import ValidationResult
        failed_msgs: List[str] = []
        for validator in validators:
            try:
                vr: ValidationResult = await validator.validate(output, context=context)
            except Exception as e:
                failed_msgs.append(f"{validator.name}: {e}")
                continue
            if not vr.is_valid:
                failed_msgs.append(f"{vr.validator_name}: {vr.feedback}")
        if failed_msgs:
            parts = plugin_feedbacks + failed_msgs
            return output, "; ".join(parts)
    return output, None