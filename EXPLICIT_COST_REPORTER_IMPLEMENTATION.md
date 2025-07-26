# ExplicitCostReporter Protocol Implementation

## Overview

This document summarizes the implementation of the `ExplicitCostReporter` protocol, which provides a standardized way for any operation within Flujo to report a direct, pre-calculated cost, completely bypassing the token-based calculation logic.

## Problem Statement

The existing `extract_usage_metrics` function was fundamentally token-centric, designed to find a `.usage()` method and process token counts. While it had a fallback to look for `cost_usd` on the raw output, this was an implicit convention, not a formal part of the framework's design. To robustly support primitives with non-token-based costs (like images, third-party API calls, etc.), we needed a formal, explicit mechanism for a step's output to declare its own cost.

## Solution: ExplicitCostReporter Protocol

### Technical Specification

The `ExplicitCostReporter` protocol is defined in `flujo/cost.py`:

```python
@runtime_checkable
class ExplicitCostReporter(Protocol):
    """A protocol for objects that can report their own pre-calculated cost.

    Attributes
    ----------
    cost_usd : float
        The explicit cost in USD for the operation.
    token_counts : int, optional
        The total token count for the operation, if applicable. If not present, will be treated as 0 by extraction logic.
    """
    cost_usd: float
    token_counts: int  # Optional; if missing, treated as 0
```

### Key Features

1. **Runtime Checkable**: The protocol uses `@runtime_checkable` decorator to enable `isinstance()` checks
2. **Flexible Token Counts**: The `token_counts` attribute defaults to 0 for non-token operations
3. **Graceful Handling**: None values are handled gracefully by defaulting to 0.0 for cost and 0 for tokens

### Priority Order

The `extract_usage_metrics` function now follows this priority order:

1. **HIGHEST PRIORITY**: `ExplicitCostReporter` protocol
2. **SECOND PRIORITY**: `hasattr(raw_output, "cost_usd")` (existing fallback)
3. **THIRD PRIORITY**: `hasattr(raw_output, "usage")` (existing token-based calculation)

## Implementation Details

### Core Changes

1. **Protocol Definition**: Added `ExplicitCostReporter` protocol with `@runtime_checkable` decorator
2. **Priority Logic**: Updated `extract_usage_metrics` to check for protocol first
3. **Graceful Handling**: Added proper handling of None values and edge cases
4. **Logging**: Enhanced logging to indicate when protocol is being used

### Code Changes

```python
# In flujo/cost.py
@runtime_checkable
class ExplicitCostReporter(Protocol):
    """A protocol for objects that can report their own pre-calculated cost.

    Attributes
    ----------
    cost_usd : float
        The explicit cost in USD for the operation.
    token_counts : int, optional
        The total token count for the operation, if applicable. If not present, will be treated as 0 by extraction logic.
    """
    cost_usd: float
    token_counts: int  # Optional; if missing, treated as 0

def extract_usage_metrics(raw_output: Any, agent: Any, step_name: str) -> Tuple[int, int, float]:
    # 1. HIGHEST PRIORITY: Check if the output object reports its own cost.
    # We check for the protocol attributes manually since token_counts is optional
    if hasattr(raw_output, "cost_usd"):
        cost_usd = getattr(raw_output, 'cost_usd', 0.0) or 0.0
        total_tokens = getattr(raw_output, 'token_counts', 0) or 0

        telemetry.logfire.info(f"Using explicit cost from '{type(raw_output).__name__}' for step '{step_name}': cost=${cost_usd}, tokens={total_tokens}")

        # Return prompt_tokens as 0 since it cannot be determined reliably here.
        return 0, total_tokens, cost_usd
```

    # 2. Check for explicit cost first - if cost is provided, trust it and don't attempt token calculation
    if hasattr(raw_output, "cost_usd"):
        # ... existing logic ...

    # 3. If explicit metrics are not fully present, proceed with usage() extraction
    if hasattr(raw_output, "usage"):
        # ... existing logic ...
```

## Testing Strategy

### Unit Tests

Comprehensive unit tests were added to `tests/unit/test_cost_tracking.py`:

1. **Protocol Priority Test**: Ensures protocol takes priority over `.usage()` method
2. **Protocol with Only Cost**: Tests objects with only `cost_usd` attribute
3. **Protocol with Cost and Tokens**: Tests objects with both attributes
4. **Edge Cases**: Tests zero cost, negative cost, None values
5. **Regression Tests**: Ensures existing functionality is preserved

### Integration Tests

New integration tests in `tests/integration/test_explicit_cost_integration.py`:

1. **End-to-End Success Case**: Full pipeline execution with protocol
2. **Usage Limits Integration**: Tests that explicit costs work with usage limits
3. **Multiple Steps**: Tests protocol with multiple pipeline steps
4. **Priority Verification**: Ensures protocol takes priority over usage method
5. **Edge Cases**: Tests None values and other edge cases
6. **Regression Testing**: Ensures existing `.usage()` method still works

## Usage Examples

### Basic Usage

```python
class ImageGenerationResult:
    def __init__(self, cost_usd: float, token_counts: int = 0):
        self.cost_usd = cost_usd
        self.token_counts = token_counts

# This object will automatically be recognized as an ExplicitCostReporter
result = ImageGenerationResult(cost_usd=0.04, token_counts=0)
```

### With Token Counts

```python
class CustomAgentResult:
    def __init__(self, cost_usd: float, token_counts: int):
        self.cost_usd = cost_usd
        self.token_counts = token_counts

# This object will report both cost and token counts
result = CustomAgentResult(cost_usd=0.25, token_counts=1000)
```

### Edge Cases

```python
class EdgeCaseResult:
    def __init__(self):
        self.cost_usd = None  # Will default to 0.0
        self.token_counts = None  # Will default to 0

# None values are handled gracefully
result = EdgeCaseResult()
```

## Benefits

1. **Clean Architecture**: Clear separation between token-based and unit-based costs
2. **Backward Compatibility**: Existing token-based cost tracking continues to work
3. **Flexibility**: Supports any type of cost calculation (images, API calls, etc.)
4. **Type Safety**: Protocol provides type hints and runtime checking
5. **Robustness**: Handles edge cases gracefully
6. **Extensibility**: Easy to extend for future cost types

## Future Enhancements

This implementation provides the foundation for:

1. **Image Generation Costs**: Per-image pricing models
2. **Third-Party API Costs**: External service cost tracking
3. **Custom Cost Models**: User-defined cost calculation logic
4. **Cost Aggregation**: Complex cost models across multiple operations

## Testing Results

All tests pass successfully:

- ✅ Protocol priority over usage() method
- ✅ Protocol with only cost attribute
- ✅ Protocol with cost and token counts
- ✅ Edge cases (None values, zero costs, negative costs)
- ✅ Backward compatibility with existing token-based tracking
- ✅ Integration with usage limits
- ✅ Multiple step scenarios
- ✅ Regression testing

## Conclusion

The `ExplicitCostReporter` protocol successfully establishes a clear, standardized protocol for any operation within Flujo to report a direct, pre-calculated cost. This creates a clean architectural separation between token-based costs (which the framework calculates) and unit-based costs (which the primitive itself calculates).

The implementation is robust, well-tested, and maintains full backward compatibility while providing the foundation for future unit-based costing features.
