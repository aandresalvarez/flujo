# TODO: Enhanced Context Merging for Parallel Steps

## Overview

During the parallel step execution deadlock fix, we simplified the context merging logic to prioritize reliability over sophistication. This document tracks the trade-off and proposes a future enhancement.

## Current Implementation

### Simplified Context Merging
```python
# Current: Simple field copying
for field_name in dir(branch_ctx):
    if not field_name.startswith('_') and hasattr(branch_ctx, field_name):
        setattr(context, field_name, getattr(branch_ctx, field_name))
```

### What Was Removed

#### 1. Deep Dictionary Merging
```python
# Previous: Deep merge
if isinstance(current_value, dict) and isinstance(actual_source_value, dict):
    merged_value = deep_merge_dict(current_value, actual_source_value)
    setattr(target_context, field_name, merged_value)
```

#### 2. Smart List Merging (Deduplication)
```python
# Previous: Smart list merge
elif isinstance(current_value, list) and isinstance(actual_source_value, list):
    new_items = [item for item in actual_source_value if item not in current_value]
    if new_items:
        current_value.extend(new_items)
```

#### 3. Counter Field Accumulation
```python
# Previous: Counter accumulation
counter_field_names = {"accumulated_value", "iteration_count", "counter", "count"}
for field_name in counter_field_names:
    if hasattr(branch_ctx, field_name) and hasattr(context, field_name):
        branch_value = getattr(branch_ctx, field_name)
        current_value = getattr(context, field_name)
        if isinstance(branch_value, (int, float)) and isinstance(current_value, (int, float)):
            accumulated_values[field_name] = current_value + branch_value
```

## Impact Analysis

### High Impact Scenarios

#### 1. Complex State Accumulation
```python
# This will break with simplified merging
context.analysis = {"sentiment": {"positive": 5, "negative": 2}}
# Branch 1 adds: {"sentiment": {"positive": 3}}
# Branch 2 adds: {"sentiment": {"negative": 1}}
# Result: Only last branch's sentiment is kept
```

#### 2. Counter-Based Workflows
```python
# This will break with simplified merging
context.processed_items = 10
# Branch 1 processes 5 items
# Branch 2 processes 3 items
# Result: Only 3 items counted (not 18)
```

#### 3. List Accumulation
```python
# This will break with simplified merging
context.collected_data = [1, 2, 3]
# Branch 1 adds: [3, 4, 5]
# Branch 2 adds: [5, 6, 7]
# Result: Only [5, 6, 7] kept (lost [1, 2, 3, 4])
```

### Low Impact Scenarios

1. **Simple Field Updates**: Most basic context updates work fine
2. **Independent Branch Results**: If branches don't share state
3. **Final Result Aggregation**: If you only care about final outputs

## Current Workarounds

### 1. Custom Merge Strategy
```python
def custom_merge_strategy(context, branch_results):
    """Implement sophisticated merging logic."""
    for branch_result in branch_results.values():
        branch_ctx = getattr(branch_result, "branch_context", None)
        if branch_ctx:
            # Implement your own deep merging logic
            if hasattr(branch_ctx, "data") and hasattr(context, "data"):
                # Deep merge dictionaries
                if isinstance(context.data, dict) and isinstance(branch_ctx.data, dict):
                    context.data = deep_merge_dict(context.data, branch_ctx.data)
            
            # Accumulate counters
            if hasattr(branch_ctx, "counter") and hasattr(context, "counter"):
                context.counter += branch_ctx.counter

# Use in parallel step
parallel_step = Step.parallel(
    branches={"branch1": pipeline1, "branch2": pipeline2},
    merge_strategy=custom_merge_strategy
)
```

### 2. Pre-Merge Preparation
```python
# Structure context to avoid conflicts
class WorkflowContext(BaseModel):
    branch1_data: Dict[str, Any] = {}
    branch2_data: Dict[str, Any] = {}
    shared_counters: Dict[str, int] = {}
    
    def accumulate_counters(self):
        """Manual counter accumulation."""
        total = sum(self.shared_counters.values())
        return total
```

### 3. Post-Merge Processing
```python
# Process results after parallel execution
def post_process_context(context, branch_results):
    """Manual post-processing of context."""
    # Accumulate counters
    total_processed = 0
    for branch_result in branch_results.values():
        if hasattr(branch_result, "processed_count"):
            total_processed += branch_result.processed_count
    context.total_processed = total_processed
```

## Proposed Implementation

### Enhanced Context Merging Strategy

#### 1. Configurable Merge Behavior
```python
class MergeConfig:
    """Configuration for context merging behavior."""
    
    def __init__(
        self,
        deep_merge_dicts: bool = True,
        deduplicate_lists: bool = True,
        accumulate_counters: bool = True,
        counter_fields: Set[str] = None,
        excluded_fields: Set[str] = None
    ):
        self.deep_merge_dicts = deep_merge_dicts
        self.deduplicate_lists = deduplicate_lists
        self.accumulate_counters = accumulate_counters
        self.counter_fields = counter_fields or {"counter", "count", "total", "processed"}
        self.excluded_fields = excluded_fields or set()
```

#### 2. Enhanced Merge Function
```python
def enhanced_merge_context(
    target_context: Any,
    source_context: Any,
    config: MergeConfig
) -> bool:
    """
    Enhanced context merging with configurable behavior.
    
    Args:
        target_context: Target context to merge into
        source_context: Source context to merge from
        config: Merge configuration
        
    Returns:
        True if merge was successful, False otherwise
    """
    if target_context is None or source_context is None:
        return False
    
    try:
        for field_name in dir(source_context):
            if field_name.startswith('_') or field_name in config.excluded_fields:
                continue
                
            if not hasattr(target_context, field_name):
                continue
                
            source_value = getattr(source_context, field_name)
            target_value = getattr(target_context, field_name)
            
            # Deep dictionary merging
            if (config.deep_merge_dicts and 
                isinstance(target_value, dict) and 
                isinstance(source_value, dict)):
                merged_dict = deep_merge_dict(target_value, source_value)
                setattr(target_context, field_name, merged_dict)
                
            # List deduplication
            elif (config.deduplicate_lists and 
                  isinstance(target_value, list) and 
                  isinstance(source_value, list)):
                new_items = [item for item in source_value if item not in target_value]
                if new_items:
                    target_value.extend(new_items)
                    
            # Counter accumulation
            elif (config.accumulate_counters and 
                  field_name in config.counter_fields and
                  isinstance(target_value, (int, float)) and 
                  isinstance(source_value, (int, float))):
                setattr(target_context, field_name, target_value + source_value)
                
            # Simple replacement for other types
            else:
                setattr(target_context, field_name, source_value)
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge context: {e}")
        return False
```

#### 3. Integration with Parallel Steps
```python
# In _handle_parallel_step method
def _handle_parallel_step(self, parallel_step, ...):
    # ... existing logic ...
    
    # Enhanced context merging
    if context is not None and parallel_step.merge_strategy != MergeStrategy.NO_MERGE:
        merge_config = getattr(parallel_step, "merge_config", MergeConfig())
        
        for branch_result in branch_results.values():
            branch_ctx = getattr(branch_result, "branch_context", None)
            if branch_ctx is not None:
                enhanced_merge_context(context, branch_ctx, merge_config)
```

#### 4. Merge Strategy Options
```python
class MergeStrategy(Enum):
    NO_MERGE = "no_merge"
    SIMPLE = "simple"  # Current implementation
    ENHANCED = "enhanced"  # New implementation
    CUSTOM = "custom"  # User-defined function
```

## Implementation Plan

### Phase 1: Core Enhancement
1. **Implement `MergeConfig` class**
2. **Create `enhanced_merge_context` function**
3. **Add unit tests for all merge scenarios**
4. **Update parallel step logic to use enhanced merging**

### Phase 2: Integration
1. **Add merge configuration to `ParallelStep`**
2. **Update DSL to support merge configuration**
3. **Add documentation and examples**
4. **Performance testing and optimization**

### Phase 3: Advanced Features
1. **Custom merge rules per field**
2. **Merge validation and error handling**
3. **Merge performance monitoring**
4. **Backward compatibility layer**

## Testing Strategy

### Unit Tests
```python
def test_deep_dict_merging():
    """Test deep dictionary merging."""
    context = TestContext(data={"a": {"b": 1, "c": 2}})
    branch_ctx = TestContext(data={"a": {"b": 3, "d": 4}})
    
    config = MergeConfig(deep_merge_dicts=True)
    enhanced_merge_context(context, branch_ctx, config)
    
    assert context.data == {"a": {"b": 3, "c": 2, "d": 4}}

def test_counter_accumulation():
    """Test counter field accumulation."""
    context = TestContext(counter=5, processed=10)
    branch_ctx = TestContext(counter=3, processed=5)
    
    config = MergeConfig(accumulate_counters=True)
    enhanced_merge_context(context, branch_ctx, config)
    
    assert context.counter == 8
    assert context.processed == 15
```

### Integration Tests
```python
def test_parallel_step_with_enhanced_merging():
    """Test parallel step with enhanced merging."""
    parallel_step = Step.parallel(
        branches={"branch1": pipeline1, "branch2": pipeline2},
        merge_strategy=MergeStrategy.ENHANCED,
        merge_config=MergeConfig(
            deep_merge_dicts=True,
            accumulate_counters=True
        )
    )
    
    result = await executor.execute(parallel_step, data, context)
    assert result.success
    # Verify enhanced merging worked correctly
```

## Performance Considerations

### Current Performance
- **Simple merging**: O(n) where n = number of fields
- **Memory usage**: Minimal overhead

### Enhanced Performance
- **Deep merging**: O(n * d) where d = depth of nested structures
- **Memory usage**: Slight increase for deep structures
- **Optimization**: Cache merge results for repeated operations

## Migration Strategy

### Backward Compatibility
1. **Default to simple merging** for existing code
2. **Opt-in enhanced merging** via configuration
3. **Gradual migration** with deprecation warnings

### Breaking Changes
- None planned - enhanced merging is additive
- Existing code continues to work unchanged

## Success Metrics

1. **Functionality**: All complex merging scenarios work correctly
2. **Performance**: <10% overhead compared to simple merging
3. **Reliability**: No deadlocks or race conditions
4. **Usability**: Intuitive configuration and error messages

## Related Issues

- **Deadlock Prevention**: Must maintain the reliability improvements
- **Performance**: Must not significantly impact execution speed
- **Compatibility**: Must work with existing parallel step patterns

## Priority

**Medium Priority** - This is a quality-of-life improvement that addresses specific use cases but doesn't block core functionality.

## Dependencies

- Parallel step execution stability (✅ Complete)
- Context management system (✅ Complete)
- Testing infrastructure (✅ Complete)

## Notes

This enhancement restores sophisticated context merging capabilities while maintaining the reliability improvements from the deadlock fix. The implementation is designed to be opt-in and backward compatible. 