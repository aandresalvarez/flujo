# TODO: Enhanced Concurrency Control for Parallel Steps

## Overview

During the parallel step execution deadlock fix, we simplified the concurrency control mechanisms to prioritize reliability over fine-grained control. This document tracks the trade-off and proposes a future enhancement.

## Current Implementation

### Simplified Concurrency Control
```python
# Current: Simple timeout-based execution
tasks = [
    asyncio.create_task(run_branch(key, branch_pipe), name=f"branch_{key}")
    for key, branch_pipe in parallel_step.branches.items()
]

# Wait for all tasks with timeout
timeout_seconds = 30  # Reasonable timeout
branch_results_list = await asyncio.wait_for(
    asyncio.gather(*tasks, return_exceptions=True),
    timeout=timeout_seconds
)
```

### What Was Removed

#### 1. Adaptive Semaphore Control
```python
# Previous: CPU-based adaptive limits
cpu_count = multiprocessing.cpu_count()
semaphore = asyncio.Semaphore(min(10, cpu_count * 2))

async def run_branch(key: str, branch_pipe: Any) -> None:
    async with semaphore:  # Limited concurrency
        # Branch execution logic
```

#### 2. Fine-Grained Task Management
```python
# Previous: Individual task tracking and cancellation
running_tasks: Dict[str, asyncio.Task[None]] = {}

for key, branch_pipe in parallel_step.branches.items():
    task = asyncio.create_task(run_branch(key, branch_pipe), name=f"branch_{key}")
    running_tasks[key] = task

# Individual task cancellation
for task_key, task in running_tasks.items():
    if not task.done():
        task.cancel()
```

#### 3. Completion Order Tracking
```python
# Previous: Track completion order for OVERWRITE strategy
completion_order = []
completion_lock = asyncio.Lock()

async def run_branch(key: str, branch_pipe: Any) -> None:
    # ... execution logic ...
    async with completion_lock:
        completion_order.append(key)
```

#### 4. Responsive Breach Watcher
```python
# Previous: Dedicated breach watcher with responsive signaling
async def breach_watcher() -> None:
    """Watch for breach events and cancel all running tasks."""
    try:
        await breach_event.wait()
        for task_key, task in running_tasks.items():
            if not task.done():
                task.cancel()
    except asyncio.CancelledError:
        pass

watcher_task = asyncio.create_task(breach_watcher(), name="breach_watcher")
```

## Impact Analysis

### High Impact Scenarios

#### 1. Resource-Intensive Workflows
```python
# Before: Limited to CPU * 2 concurrent operations
# After: All branches run simultaneously
# Risk: Memory/CPU exhaustion with many branches

# Example: Processing large datasets in parallel
parallel_step = Step.parallel(
    branches={
        f"processor_{i}": data_processing_pipeline 
        for i in range(100)  # 100 concurrent processors
    }
)
# Result: All 100 processors run simultaneously, potentially overwhelming the system
```

#### 2. Complex Cancellation Logic
```python
# Before: Fine-grained control over individual tasks
for task_key, task in running_tasks.items():
    if task_key.startswith("high_priority_"):
        # Cancel low priority tasks when high priority completes
        for low_task in [t for k, t in running_tasks.items() if k.startswith("low_priority_")]:
            if not low_task.done():
                low_task.cancel()

# After: Simple gather with timeout
# No individual task control
```

#### 3. Performance-Critical Applications
```python
# Before: Adaptive concurrency based on system load
semaphore = AdaptiveSemaphore(initial_limit=cpu_count * 2)
# Automatically adjusts based on performance metrics

# After: No adaptive behavior
# Fixed timeout regardless of system performance
```

### Low Impact Scenarios

1. **Small Parallel Workflows**: Few branches, simple operations
2. **Independent Operations**: Branches don't compete for resources
3. **Timeout-Based Safety**: 30s timeout provides basic protection

## Current Workarounds

### 1. External Concurrency Management
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ConcurrencyLimitedParallelStep:
    """Custom parallel step with external concurrency control."""
    
    def __init__(self, max_concurrency: int = 4):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def execute_branches(self, branches: Dict[str, Any]) -> Dict[str, Any]:
        """Execute branches with controlled concurrency."""
        async def limited_branch(key: str, branch: Any) -> tuple[str, Any]:
            async with self.semaphore:
                # Execute branch logic
                result = await self.execute_branch(branch)
                return key, result
        
        tasks = [
            limited_branch(key, branch) 
            for key, branch in branches.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(results)

# Usage
limited_step = ConcurrencyLimitedParallelStep(max_concurrency=4)
results = await limited_step.execute_branches(branch_pipelines)
```

### 2. Pipeline-Level Limits
```python
# Set limits at the pipeline level
pipeline = Pipeline(
    steps=[
        Step.parallel(
            branches={"branch1": pipeline1, "branch2": pipeline2},
            max_concurrency=2  # Limit concurrent execution
        )
    ],
    execution_config=ExecutionConfig(
        max_parallel_steps=4,  # Global pipeline limit
        resource_limits=ResourceLimits(
            max_memory_mb=1024,
            max_cpu_percent=80
        )
    )
)
```

### 3. Resource Monitoring
```python
import psutil
import asyncio

class ResourceAwareExecutor:
    """Executor that monitors system resources."""
    
    def __init__(self, max_cpu_percent: float = 80, max_memory_percent: float = 80):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
    
    async def execute_with_monitoring(self, parallel_step: Any) -> Any:
        """Execute with resource monitoring."""
        async def monitor_resources():
            while True:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > self.max_cpu_percent or memory_percent > self.max_memory_percent:
                    # Trigger backpressure or cancellation
                    await self.handle_resource_pressure()
                
                await asyncio.sleep(1)
        
        # Start monitoring in background
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            # Execute parallel step
            result = await self.execute_parallel_step(parallel_step)
            return result
        finally:
            monitor_task.cancel()
```

## Proposed Implementation

### Enhanced Concurrency Control Strategy

#### 1. Configurable Concurrency Limits
```python
class ConcurrencyConfig:
    """Configuration for parallel step concurrency control."""
    
    def __init__(
        self,
        max_concurrent_branches: Optional[int] = None,
        adaptive_limits: bool = True,
        cpu_based_limits: bool = True,
        memory_based_limits: bool = True,
        completion_tracking: bool = True,
        individual_cancellation: bool = True,
        breach_watcher: bool = True
    ):
        self.max_concurrent_branches = max_concurrent_branches
        self.adaptive_limits = adaptive_limits
        self.cpu_based_limits = cpu_based_limits
        self.memory_based_limits = memory_based_limits
        self.completion_tracking = completion_tracking
        self.individual_cancellation = individual_cancellation
        self.breach_watcher = breach_watcher
```

#### 2. Enhanced Parallel Step Execution
```python
async def _handle_parallel_step_enhanced(
    self,
    parallel_step: ParallelStep[TContext],
    concurrency_config: ConcurrencyConfig,
    **kwargs
) -> StepResult:
    """Enhanced parallel step execution with configurable concurrency control."""
    
    # Determine concurrency limits
    if concurrency_config.cpu_based_limits:
        cpu_count = multiprocessing.cpu_count()
        max_concurrent = concurrency_config.max_concurrent_branches or min(10, cpu_count * 2)
    else:
        max_concurrent = concurrency_config.max_concurrent_branches or 10
    
    # Create semaphore for controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track running tasks and completion order
    running_tasks: Dict[str, asyncio.Task[Any]] = {}
    completion_order = []
    completion_lock = asyncio.Lock()
    
    async def run_branch_with_control(key: str, branch_pipe: Any) -> tuple[str, StepResult]:
        """Execute branch with concurrency control."""
        async with semaphore:
            try:
                # Execute branch logic
                result = await self._execute_branch(branch_pipe, **kwargs)
                
                # Track completion order
                if concurrency_config.completion_tracking:
                    async with completion_lock:
                        completion_order.append(key)
                
                return key, result
                
            except asyncio.CancelledError:
                return key, StepResult(
                    name=f"branch::{key}",
                    success=False,
                    feedback="Cancelled due to concurrency control"
                )
    
    # Create tasks with controlled concurrency
    for key, branch_pipe in parallel_step.branches.items():
        task = asyncio.create_task(
            run_branch_with_control(key, branch_pipe),
            name=f"branch_{key}"
        )
        running_tasks[key] = task
    
    # Breach watcher for usage limits
    if concurrency_config.breach_watcher and kwargs.get("breach_event"):
        async def breach_watcher():
            try:
                await kwargs["breach_event"].wait()
                if concurrency_config.individual_cancellation:
                    for task_key, task in running_tasks.items():
                        if not task.done():
                            task.cancel()
            except asyncio.CancelledError:
                pass
        
        watcher_task = asyncio.create_task(breach_watcher(), name="breach_watcher")
        all_tasks = list(running_tasks.values()) + [watcher_task]
    else:
        all_tasks = list(running_tasks.values())
    
    # Execute with timeout
    try:
        branch_results_list = await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=kwargs.get("timeout", 30)
        )
    except asyncio.TimeoutError:
        # Cancel all tasks on timeout
        for task in all_tasks:
            if not task.done():
                task.cancel()
        branch_results_list = [None] * len(running_tasks)
    
    # Process results
    branch_results = {}
    for i, (key, result) in enumerate(zip(running_tasks.keys(), branch_results_list)):
        if isinstance(result, Exception):
            branch_results[key] = StepResult(
                name=key,
                success=False,
                feedback=f"Branch execution failed: {str(result)}"
            )
        else:
            branch_results[key] = result[1]  # Extract StepResult from tuple
    
    # Build final result
    return self._build_parallel_result(branch_results, completion_order)
```

#### 3. Integration with Parallel Steps
```python
class ParallelStep:
    """Enhanced parallel step with concurrency control."""
    
    def __init__(
        self,
        branches: Dict[str, Any],
        merge_strategy: MergeStrategy = MergeStrategy.NO_MERGE,
        concurrency_config: ConcurrencyConfig = None,
        **kwargs
    ):
        self.branches = branches
        self.merge_strategy = merge_strategy
        self.concurrency_config = concurrency_config or ConcurrencyConfig()
        self.kwargs = kwargs
```

#### 4. DSL Integration
```python
# Enhanced DSL for parallel steps
parallel_step = Step.parallel(
    branches={"branch1": pipeline1, "branch2": pipeline2},
    concurrency=ConcurrencyConfig(
        max_concurrent_branches=4,
        adaptive_limits=True,
        completion_tracking=True
    )
)

# Or using simplified syntax
parallel_step = Step.parallel(
    branches={"branch1": pipeline1, "branch2": pipeline2},
    max_concurrency=4,
    track_completion=True
)
```

## Implementation Plan

### Phase 1: Core Concurrency Control
1. **Implement `ConcurrencyConfig` class**
2. **Create enhanced parallel step execution logic**
3. **Add semaphore-based concurrency control**
4. **Implement completion tracking**

### Phase 2: Advanced Features
1. **Add breach watcher for usage limits**
2. **Implement individual task cancellation**
3. **Add adaptive concurrency limits**
4. **Performance monitoring and optimization**

### Phase 3: Integration and Polish
1. **Update DSL to support concurrency configuration**
2. **Add documentation and examples**
3. **Comprehensive testing suite**
4. **Performance benchmarking**

## Testing Strategy

### Unit Tests
```python
def test_concurrency_limits():
    """Test that concurrency limits are respected."""
    config = ConcurrencyConfig(max_concurrent_branches=2)
    
    # Create parallel step with 4 branches
    parallel_step = Step.parallel(
        branches={f"branch{i}": pipeline for i in range(4)}
    )
    
    # Execute with concurrency limit
    result = await executor._handle_parallel_step_enhanced(
        parallel_step, config
    )
    
    # Verify that only 2 branches ran concurrently
    # (This would require monitoring the semaphore usage)
    assert result.success
```

### Integration Tests
```python
def test_adaptive_concurrency():
    """Test adaptive concurrency based on system load."""
    config = ConcurrencyConfig(
        adaptive_limits=True,
        cpu_based_limits=True
    )
    
    # Mock high CPU usage
    with patch('multiprocessing.cpu_count', return_value=8):
        parallel_step = Step.parallel(branches=branches)
        result = await executor.execute(parallel_step, concurrency_config=config)
        
        # Verify adaptive behavior
        assert result.success
```

### Performance Tests
```python
def test_concurrency_performance():
    """Test performance impact of concurrency control."""
    # Baseline: No concurrency control
    baseline_time = await measure_execution_time(
        executor._handle_parallel_step_simple
    )
    
    # Enhanced: With concurrency control
    enhanced_time = await measure_execution_time(
        executor._handle_parallel_step_enhanced
    )
    
    # Ensure overhead is acceptable (<20% increase)
    assert enhanced_time <= baseline_time * 1.2
```

## Performance Considerations

### Current Performance
- **Simple execution**: O(n) where n = number of branches
- **Memory usage**: All branches run simultaneously
- **CPU usage**: Uncontrolled, potentially overwhelming

### Enhanced Performance
- **Controlled execution**: O(n) with semaphore limiting
- **Memory usage**: Controlled by concurrency limits
- **CPU usage**: Adaptive based on system capabilities

### Optimization Strategies
1. **Adaptive semaphore limits** based on system load
2. **Task prioritization** for important branches
3. **Resource monitoring** with automatic backpressure
4. **Caching** of frequently used branch results

## Migration Strategy

### Backward Compatibility
1. **Default to simple execution** for existing code
2. **Opt-in enhanced concurrency** via configuration
3. **Gradual migration** with performance monitoring

### Breaking Changes
- None planned - enhanced concurrency is additive
- Existing code continues to work unchanged

## Success Metrics

1. **Reliability**: No deadlocks or resource exhaustion
2. **Performance**: <20% overhead compared to simple execution
3. **Control**: Fine-grained control over concurrent execution
4. **Adaptability**: Automatic adjustment to system capabilities

## Related Issues

- **Resource Management**: Must prevent system overload
- **Performance**: Must not significantly impact execution speed
- **Reliability**: Must maintain deadlock-free execution
- **Compatibility**: Must work with existing parallel step patterns

## Priority

**Medium Priority** - This is a quality-of-life improvement that addresses resource management concerns but doesn't block core functionality.

## Dependencies

- Parallel step execution stability (✅ Complete)
- Resource monitoring capabilities (🔄 In Progress)
- Performance testing infrastructure (✅ Complete)

## Notes

This enhancement restores sophisticated concurrency control capabilities while maintaining the reliability improvements from the deadlock fix. The implementation is designed to be opt-in and backward compatible, with adaptive behavior based on system capabilities.

## Future Enhancements

### Advanced Concurrency Features
1. **Task prioritization** based on branch importance
2. **Resource-aware scheduling** based on memory/CPU usage
3. **Dynamic concurrency adjustment** based on performance metrics
4. **Distributed concurrency control** for multi-node deployments

### Monitoring and Observability
1. **Real-time concurrency metrics** (active tasks, queue depth)
2. **Resource utilization tracking** (CPU, memory, I/O)
3. **Performance alerts** for resource exhaustion
4. **Concurrency visualization** in monitoring dashboards 