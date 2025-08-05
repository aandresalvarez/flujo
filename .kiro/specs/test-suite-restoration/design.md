# Design Document

## Overview

This document outlines the design for systematically restoring the Flujo test suite by addressing fundamental architectural issues. The design is based on first principles analysis of the Flujo architecture, which emphasizes a dual architecture with a declarative shell and a robust execution core.

The 282 failing tests indicate systemic issues across multiple architectural layers that require coordinated fixes to restore the system to its intended design. Rather than patching individual test failures, this design addresses the root causes at the architectural level.

## Architecture

The design follows the Flujo architecture's core principles:

1. **Recursive Execution Model**: All step types pass through the same optimized, instrumented execution path
2. **Dependency Injection**: Components are designed as independent, replaceable modules with clear interfaces
3. **Algebraic Closure**: Complex structures are themselves steps that can be composed seamlessly
4. **Production Readiness**: Focus on resilience, performance, and observability

## Components and Interfaces

### 1. ExecutorCore Restoration

**Component**: `flujo/application/core/ultra_executor.py`

**Current Issues**:
- Inconsistent step handling across different step types
- Agent retry logic conflating retryable and non-retryable failures
- Missing or incomplete handler methods for certain step types
- Fallback logic not properly integrated with the execution flow

**Design Solution**:

```python
class ExecutorCore:
    """
    Restored ExecutorCore following the recursive execution model.
    All step types pass through the same optimized execution path.
    """
    
    async def execute(self, step: Step, context: ContextModelT) -> StepResult:
        """
        Central execution method that routes to appropriate handlers.
        Implements the recursive execution model consistently.
        """
        # Route to appropriate handler based on step type
        if isinstance(step, LoopStep):
            return await self._handle_loop_step(step, context)
        elif isinstance(step, ParallelStep):
            return await self._handle_parallel_step(step, context)
        elif isinstance(step, ConditionalStep):
            return await self._handle_conditional_step(step, context)
        elif hasattr(step, 'fallback') and step.fallback:
            return await self._execute_simple_step(step, context)
        else:
            return await self._execute_agent_step(step, context)
    
    async def _execute_agent_step(self, step: Step, context: ContextModelT) -> StepResult:
        """
        Execute agent steps with proper separation of failure domains.
        Distinguishes between retryable agent failures and non-retryable validation/plugin failures.
        """
        # Separate try-catch blocks for different failure types
        # Validator failures: immediate failure, no retry
        # Plugin failures: immediate failure, no retry  
        # Agent failures: retry with exponential backoff
        
    async def _execute_simple_step(self, step: Step, context: ContextModelT) -> StepResult:
        """
        Execute simple steps with fallback logic.
        Handles metrics accumulation and feedback combination properly.
        """
        # Implement comprehensive fallback support
        # Handle metrics accumulation correctly
        # Preserve step names and context
```

### 2. Context Management System

**Component**: Context propagation across all step types

**Current Issues**:
- Context modifications not properly accumulated in loop steps
- Parallel step context merging inconsistent
- Conditional step context updates not preserved
- HITL step context status not updated correctly

**Design Solution**:

```python
class ContextManager:
    """
    Centralized context management following copy-on-write principles.
    Ensures proper isolation and merging across all step types.
    """
    
    def isolate_context(self, context: ContextModelT) -> ContextModelT:
        """Create isolated context copy for branch execution."""
        return copy.deepcopy(context)
    
    def merge_context_updates(self, 
                            main_context: ContextModelT, 
                            branch_context: ContextModelT) -> ContextModelT:
        """Merge branch context updates back to main context."""
        # Use safe_merge_context_updates for proper merging
        # Handle different context types appropriately
        
    def accumulate_loop_context(self, 
                              current_context: ContextModelT,
                              iteration_context: ContextModelT) -> ContextModelT:
        """Accumulate context changes across loop iterations."""
        # Ensure context modifications persist across iterations
        # Handle scratchpad updates correctly
```

**Loop Step Context Flow**:
```
Initial Context → Iteration 1 → Updated Context → Iteration 2 → Final Context
                     ↓              ↓              ↓
                 Isolated      Accumulated    Accumulated
```

**Parallel Step Context Flow**:
```
Main Context → Branch A (isolated) → Merge Results
            → Branch B (isolated) → 
            → Branch C (isolated) →
```

### 3. Serialization System Overhaul

**Component**: `flujo/utils/serialization.py`

**Current Issues**:
- AgentResponse objects not serializable
- Enum handling incorrect
- Custom object serialization incomplete
- Circular reference handling missing

**Design Solution**:

```python
def safe_serialize(obj: Any) -> Any:
    """
    Comprehensive serialization with proper type handling.
    Follows the order: specific types → generic fallbacks.
    """
    # Handle specific types first (most specific to least specific)
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, AgentResponse):
        return serialize_agent_response(obj)
    elif hasattr(obj, 'model_dump'):  # Pydantic models
        return obj.model_dump()
    elif isinstance(obj, Mock):
        return serialize_mock_object(obj)
    elif hasattr(obj, '__dict__'):
        return serialize_custom_object(obj)
    else:
        # Fallback for unknown types
        return handle_unknown_type(obj)

def serialize_agent_response(response: AgentResponse) -> Dict[str, Any]:
    """Serialize AgentResponse objects properly."""
    return {
        'content': response.content,
        'usage': response.usage.model_dump() if response.usage else None,
        'metadata': response.metadata
    }
```

### 4. Usage Governance System

**Component**: Usage limit enforcement and cost tracking

**Current Issues**:
- Usage limits checked before costs are added
- Loop step attempt counting incorrect
- Parallel step cost aggregation wrong
- Proactive cancellation timing issues

**Design Solution**:

```python
class UsageGovernor:
    """
    Restored usage governance with proper sequencing and cost tracking.
    """
    
    def guard(self, pipeline_result: PipelineResult) -> None:
        """
        Check usage limits after costs are properly accumulated.
        Sequence: Execute → Add to history → Update totals → Check limits
        """
        # Check limits on the updated totals
        # Provide accurate error messages with current costs
        
    def track_parallel_costs(self, branch_results: List[StepResult]) -> Tuple[float, int]:
        """
        Aggregate costs from parallel branches correctly.
        Only count successful branches, avoid double-counting.
        """
        total_cost = sum(result.cost_usd for result in branch_results if result.success)
        total_tokens = sum(result.token_counts for result in branch_results if result.success)
        return total_cost, total_tokens
        
    def handle_proactive_cancellation(self, breach_event: asyncio.Event) -> None:
        """
        Implement efficient proactive cancellation.
        Cancel parallel branches immediately when limits are breached.
        """
        # Set breach event immediately
        # Cancel running tasks efficiently
        # Ensure timing constraints are met
```

### 5. Parallel Step Execution Engine

**Component**: Parallel step handling with proper isolation and merging

**Current Issues**:
- Context isolation not working correctly
- Error propagation inconsistent
- Cancellation not efficient enough
- Context merging strategies not applied properly

**Design Solution**:

```python
async def _handle_parallel_step(self, step: ParallelStep, context: ContextModelT) -> StepResult:
    """
    Handle parallel execution with proper isolation and merging.
    Implements the recursive execution model for parallel branches.
    """
    # Create isolated contexts for each branch
    branch_contexts = [self.context_manager.isolate_context(context) 
                      for _ in step.branches]
    
    # Execute branches concurrently with proper cancellation support
    branch_tasks = [
        self.execute(branch_step, branch_context)
        for branch_step, branch_context in zip(step.branches, branch_contexts)
    ]
    
    # Handle results with proper error propagation
    branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)
    
    # Merge successful branch contexts
    final_context = context
    for result in branch_results:
        if isinstance(result, StepResult) and result.success and result.branch_context:
            final_context = self.context_manager.merge_context_updates(
                final_context, result.branch_context
            )
    
    # Determine overall success based on strategy
    overall_success = self._determine_parallel_success(branch_results, step.strategy)
    
    return StepResult(
        name=step.name,
        output=self._aggregate_parallel_outputs(branch_results),
        success=overall_success,
        branch_context=final_context,
        step_history=branch_results
    )
```

### 6. Loop Step Logic Engine

**Component**: Loop step handling with proper iteration management

**Current Issues**:
- Iteration counting incorrect
- Exit condition evaluation inconsistent
- Context accumulation not working
- Max iterations logic wrong

**Design Solution**:

```python
async def _handle_loop_step(self, step: LoopStep, context: ContextModelT) -> StepResult:
    """
    Handle loop execution with proper iteration management.
    Implements context accumulation and accurate counting.
    """
    current_context = context
    iteration_count = 0
    
    while iteration_count < step.max_iterations:
        # Apply iteration input mapper if present
        iteration_input = self._apply_iteration_input_mapper(
            step, current_context, iteration_count
        )
        
        # Execute body with current context
        body_result = await self.execute(step.body, current_context)
        
        # Accumulate context changes
        if body_result.branch_context:
            current_context = self.context_manager.accumulate_loop_context(
                current_context, body_result.branch_context
            )
        
        iteration_count += 1
        
        # Check exit condition
        if self._evaluate_exit_condition(step, current_context, body_result):
            break
    
    # Apply output mapper after loop completion
    final_output = self._apply_output_mapper(step, current_context)
    
    return StepResult(
        name=step.name,
        output=final_output,
        success=True,
        attempts=iteration_count,
        branch_context=current_context,
        metadata_={'iterations': iteration_count}
    )
```

### 7. HITL Integration System

**Component**: Human-in-the-Loop step integration

**Current Issues**:
- Method signatures inconsistent
- Message formatting incorrect
- Context status not updated
- Integration with other components broken

**Design Solution**:

```python
async def _handle_hitl_step(self, 
                          step: HITLStep, 
                          context: ContextModelT,
                          context_setter: Callable[[ContextModelT], None]) -> StepResult:
    """
    Handle HITL steps with proper integration.
    Consistent method signature and proper context management.
    """
    # Update context status before pausing
    if isinstance(context, PipelineContext):
        context.scratchpad['status'] = 'paused'
        context_setter(context)
    
    # Format message appropriately for step type
    message = self._format_hitl_message(step)
    
    # Raise PausedException with proper context
    raise PausedException(message)

def _format_hitl_message(self, step: HITLStep) -> str:
    """Format HITL messages consistently across step types."""
    if hasattr(step, 'message') and step.message:
        return step.message
    else:
        return f"Human-in-the-Loop step '{step.name}' requires human input"
```

## Data Models

### Enhanced StepResult Model

```python
@dataclass
class StepResult:
    """Enhanced StepResult with proper context and metadata handling."""
    name: str
    output: Any
    success: bool
    attempts: int = 1
    latency_s: float = 0.0
    token_counts: int = 0
    cost_usd: float = 0.0
    feedback: Optional[str] = None
    branch_context: Optional[ContextModelT] = None
    metadata_: Dict[str, Any] = field(default_factory=dict)
    step_history: List['StepResult'] = field(default_factory=list)
```

### Context Management Models

```python
class ContextState(Enum):
    """Context states for proper lifecycle management."""
    RUNNING = "running"
    PAUSED = "paused" 
    COMPLETED = "completed"
    FAILED = "failed"

class ContextManager:
    """Context management with proper isolation and merging."""
    def __init__(self):
        self.isolation_strategy = "deep_copy"
        self.merge_strategy = "safe_merge"
```

## Error Handling

### Failure Domain Separation

The design implements clear separation between different types of failures:

1. **Agent Failures**: Retryable with exponential backoff
2. **Validation Failures**: Non-retryable, immediate failure
3. **Plugin Failures**: Non-retryable, immediate failure
4. **System Failures**: Handled with circuit breaker patterns

### Error Propagation Strategy

```python
class ErrorHandler:
    """Centralized error handling with proper categorization."""
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize errors for appropriate handling."""
        if isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, PluginError):
            return ErrorCategory.PLUGIN
        elif isinstance(error, AgentError):
            return ErrorCategory.AGENT
        else:
            return ErrorCategory.SYSTEM
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorResponse:
        """Handle errors based on category and context."""
        category = self.categorize_error(error)
        return self.error_strategies[category].handle(error, context)
```

## Testing Strategy

### Test Restoration Approach

1. **Architectural Fixes First**: Address core architectural issues before individual test fixes
2. **Layer-by-Layer Restoration**: Fix issues in dependency order (core → context → serialization → etc.)
3. **Regression Prevention**: Ensure fixes don't break existing functionality
4. **Performance Validation**: Verify that fixes maintain acceptable performance

### Test Categories

1. **Core Execution Tests**: Verify ExecutorCore handles all step types correctly
2. **Context Management Tests**: Validate context isolation and merging
3. **Serialization Tests**: Ensure all types are properly serializable
4. **Usage Governance Tests**: Verify cost tracking and limit enforcement
5. **Integration Tests**: Validate component interactions
6. **Performance Tests**: Ensure acceptable overhead levels

## Implementation Plan

The implementation will follow a systematic approach:

1. **Phase 1**: Restore ExecutorCore with proper step handling
2. **Phase 2**: Fix context management across all step types
3. **Phase 3**: Overhaul serialization system
4. **Phase 4**: Restore usage governance and cost tracking
5. **Phase 5**: Fix parallel and loop step execution
6. **Phase 6**: Integrate HITL system properly
7. **Phase 7**: Address performance and persistence issues
8. **Phase 8**: Validate end-to-end integration

Each phase will be implemented with careful attention to the Flujo architecture principles, ensuring that fixes align with the intended design rather than creating patches that deviate from the architectural vision.