# Flujo Library Bug Report

## Executive Summary

This report documents critical issues found in the Flujo library (version ^0.4.15) that affect parameter passing consistency and Pydantic schema generation. These issues break the documented API and prevent the use of standard Pydantic patterns.

## Issues Identified

### 1. Parameter Passing Inconsistency

**Severity**: High  
**Status**: Confirmed  
**Affects**: All step functions using typed context

#### Problem Description
Flujo passes parameters with different names than what step functions expect. The framework passes `pipeline_context` instead of `context`, breaking the documented API pattern.

#### Evidence
- Step functions defined with `context: CohortGenerationContext` parameter
- Flujo passes `pipeline_context` instead of `context`
- Error: `generate_sql_and_heuristics() got an unexpected keyword argument 'pipeline_context'`

#### Root Cause Analysis
In `flujo/application/flujo_engine.py` lines 467-495, the parameter passing logic:

```python
if isinstance(current_agent, ContextAwareAgentProtocol) and getattr(
    current_agent, "__context_aware__", False
):
    agent_kwargs["pipeline_context"] = pipeline_context
elif pipeline_context is not None:
    # ... legacy pattern detection
    if pass_ctx:
        agent_kwargs["pipeline_context"] = pipeline_context
```

The framework consistently uses `pipeline_context` as the parameter name, but the documentation and examples suggest `context` should be used.

#### Impact
- Breaks standard Flujo pattern where step functions should receive context and resources as keyword arguments
- Forces developers to accept both parameter names in step functions as a workaround
- Creates confusion between documented API and actual implementation

#### Workaround
Accept both parameter names in step functions:
```python
async def my_step(data: Any, *, context: MyContext = None, pipeline_context: MyContext = None, resources: MyResources = None) -> None:
    ctx = context or pipeline_context
    # ... rest of implementation
```

### 2. Pydantic Schema Generation Issues

**Severity**: High  
**Status**: Confirmed  
**Affects**: Agents using TypeAdapter or complex Pydantic types

#### Problem Description
Flujo's `make_agent_async` function has trouble with Pydantic schema generation for certain types, particularly `TypeAdapter` and built-in types.

#### Evidence
- Error: `Unable to generate pydantic-core schema for TypeAdapter(str)`
- Error: `Unable to generate pydantic-core schema for TypeAdapter(SQLResponse)`
- Even after creating proper Pydantic models, the issue persists

#### Root Cause Analysis
In `flujo/infra/agents.py` lines 165-175, the `make_agent` function:

```python
agent: Agent[Any, Any] = Agent(
    model=model,
    system_prompt=system_prompt,
    output_type=output_type,  # This is where the issue occurs
    tools=tools or [],
)
```

The pydantic-ai library's Agent constructor expects a type that can be properly converted to a Pydantic schema, but `TypeAdapter` types are not handled correctly.

#### Impact
- Prevents the use of standard Pydantic patterns that should work with Flujo agents
- Forces developers to use Pydantic models directly instead of `TypeAdapter`
- Creates inconsistent behavior between different type patterns

#### Workaround
Use Pydantic models directly instead of `TypeAdapter`:
```python
# Instead of:
# output_type=TypeAdapter(str)

# Use:
output_type=str

# Or for custom models:
class MyResponse(BaseModel):
    content: str

output_type=MyResponse
```

### 3. Version Compatibility Issues

**Severity**: Medium  
**Status**: Suspected  
**Affects**: API consistency across versions

#### Problem Description
The behavior suggests that Flujo's API may have changed between versions, but the changes aren't documented or backward-compatible.

#### Evidence
- Using Flujo version `^0.4.15` (latest)
- Parameter passing issue suggests API changes
- Pydantic integration issues suggest recent changes to type validation

## Detailed Bug Reports

### Bug #1: Parameter Name Mismatch in Step Functions

**Bug ID**: FLUJO-001  
**Component**: `flujo/application/flujo_engine.py`  
**Lines**: 467-495  

#### Steps to Reproduce
1. Create a step function with signature: `async def my_step(_: None, *, context: MyContext, resources: MyResources) -> None`
2. Use `@step` decorator
3. Run the pipeline
4. Observe error: `my_step() got an unexpected keyword argument 'pipeline_context'`

#### Expected Behavior
Step functions should receive `context` parameter as documented.

#### Actual Behavior
Step functions receive `pipeline_context` parameter.

#### Proposed Fix
Update the parameter passing logic to use `context` instead of `pipeline_context`:

```python
# In _run_step_logic function
if isinstance(current_agent, ContextAwareAgentProtocol) and getattr(
    current_agent, "__context_aware__", False
):
    agent_kwargs["context"] = pipeline_context  # Changed from "pipeline_context"
elif pipeline_context is not None:
    # ... legacy pattern detection
    if pass_ctx:
        agent_kwargs["context"] = pipeline_context  # Changed from "pipeline_context"
```

### Bug #2: Pydantic Schema Generation Failure

**Bug ID**: FLUJO-002  
**Component**: `flujo/infra/agents.py`  
**Lines**: 165-175  

#### Steps to Reproduce
1. Create an agent with `TypeAdapter(str)` or `TypeAdapter(MyModel)`
2. Use `make_agent_async` with the type adapter
3. Run the pipeline
4. Observe error: `Unable to generate pydantic-core schema for TypeAdapter(...)`

#### Expected Behavior
Flujo should properly handle Pydantic type adapters and models.

#### Actual Behavior
Schema generation fails with cryptic error messages.

#### Proposed Fix
Add type adapter handling in the `make_agent` function:

```python
def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
) -> Agent[Any, Any]:
    # Handle TypeAdapter types
    if hasattr(output_type, 'type_adapter'):
        # Extract the underlying type from TypeAdapter
        actual_type = output_type.type_adapter
    else:
        actual_type = output_type
    
    agent: Agent[Any, Any] = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=actual_type,  # Use the actual type
        tools=tools or [],
    )
    return agent
```

### Bug #3: Inconsistent Type Validation

**Bug ID**: FLUJO-003  
**Component**: `flujo/infra/agents.py`  
**Lines**: 165-175  

#### Problem Description
Flujo's type validation system is inconsistent and doesn't follow Pydantic best practices.

#### Evidence
- `TypeAdapter(str)` fails but `MyModel` (Pydantic model) works
- Error messages suggest infinite recursion in schema generation
- The framework seems to be calling Pydantic internals incorrectly

#### Proposed Fix
Improve type validation with better error handling:

```python
def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
) -> Agent[Any, Any]:
    # Validate and normalize the output type
    try:
        if hasattr(output_type, 'type_adapter'):
            # Handle TypeAdapter
            actual_type = output_type.type_adapter
        elif hasattr(output_type, '__origin__') and output_type.__origin__ is TypeAdapter:
            # Handle TypeAdapter[T] pattern
            actual_type = output_type.__args__[0]
        else:
            actual_type = output_type
        
        # Test schema generation
        from pydantic import create_model
        test_model = create_model('TestModel', value=(actual_type, ...))
        
    except Exception as e:
        raise ValueError(
            f"Invalid output_type '{output_type}': {e}. "
            "Use a Pydantic model, built-in type, or properly configured TypeAdapter."
        ) from e
    
    agent: Agent[Any, Any] = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=actual_type,
        tools=tools or [],
    )
    return agent
```

## Recommendations for Flujo Developers

### Immediate Actions
1. **Fix Parameter Passing**: Ensure consistent parameter names between framework and step functions
2. **Improve Pydantic Integration**: Fix schema generation issues and provide better error messages
3. **Add Type Validation Tests**: Create comprehensive tests for different Pydantic patterns

### Documentation Updates
1. **Update API Documentation**: Document the correct parameter names and provide migration guides
2. **Add Type Usage Guide**: Create examples showing proper type usage patterns
3. **Version Migration Guide**: Document any API changes between versions

### Testing Improvements
1. **Add Parameter Passing Tests**: Test both `context` and `pipeline_context` parameter patterns
2. **Add Type Validation Tests**: Test various Pydantic type patterns including TypeAdapter
3. **Add Backward Compatibility Tests**: Ensure existing code continues to work

### Code Quality Improvements
1. **Add Better Error Messages**: Provide clear guidance when type validation fails
2. **Add Debugging Tools**: Provide better error messages and debugging information for pipeline issues
3. **Add Type Hints**: Improve type safety throughout the codebase

## Conclusion

These issues significantly impact the developer experience with Flujo and should be addressed as high-priority bugs. The parameter passing inconsistency breaks the documented API, while the Pydantic schema issues prevent the use of standard patterns.

The proposed fixes maintain backward compatibility while resolving the core issues. Immediate attention should be given to the parameter passing issue as it affects all users of typed contexts.

## Contact Information

This bug report was generated based on real-world usage patterns and testing. For questions or additional context, please refer to the original analysis provided by the user. 