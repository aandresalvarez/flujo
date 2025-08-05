#!/usr/bin/env python3
"""
Script to fix the ExecutorCore implementation by removing duplicates and implementing the required changes.
"""

def fix_executor_file():
    # Read the original file
    with open('flujo/application/core/ultra_executor.py', 'r') as f:
        content = f.read()
    
    # Find the first ExecutorCore class (line 1169)
    first_executor_start = content.find('class ExecutorCore(Generic[TContext_w_Scratch]):')
    if first_executor_start == -1:
        print("Could not find first ExecutorCore class")
        return
    
    # Find the end of the first ExecutorCore class by finding the next class
    search_start = first_executor_start + 100  # Start searching after the class definition
    next_class_start = content.find('\nclass DefaultProcessorPipeline:', search_start)
    if next_class_start == -1:
        print("Could not find end of first ExecutorCore class")
        return
    
    # Remove the first ExecutorCore class and all duplicate classes until the second ExecutorCore
    second_executor_start = content.find('class ExecutorCore(Generic[TContext_w_Scratch]):', next_class_start)
    if second_executor_start == -1:
        print("Could not find second ExecutorCore class")
        return
    
    # Keep everything before the first ExecutorCore, then skip to the second ExecutorCore
    before_first = content[:first_executor_start]
    after_second = content[second_executor_start:]
    
    # Add a comment where the first ExecutorCore was
    fixed_content = before_first + "# Removed duplicate ExecutorCore class - using the one below\n\n" + after_second
    
    # Now fix the execute method in the remaining ExecutorCore
    # Find the routing logic that needs to be replaced
    old_routing = """        # Check if this is a complex step that needs special handling
        is_complex = self._is_complex_step(step)
        print(f"🔍 Step {step.name} is_complex: {is_complex}")
        
        if is_complex:
            print(f"🔍 Executing complex step: {step.name}")
            telemetry.logfire.debug(f"Complex step detected: {step.name}")
            return await self._execute_complex_step(
                step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                breach_event,
                context_setter,
                cache_key,
                _fallback_depth,
            )

        print(f"🔍 Executing agent step: {step.name}")
        telemetry.logfire.debug(f"Agent step detected: {step.name}")
        return await self._execute_simple_step(
            step,
            data,
            context,
            resources,
            limits,
            stream,
            on_chunk,
            cache_key,
            breach_event,
            _fallback_depth,
        )"""
    
    new_routing = """        # Consistent step routing following the recursive execution model
        # Route to appropriate handler based on step type
        if isinstance(step, LoopStep):
            telemetry.logfire.debug(f"Routing to loop step handler: {step.name}")
            return await self._handle_loop_step(
                step, data, context, resources, limits, context_setter, _fallback_depth
            )
        elif isinstance(step, ParallelStep):
            telemetry.logfire.debug(f"Routing to parallel step handler: {step.name}")
            return await self._handle_parallel_step(
                step, data, context, resources, limits, breach_event, context_setter
            )
        elif isinstance(step, ConditionalStep):
            telemetry.logfire.debug(f"Routing to conditional step handler: {step.name}")
            return await self._handle_conditional_step(
                step, data, context, resources, limits, context_setter, _fallback_depth
            )
        elif isinstance(step, DynamicParallelRouterStep):
            telemetry.logfire.debug(f"Routing to dynamic router step handler: {step.name}")
            return await self._handle_dynamic_router_step(
                step, data, context, resources, limits, context_setter
            )
        elif isinstance(step, HumanInTheLoopStep):
            telemetry.logfire.debug(f"Routing to HITL step handler: {step.name}")
            return await self._handle_hitl_step(
                step, data, context, resources, limits, breach_event, context_setter
            )
        elif isinstance(step, CacheStep):
            telemetry.logfire.debug(f"Routing to cache step handler: {step.name}")
            return await self._handle_cache_step(
                step, data, context, resources, limits, breach_event, context_setter, None
            )
        elif hasattr(step, 'fallback_step') and step.fallback_step is not None:
            telemetry.logfire.debug(f"Routing to simple step with fallback: {step.name}")
            return await self._execute_simple_step(
                step, data, context, resources, limits, stream, on_chunk, cache_key, breach_event, _fallback_depth
            )
        else:
            telemetry.logfire.debug(f"Routing to agent step handler: {step.name}")
            return await self._execute_agent_step(
                step, data, context, resources, limits, stream, on_chunk, cache_key, breach_event, _fallback_depth
            )"""
    
    # Replace the routing logic
    if old_routing in fixed_content:
        fixed_content = fixed_content.replace(old_routing, new_routing)
        print("Successfully replaced routing logic")
    else:
        print("Could not find routing logic to replace")
        return
    
    # Write the fixed content back
    with open('flujo/application/core/ultra_executor.py', 'w') as f:
        f.write(fixed_content)
    
    print("Successfully fixed ultra_executor.py")

if __name__ == "__main__":
    fix_executor_file()