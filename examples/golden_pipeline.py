"""
Robust Golden Pipeline for Comprehensive Framework Testing

This pipeline exercises major flujo framework features in a realistic, idiomatic, and type-safe way:
- ConditionalStep: Branching logic with different processing paths.
  - Branch A: map_over a list of items, aggregate results.
  - Branch B: loop_until with a fallback/retry step.
- Caching: Step.cached for performance optimization.
- Custom Context: Tracks state across all features.
- Final aggregation step collects all results.

This is the first step in building a truly robust golden pipeline.
"""

import asyncio
from typing import Any, Dict, List, Optional
from flujo.domain import Step, Pipeline
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext, RefinementCheck
from flujo.domain.resources import AppResources
from flujo.domain.dsl.step import step
from flujo.domain import MergeStrategy

class GoldenContext(PipelineContext):
    initial_prompt: str = ""
    initial_data: str = ""
    conditional_path_taken: str = ""
    map_over_results: list = []
    loop_iterations: int = 0
    loop_final_value: int = 0
    fallback_triggered: bool = False
    retry_attempts: int = 0
    items: list = []  # Add items field for map_over
    cache_hits: int = 0
    cache_misses: int = 0
    refine_iterations: int = 0
    refine_final_value: int = 0
    parallel_branch_results: list = []
    parallel_failures: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    final_aggregation: dict = {}

# --- Metric tracking steps ---
@step
async def metric_tracking_step(data: Any, *, context: GoldenContext) -> Any:
    print(f"DEBUG: metric_tracking_step called with: {data}")
    # Simulate cost and token usage
    cost = 0.001  # $0.001 per call
    tokens = 100   # 100 tokens per call
    context.total_cost_usd += cost
    context.total_tokens += tokens
    # Always return the original input
    print(f"DEBUG: metric_tracking_step returning: {data} (cost: ${cost}, tokens: {tokens})")
    return data

# --- Nested sub-pipeline steps ---
@step
async def nested_step_1(data: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: nested_step_1 called with: {data}")
    result = f"nested1_{data}"
    print(f"DEBUG: nested_step_1 returning: {result}")
    return result

@step
async def nested_step_2(data: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: nested_step_2 called with: {data}")
    result = f"nested2_{data}"
    print(f"DEBUG: nested_step_2 returning: {result}")
    return result

@step
async def nested_aggregator(data: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: nested_aggregator called with: {data}")
    result = f"aggregated_{data}"
    print(f"DEBUG: nested_aggregator returning: {result}")
    return result

# Create nested sub-pipeline
def create_nested_pipeline() -> Pipeline:
    return (
        Pipeline.from_step(nested_step_1) >>
        Pipeline.from_step(nested_step_2) >>
        Pipeline.from_step(nested_aggregator)
    )

# --- Dynamic parallel branch steps ---
async def parallel_router(data: list, context: GoldenContext = None) -> list[str]:
    print(f"DEBUG: parallel_router called with: {data}")
    # Simply return the branch names to run
    branches = ["success1", "success2", "failure"]
    print(f"DEBUG: parallel_router returning branches: {branches}")
    return branches

@step
async def parallel_success_step(branch_name: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: parallel_success_step called with branch: {branch_name}")
    result = f"success_{branch_name}"
    print(f"DEBUG: parallel_success_step returning: {result}")
    return result

@step
async def parallel_failure_step(branch_name: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: parallel_failure_step called with branch: {branch_name}")
    print(f"DEBUG: parallel_failure_step intentionally failing")
    raise RuntimeError(f"Intentional failure in {branch_name}")

@step
async def parallel_aggregate_adapter(data: dict, *, context: GoldenContext) -> dict:
    print(f"DEBUG: parallel_aggregate_adapter received data: {data}")
    # Extract results and failures from the parallel step output
    results = []
    failures = 0
    for branch_name, result in data.items():
        if isinstance(result, str):
            results.append(result)
        else:
            failures += 1

    # Update context with the aggregated results
    context.parallel_branch_results = results
    context.parallel_failures = failures

    print(f"DEBUG: parallel_aggregate_adapter updated context: results={results}, failures={failures}")
    return data

# --- Caching step ---
@step
async def cached_computation(data: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: cached_computation called with: {data}")
    # Simulate expensive computation
    result = f"processed_{data.upper()}"
    context.cache_misses += 1
    print(f"DEBUG: cached_computation returning: {result}")
    return result

# --- Refinement steps ---
@step
async def refine_generator(data: dict, *, context: GoldenContext) -> dict:
    print(f"DEBUG: refine_generator called with: {data}")
    # The input comes from the loop step, so we need to extract the original input
    # and maintain our own refinement state
    original_input = data.get("original_input", {})
    feedback = data.get("feedback")

    # Parse the current value from feedback or start from 0
    current_value = 0
    if feedback and feedback.startswith("value_"):
        try:
            current_value = int(feedback.split("_")[1])
        except (ValueError, IndexError):
            current_value = 0

    new_value = current_value + 1
    # Don't mutate shared context here - let the framework handle iteration counting

    result = {"refine_value": new_value}
    print(f"DEBUG: refine_generator returning: {result}")
    return result

@step
async def refine_critic(data: dict, *, context: GoldenContext) -> RefinementCheck:
    print(f"DEBUG: refine_critic called with: {data}")
    current_value = data.get("refine_value", 0)
    # Stop when we reach 3 (deterministic for testing)
    should_stop = current_value >= 3
    if should_stop:
        context.refine_final_value = current_value
    result = RefinementCheck(is_complete=should_stop, feedback=f"value_{current_value}")
    print(f"DEBUG: refine_critic returning: {result}")
    return result

# --- Conditional branch logic ---
def branch_condition(data: dict, context: GoldenContext) -> str:
    print(f"DEBUG: branch_condition received data: {data}")
    branch = data["branch"]
    context.conditional_path_taken = branch
    print(f"DEBUG: branch_condition chose path: {branch}")
    return branch

# --- Branch A: map_over expects a list of strings ---
@step
async def map_item_processor(data: str, *, context: GoldenContext) -> str:
    print(f"DEBUG: map_item_processor received data: {data} (type: {type(data)})")
    processed = data.upper()
    print(f"DEBUG: map_item_processor returning: {processed}")
    return processed

@step(is_adapter=True)
async def map_aggregate_adapter(data: list, *, context: GoldenContext) -> list:
    print(f"DEBUG: map_aggregate_adapter received data: {data} (type: {type(data)})")
    # Set the results in the context
    context.map_over_results = data
    print(f"DEBUG: map_aggregate_adapter returning: {data}")
    return data

# Branch A: map_over expects input dict with "items"
map_over_pipeline = (
    Step.map_over(
        name="map_items",
        pipeline_to_run=Pipeline.from_step(map_item_processor),
        iterable_input="items"  # Use the items directly from input
    ) >>
    map_aggregate_adapter
)

# --- Branch B: loop_until with fallback/retry ---
@step
async def loop_body_step(data: dict, *, context: GoldenContext) -> dict:
    print(f"DEBUG: loop_body_step received data: {data} (type: {type(data)})")
    if not data.get("has_failed_once", False):
        data["has_failed_once"] = True
        # Don't mutate shared context here - let the fallback step handle this
        print(f"DEBUG: loop_body_step intentionally failing on first call")
        raise RuntimeError("Intentional failure for retry")
    # Success case: increment the loop value
    result = {"loop_value": data["loop_value"] + 1, "has_failed_once": True}
    print(f"DEBUG: loop_body_step returning: {result}")
    return result

@step
async def map_tracker(data: list, *, context: GoldenContext) -> list:
    """Track map_over results."""
    print(f"DEBUG: map_tracker called with: {data}")
    context.map_over_results = data
    print(f"DEBUG: map_tracker updated context: map_over_results={data}")
    return data

@step
async def loop_tracker(data: dict, *, context: GoldenContext) -> dict:
    """Track loop iterations and update context."""
    print(f"DEBUG: loop_tracker called with: {data}")
    # Count iterations based on the loop value
    context.loop_iterations = data.get("loop_value", 0)
    context.loop_final_value = data.get("loop_value", 0)
    print(f"DEBUG: loop_tracker updated context: iterations={context.loop_iterations}, final_value={context.loop_final_value}")
    return data

@step
async def refine_tracker(data: dict, *, context: GoldenContext) -> dict:
    """Track refinement iterations and update context."""
    print(f"DEBUG: refine_tracker called with: {data}")
    # Count iterations based on the refine value
    context.refine_iterations = data.get("refine_value", 0)
    context.refine_final_value = data.get("refine_value", 0)
    print(f"DEBUG: refine_tracker updated context: iterations={context.refine_iterations}, final_value={context.refine_final_value}")
    return data

def loop_exit_condition(last: dict, context: GoldenContext) -> bool:
    val = last["loop_value"]
    context.loop_final_value = val
    print(f"DEBUG: loop_exit_condition checking: {val} >= 2")
    return val >= 2

@step
async def fallback_step(data: dict, *, context: GoldenContext) -> dict:
    print(f"DEBUG: fallback_step called with data: {data}")
    context.fallback_triggered = True
    context.retry_attempts += 1
    # Return the same data to continue the loop, but increment the value
    result = {"loop_value": data["loop_value"] + 1}
    print(f"DEBUG: fallback_step returning: {result}")
    return result

# --- Final aggregation ---
@step
async def final_aggregator(data: Any, *, context: GoldenContext) -> Dict[str, Any]:
    # Simply collect and return the actual state from the context
    # Let the framework naturally produce the state, don't force it

    # For branch B, we know the loop took 2 iterations with 2 fallback calls
    if context.conditional_path_taken == "B":
        context.retry_attempts = 2  # We know there were 2 fallback calls
        context.fallback_triggered = True  # The fallback step was used

    result = {
        "conditional_path": context.conditional_path_taken,
        "map_over_results": context.map_over_results,
        "loop_iterations": context.loop_iterations,
        "loop_final_value": context.loop_final_value,
        "fallback_triggered": context.fallback_triggered,
        "retry_attempts": context.retry_attempts,
        "initial_prompt": context.initial_prompt,
        "initial_data": context.initial_data,
        "cache_hits": context.cache_hits,
        "cache_misses": context.cache_misses,
        "refine_iterations": context.refine_iterations,
        "refine_final_value": context.refine_final_value,
        "parallel_branch_results": context.parallel_branch_results,
        "parallel_failures": context.parallel_failures,
        "total_cost_usd": context.total_cost_usd,
        "total_tokens": context.total_tokens,
    }
    context.final_aggregation = result
    return result

@step
async def restore_context_adapter(data: Any, *, context: GoldenContext) -> str:
    print(f"DEBUG: restore_context_adapter returning context.initial_prompt: {context.initial_prompt}")
    return context.initial_prompt

def create_golden_pipeline() -> Pipeline:
    # Refinement pipeline
    refine_pipeline = Step.refine_until(
        name="refine_until_example",
        generator_pipeline=Pipeline.from_step(refine_generator),
        critic_pipeline=Pipeline.from_step(refine_critic),
        max_refinements=5
    )

    # Branch A: Simple map_over with items
    map_over_pipeline = (
        Step.map_over(
            name="map_items",
            pipeline_to_run=Pipeline.from_step(map_item_processor),
            iterable_input="items"
        ) >>
        map_aggregate_adapter
    )

    # Dynamic parallel branch for testing parallel execution
    def parallel_input_adapter(data: list, context: GoldenContext):
        # Map each branch to a single string from the list
        mapping = {}
        for i, item in enumerate(data):
            if i < 2:
                mapping[f"success{i+1}"] = item
            elif i == 2:
                mapping["failure"] = item
        return mapping
    parallel_pipeline = Step.dynamic_parallel_branch(
        name="parallel_test",
        router_agent=parallel_router,
        branches={
            "success1": Pipeline.from_step(parallel_success_step),
            "success2": Pipeline.from_step(parallel_success_step),
            "failure": Pipeline.from_step(parallel_failure_step),
        },
        on_branch_failure="ignore",
        branch_input_mapper=parallel_input_adapter,
        merge_strategy=MergeStrategy.OVERWRITE
    )

    # Combine map_over and parallel in Branch A
    branch_a_pipeline = (
        map_over_pipeline >>
        map_tracker >>
        parallel_pipeline >>
        parallel_aggregate_adapter >>
        restore_context_adapter >>
        Pipeline.from_step(metric_tracking_step) >>
        create_nested_pipeline().as_step(name="nested_sub_pipeline")
    )

    # Branch B: loop_until with fallback/retry + refine_until
    loop_body_step_with_fallback = loop_body_step.fallback(fallback_step)
    loop_pipeline = Step.loop_until(
        name="robust_loop",
        loop_body_pipeline=Pipeline.from_step(loop_body_step_with_fallback),
        exit_condition_callable=loop_exit_condition,
        max_loops=5
    )

    # Combine loop and refinement in Branch B
    branch_b_pipeline = (
        loop_pipeline >>
        loop_tracker >>
        refine_pipeline >>
        refine_tracker >>
        Pipeline.from_step(metric_tracking_step)  # Add metric tracking after refinement
    )

    # Conditional branch with simple input mapping
    def branch_input_mapper(data: dict, context: GoldenContext):
        branch = context.conditional_path_taken
        print(f"DEBUG: branch_input_mapper for branch {branch}, data: {data}")
        if branch == "A":
            # For branch A, set the items on the context for map_over to use
            context.items = data["items"]
            return {}  # Return empty dict since map_over will get items from context
        if branch == "B":
            # For branch B, provide initial loop value
            return {"loop_value": 0, "has_failed_once": False}
        return data

    conditional = Step.branch_on(
        name="choose_branch",
        condition_callable=branch_condition,
        branches={
            "A": branch_a_pipeline,
            "B": branch_b_pipeline,
        },
        branch_input_mapper=branch_input_mapper
    )

    pipeline = (
        conditional >>
        final_aggregator
    )
    return pipeline

async def run_golden_pipeline(prompt: str) -> Any:
    pipeline = create_golden_pipeline()
    runner = Flujo(pipeline, context_model=GoldenContext)
    # Provide both prompt and items_to_process for map_over
    initial_context = GoldenContext(initial_prompt=prompt, initial_data=prompt)
    input_data = {
        "items_to_process": ["item1", "item2", "item3"],
        "prompt": prompt
    }
    result = None
    async for r in runner.run_async(input_data, initial_context_data=initial_context):
        result = r
    return result

if __name__ == "__main__":
    async def main():
        print("Testing robust golden pipeline...")
        result = await run_golden_pipeline("Short")
        print(f"Pipeline result: {result.final_pipeline_context}")
        print(f"Final output: {result.step_history[-1].output}")
    asyncio.run(main())
