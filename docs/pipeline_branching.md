# ConditionalStep: Branching Pipelines

`ConditionalStep` lets you choose between multiple sub-pipelines at runtime. A callable decides which branch to execute based on the previous output or the shared pipeline context.

## Parameters

- **`name`** – Step name.
- **`condition_callable`** – Function accepting `(previous_step_output, context)` and returning a key.
- **`branches`** – Dictionary mapping keys to `Pipeline` objects.
- **`default_branch_pipeline`** – Optional pipeline used when no key matches.
- **`branch_input_mapper`** – Optional function mapping the `ConditionalStep` input to the branch input.
- **`branch_output_mapper`** – Optional function mapping the branch output to the `ConditionalStep` output.

All callables receive the shared typed pipeline context if provided.

## Success and Failure

The `ConditionalStep` succeeds when the selected branch completes successfully. If no branch matches and no default is provided, the step fails. Failures inside the chosen branch propagate to the `ConditionalStep`.

`StepResult.metadata_['executed_branch_key']` stores the branch key that was executed.

## Example

```python
from flujo.domain import Step, Pipeline
from pydantic import BaseModel

async def classify(x: str) -> str:
    return "numbers" if x.isdigit() else "text"

async def process_numbers(data: str) -> str:
    return str(int(data) * 2)

async def process_text(data: str) -> str:
    return data.upper()

branches = {
    "numbers": Pipeline.from_step(Step("num", process_numbers)),
    "text": Pipeline.from_step(Step("txt", process_text)),
}

def select_branch(result: str, ctx: BaseModel | None) -> str:
    return result

branch_step = Step.branch_on(
    name="router",
    condition_callable=select_branch,
    branches=branches,
)

pipeline = Step("classify", classify) >> branch_step
```

Running the pipeline will execute either the `process_numbers` or `process_text` branch based on the classification result.

See [pipeline_dsl.md](pipeline_dsl.md) for an overview of the DSL. A runnable example can be found in [this script on GitHub](https://github.com/aandresalvarez/flujo/blob/main/examples/08_branch_step.py).


## Dynamic Parallel Router

`DynamicParallelRouterStep` extends branching by letting an agent decide which branches to execute in parallel at runtime. The router agent returns a list of branch names, and only those branches run.

```python
router = Step.dynamic_parallel_branch(
    name="router",
    router_agent=my_router_agent,
    branches={"billing": billing_pipe, "support": support_pipe},
)
```

The executed branch names are stored in `StepResult.metadata_["executed_branches"]`.
 
