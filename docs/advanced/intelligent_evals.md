# Intelligent Evaluations

This guide explains how to run automated evaluations and use the self-improvement agent introduced in v2.1.

## Quick start

`flujo` integrates with `pydantic-evals` to provide automated evaluation and self-improvement capabilities. The `run_pipeline_async` function acts as an adapter, allowing you to use a `Flujo` instance as a task function for `pydantic-evals`.

```python
from flujo.application.eval_adapter import run_pipeline_async
from flujo.application.self_improvement import evaluate_and_improve, SelfImprovementAgent
from flujo import Flujo
from flujo.domain import Step
from pydantic_evals import Dataset, Case
from flujo.infra.agents import self_improvement_agent

# Define your Flujo pipeline
pipeline = Step.solution(lambda x: x)  # Replace with your actual pipeline
runner = Flujo(pipeline)

# Define your evaluation dataset
dataset = Dataset(cases=[
    Case(inputs="What is the capital of France?", expected_output="Paris"),
    Case(inputs="What is 2 + 2?", expected_output="4"),
])

# Create a SelfImprovementAgent
agent = SelfImprovementAgent(self_improvement_agent)

# Run the evaluation and self-improvement process
report = await evaluate_and_improve(
    lambda x: run_pipeline_async(x, runner=runner),  # Use the adapter here
    dataset,
    agent,
)

print(report)
```

The `ImprovementReport` contains structured suggestions for updating your pipeline or evaluation suite based on the evaluation results.

## ImprovementSuggestion Model

The self-improvement agent returns an `ImprovementReport` which contains a list of `ImprovementSuggestion` objects. Each suggestion has a `suggestion_type` indicating the general category of improvement along with additional fields describing the issue and proposed fix.

```
class ImprovementSuggestion(BaseModel):
    target_step_name: Optional[str]
    suggestion_type: SuggestionType
    failure_pattern_summary: str
    detailed_explanation: str
    prompt_modification_details: Optional[PromptModificationDetail]
    config_change_details: Optional[List[ConfigChangeDetail]]
    example_failing_input_snippets: List[str]
    suggested_new_eval_case_description: Optional[str]
    estimated_impact: Optional[Literal["HIGH", "MEDIUM", "LOW"]]
    estimated_effort_to_implement: Optional[Literal["HIGH", "MEDIUM", "LOW"]]
```

`SuggestionType` values include things like `PROMPT_MODIFICATION`, `CONFIG_ADJUSTMENT`, and `NEW_EVAL_CASE`. For prompt modifications or config adjustments, the relevant detail objects provide the exact change proposed.

## End-to-end Example

1. Define a simple pipeline and dataset with a failing case.
2. Run `flujo improve pipeline.py data.py`.
3. Review the suggestions printed in the formatted table.
4. Apply one of the suggested prompt tweaks.
5. Re-run the evaluation to see the improvement.

Suggestions are advisory and may vary in quality depending on the underlying LLM.

## Context for SelfImprovementAgent

When building the prompt for the self-improvement agent, each step now includes its `StepConfig` parameters and a redacted summary of the step's system prompt. This gives the agent more insight into how your pipeline is configured and helps it provide targeted `CONFIG_ADJUSTMENT` or `PROMPT_MODIFICATION` suggestions.

The `evaluate_and_improve` function constructs a detailed prompt for the `SelfImprovementAgent` by summarizing both failed and successful evaluation cases. This context includes:

*   **Failed Cases**: For each failed case, it provides the input, the output of each step in the pipeline, whether the step succeeded or failed, and any feedback. It also includes the `StepConfig` parameters (retries, timeout, temperature) and a redacted summary of the agent's system prompt for each step.
*   **Successful Example (if available)**: If there's at least one successful case, it's included to provide a positive example for the agent to learn from.

This structured context helps the `SelfImprovementAgent` to accurately diagnose issues and propose relevant improvements.

Example snippet of the context:

```
Case: test_sql_error
- GenerateSQL: Output(content="SELEC * FROM t") (success=True)
  Config(retries=1, timeout=30s, temperature=0.7)
  SystemPromptSummary: "You are a SQL expert..."
```

## Acting on Suggestions: Adding New Evaluation Cases

For suggestions of type `NEW_EVAL_CASE`, use the helper command:

```bash
flujo add-eval-case -d path/to/my_evals.py -n test_new_case -i "user input"
```

The command prints a `Case(...)` definition that you can copy into your dataset
file.

## Configuring the Self-Improvement Agent

The model used by the self-improvement agent can be changed via the
`default_self_improvement_model` setting or overridden at the CLI using
`flujo improve --improvement-model MODEL_NAME`.
### Interpreting Suggestion Types
The `suggestion_type` field indicates how you might act on the advice:
- **PROMPT_MODIFICATION** – adjust the text of a step's system prompt as described.
- **CONFIG_ADJUSTMENT** – tweak temperature or other parameters in the step configuration.
- **NEW_EVAL_CASE / EVAL_CASE_REFINEMENT** – create or refine dataset cases to exercise the pipeline more thoroughly.
- **OTHER** – miscellaneous guidance not captured by the above categories.

### Dataset Best Practices
When authoring evaluation datasets:
- Provide clear `expected_output` values so failures are easy to diagnose.
- Give cases descriptive names using the `name` field.
- Include metadata if extra context helps the agent understand the scenario.

### Limitations
Self‑improvement suggestions are generated by an LLM and should be reviewed
critically. The agent does not modify your code automatically.

## Custom Evaluators

`flujo` provides custom evaluators to simplify the integration with `pydantic-evals`.

### Multi-Signal Evaluator (Critic/Judge Loop)

The critic-judge loop does not require a separate pattern. `MultiSignalEvaluator` already
supports it via `review_agent` (critic) and `validator_agent` (judge), with optional
objective validators for hard checks.

```python
from flujo.domain.evaluation import MultiSignalEvaluator
from flujo.agents import make_agent_async

critic = make_agent_async("openai:gpt-4o", "Critique the solution using a checklist.", dict)
judge = make_agent_async("openai:gpt-4o", "Validate the checklist strictly.", dict)

evaluator = MultiSignalEvaluator(
    review_agent=critic,
    validator_agent=judge,
    objective_validators=[],
)

report = await evaluator.run({"solution": output, "task": prompt})
```

### `FinalSolutionEvaluator`

The `FinalSolutionEvaluator` extracts the output of the last step in a `PipelineResult` and compares it against the `expected_output` defined in your evaluation `Case`. This is useful when you want to evaluate the overall outcome of your pipeline.

```python
from flujo import Flujo, Step
from flujo.application.evaluators import FinalSolutionEvaluator
from flujo.testing.utils import StubAgent
from pydantic_evals import Dataset, Case

# Define a simple pipeline
pipeline = (
    Step.solution(StubAgent(["intermediate result"]))
    >> Step.solution(StubAgent(["final result"]))
)
runner = Flujo(pipeline)

# Define a dataset with an expected final output
dataset = Dataset(cases=[
    Case(
        inputs="some input",
        expected_output="final result",
        evalu_fn=FinalSolutionEvaluator(),
    ),
])

async def run_evaluation():
    report = await dataset.evaluate(lambda x: runner.arun(x))
    print(report)

# To run this example, you would typically call run_evaluation() in an async context.
```
