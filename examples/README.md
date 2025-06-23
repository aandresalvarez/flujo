# Examples – flujo

| File | What it shows |
|------|---------------|
| **00_quickstart.py** | Hello World with the Default recipe. |
| **01_weighted_scoring.py** | Weighted scoring to prioritize docstrings. |
| **02_custom_agents.py** | Building creative agents with custom prompts. |
| **03_reward_scorer.py** | Using an LLM judge via RewardScorer. |
| **04_batch_processing.py** | Running multiple workflows concurrently. |
| **05_pipeline_sql.py** | Pipeline DSL with SQL validation plugin. |
| **06_typed_context.py** | Sharing state with Typed Pipeline Context. |
| **07_loop_step.py** | Iterative refinement using LoopStep. |
| **08_branch_step.py** | Dynamic routing with ConditionalStep. |
| **10_cost_control.py** | Enforcing usage limits to control cost. |
| **11_stateful_hitl.py** | Multi-turn correction loop with simulated HITL. |
| **12_using_resources.py** | Dependency injection via AppResources. |
| **13_lifecycle_hooks.py** | Observing pipeline events with hooks. |

Each script is standalone – activate your virtualenv, set `OPENAI_API_KEY`, then:

```bash
python examples/00_quickstart.py
``` 