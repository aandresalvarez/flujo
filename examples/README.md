# Examples – pydantic-ai-orchestrator

| File | What it shows |
|------|---------------|
| **00_quickstart.py** | Minimal, single-task run (ratio scorer). |
| **01_weighted_scoring.py** | Passing custom checklist weights. |
| **02_custom_agents.py** | Mixing models / replacing the solution agent. |
| **03_reward_scorer.py** | Using the reward-model scorer. |
| **04_batch_processing.py** | Running many prompts from a CSV and exporting results. |

Each script is standalone – activate your virtualenv, set `ORCH_OPENAI_API_KEY`, then:

```bash
python examples/00_quickstart.py
``` 