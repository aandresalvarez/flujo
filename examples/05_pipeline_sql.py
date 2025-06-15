"""
05_pipeline_sql.py
-------------------
Demonstrates the Pipeline DSL and SQL validator plugin.
"""

from flujo import Step, Flujo
from flujo.plugins.sql_validator import SQLSyntaxValidator
from flujo.testing.utils import StubAgent

# Solution agent that returns an invalid SQL statement
solution = StubAgent(["SELECT FROM"])
# Validator agent doesn't matter for syntax check
validator = StubAgent([None])

solution_step = Step.solution(solution)
validation_step = Step.validate(validator, plugins=[SQLSyntaxValidator()])

pipeline = solution_step >> validation_step
runner = Flujo(pipeline)

result = runner.run("SELECT FROM")
for step_result in result.step_history:
    print(step_result.name, step_result.success, step_result.feedback)
