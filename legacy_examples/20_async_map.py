"""Example: Mapping a pipeline over context data asynchronously.

See docs/cookbook/async_map.md for details.
"""

from flujo import Flujo, Step, Pipeline
from flujo.domain.models import PipelineContext


class Numbers(PipelineContext):
    values: list[int]


body = Pipeline.from_step(Step.from_mapper(lambda x: x * 2, name="double"))
map_step = Step.map_over("map", body, iterable_input="values")

runner = Flujo(map_step, context_model=Numbers)

if __name__ == "__main__":
    result = runner.run(None, initial_context_data={"values": [1, 2, 3]})
    print("Mapped results:", result.step_history[-1].output)
