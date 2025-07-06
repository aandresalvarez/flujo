from flujo import Pipeline, step
from flujo.domain.models import BaseModel


class Ctx(BaseModel):
    num: int = 0


@step
async def deprecated_step(x: int, *, context: Ctx) -> int:
    return x


@step
async def modern_step(x: int, *, context: Ctx) -> int:
    return x


def test_no_warning_for_context() -> None:
    pipeline = Pipeline.from_step(deprecated_step)
    report = pipeline.validate_graph()
    assert not any(f.rule_id == "V-A4" for f in report.warnings)


def test_no_warning_for_modern_context() -> None:
    pipeline = Pipeline.from_step(modern_step)
    report = pipeline.validate_graph()
    assert not any(f.rule_id == "V-A4" for f in report.warnings)
