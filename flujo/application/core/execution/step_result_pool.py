"""Lightweight StepResult pooling for hot-path creation."""

from __future__ import annotations

from dataclasses import dataclass, field

from flujo.domain.models import StepResult


@dataclass
class _StepResultPool:
    max_size: int = 256
    _pool: list[StepResult] = field(default_factory=list)

    def acquire(self) -> StepResult:
        if self._pool:
            sr = self._pool.pop()
            self._reset(sr)
            return sr
        return StepResult(
            name="",
            output=None,
            success=True,
            attempts=0,
            latency_s=0.0,
            token_counts=0,
            cost_usd=0.0,
            feedback=None,
            branch_context=None,
            metadata_={},
            step_history=[],
        )

    def release(self, sr: StepResult) -> None:
        if len(self._pool) >= self.max_size:
            return
        self._reset(sr)
        self._pool.append(sr)

    def _reset(self, sr: StepResult) -> None:
        sr.name = ""
        sr.output = None
        sr.success = True
        sr.attempts = 0
        sr.latency_s = 0.0
        sr.token_counts = 0
        sr.cost_usd = 0.0
        sr.feedback = None
        sr.branch_context = None
        sr.metadata_.clear()
        sr.step_history.clear()


_STEP_RESULT_POOL = _StepResultPool()


def build_pooled_step_result(
    *,
    name: str,
    success: bool,
    output: object = None,
    attempts: int = 0,
    latency_s: float = 0.0,
    token_counts: int | dict[str, object] = 0,
    cost_usd: float = 0.0,
    feedback: str | None = None,
    branch_context: object | None = None,
    metadata: dict[str, object] | None = None,
    step_history: list[StepResult] | None = None,
) -> StepResult:
    """Build a StepResult using a pooled instance to reduce allocations."""
    sr = _STEP_RESULT_POOL.acquire()
    sr.name = name
    sr.success = success
    sr.output = output
    sr.attempts = attempts
    sr.latency_s = latency_s
    if isinstance(token_counts, dict):
        token_total = token_counts.get("total", 0)
        if isinstance(token_total, (int, bool)):
            sr.token_counts = int(token_total)
        elif isinstance(token_total, float):
            sr.token_counts = int(token_total)
        elif isinstance(token_total, str):
            try:
                sr.token_counts = int(token_total)
            except ValueError:
                sr.token_counts = 0
        else:
            sr.token_counts = 0
    else:
        sr.token_counts = token_counts
    sr.cost_usd = cost_usd
    sr.feedback = feedback
    sr.branch_context = branch_context
    sr.metadata_.update(metadata or {})
    if step_history:
        sr.step_history.extend(step_history)
    # Return a deep copy to callers to avoid shared mutable state, then recycle base object.
    result = sr.model_copy(deep=True)
    _STEP_RESULT_POOL.release(sr)
    return result


__all__ = ["build_pooled_step_result"]
