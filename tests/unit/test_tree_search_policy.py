import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultTreeSearchStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.tree_search import TreeSearchStep
from flujo.domain.evaluation import EvaluationReport
from flujo.domain.models import PipelineContext, Quota, UsageEstimate, UsageLimits
from flujo.domain.validation import ValidationResult
from flujo.exceptions import PausedException, UsageLimitExceededError


class _ProposerAgent:
    def __init__(self, proposals: list[str]) -> None:
        self.proposals = proposals
        self.last_prompt: str | None = None

    async def run(self, data, **_kwargs):
        self.last_prompt = str(data)
        return list(self.proposals)


class _EvaluatorAgent:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0
        self.last_prompt: str | None = None

    async def run(self, data, **_kwargs):
        self.calls += 1
        self.last_prompt = str(data)
        return float(self.score)


def _extract_candidate(prompt: str) -> str:
    marker = "Candidate:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip().splitlines()[0]
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    return lines[-1] if lines else ""


class _ContextTaggingEvaluator:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def run(self, data, *, context=None, **_kwargs):
        candidate = _extract_candidate(str(data))
        if context is not None:
            tags = getattr(context, "branch_tags", None)
            if tags is None:
                tags = []
                setattr(context, "branch_tags", tags)
            tags.append(candidate)
        return float(self.scores.get(candidate, 0.0))


class _StagedProposerAgent:
    def __init__(self, first: list[str], later: list[str]) -> None:
        self.first = list(first)
        self.later = list(later)
        self.calls = 0

    async def run(self, data, **_kwargs):
        self.calls += 1
        return list(self.first) if self.calls == 1 else list(self.later)


class _RecordingStagedProposerAgent(_StagedProposerAgent):
    def __init__(self, first: list[str], later: list[str]) -> None:
        super().__init__(first, later)
        self.prompts: list[str] = []

    async def run(self, data, **_kwargs):
        self.prompts.append(str(data))
        return await super().run(data, **_kwargs)


class _DiscoveryAgent:
    def __init__(self, rules: list[str]) -> None:
        self.rules = list(rules)
        self.calls = 0

    async def run(self, _data, **_kwargs):
        self.calls += 1
        return list(self.rules)


class _PausingEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self._paused = False

    async def run(self, data, **_kwargs):
        self.calls += 1
        candidate = _extract_candidate(str(data))
        if candidate == "second" and not self._paused:
            self._paused = True
            raise PausedException("pause")
        if candidate == "finish":
            return 1.0
        return 0.1


class _DiffEvaluator:
    def __init__(self, patch_len: int) -> None:
        self.patch_len = patch_len

    async def run(self, _data, **_kwargs):
        patch = [{"op": "replace", "path": "/value", "value": idx} for idx in range(self.patch_len)]
        return EvaluationReport(score=0.9, diff={"patch": patch})


class _ValidationResultEvaluator:
    def __init__(self, score: float, patch_len: int) -> None:
        self.score = score
        self.patch_len = patch_len

    async def run(self, _data, **_kwargs):
        patch = [{"op": "replace", "path": "/value", "value": idx} for idx in range(self.patch_len)]
        return ValidationResult(
            is_valid=True,
            score=self.score,
            diff={"patch": patch},
            validator_name="TestValidator",
        )


@pytest.mark.asyncio
async def test_tree_search_dedup_skips_duplicate_candidates():
    core = ExecutorCore()
    proposer = _ProposerAgent(["dup", "dup"])
    evaluator = _EvaluatorAgent(0.5)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=2,
        beam_width=2,
        max_depth=1,
    )
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=PipelineContext(initial_prompt="goal"),
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True
    assert evaluator.calls == 1


@pytest.mark.asyncio
async def test_tree_search_goal_pinning_in_prompts():
    core = ExecutorCore()
    proposer = _ProposerAgent(["next"])
    evaluator = _EvaluatorAgent(0.4)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=1,
        beam_width=1,
        max_depth=1,
    )
    ctx = PipelineContext(initial_prompt="find solution")
    frame = make_execution_frame(
        core,
        step,
        data="find solution",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True
    assert proposer.last_prompt is not None
    assert evaluator.last_prompt is not None
    assert proposer.last_prompt.startswith("Primary Objective: find solution")
    assert evaluator.last_prompt.startswith("Primary Objective: find solution")


@pytest.mark.asyncio
async def test_tree_search_quota_reservation_denied():
    core = ExecutorCore()

    class _Estimator:
        def estimate(self, *_args, **_kwargs) -> UsageEstimate:
            return UsageEstimate(cost_usd=0.0, tokens=999)

    core._usage_estimator = _Estimator()
    core._estimator_factory = None
    proposer = _ProposerAgent(["next"])
    evaluator = _EvaluatorAgent(0.0)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=1,
        beam_width=1,
        max_depth=1,
    )
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=PipelineContext(initial_prompt="goal"),
        resources=None,
        limits=UsageLimits(total_tokens_limit=1),
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    with pytest.raises(UsageLimitExceededError):
        await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)


@pytest.mark.asyncio
async def test_tree_search_quota_reconciles_estimates():
    core = ExecutorCore()

    class _Estimator:
        def estimate(self, *_args, **_kwargs) -> UsageEstimate:
            return UsageEstimate(cost_usd=0.0, tokens=5)

    core._usage_estimator = _Estimator()
    core._estimator_factory = None
    proposer = _ProposerAgent(["next"])
    evaluator = _EvaluatorAgent(0.0)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=1,
        beam_width=1,
        max_depth=1,
    )
    ctx = PipelineContext(initial_prompt="goal")
    quota = Quota(remaining_cost_usd=float("inf"), remaining_tokens=10)
    token = core._set_current_quota(quota)
    try:
        frame = make_execution_frame(
            core,
            step,
            data="goal",
            context=ctx,
            resources=None,
            limits=UsageLimits(total_tokens_limit=10),
            context_setter=None,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            result=None,
            quota=quota,
        )
        outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    finally:
        core._reset_current_quota(token)
    assert outcome.step_result.success is True
    remaining_cost_usd, remaining_tokens = quota.get_remaining()
    assert remaining_tokens == 10


@pytest.mark.asyncio
async def test_tree_search_context_isolation_merges_winner_only():
    core = ExecutorCore()
    proposer = _ProposerAgent(["loser", "winner"])
    evaluator = _ContextTaggingEvaluator({"loser": 0.1, "winner": 1.0})
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=2,
        beam_width=2,
        max_depth=1,
        goal_score_threshold=1.0,
        require_goal=False,
    )
    ctx = PipelineContext(initial_prompt="goal", branch_tags=[])
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True
    assert ctx.branch_tags == ["winner"]


@pytest.mark.asyncio
async def test_tree_search_resume_after_pause_restores_frontier():
    core = ExecutorCore()
    proposer = _StagedProposerAgent(["first", "second"], ["finish"])
    evaluator = _PausingEvaluator()
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=2,
        beam_width=2,
        max_depth=2,
        goal_score_threshold=0.9,
        require_goal=True,
    )
    ctx = PipelineContext(initial_prompt="goal")
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(PausedException):
        await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)

    state = ctx.tree_search_state
    assert state is not None
    assert state.status == "paused"
    assert state.open_set
    iterations_before = state.iterations

    frame2 = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame2)
    assert outcome.step_result.success is True
    assert ctx.tree_search_state is not None
    assert ctx.tree_search_state.status == "completed"
    assert ctx.tree_search_state.iterations > iterations_before


@pytest.mark.asyncio
async def test_tree_search_discovery_runs_once_on_resume():
    core = ExecutorCore()
    proposer = _StagedProposerAgent(["first", "second"], ["finish"])
    evaluator = _PausingEvaluator()
    discovery = _DiscoveryAgent(["output != 'forbidden'"])
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        discovery_agent=discovery,
        branching_factor=2,
        beam_width=2,
        max_depth=2,
        goal_score_threshold=0.9,
        require_goal=True,
    )
    ctx = PipelineContext(initial_prompt="goal")
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )

    with pytest.raises(PausedException):
        await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)

    assert discovery.calls == 1
    assert ctx.tree_search_state is not None
    assert ctx.tree_search_state.deduced_invariants == ["output != 'forbidden'"]

    frame2 = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame2)
    assert outcome.step_result.success is True
    assert discovery.calls == 1


@pytest.mark.asyncio
async def test_tree_search_uses_diff_heuristic_when_present():
    core = ExecutorCore()
    proposer = _ProposerAgent(["next"])
    evaluator = _DiffEvaluator(2)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=1,
        beam_width=1,
        max_depth=1,
        require_goal=False,
    )
    ctx = PipelineContext(initial_prompt="goal")
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True

    state = ctx.tree_search_state
    assert state is not None
    depth_nodes = [node for node in state.nodes.values() if node.depth == 1]
    assert depth_nodes
    assert depth_nodes[0].h_cost == 2.0
    assert depth_nodes[0].evaluation is not None
    assert depth_nodes[0].evaluation.get("heuristic_source") == "diff"


@pytest.mark.asyncio
async def test_tree_search_invariant_violation_injects_prompt():
    core = ExecutorCore()
    proposer = _RecordingStagedProposerAgent(["bad", "good"], ["finish"])
    evaluator = _ContextTaggingEvaluator({"good": 0.4, "finish": 1.0})
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        static_invariants=["output != 'bad'"],
        branching_factor=2,
        beam_width=2,
        max_depth=2,
        goal_score_threshold=1.0,
        require_goal=True,
    )
    ctx = PipelineContext(initial_prompt="goal")
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True
    assert len(proposer.prompts) >= 2
    assert "Invariant Violations:" in proposer.prompts[1]
    assert "output != 'bad'" in proposer.prompts[1]


@pytest.mark.asyncio
async def test_tree_search_uses_validation_result_score_and_diff():
    core = ExecutorCore()
    proposer = _ProposerAgent(["next"])
    evaluator = _ValidationResultEvaluator(score=0.8, patch_len=3)
    step = TreeSearchStep(
        name="ts",
        proposer=Step(name="proposer", agent=proposer),
        evaluator=Step(name="evaluator", agent=evaluator),
        branching_factor=1,
        beam_width=1,
        max_depth=1,
        goal_score_threshold=0.7,
        require_goal=True,
    )
    ctx = PipelineContext(initial_prompt="goal")
    frame = make_execution_frame(
        core,
        step,
        data="goal",
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultTreeSearchStepExecutor().execute(core=core, frame=frame)
    assert outcome.step_result.success is True

    state = ctx.tree_search_state
    assert state is not None
    depth_nodes = [node for node in state.nodes.values() if node.depth == 1]
    assert depth_nodes
    assert depth_nodes[0].h_cost == 3.0
    assert depth_nodes[0].metadata.get("rubric_score") == 0.8
