def test_loop_meta_overrides_config_defaults(monkeypatch):
    # Capture HistoryManager cfg passed by policy wiring
    import flujo.application.conversation.history_manager as hm_mod

    captured = {"cfgs": []}
    orig_init = hm_mod.HistoryManager.__init__

    def _wrap_init(self, cfg=None):  # type: ignore[no-redef]
        captured["cfgs"].append(cfg)
        return orig_init(self, cfg)

    monkeypatch.setattr(hm_mod.HistoryManager, "__init__", _wrap_init)

    # Fake config manager defaults (truncate_tokens)
    class _FakeCfgMgr:
        def load_config(self):
            return {
                "conversation": {
                    "history_management": {"strategy": "truncate_tokens", "max_tokens": 9999}
                }
            }

    monkeypatch.setattr(hm_mod, "get_config_manager", lambda: _FakeCfgMgr(), raising=False)

    # Build a minimal conversational loop with explicit loop meta overrides
    from flujo.domain.dsl.step import Step
    from flujo.domain.dsl.pipeline import Pipeline
    from typing import Optional
    from flujo.domain.dsl.loop import LoopStep
    from flujo.application.runner import Flujo
    from flujo.domain.models import PipelineContext

    async def agent(_: str, *, context: Optional[PipelineContext] = None) -> str:
        return "ok"

    s = Step.from_callable(agent, name="s")
    body = Pipeline.from_step(s)

    def _exit(_out: str, _ctx: Optional[PipelineContext]) -> bool:
        return True

    loop = LoopStep(name="L", loop_body_pipeline=body, exit_condition_callable=_exit, max_retries=1)
    loop.meta["conversation"] = True
    # Override config defaults via loop meta
    loop.meta["history_management"] = {"strategy": "truncate_turns", "max_turns": 1}

    pipe = Pipeline.from_step(loop)
    runner = Flujo(pipe)

    # Execute to trigger policy wiring and processor creation
    async def _run():
        res = None
        async for r in runner.run_async("hello"):
            res = r
        return res

    import asyncio

    result = asyncio.run(_run())
    # Accept either a StepResult (legacy) or PipelineResult (current)
    assert result is not None
    success_attr = getattr(result, "success", None)
    if success_attr is None:
        # Assume PipelineResult; check last step
        assert getattr(result, "step_history", None), "No steps executed"
        assert result.step_history[-1].success is True
    else:
        assert success_attr is True

    # Assert that the cfg used by HistoryManager reflects loop meta, not config default
    assert captured["cfgs"], "HistoryManager was not constructed by policy"
    last_cfg = next(c for c in captured["cfgs"] if c is not None)
    assert last_cfg.strategy == "truncate_turns"
