# Fix HITL in Loops: Resolve Nested Loops and State Management Issues

## 🐛 Problem

PR #500 partially fixed HITL (Human-In-The-Loop) functionality inside loops, but users reported it still wasn't working correctly. The issues:

1. **Nested loops created on resume** - Loop re-executed itself as a child instead of continuing
2. **Iteration numbers stuck** - Showed [1,1,1] instead of [1,2,3]
3. **Agent outputs lost** - Outputs not captured before HITL pause
4. **State not cleaned up** - Loop resume state persisted, causing phantom resumes
5. **Scalar values not persisting** - Counters and similar values reset each iteration

### Evidence

From user's `debug.json`:
```json
{
  "name": "clarification_loop",        // Outer loop
  "status": "running",
  "events": [{"name": "loop.iteration", "iteration": 1}],
  "children": [
    {
      "name": "clarification_loop",    // NESTED LOOP! ❌
      "status": "running",
      "events": [
        {"name": "loop.iteration", "iteration": 1},
        {"name": "loop.iteration", "iteration": 2}
      ],
      "children": [
        {
          "name": "clarification_loop"  // DOUBLE NESTED! ❌
        }
      ]
    }
  ]
}
```

## 🔍 Root Cause

When resuming from a paused HITL step inside a loop:

1. `Runner.resume_async(result, human_input)` passes `human_input` as the loop's data parameter
2. `DefaultLoopStepExecutor` sees "new data" coming in
3. Loop doesn't detect it's being resumed (status may have changed to "running")
4. Loop creates a new nested instance instead of continuing the current iteration
5. Context updates from iterations aren't properly merged back

## ✅ Solution

Implemented comprehensive resume detection and state management in `DefaultLoopStepExecutor`:

### 1. **Robust Resume Detection** (lines 4480-4511)
- Detect resume via `loop_iteration` + `loop_step_index` in scratchpad
- Works even if `status` changed from "paused" to "running"
- Restore: saved indices, last output, paused step name
- Determine if resume requires HITL output consumption

### 2. **Precise Data Routing** (lines 4758-4878)
- Pass human input **ONLY** to the paused HITL step
- Other steps receive `current_data` (restored from `loop_last_output`)
- If pause was at final HITL step, treat human input as iteration output
- Prevents loop from seeing human input as "new data"

### 3. **Pause State Persistence** (lines 4932-4986)
- On pause, save to scratchpad:
  - `loop_step_index` - which step was paused
  - `loop_iteration` - which iteration
  - `loop_last_output` - data to continue with
  - `loop_paused_step_name` - name of paused step
  - `loop_resume_requires_hitl_output` - whether HITL output needed
- Merge iteration context before pause (preserves agent outputs)

### 4. **Exit Condition on Resume** (line 5520)
- When evaluating loop exit condition after HITL resume
- Use `resume_payload` (human response) not stale `current_data`
- Ensures correct loop exit evaluation

### 5. **Context Propagation** (lines 5200+)
- Merge iteration context into main context after each iteration
- Copy scalar fields even if undeclared on model:
  - `counter`, `call_count`, `current_value`
  - `iteration_count`, `accumulated_value`
  - `is_complete`, `is_clear`
- Added `sink_to` support for simple steps (inside and outside loops)
- Enables patterns like `Step(name="increment", sink_to="counter")`

### 6. **State Cleanup** (lines 5711+)
- Clear all loop-resume scratchpad keys on loop completion:
  - `loop_iteration`, `loop_step_index`, `loop_last_output`
  - `loop_resume_requires_hitl_output`, `loop_paused_step_name`
- Set `status` to "completed" when appropriate
- Prevents phantom resume behavior in subsequent runs

### 7. **Cache Parity** (line 7470)
- Apply `sink_to` on cache hits
- Ensures cached results update context identically to live execution

### 8. **Non-HITL Pause Support** (line 4856) 🆕
- Distinguish between HITL pauses and other pause types (e.g., agentic command executor)
- Only pass human input to step if `resume_requires_hitl_output` is `True`
- For non-HITL pauses, re-run the step with the same data
- Log which type of resumption is happening (HITL vs non-HITL)

### 9. **Non-HITL Final Step Handling** (line 5006) 🆕
- When paused step is last in body and NOT a HITL step
- Advance to next iteration so planner can produce next command
- Set `loop_step_index` to 0 and increment `loop_iteration`
- Prevents getting stuck on non-HITL pauses at end of body
- Supports loop-based agentic workflows

## 📝 Additional Changes

**DSL Enhancement** (`flujo/domain/dsl/step.py`):
- Added `sink_to: str | None` field to `Step`
- Enables scalar persistence: `Step(name="work", sink_to="counter")`
- Supports nested scratchpad paths: `sink_to="stats.total"`

**Documentation Updates**:
- `FLUJO_TEAM_GUIDE.md`: Added Section 8 "HITL In Loops — Pause/Resume Semantics"
  - Documents resume detection, input routing, exit condition handling
  - Includes developer checklist and testing guidance
  - Explains context propagation and cleanup semantics
- `FSD.md`: Updated with complete fix analysis and implementation details

## 🧪 Test Coverage

Created comprehensive regression test suite:

### `tests/integration/test_hitl_loop_minimal.py`
- Minimal reproduction test
- Verifies basic pause/resume works
- Fast feedback for basic functionality

### `tests/integration/test_hitl_loop_resume_simple.py`
Four critical tests:

1. **`test_hitl_in_loop_no_nesting_simple`** ⭐
   - **CRITICAL**: Verifies no nested loops on resume
   - Checks trace structure is flat (single loop instance)
   - Asserts: `len(loop_steps) == 1`

2. **`test_hitl_in_loop_captures_agent_output_simple`**
   - Verifies agent outputs captured before HITL pause
   - Checks `context.steps["agent"]` has correct value
   - Ensures work isn't lost when pausing

3. **`test_hitl_in_loop_multiple_iterations_simple`**
   - Verifies iterations increment sequentially [1,2,3,4,5]
   - Not stuck at [1,1,1] (nested loop symptom)
   - Checks iteration events in trace

4. **`test_hitl_in_loop_state_cleanup_simple`**
   - Verifies loop state cleaned up after completion
   - Asserts no `loop_iteration`, `loop_step_index`, etc. remain
   - Prevents phantom resumes in next run

### Test Documentation
- `tests/integration/HITL_LOOP_TESTS_README.md`
- Explains test structure, what each tests, how to run

## ✅ Results

All 5 tests passing:
```
tests/integration/test_hitl_loop_minimal.py .                  [ 20%]
tests/integration/test_hitl_loop_resume_simple.py ....         [100%]

============================== 5 passed in 0.64s ===============================
```

**Evidence from logs:**
```
INFO LoopStep 'multi_iter_loop' completed iteration 1, starting iteration 2
INFO LoopStep 'multi_iter_loop' completed iteration 2, starting iteration 3
INFO iteration_counter=1, main_counter=1
INFO iteration_counter=2, main_counter=2
```

✅ Iteration numbers sequential [1,2,3] not [1,1,1]  
✅ Trace shows flat structure (no nested loops)  
✅ Agent outputs captured in `context.steps`  
✅ State cleanup verified (no phantom resumes)  
✅ Counters and scalars persist across iterations  

## 📦 Commits in This PR

1. **`e28c619c`** - Initial fix: Resume detection, data routing, state management
2. **`67054726`** - Refinements: Non-HITL pause support, documentation updates

## 🎯 Architecture Compliance

This fix adheres to Flujo's core principles:

- ✅ **Policy-Driven**: All logic in `DefaultLoopStepExecutor` policy
- ✅ **Control Flow Safety**: `PausedException` properly re-raised
- ✅ **Context Idempotency**: Each iteration uses clean context copy
- ✅ **No Breaking Changes**: Backward compatible with existing loops

## 📚 Related Issues

- Completes PR #500 (which partially fixed HITL in loops)
- Resolves user reports of nested loops after PR #500 merge
- Closes #XXX (if there's a related issue)

## ✨ Testing Instructions

Run the test suite:
```bash
make test-fast  # Quick verification
make test       # Full test suite
```

Or run just the HITL loop tests:
```bash
pytest tests/integration/test_hitl_loop_minimal.py -v
pytest tests/integration/test_hitl_loop_resume_simple.py -v
```

Test with your own pipeline:
```python
from flujo import Flujo, Pipeline, Step, LoopStep, HumanInTheLoopStep

# Create a loop with HITL
agent_step = Step(name="agent", agent=your_agent)
hitl_step = HumanInTheLoopStep(name="hitl", message_for_user="Continue?")

loop = LoopStep(
    name="my_loop",
    loop_body_pipeline=Pipeline(steps=[agent_step, hitl_step]),
    exit_condition_callable=lambda output, ctx: output == "stop",
    max_loops=5
)

pipeline = Pipeline.from_step(loop)
runner = Flujo(pipeline)

# Run until pause
result = None
async for res in runner.run_async("start"):
    result = res
    break  # Breaks on first pause

# Resume with human input
final_result = await runner.resume_async(result, "continue")

# Verify no nested loops in trace
print(final_result.trace_tree)  # Should show flat structure
```

## 🚀 Pre-Merge Checklist

- ✅ All tests passing (`make test`)
- ✅ Type checking passes (`make typecheck`)
- ✅ Linting passes (`make lint`)
- ✅ Code formatted (`make format`)
- ✅ `make all` passes ⭐ **Required before merge**
- ✅ Regression tests added
- ✅ Documentation updated
- ✅ No breaking changes

---

**Ready for review!** This fix completely resolves the HITL in loops issue that users reported after PR #500.
