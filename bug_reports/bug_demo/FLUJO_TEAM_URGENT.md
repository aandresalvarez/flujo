# 🚨 URGENT: Nested Loop Bug Still NOT Fixed

**To**: Flujo Development Team  
**From**: Production User  
**Date**: October 4, 2025  
**Flujo Version**: 0.4.37

---

## 🎯 Quick Summary

**The nested loop bug on HITL resume is STILL HAPPENING in version 0.4.37.**

Your recent PR description mentioned:
> "fixes ... HITL-in-loop resume semantics to prevent nested loop creation"

But our testing shows **nested loops are still being created**.

---

## 📊 Evidence (30 seconds to verify)

### What We See

```
Agent: "Thank you! Press Enter to generate..." (action: "finish")
User: [presses Enter]
→ Creates NESTED loop instead of exiting ❌
→ Loop continues indefinitely
→ Never reaches next step
```

### Debug Trace Shows

```
clarification_loop (iteration 1)
  └── clarification_loop (nested, iteration 1-2) ← NESTED!
      └── clarification_loop (nested, iteration 2-3) ← NESTED!
          └── clarification_loop (nested, iteration 3-4) ← NESTED!
              └── ... continues until user cancels
```

**This is the EXACT bug we reported before.**

---

## ❓ Critical Questions

### 1. Does version 0.4.37 include the nested loop fix?

```bash
$ uv pip show flujo
Name: flujo
Version: 0.4.37
```

If yes → **The fix doesn't work** (evidence above)  
If no → **When will it be released?**

---

### 2. What did your PR actually fix?

Your PR description mentioned:
- ✅ "Fixed sink_to parameter forwarding" (this works now!)
- ✅ "Added comprehensive documentation" (thanks!)
- ❓ "HITL-in-loop resume semantics to prevent nested loop creation"

**But we still see nested loops being created!**

---

### 3. Can you reproduce this?

**Minimal test case**:

```yaml
version: '2.0'
name: test_hitl_loop

agents:
  agent:
    model: openai:gpt-4o-mini
    output_schema:
      type: object
      properties:
        action: { type: string, enum: [ask, finish] }
        question: { type: string }

steps:
  - kind: loop
    name: test_loop
    loop:
      body:
        - kind: step
          name: ask
          uses: agents.agent
          input: "Ask question or finish"
          updates_context: true
        
        - kind: hitl
          name: user_input
          message: "{{ previous_step.question }}"
      
      exit_expression: "context.action == 'finish'"
      max_loops: 5
```

**Expected**: 3 iterations, then exit  
**Actual**: Nested loops, never exits

---

## 🔍 What You Said vs What We See

### You Said (PR Description)

> "It's fixed in code — not just documented. The nested loop issue on HITL resume is addressed by code in step_policies.py"

### We See (Debug Trace)

```
167|└── clarification_loop  (0.000s)
168|    ├── flujo.step.type: LoopStep
169|    ├── step_input: patient count
172|    ├── event loop.iteration: {'iteration': 1}
173|    ├── event loop.iteration: {'iteration': 2}  ← Increments TWICE

204|└── clarification_loop  (0.000s)  ← NESTED LOOP CREATED!
205|    ├── flujo.step.type: LoopStep
206|    ├── step_input: past 12 months
209|    ├── event loop.iteration: {'iteration': 2}
210|    ├── event loop.iteration: {'iteration': 3}  ← Increments TWICE AGAIN
```

**This shows nested loop creation is still happening.**

---

## 🎯 What We Need to Know

### Option A: Fix is in 0.4.37 but doesn't work
→ We have evidence it's still broken  
→ Need deeper investigation  
→ Can share full debug traces

### Option B: Fix is NOT in 0.4.37 yet
→ When will it be released?  
→ What version should we upgrade to?  
→ Can we test from a branch?

### Option C: PR only fixed sink_to, not nested loops
→ Nested loop bug is STILL UNFIXED  
→ Need separate fix for execution flow  
→ This is blocking our production use

---

## 📋 Supporting Evidence

**Full debug trace**: `debug/20251004_215628_run.json`  
**Pipeline**: `pipeline.yaml` (validated successfully)  
**Documentation**: `NESTED_LOOP_STILL_BROKEN.md` (detailed analysis)

**Key observations**:
1. ✅ `sink_to` now works (thank you!)
2. ✅ Validation passes (no warnings)
3. ❌ Nested loops still created on HITL resume
4. ❌ `exit_expression` never evaluated in correct context
5. ❌ Loop never exits, even when agent outputs `action: "finish"`

---

## 🚨 Impact

**CRITICAL**: Cannot use HITL in loops in production

This blocks our entire clarification workflow, which is core to our product.

---

## 🙏 Request

Please clarify:
1. **Is the nested loop fix in version 0.4.37?**
2. **If yes, why is it still broken?** (we have evidence)
3. **If no, when will it be released?**

We're ready to:
- Share full debug traces
- Test any fixes
- Provide more details

Thank you for your rapid response on previous issues! 🙏

---

**Status**: ❌ **PRODUCTION BLOCKED**  
**Flujo Version**: 0.4.37  
**Date**: October 4, 2025  
**Debug Log**: `debug/20251004_215628_run.json`

