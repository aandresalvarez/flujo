# 🐛 Nested Loop Bug Demonstration

**Location**: `/Users/alvaro1/Documents/Coral/Code/cohortgen/projects/clarification/bug_demo/`

> **Note**: The folder now includes a minimal `.flujo/` directory (with `state.db`) so that CLI runs stop complaining about the missing state backend. Keep it in place when zipping or copying the evidence to ensure `uv run flujo run …` works out of the box.

---

## 🚀 **Quick Demo (5 seconds)**

```bash
cd /Users/alvaro1/Documents/Coral/Code/cohortgen/projects/clarification/bug_demo
./DEMO_BUG.sh
```

**Output**: Shows nested loops detected at depth 7

---

## 📁 **Key Files**

### 1. **DEMO_BUG.sh** ⭐
   **The main demonstration script**
   
   Shows:
   - Analysis of actual failing pipeline run
   - Visual tree of nested loops (7 levels deep)
   - Explanation of expected vs actual behavior
   - Root cause analysis

   ```bash
   ./DEMO_BUG.sh
   ```

---

### 2. **analyze_nested_loops.py**
   **Automated analysis tool**
   
   Parses Flujo debug trace JSON and counts nested loop depth.
   
   ```bash
   python3 analyze_nested_loops.py debug/20251004_215628_run.json
   ```
   
   **Output**:
   ```
   └── clarification_loop (depth 1)
     └── clarification_loop (depth 2)  ← NESTED!
       └── clarification_loop (depth 3)  ← NESTED!
         ...
   
   Maximum nesting depth: 7
   ❌ NESTED LOOPS DETECTED!
   ```

---

### 3. **20251004_215628_run.json**
   **The evidence - actual failing pipeline run**
   
   Contains complete trace showing nested loop structures.
   Can be analyzed with `analyze_nested_loops.py`.

---

### 4. **../pipeline.yaml**
   **The pipeline that exhibits the bug** (in parent directory)
   
   Interactive clarification loop with HITL steps.
   To reproduce:
   ```bash
   cd ..
   uv run flujo run --debug pipeline.yaml
   # Answer 5-6 questions
   # Agent will say "finish" but loop won't exit
   # Instead creates nested loops
   ```

---

## 🧪 **How to Reproduce from Scratch**

### Option A: Use Existing Evidence (Fastest)
```bash
./DEMO_BUG.sh
```
Analyzes the already-captured failing run.

---

### Option B: Fresh Reproduction
```bash
# 1. Run pipeline from parent directory
cd /Users/alvaro1/Documents/Coral/Code/cohortgen/projects/clarification
uv run flujo run --debug pipeline.yaml
# Answer questions: "patients with flu", "count", "2020", "none", "none"
# Agent will finish but loop won't exit

# 2. Copy debug trace to bug_demo folder and analyze
cp debug/$(ls -t debug/*.json | head -1 | xargs basename) bug_demo/
cd bug_demo
python3 analyze_nested_loops.py $(ls -t *.json | head -1)
```

---

## 📊 **What the Bug Looks Like**

### Expected Behavior ✅
```
Loop iteration 1 → iteration 2 → iteration 3 → ... → finish (exit)
```
**Depth**: 1 (flat)

---

### Actual Behavior ❌
```
Loop 1
  └── Loop 2 (nested on HITL resume)
      └── Loop 3 (nested on HITL resume)
          └── Loop 4 (nested on HITL resume)
              └── ... (infinite nesting)
```
**Depth**: N (tree structure)

---

## 🎯 **The Core Problem**

**When a HITL step pauses and resumes inside a loop:**

❌ **Current**: Flujo creates a NEW nested loop instance  
✅ **Expected**: Flujo continues the CURRENT loop iteration

**Result**:
- `exit_expression` evaluated in wrong (nested) context
- Loop never exits even when condition is met
- Pipeline stuck until timeout or manual cancel

---

## 📋 **Evidence Package for Flujo Team**

**Share the entire `bug_demo/` folder** containing:

1. ✅ **DEMO_BUG.sh** - Runnable demonstration
2. ✅ **analyze_nested_loops.py** - Analysis tool
3. ✅ **20251004_215628_run.json** - Evidence trace
4. ✅ **FLUJO_TEAM_URGENT.md** - Detailed report
5. ✅ **BUG_CONFIRMED.md** - Test results
6. ✅ **NESTED_LOOP_STILL_BROKEN.md** - Technical analysis
7. ✅ **README_BUG_DEMO.md** - This file

**They can run**:
```bash
cd bug_demo
./DEMO_BUG.sh
```
To see the bug immediately.

**Or zip and send**:
```bash
cd /Users/alvaro1/Documents/Coral/Code/cohortgen/projects/clarification
tar -czf flujo_nested_loop_bug.tar.gz bug_demo/
# Send flujo_nested_loop_bug.tar.gz to Flujo team
```

---

## 🔍 **Technical Details**

### Debug Trace Structure

**File**: `20251004_215628_run.json` (in this folder)

```json
{
  "trace_tree": {
    "name": "pipeline_run",
    "children": [
      {
        "name": "clarification_loop",
        "children": [
          {
            "name": "clarification_loop",  ← NESTED!
            "children": [
              {
                "name": "clarification_loop",  ← NESTED!
                ...
```

Each HITL resume creates a new nested `clarification_loop` node instead of continuing the parent loop.

---

## 🚨 **Impact**

**CRITICAL**: Makes HITL in loops completely unusable

- ❌ Cannot build interactive clarification workflows
- ❌ Cannot do slot-filling with HITL
- ❌ Blocks production use of Flujo for conversational pipelines

---

## ✅ **Validation Status**

| Check | Status |
|-------|--------|
| User experience | ❌ Stuck in loop |
| Terminal output | ❌ Repeated prompts |
| Debug trace inspection | ❌ Nested structure visible |
| **Automated analysis** | ❌ **7 levels of nesting** |
| Bug confirmed | ✅ **DEFINITIVELY** |

---

## 📊 **Summary**

- **Bug**: HITL resume in loops creates nested loops
- **Flujo Version**: 0.4.37
- **Evidence**: Debug trace + automated analysis
- **Nesting Depth**: 7 levels (should be 1)
- **Status**: ❌ **PRODUCTION BLOCKED**
- **Demo**: `./DEMO_BUG.sh` (5 seconds to verify)

---

## 🎯 **Next Steps**

1. ✅ Run `./DEMO_BUG.sh` to see the bug
2. ✅ Share files with Flujo team
3. ⏳ Wait for fix in next Flujo version
4. ⏳ Test again with updated version

---

**Created**: October 4, 2025  
**Flujo Version**: 0.4.37  
**Status**: Bug confirmed and documented
