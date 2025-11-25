# Feature Request: Background Task Resumability

**Status:** ✅ APPROVED with Minor Revisions  
**Priority:** Medium-High  
**Category:** Reliability / State Management  
**Related:** #FSD-008 (StepOutcome), StateBackend, Resume API, FLUJO_TEAM_GUIDE.md  
**Reviewer:** Flujo Architecture Team  
**Review Date:** 2025-01-XX  
**Last Updated:** 2025-01-XX

---

## Executive Summary

Enable **automatic resumability for background tasks** by persisting their state separately and providing a mechanism to resume failed background tasks independently of the main pipeline.

This proposal has been reviewed against FLUJO_TEAM_GUIDE.md and updated to ensure full architectural compliance.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architectural Alignment](#architectural-alignment)
3. [Proposed Solution](#proposed-solution)
4. [Implementation Details](#implementation-details)
5. [API Design](#api-design)
6. [Configuration Management](#configuration-management)
7. [Quota Management](#quota-management)
8. [Error Handling & Control Flow](#error-handling--control-flow)
9. [Testing Requirements](#testing-requirements)
10. [Migration & Schema Changes](#migration--schema-changes)
11. [Use Cases](#use-cases)
12. [Backward Compatibility](#backward-compatibility)
13. [Success Criteria](#success-criteria)
14. [References](#references)

---

## Problem Statement

### Current Behavior

Flujo has excellent resumability for regular pipeline steps:
- ✅ State persisted to SQLite after each step
- ✅ Can resume from any step index
- ✅ Crash recovery supported
- ✅ HITL pause/resume works

However, **background tasks** (`execution_mode="background"`) have limitations:

1. **Failures Not Tracked**: Background task failures are logged but don't update workflow state
2. **No Resume Capability**: Can't resume a failed background task independently
3. **State Timing Issue**: Main pipeline completes → state marked "completed" → background failure happens later → too late to update state
4. **Fire-and-Forget Design**: Failures are intentionally swallowed (by design) but this prevents recovery

### Current Implementation Analysis

```python
# flujo/application/core/executor_core.py:807-879
async def _execute_background_task(
    self,
    step: Step[Any, Any],
    data: Any,
    context: Optional[TContext_w_Scratch],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
) -> None:
    """Execute a step in the background, logging any errors."""
    try:
        isolated_context = self._isolate_context(context)  # ⚠️ Should use ContextManager.isolate()
        # ... execution ...
    except (PausedException, PipelineAbortSignal) as control_flow_err:
        # ⚠️ Swallows control flow exceptions - needs clarification
        telemetry.logfire.warning(...)
    except Exception as e:
        # ⚠️ Fire-and-forget - no state persistence
        telemetry.logfire.error(...)
```

### Real-World Impact

**Example: MED13 Research Engine**
```
1. User submits document → Shadow Scribe (background) launched
2. Hero Researcher responds immediately ✅
3. Shadow Scribe fails (API timeout) → Error logged ⚠️
4. Knowledge Graph never updated ❌
5. User never notified ❌
6. No way to retry ❌
```

**Production Impact:**
- Data loss in background processing pipelines
- No visibility into background task failures
- Manual workarounds required (separate run_id, manual state tracking)
- Inconsistent with Flujo's otherwise excellent resumability story

---

## Architectural Alignment

This section documents how the proposal aligns with FLUJO_TEAM_GUIDE.md.

### ✅ Policy-Driven Architecture (Section 1)

Background task execution still routes through the policy system:

```python
# Current (correct) pattern in executor_core.py:866-867
await self.execute(frame)  # Routes through policy dispatch
```

**Compliance:** The proposal does NOT add step-specific logic to ExecutorCore. All execution goes through policies. State tracking is infrastructure-level code (acceptable in ExecutorCore).

### ✅ Context Isolation (Section 3.5)

**Current Issue:** Uses `self._isolate_context()` instead of `ContextManager.isolate()`.

**Required Fix:**

```python
# ❌ WRONG - Current implementation
isolated_context = self._isolate_context(context)

# ✅ CORRECT - Required pattern (FLUJO_TEAM_GUIDE.md Section 3.5)
from flujo.application.core.context_manager import ContextManager

isolated_context = ContextManager.isolate(context)
```

### ✅ Proactive Quota System (Section 7)

Background tasks must participate in the quota system:

```python
# Background tasks should use split quotas
if parent_quota is not None:
    bg_quota = parent_quota.split(1)[0]  # Reserve portion for background task
else:
    bg_quota = None
```

### ✅ Centralized Configuration (Section 4)

Configuration must go through `ConfigManager`:

```toml
# flujo.toml
[background_tasks]
enable_state_tracking = true
enable_resumability = true
default_retry_limit = 3
```

### ✅ Control Flow Exception Pattern (Section 2)

Clarification needed for background tasks:

| Exception Type | Regular Steps | Background Tasks (with resumability) |
|----------------|---------------|-------------------------------------|
| `PausedException` | Re-raise | Log + persist paused state |
| `PipelineAbortSignal` | Re-raise | Log + persist failed state |
| Other exceptions | Return `StepResult(success=False)` | Log + persist failed state |

**Rationale:** Background tasks cannot re-raise to the main pipeline (it has moved on), but with resumability enabled, we persist state to allow later resume.

---

## Proposed Solution

### Approach: Automatic Separate State Tracking

**Automatically create separate state entries for background tasks:**

```python
from flujo.application.core.context_manager import ContextManager
from flujo.application.core.state_manager import StateManager

async def _execute_background_task(
    self,
    step: Step[Any, Any],
    data: Any,
    context: Optional[TContext_w_Scratch],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
) -> None:
    """Execute a step in the background with optional state tracking."""
    
    # Check if resumability is enabled
    settings = get_settings()
    enable_resumability = settings.background_tasks.enable_resumability
    
    # Generate unique identifiers
    task_id = f"bg_{uuid.uuid4().hex}"
    parent_run_id = getattr(context, "run_id", None) if context else None
    bg_run_id = f"{parent_run_id}_bg_{task_id}" if parent_run_id else task_id
    
    # ✅ Use ContextManager.isolate() for proper context isolation
    bg_context = ContextManager.isolate(context)
    if bg_context is not None:
        bg_context.run_id = bg_run_id
        bg_context.parent_run_id = parent_run_id
    
    # Register task if resumability enabled
    if enable_resumability and self.state_manager is not None:
        await self._register_background_task(
            task_id=task_id,
            bg_run_id=bg_run_id,
            parent_run_id=parent_run_id,
            step_name=self._safe_step_name(step),
            data=data,
            context=bg_context,
        )
    
    try:
        # Execute through policy system
        step_copy = step.model_copy(deep=True)
        if hasattr(step_copy, "config"):
            step_copy.config.execution_mode = "sync"
        
            # Get quota (prefer splitting from parent if available)
            parent_quota = self.CURRENT_QUOTA.get() if hasattr(self, 'CURRENT_QUOTA') else None
            bg_quota = self._get_background_quota(parent_quota=parent_quota)
            
            frame = ExecutionFrame(
                step=step_copy,
                data=data,
                context=bg_context,
                resources=resources,
                limits=limits,
                quota=bg_quota,  # Quota management (split from parent or config)
                stream=False,
                on_chunk=None,
                context_setter=lambda _res, _ctx: None,
                result=None,
                _fallback_depth=0,
            )
        
        await self.execute(frame)
        
        # Mark completed if resumability enabled
        if enable_resumability and self.state_manager is not None:
            await self._mark_background_task_completed(task_id, bg_context)
            
    except (PausedException, PipelineAbortSignal) as control_flow_err:
        # Control-flow exceptions: persist state if resumability enabled
        if enable_resumability and self.state_manager is not None:
            await self._mark_background_task_paused(
                task_id=task_id,
                context=bg_context,
                error=control_flow_err,
            )
        telemetry.logfire.warning(
            f"Background task '{self._safe_step_name(step)}' raised control-flow signal: {control_flow_err}"
        )
        
    except Exception as e:
        # Other exceptions: persist failed state if resumability enabled
        if enable_resumability and self.state_manager is not None:
            await self._mark_background_task_failed(
                task_id=task_id,
                context=bg_context,
                error=e,
            )
        # Classify error per FLUJO_TEAM_GUIDE.md Section 2
        error_context = ErrorContext.from_exception(e)
        classifier = ErrorClassifier()
        classifier.classify_error(error_context)
        
        telemetry.logfire.error(
            f"Background task failed for step '{self._safe_step_name(step)}': {e}",
            extra={"error_category": error_context.category.value}
        )
```

**Benefits:**
- ✅ Automatic (no code changes needed for existing pipelines)
- ✅ Consistent with Flujo's resumability model
- ✅ Uses `ContextManager.isolate()` for proper context isolation
- ✅ Participates in quota system
- ✅ Errors properly classified
- ✅ Main pipeline state unaffected
- ✅ Can query and resume failed background tasks

---

## Implementation Details

### 1. ExecutorCore Changes

**File:** `flujo/application/core/executor_core.py`

```python
from flujo.application.core.context_manager import ContextManager
from flujo.application.core.error_categories import ErrorCategory
from flujo.application.core.error_context import ErrorContext
from flujo.application.core.error_classifier import ErrorClassifier
from flujo.infra.config_manager import get_settings

class ExecutorCore(Generic[TContext_w_Scratch]):
    """Executor core with background task resumability support."""
    
    def __init__(
        self,
        # ... existing parameters ...
        state_manager: Optional[StateManager] = None,
    ) -> None:
        # ... existing initialization ...
        self.state_manager = state_manager
    
    async def _register_background_task(
        self,
        task_id: str,
        bg_run_id: str,
        parent_run_id: Optional[str],
        step_name: str,
        data: Any,
        context: Optional[TContext_w_Scratch],
    ) -> None:
        """Register a background task for state tracking.
        
        Uses StateManager for persistence to maintain consistency
        with the existing state management patterns.
        """
        if self.state_manager is None:
            return
            
        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="running",
            metadata={
                "is_background_task": True,
                "task_id": task_id,
                "parent_run_id": parent_run_id,
                "step_name": step_name,
                "input_data": data,
            },
        )
    
    async def _mark_background_task_completed(
        self,
        task_id: str,
        context: Optional[TContext_w_Scratch],
    ) -> None:
        """Mark background task as completed."""
        if self.state_manager is None:
            return
            
        bg_run_id = getattr(context, "run_id", None) if context else None
        if bg_run_id is None:
            return
            
        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=1,  # Single step completed
            last_step_output=None,
            status="completed",
        )
        
        telemetry.logfire.info(f"Background task '{task_id}' completed successfully")
    
    async def _mark_background_task_failed(
        self,
        task_id: str,
        context: Optional[TContext_w_Scratch],
        error: Exception,
    ) -> None:
        """Mark background task as failed with error details."""
        if self.state_manager is None:
            return
            
        bg_run_id = getattr(context, "run_id", None) if context else None
        if bg_run_id is None:
            return
        
        # Store error details in context for later retrieval
        if context is not None and hasattr(context, "scratchpad"):
            context.scratchpad["background_error"] = str(error)
            context.scratchpad["background_error_type"] = type(error).__name__
            
        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="failed",
        )
        
        telemetry.logfire.error(
            f"Background task '{task_id}' failed",
            extra={"error": str(error), "error_type": type(error).__name__}
        )
    
    async def _mark_background_task_paused(
        self,
        task_id: str,
        context: Optional[TContext_w_Scratch],
        error: Exception,
    ) -> None:
        """Mark background task as paused (for control flow exceptions)."""
        if self.state_manager is None:
            return
            
        bg_run_id = getattr(context, "run_id", None) if context else None
        if bg_run_id is None:
            return
            
        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="paused",
        )
        
        telemetry.logfire.info(
            f"Background task '{task_id}' paused",
            extra={"reason": str(error)}
        )
    
    def _get_background_quota(self, parent_quota: Optional[Quota] = None) -> Optional[Quota]:
        """Get quota for background task execution.
        
        Per FLUJO_TEAM_GUIDE.md Section 7: use proactive quota system.
        
        Priority 1: Split from parent quota (atomic reservation)
        - Ensures total spend doesn't exceed user's intent
        - Example: User sets $5.00 limit, spawns 10 background tasks
          → Each gets $0.50 slice, total stays within $5.00
        
        Priority 2: Config fallback (isolated budget)
        - Used when no parent quota exists (standalone background tasks)
        - Independent budget per task from config
        
        Args:
            parent_quota: Optional parent quota to split from
            
        Returns:
            Quota for background task, or None if quota disabled or parent exhausted
        """
        settings = get_settings()
        bg_settings = settings.background_tasks
        
        if not bg_settings.enable_quota:
            return None
        
        # Priority 1: Split from parent quota (atomic reservation)
        if parent_quota is not None:
            try:
                # Take a slice of the parent budget
                # split(1) returns list with one quota slice
                bg_quota = parent_quota.split(1)[0]
                telemetry.logfire.debug(
                    f"Background task quota split from parent: "
                    f"${bg_quota.max_cost:.2f} / {bg_quota.max_tokens} tokens"
                )
                return bg_quota
            except ValueError:
                # Parent budget exhausted - cannot spawn background task
                telemetry.logfire.warning(
                    "Cannot spawn background task: parent quota exhausted"
                )
                return None
        
        # Priority 2: Config fallback (isolated budget)
        # Used when no parent quota exists (standalone background tasks)
        bg_quota = Quota(
            max_cost=bg_settings.max_cost_per_task,
            max_tokens=bg_settings.max_tokens_per_task,
        )
        telemetry.logfire.debug(
            f"Background task quota from config: "
            f"${bg_quota.max_cost:.2f} / {bg_quota.max_tokens} tokens"
        )
        return bg_quota
```

### 2. StateBackend Extension

**File:** `flujo/state/backends/base.py`

```python
class StateBackend(ABC):
    """Abstract base class for state backends.
    
    Extended to support background task state management.
    """

    # ... existing methods ...
    
    async def list_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List background tasks with optional filtering.
        
        Args:
            parent_run_id: Filter by parent pipeline run ID
            status: Filter by status ("running", "completed", "failed", "paused")
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of background task state dictionaries
        """
        # Default implementation using list_workflows with filtering
        all_workflows = await self.list_workflows(status=status, limit=limit, offset=offset)
        
        background_tasks = [
            wf for wf in all_workflows
            if wf.get("metadata", {}).get("is_background_task", False)
        ]
        
        if parent_run_id is not None:
            background_tasks = [
                wf for wf in background_tasks
                if wf.get("metadata", {}).get("parent_run_id") == parent_run_id
            ]
        
        return background_tasks
    
    async def get_failed_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        hours_back: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get failed background tasks within time window.
        
        Args:
            parent_run_id: Filter by parent pipeline run ID
            hours_back: How far back to look (default 24 hours)
            
        Returns:
            List of failed background task state dictionaries
        """
        failed_tasks = await self.list_background_tasks(
            parent_run_id=parent_run_id,
            status="failed",
        )
        
        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=hours_back)
        return [
            task for task in failed_tasks
            if datetime.fromisoformat(task.get("updated_at", "1970-01-01")) > cutoff
        ]
```

### 3. SQLite Backend Extension

**File:** `flujo/state/backends/sqlite.py`

```python
# Schema extension for background task metadata
# Per FLUJO_TEAM_GUIDE.md: New columns must be in ALLOWED_COLUMNS

ALLOWED_COLUMNS = frozenset({
    "total_steps INTEGER DEFAULT 0",
    "error_message TEXT",
    "execution_time_ms INTEGER",
    "memory_usage_mb REAL",
    "step_history TEXT",
    # New columns for background task support
    "is_background_task INTEGER DEFAULT 0",
    "parent_run_id TEXT",
    "task_id TEXT",
    "background_error TEXT",
})

class SQLiteBackend(StateBackend):
    """SQLite backend with background task support."""
    
    async def list_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List background tasks with efficient SQL filtering."""
        query = """
            SELECT * FROM workflow_state 
            WHERE is_background_task = 1
        """
        params: List[Any] = []
        
        if parent_run_id is not None:
            query += " AND parent_run_id = ?"
            params.append(parent_run_id)
        
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY updated_at DESC"
        
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        async with self._get_connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
```

### 4. Flujo Runner Extension

**File:** `flujo/application/runner.py`

```python
class Flujo(Generic[RunnerInT, RunnerOutT, ContextT]):
    """Flujo runner with background task management."""
    
    async def get_failed_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        hours_back: int = 24,
    ) -> List[BackgroundTaskInfo]:
        """Get failed background tasks.
        
        Args:
            parent_run_id: Filter by parent pipeline run ID
            hours_back: How far back to look (default 24 hours)
            
        Returns:
            List of BackgroundTaskInfo objects
        """
        if self.state_backend is None:
            return []
        
        failed = await self.state_backend.get_failed_background_tasks(
            parent_run_id=parent_run_id,
            hours_back=hours_back,
        )
        
        return [
            BackgroundTaskInfo(
                task_id=task.get("metadata", {}).get("task_id"),
                run_id=task.get("run_id"),
                parent_run_id=task.get("metadata", {}).get("parent_run_id"),
                step_name=task.get("metadata", {}).get("step_name"),
                status=task.get("status"),
                error=task.get("background_error"),
                created_at=task.get("created_at"),
                updated_at=task.get("updated_at"),
            )
            for task in failed
        ]
    
    async def resume_background_task(
        self,
        task_id: str,
        new_data: Optional[Any] = None,
    ) -> PipelineResult[ContextT]:
        """Resume a failed background task.
        
        Args:
            task_id: The background task ID to resume
            new_data: Optional new input data (uses original if not provided)
            
        Returns:
            PipelineResult from the resumed execution
        """
        if self.state_backend is None:
            raise ValueError("State backend required for background task resumption")
        
        # Find the task
        tasks = await self.state_backend.list_background_tasks(status="failed")
        task = next(
            (t for t in tasks if t.get("metadata", {}).get("task_id") == task_id),
            None
        )
        
        if task is None:
            raise ValueError(f"Background task '{task_id}' not found")
        
        # Load the task state
        bg_run_id = task.get("run_id")
        loaded_state = await self.state_backend.load_state(bg_run_id)
        
        if loaded_state is None:
            raise ValueError(f"State not found for background task '{task_id}'")
        
        wf_state = WorkflowState.model_validate(loaded_state)
        
        # Reconstruct context
        context = self._serializer.deserialize_context(
            wf_state.pipeline_context,
            self.context_model,
        )
        
        # Get input data
        data = new_data if new_data is not None else task.get("metadata", {}).get("input_data")
        
        # Execute the background task synchronously
        # (We're explicitly resuming, so it runs in foreground)
        step_name = task.get("metadata", {}).get("step_name")
        step = self._find_step_by_name(step_name)
        
        if step is None:
            raise ValueError(f"Step '{step_name}' not found in pipeline")
    
    def _find_step_by_name(self, name: str) -> Optional[Step]:
        """Find a step by name in the pipeline.
        
        Supports nested structures (LoopStep, ParallelStep) by recursively
        traversing the pipeline structure.
        
        Args:
            name: The step name to find
            
        Returns:
            The Step object if found, None otherwise
        """
        from flujo.domain.dsl.loop import LoopStep
        from flujo.domain.dsl.parallel import ParallelStep
        
        # Use queue-based BFS to traverse nested structures
        queue: List[Step] = list(self.pipeline.steps)
        visited: Set[Step] = set()
        
        while queue:
            curr = queue.pop(0)
            
            # Skip if already visited (prevent cycles)
            if curr in visited:
                continue
            visited.add(curr)
            
            # Check if this step matches
            step_name = getattr(curr, "name", None)
            if step_name == name:
                return curr
            
            # Add children to queue for nested structures
            if isinstance(curr, LoopStep):
                # LoopStep has a body step
                if hasattr(curr, "body") and curr.body is not None:
                    queue.append(curr.body)
            elif isinstance(curr, ParallelStep):
                # ParallelStep has branches
                if hasattr(curr, "branches") and curr.branches:
                    queue.extend(curr.branches)
            elif hasattr(curr, "steps"):
                # Generic step container (e.g., SequentialStep)
                if curr.steps:
                    queue.extend(curr.steps)
        
        return None
        
        # Mark as running
        await self.state_backend.save_state(bg_run_id, {
            **loaded_state,
            "status": "running",
            "updated_at": datetime.now().isoformat(),
        })
        
        try:
            # Execute
            result = await self.run_async(data, context=context, run_id=bg_run_id)
            return result
        except Exception as e:
            # Mark as failed again
            await self.state_backend.save_state(bg_run_id, {
                **loaded_state,
                "status": "failed",
                "background_error": str(e),
                "updated_at": datetime.now().isoformat(),
            })
            raise
    
    async def retry_failed_background_tasks(
        self,
        parent_run_id: str,
        max_retries: int = 3,
    ) -> List[PipelineResult[ContextT]]:
        """Retry all failed background tasks for a parent run.
        
        Args:
            parent_run_id: The parent pipeline run ID
            max_retries: Maximum retry attempts per task
            
        Returns:
            List of PipelineResults from retry attempts
        """
        results: List[PipelineResult[ContextT]] = []
        
        failed_tasks = await self.get_failed_background_tasks(parent_run_id=parent_run_id)
        
        for task in failed_tasks:
            for attempt in range(max_retries):
                try:
                    result = await self.resume_background_task(task.task_id)
                    results.append(result)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        telemetry.logfire.error(
                            f"Background task '{task.task_id}' failed after {max_retries} retries: {e}"
                        )
                    else:
                        telemetry.logfire.warning(
                            f"Background task '{task.task_id}' retry {attempt + 1} failed: {e}"
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return results
```

---

## API Design

### Data Models

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

@dataclass
class BackgroundTaskInfo:
    """Information about a background task."""
    task_id: str
    run_id: str
    parent_run_id: Optional[str]
    step_name: str
    status: str  # "running" | "completed" | "failed" | "paused"
    error: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class BackgroundTaskResult:
    """Result of a background task execution."""
    task_id: str
    success: bool
    output: Optional[Any]
    error: Optional[str]
    execution_time_ms: int
```

### Public API Summary

| Method | Description | Returns |
|--------|-------------|---------|
| `runner.get_failed_background_tasks()` | Get failed background tasks | `List[BackgroundTaskInfo]` |
| `runner.resume_background_task(task_id)` | Resume a specific task | `PipelineResult` |
| `runner.retry_failed_background_tasks(parent_run_id)` | Retry all failed tasks | `List[PipelineResult]` |
| `state_backend.list_background_tasks()` | List background tasks | `List[Dict]` |
| `state_backend.get_failed_background_tasks()` | Get failed tasks | `List[Dict]` |

---

## Configuration Management

Per FLUJO_TEAM_GUIDE.md Section 4, all configuration goes through `ConfigManager`.

### flujo.toml Configuration

```toml
[background_tasks]
# Enable state persistence for background tasks
enable_state_tracking = true

# Enable resume capability for failed background tasks
enable_resumability = true

# Enable quota management for background tasks
enable_quota = true

# Maximum cost per background task (USD)
max_cost_per_task = 1.0

# Maximum tokens per background task
max_tokens_per_task = 10000

# Default retry limit for automatic retries
default_retry_limit = 3

# Retention period for background task state (days)
state_retention_days = 30
```

### Settings Model

**File:** `flujo/infra/settings.py`

```python
class BackgroundTaskSettings(BaseModel):
    """Settings for background task management."""
    enable_state_tracking: bool = True
    enable_resumability: bool = True
    enable_quota: bool = True
    max_cost_per_task: float = 1.0
    max_tokens_per_task: int = 10000
    default_retry_limit: int = 3
    state_retention_days: int = 30


class Settings(BaseSettings):
    # ... existing settings ...
    
    background_tasks: BackgroundTaskSettings = BackgroundTaskSettings()
```

### Environment Variable Overrides

```bash
# Override via environment variables
FLUJO_BACKGROUND_TASKS__ENABLE_STATE_TRACKING=true
FLUJO_BACKGROUND_TASKS__ENABLE_RESUMABILITY=true
FLUJO_BACKGROUND_TASKS__MAX_COST_PER_TASK=2.0
```

---

## Quota Management

Per FLUJO_TEAM_GUIDE.md Section 7: Proactive Quota System.

### Background Task Quota Strategy

**CRITICAL FIX:** Background tasks must split quota from parent first to prevent budget violations.

**Problem:** If a user sets a global limit of $5.00 and spawns 10 background tasks, each reading a new $1.00 limit from config would allow $15.00 total spend, violating the user's intent.

**Solution:** Two-tier quota system:

1. **Priority 1: Split from parent quota** (atomic reservation)
   - Ensures total spend doesn't exceed user's intent
   - Example: User sets $5.00 limit, spawns 10 background tasks → Each gets $0.50 slice, total stays within $5.00

2. **Priority 2: Config fallback** (isolated budget)
   - Used when no parent quota exists (standalone background tasks)
   - Independent budget per task from config

```python
def _get_background_quota(self, parent_quota: Optional[Quota] = None) -> Optional[Quota]:
    """Get quota for background task execution.
    
    Priority 1: Split from parent quota (atomic reservation)
    Priority 2: Config fallback (isolated budget)
    
    Background tasks split quotas because:
    1. They run asynchronously after the main pipeline returns
    2. Must respect parent pipeline's total budget constraint
    3. Background failures shouldn't affect main pipeline quota accounting
    """
    # See Implementation Details section for full code
```

### Quota Enforcement

```python
# In _execute_background_task
try:
    # Reserve quota before execution (proactive)
    # Get parent quota if available (for splitting)
    parent_quota = self.CURRENT_QUOTA.get() if hasattr(self, 'CURRENT_QUOTA') else None
    quota = self._get_background_quota(parent_quota=parent_quota)
    if quota is not None:
        estimate = UsageEstimate(tokens=1000, cost=0.01)  # Conservative estimate
        if not quota.can_reserve(estimate):
            raise UsageLimitError(
                f"Background task quota exceeded: {format_cost(quota.max_cost)}"
            )
        quota.reserve(estimate)
    
    # Execute
    await self.execute(frame)
    
    # Reconcile after execution
    if quota is not None:
        actual_usage = frame.get_actual_usage()
        quota.reconcile(estimate, actual_usage)
        
except UsageLimitError as e:
    await self._mark_background_task_failed(task_id, bg_context, e)
    telemetry.logfire.warning(f"Background task '{task_id}' exceeded quota: {e}")
```

---

## Error Handling & Control Flow

Per FLUJO_TEAM_GUIDE.md Section 2: Exception Handling.

### Control Flow Exception Strategy

| Exception | Without Resumability | With Resumability |
|-----------|---------------------|-------------------|
| `PausedException` | Log warning, swallow | Persist "paused" state, log |
| `PipelineAbortSignal` | Log warning, swallow | Persist "failed" state, log |
| `UsageLimitError` | Log error, swallow | Persist "failed" state, log |
| Other exceptions | Log error, swallow | Persist "failed" state, log |

### Error Classification Integration

```python
from flujo.application.core.error_categories import ErrorCategory
from flujo.application.core.error_context import ErrorContext
from flujo.application.core.error_classifier import ErrorClassifier

async def _handle_background_task_error(
    self,
    task_id: str,
    context: Optional[TContext_w_Scratch],
    error: Exception,
) -> None:
    """Handle background task error with proper classification."""
    
    # Classify the error
    error_context = ErrorContext.from_exception(error)
    classifier = ErrorClassifier()
    classifier.classify_error(error_context)
    
    # Store classification in context
    if context is not None and hasattr(context, "scratchpad"):
        context.scratchpad["background_error"] = str(error)
        context.scratchpad["background_error_type"] = type(error).__name__
        context.scratchpad["background_error_category"] = error_context.category.value
    
    # Determine status based on error category
    if error_context.category == ErrorCategory.CONTROL_FLOW:
        status = "paused"
    else:
        status = "failed"
    
    # Persist state
    await self.state_manager.persist_workflow_state(
        run_id=getattr(context, "run_id", None) if context else None,
        context=context,
        current_step_index=0,
        last_step_output=None,
        status=status,
    )
    
    # Emit telemetry
    telemetry.logfire.error(
        f"Background task '{task_id}' failed",
        extra={
            "error": str(error),
            "error_type": type(error).__name__,
            "error_category": error_context.category.value,
            "status": status,
        }
    )
```

### HITL in Background Tasks

**Important Clarification:** HITL steps in background tasks will pause but cannot interact with users (the main pipeline has returned). The paused state is preserved for later inspection/debugging.

```python
# If a background task contains HITL and pauses:
# 1. State is persisted as "paused"
# 2. Cannot be resumed via normal HITL flow (no user waiting)
# 3. Must be resumed programmatically via resume_background_task()
# 4. New data can be provided at resume time

# Example: Resume paused HITL background task with answer
await runner.resume_background_task(
    task_id="bg_abc123",
    new_data="User's answer to HITL question",
)
```

---

## Telemetry & Observability

### Hook Telemetry Noise Reduction

**Problem:** When `run_async` is called for a background task, it triggers `pre_run` and `post_run` hooks. This causes metrics dashboards to show double the number of "Pipeline Runs" (Parent + Background), creating noise in telemetry.

**Solution:** Add `is_background: bool` flag to hook payloads and context to allow telemetry processors to distinguish between user-facing runs and background workers.

**Implementation:**

```python
# flujo/application/core/hooks.py or similar

class HookPayload:
    """Payload for pipeline hooks."""
    # ... existing fields ...
    is_background: bool = False  # NEW: Flag for background task runs

# In ExecutorCore._execute_background_task
async def _execute_background_task(...):
    # ... setup ...
    
    # Mark context as background task
    if bg_context is not None:
        bg_context.is_background = True
        bg_context.scratchpad["is_background_task"] = True
    
    # When calling run_async for background task resume
    # Pass is_background flag in hook payload
    result = await self.run_async(
        data,
        context=bg_context,
        run_id=bg_run_id,
        is_background=True,  # NEW: Suppress telemetry noise
    )
```

**Telemetry Integration:**

```python
# Example: Prometheus exporter can filter background tasks
def record_pipeline_run(payload: HookPayload) -> None:
    """Record pipeline run metric."""
    if payload.is_background:
        # Record to separate metric: flujo_background_runs_total
        background_runs_counter.inc()
    else:
        # Record to main metric: flujo_pipeline_runs_total
        pipeline_runs_counter.inc()
```

---

## Zombie Task Cleanup

**Problem:** Background tasks are less visible. If the process dies (`kill -9`), the background task remains in `running` state in SQLite forever, creating "zombie" tasks that never complete.

**Solution:** Add a `cleanup_stale_background_tasks` method to `SQLiteBackend` that checks for `running` tasks older than X hours and marks them as `failed` (timeout).

**Implementation:**

```python
# flujo/state/backends/sqlite.py

class SQLiteBackend(StateBackend):
    """SQLite backend with zombie task cleanup."""
    
    async def cleanup_stale_background_tasks(
        self,
        stale_hours: int = 24,
    ) -> int:
        """Mark stale background tasks as failed (timeout).
        
        Background tasks that have been in 'running' state for longer
        than stale_hours are likely zombies from crashed processes.
        
        Args:
            stale_hours: Hours after which a running task is considered stale
            
        Returns:
            Number of tasks marked as failed
        """
        cutoff = datetime.now() - timedelta(hours=stale_hours)
        
        async with self._get_connection() as conn:
            # Find stale running background tasks
            cursor = await conn.execute(
                """
                SELECT run_id, updated_at 
                FROM workflow_state 
                WHERE is_background_task = 1 
                  AND status = 'running'
                  AND updated_at < ?
                """,
                (cutoff.isoformat(),)
            )
            stale_tasks = await cursor.fetchall()
            
            if not stale_tasks:
                return 0
            
            # Mark as failed with timeout error
            count = 0
            for run_id, updated_at in stale_tasks:
                await conn.execute(
                    """
                    UPDATE workflow_state 
                    SET status = 'failed',
                        background_error = 'Task timeout: process likely crashed',
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (datetime.now().isoformat(), run_id)
                )
                count += 1
            
            await conn.commit()
            
            if count > 0:
                telemetry.logfire.warning(
                    f"Cleaned up {count} stale background tasks (older than {stale_hours} hours)"
                )
            
            return count
    
    async def get_stale_background_tasks(
        self,
        stale_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get stale background tasks without modifying them.
        
        Useful for monitoring and reporting before cleanup.
        
        Args:
            stale_hours: Hours after which a running task is considered stale
            
        Returns:
            List of stale task dictionaries
        """
        cutoff = datetime.now() - timedelta(hours=stale_hours)
        
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM workflow_state 
                WHERE is_background_task = 1 
                  AND status = 'running'
                  AND updated_at < ?
                ORDER BY updated_at ASC
                """,
                (cutoff.isoformat(),)
            )
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
```

**Usage:**

```python
# Periodic cleanup (e.g., in a cron job or background worker)
async def periodic_cleanup():
    """Run periodic cleanup of stale background tasks."""
    async with SQLiteBackend("flujo.db") as backend:
        cleaned = await backend.cleanup_stale_background_tasks(stale_hours=24)
        print(f"Cleaned up {cleaned} stale background tasks")

# Or integrate into Flujo runner
class Flujo:
    async def cleanup_stale_background_tasks(self, stale_hours: int = 24) -> int:
        """Clean up stale background tasks."""
        if self.state_backend is None:
            return 0
        
        if hasattr(self.state_backend, 'cleanup_stale_background_tasks'):
            return await self.state_backend.cleanup_stale_background_tasks(stale_hours)
        return 0
```

**Configuration:**

```toml
# flujo.toml
[background_tasks]
# ... other settings ...
stale_task_timeout_hours = 24  # Hours before marking as failed
```

---

## Testing Requirements

Per FLUJO_TEAM_GUIDE.md Section "Testing Standards":

### Test Markers

```python
# Background task resumability tests should use:
pytestmark = [
    pytest.mark.slow,      # Uses state persistence
    pytest.mark.serial,    # SQLite backend access
]
```

### Unit Tests

```python
# tests/unit/test_background_task_resumability.py

@pytest.mark.asyncio
async def test_background_task_state_persistence():
    """Test that background task state is persisted correctly."""
    # ...

@pytest.mark.asyncio
async def test_background_task_failure_tracking():
    """Test that failed background tasks are tracked."""
    # ...

@pytest.mark.asyncio
async def test_background_task_resume():
    """Test that failed background tasks can be resumed."""
    # ...

@pytest.mark.asyncio
async def test_background_task_quota_enforcement():
    """Test quota enforcement for background tasks."""
    # ...

@pytest.mark.asyncio
async def test_background_task_context_isolation():
    """Test that ContextManager.isolate() is used correctly."""
    # ...
```

### Integration Tests

```python
# tests/integration/test_background_task_e2e.py

@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.asyncio
async def test_background_task_failure_and_resume_e2e():
    """End-to-end test: background task fails → resume → success."""
    # ...

@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.asyncio
async def test_multiple_background_tasks_per_pipeline():
    """Test multiple background tasks in a single pipeline."""
    # ...

@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.asyncio
async def test_background_task_crash_recovery():
    """Test crash recovery for background tasks."""
    # ...
```

### Performance Tests

```python
# tests/benchmarks/test_background_task_performance.py

@pytest.mark.benchmark
@pytest.mark.slow
def test_background_task_state_persistence_overhead():
    """Test that state persistence overhead is < 5%."""
    # ...
```

---

## Migration & Schema Changes

Per FLUJO_TEAM_GUIDE.md Section "SQLite Schema Safeguards":

### Schema Migration

```python
# flujo/state/backends/sqlite.py

ALLOWED_COLUMNS = frozenset({
    # Existing columns
    "total_steps INTEGER DEFAULT 0",
    "error_message TEXT",
    "execution_time_ms INTEGER",
    "memory_usage_mb REAL",
    "step_history TEXT",
    # New columns for background task support (add to whitelist FIRST)
    "is_background_task INTEGER DEFAULT 0",
    "parent_run_id TEXT",
    "task_id TEXT",
    "background_error TEXT",
})

async def _migrate_schema(self) -> None:
    """Apply schema migrations for background task support."""
    async with self._get_connection() as conn:
        # Check if columns exist
        cursor = await conn.execute("PRAGMA table_info(workflow_state)")
        existing_columns = {row[1] for row in await cursor.fetchall()}
        
        # Add missing columns (migrations are idempotent)
        migrations = [
            ("is_background_task", "INTEGER DEFAULT 0"),
            ("parent_run_id", "TEXT"),
            ("task_id", "TEXT"),
            ("background_error", "TEXT"),
        ]
        
        for col_name, col_type in migrations:
            if col_name not in existing_columns:
                # Validate column is in whitelist
                col_def = f"{col_name} {col_type}"
                if col_def not in ALLOWED_COLUMNS:
                    raise ValueError(f"Column '{col_def}' not in ALLOWED_COLUMNS whitelist")
                
                await conn.execute(
                    f"ALTER TABLE workflow_state ADD COLUMN {col_name} {col_type}"
                )
        
        await conn.commit()
```

### Identifier Validation

```python
# Per FLUJO_TEAM_GUIDE.md: SQL identifiers must be validated
import re

MAX_SQL_IDENTIFIER_LENGTH = 1000
SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

def _validate_sql_identifier(identifier: str) -> None:
    """Validate SQL identifier per FLUJO_TEAM_GUIDE.md."""
    if len(identifier) > MAX_SQL_IDENTIFIER_LENGTH:
        raise ValueError(f"SQL identifier too long: {len(identifier)} > {MAX_SQL_IDENTIFIER_LENGTH}")
    
    if not SQL_IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    
    # Check for null/control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in identifier):
        raise ValueError(f"SQL identifier contains control characters: {identifier}")
```

---

## Use Cases

### Use Case 1: MED13 Research Engine

**Current:**
```python
@step(execution_mode="background")
async def shadow_scribe_step(...):
    # If this fails, data is lost
    ...
```

**With Feature:**
```python
# No code changes needed! Just enable in config:
# flujo.toml: [background_tasks] enable_resumability = true

@step(execution_mode="background")
async def shadow_scribe_step(...):
    # If this fails, state is persisted automatically
    ...

# Later, retry failed extractions
async def retry_failed_extractions():
    async with Flujo(pipeline, state_backend=SQLiteBackend("flujo.db")) as runner:
        failed = await runner.get_failed_background_tasks()
        for task in failed:
            await runner.resume_background_task(task.task_id)
```

### Use Case 2: Batch Processing

**Scenario:** Process 1000 documents in background, some fail

```python
# Submit background tasks
async with Flujo(batch_pipeline, state_backend=backend) as runner:
    for doc in documents:
        await runner.run_async(doc)  # Background extraction
    
    # Main pipeline returns immediately, background tasks run

# Later, retry failures
async with Flujo(batch_pipeline, state_backend=backend) as runner:
    await runner.retry_failed_background_tasks(
        parent_run_id=main_run_id,
        max_retries=3,
    )
```

### Use Case 3: Monitoring Dashboard

```python
# Get background task status
async with Flujo(pipeline, state_backend=backend) as runner:
    # All running background tasks
    running = await backend.list_background_tasks(status="running")
    
    # Failed tasks in last 24 hours
    failed = await runner.get_failed_background_tasks(hours_back=24)
    
    # Display in monitoring dashboard
    for task in failed:
        print(f"Task {task.task_id}: {task.error}")
```

---

## Backward Compatibility

### ✅ Fully Backward Compatible

- Existing background tasks continue to work
- No breaking changes to API
- New features are opt-in (via configuration)
- Default behavior unchanged (failures still logged)

### Migration Path

```python
# Old code (still works - no changes needed)
@step(execution_mode="background")
async def my_step(...):
    ...

# Enable resumability via config (no code changes)
# flujo.toml:
# [background_tasks]
# enable_resumability = true
```

### Configuration Defaults

| Setting | Default | Behavior |
|---------|---------|----------|
| `enable_state_tracking` | `true` | Track background task state |
| `enable_resumability` | `true` | Allow resume of failed tasks |
| `enable_quota` | `true` | Enforce quotas on background tasks |

---

## Success Criteria

### Critical Requirements (Must Fix Before PR)

- [ ] **Quota Logic Fixed**: `_get_background_quota()` prefers splitting from parent quota first, falls back to config only if no parent exists
- [ ] **Step Retrieval Implemented**: `_find_step_by_name()` implemented with recursive traversal for nested structures (LoopStep, ParallelStep)

### Core Functionality

- [ ] Background task failures are persisted to StateBackend
- [ ] Can query failed background tasks via API
- [ ] Can resume failed background tasks
- [ ] Can retry failed background tasks automatically
- [ ] Uses `ContextManager.isolate()` for context isolation
- [ ] Participates in quota system (proactive enforcement with parent splitting)
- [ ] Errors are properly classified
- [ ] Control flow exceptions are handled per specification
- [ ] SQLite schema changes follow migration whitelist
- [ ] Backward compatible (existing code works)
- [ ] Performance impact < 5% for background tasks

### Observability & Maintenance

- [ ] Telemetry hook noise reduction (`is_background` flag implemented)
- [ ] Zombie task cleanup mechanism implemented
- [ ] Documentation updated
- [ ] Tests added (unit + integration + benchmarks)
- [ ] `make all` passes with 0 errors

---

## References

- **Current Implementation**: `flujo/application/core/executor_core.py:807-879`
- **State Persistence**: `flujo/application/core/state_manager.py`
- **Resume API**: `flujo/application/runner.py:827-1088`
- **Context Manager**: `flujo/application/core/context_manager.py`
- **Error Classification**: `flujo/application/core/error_classifier.py`
- **Quota System**: `flujo/application/core/quota.py`
- **FLUJO_TEAM_GUIDE.md**: Architecture and patterns reference
- **Example Use Case**: `examples/example4/med13_research_engine.py`

---

## Appendix A: Architectural Review Summary

### Review Decision: ✅ APPROVED with Minor Revisions

**Reviewer Assessment:** This is a **high-quality feature request** that demonstrates a deep understanding of the Flujo architecture and the specific pain points of the "Research Engine" use case. It correctly identifies that while Flujo handles synchronous state recovery well, "fire-and-forget" background tasks create a visibility and recovery gap.

### Architectural Fit: Strong ✅

- **Decoupling:** Creating a new `run_id` (linked via `parent_run_id`) is the correct database design. Trying to jam background steps into the parent's `step_history` JSON blob would cause massive concurrency locking issues in SQLite.
- **Isolation:** Explicitly using `ContextManager.isolate(context)` prevents race conditions where the background task modifies data the main pipeline is reading.
- **Recursion Prevention:** Setting `execution_mode="sync"` on the *copy* of the step before execution is a clever and necessary detail to prevent infinite background spawning logic.

### Critical Gaps Identified and Fixed

| Issue | Original Proposal | Updated Proposal | Status |
|-------|-------------------|------------------|--------|
| **Quota Logic Discrepancy** | Ignored parent quota, read fresh limits from config | **Priority 1:** Split from parent quota first<br>**Priority 2:** Config fallback only if no parent | ✅ Fixed |
| **Step Retrieval Missing** | `_find_step_by_name()` not implemented | Implemented with recursive BFS traversal for nested structures | ✅ Fixed |
| Context isolation | `self._isolate_context()` | `ContextManager.isolate()` | ✅ Fixed |
| Configuration | Step-level config | `ConfigManager` via `flujo.toml` | ✅ Fixed |
| Error handling | Generic logging | Error classification + telemetry | ✅ Fixed |
| Control flow exceptions | Unclear | Documented strategy per exception type | ✅ Fixed |
| Schema changes | Not specified | Migration whitelist compliance | ✅ Fixed |
| HITL in background | Not addressed | Documented limitation + workaround | ✅ Fixed |

### Suggested Improvements Implemented

| Improvement | Status | Implementation |
|-------------|--------|----------------|
| **Hook Telemetry Noise** | ✅ Added | `is_background` flag in `HookPayload` and context to distinguish user-facing vs background runs |
| **Zombie Task Cleanup** | ✅ Added | `cleanup_stale_background_tasks()` method with configurable timeout |

### Security & Schema Review ✅

- **Schema:** The proposed schema changes (`parent_run_id`, `task_id`, `is_background_task`) are safe and follow the `ALLOWED_COLUMNS` pattern.
- **Sanitization:** Input/output of background tasks respects `FLUJO_TRACE_PREVIEW_LEN` to avoid bloating the SQLite file.

### Next Steps

1. ✅ Update `_get_background_quota` to prefer parent splitting (DONE)
2. ✅ Implement `_find_step_by_name` with traversal support (DONE)
3. ✅ Add telemetry hook noise reduction (DONE)
4. ✅ Add zombie task cleanup mechanism (DONE)
5. ⏭️ Proceed with PR implementation

### Reviewer Sign-off

- [x] Architecture Team Review - **APPROVED**
- [ ] Security Review (state persistence) - Pending
- [ ] Performance Review - Pending
- [ ] Documentation Review - Pending

---

**Submitted by:** Example 4 Implementation Team  
**Reviewed by:** Flujo Architecture Team  
**Review Date:** 2025-01-XX  
**Priority:** Medium-High (affects production reliability)  
**Status:** ✅ APPROVED - Ready for Implementation (Critical fixes applied)

