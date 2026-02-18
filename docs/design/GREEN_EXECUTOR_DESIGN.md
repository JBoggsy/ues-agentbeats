# GreenAgentExecutor Design

**Source**: `src/green/executor.py`
**Status**: ✅ COMPLETE — see `GREEN_EXECUTOR_IMPLEMENTATION_PLAN.md` for details

---

## Table of Contents

1. [Overview](#1-overview)
2. [Responsibilities](#2-responsibilities)
3. [Interface Contract](#3-interface-contract)
4. [Request Parsing & Validation](#4-request-parsing--validation)
5. [GreenAgent Lifecycle Management](#5-greenagent-lifecycle-management)
6. [UES Port Allocation](#6-ues-port-allocation)
7. [TaskUpdateEmitter Wiring](#7-taskupdateemitter-wiring)
8. [Assessment Dispatch](#8-assessment-dispatch)
9. [Cancellation Handling](#9-cancellation-handling)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Shutdown & Cleanup](#11-shutdown--cleanup)
12. [Method Inventory](#12-method-inventory)
13. [Dependencies](#13-dependencies)

---

## 1. Overview

The `GreenAgentExecutor` is the **bridge** between the A2A protocol layer and
the `GreenAgent` assessment orchestrator. It implements the `AgentExecutor`
interface from the `a2a-python` SDK, translating A2A `RequestContext` /
`EventQueue` objects into the higher-level abstractions that `GreenAgent.run()`
expects.

### Position in the Architecture

```
  A2A HTTP request
        │
        ▼
  ┌──────────────────────┐
  │  DefaultRequestHandler│  (from a2a-python SDK)
  │  + InMemoryTaskStore  │
  └──────────┬───────────┘
             │ calls execute() / cancel()
             ▼
  ┌──────────────────────┐
  │  GreenAgentExecutor  │  ← THIS MODULE
  │  (AgentExecutor)     │
  └──────────┬───────────┘
             │ creates / manages
             ▼
  ┌──────────────────────┐
  │  GreenAgent(s)       │  one per context_id
  │  + UES servers       │
  └──────────────────────┘
```

The executor is **not** responsible for HTTP/ASGI details — those are handled by
`A2AServer` in `src/common/a2a/server.py`. The executor is solely concerned
with:

- Parsing incoming A2A messages into assessment parameters
- Managing `GreenAgent` instances (one per `context_id`)
- Wiring up `TaskUpdater`/`TaskUpdateEmitter` for each task
- Dispatching `GreenAgent.run()` and converting results to A2A artifacts
- Handling cancellation requests

---

## 2. Responsibilities

| Responsibility | How |
|----------------|-----|
| Implement `AgentExecutor` interface | `execute()` and `cancel()` methods |
| Parse assessment requests | Extract `participants` and `config` from A2A message |
| Validate requests | Ensure required fields present, scenario exists |
| Manage `GreenAgent` instances | Create/cache per `context_id`, allocate UES ports |
| Wire `TaskUpdateEmitter` | Create `TaskUpdater` from `EventQueue`, wrap in `TaskUpdateEmitter` |
| Create `A2AClientWrapper` | For Purple agent communication, from `participants` URL |
| Load scenarios & evaluators | Via `ScenarioManager` |
| Dispatch assessment | Call `GreenAgent.run()` with all dependencies |
| Emit result artifacts | Convert `AssessmentResults` to A2A artifact events |
| Handle errors | Catch exceptions, emit error status updates |
| Handle cancellation | Forward `cancel()` to the correct `GreenAgent` |
| Manage shutdown | Clean up all `GreenAgent` instances on executor teardown |

---

## 3. Interface Contract

The `AgentExecutor` ABC requires two methods:

```python
class AgentExecutor(ABC):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None: ...
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None: ...
```

### `execute()` Flow

1. Extract `task_id` and `context_id` from `context`
2. Parse the incoming message to get `participants` and `config`
3. Acquire the per-context asyncio lock (prevents concurrent assessments)
4. Look up or create a `GreenAgent` for this `context_id`
5. Create `TaskUpdater` → `TaskUpdateEmitter` from `event_queue`
6. Load scenario and evaluators from `config.scenario_id`
7. Create `A2AClientWrapper` for the Purple agent from `participants`
   (created per-assessment, not cached — Purple URL may differ between assessments)
8. Call `agent.run(task_id, emitter, scenario, evaluators, purple_client, config)`
9. Emit the `AssessmentResults` as an A2A artifact via `updater.add_artifact()`
10. Emit `TaskState.completed` via `updater.update_status()` (executor owns terminal state)
11. Close the Purple client and release the context lock

### `cancel()` Flow

1. Extract `task_id` and `context_id` from `context`
2. Look up the `GreenAgent` for this `context_id`
3. Call `agent.cancel(task_id)` to set the cancellation flag
4. Emit a cancellation status update via `event_queue`

---

## 4. Request Parsing & Validation

### Incoming Message Format

Per the AgentBeats competition spec and `ASSESSMENT_FLOW.md`, the Green agent
receives an A2A message with a JSON `DataPart` containing:

```json
{
    "participants": {
        "personal_assistant": "http://purple-agent:8001"
    },
    "config": {
        "scenario_id": "email_triage_basic",
        "verbose_updates": true,
        "seed": 12345
    }
}
```

### Validation Model

Define a Pydantic model for the incoming request:

```python
class AssessmentRequest(BaseModel):
    """Parsed assessment request from the AgentBeats platform.

    Attributes:
        participants: Mapping of role names to A2A endpoint URLs.
        config: Assessment-specific configuration.
    """

    model_config = ConfigDict(frozen=True)

    participants: dict[str, str]
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        """Get the scenario ID from the config."""
        scenario_id = self.config.get("scenario_id")
        if not scenario_id:
            raise ValueError("config.scenario_id is required")
        return scenario_id

    @property
    def purple_agent_url(self) -> str:
        """Get the Purple agent URL from participants.

        Uses the first participant URL found.
        """
        if not self.participants:
            raise ValueError("participants must contain at least one entry")
        return next(iter(self.participants.values()))
```

### Extracting From `RequestContext`

The incoming data arrives as a `DataPart` in the A2A message. The executor
should use `get_data_content(context.message)` from `src/common/a2a/messages.py`
to extract the JSON dict, then validate it with `AssessmentRequest`.

If the message contains no data part, or the data fails validation, the executor
should emit a `TaskState.failed` status update with a descriptive error message
and return from `execute()`.

---

## 5. GreenAgent Lifecycle Management

### Instance Cache

The executor maintains a dictionary mapping `context_id` to `GreenAgent`
instances, plus a per-context asyncio lock to prevent concurrent assessments
(since `GreenAgent.run()` is not designed for concurrent execution):

```python
self._agents: dict[str, GreenAgent] = {}
self._context_locks: dict[str, asyncio.Lock] = {}
```

A lock is created lazily for each `context_id` on first access. The
`execute()` method acquires the lock before dispatching `agent.run()` and
releases it when done. This serializes assessments within a context while
allowing different contexts to run in parallel.

### Creation Flow

When `execute()` is called with a `context_id` that doesn't have an existing
`GreenAgent`:

1. Allocate a UES port via the port allocator
2. Create a new `GreenAgent(ues_port=port, config=self._config)`
3. Call `await agent.startup()`
4. Store in `self._agents[context_id]`

### Reuse

If a `GreenAgent` already exists for the `context_id`, reuse it — it can run
multiple sequential assessments (different scenarios) within the same context.

### Cleanup

When the executor shuts down, it must call `shutdown()` on every cached
`GreenAgent` to stop their UES server subprocesses.

---

## 6. UES Port Allocation

Each `GreenAgent` needs a unique UES port. The executor uses a simple
counter-based allocator starting from `config.ues_base_port`:

```python
class _PortAllocator:
    """Simple sequential port allocator."""

    def __init__(self, base_port: int) -> None:
        self._next_port = base_port

    def allocate(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port
```

This is sufficient for the expected concurrency (a handful of parallel
assessments). If port reuse becomes necessary (e.g., after `GreenAgent`
shutdown), a pool-based allocator could replace this.

---

## 7. TaskUpdateEmitter Wiring

The A2A SDK provides `TaskUpdater` which takes an `EventQueue`, `task_id`, and
`context_id`. The executor wraps this in our `TaskUpdateEmitter`:

```python
from a2a.server.tasks import TaskUpdater

updater = TaskUpdater(
    event_queue=event_queue,
    task_id=context.task_id,
    context_id=context.context_id,
)
emitter = TaskUpdateEmitter(updater)
```

The `emitter` is then passed to `GreenAgent.run()`, which uses it to emit
structured assessment updates throughout the assessment lifecycle.

---

## 8. Assessment Dispatch

The core of `execute()` is:

```python
# Load scenario and evaluators
scenario = self._scenario_manager.get_scenario(request.scenario_id)
evaluators = self._scenario_manager.get_evaluators(request.scenario_id)

# Create Purple agent client
async with A2AClientWrapper(request.purple_agent_url) as purple_client:
    # Run the assessment
    results = await agent.run(
        task_id=context.task_id,
        emitter=emitter,
        scenario=scenario,
        evaluators=evaluators,
        purple_client=purple_client,
        assessment_config=request.config,
    )

# Emit results as artifact (before terminal state)
artifact = create_json_artifact(
    data=results.model_dump(mode="json"),
    name="assessment_results",
    description=f"Assessment results for scenario {request.scenario_id}",
)
await updater.add_artifact(parts=artifact.parts, name=artifact.name)

# Executor owns the terminal state
await updater.update_status(state=TaskState.completed)
```

### End State and Terminal State Ownership

The **executor** owns the terminal task state, not `GreenAgent.run()`. The
ordering after `run()` completes:

1. `GreenAgent.run()` emits an `assessment_completed` update with
   `TaskState.working` (informational only — not terminal).
2. The executor emits the results artifact via `updater.add_artifact()`.
3. The executor emits `TaskState.completed` via `updater.update_status()`.

This requires `TaskUpdateEmitter.assessment_completed()` to use
`TaskState.working` instead of `TaskState.completed`. This change ensures
the artifact is emitted before the task reaches terminal state (since
`TaskUpdater.update_status()` raises `RuntimeError` after terminal state).

If `run()` raises an exception, the executor catches it, emits a
`TaskState.failed` status, and returns.

---

## 9. Cancellation Handling

```python
async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
    context_id = context.context_id
    task_id = context.task_id

    agent = self._agents.get(context_id)
    if agent is not None and task_id is not None:
        await agent.cancel(task_id)

    # Emit cancellation acknowledged
    updater = TaskUpdater(event_queue, task_id, context_id)
    await updater.update_status(state=TaskState.canceled)
```

Note: `GreenAgent.cancel()` just sets a flag — the actual cleanup happens
within `run()` when it checks `_cancelled` between turns. The `TaskState.canceled`
status emitted here tells the A2A framework the cancellation was acknowledged.

**Edge case**: If no agent exists for the `context_id` (request was never
started or already completed), we still emit `TaskState.canceled` to
satisfy the A2A protocol.

---

## 10. Error Handling Strategy

### Exception Categories

| Exception | Source | Recovery |
|-----------|--------|----------|
| `ValueError` (bad request) | Request parsing | Emit `TaskState.failed`, return |
| `ScenarioNotFoundError` | `ScenarioManager` | Emit `TaskState.failed`, return |
| `ScenarioValidationError` | `ScenarioManager` | Emit `TaskState.failed`, return |
| `EvaluatorLoadError` | `ScenarioManager` | Emit `TaskState.failed`, return |
| `UESServerError` | `GreenAgent.startup()` | Emit `TaskState.failed`, clean up agent |
| `RuntimeError` | `GreenAgent.run()` | Emit `TaskState.failed`, *agent remains* |
| Unexpected exceptions | Anywhere | Emit `TaskState.failed`, log traceback |

### Error Message Format

Failed tasks emit a status update with a `Message` containing a text description
of the error:

```python
from src.common.a2a.messages import create_text_message

error_msg = create_text_message(
    text=f"Assessment failed: {error}",
    role="agent",
)
await updater.update_status(state=TaskState.failed, message=error_msg)
```

### Agent Recovery

If `GreenAgent.startup()` fails, the agent is removed from the cache and its
port is lost (acceptable since sequential allocation means the next agent gets
the next port). If `GreenAgent.run()` fails, the agent remains in the cache —
it can still serve future assessment requests for the same `context_id`.

---

## 11. Shutdown & Cleanup

The executor needs a `shutdown()` method (not part of `AgentExecutor` ABC) that
the server layer calls on application teardown:

```python
async def shutdown(self) -> None:
    """Shut down all managed GreenAgent instances."""
    for context_id, agent in self._agents.items():
        try:
            await agent.shutdown()
        except Exception:
            logger.exception("Error shutting down agent for context %s", context_id)
    self._agents.clear()
```

This should be called from `server.py`'s shutdown handler or `atexit` hook.

---

## 12. Method Inventory

### Constructor

```python
def __init__(
    self,
    config: GreenAgentConfig,
    scenario_manager: ScenarioManager,
) -> None:
```

**Accepts**: Configuration and a pre-built `ScenarioManager`.
**Creates**: Port allocator, empty agent cache (`_agents`), empty context
lock cache (`_context_locks`).

### `execute()` — from AgentExecutor ABC

```python
async def execute(
    self, context: RequestContext, event_queue: EventQueue
) -> None:
```

**Steps**:
1. Parse and validate request
2. Acquire per-context asyncio lock
3. Get or create `GreenAgent` for `context_id`
4. Create `TaskUpdater` → `TaskUpdateEmitter`
5. Load scenario + evaluators
6. Create `A2AClientWrapper` for Purple (per-assessment, not cached)
7. Run assessment via `agent.run()`
8. Emit results artifact
9. Emit `TaskState.completed` (executor owns terminal state)
10. Release context lock, handle errors

### `cancel()` — from AgentExecutor ABC

```python
async def cancel(
    self, context: RequestContext, event_queue: EventQueue
) -> None:
```

**Steps**:
1. Look up agent by `context_id`
2. Forward cancellation to `agent.cancel(task_id)`
3. Emit `TaskState.canceled`

### `shutdown()`

```python
async def shutdown(self) -> None:
```

**Steps**:
1. Iterate all cached agents
2. Call `agent.shutdown()` on each
3. Clear agent and lock caches

### `_parse_request()` (private)

```python
def _parse_request(self, context: RequestContext) -> AssessmentRequest:
```

Extracts and validates the assessment request from the A2A message.

### `_get_or_create_agent()` (private)

```python
async def _get_or_create_agent(self, context_id: str) -> GreenAgent:
```

Looks up existing agent or creates + starts a new one.

---

## 13. Dependencies

### Internal

| Module | Provides |
|--------|----------|
| `src/green/agent` | `GreenAgent` |
| `src/green/scenarios/loader` | `ScenarioManager`, `ScenarioNotFoundError`, etc. |
| `src/common/agentbeats/config` | `GreenAgentConfig` |
| `src/common/agentbeats/updates` | `TaskUpdateEmitter` |
| `src/common/a2a/client` | `A2AClientWrapper` |
| `src/common/a2a/messages` | `get_data_content`, `create_text_message` |
| `src/common/a2a/artifacts` | `create_json_artifact` |

### External (from `a2a-python` SDK)

| Class | Used for |
|-------|----------|
| `AgentExecutor` (ABC) | Interface to implement |
| `RequestContext` | Incoming request data |
| `EventQueue` | Outgoing event publication |
| `TaskUpdater` | Status/artifact emission helper |
| `TaskState` | Task state constants |

---

*Document created: February 18, 2026*
