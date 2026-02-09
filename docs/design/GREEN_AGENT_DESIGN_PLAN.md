# GreenAgent Class Design Document

This document provides the comprehensive design for the `GreenAgent` class (Phase 3.8 from `IMPLEMENTATION_PLAN.md`) and identifies open questions, potential issues, and blockers.

**Date**: February 7, 2026  
**Status**: 📋 DESIGN REVIEW

---

## Table of Contents

1. [Design Overview](#1-design-overview)
2. [Core Responsibilities](#2-core-responsibilities)
3. [Class Architecture](#3-class-architecture)
4. [Lifecycle Management](#4-lifecycle-management)
5. [UES Server Management](#5-ues-server-management)
6. [API Key Management](#6-api-key-management)
7. [Assessment Execution Flow](#7-assessment-execution-flow)
8. [Turn Loop Implementation](#8-turn-loop-implementation)
9. [Purple Agent Communication](#9-purple-agent-communication)
10. [Response Scheduling](#10-response-scheduling)
11. [State Management](#11-state-management)
12. [Error Handling](#12-error-handling)
13. [Supporting Data Models](#13-supporting-data-models)
14. [Dependencies Summary](#14-dependencies-summary)
15. [Open Questions and Issues](#15-open-questions-and-issues)
16. [Implementation Plan](#16-implementation-plan)

---

## 1. Design Overview

### Purpose

The `GreenAgent` class is the core orchestrator for AgentBeats assessments. Each instance:
- Owns and manages its own UES server instance
- Runs assessments for a single `context_id` (Purple agent session)
- Coordinates response generation, action logging, and evaluation
- Produces assessment results as A2A artifacts

### Position in Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Green Agent System                               │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │  A2AServer       │───▶│ GreenAgentExecutor│                       │
│  │  (server.py)     │    │  (executor.py)    │                       │
│  └──────────────────┘    └────────┬─────────┘                       │
│                                   │                                  │
│                     ┌─────────────┴─────────────┐                   │
│                     ▼                           ▼                   │
│            ┌──────────────────┐       ┌──────────────────┐          │
│            │   GreenAgent     │       │   GreenAgent     │          │
│            │   (context A)    │       │   (context B)    │          │
│            │                  │       │                  │          │
│            │  ┌────────────┐  │       │  ┌────────────┐  │          │
│            │  │ UES Server │  │       │  │ UES Server │  │          │
│            │  │ (port 8100)│  │       │  │ (port 8101)│  │          │
│            │  └────────────┘  │       │  └────────────┘  │          │
│            └──────────────────┘       └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions (from `IMPLEMENTATION_PLAN.md`)

1. **Context Isolation**: Each `context_id` gets its own `GreenAgent` with its own UES server
2. **Resource Reuse**: LLMs and UES server persist across assessments within a context
3. **Per-Assessment Helpers**: `ResponseGenerator`, `CriteriaJudge`, `ActionLogBuilder`, `NewMessageCollector` are created fresh per assessment

---

## 2. Core Responsibilities

| Responsibility | Description | Component Used |
|----------------|-------------|----------------|
| **UES Lifecycle** | Start, manage, and shutdown UES server | subprocess, AsyncUESClient |
| **Scenario Setup** | Load scenario initial state into UES | AsyncUESClient.simulation, scenario import |
| **API Key Generation** | Create user-level keys for Purple agent | UES /keys endpoint |
| **Turn Orchestration** | Run the assessment turn loop | A2A messaging, time control |
| **Action Tracking** | Build action log from UES events | ActionLogBuilder |
| **Response Generation** | Generate character responses | ResponseGenerator |
| **Response Scheduling** | Inject character responses into UES | Email/SMS receive endpoints |
| **Evaluation** | Score Purple agent performance | CriteriaJudge |
| **Result Assembly** | Build and return assessment results | AssessmentResults model |

---

## 3. Class Architecture

### 3.1 Per-Agent Resources (Long-Lived)

These resources are created once when the `GreenAgent` is instantiated and persist across multiple assessments:

```python
# Owned by GreenAgent instance
self._ues_process: subprocess.Popen      # UES server process
self._proctor_api_key: str                # Admin key (pre-generated, passed via env var)
self._ues_port: int                       # Assigned port for UES
self.ues_client: AsyncUESClient           # Client using proctor key
self.response_llm: BaseChatModel          # LLM for response generation
self.evaluation_llm: BaseChatModel        # LLM for evaluation (temp=0)
```

### 3.2 Per-Assessment Resources (Short-Lived)

These resources are created fresh for each `run()` call:

```python
# Created in run() for each assessment
action_log_builder: ActionLogBuilder       # Tracks Purple actions
message_collector: NewMessageCollector     # Collects new messages
response_generator: ResponseGenerator      # Generates character responses
criteria_judge: CriteriaJudge              # Evaluates criteria
summarization_llm: BaseChatModel           # For thread summarization (optional)
```

### 3.3 Class Skeleton

```python
class GreenAgent:
    """High-level orchestrator for assessments within a context.
    
    Each GreenAgent instance owns its own UES server and can run multiple
    sequential assessments. The executor creates one GreenAgent per context_id.
    
    Lifecycle:
        1. Executor creates GreenAgent with allocated port
        2. Executor calls startup() to spin up UES server
        3. Executor calls run() for each assessment task
        4. Executor calls shutdown() when context is no longer needed
    
    Attributes:
        config: Green agent configuration.
        ues_port: Port for this agent's UES server.
        ues_client: Async client for UES API calls (proctor-level).
        response_llm: LLM for generating character responses.
        evaluation_llm: LLM for criteria evaluation.
    """
    
    def __init__(self, ues_port: int, config: GreenAgentConfig) -> None:
        """Initialize GreenAgent (does not start UES server)."""
        ...
    
    async def startup(self) -> None:
        """Start UES server and create client. Call after __init__."""
        ...
    
    async def shutdown(self) -> None:
        """Stop UES server and cleanup resources."""
        ...
    
    async def cancel(self, task_id: str) -> None:
        """Request cancellation of an ongoing assessment."""
        ...
    
    async def run(
        self,
        task_id: str,
        emitter: TaskUpdateEmitter,
        scenario: ScenarioConfig,
        evaluators: EvaluatorRegistry,
        purple_client: A2AClientWrapper,
        assessment_config: dict[str, Any],
    ) -> AssessmentResults:
        """Run a complete assessment. Main entry point."""
        ...
    
    # -- Turn orchestration (private) --
    
    async def _run_turn(self, ...) -> TurnResult:
        """Execute a single turn (send/receive + end-of-turn processing)."""
        ...
    
    async def _process_turn_end(self, ...) -> EndOfTurnResult:
        """Process end-of-turn: apply events, generate responses, advance."""
        ...
```

---

## 4. Lifecycle Management

### 4.1 GreenAgent Lifecycle

```
             ┌───────────────┐
             │  __init__()   │
             │               │
             │ - Store port  │
             │ - Store config│
             │ - Create LLMs │
             └──────┬────────┘
                    │
                    ▼
             ┌──────────────┐
             │  startup()   │
             │              │
             │ - Start UES  │
             │ - Get admin  │
             │   key        │
             │ - Create     │
             │   client     │
             │ - Wait ready │
             └──────┬───────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    ┌───────┐   ┌───────┐   ┌───────┐
    │ run() │   │ run() │   │ run() │   (Multiple assessments)
    │ task1 │   │ task2 │   │ task3 │
    └───────┘   └───────┘   └───────┘
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
             ┌───────────────┐
             │  shutdown()   │
             │               │
             │ - Close client│
             │ - Terminate   │
             │   UES process │ 
             └───────────────┘
```

### 4.2 Assessment Lifecycle (within `run()`)

```
run() called
    │
    ▼
┌─────────────────────────┐
│ Phase 1: SETUP          │
│ - Reset UES             │
│ - Load scenario state   │
│ - Create user API key   │
│ - Initialize collectors │
│ - Capture initial state │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Phase 2: START          │
│ - Build initial summary │
│ - Send AssessmentStart  │
│   to Purple             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Phase 3: TURN LOOP              │
│                                 │
│  while turn < max_turns         │
│    and not cancelled            │
│    and not early_completion:    │
│                                 │
│    ├── Send TurnStart to Purple │
│    ├── Wait for TurnComplete    │
│    ├── _process_turn_end():     │
│    │   ├── Advance UES time 1s  │
│    │   │   (apply Purple events)│
│    │   ├── Collect UES events   │
│    │   ├── Build action log     │
│    │   ├── Collect new messages │
│    │   ├── Generate responses   │
│    │   ├── Schedule responses   │
│    │   └── Advance UES time by  │
│    │       remainder            │
│    └── turn += 1                │
│                                 │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ Phase 4: EVALUATION     │
│ - Capture final state   │
│ - Build eval context    │
│ - Run CriteriaJudge     │
│ - Aggregate scores      │
└───────────┬─────────────┘
            │
            ▼
┌──────────────────────────┐
│ Phase 5: COMPLETION      │
│ - Send AssessmentComplete│
│   to Purple              │
│ - Revoke user API key    │
│ - Build results          │
│ - Return results         │
└──────────────────────────┘
```

---

## 5. UES Server Management

### 5.1 Starting UES Server

The GreenAgent needs to start a UES server as a subprocess with a known admin API key.

**Approach**: UES supports the `UES_ADMIN_KEY` environment variable (and equivalent `--admin-key`
CLI flag) to supply a pre-set admin key secret. The GreenAgent generates a random
key at `__init__` time, passes it to the UES subprocess via the environment
variable, and immediately knows the admin key without parsing stdout.

See UES `docs/api/AUTHENTICATION.md` § "Programmatic Key Retrieval" for details.

**Key generation requirements** (from UES docs):
- The secret must be at least 32 characters long
- Any string format is accepted (hex, alphanumeric, etc.)
- The secret is used as-is — UES does not modify or hash it

**Proposed Approach**:

```python
import secrets

def _generate_admin_key(self) -> str:
    """Generate a random admin API key for the UES server.
    
    Returns:
        A 64-character hex string suitable for UES_ADMIN_KEY.
    """
    return secrets.token_hex(32)  # 64 hex chars, exceeds 32-char minimum

async def _start_ues_server(self) -> subprocess.Popen:
    """Start UES server with a pre-set admin key.
    
    The admin key is generated at __init__ time and passed to the
    UES server via the UES_ADMIN_KEY environment variable. This
    avoids the need to parse stdout for the key.
    
    Returns:
        The UES server subprocess.
    """
    env = {**os.environ, "UES_ADMIN_KEY": self._proctor_api_key}
    
    process = subprocess.Popen(
        [
            "uv", "run", "python", "-m", "ues.cli", "server",
            "--host", "127.0.0.1",
            "--port", str(self._ues_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        env=env,
    )
    
    return process
```

**Note**: UES server stdout/stderr is still captured for logging and
diagnostics. A background task should drain the pipe to prevent the
subprocess from blocking on a full buffer.

### 5.2 Waiting for UES Ready

After starting the server, we need to wait until it's accepting connections:

```python
async def _wait_for_ues_ready(self, timeout: float = 30.0) -> None:
    """Wait for UES server to be ready.
    
    Polls the /health endpoint until it responds or timeout.
    
    Args:
        timeout: Maximum seconds to wait.
        
    Raises:
        TimeoutError: If server doesn't become ready in time.
    """
    import httpx
    import asyncio
    
    start = time.monotonic()
    health_url = f"http://127.0.0.1:{self._ues_port}/health"
    
    async with httpx.AsyncClient() as client:
        while time.monotonic() - start < timeout:
            try:
                response = await client.get(health_url)
                if response.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.1)
    
    raise TimeoutError(f"UES server not ready after {timeout}s")
```

### 5.3 UES Server Configuration Considerations

| Setting | Value | Rationale |
|---------|-------|-----------|
| Host | `127.0.0.1` | Local only; Purple accesses via Green-provided URL |
| Port | Dynamically assigned | Each GreenAgent gets unique port |
| auto_advance | `False` | Green controls time via explicit advance calls |
| Reload | `False` | Production mode |

---

## 6. API Key Management

### 6.1 Key Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│                     API Key Hierarchy                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐                                         │
│  │   Admin Key         │   Pre-generated by GreenAgent           │
│  │   (proctor-level)   │   Passed via UES_ADMIN_KEY env var      │
│  │                     │   Full access to all endpoints           │
│  │   Permissions: ["*"]│                                         │
│  └──────────┬──────────┘                                         │
│             │                                                     │
│             │ Creates                                             │
│             ▼                                                     │
│  ┌─────────────────────┐                                         │
│  │   User Key          │   Created per-assessment                │
│  │   (purple-level)    │   Given to Purple agent                 │
│  │                     │   Limited permissions                    │
│  │   Permissions:      │   key_id used as agent_id for           │
│  │   [user-level perms]│   event filtering                       │
│  └─────────────────────┘                                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 User-Level Permissions

Based on `ASSESSMENT_FLOW.md` Section 3, Purple agents should have access to:

**ALLOWED** (User-Level Permissions):

```python
USER_PERMISSIONS = [
    # State & Query
    "email:state", "email:query",
    "sms:state", "sms:query",
    "calendar:state", "calendar:query",
    "chat:state", "chat:query",
    "time:read",  # Read-only time access
    
    # Email user-side actions
    "email:send", "email:read", "email:unread", "email:star", "email:unstar",
    "email:archive", "email:delete", "email:label", "email:unlabel", "email:move",
    
    # SMS user-side actions
    "sms:send", "sms:read", "sms:unread", "sms:delete", "sms:react",
    
    # Calendar user-side actions
    "calendar:create", "calendar:update", "calendar:delete", "calendar:respond",
    
    # Chat user-side actions
    "chat:send",
]
```

**FORBIDDEN** (Proctor-Only):

```python
PROCTOR_ONLY_PERMISSIONS = [
    # Simulator-side actions (inject external events)
    "email:receive", "sms:receive", "chat:receive",
    "location:update", "weather:update",
    
    # Time control
    "time:advance", "time:set", "time:skip", "time:pause", "time:resume",
    
    # Simulation control
    "simulation:*",
    
    # Scenario import/export
    "scenario:*",
    
    # Events management
    "events:*",
    
    # Holds system
    "simulation:hold", "simulation:release", "simulation:holds",
    
    # Key management
    "keys:*",
    
    # Access logs
    "logs:*",
    
    # Webhooks
    "webhooks:*",
]
```

### 6.3 Key Creation Implementation

The `key_id` returned by UES serves a dual purpose: it identifies the key for revocation, and it becomes the `agent_id` for all events created by the Purple agent. This allows the Green agent to filter UES event history to only those actions performed by Purple.

```python
async def _create_user_api_key(self, assessment_id: str) -> tuple[str, str]:
    """Create a user-level API key for Purple agent.
    
    Args:
        assessment_id: Unique assessment ID for key naming.
        
    Returns:
        Tuple of (api_key_secret, key_id). The key_id is used as the
        agent_id for filtering UES events created by Purple.
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{self._ues_port}/keys",
            headers={"X-API-Key": self._proctor_api_key},
            json={
                "name": f"Purple Agent ({assessment_id})",
                "permissions": USER_PERMISSIONS,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["secret"], data["key_id"]
```

### 6.4 Key Revocation

At assessment end, revoke the user key:

```python
async def _revoke_user_api_key(self, key_id: str) -> None:
    """Revoke a user API key after assessment ends."""
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.delete(
            f"http://127.0.0.1:{self._ues_port}/keys/{key_id}",
            headers={"X-API-Key": self._proctor_api_key},
        )
        # Ignore 404 (key already revoked or doesn't exist)
        if response.status_code not in (200, 404):
            response.raise_for_status()
```

---

## 7. Assessment Execution Flow

### 7.1 run() Method Flow

```python
async def run(
    self,
    task_id: str,
    emitter: TaskUpdateEmitter,
    scenario: ScenarioConfig,
    evaluators: EvaluatorRegistry,
    purple_client: A2AClientWrapper,
    assessment_config: dict[str, Any],
) -> AssessmentResults:
    """Run a complete assessment.
    
    This is the main entry point for running an assessment.
    """
    # Track start time for duration calculation
    start_time = datetime.now(timezone.utc)
    assessment_id = str(uuid.uuid4())
    
    # Mark this assessment as active
    self._current_task_id = task_id
    self._cancelled = False
    
    try:
        # === Phase 1: Setup ===
        await emitter.assessment_started(
            assessment_id=assessment_id,
            scenario_id=scenario.scenario_id,
            participant_url=purple_client.agent_url,
            start_time=start_time,
        )
        
        # Reset UES and load scenario
        await self._setup_ues(scenario)
        
        # Create user API key for Purple
        user_api_key, user_key_id = await self._create_user_api_key(assessment_id)
        
        # Create per-assessment helpers
        # The user_key_id becomes the agent_id for all Purple actions
        action_log_builder = ActionLogBuilder(purple_agent_id=user_key_id)
        message_collector = NewMessageCollector(client=self.ues_client)
        response_generator = ResponseGenerator(
            client=self.ues_client,
            scenario_config=scenario,
            response_llm=self.response_llm,
            summarization_llm=self.response_llm,  # Reuse for now
        )
        criteria_judge = CriteriaJudge(
            llm=self.evaluation_llm,
            criteria=scenario.criteria,
            evaluators=evaluators,
            emitter=emitter,
        )
        
        # Initialize message collector
        time_state = await self.ues_client.time.get_state()
        await message_collector.initialize(time_state.current_time)
        
        # Capture initial state
        initial_state = await self._capture_state_snapshot()
        
        # === Phase 2: Send Start to Purple ===
        initial_summary = await self._build_initial_state_summary()
        await self._send_assessment_start(
            purple_client=purple_client,
            scenario=scenario,
            initial_summary=initial_summary,
            ues_url=f"http://127.0.0.1:{self._ues_port}",
            api_key=user_api_key,
        )
        
        # === Phase 3: Turn Loop ===
        max_turns = assessment_config.get("max_turns", self.config.default_max_turns)
        turn = 0
        completion_reason = "scenario_complete"
        
        while turn < max_turns and not self._cancelled:
            turn += 1
            
            turn_result = await self._run_turn(
                turn=turn,
                emitter=emitter,
                purple_client=purple_client,
                action_log_builder=action_log_builder,
                message_collector=message_collector,
                response_generator=response_generator,
            )
            
            if turn_result.early_completion:
                completion_reason = "early_completion"
                break
        
        if self._cancelled:
            completion_reason = "cancelled"
        elif turn >= max_turns:
            completion_reason = "max_turns_reached"
        
        # === Phase 4: Evaluation ===
        await emitter.evaluation_started()
        final_state = await self._capture_state_snapshot()
        
        eval_context = AgentBeatsEvalContext(
            client=self.ues_client,
            scenario_config=scenario.model_dump(),
            action_log=action_log_builder.get_log_with_turns(),
            initial_state=initial_state,
            final_state=final_state,
            user_prompt=scenario.user_prompt,
        )
        
        criteria_results = await criteria_judge.evaluate_all(eval_context)
        scores = criteria_judge.aggregate_scores(criteria_results)
        
        # === Phase 5: Completion ===
        await self._send_assessment_complete(purple_client, completion_reason)
        await self._revoke_user_api_key(user_key_id)
        
        # Build results
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        results = self._build_results(
            assessment_id=assessment_id,
            scenario=scenario,
            scores=scores,
            criteria_results=criteria_results,
            action_log=action_log_builder.get_log_with_turns(),
            turns_completed=turn,
            duration=duration,
            status="completed" if completion_reason != "cancelled" else "cancelled",
        )
        
        await emitter.assessment_completed(
            reason=completion_reason,
            total_turns=turn,
            total_actions=action_log_builder.get_total_actions(),
            duration_seconds=duration,
            overall_score=int(scores.overall.score),
            max_score=int(scores.overall.max_score),
        )
        
        return results
        
    finally:
        self._current_task_id = None
```

---

## 8. Turn Loop Implementation

### 8.1 Single Turn Execution

```python
async def _run_turn(
    self,
    turn: int,
    emitter: TaskUpdateEmitter,
    purple_client: A2AClientWrapper,
    action_log_builder: ActionLogBuilder,
    message_collector: NewMessageCollector,
    response_generator: ResponseGenerator,
) -> TurnResult:
    """Execute a single assessment turn.
    
    Time is advanced in two phases:
      1. Advance by 1 second to apply Purple's scheduled events so they
         become visible in UES modality state.
      2. After processing events and scheduling character responses,
         advance by the remainder (time_step - 1s) to fire those
         responses before Purple's next turn.
    
    Returns:
        TurnResult with turn outcome details.
    """
    # Get current time before turn
    time_state = await self.ues_client.time.get_state()
    turn_start_time = time_state.current_time
    
    # == Step 1: Send TurnStart to Purple ==
    await emitter.turn_started(turn, turn_start_time)
    
    turn_start_msg = TurnStartMessage(
        turn_number=turn,
        current_time=turn_start_time,
        events_processed=0,  # From previous turn
    )
    
    # == Step 2: Wait for Purple's response ==
    response = await self._send_and_wait_purple(
        purple_client=purple_client,
        message=turn_start_msg,
        timeout=self.config.default_turn_timeout,
    )
    
    # Check for early completion
    if isinstance(response, EarlyCompletionMessage):
        await emitter.turn_completed(
            turn_number=turn,
            actions_taken=0,
            time_advanced="PT0S",
            early_completion_requested=True,
        )
        return TurnResult(
            turn_number=turn,
            actions_taken=0,
            time_step="PT0S",
            events_processed=0,
            early_completion=True,
            notes=response.reason,
        )
    
    turn_complete_msg = response  # TurnCompleteMessage
    time_step = turn_complete_msg.time_step or "PT1H"
    
    # == Step 3: End-of-turn processing ==
    # Delegates to _process_turn_end for apply advance, event collection,
    # response generation, response scheduling, and remainder advance.
    # Extracted as a helper to allow future changes (e.g., event-by-event
    # processing) without modifying the overall turn structure.
    end_of_turn = await self._process_turn_end(
        turn=turn,
        turn_start_time=turn_start_time,
        time_step=time_step,
        emitter=emitter,
        action_log_builder=action_log_builder,
        message_collector=message_collector,
        response_generator=response_generator,
    )
    
    await emitter.turn_completed(
        turn_number=turn,
        actions_taken=end_of_turn.actions_taken,
        time_advanced=time_step,
        early_completion_requested=False,
    )
    
    return TurnResult(
        turn_number=turn,
        actions_taken=end_of_turn.actions_taken,
        time_step=time_step,
        events_processed=end_of_turn.total_events,
        early_completion=False,
        notes=turn_complete_msg.notes,
    )
```

### 8.2 End-of-Turn Processing

The `_process_turn_end` helper encapsulates all processing between Purple's
turn completion and the start of the next turn: applying Purple's events,
collecting and logging actions, generating character responses, scheduling
them into UES, and advancing the remaining time. This separation makes it
straightforward to change the processing strategy (e.g., switching from
batch processing to event-by-event processing) without modifying the
overall turn orchestration in `_run_turn`.

```python
@dataclass
class EndOfTurnResult:
    """Result from end-of-turn processing.
    
    Attributes:
        actions_taken: Number of Purple agent actions observed.
        total_events: Total UES events executed across both time
            advances (apply + remainder).
        responses_generated: Number of character responses scheduled.
    """
    actions_taken: int
    total_events: int
    responses_generated: int

async def _process_turn_end(
    self,
    turn: int,
    turn_start_time: datetime,
    time_step: str,
    emitter: TaskUpdateEmitter,
    action_log_builder: ActionLogBuilder,
    message_collector: NewMessageCollector,
    response_generator: ResponseGenerator,
) -> EndOfTurnResult:
    """Process end-of-turn: apply events, generate responses, advance time.
    
    This method handles all processing after Purple completes its turn
    and before the next turn begins. It is intentionally separated from
    _run_turn to allow the processing strategy to be changed
    independently of the turn orchestration logic.
    
    The current implementation uses a batch strategy:
      1. Apply advance (1s) to make Purple's events visible
      2. Collect all events and new messages at once
      3. Generate all character responses
      4. Schedule all responses
      5. Remainder advance to fire responses
    
    Future alternatives (e.g., event-by-event processing with
    per-event response generation) can be implemented by replacing
    this method without touching _run_turn.
    
    Args:
        turn: Current turn number.
        turn_start_time: Simulation time at the start of this turn.
        time_step: ISO 8601 duration for total time advancement.
        emitter: Task update emitter for observability.
        action_log_builder: Tracks Purple agent actions.
        message_collector: Collects new messages from UES state.
        response_generator: Generates character responses.
    
    Returns:
        EndOfTurnResult with processing outcome details.
    """
    # == Phase 1: Apply advance — advance by 1s to apply Purple's events ==
    apply_result = await self._advance_time("PT1S")
    
    # Get time after apply advance
    time_state = await self.ues_client.time.get_state()
    apply_time = time_state.current_time
    
    # == Phase 2: Collect and process events ==
    action_log_builder.start_turn(turn)
    
    events = await self.ues_client.events.list_events(
        start_time=turn_start_time,
        end_time=apply_time,
        status="executed",
    )
    
    purple_entries = action_log_builder.add_events_from_turn(events.events)
    action_log_builder.end_turn()
    
    # Emit action updates for observability
    for entry in purple_entries:
        await emitter.action_observed(
            turn_number=turn,
            timestamp=entry.timestamp,
            action=entry.action,
            parameters=entry.parameters,
            success=entry.success,
            error_message=entry.error_message,
        )
    
    # == Phase 3: Collect new messages ==
    new_messages = await message_collector.collect(apply_time)
    
    # == Phase 4: Generate character responses ==
    responses = await response_generator.process_new_messages(
        new_messages=new_messages,
        current_time=apply_time,
    )
    
    # == Phase 5: Schedule responses in UES ==
    for scheduled in responses:
        await self._schedule_response(scheduled)
    
    await emitter.responses_generated(
        count=len(responses),
        modalities=[r.modality for r in responses],
    )
    
    # == Phase 6: Remainder advance — advance by (time_step - 1s) ==
    # This fires the character responses scheduled above
    remainder_result = await self._advance_remainder(
        time_step=time_step, apply_seconds=1
    )
    
    return EndOfTurnResult(
        actions_taken=len(purple_entries),
        total_events=(
            apply_result.events_executed
            + remainder_result.events_executed
        ),
        responses_generated=len(responses),
    )
```

### 8.3 Time Advancement

Time is advanced in two phases per turn. The `_advance_time()` method handles
a single advance call, while `_advance_remainder()` computes and applies the
remainder after the initial 1-second apply advance.

```python
async def _advance_time(self, time_step: str) -> AdvanceTimeResponse:
    """Advance UES simulation time by a given duration.
    
    Args:
        time_step: ISO 8601 duration string (e.g., "PT1H", "PT1S").
        
    Returns:
        AdvanceTimeResponse from UES.
    """
    from src.green.scenarios.schema import parse_iso8601_duration
    
    duration = parse_iso8601_duration(time_step)
    seconds = int(duration.total_seconds())
    
    return await self.ues_client.time.advance(seconds=seconds)

async def _advance_remainder(
    self, time_step: str, apply_seconds: int = 1,
) -> AdvanceTimeResponse:
    """Advance UES simulation time by the remainder after apply advance.
    
    Computes time_step - apply_seconds and advances by that amount.
    If the remainder is zero or negative (time_step <= apply_seconds),
    no advance is performed.
    
    Args:
        time_step: Original ISO 8601 duration string (e.g., "PT1H").
        apply_seconds: Seconds already advanced in the apply phase.
        
    Returns:
        AdvanceTimeResponse from UES (or a zero-event response if
        no advancement was needed).
    """
    from src.green.scenarios.schema import parse_iso8601_duration
    
    duration = parse_iso8601_duration(time_step)
    total_seconds = int(duration.total_seconds())
    remainder = total_seconds - apply_seconds
    
    if remainder <= 0:
        # time_step was <= 1s, no remainder to advance
        return AdvanceTimeResponse(events_executed=0)
    
    return await self.ues_client.time.advance(seconds=remainder)
```

### 8.4 Two-Phase Time Advancement Rationale

The two-phase approach solves a critical ordering problem:

1. **Apply advance (1s)**: Purple's actions during its turn are submitted as UES
   events that need a time advance to be applied. The 1-second advance ensures
   these events fire and their effects (sent emails, created calendar events,
   etc.) become visible in UES modality state. Without this, Green would query
   modality state and see no new messages from Purple's actions.

2. **Remainder advance (time_step - 1s)**: After Green generates character
   responses and schedules them as future UES events, the remainder advance
   fires those responses. This means Purple will see character replies when it
   queries UES state on its next turn, creating a natural conversation flow.

```
    Purple's turn          Apply (1s)        Green processes       Remainder
    ─────────────────┬───────────────┬────────────────────┬────────────────────
    Purple sends     │ Events fire,  │ Green reads state, │ Character responses
    emails, creates  │ emails appear │ generates replies,  │ fire, Purple sees
    events via API   │ in UES state  │ schedules responses │ them next turn
    ─────────────────┴───────────────┴────────────────────┴────────────────────
```

---

## 9. Purple Agent Communication

### 9.1 Message Exchange Pattern

```
    GreenAgent                              PurpleAgent
        │                                        │
        │   AssessmentStartMessage               │
        │────────────────────────────────────────▶
        │                                        │
        │   (Purple reads chat for instructions) │
        │                                        │
        │   TurnStartMessage                     │
        │────────────────────────────────────────▶
        │                                        │
        │                    TurnCompleteMessage │
        │◀────────────────────────────────────────
        │      or EarlyCompletionMessage         │
        │                                        │
        │   [Green advances time, generates      │
        │    responses, schedules them]          │
        │                                        │
        │   TurnStartMessage                     │
        │────────────────────────────────────────▶
        │                                        │
        │   ... repeat ...                       │
        │                                        │
        │   AssessmentCompleteMessage            │
        │────────────────────────────────────────▶
        │                                        │
```

### 9.2 Sending Messages to Purple

```python
async def _send_and_wait_purple(
    self,
    purple_client: A2AClientWrapper,
    message: BaseModel,
    timeout: float,
) -> TurnCompleteMessage | EarlyCompletionMessage:
    """Send message to Purple and wait for response.
    
    Args:
        purple_client: A2A client for Purple agent.
        message: Pydantic model to send (serialized to JSON).
        timeout: Maximum seconds to wait for response.
        
    Returns:
        Parsed response message.
        
    Raises:
        TimeoutError: If Purple doesn't respond in time.
        ValueError: If response type is unexpected.
    """
    from a2a.types import DataPart, Message, Part, Role
    
    # Create A2A message with data payload
    a2a_message = Message(
        role=Role.user,
        parts=[Part(root=DataPart(data=message.model_dump()))],
        message_id=str(uuid.uuid4()),
    )
    
    # Send and wait for response (blocking=True)
    response = await asyncio.wait_for(
        purple_client.send_message(a2a_message, blocking=True),
        timeout=timeout,
    )
    
    # Parse response
    response_data = self._extract_response_data(response)
    message_type = response_data.get("message_type")
    
    if message_type == "turn_complete":
        return TurnCompleteMessage.model_validate(response_data)
    elif message_type == "early_completion":
        return EarlyCompletionMessage.model_validate(response_data)
    else:
        raise ValueError(f"Unexpected message type: {message_type}")
```

---

## 10. Response Scheduling

### 10.1 Scheduling Flow

```
ResponseGenerator.process_new_messages()
         │
         │ Returns list[ScheduledResponse]
         ▼
GreenAgent._schedule_response(scheduled)
         │
         ├── modality == "email" ──▶ ues_client.email.receive(...)
         │
         ├── modality == "sms" ──▶ ues_client.sms.receive_message(...)
         │
         └── modality == "calendar" ──▶ ues_client.calendar.respond_to_event(...)
```

### 10.2 Implementation

```python
async def _schedule_response(self, scheduled: ScheduledResponse) -> None:
    """Schedule a character response in UES.
    
    Uses proctor-level API calls to inject responses as if they came
    from external sources.
    
    Args:
        scheduled: The response to schedule.
    """
    if scheduled.modality == "email":
        await self._schedule_email_response(scheduled)
    elif scheduled.modality == "sms":
        await self._schedule_sms_response(scheduled)
    elif scheduled.modality == "calendar":
        await self._schedule_calendar_response(scheduled)
    else:
        raise ValueError(f"Unknown modality: {scheduled.modality}")

async def _schedule_email_response(self, scheduled: ScheduledResponse) -> None:
    """Schedule an email response."""
    await self.ues_client.email.receive(
        from_address=scheduled.character_email,
        to_addresses=scheduled.recipients,
        subject=scheduled.subject,
        body_text=scheduled.content,
        cc_addresses=scheduled.cc_recipients or None,
        thread_id=scheduled.thread_id,
        in_reply_to=scheduled.in_reply_to,
        references=scheduled.references or None,
        sent_at=scheduled.scheduled_time,
    )

async def _schedule_sms_response(self, scheduled: ScheduledResponse) -> None:
    """Schedule an SMS response."""
    await self.ues_client.sms.receive(
        from_number=scheduled.character_phone,
        to_numbers=scheduled.recipients,
        body=scheduled.content,
        replied_to_message_id=scheduled.original_message_id,
        sent_at=scheduled.scheduled_time,
    )

async def _schedule_calendar_response(self, scheduled: ScheduledResponse) -> None:
    """Schedule a calendar RSVP response."""
    await self.ues_client.calendar.respond_to_event(
        event_id=scheduled.event_id,
        attendee_email=scheduled.character_email,
        response=scheduled.rsvp_status,
        comment=scheduled.rsvp_comment,
    )
```

### 10.3 Response Scheduling via UES Events

Character responses are scheduled using UES's event system with the two-phase
time advancement approach:

1. Purple completes turn at time T
2. Green advances time by 1 second (T → T+1s) — applies Purple's events
3. Green collects messages from Purple's actions (now visible in state)
4. Green generates responses with `scheduled_time`
5. Scheduled responses are created as UES events via `events.create()`
6. Green advances time by remainder (T+1s → T+time_step) — fires responses
7. Purple sees character responses on its next turn

**Implementation**: Use UES events scheduling for realistic timing:

```python
await self.ues_client.events.create(
    scheduled_time=scheduled.scheduled_time,
    modality="email",
    data={
        "action": "receive",
        "from_address": character.email,
        ...
    },
)
```

**Note on scheduling times**: Response `scheduled_time` values should fall
between the apply advance time (T+1s) and the end of the remainder advance
(T+time_step). This ensures they fire during the remainder advance and are
visible to Purple before its next turn.

---

## 11. State Management

### 11.1 Capturing State Snapshots

```python
async def _capture_state_snapshot(self) -> dict[str, Any]:
    """Capture current state of all modalities.
    
    Used for initial/final state comparison in evaluation.
    
    Returns:
        Dict with modality states.
    """
    return {
        "email": (await self.ues_client.email.get_state()).model_dump(),
        "sms": (await self.ues_client.sms.get_state()).model_dump(),
        "calendar": (await self.ues_client.calendar.get_state()).model_dump(),
        "chat": (await self.ues_client.chat.get_state()).model_dump(),
        "time": (await self.ues_client.time.get_state()).model_dump(),
    }
```

### 11.2 Building Initial State Summary

```python
async def _build_initial_state_summary(self) -> InitialStateSummary:
    """Build summary of initial UES state for Purple.
    
    Returns:
        InitialStateSummary with counts for each modality.
    """
    email_state = await self.ues_client.email.get_state()
    sms_state = await self.ues_client.sms.get_state()
    calendar_state = await self.ues_client.calendar.get_state()
    chat_state = await self.ues_client.chat.get_state()
    
    return InitialStateSummary(
        email=EmailSummary(
            total_emails=email_state.total_email_count,
            total_threads=len(email_state.threads),
            unread=email_state.unread_count,
            draft_count=len(email_state.drafts),
        ),
        sms=SMSSummary(
            total_messages=sms_state.total_messages,
            total_conversations=len(sms_state.conversations),
            unread=sms_state.unread_count,
        ),
        calendar=CalendarSummary(
            event_count=len(calendar_state.events),
            calendar_count=len(calendar_state.calendars),
            events_today=self._count_events_today(calendar_state),
        ),
        chat=ChatSummary(
            total_messages=chat_state.total_message_count,
            conversation_count=chat_state.conversation_count,
        ),
    )
```

### 11.3 UES Reset and Scenario Loading

```python
async def _setup_ues(self, scenario: ScenarioConfig) -> None:
    """Reset UES and load scenario initial state.
    
    Args:
        scenario: Scenario configuration with initial_state.
    """
    import httpx
    
    # Clear UES state
    await self.ues_client.simulation.clear()
    
    # Load scenario initial state via direct HTTP (no client method)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{self._ues_port}/scenario/import/full",
            headers={"X-API-Key": self._proctor_api_key},
            json={"scenario": scenario.initial_state},
        )
        response.raise_for_status()
    
    # Start simulation
    await self.ues_client.simulation.start(auto_advance=False)
```

---

## 12. Error Handling

### 12.1 Error Categories

| Category | Handling | Recovery |
|----------|----------|----------|
| **UES Server Crash** | Detect via process poll | Restart or fail assessment |
| **Purple Timeout** | asyncio.TimeoutError | Fail turn, mark assessment failed |
| **Purple Invalid Response** | ValidationError | Fail turn, mark assessment failed |
| **LLM Failure** | Catch in ResponseGenerator/Judge | Log, skip response / award 0 points |
| **Cancellation Request** | Check `_cancelled` flag | Clean exit from turn loop |

### 12.2 Graceful Degradation

```python
async def _run_turn(...) -> TurnResult:
    try:
        # ... normal turn execution
    except asyncio.TimeoutError:
        logger.error(f"Turn {turn} timed out waiting for Purple")
        return TurnResult(
            turn_number=turn,
            actions_taken=0,
            time_step="PT0S",
            events_processed=0,
            early_completion=False,
            error="timeout",
        )
    except Exception as e:
        logger.exception(f"Error in turn {turn}: {e}")
        return TurnResult(
            turn_number=turn,
            actions_taken=0,
            time_step="PT0S",
            events_processed=0,
            early_completion=False,
            error=str(e),
        )
```

### 12.3 UES Server Health Monitoring

```python
async def _check_ues_health(self) -> bool:
    """Check if UES server is still running.
    
    Returns:
        True if healthy, False if dead or unresponsive.
    """
    # Check process is still running
    if self._ues_process.poll() is not None:
        return False
    
    # Check HTTP health
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://127.0.0.1:{self._ues_port}/health"
            )
            return response.status_code == 200
    except Exception:
        return False
```

---

## 13. Supporting Data Models

### 13.1 TurnResult (New Model)

```python
@dataclass
class TurnResult:
    """Result from executing a single turn.
    
    Returned by _run_turn() to communicate turn outcome to the main loop.
    """
    turn_number: int
    actions_taken: int
    time_step: str  # ISO 8601 duration
    events_processed: int
    early_completion: bool
    notes: str | None = None
    error: str | None = None  # Set if turn failed
```

**Location**: `src/green/assessment/models.py` (new file)

### 13.2 EndOfTurnResult (New Model)

```python
@dataclass
class EndOfTurnResult:
    """Result from end-of-turn processing.
    
    Returned by _process_turn_end() to communicate processing outcome
    back to _run_turn().
    """
    actions_taken: int
    total_events: int
    responses_generated: int
```

**Location**: `src/green/assessment/models.py`

### 13.3 Other Supporting Types

| Type | Location | Purpose |
|------|----------|---------|
| `ScenarioConfig` | `src/green/scenarios/schema.py` | ✅ Exists |
| `AssessmentResults` | `src/common/agentbeats/results.py` | ✅ Exists |
| `InitialStateSummary` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `TurnStartMessage` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `TurnCompleteMessage` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `EarlyCompletionMessage` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `AssessmentStartMessage` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `AssessmentCompleteMessage` | `src/common/agentbeats/messages.py` | ✅ Exists |
| `ScheduledResponse` | `src/green/response/models.py` | ✅ Exists |
| `AgentBeatsEvalContext` | `src/green/scenarios/schema.py` | ✅ Exists |
| `TaskUpdateEmitter` | `src/common/agentbeats/updates.py` | ✅ Exists |

---

## 14. Dependencies Summary

### 14.1 Internal Dependencies

```
GreenAgent
    ├── src/green/scenarios/schema.py
    │       ScenarioConfig, EvaluatorRegistry, AgentBeatsEvalContext
    │
    ├── src/green/core/llm_config.py
    │       LLMFactory
    │
    ├── src/green/core/action_log.py
    │       ActionLogBuilder
    │
    ├── src/green/core/message_collector.py
    │       NewMessageCollector, NewMessages
    │
    ├── src/green/response/generator.py
    │       ResponseGenerator
    │
    ├── src/green/response/models.py
    │       ScheduledResponse
    │
    ├── src/green/evaluation/judge.py
    │       CriteriaJudge
    │
    ├── src/common/agentbeats/messages.py
    │       AssessmentStartMessage, TurnStartMessage, TurnCompleteMessage,
    │       EarlyCompletionMessage, AssessmentCompleteMessage, InitialStateSummary
    │
    ├── src/common/agentbeats/results.py
    │       AssessmentResults, Scores, CriterionResult
    │
    ├── src/common/agentbeats/updates.py
    │       TaskUpdateEmitter
    │
    ├── src/common/agentbeats/config.py
    │       GreenAgentConfig
    │
    └── src/common/a2a/client.py
            A2AClientWrapper
```

### 14.2 External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `ues` | local | User Environment Simulator |
| `a2a-python` | ^0.x | A2A protocol SDK |
| `langchain-core` | ^0.x | LLM abstraction |
| `langchain-openai` | ^0.x | OpenAI LLMs |
| `langchain-anthropic` | ^0.x | Claude LLMs |
| `langchain-google-genai` | ^0.x | Gemini LLMs |
| `httpx` | ^0.x | HTTP client |
| `pydantic` | ^2.x | Data validation |

---

### 15.2 Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| UES subprocess stability | Medium | High | Health checks, restart logic |
| A2A protocol compliance | Low | High | Use SDK correctly, test with real agents |
| LLM rate limits | Medium | Medium | Retry logic, caching |
| Port allocation conflicts | Low | Medium | Track allocated ports, cleanup on shutdown |
| Long-running assessments | Low | Medium | Timeout enforcement, cancellation support |

---

## 16. Implementation Plan

### 16.1 File Structure

```
src/green/
├── __init__.py                      # Export GreenAgent
├── README.md                        # Update with agent.py docs
├── agent.py                         # GreenAgent class (NEW)
├── assessment/
│   ├── __init__.py                  # ✅ Exists
│   └── models.py                    # TurnResult, EndOfTurnResult ✅ Exists
├── core/
│   ├── ues_server.py                # UESServerManager ✅ Exists
│   └── ... (existing modules)
└── ... (existing modules)
```

### 16.2 Implementation Steps

| Step | Description | Dependencies | Est. Time |
|------|-------------|--------------|-----------|
| 1 | ~~Create `assessment/models.py` with `TurnResult`~~ | None | ✅ Done |
| 2 | ~~Implement UES server management (`UESServerManager`)~~ | UES CLI | ✅ Done |
| 3 | Implement API key management (`_create_user_api_key`, `_revoke_user_api_key`) | UES /keys endpoint | 1h |
| 4 | Implement UES setup (`_setup_ues`) | UES scenario import | 1h |
| 5 | Implement state helpers (`_capture_state_snapshot`, `_build_initial_state_summary`) | AsyncUESClient | 1h |
| 6 | Implement Purple communication (`_send_and_wait_purple`, `_extract_response_data`) | A2AClientWrapper | 1.5h |
| 7 | Implement response scheduling (`_schedule_response` and variants) | ScheduledResponse | 1.5h |
| 8 | Implement `_process_turn_end()` | Steps 4-7 | 1.5h |
| 9 | Implement `_run_turn()` | `_process_turn_end`, Purple comm | 1.5h |
| 10 | Implement `run()` main flow | `_run_turn`, CriteriaJudge | 2h |
| 11 | Implement `cancel()` | Turn loop cancellation | 0.5h |
| 12 | Add error handling throughout | All | 1h |
| 13 | Write unit tests | pytest, mocking | 4h |
| 14 | Write integration tests | Real UES instance | 3h |

**Total Estimated Time**: ~21 hours

### 16.3 Testing Strategy

**Unit Tests**:
- Mock UES client for all state operations
- Mock A2A client for Purple communication
- Mock LLMs for response generation and evaluation
- Test each private method in isolation

**Integration Tests**:
- Real UES server (started by test)
- Mock Purple agent (A2A server returning fixed responses)
- Test full assessment flow end-to-end

**Edge Cases to Test**:
- Purple timeout
- Purple early completion
- Max turns reached
- Cancellation mid-turn
- UES server crash
- Empty scenario (no actions required)
- High-volume scenario (many messages)

---

## Appendix A: Code Snippets from Existing Components

### A.1 ActionLogBuilder Usage

```python
# From src/green/core/action_log.py
builder = ActionLogBuilder(purple_agent_id="purple-123")
builder.start_turn(1)
entries = builder.add_events_from_turn(events)
builder.end_turn()
log = builder.get_log()  # List[ActionLogEntry]
```

### A.2 NewMessageCollector Usage

```python
# From src/green/core/message_collector.py
collector = NewMessageCollector(client=ues_client)
await collector.initialize(current_time)
new_messages = await collector.collect(new_time)
# new_messages.emails, new_messages.sms_messages, new_messages.calendar_events
```

### A.3 ResponseGenerator Usage

```python
# From src/green/response/generator.py
generator = ResponseGenerator(
    client=ues_client,
    scenario_config=scenario,
    response_llm=llm,
    summarization_llm=llm,
)
responses = await generator.process_new_messages(
    new_messages=new_messages,
    current_time=current_time,
)
# responses: List[ScheduledResponse]
```

### A.4 CriteriaJudge Usage

```python
# From src/green/evaluation/judge.py
judge = CriteriaJudge(
    llm=evaluation_llm,
    criteria=scenario.criteria,
    evaluators=evaluator_registry,
    emitter=emitter,
)
results = await judge.evaluate_all(eval_context)
scores = judge.aggregate_scores(results)
```

---

*Document created: February 7, 2026*
*Last updated: February 9, 2026*
