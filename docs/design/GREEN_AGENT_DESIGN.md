# GreenAgent Design

The `GreenAgent` class is the core orchestrator for AgentBeats assessments. This document describes its architecture, responsibilities, and key design decisions.

**Source**: `src/green/agent.py`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Lifecycle](#3-lifecycle)
4. [UES Server Management](#4-ues-server-management)
5. [API Key Management](#5-api-key-management)
6. [Assessment Execution](#6-assessment-execution)
7. [Turn Loop and Time Advancement](#7-turn-loop-and-time-advancement)
8. [Purple Agent Communication](#8-purple-agent-communication)
9. [Response Scheduling](#9-response-scheduling)
10. [Error Handling](#10-error-handling)
11. [Dependencies](#11-dependencies)
12. [Component Usage Examples](#12-component-usage-examples)

---

## 1. Overview

Each `GreenAgent` instance:

- Owns and manages its own UES server subprocess
- Runs assessments for a single `context_id` (Purple agent session)
- Coordinates response generation, action logging, and evaluation
- Produces assessment results as A2A artifacts

### Key Design Decisions

- **Context Isolation**: Each `context_id` gets its own `GreenAgent` with its own UES server.
- **Resource Reuse**: LLMs and the UES server persist across assessments within a context.
- **Per-Assessment Helpers**: `ResponseGenerator`, `CriteriaJudge`, `ActionLogBuilder`, and `NewMessageCollector` are created fresh for each assessment.

---

## 2. Architecture

### Position in the System

```
┌──────────────────────────────────────────────────────────────────┐
│                     Green Agent System                            │
│                                                                   │
│  ┌──────────────┐    ┌────────────────────┐                      │
│  │  A2AServer   │───▶│ GreenAgentExecutor │                      │
│  └──────────────┘    └─────────┬──────────┘                      │
│                                │                                  │
│                  ┌─────────────┴──────────────┐                  │
│                  ▼                            ▼                   │
│         ┌──────────────┐            ┌──────────────┐             │
│         │  GreenAgent  │            │  GreenAgent  │             │
│         │ (context A)  │            │ (context B)  │             │
│         │ ┌──────────┐ │            │ ┌──────────┐ │             │
│         │ │UES Server│ │            │ │UES Server│ │             │
│         │ └──────────┘ │            │ └──────────┘ │             │
│         └──────────────┘            └──────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

The `A2AServer` receives protocol messages and delegates to `GreenAgentExecutor`, which manages `GreenAgent` instances — one per context. Each agent owns its own UES server process.

### Resource Lifetimes

**Long-lived** (persist across assessments within a context): UES server process, proctor API key, UES client, response LLM, evaluation LLM.

**Short-lived** (created fresh per `run()` call): `ActionLogBuilder`, `NewMessageCollector`, `ResponseGenerator`, `CriteriaJudge`.

### Core Responsibilities

| Responsibility | Component |
|----------------|-----------|
| UES lifecycle management | `UESServerManager` |
| Scenario setup | `AsyncUESClient` |
| API key generation/revocation | UES `/keys` endpoint |
| User-prompt injection into chat | `AsyncUESClient` chat API |
| Turn orchestration | A2A messaging + time control |
| Action tracking | `ActionLogBuilder` |
| Character response generation | `ResponseGenerator` |
| Response injection into UES | Modality `receive` APIs |
| Performance evaluation | `CriteriaJudge` |
| Result assembly | `AssessmentResults` model |

---

## 3. Lifecycle

### Agent Lifecycle

```
  __init__()          startup()           run() × N            shutdown()
 ┌───────────┐       ┌───────────┐      ┌───────────┐        ┌───────────┐
 │ Store port│       │ Start UES │      │ Run       │        │ Close     │
 │ Store conf│─────▶│ Create    │─────▶│ assessment│──···──▶│ client    │
 │ Create    │       │ client    │      │ tasks     │        │ Terminate │
 │ LLMs      │       │ Wait ready│      │           │        │ UES proc  │
 └───────────┘       └───────────┘      └───────────┘        └───────────┘
```

### Assessment Lifecycle (within `run()`)

Each `run()` invocation proceeds through six phases:

1. **Setup** — Reset UES, load scenario state, create user API key for Purple, initialize per-assessment helpers, capture initial state snapshot.
2. **Prompt Injection** — Inject the scenario's `user_prompt` into UES chat via `_inject_user_prompt_chat()`. This places the task description in the `"user-assistant"` conversation so Purple discovers it by reading chat state.
3. **Start** — Send `AssessmentStartMessage` to Purple. The message carries a fixed `assessment_instructions` string (the `DEFAULT_ASSESSMENT_INSTRUCTIONS` constant) that tells Purple to act as a personal assistant and read tasks from chat. The scenario-specific task is *not* included in this A2A message — it lives in UES chat.
4. **Turn Loop** — Iterate turns until `max_turns`, cancellation, or early completion. Each turn sends `TurnStartMessage` to Purple, waits for a response, then processes end-of-turn events and responses.
5. **Evaluation** — Capture final state, build evaluation context, run `CriteriaJudge`, aggregate scores.
6. **Completion** — Send `AssessmentCompleteMessage` to Purple, revoke user API key, build and return `AssessmentResults`.

---

## 4. UES Server Management

The `GreenAgent` starts a UES server as a subprocess via `UESServerManager`. A random 64-character hex admin key is generated at `__init__` time and passed to the UES process via the `UES_ADMIN_KEY` environment variable, so the admin key is immediately known without parsing stdout.

UES server stdout/stderr is captured by a background drain task to prevent the subprocess from blocking on a full pipe buffer.

After starting the process, `startup()` polls the `/health` endpoint until the server is accepting connections (with a configurable timeout).

**Server configuration**:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Host | `127.0.0.1` | Local only; Purple accesses via Green-provided URL |
| Port | Dynamically assigned | Each `GreenAgent` gets a unique port |
| `auto_advance` | `False` | Green controls time via explicit advance calls |

---

## 5. API Key Management

### Key Hierarchy

Two tiers of API keys control access to the UES server:

- **Admin key** (proctor-level): Pre-generated by `GreenAgent`, passed to the UES subprocess via environment variable. Has wildcard (`"*"`) permissions — full access to all endpoints including time control, simulation management, event injection, and key management.

- **User key** (purple-level): Created per-assessment and given to the Purple agent. Has limited permissions covering only user-side actions: reading state, sending emails/SMS, managing calendar events, etc. Explicitly excluded from simulator-side injection (`email:receive`, `sms:receive`), time control, simulation control, scenario management, event management, and key management.

### Dual-Purpose `key_id`

The `key_id` returned by UES when creating a user key serves two purposes:

1. **Key revocation**: Used to revoke the key when the assessment ends.
2. **Event attribution**: Becomes the `agent_id` on all UES events created by the Purple agent, allowing Green to filter event history to only Purple's actions.

---

## 6. Assessment Execution

The `run()` method is the main entry point. It creates per-assessment helpers (scoped to the assessment's user key and scenario), injects the scenario's `user_prompt` into UES chat, sends the assessment start message, runs the turn loop, evaluates results, and cleans up.

### Prompt Delivery Contract

Task delivery to Purple is split across two channels:

- **UES chat** (scenario-specific): Green calls `_inject_user_prompt_chat()` to post the scenario's `user_prompt` as a user message in the `"user-assistant"` conversation. This is how Purple learns *what* to do.
- **A2A message** (fixed): The `AssessmentStartMessage` carries a constant `assessment_instructions` (`DEFAULT_ASSESSMENT_INSTRUCTIONS`) that tells Purple to act as a personal assistant and read tasks from chat. This never varies between scenarios.

This separation keeps the A2A protocol stable — Purple always receives the same top-level instruction — while scenarios control the specific task via chat content.

The turn loop runs until one of three exit conditions is met:

- **Max turns reached**: Configured via `assessment_config` or `GreenAgentConfig.default_max_turns`.
- **Early completion**: Purple sends an `EarlyCompletionMessage` instead of `TurnCompleteMessage`.
- **Cancellation**: The `cancel()` method sets a flag checked between turns.

After the loop, `CriteriaJudge` evaluates the Purple agent's performance against scenario criteria using the action log and initial/final state snapshots.

---

## 7. Turn Loop and Time Advancement

### Single Turn Structure

Each turn follows three steps:

1. **Send `TurnStartMessage` to Purple** with the current simulation time.
2. **Wait for Purple's response** — either `TurnCompleteMessage` (with an optional `time_step`) or `EarlyCompletionMessage`.
3. **End-of-turn processing** via `_process_turn_end()`.

### End-of-Turn Processing

The `_process_turn_end()` helper encapsulates all processing between Purple's turn completion and the next turn. It is separated from `_run_turn()` to allow the processing strategy to change independently. The current batch strategy:

1. Apply advance (1s) — make Purple's events visible
2. Collect all events and new messages
3. Generate all character responses
4. Schedule all responses into UES
5. Remainder advance — fire character responses

### Two-Phase Time Advancement

Time is advanced in two phases per turn to solve a critical ordering problem:

```
  Purple's turn       Apply (1s)      Green processes        Remainder
  ────────────────┬──────────────┬──────────────────────┬──────────────────
  Purple sends    │ Events fire, │ Green reads state,   │ Character responses
  emails, creates │ emails appear│ generates replies,   │ fire; Purple sees
  events via API  │ in UES state │ schedules responses  │ them next turn
  ────────────────┴──────────────┴──────────────────────┴──────────────────
```

**Phase 1 — Apply advance (1s)**: Purple's actions are submitted as UES events that need a time advance to execute. The 1-second advance ensures these events fire and their effects become visible in modality state. Without this, Green would see no messages from Purple's actions.

**Phase 2 — Remainder advance (time_step - 1s)**: After generating and scheduling character responses as future UES events, the remainder advance fires those responses. Purple will see character replies when it queries UES state on its next turn, creating a natural conversation flow.

Response `scheduled_time` values must fall between the apply time (T+1s) and the end of the remainder advance (T+time_step) to ensure they fire during the remainder and are visible to Purple.

---

## 8. Purple Agent Communication

All communication with Purple agents uses the A2A protocol. Messages are Pydantic models serialized to JSON and wrapped in A2A `DataPart` messages.

### Message Exchange Pattern

```
    GreenAgent                              PurpleAgent
        │                                        │
        │   [inject user_prompt into UES chat]    │
        │                                        │
        │   AssessmentStartMessage               │
        │   (fixed assessment_instructions)       │
        │───────────────────────────────────────▶│
        │                                        │
        │   TurnStartMessage                     │
        │───────────────────────────────────────▶│
        │                                        │
        │                    TurnCompleteMessage  │
        │◀───────────────────────────────────────│
        │      or EarlyCompletionMessage         │
        │                                        │
        │   [Green advances time, generates      │
        │    responses, schedules them]           │
        │                                        │
        │   TurnStartMessage                     │
        │───────────────────────────────────────▶│
        │                                        │
        │   ... repeat ...                       │
        │                                        │
        │   AssessmentCompleteMessage             │
        │───────────────────────────────────────▶│
        │                                        │
```

Green sends messages via `A2AClientWrapper.send_message()` with `blocking=True` and wraps the call in `asyncio.wait_for()` for timeout enforcement. The response is parsed based on its `message_type` field.

Note that `AssessmentStartMessage.assessment_instructions` is a fixed constant (`DEFAULT_ASSESSMENT_INSTRUCTIONS`) enforced by a Pydantic validator — it cannot be customized per scenario. The scenario-specific task reaches Purple exclusively through UES chat, injected by `_inject_user_prompt_chat()` before the assessment start message is sent.

---

## 9. Response Scheduling

Character responses generated by `ResponseGenerator` are injected into UES using proctor-level modality APIs, dispatched by modality:

- **Email**: `ues_client.email.receive()` — injects an incoming email with sender, recipients, subject, body, and thread metadata.
- **SMS**: `ues_client.sms.receive()` — injects an incoming SMS with sender, recipients, and body.
- **Calendar**: `ues_client.calendar.respond_to_event()` — submits an RSVP response from a character.

These use proctor-level permissions (the `receive` and `respond_to_event` endpoints are forbidden to Purple's user key) to simulate external characters replying to Purple's actions.

---

## 10. Error Handling

| Category | Detection | Recovery |
|----------|-----------|----------|
| UES server crash | Process poll + `/health` check | Fail assessment |
| Purple timeout | `asyncio.TimeoutError` | Return error `TurnResult` |
| Purple invalid response | `ValidationError` | Return error `TurnResult` |
| LLM failure | Exception in `ResponseGenerator`/`CriteriaJudge` | Log error, skip response or award 0 points |
| Cancellation | `_cancelled` flag checked between turns | Clean exit from turn loop |

UES server health is monitored via process polling and HTTP health checks. Turns that fail (timeout, bad response) produce a `TurnResult` with an `error` field, allowing the main loop to decide whether to continue or abort.

---

## 11. Dependencies

### Internal

| Module | Provides |
|--------|----------|
| `src/green/scenarios/schema` | `ScenarioConfig`, `EvaluatorRegistry`, `AgentBeatsEvalContext` |
| `src/green/core/llm_config` | `LLMFactory` |
| `src/green/core/action_log` | `ActionLogBuilder` |
| `src/green/core/message_collector` | `NewMessageCollector` |
| `src/green/core/ues_server` | `UESServerManager` |
| `src/green/response/generator` | `ResponseGenerator` |
| `src/green/response/models` | `ScheduledResponse` |
| `src/green/evaluation/judge` | `CriteriaJudge` |
| `src/green/assessment/models` | `TurnResult`, `EndOfTurnResult` |
| `src/common/agentbeats/messages` | A2A message types (`AssessmentStartMessage`, `TurnStartMessage`, etc.) |
| `src/common/agentbeats/results` | `AssessmentResults`, `Scores`, `CriterionResult` |
| `src/common/agentbeats/updates` | `TaskUpdateEmitter` |
| `src/common/agentbeats/config` | `GreenAgentConfig` |
| `src/common/a2a/client` | `A2AClientWrapper` |

### External

| Dependency | Purpose |
|------------|---------|
| `ues` (local editable) | User Environment Simulator |
| `a2a-python` | A2A protocol SDK |
| `langchain-core` | LLM abstraction |
| `langchain-openai` / `langchain-anthropic` / `langchain-google-genai` | LLM providers |
| `pydantic` | Data validation |

---

## 12. Component Usage Examples

### ActionLogBuilder

```python
builder = ActionLogBuilder(purple_agent_id="purple-123")
builder.start_turn(1)
purple_entries, green_events = builder.add_events_from_turn(events)
builder.end_turn()
log = builder.get_log()  # list[ActionLogEntry]
```

### NewMessageCollector

```python
collector = NewMessageCollector(client=ues_client)
await collector.initialize(current_time)
new_messages = await collector.collect(new_time)
# new_messages.emails, new_messages.sms_messages, new_messages.calendar_events
```

### ResponseGenerator

```python
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
# responses: list[ScheduledResponse]
```

### CriteriaJudge

```python
judge = CriteriaJudge(
    llm=evaluation_llm,
    criteria=scenario.criteria,
    evaluators=evaluator_registry,
    emitter=emitter,
)
results = await judge.evaluate_all(eval_context)
scores = judge.aggregate_scores(results)
```

### GreenAgent

```python
config = GreenAgentConfig()
agent = GreenAgent(ues_port=8100, config=config)
await agent.startup()

results = await agent.run(
    task_id="task-1",
    emitter=emitter,
    scenario=scenario,
    evaluators=evaluators,
    purple_client=purple_client,
    assessment_config={"max_turns": 10},
)

await agent.shutdown()
```

---

*Document created: February 7, 2026*
*Last updated: February 12, 2026*
