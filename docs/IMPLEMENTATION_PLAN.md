# UES AgentBeats Implementation Plan

This document outlines the high-level implementation plan for building Green and Purple agents for the AgentBeats competition using the User Environment Simulator (UES).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1: Common A2A Helper Code](#phase-1-common-a2a-helper-code)
3. [Phase 2: Common AgentBeats Helper Code](#phase-2-common-agentbeats-helper-code)
4. [Phase 3: Green Agent Implementation](#phase-3-green-agent-implementation)
5. [Phase 4: Purple Agent Template](#phase-4-purple-agent-template)
6. [Phase 5: Submission Requirements](#phase-5-submission-requirements)
7. [Project Structure](#project-structure)
8. [Development Timeline](#development-timeline)

---

## Project Overview

### Goals
- Build a **Green Agent** (evaluator) that uses UES as the testing environment for AI personal assistants
- Provide a **Purple Agent template** that makes it easy for participants to build compliant agents
- Implement the **turn-based assessment flow** defined in `ASSESSMENT_FLOW.md`
- Ensure full **A2A protocol compliance** for interoperability

### Key Dependencies
- `a2a-python` - Official A2A Python SDK
- `langchain` / `langchain-core` - LLM agent framework
- `ues` - User Environment Simulator (local dependency)
- `uvicorn` / `starlette` - ASGI server framework
- `pydantic` - Data validation and settings

---

## Phase 1: Common A2A Helper Code ✅ COMPLETE

**Location**: `src/common/a2a/`

**Status**: Implemented and tested (107 tests passing). See [src/common/a2a/README.md](../src/common/a2a/README.md) for detailed documentation.

This module provides reusable A2A protocol utilities for both Green and Purple agents.

### 1.1 Agent Card Builder (`agent_card.py`)

Simplify the creation of A2A-compliant Agent Cards.

```python
# Key components:
class AgentCardBuilder:
    """Fluent builder for creating AgentCard objects."""
    
    def with_name(self, name: str) -> Self: ...
    def with_description(self, description: str) -> Self: ...
    def with_url(self, url: str) -> Self: ...
    def with_version(self, version: str) -> Self: ...
    def with_skill(self, skill: AgentSkill) -> Self: ...
    def with_capabilities(self, **kwargs) -> Self: ...
    def build(self) -> AgentCard: ...

def create_skill(
    id: str,
    name: str,
    description: str,
    tags: list[str],
    examples: list[str] | None = None,
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
) -> AgentSkill: ...
```

**Tests**: Unit tests for builder patterns, validation of required fields.

### 1.2 Server Utilities (`server.py`)

Wrapper around A2A SDK server components with sensible defaults.

```python
class A2AServer:
    """Simplified A2A server setup."""
    
    def __init__(
        self,
        agent_card: AgentCard,
        executor: AgentExecutor,
        host: str = "0.0.0.0",
        port: int = 8000,
    ): ...
    
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def run(self) -> None:  # Blocking run
        ...
```

**Tests**: Integration tests for server startup/shutdown, agent card endpoint.

### 1.3 Client Utilities (`client.py`)

Helpers for making A2A calls to other agents.

```python
class A2AClientWrapper:
    """Client for communicating with A2A agents."""
    
    def __init__(self, agent_url: str): ...
    
    async def get_agent_card(self) -> AgentCard: ...
    
    async def send_message(
        self,
        message: Message,
        task_id: str | None = None,
        context_id: str | None = None,
        blocking: bool = False,
    ) -> Task | Message: ...
    
    async def send_streaming_message(
        self,
        message: Message,
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator[StreamResponse]: ...
    
    async def get_task(self, task_id: str) -> Task: ...
    async def cancel_task(self, task_id: str) -> Task: ...
```

**Tests**: Mock server tests, streaming tests, error handling.

### 1.4 Message Helpers (`messages.py`)

Utilities for creating and parsing A2A messages.

```python
# Message creation helpers
def create_text_message(
    text: str,
    role: Role = Role.USER,
    message_id: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
    metadata: dict | None = None,
) -> Message: ...

def create_data_message(
    data: dict,
    role: Role = Role.USER,
    message_id: str | None = None,
) -> Message: ...

# Part creation helpers
def text_part(text: str) -> Part: ...
def data_part(data: dict) -> Part: ...
def file_part(content: bytes, filename: str, media_type: str) -> Part: ...

# Message parsing helpers
def get_text_content(message: Message) -> str: ...
def get_data_content(message: Message) -> dict | None: ...
def get_all_text_parts(message: Message) -> list[str]: ...
```

**Tests**: Round-trip serialization, edge cases for multi-part messages.

### 1.5 Task Management (`tasks.py`)

Helpers for task state management and updates.

```python
# Task state checks
def is_terminal_state(task: Task) -> bool: ...
def is_completed(task: Task) -> bool: ...
def is_failed(task: Task) -> bool: ...
def is_input_required(task: Task) -> bool: ...

# Task update event helpers
def create_status_update(
    task_id: str,
    context_id: str,
    state: TaskState,
    message: Message | None = None,
) -> TaskStatusUpdateEvent: ...

def create_artifact_update(
    task_id: str,
    context_id: str,
    artifact: Artifact,
    append: bool = False,
    last_chunk: bool = True,
) -> TaskArtifactUpdateEvent: ...
```

**Tests**: State transition validation, event creation.

### 1.6 Artifact Helpers (`artifacts.py`)

Utilities for creating and working with artifacts.

```python
def create_artifact(
    artifact_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    parts: list[Part] | None = None,
    text: str | None = None,  # Convenience: creates TextPart
    data: dict | None = None,  # Convenience: creates DataPart
) -> Artifact: ...

def create_json_artifact(
    data: dict,
    name: str = "result",
    description: str | None = None,
) -> Artifact: ...

def extract_json_from_artifact(artifact: Artifact) -> dict | None: ...
```

**Tests**: JSON serialization, multi-part artifacts.

---

## Phase 2: Common AgentBeats Helper Code ✅ COMPLETE

**Location**: `src/common/agentbeats/`

**Status**: Implemented and tested (237 tests passing). See [src/common/agentbeats/README.md](../src/common/agentbeats/README.md) for detailed documentation.

AgentBeats-specific patterns and message types used by both Green and Purple agents.

### 2.1 Assessment Messages (`messages.py`) ✅ COMPLETE

Pydantic models for AgentBeats-specific message payloads.

> **Design Note**: All message models include a `message_type` field with a fixed
> literal string value. This allows agents to easily parse incoming messages by
> checking the `message_type` field rather than inferring the type from which
> fields are present. The `message_type` is set as a class-level default and
> cannot be overridden.

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Literal

# NOTE: Modality summary fields are derived from UES `get_snapshot()` outputs.
# Different modalities have different available fields:
#   - Email: total_emails, unread (computed from folders[*].unread_count)
#   - Calendar: event_count, events_today (requires get_compact_snapshot)
#   - SMS: total_messages, unread_total
#   - Chat: total_message_count (no unread concept)

class EmailSummary(BaseModel):
    """Summary counts for email modality."""
    message_type: Literal["email_summary"] = "email_summary"
    total_emails: int
    total_threads: int
    unread: int
    draft_count: int

class CalendarSummary(BaseModel):
    """Summary counts for calendar modality."""
    message_type: Literal["calendar_summary"] = "calendar_summary"
    event_count: int
    calendar_count: int
    events_today: int  # Requires get_compact_snapshot(current_time)

class SMSSummary(BaseModel):
    """Summary counts for SMS modality."""
    message_type: Literal["sms_summary"] = "sms_summary"
    total_messages: int
    total_conversations: int
    unread: int

class ChatSummary(BaseModel):
    """Summary counts for chat modality."""
    message_type: Literal["chat_summary"] = "chat_summary"
    total_messages: int
    conversation_count: int
    # Note: Chat has no "unread" concept (user-assistant pairs)

class InitialStateSummary(BaseModel):
    """Summary of initial UES state, derived from modality snapshots."""
    message_type: Literal["initial_state_summary"] = "initial_state_summary"
    email: EmailSummary
    calendar: CalendarSummary
    sms: SMSSummary
    chat: ChatSummary

class AssessmentStartMessage(BaseModel):
    """Message sent from Green to Purple at assessment start."""
    message_type: Literal["assessment_start"] = "assessment_start"
    ues_url: str  # Generated by GreenAgent (e.g., "http://localhost:8100")
    api_key: str  # User-level token generated per-assessment
    assessment_instructions: str
    current_time: datetime
    initial_state_summary: InitialStateSummary

class TurnStartMessage(BaseModel):
    """Message sent from Green to Purple at turn start."""
    message_type: Literal["turn_start"] = "turn_start"
    turn_number: int
    current_time: datetime
    events_processed: int

class ActionLogEntry(BaseModel):
    """Single action taken by Purple agent during a turn.
    
    Built from UES event history, filtered by Purple agent's agent_id.
    Green agent adds turn number when aggregating into assessment results.
    """
    message_type: Literal["action_log_entry"] = "action_log_entry"
    turn: int  # Turn number when action occurred
    timestamp: datetime
    action: str  # e.g., "email.send", "calendar.create", "sms.reply"
    parameters: dict  # Action-specific parameters
    success: bool
    error_message: str | None = None  # Populated if success=False

class TurnCompleteMessage(BaseModel):
    """Message sent from Purple to Green when turn completes.
    
    Purple agent signals turn completion. The Green agent builds the
    action log by querying UES event history, filtering for events
    attributed to the Purple agent.
    """
    message_type: Literal["turn_complete"] = "turn_complete"
    notes: str | None = None  # Optional reasoning/transparency
    time_step: str = "PT1H"  # ISO 8601 duration

class AssessmentCompleteMessage(BaseModel):
    """Message sent from Green to Purple when assessment ends."""
    message_type: Literal["assessment_complete"] = "assessment_complete"
    reason: Literal[
        "scenario_complete",
        "early_completion",
        "timeout",
        "error",
    ]

class EarlyCompletionMessage(BaseModel):
    """Message sent from Purple to Green to signal early completion."""
    message_type: Literal["early_completion"] = "early_completion"
    reason: str | None = None
```

**Tests**: Serialization/deserialization, validation, message_type field immutability.

### 2.2 Assessment Results (`results.py`) ✅ COMPLETE

Pydantic models for assessment results (artifacts).

> **Design Note**: Like message models, all result models include a `message_type`
> field with a fixed literal value for consistent parsing. Results also include
> validation logic to ensure score consistency (e.g., overall score matches sum
> of dimension scores).

```python
class CriterionResult(BaseModel):
    """Result for a single evaluation criterion.
    
    The `details` field carries structured information about individual
    evaluations, e.g., politeness scores for each email in an "email
    politeness" criterion.
    """
    message_type: Literal["criterion_result"] = "criterion_result"
    criterion_id: str
    name: str
    dimension: Literal[
        "accuracy",
        "instruction_following", 
        "efficiency",
        "safety",
        "politeness",
    ]
    score: int
    max_score: int
    explanation: str
    details: dict | None = None  # Structured breakdown (e.g., per-item scores)

class DimensionScore(BaseModel):
    """Aggregated score for a dimension."""
    message_type: Literal["dimension_score"] = "dimension_score"
    score: int
    max_score: int

class OverallScore(BaseModel):
    """Overall assessment score."""
    message_type: Literal["overall_score"] = "overall_score"
    score: int
    max_score: int

class Scores(BaseModel):
    """All scores for an assessment."""
    message_type: Literal["scores"] = "scores"
    overall: OverallScore
    dimensions: dict[str, DimensionScore]

class ActionLogEntryWithTurn(BaseModel):
    """ActionLogEntry with turn number added by Green agent.
    
    This extends the ActionLogEntry from messages.py with the turn context
    that the Green agent adds when building the assessment results.
    """
    message_type: Literal["action_log_entry_with_turn"] = "action_log_entry_with_turn"
    turn: int
    timestamp: datetime
    action: str
    parameters: dict
    success: bool
    error_message: str | None = None

class AssessmentResults(BaseModel):
    """Complete assessment results."""
    message_type: Literal["assessment_results"] = "assessment_results"
    assessment_id: str
    scenario_id: str
    participant: str
    status: Literal["completed", "failed", "timeout"]
    duration_seconds: float
    turns_taken: int
    actions_taken: int
    scores: Scores
    criteria_results: list[CriterionResult]
    action_log: list[ActionLogEntryWithTurn]
```

**Tests**: Score calculation consistency, JSON schema validation, score validation.

### 2.3 Task Update Helpers (`updates.py`) ✅ COMPLETE

Pydantic models for structured task updates and a `TaskUpdateEmitter` helper class.
Task updates are emitted via A2A `TaskStatusUpdateEvent` messages for logging,
debugging, and observability.

**Update Types - Green Agent:**
- `AssessmentStartedUpdate`: Assessment has begun
- `ScenarioLoadedUpdate`: Scenario configuration loaded
- `TurnStartedUpdate`: New turn has begun
- `TurnCompletedUpdate`: Turn has completed
- `ResponsesGeneratedUpdate`: Character responses generated
- `SimulationAdvancedUpdate`: UES simulation time advanced
- `EvaluationStartedUpdate`: Evaluation phase has begun
- `CriterionEvaluatedUpdate`: Single criterion evaluated
- `AssessmentCompletedUpdate`: Assessment has ended
- `ErrorOccurredUpdate`: An error occurred

- `ActionObservedUpdate`: Logs a Purple agent action (emitted by Green when
  processing UES events)

**Note**: All task updates are emitted by the Green agent. Purple agents do not
emit updates directly - actions are captured from UES event history, and the
Green agent logs each Purple action as an `ActionObservedUpdate`.

```python
class TaskUpdateType(str, Enum):
    """Types of task updates for logging (all emitted by Green agent)."""
    ASSESSMENT_STARTED = "update_assessment_started"
    SCENARIO_LOADED = "update_scenario_loaded"
    TURN_STARTED = "update_turn_started"
    TURN_COMPLETED = "update_turn_completed"
    RESPONSES_GENERATED = "update_responses_generated"
    SIMULATION_ADVANCED = "update_simulation_advanced"
    EVALUATION_STARTED = "update_evaluation_started"
    CRITERION_EVALUATED = "update_criterion_evaluated"
    ASSESSMENT_COMPLETED = "update_assessment_completed"
    ERROR_OCCURRED = "update_error_occurred"
    ACTION_OBSERVED = "update_action_observed"

class ActionObservedUpdate(BaseModel):
    """Update emitted by Green agent when logging a Purple agent action.
    
    The Green agent emits this for each action found in UES event history
    that is attributed to the Purple agent. Captures all details for audit/debugging.
    """
    message_type: Literal["update_action_observed"] = "update_action_observed"
    turn_number: int
    timestamp: datetime
    action: str
    parameters: dict[str, Any]
    success: bool
    error_message: str | None = None
    notes: str | None = None

class TaskUpdateEmitter:
    """Helper for emitting structured task updates as A2A events.
    
    Used by the Green agent to emit all task updates during assessment.
    """
    
    def __init__(self, task_id: str, context_id: str): ...
    
    def assessment_started(
        self,
        assessment_id: str,
        scenario_id: str,
        participant_url: str,
        start_time: datetime,
    ) -> TaskStatusUpdateEvent: ...
    
    def turn_completed(
        self,
        turn_number: int,
        actions_taken: int,
        time_advanced: str,
        early_completion_requested: bool = False,
    ) -> TaskStatusUpdateEvent: ...
    
    def assessment_completed(
        self,
        reason: Literal["scenario_complete", "early_completion", "timeout", "error"],
        total_turns: int,
        total_actions: int,
        duration_seconds: float,
        overall_score: int,
        max_score: int,
    ) -> TaskStatusUpdateEvent: ...
    
    # Called when processing UES events from Purple agent
    def action_observed(
        self,
        turn_number: int,
        timestamp: datetime,
        action: str,
        parameters: dict[str, Any],
        success: bool,
        error_message: str | None = None,
        notes: str | None = None,
    ) -> TaskStatusUpdateEvent: ...
    
    # ... more convenience methods

def parse_update(data: dict[str, Any]) -> AgentBeatsUpdate: ...
```

**Tests**: 60 tests covering all update models, serialization, validation, parsing,
and emitter methods.

### 2.4 Configuration (`config.py`) ✅ COMPLETE

**Status**: Implemented and tested (83 tests passing). Provides comprehensive
configuration management for Green and Purple agents with CLI argument parsing
and environment variable support.

**Key Features**:
- Base `AgentBeatsConfig` with common settings (host, port, card_url, log_level)
- `GreenAgentConfig` with evaluator-specific settings (UES URL, scenarios, LLM models)
- `PurpleAgentConfig` with participant-specific settings (model, temperature, actions)
- `from_cli_args()` class method for CLI argument parsing
- Environment variable loading via pydantic-settings (prefixed with `AGENTBEATS_`)
- `merge_configs()` utility for non-destructive config updates
- `validate_config()` for detecting potential issues

**Environment Variable Prefixes**:
- Base config: `AGENTBEATS_*`
- Green config: `AGENTBEATS_GREEN_*`
- Purple config: `AGENTBEATS_PURPLE_*`

Shared configuration patterns.

```python
from pydantic_settings import BaseSettings

class AgentBeatsConfig(BaseSettings):
    """Base configuration for AgentBeats agents."""
    host: str = "0.0.0.0"
    port: int = 8000
    card_url: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    @classmethod
    def from_cli_args(cls, args: Sequence[str] | None = None) -> Self:
        """Parse from command line arguments."""
        ...
    
    @property
    def effective_card_url(self) -> str:
        """Get card URL with sensible default."""
        ...

class GreenAgentConfig(AgentBeatsConfig):
    """Configuration specific to Green agents."""
    port: int = 8000  # Default for Green
    verbose_updates: bool = True
    ues_base_port: int = 8100  # Base port for UES servers (each GreenAgent gets unique port)
    scenarios_dir: str = "scenarios"
    default_max_turns: int = 100
    default_turn_timeout: float = 300.0
    response_generator_model: str = "gpt-4o"
    evaluation_model: str = "gpt-4o"

class PurpleAgentConfig(AgentBeatsConfig):
    """Configuration specific to Purple agents."""
    port: int = 8001  # Default for Purple
    model: str = "gpt-4o"
    max_actions_per_turn: int = 50
    temperature: float = 0.7
    enable_reflection: bool = True
    action_delay: float = 0.0

# Utility functions
def merge_configs(base: AgentBeatsConfig, overrides: dict) -> AgentBeatsConfig: ...
def validate_config(config: AgentBeatsConfig) -> list[str]: ...  # Returns warnings
```

**Tests**: 83 tests covering CLI argument parsing, environment variable loading,
field validation, configuration merging, and validation warnings.

---

## Phase 3: Green Agent Implementation

**Location**: `src/green/`

The Green Agent orchestrates assessments, manages UES, and evaluates Purple agents.

### 3.1 Architecture Overview

The Green agent system consists of three main layers:

1. **Server Layer** (`server.py`): A2A server setup, agent card, request routing
2. **Executor Layer** (`executor.py`): Request validation, task/context management, `GreenAgent` lifecycle
3. **Agent Layer** (`agent.py`): Assessment orchestration, UES management, turn loop, evaluation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Green Agent System                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  A2A Server (server.py)                                            │ │
│  │  - Agent card configuration                                        │ │
│  │  - HTTP request handling via DefaultRequestHandler                 │ │
│  │  - Routes requests to GreenAgentExecutor                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  GreenAgentExecutor (executor.py)                                  │ │
│  │  - Implements AgentExecutor interface (execute, cancel)            │ │
│  │  - Validates assessment requests (EvalRequest)                     │ │
│  │  - Manages GreenAgent instances (keyed by context_id)              │ │
│  │  - Allocates UES ports for new GreenAgent instances                │ │
│  │  - Creates TaskUpdater for A2A event emission                      │ │
│  │  - Loads scenarios and evaluators via ScenarioManager              │ │
│  │  - Creates A2A client for Purple agent communication               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│         ┌──────────────────────────┼──────────────────────────┐         │
│         ▼                          ▼                          ▼         │
│  ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│  │  GreenAgent     │    │  GreenAgent         │    │  GreenAgent     │  │
│  │  (context A)    │    │  (context B)        │    │  (context C)    │  │
│  │                 │    │                     │    │                 │  │
│  │  - Own UES      │    │  - Own UES          │    │  - Own UES      │  │
│  │    server       │    │    server           │    │    server       │  │
│  │  - Own LLMs     │    │  - Own LLMs         │    │  - Own LLMs     │  │
│  │  - Can run      │    │  - Can run          │    │  - Can run      │  │
│  │    multiple     │    │    multiple         │    │    multiple     │  │
│  │    assessments  │    │    assessments      │    │    assessments  │  │
│  └─────────────────┘    └─────────────────────┘    └─────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions:**

1. **Task/Context Semantics**: In AgentBeats, `context_id` represents the long-running
   conversation between a Green and Purple agent pair. Within that conversation, there
   may be multiple assessment `task_id`s (one per assessment request). A single
   `GreenAgent` instance handles all assessments for its `context_id`.

2. **UES Isolation**: Each `GreenAgent` owns its own UES server instance (on a unique
   port). This ensures Purple agents being assessed in parallel don't interfere with
   each other's simulation state.

3. **Assessment Independence**: A `GreenAgent` can run multiple sequential assessments
   with different scenarios. Scenario-specific objects (`ResponseGenerator`,
   `CriteriaJudge`, `ActionLogBuilder`) are created fresh for each `run()` call, while
   stable resources (UES server, LLM instances) persist across assessments.

4. **Lifecycle Management**: The `GreenAgentExecutor` is responsible for creating and
   cleaning up `GreenAgent` instances. When a context is no longer needed, its
   `GreenAgent` is shutdown (including its UES server).

### 3.2 Scenario Management (`scenarios/`) ✅ COMPLETE

**Status**: Implemented and tested (141 tests passing). See [src/green/scenarios/README.md](../src/green/scenarios/README.md) for detailed documentation.

This module provides schema definitions and loading utilities for assessment scenarios.

#### 3.2.1 Scenario Schema (`schema.py`) ✅ COMPLETE

Pydantic models for scenario configuration with comprehensive validation:

- **`ResponseTiming`**: Timing configuration for character responses (ISO 8601 durations)
- **`CharacterProfile`**: Profile for simulated characters with contact methods and personality
- **`EvaluationCriterion`**: Evaluation criterion with programmatic or LLM-based evaluation
- **`ScenarioConfig`**: Complete scenario configuration with validation

**Key Features**:
- ISO 8601 duration parsing with `parse_iso8601_duration()` utility
- Timezone-aware datetime validation
- Cross-field validation (unique emails/phones, unique criterion IDs, time ordering)
- Frozen models for immutability
- Helper methods (`get_character_by_email`, `get_criteria_by_dimension`, etc.)

**Evaluator Types** (also defined in `schema.py`):

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Awaitable
from ues.client import AsyncUESClient
from src.common.agentbeats.results import ActionLogEntryWithTurn

@dataclass
class EvalResult:
    """Result returned by an evaluator function.
    
    Follows UES agent_testing pattern. Evaluators return this; CriteriaJudge
    converts to CriterionResult (adding criterion_id, name, dimension).
    """
    score: float
    max_score: float
    explanation: str
    details: dict[str, Any] | None = None

@dataclass
class AgentBeatsEvalContext:
    """Context passed to evaluator functions.
    
    Extends UES EvalContext pattern with AgentBeats-specific fields:
    action_log, initial_state, final_state, user_prompt.
    """
    client: AsyncUESClient
    scenario_config: dict[str, Any]
    action_log: list[ActionLogEntryWithTurn]
    initial_state: dict[str, Any]  # Modality snapshots before assessment
    final_state: dict[str, Any]    # Modality snapshots after assessment
    user_prompt: str
    
    async def get_state(self, modality: str) -> Any:
        """Get current modality state from UES."""
        modality_clients = {
            "email": self.client.email,
            "sms": self.client.sms,
            "calendar": self.client.calendar,
            "chat": self.client.chat,
        }
        if modality not in modality_clients:
            raise ValueError(f"Unknown modality: {modality}")
        return await modality_clients[modality].get_state()
    
    async def get_time(self) -> datetime:
        """Get current simulation time."""
        time_state = await self.client.time.get_state()
        return time_state.current_time

# Type aliases
EvaluatorFunc = Callable[[AgentBeatsEvalContext, dict[str, Any]], Awaitable[EvalResult]]
EvaluatorRegistry = dict[str, EvaluatorFunc]
```

#### 3.2.2 Scenario Loader (`loader.py`) ✅ COMPLETE

Loading and management utilities for scenarios:

- **`ScenarioLoader`**: Low-level loader for individual scenario files
- **`ScenarioManager`**: High-level interface for discovering and loading scenarios
- **`ScenarioNotFoundError`**: Exception for missing scenarios
- **`ScenarioValidationError`**: Exception for validation failures
- **`EvaluatorLoadError`**: Exception for evaluator module loading failures

**Key Features**:
- Load scenarios from JSON files
- Support for embedded or external initial_state files
- Scenario caching with clear/reload capabilities
- Validation warnings for potential configuration issues
- **Dynamic evaluator loading** from scenario-specific `evaluators.py` files

#### 3.2.3 Scenario Evaluators (`evaluators.py`) ✅ COMPLETE

Each scenario directory can include an `evaluators.py` file that defines
programmatic evaluation functions specific to that scenario:

```
scenarios/
└── email_triage_basic/
    ├── scenario.json        # Scenario configuration
    ├── initial_state.json   # UES initial state
    └── evaluators.py        # Scenario-specific evaluators
```

**Evaluator Function Signature**:
```python
from src.green.scenarios.schema import EvaluatorFunc, EvalResult

# Type alias: Callable[[AgentBeatsEvalContext, dict], Awaitable[EvalResult]]
async def check_urgent_email_responses(
    ctx: AgentBeatsEvalContext,
    params: dict,
) -> EvalResult:
    """Check that urgent emails were responded to within the time limit."""
    # Access UES state via ctx.ues_client
    # Access action history via ctx.action_log
    # Return EvalResult with score and explanation
    ...
```

**Loading Evaluators**:
```python
# ScenarioManager automatically loads evaluators alongside scenarios
manager = ScenarioManager("/path/to/scenarios")
scenario = manager.get_scenario("email_triage_basic")
evaluators = manager.get_evaluators("email_triage_basic")

# Returns dict[str, EvaluatorFunc] mapping evaluator_id -> function
# e.g., {"check_urgent_email_responses": <function>, ...}
```

**Design Rationale**:
- Evaluators are co-located with scenarios for portability
- Scenarios can be shared/distributed as self-contained directories
- No need to modify Green agent code to add new evaluation criteria
- Evaluation logic can be complex and scenario-specific

**Tests**: 141 tests covering schema validation, loading from JSON files, evaluator loading, error handling, caching, and validation warnings.

### 3.3 LLM Configuration (`core/llm_config.py`) ✅ COMPLETE

**Location**: `src/green/core/llm_config.py`

**Status**: Implemented and tested (72 tests passing). Factory for creating LangChain
LLM instances supporting multiple providers.

**Supported Providers:**
- **OpenAI**: `gpt-*`, `o1-*`, `o3-*`, `chatgpt-*` prefixes
- **Anthropic**: `claude-*` prefix
- **Google**: `gemini-*` prefix
- **Ollama**: `ollama/*` prefix (e.g., `ollama/llama3.2`)

**Key Classes:**

```python
from enum import Enum
from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"

@dataclass(frozen=True)
class LLMConfig:
    """Configuration for creating an LLM instance."""
    model: str
    temperature: float = 0.7
    seed: int | None = None
    base_url: str | None = None  # Custom API endpoint override
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

class UnsupportedModelError(Exception):
    """Raised when a model identifier is not recognized."""

class LLMFactory:
    """Factory for creating LangChain chat model instances."""
    
    @classmethod
    def detect_provider(cls, model: str) -> LLMProvider:
        """Detect provider from model identifier string."""
    
    @classmethod
    def create(
        cls,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
        base_url: str | None = None,
        **extra_kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM instance from parameters."""
    
    @classmethod
    def create_from_config(cls, config: LLMConfig) -> BaseChatModel:
        """Create an LLM instance from a configuration object."""
```

**Key Features:**
- Case-insensitive model prefix detection
- Temperature warning (not exception) for OpenAI reasoning models (o1/o3 family)
- Custom `base_url` support for OpenAI, Anthropic, and Ollama
- Seed parameter passed only to providers that support it (OpenAI)
- Extra kwargs passthrough for provider-specific parameters
- Frozen `LLMConfig` dataclass for immutability and safety

**Tests**: 72 tests covering provider detection, model creation, parameter passing,
reasoning model handling, error cases, and edge cases.

### 3.4 Action Log Builder (`core/action_log.py`) ✅ COMPLETE

**Location**: `src/green/core/action_log.py`

**Status**: Implemented and tested (67 tests passing).

Helper class for building the assessment action log from UES event history.
Queries UES events after time advances and filters for Purple agent actions.

**Key Features:**
- Takes `purple_agent_id` in constructor for event filtering
- `add_events_from_turn()` processes UES events, returns Purple entries and Green events separately
- Converts UES `EventResponse` dicts to `ActionLogEntry` models
- Statistics methods (`get_total_actions()`, `get_successful_actions()`, `get_failed_actions()`)
- Filtering methods (`get_actions_by_turn()`, `get_actions_by_type()`)
- Reset capability for reuse across assessments (`reset()`)

**Exceptions:**
- `ActionLogBuilderError`: Base exception class
- `InvalidTurnStateError`: Raised for invalid turn lifecycle operations
- `InvalidTurnNumberError`: Raised for invalid turn numbers (with `expected`/`actual` attributes)

```python
class ActionLogBuilder:
    """Builds assessment action log from UES event history.
    
    The action log is built from UES event history after time advances,
    filtering for events attributed to the Purple agent via agent_id.
    This provides accurate, tamper-proof action recording.
    
    Lifecycle:
    1. Create with purple_agent_id
    2. After each time advance, call add_events_from_turn(events)
    3. Returns (purple_entries, green_events) tuple
    4. Use purple_entries for action log, green_events for response generation
    5. Call get_log() to retrieve all Purple entries
    
    Use reset() to clear all entries and start fresh for a new assessment.
    """
    
    def __init__(self, purple_agent_id: str):
        """Initialize builder with Purple agent's agent_id for filtering."""
    
    @property
    def current_turn(self) -> int:
        """Return highest turn number processed (0 if none)."""
    
    def add_events_from_turn(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[list[ActionLogEntry], list[dict[str, Any]]]:
        """Process UES events from a turn.
        
        Filters events by agent_id to separate Purple actions from
        Green-scheduled events and scenario-initiated events.
        
        Args:
            events: List of UES EventResponse dicts from the turn.
            
        Returns:
            Tuple of (purple_entries, green_events) where:
            - purple_entries: ActionLogEntry objects for Purple's actions
            - green_events: Raw event dicts for response generation
        """
    
    def get_log(self) -> list[ActionLogEntry]:
        """Return copy of complete action log (Purple actions only)."""
    
    def get_total_actions(self) -> int:
        """Return total number of Purple actions logged."""
    
    def get_successful_actions(self) -> int:
        """Return number of successful Purple actions."""
    
    def get_failed_actions(self) -> int:
        """Return number of failed Purple actions."""
    
    def get_actions_by_turn(self, turn_number: int) -> list[ActionLogEntry]:
        """Return all Purple actions for a specific turn."""
    
    def get_actions_by_type(self, action_type: str) -> list[ActionLogEntry]:
        """Return all Purple actions of a specific type."""
    
    def reset(self) -> None:
        """Clear all entries and reset for new assessment."""
```

**Tests**: 67 tests covering initial state, turn management, action logging,
statistics, filtering, reset functionality, exception handling, and edge cases.

### 3.5 New Message Collector (`core/message_collector.py`) ✅ COMPLETE

**Location**: `src/green/core/message_collector.py`

**Status**: Implemented and tested (41 tests passing).

The `NewMessageCollector` queries UES modality states for new messages since the
last check. Unlike querying the event log, this approach gives direct access to
full message objects with all fields needed for response generation (including
`message_id` and `thread_id`).

**Why not use the event log?**
1. UES events may not include `message_id` or `thread_id` for emails/SMS
2. Most events aren't response-triggering, making event scanning wasteful

**Query strategies by modality:**
- **Email**: `client.email.query(received_after=timestamp)`
- **SMS**: `client.sms.query(sent_after=timestamp)`
- **Calendar**: Compare event IDs (no creation-time filter available)

**Key Classes:**

```python
@dataclass
class NewMessages:
    """Container for new messages collected from UES modalities."""
    emails: list[Email] = field(default_factory=list)
    sms_messages: list[SMSMessage] = field(default_factory=list)
    calendar_events: list[CalendarEvent] = field(default_factory=list)
    
    @property
    def total_count(self) -> int: ...
    def is_empty(self) -> bool: ...


class NewMessageCollector:
    """Collects new messages from UES modalities since a timestamp.
    
    Lifecycle:
    1. Create with UES client
    2. Call initialize() to set baseline (records time, captures event IDs)
    3. After each time advance, call collect() to get new messages
    4. Call reset() to start fresh for a new assessment
    """
    
    def __init__(self, client: AsyncUESClient) -> None: ...
    
    @property
    def is_initialized(self) -> bool: ...
    
    @property
    def last_check_time(self) -> datetime | None: ...
    
    async def initialize(self, current_time: datetime) -> None:
        """Initialize collector with baseline state."""
    
    async def collect(self, current_time: datetime) -> NewMessages:
        """Collect new messages since last check."""
    
    def reset(self) -> None:
        """Reset collector state for a new assessment."""
```

**Exceptions:**
- `MessageCollectorError`: Base exception class
- `CollectorNotInitializedError`: Raised when `collect()` called before `initialize()`

**Tests**: 41 tests covering initialization, collection, reset, edge cases, and
exception handling.

### 3.6 Response Generation (`response/`) ✅ COMPLETE

**Location**: `src/green/response/`

**Status**: Implemented and tested (54 tests for generator, plus 40 tests for models and 29 tests for prompts, totaling 123 response-related tests).

**Implementation Details**:
- See `docs/design/RESPONSE_GENERATION_DESIGN.md` for comprehensive design documentation
- `src/green/response/models.py` - Data models including `ScheduledResponse`, `ShouldRespondResult`, `CalendarRSVPResult`, `ThreadContext`, `MessageContext`, `CalendarEventContext`
- `src/green/response/prompts.py` - LLM prompt templates for response generation decisions
- `src/green/response/generator.py` - Main `ResponseGenerator` class (~1200 lines)

The `ResponseGenerator` creates in-character responses from simulated contacts
based on new messages collected during the assessment. Depends on `LLMFactory` (3.3)
and `NewMessageCollector` (3.5).

**Responsibilities:**
1. Process new emails, SMS messages, and calendar events from `NewMessageCollector`
2. Determine if characters should respond (based on message content and character personality)
3. Generate in-character response content using LLM
4. Return scheduled responses for the `GreenAgent` to inject into UES

**Message Types Processed:**
- **Emails**: All new emails received since last check (from `Email` objects)
- **SMS**: All new SMS messages since last check (from `SMSMessage` objects)
- **Calendar**: All new calendar events since last check (from `CalendarEvent` objects)

```python
from ues.client._email import Email
from ues.client._sms import SMSMessage
from ues.client._calendar import CalendarEvent
from src.green.core.message_collector import NewMessages

class ScheduledResponse(BaseModel):
    """A response to be scheduled in UES."""
    character_id: str
    modality: Literal["email", "sms", "calendar"]
    response_content: str | dict  # str for message, dict for calendar RSVP
    scheduled_time: datetime
    # Context for scheduling (thread_id, reply_to, etc.)
    thread_id: str | None = None
    in_reply_to: str | None = None  # message_id for email replies
    subject: str | None = None  # For email responses


class ResponseGenerator:
    """Generates character responses to new messages during assessment.
    
    Created per-assessment since it depends on the scenario's character
    profiles. The GreenAgent handles actually scheduling the responses
    in UES after receiving them from this generator.
    
    Processes new messages from all modalities uniformly. Each message
    is checked to see if any recipient character should respond.
    
    Attributes:
        ues_client: For fetching thread history and current state.
        characters: Map of character_id -> CharacterProfile from scenario.
        user_character: The CharacterProfile representing the user (for exclusion).
        llm: LangChain LLM for generating response content.
    """
    
    def __init__(
        self,
        ues_client: AsyncUESClient,
        characters: dict[str, CharacterProfile],
        user_character: CharacterProfile,
        llm: BaseChatModel,
    ):
        self.ues_client = ues_client
        self.characters = characters
        self.user_character = user_character
        self.llm = llm
    
    async def process_new_messages(
        self,
        new_messages: NewMessages,
    ) -> list[ScheduledResponse]:
        """Generate character responses to new messages.
        
        Processes all new messages collected by NewMessageCollector.
        For each message, identifies recipient characters and determines
        if they should respond.
        
        Args:
            new_messages: Container with new emails, SMS, and calendar events.
            
        Returns:
            List of responses to schedule in UES.
        """
        responses: list[ScheduledResponse] = []
        
        for email in new_messages.emails:
            responses.extend(await self._process_email(email))
        
        for sms in new_messages.sms_messages:
            responses.extend(await self._process_sms(sms))
        
        for event in new_messages.calendar_events:
            responses.extend(await self._process_calendar_event(event))
        
        return responses
    
    async def _process_email(self, email: Email) -> list[ScheduledResponse]:
        """Process a single email and generate any character responses.
        
        Checks all recipients (to, cc) against scenario characters.
        The sender is excluded from potential responders.
        """
        responses: list[ScheduledResponse] = []
        sender_address = email.from_address
        
        # Collect all recipient addresses
        recipient_addresses = set(email.to_addresses + email.cc_addresses)
        
        for address in recipient_addresses:
            character = self._find_character_by_email(address)
            if not character:
                continue
            
            # Skip if this character was the sender
            if self._character_has_email(character, sender_address):
                continue
            
            # Skip if this is the user character
            if character.character_id == self.user_character.character_id:
                continue
            
            if await self._should_respond_to_email(email, character):
                response = await self._generate_email_response(email, character)
                if response:
                    responses.append(response)
        
        return responses
    
    async def _process_sms(self, sms: SMSMessage) -> list[ScheduledResponse]:
        """Process a single SMS message and generate any character responses.
        
        Checks all recipient numbers against scenario characters.
        The sender is excluded from potential responders.
        """
        responses: list[ScheduledResponse] = []
        sender_number = sms.from_number
        
        for number in sms.to_numbers:
            character = self._find_character_by_phone(number)
            if not character:
                continue
            
            # Skip if this character was the sender
            if self._character_has_phone(character, sender_number):
                continue
            
            # Skip if this is the user character
            if character.character_id == self.user_character.character_id:
                continue
            
            if await self._should_respond_to_sms(sms, character):
                response = await self._generate_sms_response(sms, character)
                if response:
                    responses.append(response)
        
        return responses
    
    async def _process_calendar_event(
        self,
        event: CalendarEvent,
    ) -> list[ScheduledResponse]:
        """Process a calendar event and generate RSVP responses from attendees.
        
        For calendar events, character attendees may respond with RSVP.
        """
        responses: list[ScheduledResponse] = []
        
        for attendee in event.attendees:
            # Skip if already responded
            if attendee.response_status != "needs_action":
                continue
            
            character = self._find_character_by_email(attendee.email)
            if not character:
                continue
            
            # Skip if this is the user character
            if character.character_id == self.user_character.character_id:
                continue
            
            # Generate RSVP response
            rsvp = await self._generate_calendar_rsvp(event, character)
            if rsvp:
                responses.append(rsvp)
        
        return responses
    
    # --- Helper methods ---
    
    def _find_character_by_email(self, email: str) -> CharacterProfile | None:
        """Find a character by email address."""
        for character in self.characters.values():
            if email in character.email_addresses:
                return character
        return None
    
    def _find_character_by_phone(self, phone: str) -> CharacterProfile | None:
        """Find a character by phone number."""
        for character in self.characters.values():
            if phone in character.phone_numbers:
                return character
        return None
    
    def _character_has_email(self, character: CharacterProfile, email: str) -> bool:
        """Check if a character owns an email address."""
        return email in character.email_addresses
    
    def _character_has_phone(self, character: CharacterProfile, phone: str) -> bool:
        """Check if a character owns a phone number."""
        return phone in character.phone_numbers
    
    # --- Response decision methods ---
    
    async def _should_respond_to_email(
        self,
        email: Email,
        character: CharacterProfile,
    ) -> bool:
        """Determine if character should respond to this email.
        
        Uses LLM to decide based on:
        - Message content (does it expect a reply?)
        - Character personality and special instructions
        - Thread context (conversation history)
        """
        # Pre-LLM heuristic checks
        if self._is_non_responder(character):
            return False
        
        # Get thread history for context
        thread_history = await self._get_email_thread_history(email.thread_id)
        
        # Use LLM to decide
        return await self._llm_should_respond(
            modality="email",
            character=character,
            message_content=email.body_text,
            sender_name=email.from_address,
            thread_history=thread_history,
        )
    
    async def _should_respond_to_sms(
        self,
        sms: SMSMessage,
        character: CharacterProfile,
    ) -> bool:
        """Determine if character should respond to this SMS."""
        if self._is_non_responder(character):
            return False
        
        thread_history = await self._get_sms_thread_history(sms.thread_id)
        
        return await self._llm_should_respond(
            modality="sms",
            character=character,
            message_content=sms.body,
            sender_name=sms.from_number,
            thread_history=thread_history,
        )
    
    def _is_non_responder(self, character: CharacterProfile) -> bool:
        """Check if character is configured as a non-responder."""
        # Check special instructions for "no response", "automated", etc.
        if character.special_instructions:
            lower = character.special_instructions.lower()
            if any(kw in lower for kw in ["no response", "automated", "do not respond"]):
                return True
        
        # Check for "never responds" timing pattern
        timing = character.response_timing
        if timing.base_delay >= timedelta(hours=24) and timing.variance == timedelta(0):
            return True
        
        return False
    
    # --- Response generation methods ---
    
    async def _generate_email_response(
        self,
        email: Email,
        character: CharacterProfile,
    ) -> ScheduledResponse | None:
        """Generate an in-character email response."""
        thread_history = await self._get_email_thread_history(email.thread_id)
        
        content = await self._llm_generate_response(
            modality="email",
            character=character,
            message_content=email.body_text,
            sender_name=email.from_address,
            thread_history=thread_history,
        )
        
        if not content:
            return None
        
        scheduled_time = self._calculate_response_time(
            character.response_timing,
            email.received_at,
        )
        
        return ScheduledResponse(
            character_id=character.character_id,
            modality="email",
            response_content=content,
            scheduled_time=scheduled_time,
            thread_id=email.thread_id,
            in_reply_to=email.message_id,
            subject=self._derive_email_subject(email.subject),
        )
    
    async def _generate_sms_response(
        self,
        sms: SMSMessage,
        character: CharacterProfile,
    ) -> ScheduledResponse | None:
        """Generate an in-character SMS response."""
        thread_history = await self._get_sms_thread_history(sms.thread_id)
        
        content = await self._llm_generate_response(
            modality="sms",
            character=character,
            message_content=sms.body,
            sender_name=sms.from_number,
            thread_history=thread_history,
        )
        
        if not content:
            return None
        
        scheduled_time = self._calculate_response_time(
            character.response_timing,
            sms.sent_at,
        )
        
        return ScheduledResponse(
            character_id=character.character_id,
            modality="sms",
            response_content=content,
            scheduled_time=scheduled_time,
            thread_id=sms.thread_id,
        )
    
    async def _generate_calendar_rsvp(
        self,
        event: CalendarEvent,
        character: CharacterProfile,
    ) -> ScheduledResponse | None:
        """Generate a calendar RSVP response."""
        # Use LLM to decide accept/decline/tentative based on character profile
        rsvp_decision = await self._llm_calendar_rsvp(event, character)
        
        if not rsvp_decision:
            return None
        
        scheduled_time = self._calculate_response_time(
            character.response_timing,
            event.created_at or datetime.now(timezone.utc),
        )
        
        return ScheduledResponse(
            character_id=character.character_id,
            modality="calendar",
            response_content=rsvp_decision,  # {"status": "accepted"|"declined"|"tentative"}
            scheduled_time=scheduled_time,
        )
    
    # --- Thread history retrieval ---
    
    async def _get_email_thread_history(self, thread_id: str) -> list[Email]:
        """Retrieve full email thread history."""
        result = await self.ues_client.email.query(
            thread_id=thread_id,
            sort_order="asc",
        )
        return result.emails
    
    async def _get_sms_thread_history(self, thread_id: str) -> list[SMSMessage]:
        """Retrieve full SMS thread history."""
        result = await self.ues_client.sms.query(
            thread_id=thread_id,
            sort_order="asc",
        )
        return result.messages
    
    # --- Utility methods ---
    
    def _calculate_response_time(
        self,
        timing: ResponseTiming,
        reference_time: datetime,
    ) -> datetime:
        """Calculate when a response should be scheduled."""
        base_seconds = timing.base_delay.total_seconds()
        variance_seconds = timing.variance.total_seconds()
        
        min_delay = max(0, base_seconds - variance_seconds)
        max_delay = base_seconds + variance_seconds
        delay_seconds = random.uniform(min_delay, max_delay)
        
        return reference_time + timedelta(seconds=delay_seconds)
    
    def _derive_email_subject(self, original_subject: str) -> str:
        """Derive reply subject line."""
        if original_subject.lower().startswith("re:"):
            return original_subject
        return f"Re: {original_subject}"
    
    # --- LLM interaction methods (stubs) ---
    
    async def _llm_should_respond(
        self,
        modality: str,
        character: CharacterProfile,
        message_content: str,
        sender_name: str,
        thread_history: list,
    ) -> bool:
        """Use LLM to decide if character should respond."""
        # Build prompt with character profile, message, thread history
        # Call LLM with structured output for {"should_respond": bool, "reasoning": str}
        ...
    
    async def _llm_generate_response(
        self,
        modality: str,
        character: CharacterProfile,
        message_content: str,
        sender_name: str,
        thread_history: list,
    ) -> str | None:
        """Use LLM to generate in-character response content."""
        # Build prompt with character profile, conversation context
        # Call LLM to generate response text
        ...
    
    async def _llm_calendar_rsvp(
        self,
        event: CalendarEvent,
        character: CharacterProfile,
    ) -> dict | None:
        """Use LLM to decide calendar RSVP."""
        # Consider event details, character's schedule/personality
        # Return {"status": "accepted"|"declined"|"tentative", "comment": "..."}
        ...
```

### 3.7 Criteria Judge (`evaluation/`) ✅ COMPLETE

**Location**: `src/green/evaluation/`

The `CriteriaJudge` evaluates Purple agent performance against scenario criteria.
Depends on `LLMFactory` (3.3).

**Responsibilities:**
1. Orchestrate evaluation of all criteria
2. Dispatch to programmatic evaluators or LLM-based evaluation
3. Convert `EvalResult` to `CriterionResult` with metadata
4. Aggregate scores by dimension

```python
from src.green.evaluation import CriteriaJudge
from src.green.core import LLMFactory
from src.common.agentbeats.updates import TaskUpdateEmitter

# Create judge with scenario criteria and evaluators
judge = CriteriaJudge(
    llm=LLMFactory.create("gpt-4o-mini"),
    criteria=scenario.criteria,
    evaluators=evaluator_registry,
    emitter=task_update_emitter,  # Optional, for observability
)

# Evaluate all criteria (runs in parallel)
results = await judge.evaluate_all(eval_context)

# Aggregate scores by dimension
scores = judge.aggregate_scores(results)
# Returns: Scores(overall=OverallScore(...), dimensions={...})

# Get unique dimensions from criteria
dimensions = judge.get_dimensions()
# Returns: ["accuracy", "efficiency", ...]
```

**Key Features:**
- **Parallel evaluation**: All criteria evaluated concurrently via `asyncio.gather()`
- **Score scaling**: Evaluator scores scaled to match `criterion.max_score`
- **Error handling**: Catches exceptions per-criterion, awards 0 points with explanation
- **TaskUpdateEmitter integration**: Emits `criterion_evaluated` updates for observability
- **Float scores**: Uses `float` for scores throughout to handle scaling accurately

**Supporting Modules:**
- `evaluation/models.py`: `LLMEvaluationResult` Pydantic model for structured LLM output
- `evaluation/prompts.py`: Prompt templates and context builders for LLM evaluation

### 3.8 Green Agent (`agent.py`)

The `GreenAgent` class is the high-level orchestrator for assessments. Each instance
owns its own UES server and can run multiple sequential assessments (one per task).

**Responsibilities:**
1. Own and manage UES server lifecycle (startup/shutdown)
2. Own LLM instances (shared across assessments)
3. Run the assessment turn loop
4. Coordinate response generation and evaluation
5. Build and return assessment results

**Per-Assessment vs Per-Agent Resources:**

| Resource | Lifetime | Reason |
|----------|----------|--------|
| UES server | Per-agent | Expensive to spin up; can be reset between assessments |
| `AsyncUESClient` | Per-agent | Reuses connection to UES server |
| LLM instances | Per-agent | Expensive to initialize; config-driven, not scenario-driven |
| `ResponseGenerator` | Per-assessment | Needs scenario's character profiles |
| `CriteriaJudge` | Per-assessment | Needs scenario's evaluation criteria and evaluators |
| `ActionLogBuilder` | Per-assessment | Tracks actions for one assessment |
| `NewMessageCollector` | Per-assessment | Tracks new messages for one assessment |

```python
class GreenAgent:
    """High-level orchestrator for a single assessment context.
    
    Each GreenAgent instance owns its own UES server and can run multiple
    sequential assessments. The agent is created once per context_id and
    reused across assessment tasks within that context.
    
    Attributes:
        config: Green agent configuration.
        ues_port: Port for this agent's UES server.
        ues_client: Async client for UES API calls.
        response_llm: LLM for generating character responses.
        evaluation_llm: LLM for criteria evaluation.
    """
    
    def __init__(self, ues_port: int, config: GreenAgentConfig):
        """Initialize the GreenAgent.
        
        Note: Call startup() after construction to spin up the UES server.
        
        Args:
            ues_port: Port to run the UES server on.
            config: Green agent configuration.
        """
        self.config = config
        self.ues_port = ues_port
        
        # Initialized in startup()
        self._ues_process: subprocess.Popen | None = None
        self.ues_client: AsyncUESClient | None = None
        
        # LLMs (created once, reused across assessments)
        self.response_llm = LLMFactory.create(
            config.response_generator_model,
            temperature=0.7,
        )
        self.evaluation_llm = LLMFactory.create(
            config.evaluation_model,
            temperature=0.0,  # Deterministic for evaluation
        )
        
        # Per-assessment state (set in run())
        self._current_task_id: str | None = None
        self._cancelled: bool = False
    
    async def startup(self) -> None:
        """Start the UES server and create client.
        
        Spins up a UES server process on the configured port and waits
        for it to become ready. Creates an AsyncUESClient for API calls.
        The proctor API key is auto-generated by the UES server at startup.
        """
        self._ues_process, self._proctor_api_key = await self._start_ues_server()
        self.ues_client = AsyncUESClient(
            base_url=f"http://localhost:{self.ues_port}",
            api_key=self._proctor_api_key,
        )
        # Wait for server to be ready
        await self._wait_for_ues_ready()
    
    async def shutdown(self) -> None:
        """Stop the UES server and cleanup resources."""
        if self.ues_client:
            await self.ues_client.close()
            self.ues_client = None
        if self._ues_process:
            self._ues_process.terminate()
            self._ues_process.wait(timeout=10)
            self._ues_process = None
    
    async def cancel(self, task_id: str) -> None:
        """Request cancellation of an ongoing assessment.
        
        Sets a flag that the turn loop checks to exit early.
        
        Args:
            task_id: The task to cancel (must match current task).
        """
        if self._current_task_id == task_id:
            self._cancelled = True
    
    async def run(
        self,
        task_id: str,
        updater: TaskUpdater,
        scenario: ScenarioConfig,
        evaluators: EvaluatorRegistry,
        purple_client: A2AClientWrapper,
        assessment_config: dict[str, Any],
    ) -> AssessmentResults:
        """Run a complete assessment.
        
        This is the main entry point for running an assessment. It:
        1. Sets up the assessment environment (resets UES, creates helpers)
        2. Sends AssessmentStartMessage to Purple agent
        3. Runs the turn loop until completion or max turns
        4. Evaluates the Purple agent's performance
        5. Returns the assessment results
        
        Args:
            task_id: Unique identifier for this assessment task.
            updater: TaskUpdater for emitting A2A events.
            scenario: Loaded scenario configuration.
            evaluators: Map of evaluator_id -> evaluator function.
            purple_client: A2A client for Purple agent communication.
            assessment_config: Additional config from the request.
            
        Returns:
            Complete assessment results including scores and action log.
        """
        self._current_task_id = task_id
        self._cancelled = False
        
        # Create per-assessment helper objects
        # Note: ActionLogBuilder needs purple_agent_id for event filtering
        purple_agent_id = await self._get_purple_agent_id(purple_client)
        action_log_builder = ActionLogBuilder(purple_agent_id=purple_agent_id)
        
        # NewMessageCollector queries modality states for new messages
        message_collector = NewMessageCollector(client=self.ues_client)
        
        # ResponseGenerator uses NewMessages objects (not raw events)
        response_generator = ResponseGenerator(
            ues_client=self.ues_client,
            characters=scenario.characters,
            user_character=scenario.get_user_character_profile(),
            llm=self.response_llm,
        )
        criteria_judge = CriteriaJudge(
            llm=self.evaluation_llm,
            criteria=scenario.evaluation_criteria,
            evaluators=evaluators,
        )
        
        try:
            # Phase 1: Setup
            await self._setup_assessment(scenario, updater)
            
            # Initialize message collector AFTER UES is set up
            # This establishes baseline time and captures existing calendar events
            time_state = await self.ues_client.time.get_state()
            await message_collector.initialize(time_state.current_time)
            
            # Phase 2: Send start message to Purple
            initial_state_summary = await self._build_initial_state_summary()
            await self._send_assessment_start(
                purple_client, scenario, initial_state_summary, assessment_config
            )
            
            # Capture initial state for evaluation context
            initial_state = await self._capture_state_snapshot()
            
            # Phase 3: Turn loop
            turn = 0
            max_turns = assessment_config.get("max_turns", self.config.default_max_turns)
            early_completion = False
            
            while turn < max_turns and not self._cancelled and not early_completion:
                turn += 1
                turn_result = await self._run_turn(
                    turn=turn,
                    updater=updater,
                    purple_client=purple_client,
                    action_log_builder=action_log_builder,
                    message_collector=message_collector,
                    response_generator=response_generator,
                )
                early_completion = turn_result.early_completion
            
            # Phase 4: Evaluation
            final_state = await self._capture_state_snapshot()
            
            eval_context = AgentBeatsEvalContext(
                client=self.ues_client,
                scenario_config=scenario.model_dump(),
                action_log=action_log_builder.get_log(),
                initial_state=initial_state,
                final_state=final_state,
                user_prompt=scenario.user_prompt,
            )
            
            criteria_results = await criteria_judge.evaluate_all(eval_context)
            
            # Phase 5: Build and return results
            results = self._build_results(
                scenario=scenario,
                criteria_results=criteria_results,
                action_log=action_log_builder.get_log(),
                turns_completed=turn,
                early_completion=early_completion,
            )
            
            # Notify Purple of completion
            await self._send_assessment_complete(purple_client, results)
            
            return results
            
        finally:
            self._current_task_id = None
    
    async def _setup_assessment(
        self,
        scenario: ScenarioConfig,
        updater: TaskUpdater,
    ) -> None:
        """Initialize UES with scenario's initial state.
        
        Resets UES to a clean state and loads the scenario's initial
        state configuration.
        """
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text=f"Setting up scenario: {scenario.name}"))
            ])
        )
        
        # Reset UES and load initial state
        await self.ues_client.reset()
        await self.ues_client.load_state(scenario.initial_state)
        
        # Set simulation start time
        if scenario.start_time:
            await self.ues_client.time.set_time(scenario.start_time)
    
    async def _run_turn(
        self,
        turn: int,
        updater: TaskUpdater,
        purple_client: A2AClientWrapper,
        action_log_builder: ActionLogBuilder,
        message_collector: NewMessageCollector,
        response_generator: ResponseGenerator,
    ) -> TurnResult:
        """Execute a single assessment turn.
        
        A turn consists of:
        1. Send TurnStartMessage to Purple
        2. Wait for TurnCompleteMessage (or EarlyCompletionMessage)
        3. Advance simulation time
        4. Query UES events for action log (Purple agent actions)
        5. Collect new messages from modalities (for response generation)
        6. Generate character responses to new messages
        7. Schedule responses in UES (will fire on next time advance)
        
        Args:
            turn: Current turn number (1-indexed).
            updater: For emitting task updates.
            purple_client: For communicating with Purple agent.
            action_log_builder: For recording actions from UES events.
            message_collector: For collecting new messages from modalities.
            response_generator: For generating character responses.
            
        Returns:
            TurnResult with turn outcome details.
        """
        # Get current simulation time
        time_state = await self.ues_client.time.get_state()
        turn_start_time = time_state.current_time
        
        # Send turn start to Purple
        turn_start = TurnStartMessage(
            current_time=turn_start_time,
            turn_number=turn,
            events_processed=0,  # TODO: Track from previous turn
        )
        
        # Wait for Purple's response
        response = await purple_client.send_message(
            turn_start.model_dump_json(),
            timeout=self.config.default_turn_timeout,
        )
        
        # Parse response
        response_data = json.loads(response)
        message_type = response_data.get("message_type")
        
        if message_type == "early_completion":
            early_completion = EarlyCompletionMessage.model_validate(response_data)
            return TurnResult(
                turn_number=turn,
                actions_taken=0,
                notes=early_completion.reason,
                time_step="PT0S",
                events_processed=0,
                early_completion=True,
            )
        
        turn_complete = TurnCompleteMessage.model_validate(response_data)
        
        # Advance simulation time FIRST (scheduled events fire during this)
        events_processed = await self._advance_time(turn_complete.time_step)
        
        # Get the new current time after advancement
        time_state = await self.ues_client.time.get_state()
        current_time = time_state.current_time
        
        # --- Action Log: Query events for Purple agent tracking ---
        # This is for scoring and audit purposes
        events = await self.ues_client.events.get_events(
            since=turn_start_time,
            until=current_time,
        )
        
        # Build action log from events (filters by Purple agent_id)
        purple_entries = action_log_builder.add_events_from_turn(
            turn=turn,
            events=events,
        )
        
        # Emit action updates for observability
        for entry in purple_entries:
            await self._emit_action_update(updater, turn, entry)
        
        # --- Response Generation: Collect new messages from modalities ---
        # This gives us full message objects with thread_id, message_id, etc.
        new_messages = await message_collector.collect(current_time)
        
        # Generate character responses to new messages
        responses = await response_generator.process_new_messages(new_messages)
        
        # Schedule responses in UES (will fire on future time advances)
        for scheduled in responses:
            await self._schedule_response(scheduled)
        
        return TurnResult(
            turn_number=turn,
            actions_taken=len(purple_entries),
            notes=turn_complete.notes,
            time_step=turn_complete.time_step,
            events_processed=events_processed,
            early_completion=False,
        )
    
    # ... additional helper methods (_start_ues_server, _wait_for_ues_ready,
    #     _build_initial_state_summary, _send_assessment_start, _capture_state_snapshot,
    #     _emit_action_update, _schedule_response, _advance_time,
    #     _send_assessment_complete, _build_results)
```

**Helper Methods Summary:**

| Method | Purpose |
|--------|---------|
| `_start_ues_server()` | Spawn UES server subprocess |
| `_wait_for_ues_ready()` | Poll until UES responds to health check |
| `_build_initial_state_summary()` | Create `InitialStateSummary` from UES state |
| `_send_assessment_start()` | Send `AssessmentStartMessage` to Purple |
| `_capture_state_snapshot()` | Get current UES state for evaluation |
| `_emit_action_update()` | Emit `ActionObservedUpdate` for each action |
| `_schedule_response()` | Schedule a character response in UES |
| `_advance_time()` | Advance UES simulation time, return events processed |
| `_send_assessment_complete()` | Send `AssessmentCompleteMessage` to Purple |
| `_build_results()` | Construct `AssessmentResults` from evaluation data |

### 3.9 Green Agent Executor (`executor.py`)

The executor is the entry point for A2A requests. It implements the `AgentExecutor`
interface from the A2A SDK and is responsible for managing `GreenAgent` instances
and routing assessment requests to them. Depends on `GreenAgent` (3.8) and
`ScenarioManager` (3.2). Uses `A2AClientWrapper` from `src/common/a2a/` for
Purple agent communication.

**Responsibilities:**
1. Validate incoming assessment requests (`EvalRequest` format)
2. Manage `GreenAgent` instances keyed by `context_id`
3. Allocate unique UES ports for new `GreenAgent` instances
4. Create `TaskUpdater` for A2A event emission
5. Load scenarios and evaluators via `ScenarioManager`
6. Create A2A client wrapper for Purple agent communication
7. Delegate assessment execution to `GreenAgent.run()`
8. Handle task cancellation

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Task, TaskState
from a2a.utils import new_task, new_agent_text_message

class EvalRequest(BaseModel):
    """Assessment request format from AgentBeats platform.
    
    Follows the AgentBeats green agent template pattern.
    """
    participants: dict[str, HttpUrl]  # role -> agent URL (e.g., {"assistant": "http://..."})
    config: dict[str, Any]  # Assessment config including scenario_id

class GreenAgentExecutor(AgentExecutor):
    """A2A executor for the Green Agent.
    
    Manages GreenAgent instances and routes assessment requests. Each unique
    context_id gets its own GreenAgent instance with its own UES server.
    
    Attributes:
        scenario_manager: Loads scenarios and evaluators from disk.
        config: Green agent configuration.
        agents: Map of context_id -> GreenAgent instances.
        _next_ues_port: Port allocator for UES servers.
    """
    
    def __init__(
        self,
        scenario_manager: ScenarioManager,
        config: GreenAgentConfig,
    ):
        self.scenario_manager = scenario_manager
        self.config = config
        self.agents: dict[str, GreenAgent] = {}  # context_id -> GreenAgent
        self._next_ues_port = config.ues_base_port  # e.g., 8100
        self._port_lock = asyncio.Lock()
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle incoming assessment request.
        
        Called by the A2A server when a new message arrives. This method:
        1. Validates the request format
        2. Gets or creates a Task for tracking
        3. Gets or creates a GreenAgent for this context
        4. Loads the scenario and evaluators
        5. Creates the Purple agent A2A client
        6. Runs the assessment
        7. Emits results as an artifact
        
        Args:
            context: Request context with message, task_id, context_id, etc.
            event_queue: Queue for emitting A2A events (task updates, artifacts).
        """
        # Validate request has a message
        if not context.message:
            raise ServerError(error=InvalidRequestError(message="Missing message"))
        
        # Get or create task
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already completed"
            ))
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        context_id = task.context_id
        task_id = task.id
        
        # Create TaskUpdater for emitting events
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        
        try:
            # Parse and validate the assessment request
            request_text = get_message_text(context.message)
            eval_request = EvalRequest.model_validate_json(request_text)
            
            # Validate required fields
            if "assistant" not in eval_request.participants:
                await updater.reject(new_agent_text_message(
                    "Missing 'assistant' role in participants"
                ))
                return
            
            scenario_id = eval_request.config.get("scenario_id")
            if not scenario_id:
                await updater.reject(new_agent_text_message(
                    "Missing 'scenario_id' in config"
                ))
                return
            
            # Get or create GreenAgent for this context
            agent = await self._get_or_create_agent(context_id)
            
            # Load scenario and evaluators
            try:
                scenario = self.scenario_manager.get_scenario(scenario_id)
                evaluators = self.scenario_manager.get_evaluators(scenario_id)
            except ScenarioNotFoundError as e:
                await updater.reject(new_agent_text_message(str(e)))
                return
            
            # Create A2A client for Purple agent
            purple_url = str(eval_request.participants["assistant"])
            purple_client = await self._create_purple_client(purple_url)
            
            # Run the assessment
            results = await agent.run(
                task_id=task_id,
                updater=updater,
                scenario=scenario,
                evaluators=evaluators,
                purple_client=purple_client,
                assessment_config=eval_request.config,
            )
            
            # Emit results as artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text="Assessment completed successfully.")),
                    Part(root=DataPart(data=results.model_dump())),
                ],
                name="assessment_results",
            )
            await updater.complete()
            
        except Exception as e:
            logger.exception(f"Assessment failed: {e}")
            await updater.failed(new_agent_text_message(f"Assessment error: {e}"))
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle cancellation request.
        
        Cancels an ongoing assessment for the given task. Signals the
        GreenAgent to stop and emits a cancelled status.
        
        Args:
            context: Request context with task_id to cancel.
            event_queue: Queue for emitting cancellation status.
        """
        task_id = context.task_id
        context_id = context.context_id
        
        # Find the agent handling this context
        if context_id and context_id in self.agents:
            agent = self.agents[context_id]
            await agent.cancel(task_id)
        
        # Emit cancellation status
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()
    
    async def _get_or_create_agent(self, context_id: str) -> GreenAgent:
        """Get existing GreenAgent or create a new one for this context.
        
        Each context_id gets its own GreenAgent with its own UES server.
        This ensures Purple agents being assessed in parallel don't
        interfere with each other.
        
        Args:
            context_id: The conversation context identifier.
            
        Returns:
            GreenAgent instance for this context.
        """
        if context_id not in self.agents:
            # Allocate a unique port for this agent's UES server
            async with self._port_lock:
                ues_port = self._next_ues_port
                self._next_ues_port += 1
            
            # Create new GreenAgent
            agent = GreenAgent(ues_port=ues_port, config=self.config)
            await agent.startup()
            self.agents[context_id] = agent
        
        return self.agents[context_id]
    
    async def _create_purple_client(self, url: str) -> A2AClientWrapper:
        """Create an A2A client wrapper for communicating with Purple agent.
        
        Args:
            url: The Purple agent's base URL.
            
        Returns:
            Configured A2A client wrapper.
        """
        return A2AClientWrapper(base_url=url, timeout=self.config.default_turn_timeout)
    
    async def cleanup(self) -> None:
        """Shutdown all GreenAgent instances.
        
        Called when the server is shutting down. Ensures all UES servers
        are properly terminated.
        """
        for context_id, agent in self.agents.items():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent for {context_id}: {e}")
        self.agents.clear()
```

### 3.10 Module Structure Summary

```
src/green/
├── __init__.py
├── server.py              # A2A server setup, agent card, entry point
├── llm_config.py          # 3.3 ✅ COMPLETE - LangChain LLM creation
├── action_log.py          # 3.4 ✅ COMPLETE - action log construction
├── message_collector.py   # 3.5 ✅ COMPLETE - new message collection from UES
├── response_generator.py  # 3.6 ResponseGenerator - character response generation
├── judge.py               # 3.7 CriteriaJudge - evaluation orchestration
├── agent.py               # 3.8 GreenAgent - assessment orchestration, UES management
├── executor.py            # 3.9 GreenAgentExecutor - request handling, agent lifecycle
└── scenarios/             # 3.2 ✅ COMPLETE - scenario schema and loading
    ├── __init__.py
    ├── schema.py
    ├── loader.py
    └── README.md
```

**Implementation Order** (dependencies flow upward):
1. `llm_config.py` - No dependencies ✅ COMPLETE
2. `action_log.py` - No dependencies ✅ COMPLETE
3. `message_collector.py` - Depends on UES client ✅ COMPLETE
4. `response_generator.py` - Depends on `llm_config.py`, `message_collector.py`
5. `judge.py` - Depends on `llm_config.py`
6. `agent.py` - Depends on all of the above
7. `executor.py` - Depends on `agent.py`, `scenarios/`, and `A2AClientWrapper` from `src/common/a2a/`
8. `server.py` - Depends on `executor.py`

### 3.11 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Assessment Request Flow                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AgentBeats Platform                                                         │
│         │                                                                    │
│         │ EvalRequest (participants, config)                                 │
│         ▼                                                                    │
│  ┌─────────────────┐                                                        │
│  │ GreenAgentExecutor │                                                      │
│  │  1. Validate request                                                      │
│  │  2. Get/create GreenAgent (by context_id)                                │
│  │  3. Load scenario + evaluators                                           │
│  │  4. Create Purple A2A client                                              │
│  └─────────┬───────┘                                                        │
│            │                                                                 │
│            │ run(task_id, updater, scenario, evaluators, purple_client, config)
│            ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   GreenAgent    │                                                        │
│  │  1. Setup UES with scenario initial state                                │
│  │  2. Send AssessmentStartMessage to Purple                                │
│  │  3. Turn Loop:                                                            │
│  │     a. Send TurnStartMessage to Purple                                   │
│  │     b. Receive TurnCompleteMessage                                       │
│  │     c. Log actions (ActionLogBuilder)                                    │
│  │     d. Generate responses (ResponseGenerator)                            │
│  │     e. Schedule responses in UES                                         │
│  │     f. Advance UES time                                                  │
│  │  4. Evaluate (CriteriaJudge)                                             │
│  │  5. Send AssessmentCompleteMessage to Purple                             │
│  │  6. Return AssessmentResults                                             │
│  └─────────┬───────┘                                                        │
│            │                                                                 │
│            │ AssessmentResults                                               │
│            ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ GreenAgentExecutor │                                                      │
│  │  - Emit results as A2A artifact                                          │
│  │  - Mark task complete                                                    │
│  └─────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
```

---

## Phase 4: Purple Agent Template

**Location**: `src/purple/`

A lightweight template to accelerate Purple agent development.

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Purple Agent Template                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Server    │  │  Executor   │  │  Assessment │             │
│  │  (A2A)      │──│  (A2A)      │──│  Handler    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                │                     │
│                          ▼                ▼                     │
│                   ┌─────────────┐  ┌─────────────┐             │
│                   │    Agent    │  │    UES      │             │
│                   │   (Custom)  │  │   Client    │             │
│                   └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Base Agent Class (`base_agent.py`)

```python
from abc import ABC, abstractmethod

class BasePurpleAgent(ABC):
    """Base class for Purple agent implementations."""
    
    def __init__(self, config: PurpleAgentConfig):
        self.config = config
        self.ues_client: UESClient | None = None
        self.current_time: datetime | None = None
    
    def initialize(
        self,
        ues_url: str,
        api_key: str,
        current_time: datetime,
    ) -> None:
        """Initialize with UES access (called by executor)."""
        self.ues_client = UESClient(ues_url, api_key=api_key)
        self.current_time = current_time
    
    @abstractmethod
    async def get_user_instructions(self) -> str:
        """Retrieve and parse user instructions from chat."""
        ...
    
    @abstractmethod
    async def execute_turn(self) -> TurnCompleteMessage:
        """Execute a single turn of the assessment."""
        ...
    
    @abstractmethod
    async def should_complete_early(self) -> tuple[bool, str | None]:
        """Check if assessment goals are met."""
        ...
```

### 4.3 Assessment Handler (`handler.py`)

```python
class AssessmentHandler:
    """Handles the assessment lifecycle for Purple agents."""
    
    def __init__(self, agent: BasePurpleAgent): ...
    
    async def handle_assessment_start(
        self,
        message: AssessmentStartMessage,
    ) -> None:
        """Process assessment start and initialize agent."""
        ...
    
    async def handle_turn_start(
        self,
        message: TurnStartMessage,
    ) -> TurnCompleteMessage | EarlyCompletionMessage:
        """Process turn start and execute turn."""
        ...
    
    async def handle_assessment_complete(
        self,
        message: AssessmentCompleteMessage,
    ) -> None:
        """Handle assessment completion notification."""
        ...
```

### 4.4 Purple Agent Executor (`executor.py`)

```python
class PurpleAgentExecutor(AgentExecutor):
    """A2A executor for Purple agents."""
    
    def __init__(self, agent: BasePurpleAgent):
        self.agent = agent
        self.handler = AssessmentHandler(agent)
        self.task_id: str | None = None
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle incoming messages from Green agent."""
        message = context.message
        
        # Route to appropriate handler based on message type
        if is_assessment_start(message):
            await self._handle_assessment_start(message, event_queue)
        elif is_turn_start(message):
            await self._handle_turn_start(message, event_queue)
        elif is_assessment_complete(message):
            await self._handle_assessment_complete(message, event_queue)
        else:
            raise ValueError(f"Unknown message type")
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle cancellation."""
        ...
```

### 4.5 UES Client Wrapper (`ues_wrapper.py`)

Convenience wrapper around UES client for common Purple agent operations.

```python
class PurpleUESClient:
    """UES client wrapper with convenience methods for Purple agents."""
    
    def __init__(self, ues_client: UESClient):
        self._client = ues_client
    
    # Email operations
    async def get_unread_emails(self) -> list[Email]: ...
    async def get_emails_by_label(self, label: str) -> list[Email]: ...
    async def reply_to_email(
        self, email_id: str, body: str, cc: list[str] | None = None
    ) -> Email: ...
    async def send_email(
        self, to: list[str], subject: str, body: str
    ) -> Email: ...
    
    # Calendar operations
    async def get_todays_events(self) -> list[CalendarEvent]: ...
    async def get_events_in_range(
        self, start: datetime, end: datetime
    ) -> list[CalendarEvent]: ...
    async def create_event(
        self, title: str, start: datetime, end: datetime, **kwargs
    ) -> CalendarEvent: ...
    async def rsvp_to_event(
        self, event_id: str, status: Literal["accepted", "declined", "tentative"]
    ) -> CalendarEvent: ...
    
    # SMS operations
    async def get_unread_sms(self) -> list[SMS]: ...
    async def send_sms(self, to: str, message: str) -> SMS: ...
    
    # Chat operations
    async def get_chat_messages(self) -> list[ChatMessage]: ...
    async def send_chat_message(self, content: str) -> ChatMessage: ...
    
    # State queries
    async def get_current_time(self) -> datetime: ...
    async def get_full_state(self) -> dict: ...
```

### 4.6 Example Implementation (`examples/simple_agent.py`)

```python
"""Example: Simple email triage agent."""
from datetime import datetime

class SimpleTriageAgent(BasePurpleAgent):
    """A simple agent that triages emails based on keywords."""
    
    async def get_user_instructions(self) -> str:
        messages = await self.ues_client.get_chat_messages()
        # Return the most recent user message
        return messages[-1].content if messages else ""
    
    async def execute_turn(self) -> TurnCompleteMessage:
        actions: list[ActionLogEntry] = []
        notes = []
        
        # Get unread emails
        emails = await self.ues_client.get_unread_emails()
        
        for email in emails:
            if "[URGENT]" in email.subject:
                # Mark as read and flag
                await self.ues_client.mark_email_read(email.id)
                actions.append(ActionLogEntry(
                    timestamp=datetime.now(tz=timezone.utc),
                    action="email.mark_read",
                    parameters={"email_id": email.id},
                    success=True,
                ))
                
                await self.ues_client.label_email(email.id, "urgent")
                actions.append(ActionLogEntry(
                    timestamp=datetime.now(tz=timezone.utc),
                    action="email.label",
                    parameters={"email_id": email.id, "label": "urgent"},
                    success=True,
                ))
                
                notes.append(f"Flagged urgent: {email.subject}")
        
        return TurnCompleteMessage(
            actions=actions,
            notes="\n".join(notes) if notes else None,
            time_step="PT1H",
        )
    
    async def should_complete_early(self) -> tuple[bool, str | None]:
        # Check if all tasks are done
        unread = await self.ues_client.get_unread_emails()
        if not unread:
            return True, "All emails processed"
        return False, None
```

---

## Phase 5: Submission Requirements

### 5.1 Required Deliverables

| Requirement | Description | Location |
|-------------|-------------|----------|
| **Abstract** | Brief description of tasks the Green agent evaluates | `README.md` |
| **GitHub Repository** | Complete source code with documentation | This repo |
| **Baseline Purple Agent(s)** | A2A-compatible agent(s) demonstrating the benchmark | `src/purple/examples/` |
| **Docker Image (Green)** | Packaged Green agent that runs without intervention | `Dockerfile.green` |
| **Docker Image (Purple)** | Packaged baseline Purple agent | `Dockerfile.purple` |
| **AgentBeats Registration** | Register on agentbeats.dev | N/A (manual) |
| **Demo Video** | Up to 3 minutes demonstrating the Green agent | N/A (external) |

### 5.2 Dockerization

#### Green Agent Dockerfile (`Dockerfile.green`)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy source
COPY src/ ./src/
COPY scenarios/ ./scenarios/

# Entry point
ENTRYPOINT ["uv", "run", "python", "-m", "src.green"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
```

#### Purple Agent Dockerfile (`Dockerfile.purple`)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy source
COPY src/ ./src/

# Entry point
ENTRYPOINT ["uv", "run", "python", "-m", "src.purple.examples.simple_agent"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
```

### 5.3 CLI Interface

Both agents must support these arguments:
- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to listen on (default: `8000` for Green, `8001` for Purple)
- `--card-url`: URL where agent card will be accessible

### 5.4 Documentation Requirements

1. **README.md**: Project overview, setup instructions, usage
2. **docs/SCENARIOS.md**: Description of available scenarios
3. **docs/EVALUATION_CRITERIA.md**: Explanation of scoring dimensions and criteria
4. **docs/PURPLE_AGENT_GUIDE.md**: Guide for building Purple agents

---

## Project Structure

```
ues-agentbeats/
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── a2a/
│   │   │   ├── __init__.py
│   │   │   ├── agent_card.py      # Agent card builder
│   │   │   ├── server.py          # Server utilities
│   │   │   ├── client.py          # Client utilities
│   │   │   ├── messages.py        # Message helpers
│   │   │   ├── tasks.py           # Task management
│   │   │   └── artifacts.py       # Artifact helpers
│   │   └── agentbeats/
│   │       ├── __init__.py
│   │       ├── messages.py        # AgentBeats message types
│   │       ├── results.py         # Assessment results models
│   │       ├── updates.py         # Task update helpers
│   │       └── config.py          # Configuration models
│   │
│   ├── green/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── __main__.py            # Entry point
│   │   ├── executor.py            # A2A executor
│   │   ├── core/                  # Core infrastructure (Phase 3.3-3.5)
│   │   │   ├── __init__.py
│   │   │   ├── llm_config.py      # LLM factory
│   │   │   ├── action_log.py      # Action log builder
│   │   │   └── message_collector.py  # New message collector
│   │   ├── response/              # Response generation (Phase 3.6)
│   │   │   ├── __init__.py
│   │   │   ├── generator.py       # ResponseGenerator class
│   │   │   ├── models.py          # Response data models
│   │   │   └── prompts.py         # Response prompt templates
│   │   ├── evaluation/            # Criteria evaluation (Phase 3.7)
│   │   │   ├── __init__.py
│   │   │   ├── judge.py           # CriteriaJudge class
│   │   │   ├── models.py          # LLMEvaluationResult model
│   │   │   └── prompts.py         # Evaluation prompt templates
│   │   ├── scenarios/             # Scenario management (Phase 3.2)
│   │   │   ├── __init__.py
│   │   │   ├── schema.py          # Scenario schema
│   │   │   └── loader.py          # Scenario loader
│   │   ├── assessment/            # Assessment orchestration (Phase 3.8)
│   │   │   └── ...
│   │   └── prompts/               # Re-exports (deprecated)
│   │       └── __init__.py
│   │
│   └── purple/
│       ├── __init__.py
│       ├── __main__.py            # Entry point (template)
│       ├── base_agent.py          # Base agent class
│       ├── handler.py             # Assessment handler
│       ├── executor.py            # A2A executor
│       ├── ues_wrapper.py         # UES client wrapper
│       └── examples/
│           ├── __init__.py
│           └── simple_agent.py    # Example implementation
│
├── scenarios/
│   └── email_triage_basic/
│       ├── scenario.json          # Scenario config
│       ├── initial_state.json     # UES state export
│       └── evaluators.py          # Programmatic evaluators
│
├── tests/
│   ├── common/
│   │   ├── a2a/
│   │   └── agentbeats/
│   ├── green/
│   │   ├── core/                  # Tests for core infrastructure
│   │   ├── response/              # Tests for response generation
│   │   ├── evaluation/            # Tests for criteria evaluation
│   │   └── scenarios/             # Tests for scenario management
│   └── purple/
│
├── docs/
│   ├── AGENTBEATS_COMPETITION_DESCRIPTION.md
│   ├── ASSESSMENT_FLOW.md
│   ├── IMPLEMENTATION_PLAN.md     # This document
│   ├── SCENARIOS.md
│   ├── EVALUATION_CRITERIA.md
│   └── PURPLE_AGENT_GUIDE.md
│
├── Dockerfile.green
├── Dockerfile.purple
├── docker-compose.yaml            # For local testing
├── pyproject.toml
├── README.md
└── main.py
```

---

## Development Timeline

### Week 1: Foundation (Days 1-3) ✅ COMPLETE
- [x] Set up project structure
- [x] Implement `src/common/a2a/` module
- [x] Write tests for A2A helpers
- [x] Validate with simple A2A server test

### Week 1: AgentBeats Common (Days 4-5) ✅ COMPLETE
- [x] Implement `src/common/agentbeats/` module
- [x] Define message types and results schema
- [x] Write tests for serialization/validation
- [x] Implement configuration module with CLI/env support

### Week 2: Green Agent Core (Days 6-10) 🔄 IN PROGRESS
- [x] Implement scenario schema and loader (141 tests)
- [x] Create example scenario (email_triage_basic)
- [x] Implement LLM configuration/factory (59 tests)
- [x] Implement action log builder (20 tests)
- [x] Implement new message collector (41 tests)
- [x] Implement response generator with LLM integration (54 unit + 11 integration tests)
- [x] Implement response data models (40 tests)
- [x] Implement LLM prompt templates (29 tests)
- [ ] Implement assessment orchestrator
- [ ] Implement turn handler
- [ ] Implement criteria judge

### Week 3: Green Agent LLM Integration (Days 11-14) ✅ COMPLETE (merged into Week 2)
- [x] Integrate LangChain for response generation
- [x] Implement LLM configuration/factory
- [x] Integration tests with Ollama and OpenAI
- [ ] Integrate LangChain for criteria judging
- [ ] Create first complete scenario

### Week 4: Purple Agent Template (Days 15-17)
- [ ] Implement base agent class
- [ ] Implement assessment handler
- [ ] Implement Purple executor
- [ ] Create simple example agent
- [ ] Test against Green agent

### Week 4: Documentation & Submission (Days 18-21)
- [ ] Dockerize both agents
- [ ] Write documentation
- [ ] Create demo video
- [ ] Register on AgentBeats
- [ ] Final testing and submission

---

## Notes and Considerations

### API Key Security
- Green agent generates user-level API keys for Purple agents
- Keys are scoped to single assessments
- Keys are invalidated when assessment ends

### Reproducibility
- All scenarios start from identical initial states
- Response generation uses seeded randomness
- LLM calls for judging use temperature=0

### Testing Strategy
- Unit tests for all helper modules (768+ tests total)
- Integration tests with real LLMs (Ollama and OpenAI)
- Integration tests with mock A2A agents
- End-to-end tests with actual UES instance
- Docker build tests in CI

### Future Enhancements (Post-Competition)
- Multi-agent assessment support
- More complex scenarios
- Additional modalities
- Performance benchmarking tools

---

*Document created: January 28, 2026*
*Last updated: January 29, 2026*
