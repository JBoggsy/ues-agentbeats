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
class A2AClient:
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

## Phase 2: Common AgentBeats Helper Code

**Location**: `src/common/agentbeats/`

AgentBeats-specific patterns and message types used by both Green and Purple agents.

### 2.1 Assessment Messages (`messages.py`)

Pydantic models for AgentBeats-specific message payloads.

```python
from pydantic import BaseModel
from datetime import datetime

# NOTE: Modality summary fields are derived from UES `get_snapshot()` outputs.
# Different modalities have different available fields:
#   - Email: total_emails, unread (computed from folders[*].unread_count)
#   - Calendar: event_count, events_today (requires get_compact_snapshot)
#   - SMS: total_messages, unread_total
#   - Chat: total_message_count (no unread concept)

class EmailSummary(BaseModel):
    """Summary counts for email modality."""
    total_emails: int
    total_threads: int
    unread: int
    draft_count: int

class CalendarSummary(BaseModel):
    """Summary counts for calendar modality."""
    event_count: int
    calendar_count: int
    events_today: int  # Requires get_compact_snapshot(current_time)

class SMSSummary(BaseModel):
    """Summary counts for SMS modality."""
    total_messages: int
    total_conversations: int
    unread: int

class ChatSummary(BaseModel):
    """Summary counts for chat modality."""
    total_messages: int
    conversation_count: int
    # Note: Chat has no "unread" concept (user-assistant pairs)

class InitialStateSummary(BaseModel):
    """Summary of initial UES state, derived from modality snapshots."""
    email: EmailSummary
    calendar: CalendarSummary
    sms: SMSSummary
    chat: ChatSummary

class AssessmentStartMessage(BaseModel):
    """Message sent from Green to Purple at assessment start."""
    ues_url: str
    api_key: str
    assessment_instructions: str
    current_time: datetime
    initial_state_summary: InitialStateSummary

class TurnStartMessage(BaseModel):
    """Message sent from Green to Purple at turn start."""
    turn_number: int
    current_time: datetime
    events_processed: int

class ActionLogEntry(BaseModel):
    """Single action taken by Purple agent during a turn.
    
    Purple agent reports these in TurnCompleteMessage.
    Green agent adds turn number when aggregating into assessment results.
    """
    timestamp: datetime
    action: str  # e.g., "email.send", "calendar.create", "sms.reply"
    parameters: dict  # Action-specific parameters
    success: bool
    error_message: str | None = None  # Populated if success=False

class TurnCompleteMessage(BaseModel):
    """Message sent from Purple to Green when turn completes.
    
    Purple agent reports all actions taken during the turn. The Green agent
    uses this to build the assessment action log directly, rather than
    reconstructing it from UES event history.
    """
    actions: list[ActionLogEntry]
    notes: str | None = None  # Optional reasoning/transparency
    time_step: str = "PT1H"  # ISO 8601 duration

class AssessmentCompleteMessage(BaseModel):
    """Message sent from Green to Purple when assessment ends."""
    reason: Literal[
        "scenario_complete",
        "early_completion",
        "timeout",
        "error",
    ]

class EarlyCompletionMessage(BaseModel):
    """Message sent from Purple to Green to signal early completion."""
    reason: str | None = None
```

**Tests**: Serialization/deserialization, validation.

### 2.2 Assessment Results (`results.py`)

Pydantic models for assessment results (artifacts).

```python
class CriterionResult(BaseModel):
    """Result for a single evaluation criterion.
    
    The `details` field carries structured information about individual
    evaluations, e.g., politeness scores for each email in an "email
    politeness" criterion.
    """
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
    score: int
    max_score: int

class OverallScore(BaseModel):
    """Overall assessment score."""
    score: int
    max_score: int

class Scores(BaseModel):
    """All scores for an assessment."""
    overall: OverallScore
    dimensions: dict[str, DimensionScore]

class ActionLogEntryWithTurn(BaseModel):
    """ActionLogEntry with turn number added by Green agent.
    
    This extends the ActionLogEntry from messages.py with the turn context
    that the Green agent adds when building the assessment results.
    """
    turn: int
    timestamp: datetime
    action: str
    parameters: dict
    success: bool
    error_message: str | None = None

class AssessmentResults(BaseModel):
    """Complete assessment results."""
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

**Tests**: Score calculation consistency, JSON schema validation.

### 2.3 Task Update Helpers (`updates.py`)

Helpers for emitting AgentBeats-style task updates (logs).

```python
class TaskUpdateType(str, Enum):
    """Types of task updates for logging."""
    ASSESSMENT_STARTED = "log_assessment_started"
    SCENARIO_LOADED = "log_scenario_loaded"
    TURN_STARTED = "log_turn_started"
    TURN_COMPLETED = "log_turn_completed"
    RESPONSES_GENERATED = "log_responses_generated"
    SIMULATION_ADVANCED = "log_simulation_advanced"
    ASSESSMENT_COMPLETE = "log_assessment_complete"

class TaskUpdateEmitter:
    """Helper for emitting structured task updates."""
    
    def __init__(self, event_queue: EventQueue): ...
    
    async def emit(
        self,
        update_type: TaskUpdateType,
        message: str,
        details: dict | None = None,
    ) -> None: ...
    
    async def assessment_started(
        self,
        assessment_id: str,
        scenario_id: str,
        participant: str,
        user_prompt: str,
    ) -> None: ...
    
    async def turn_completed(
        self,
        turn: int,
        actions_taken: int,
    ) -> None: ...
    
    # ... more convenience methods
```

**Tests**: Update format validation, async behavior.

### 2.4 Configuration (`config.py`)

Shared configuration patterns.

```python
class AgentBeatsConfig(BaseModel):
    """Base configuration for AgentBeats agents."""
    host: str = "0.0.0.0"
    port: int = 8000
    card_url: str | None = None  # For Dockerized agents
    
    @classmethod
    def from_cli_args(cls) -> Self:
        """Parse from command line arguments."""
        ...

class GreenAgentConfig(AgentBeatsConfig):
    """Configuration specific to Green agents."""
    ues_url: str = "http://localhost:8080"
    proctor_api_key: str | None = None
    
    # LLM configuration
    judging_model: str = "gpt-4o"
    response_model: str = "gpt-4o-mini"
    
    # Assessment defaults
    default_turn_timeout: int = 300  # seconds
    default_time_step: str = "PT1H"
    verbose_updates: bool = True

class PurpleAgentConfig(AgentBeatsConfig):
    """Configuration specific to Purple agents."""
    # Model to use for reasoning
    model: str = "gpt-4o"
```

**Tests**: CLI argument parsing, environment variable loading.

---

## Phase 3: Green Agent Implementation

**Location**: `src/green/`

The Green Agent orchestrates assessments, manages UES, and evaluates Purple agents.

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Green Agent                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Server    │  │  Executor   │  │  Scenario   │             │
│  │  (A2A)      │──│  (A2A)      │──│  Manager    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                │                     │
│                          ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Assessment  │  │  Response   │  │  Criteria   │             │
│  │   Loop      │◀─│  Generator  │  │   Judge     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │              UES Client (proctor-level)         │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Scenario Management (`scenarios/`)

#### 3.2.1 Scenario Schema (`schema.py`)

```python
class CharacterProfile(BaseModel):
    """Profile for a simulated character."""
    name: str
    role: str  # e.g., "Friend", "Vendor", "Boss"
    personality: str
    email: str | None = None
    phone: str | None = None
    response_timing: ResponseTiming
    special_instructions: str | None = None
    # Character-specific config (e.g., pricing for vendors)
    config: dict | None = None

class ResponseTiming(BaseModel):
    """Timing configuration for character responses."""
    base_delay: str  # ISO 8601 duration
    variance: str  # ISO 8601 duration

class EvaluationCriterion(BaseModel):
    """Definition of a single evaluation criterion.
    
    Supports two evaluation modes:
    1. Programmatic: Set `evaluator_id` to reference a registered function
    2. LLM-based: Set `evaluation_prompt` for LLM judging
    
    At least one of `evaluator_id` or `evaluation_prompt` must be provided.
    """
    criterion_id: str
    name: str
    description: str
    dimension: Literal[
        "accuracy",
        "instruction_following",
        "efficiency",
        "safety",
        "politeness",
    ]
    max_score: int
    evaluator_id: str | None = None  # Reference to programmatic evaluator
    evaluation_prompt: str | None = None  # LLM prompt for evaluation
    params: dict | None = None  # Parameters passed to evaluator function
    
class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""
    scenario_id: str
    name: str
    description: str
    
    # Timing
    start_time: datetime
    end_time: datetime
    default_time_step: str
    
    # User prompt (delivered via chat)
    user_prompt: str
    
    # Characters
    characters: dict[str, CharacterProfile]
    
    # Initial UES state (to be imported)
    initial_state: dict  # UES scenario export format
    
    # Evaluation criteria
    criteria: list[EvaluationCriterion]
    
    # Termination conditions
    early_completion_conditions: list[str] | None = None
```

#### 3.2.2 Scenario Loader (`loader.py`)

```python
class ScenarioManager:
    """Manages loading and accessing scenarios."""
    
    def __init__(self, scenarios_dir: Path): ...
    
    def list_scenarios(self) -> list[str]: ...
    def load_scenario(self, scenario_id: str) -> ScenarioConfig: ...
    def validate_scenario(self, config: ScenarioConfig) -> list[str]: ...
```

**Tests**: Schema validation, loading from YAML/JSON files.

### 3.3 Assessment Loop (`assessment/`)

#### 3.3.1 Assessment Orchestrator (`orchestrator.py`)

```python
class AssessmentOrchestrator:
    """Orchestrates a single assessment run."""
    
    def __init__(
        self,
        ues_client: UESClient,  # Proctor-level access
        purple_client: A2AClient,
        scenario: ScenarioConfig,
        response_generator: ResponseGenerator,
        config: GreenAgentConfig,
        update_emitter: TaskUpdateEmitter,
    ): ...
    
    async def run(self) -> AssessmentResults:
        """Execute the full assessment loop."""
        ...
    
    async def setup_environment(self) -> str:
        """Initialize UES and return user-level API key."""
        ...
    
    async def run_turn(self, turn: int) -> TurnResult:
        """Execute a single turn of the assessment."""
        ...
    
    async def evaluate(self) -> AssessmentResults:
        """Evaluate Purple agent performance."""
        ...
```

#### 3.3.2 Turn Handler (`turn_handler.py`)

```python
class TurnResult(BaseModel):
    """Result of a single turn."""
    turn_number: int
    actions_taken: int
    notes: str | None
    time_step: str
    events_processed: int
    early_completion: bool = False

class TurnHandler:
    """Handles the logic for a single assessment turn."""
    
    async def send_turn_start(
        self,
        purple_client: A2AClient,
        task_id: str,
        current_time: datetime,
        events_processed: int,
    ) -> None: ...
    
    async def wait_for_turn_complete(
        self,
        purple_client: A2AClient,
        task_id: str,
        timeout: int,
    ) -> TurnCompleteMessage | EarlyCompletionMessage: ...
    
    async def process_time_advance(
        self,
        ues_client: UESClient,
        time_step: str,
    ) -> int:  # Returns events_processed
        ...
```

### 3.4 Response Generation (`response_generator/`)

#### 3.4.1 Response Generator (`generator.py`)

```python
class ResponseGenerator:
    """Generates character responses to Purple agent actions."""
    
    def __init__(
        self,
        ues_client: UESClient,  # Proctor-level
        characters: dict[str, CharacterProfile],
        llm: BaseChatModel,  # LangChain LLM
        seed: int | None = None,
    ): ...
    
    async def process_turn_actions(
        self,
        actions: list[ActionLogEntry],
    ) -> list[ScheduledResponse]:
        """Analyze Purple's turn actions and generate character responses.
        
        Scans actions for outgoing messages (email.send, sms.send, etc.)
        and generates appropriate character responses.
        """
        ...
    
    async def should_respond(
        self,
        message: str,
        character: CharacterProfile,
        thread_history: list[dict],
    ) -> bool:
        """Determine if a character response is warranted."""
        ...
    
    async def generate_response(
        self,
        character: CharacterProfile,
        context: ResponseContext,
    ) -> str:
        """Generate an in-character response."""
        ...
    
    async def schedule_responses(
        self,
        responses: list[ScheduledResponse],
    ) -> None:
        """Schedule responses as UES events."""
        ...
```

#### 3.4.2 Character Context (`context.py`)

```python
class ResponseContext(BaseModel):
    """Context provided to response generation."""
    character: CharacterProfile
    modality: Literal["email", "sms", "calendar"]
    incoming_message: str
    thread_history: list[dict]
    current_time: datetime
    
class ScheduledResponse(BaseModel):
    """A response to be scheduled in UES."""
    character_id: str
    modality: Literal["email", "sms", "calendar"]
    response_content: str | dict  # str for message, dict for calendar RSVP
    scheduled_time: datetime
```

### 3.5 Evaluation (`evaluation/`)

> **Design Note**: This module adopts patterns from the UES `agent_testing` framework
> while customizing orchestration for AgentBeats' A2A turn-based flow. See
> `docs/UES_TESTING_HARNESS_ANALYSIS.md` for the full analysis.
>
> **Adopted from UES**: `EvalResult` structure, `EvalContext` pattern, evaluator function signature
> **Custom for AgentBeats**: `CriteriaJudge` orchestration, LLM-based evaluation, dimension aggregation

#### 3.5.1 Evaluation Context (`context.py`)

Extends the UES `EvalContext` pattern with AgentBeats-specific data.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from ues.client import AsyncUESClient

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
    action_log: list["ActionLogEntryWithTurn"]
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
```

#### 3.5.2 Criteria Judge (`judge.py`)

Orchestrates evaluation using both programmatic and LLM-based evaluators.

```python
from typing import Callable, Awaitable

# Evaluator function signature (follows UES pattern)
EvaluatorFunc = Callable[
    [AgentBeatsEvalContext, dict],  # (context, params)
    Awaitable[EvalResult]
]

class CriteriaJudge:
    """Evaluates Purple agent performance against criteria.
    
    Supports two evaluation modes:
    1. Programmatic evaluators - registered functions that check state
    2. LLM-based evaluators - use evaluation_prompt from criterion
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        criteria: list[EvaluationCriterion],
    ):
        self.llm = llm
        self.criteria = criteria
        self._evaluators: dict[str, EvaluatorFunc] = {}
        self._register_builtin_evaluators()
    
    def register_evaluator(
        self,
        evaluator_id: str,
        func: EvaluatorFunc,
    ) -> None:
        """Register a programmatic evaluator function."""
        self._evaluators[evaluator_id] = func
    
    async def evaluate_all(
        self,
        ctx: AgentBeatsEvalContext,
    ) -> list[CriterionResult]:
        """Evaluate all criteria and return results."""
        results = []
        for criterion in self.criteria:
            result = await self._evaluate_criterion(ctx, criterion)
            results.append(result)
        return results
    
    async def _evaluate_criterion(
        self,
        ctx: AgentBeatsEvalContext,
        criterion: EvaluationCriterion,
    ) -> CriterionResult:
        """Evaluate a single criterion."""
        # Try registered programmatic evaluator first
        if criterion.evaluator_id and criterion.evaluator_id in self._evaluators:
            evaluator = self._evaluators[criterion.evaluator_id]
            eval_result = await evaluator(ctx, criterion.params or {})
        
        # Fall back to LLM-based evaluation
        elif criterion.evaluation_prompt:
            eval_result = await self._llm_evaluate(ctx, criterion)
        
        else:
            raise ValueError(
                f"Criterion '{criterion.criterion_id}' has no evaluator_id "
                "or evaluation_prompt"
            )
        
        return CriterionResult(
            criterion_id=criterion.criterion_id,
            name=criterion.name,
            dimension=criterion.dimension,
            score=int(eval_result.score),
            max_score=criterion.max_score,
            explanation=eval_result.explanation,
            details=eval_result.details,
        )
    
    async def _llm_evaluate(
        self,
        ctx: AgentBeatsEvalContext,
        criterion: EvaluationCriterion,
    ) -> EvalResult:
        """Use LLM to evaluate a criterion based on evaluation_prompt."""
        # Build context for the prompt
        # Call LLM with structured output
        # Return EvalResult
        ...
```

#### 3.5.3 Action Log Builder (`action_log.py`)

The action log is built directly from Purple agent's `TurnCompleteMessage` reports,
rather than being reconstructed from UES event history. This approach:
- Ensures Purple agent is responsible for accurate action reporting
- Simplifies Green agent implementation (no need to parse UES events)
- Gives Purple agent control over action naming/categorization

```python
class ActionLogBuilder:
    """Builds assessment action log from Purple agent turn reports."""
    
    def __init__(self) -> None:
        self._entries: list[ActionLogEntryWithTurn] = []
        self._current_turn: int = 0
    
    def start_turn(self, turn_number: int) -> None:
        """Mark the start of a new turn."""
        self._current_turn = turn_number
    
    def add_turn_actions(self, actions: list[ActionLogEntry]) -> None:
        """Add actions from a TurnCompleteMessage to the log.
        
        Converts ActionLogEntry (without turn) to ActionLogEntryWithTurn
        by adding the current turn number.
        """
        for action in actions:
            self._entries.append(
                ActionLogEntryWithTurn(
                    turn=self._current_turn,
                    timestamp=action.timestamp,
                    action=action.action,
                    parameters=action.parameters,
                    success=action.success,
                    error_message=action.error_message,
                )
            )
    
    def get_log(self) -> list[ActionLogEntryWithTurn]:
        """Return the complete action log."""
        return self._entries.copy()
    
    def get_actions_for_turn(self, turn: int) -> list[ActionLogEntryWithTurn]:
        """Get all actions for a specific turn."""
        return [e for e in self._entries if e.turn == turn]
    
    def categorize_actions(self) -> dict[str, list[ActionLogEntryWithTurn]]:
        """Group actions by modality/type (e.g., 'email', 'calendar')."""
        categorized: dict[str, list[ActionLogEntryWithTurn]] = {}
        for entry in self._entries:
            # Extract modality from action string (e.g., "email.send" -> "email")
            modality = entry.action.split(".")[0] if "." in entry.action else "other"
            if modality not in categorized:
                categorized[modality] = []
            categorized[modality].append(entry)
        return categorized
```

### 3.6 Green Agent Executor (`executor.py`)

The main A2A executor that handles assessment requests.

```python
class GreenAgentExecutor(AgentExecutor):
    """A2A executor for the Green Agent."""
    
    def __init__(
        self,
        scenario_manager: ScenarioManager,
        config: GreenAgentConfig,
    ): ...
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle incoming assessment request."""
        # 1. Parse assessment request
        # 2. Load scenario
        # 3. Create orchestrator
        # 4. Run assessment
        # 5. Emit results as artifact
        ...
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Handle cancellation request."""
        ...
```

### 3.7 LLM Configuration (`llm_config.py`)

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class LLMFactory:
    """Factory for creating LangChain LLM instances."""
    
    @staticmethod
    def create(
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> BaseChatModel:
        """Create an LLM instance based on model string."""
        # Support: gpt-4o, gpt-4o-mini, claude-3-opus, etc.
        ...

class GreenAgentLLMs:
    """Container for Green Agent LLM instances."""
    
    def __init__(self, config: GreenAgentConfig):
        self.judging_llm = LLMFactory.create(
            config.judging_model,
            temperature=0.0,  # Deterministic for judging
        )
        self.response_llm = LLMFactory.create(
            config.response_model,
            temperature=0.7,
            seed=config.seed,
        )
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
│   │   ├── __main__.py            # Entry point
│   │   ├── executor.py            # A2A executor
│   │   ├── llm_config.py          # LLM configuration
│   │   ├── scenarios/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py          # Scenario schema
│   │   │   └── loader.py          # Scenario loader
│   │   ├── assessment/
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py    # Assessment orchestrator
│   │   │   └── turn_handler.py    # Turn handling
│   │   ├── response_generator/
│   │   │   ├── __init__.py
│   │   │   ├── generator.py       # Response generation
│   │   │   └── context.py         # Response context
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── judge.py           # Criteria judge
│   │       └── action_log.py      # Action log builder
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
│       ├── scenario.yaml          # Scenario config
│       └── initial_state.json     # UES state export
│
├── tests/
│   ├── common/
│   │   ├── a2a/
│   │   └── agentbeats/
│   ├── green/
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

### Week 1: AgentBeats Common (Days 4-5)
- [ ] Implement `src/common/agentbeats/` module
- [ ] Define message types and results schema
- [ ] Write tests for serialization/validation

### Week 2: Green Agent Core (Days 6-10)
- [ ] Implement scenario schema and loader
- [ ] Implement assessment orchestrator
- [ ] Implement turn handler
- [ ] Basic response generator (no LLM yet)
- [ ] Basic criteria judge (no LLM yet)
- [ ] Integration tests with mock UES

### Week 3: Green Agent LLM Integration (Days 11-14)
- [ ] Integrate LangChain for response generation
- [ ] Integrate LangChain for criteria judging
- [ ] Implement LLM configuration/factory
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
- Unit tests for all helper modules
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
*For the AgentX AgentBeats Competition - Phase 1 deadline: January 31, 2026*
