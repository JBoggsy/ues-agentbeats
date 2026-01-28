# AgentBeats Helper Module

This module provides AgentBeats-specific patterns, message types, and utilities for building Green and Purple agents in the AgentBeats competition. It defines the communication protocol between agents during assessments.

## Overview

The AgentBeats competition uses a turn-based assessment flow where:
- **Green Agent** (evaluator) orchestrates assessments, manages UES, and evaluates Purple agents
- **Purple Agent** (participant) interacts with UES to demonstrate AI assistant capabilities

This module provides:

| Module | Purpose |
|--------|---------|
| `messages.py` | Pydantic models for all messages exchanged between agents |
| `results.py` | Pydantic models for assessment results artifacts |
| `updates.py` | Task update models and `TaskUpdateEmitter` for logging/tracing |
| `config.py` | Configuration models with CLI and environment variable support |

## Installation

The module requires the following dependencies (already configured in `pyproject.toml`):

```toml
[project.dependencies]
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
a2a-sdk = { version = ">=0.3.22", extras = ["http-server"] }
```

Install with:

```bash
uv sync
```

## Quick Start

### Assessment Message Flow

```python
from datetime import datetime, timezone
from src.common.agentbeats import (
    AssessmentStartMessage,
    TurnStartMessage,
    TurnCompleteMessage,
    ActionLogEntry,
    AssessmentCompleteMessage,
    InitialStateSummary,
    EmailSummary,
    CalendarSummary,
    SMSSummary,
    ChatSummary,
)

# Green agent sends AssessmentStartMessage to Purple agent
initial_state = InitialStateSummary(
    email=EmailSummary(total_emails=42, total_threads=15, unread=5, draft_count=2),
    calendar=CalendarSummary(event_count=20, calendar_count=3, events_today=4),
    sms=SMSSummary(total_messages=100, total_conversations=10, unread=3),
    chat=ChatSummary(total_messages=50, conversation_count=1),
)

start_msg = AssessmentStartMessage(
    ues_url="http://localhost:8080",
    api_key="secret-key",
    assessment_instructions="Triage all unread emails by priority.",
    current_time=datetime.now(tz=timezone.utc),
    initial_state_summary=initial_state,
)

# Green agent sends TurnStartMessage at each turn
turn_msg = TurnStartMessage(
    turn_number=1,
    current_time=datetime.now(tz=timezone.utc),
    events_processed=0,
)

# Purple agent responds with TurnCompleteMessage
action = ActionLogEntry(
    timestamp=datetime.now(tz=timezone.utc),
    action="email.archive",
    parameters={"email_id": "msg-123"},
    success=True,
)

complete_msg = TurnCompleteMessage(
    actions=[action],
    notes="Archived spam email.",
    time_step="PT30M",
)

# Green agent sends AssessmentCompleteMessage when done
end_msg = AssessmentCompleteMessage(reason="scenario_complete")
```

### Working with Assessment Results

```python
from src.common.agentbeats import (
    AssessmentResults,
    Scores,
    OverallScore,
    DimensionScore,
    CriterionResult,
    ActionLogEntryWithTurn,
)

# Build criterion results
criterion = CriterionResult(
    criterion_id="email_accuracy",
    name="Email Classification Accuracy",
    dimension="accuracy",
    score=8,
    max_score=10,
    explanation="Correctly classified 8 out of 10 emails.",
    details={"correct": 8, "incorrect": 2},
)

# Build dimension scores
dimension_scores = {
    "accuracy": DimensionScore(score=8, max_score=10),
    "efficiency": DimensionScore(score=7, max_score=10),
}

# Build overall scores
scores = Scores(
    overall=OverallScore(score=15, max_score=20),
    dimensions=dimension_scores,
)

# Build action log
action_entry = ActionLogEntryWithTurn(
    turn=1,
    timestamp=datetime.now(tz=timezone.utc),
    action="email.archive",
    parameters={"email_id": "msg-123"},
    success=True,
)

# Build complete results
results = AssessmentResults(
    assessment_id="assess-123",
    scenario_id="email_triage_basic",
    participant="purple-agent-1",
    status="completed",
    duration_seconds=1234.5,
    turns_taken=10,
    actions_taken=1,
    scores=scores,
    criteria_results=[criterion],
    action_log=[action_entry],
)

# Serialize for A2A artifact
artifact_data = results.model_dump(mode="json")
```

### Emitting Task Updates (Green Agent)

```python
from datetime import datetime, timezone
from src.common.agentbeats import TaskUpdateEmitter

# Create emitter with A2A task context
emitter = TaskUpdateEmitter(task_id="task-123", context_id="ctx-456")

# Emit assessment started update
event = emitter.assessment_started(
    assessment_id="assess-001",
    scenario_id="email_triage_basic",
    participant_url="http://purple:8001",
    start_time=datetime.now(tz=timezone.utc),
)
# event is a TaskStatusUpdateEvent ready to be yielded/enqueued

# Emit turn updates
event = emitter.turn_started(
    turn_number=1,
    current_time=datetime.now(tz=timezone.utc),
    events_pending=5,
)

# Log Purple agent actions
event = emitter.action_observed(
    turn_number=1,
    timestamp=datetime.now(tz=timezone.utc),
    action="email.send",
    parameters={"to": ["alice@example.com"]},
    success=True,
    notes="Responding to Alice's question",
)

# Emit assessment completed
event = emitter.assessment_completed(
    reason="scenario_complete",
    total_turns=10,
    total_actions=25,
    duration_seconds=1234.5,
    overall_score=85,
    max_score=100,
)
```

### Configuration

```python
from src.common.agentbeats import (
    GreenAgentConfig,
    PurpleAgentConfig,
    merge_configs,
    validate_config,
)

# From defaults
green_config = GreenAgentConfig()
print(green_config.port)  # 8000

purple_config = PurpleAgentConfig()
print(purple_config.port)  # 8001

# From CLI arguments
green_config = GreenAgentConfig.from_cli_args([
    "--port", "9000",
    "--ues-url", "http://ues:8080",
    "--verbose-updates",
])

# From environment variables
# Set AGENTBEATS_GREEN_UES_URL=http://ues:8080
# Set AGENTBEATS_PURPLE_MODEL=gpt-4-turbo
green_config = GreenAgentConfig()  # Reads from environment
purple_config = PurpleAgentConfig()

# Merge configurations
updated = merge_configs(green_config, {"port": 9001})

# Validate and get warnings
warnings = validate_config(green_config)
for warning in warnings:
    print(f"Warning: {warning}")
```

### Parsing Messages

```python
from src.common.agentbeats import parse_message, parse_update, parse_result

# Parse incoming message by message_type
data = {
    "message_type": "turn_start",
    "turn_number": 1,
    "current_time": "2026-01-28T12:00:00Z",
    "events_processed": 0,
}
msg = parse_message(data)
# msg is a TurnStartMessage instance

# Parse task update
update_data = {
    "message_type": "update_turn_started",
    "turn_number": 1,
    "current_time": "2026-01-28T12:00:00Z",
    "events_pending": 5,
}
update = parse_update(update_data)
# update is a TurnStartedUpdate instance

# Parse result
result_data = {
    "message_type": "dimension_score",
    "score": 8,
    "max_score": 10,
}
result = parse_result(result_data)
# result is a DimensionScore instance
```

## API Reference

### messages.py

| Model | Description |
|-------|-------------|
| `EmailSummary` | Summary counts for email modality |
| `CalendarSummary` | Summary counts for calendar modality |
| `SMSSummary` | Summary counts for SMS modality |
| `ChatSummary` | Summary counts for chat modality |
| `InitialStateSummary` | Aggregates all modality summaries |
| `AssessmentStartMessage` | Green → Purple at assessment start |
| `TurnStartMessage` | Green → Purple at each turn start |
| `ActionLogEntry` | Single action taken by Purple agent |
| `TurnCompleteMessage` | Purple → Green when turn completes |
| `AssessmentCompleteMessage` | Green → Purple when assessment ends |
| `EarlyCompletionMessage` | Purple → Green to signal early completion |
| `parse_message(data)` | Parse dict into appropriate message type |

### results.py

| Model/Type | Description |
|------------|-------------|
| `ScoringDimension` | Literal type for valid dimension names |
| `ALL_DIMENSIONS` | List of all scoring dimensions |
| `DimensionScore` | Score for a single dimension with percentage calculation |
| `OverallScore` | Total score across all dimensions |
| `Scores` | Container for overall and dimension scores |
| `CriterionResult` | Result for a single evaluation criterion |
| `ActionLogEntryWithTurn` | Action log entry with turn context |
| `AssessmentResults` | Complete assessment results artifact |
| `AssessmentStatus` | Literal type for assessment status values |
| `parse_result(data)` | Parse dict into appropriate result type |

**Scoring Dimensions:**
- `accuracy` - Correctness of actions and responses
- `instruction_following` - Adherence to user instructions
- `efficiency` - Resource usage and action economy
- `safety` - Avoidance of harmful or risky actions
- `politeness` - Appropriate tone and communication style

### updates.py

| Model | Description |
|-------|-------------|
| `TaskUpdateType` | Enum of all update type values |
| `AssessmentStartedUpdate` | Assessment has begun |
| `ScenarioLoadedUpdate` | Scenario configuration loaded |
| `TurnStartedUpdate` | New turn has begun |
| `TurnCompletedUpdate` | Turn has completed |
| `ResponsesGeneratedUpdate` | Character responses generated |
| `SimulationAdvancedUpdate` | UES simulation time advanced |
| `EvaluationStartedUpdate` | Evaluation phase has begun |
| `CriterionEvaluatedUpdate` | Single criterion evaluated |
| `AssessmentCompletedUpdate` | Assessment has ended |
| `ErrorOccurredUpdate` | An error occurred |
| `ActionObservedUpdate` | Logs a Purple agent action |
| `TaskUpdateEmitter` | Helper class for emitting updates |
| `parse_update(data)` | Parse dict into appropriate update type |

**TaskUpdateEmitter Methods:**
- `assessment_started(...)` - Emit assessment started update
- `scenario_loaded(...)` - Emit scenario loaded update
- `turn_started(...)` - Emit turn started update
- `turn_completed(...)` - Emit turn completed update
- `responses_generated(...)` - Emit responses generated update
- `simulation_advanced(...)` - Emit simulation advanced update
- `evaluation_started(...)` - Emit evaluation started update
- `criterion_evaluated(...)` - Emit criterion evaluated update
- `assessment_completed(...)` - Emit assessment completed update (final)
- `error_occurred(...)` - Emit error occurred update
- `action_observed(...)` - Log a Purple agent action

### config.py

| Class/Function | Description |
|----------------|-------------|
| `AgentBeatsConfig` | Base configuration for all agents |
| `GreenAgentConfig` | Configuration for Green agents (evaluators) |
| `PurpleAgentConfig` | Configuration for Purple agents (participants) |
| `merge_configs(base, overrides)` | Create new config with overrides applied |
| `validate_config(config)` | Validate config and return warnings |

**AgentBeatsConfig Fields:**
- `host` - Server host address (default: "0.0.0.0")
- `port` - Server port (default: 8000)
- `card_url` - URL where agent card is accessible
- `log_level` - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**GreenAgentConfig Fields** (extends AgentBeatsConfig):
- `port` - Server port (default: 8000)
- `verbose_updates` - Emit detailed task updates (default: True)
- `ues_url` - UES instance URL (default: "http://localhost:8080")
- `ues_proctor_api_key` - API key for proctor-level UES access
- `scenarios_dir` - Directory containing scenarios (default: "scenarios")
- `default_max_turns` - Default max turns (default: 100)
- `default_turn_timeout` - Default turn timeout in seconds (default: 300.0)
- `response_generator_model` - LLM for character responses (default: "gpt-4o")
- `evaluation_model` - LLM for evaluation (default: "gpt-4o")

**PurpleAgentConfig Fields** (extends AgentBeatsConfig):
- `port` - Server port (default: 8001)
- `model` - LLM model for reasoning (default: "gpt-4o")
- `max_actions_per_turn` - Max actions per turn (default: 50)
- `temperature` - LLM temperature 0.0-2.0 (default: 0.7)
- `enable_reflection` - Enable self-reflection (default: True)
- `action_delay` - Delay between actions in seconds (default: 0.0)

**Configuration Methods:**
- `from_cli_args(args, **overrides)` - Create config from CLI arguments
- `effective_card_url` - Property returning configured or default card URL

**Environment Variable Prefixes:**
- Base: `AGENTBEATS_*`
- Green: `AGENTBEATS_GREEN_*`
- Purple: `AGENTBEATS_PURPLE_*`

## Design Notes

### Message Type Field

All message, result, and update models include a `message_type` field with a fixed literal value. This enables easy parsing:

```python
# Instead of inferring type from fields present:
if "turn_number" in data and "events_processed" in data:
    msg = TurnStartMessage(**data)

# Just check message_type:
msg = parse_message(data)  # Automatically dispatches by message_type
```

The `message_type` is set as a class-level default and is frozen (immutable).

### Score Validation

Score models include validation to ensure consistency:

- `DimensionScore` and `OverallScore` validate that `score <= max_score`
- `Scores` validates that `overall.score == sum(dimensions[*].score)`
- `AssessmentResults` validates that `actions_taken == len(action_log)`
- `CriterionResult` validates that `dimension` exists in the parent `Scores`

### Update Emission Pattern

All task updates are emitted by the Green agent. Purple agents report actions via `TurnCompleteMessage`, and the Green agent logs each action as an `ActionObservedUpdate`. This:

- Simplifies Purple agent implementation
- Ensures consistent logging under Green agent control
- Provides a single source of truth for assessment traces

### Configuration Precedence

Configuration values are resolved in this order (highest precedence first):

1. Explicit constructor arguments
2. CLI arguments (via `from_cli_args`)
3. Environment variables (automatic via pydantic-settings)
4. Default values

## Testing

Run the module tests with:

```bash
uv run pytest tests/common/agentbeats/ -v
```

The module has comprehensive test coverage (237 tests total):
- `test_messages.py` - 46 tests for message models and parsing
- `test_results.py` - 48 tests for result models and validation
- `test_updates.py` - 60 tests for update models and emitter
- `test_config.py` - 83 tests for configuration and CLI parsing

## Related Resources

- [A2A Protocol Specification](https://a2a-protocol.org/latest/)
- [AgentBeats Competition Documentation](https://docs.agentbeats.org/)
- [Implementation Plan](../../../docs/IMPLEMENTATION_PLAN.md)
- [Assessment Flow](../../../docs/ASSESSMENT_FLOW.md)
