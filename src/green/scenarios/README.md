# Green Agent Scenarios Module

This module provides scenario management for the Green Agent, including schema definitions and loading utilities for assessment scenarios.

## Overview

Scenarios define everything needed to run an assessment:

- **Metadata**: ID, name, description
- **Timing**: Start/end times, default time step
- **User Prompt**: Scenario-specific task delivered to Purple via chat at assessment start
- **Characters**: Simulated contacts with personalities and response patterns
- **Initial State**: UES environment state (emails, calendar, SMS, etc.)
- **Evaluation Criteria**: How to score Purple agent performance
- **Early Completion**: Optional conditions to end early

## Module Structure

```
src/green/scenarios/
├── __init__.py      # Public API exports
├── schema.py        # Pydantic models and evaluator types
├── loader.py        # Scenario loading and management utilities
└── README.md        # This file
```

## Key Classes

### Schema Models (`schema.py`)

- **`ResponseTiming`**: Timing configuration for character responses (base delay + variance)
- **`CharacterProfile`**: Profile for a simulated character (name, relationships, personality, contact methods)
- **`EvaluationCriterion`**: Single evaluation criterion (programmatic or LLM-based)
- **`ScenarioConfig`**: Complete scenario configuration

### Evaluator Types (`schema.py`)

- **`EvalResult`**: Return type for programmatic evaluators (score, max_score, explanation, details)
- **`AgentBeatsEvalContext`**: Context passed to evaluators during assessment
- **`EvaluatorFunc`**: Type alias for evaluator function signature
- **`EvaluatorRegistry`**: Type alias for evaluator ID -> function mapping

### Loader Utilities (`loader.py`)

- **`ScenarioManager`**: High-level interface for discovering and loading scenarios
- **`ScenarioLoader`**: Low-level loader for individual scenario files
- **`ScenarioNotFoundError`**: Raised when a scenario doesn't exist
- **`ScenarioValidationError`**: Raised when a scenario fails validation
- **`EvaluatorLoadError`**: Raised when evaluator loading fails

## Usage Examples

### Loading a Scenario

```python
from pathlib import Path
from src.green.scenarios import ScenarioManager

# Initialize manager with scenarios directory
manager = ScenarioManager(Path("scenarios"))

# List available scenarios
scenarios = manager.list_scenarios()
print(scenarios)  # ['email_triage_basic', 'calendar_scheduling']

# Load a specific scenario
config = manager.get_scenario("email_triage_basic")
print(config.name)  # "Basic Email Triage"

# Load evaluators for the scenario
evaluators = manager.get_evaluators("email_triage_basic")
print(list(evaluators.keys()))  # ['check_urgent_email_responses', ...]
```

Note: `scenario_id` must match the scenario directory name (for example, directory
`scenarios/email_triage_basic/` must contain `"scenario_id": "email_triage_basic"`).

### Accessing Scenario Data

```python
# Get scenario timing
print(config.duration)  # timedelta(hours=8)
print(config.default_time_step_timedelta)  # timedelta(hours=1)

# Access the user being assisted (the main character)
user_profile = config.get_user_character_profile()
print(f"User: {user_profile.name} ({user_profile.email})")

# Access all characters
for char_id, character in config.characters.items():
    print(f"{char_id}: {character.name} (relationships: {character.relationships})")

# Get criteria by dimension
accuracy_criteria = config.get_criteria_by_dimension("accuracy")
print(f"Accuracy criteria: {len(accuracy_criteria)}")

# Calculate scores
print(f"Total max score: {config.get_total_max_score()}")
print(f"Max by dimension: {config.get_max_score_by_dimension()}")
```

### Validating Scenarios

```python
# Validate a scenario (returns warnings, not errors)
warnings = manager.validate_scenario(config)
for warning in warnings:
    print(f"Warning: {warning}")
```

## Scenario File Format

### Prompt Delivery Contract

At assessment start, Green applies this fixed behavior:

1. Sends a **fixed** `assessment_start.assessment_instructions` message telling Purple
  to act as a personal assistant and read specific tasks from chat.
2. Injects `scenario.user_prompt` as an **immediate chat message from the user** before
  Purple's first turn.

Scenario authors should treat `user_prompt` as the canonical scenario task request that
will be delivered through the chat modality by Green.

Scenarios are stored as JSON files in subdirectories of the `scenarios/` directory:

```
scenarios/
└── email_triage_basic/
    ├── scenario.json      # Main scenario configuration
    ├── initial_state.json # UES state (optional, can be embedded)
    └── evaluators.py      # Programmatic evaluators (optional)
```

### scenario.json Structure

```json
{
  "scenario_id": "email_triage_basic",
  "name": "Basic Email Triage",
  "description": "Handle incoming emails appropriately.",
  "start_time": "2026-01-28T09:00:00Z",
  "end_time": "2026-01-28T17:00:00Z",
  "default_time_step": "PT1H",
  "user_prompt": "Please triage my inbox...",
  "user_character": "alex",
  "characters": {
    "alex": {
      "name": "Alex Thompson",
      "relationships": {},
      "personality": "A busy professional relying on their AI assistant.",
      "email": "alex@example.com",
      "response_timing": {
        "base_delay": "PT5M",
        "variance": "PT2M"
      }
    },
    "alice": {
      "name": "Alice Chen",
      "relationships": {
        "Alex Thompson": "direct report"
      },
      "personality": "Professional but friendly.",
      "email": "alice@example.com",
      "response_timing": {
        "base_delay": "PT15M",
        "variance": "PT5M"
      }
    }
  },
  "initial_state": { ... },  // Or a file path like "initial_state.json"
  "criteria": [
    {
      "criterion_id": "accuracy_1",
      "name": "Response Accuracy",
      "description": "Evaluates response accuracy.",
      "dimension": "accuracy",
      "max_score": 10,
      "evaluator_id": "check_responses"
    }
  ],
  "early_completion_conditions": ["all_emails_processed"]
}
```

### User Character

The `user_character` field is required and must reference a key in the `characters` dictionary. This designates which character represents the user that the Purple agent is assisting.

The user character typically has an empty `relationships` dict (since relationships are defined from other characters' perspectives). The user character's profile defines their email, phone, and other contact information that the Purple agent will use when acting on their behalf.

```python
# Access the user character profile in code
user_profile = scenario_config.get_user_character_profile()
print(f"Assisting: {user_profile.name} ({user_profile.email})")
```

### Character Validation Rules

Each character must define at least one contact method:

- `email`, or
- `phone`, or
- both.

Additional validation enforced by `ScenarioConfig`:

- Character emails must be unique (if present)
- Character phone numbers must be unique (if present)
- `user_character` must reference a key in `characters`

### Initial State Options

The `initial_state` field can be:

1. **Embedded**: A JSON object directly in scenario.json
2. **Default file**: Omit the field, and `initial_state.json` will be loaded
3. **Custom path**: A string path relative to the scenario directory

If `initial_state` is omitted and `initial_state.json` does not exist, loading fails.

#### Initial State Format (UES Export Format)

The initial state must use the UES `/scenario/import/full` endpoint format with the following structure:

```json
{
  "metadata": {
    "ues_version": "0.2.1",
    "scenario_version": "1",
    "created_at": "2026-01-28T09:00:00+00:00",
    "description": "Optional scenario description"
  },
  "environment": {
    "time_state": {
      "current_time": "2026-01-28T09:00:00+00:00",
      "time_scale": 1.0,
      "is_paused": false,
      "auto_advance": false,
      "last_wall_time_update": "2026-01-28T09:00:00+00:00"
    },
    "modality_states": {
      "email": {
        "modality_type": "email",
        "last_updated": "2026-01-28T09:00:00+00:00",
        "update_count": 0,
        "user_email_address": "user@example.com",
        "emails": {},
        "threads": {},
        "folders": {},
        "labels": {},
        "drafts": {}
      },
      "calendar": { ... },
      "sms": { ... },
      "chat": { ... }
    }
  },
  "events": {
    "events": []
  }
}
```

**Required sections:**
- `metadata` - Version and creation metadata
- `environment.time_state` - Simulation time configuration
- `environment.modality_states` - State for each modality (email, calendar, sms, chat)
- `events` - Event log (typically empty at start)

Each modality in `modality_states` must include a `modality_type` field matching its key name.

## Time Formats

All durations use ISO 8601 format:

- `PT1H` - 1 hour
- `PT30M` - 30 minutes
- `PT1H30M` - 1 hour 30 minutes
- `P1D` - 1 day
- `P1DT12H` - 1 day 12 hours

All datetimes must be timezone-aware (include `Z` or offset).

## Scoring Dimensions

Criteria must belong to one of these dimensions:

- `accuracy` - Correctness of actions and responses
- `instruction_following` - Adherence to user instructions
- `efficiency` - Resource usage and action economy
- `safety` - Avoidance of harmful actions
- `politeness` - Appropriate tone and communication

## Evaluation Methods

Criteria support two evaluation methods:

1. **Programmatic** (`evaluator_id`): Reference an evaluator function in `evaluators.py`
2. **LLM-based** (`evaluation_prompt`): Prompt for LLM judging

At least one method must be specified. Both can be used together.

Important: `assessment_start.assessment_instructions` is fixed and shared across all
scenarios. Scenario-specific instructions should be authored in `user_prompt`, which
Green injects into chat at assessment start.

## Programmatic Evaluators

Scenarios can include an `evaluators.py` file with programmatic evaluation functions:

```python
# scenarios/my_scenario/evaluators.py
from src.green.scenarios.schema import AgentBeatsEvalContext, EvalResult

async def check_response_accuracy(
    ctx: AgentBeatsEvalContext,
    params: dict,
) -> EvalResult:
    """Check that responses are accurate."""
  # Access UES state via ctx.client
    # Access action history via ctx.action_log
    # Access scenario config via ctx.scenario_config
    # Use params from the criterion definition
    
    correct_count = 0
    total_count = len(ctx.action_log)
    # ... evaluate actions ...
    
    return EvalResult(
        score=float(correct_count),
        max_score=float(total_count) if total_count else 1.0,
        explanation=f"Correctly handled {correct_count} of {total_count} items",
        details={"correct": correct_count, "total": total_count},
    )
```

### Evaluator Requirements

- Must be `async def` (async function)
- Must accept at least 2 parameters (first two should be `ctx`, `params`)
- Must not start with underscore (private functions are excluded)
- Function name becomes the `evaluator_id` referenced in criteria

Runtime expectation (not enforced at import-time): evaluators should return
`EvalResult` so the `CriteriaJudge` can scale scores correctly.

### Loading and Validating Evaluators

```python
from src.green.scenarios import ScenarioManager

manager = ScenarioManager(Path("scenarios"))
config = manager.get_scenario("email_triage_basic")
evaluators = manager.get_evaluators("email_triage_basic")

# Validate all required evaluators are present
warnings = manager.validate_evaluators(config, evaluators)
if warnings:
    print("Missing evaluators:", warnings)
```

## Error Handling

```python
from src.green.scenarios import (
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
    EvaluatorLoadError,
)

manager = ScenarioManager(Path("scenarios"))

try:
    config = manager.get_scenario("nonexistent")
except ScenarioNotFoundError as e:
    print(f"Scenario not found: {e.scenario_id}")
except ScenarioValidationError as e:
    print(f"Validation failed: {e.errors}")

try:
    evaluators = manager.get_evaluators("bad_evaluators")
except EvaluatorLoadError as e:
    print(f"Failed to load evaluators: {e.reason}")
```

## Caching

The `ScenarioManager` caches loaded scenarios and evaluators. Use these methods to manage the cache:

```python
# Check cached scenarios and evaluators
cached_scenarios = manager.get_cached_scenarios()
cached_evaluators = manager.get_cached_evaluators()

# Reload a scenario (bypass cache)
config = manager.reload_scenario("email_triage_basic")

# Reload evaluators (bypass cache)
evaluators = manager.reload_evaluators("email_triage_basic")

# Clear all caches
manager.clear_cache()

# Load without caching
config = manager.get_scenario("test", use_cache=False)
evaluators = manager.get_evaluators("test", use_cache=False)
```

## Testing

Run the tests with:

```bash
uv run pytest tests/green/scenarios/ -v
```

Tests cover:
- Model creation and validation
- Field validation (patterns, constraints)
- Cross-field validation (time ordering, uniqueness)
- Serialization/deserialization
- Evaluator types (EvalResult, AgentBeatsEvalContext)
- File loading (embedded, external, referenced states)
- Evaluator loading (dynamic module import)
- Error handling (missing files, invalid data, syntax errors)
- Manager caching behavior (scenarios and evaluators)
- Validation warnings
