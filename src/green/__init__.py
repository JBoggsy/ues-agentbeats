"""Green Agent package for AgentBeats competition.

The Green Agent orchestrates assessments, manages UES environments, and evaluates
Purple agents' performance. This package provides:

- Scenario management (loading, validation, configuration)
- Assessment orchestration (turn-based execution loop)
- Action log building (tracking Purple agent actions)
- Response generation (character simulation)
- Evaluation (criteria judging, scoring)

Modules:
    scenarios: Scenario schema and loading utilities
    action_log: Action log building for assessments
    assessment: Assessment orchestration and turn handling
    response_generator: Character response generation
    evaluation: Criteria evaluation and scoring

Example:
    >>> from pathlib import Path
    >>> from src.green.scenarios import ScenarioManager
    >>> manager = ScenarioManager(Path("scenarios"))
    >>> config = manager.load_scenario("email_triage_basic")
"""

from src.green.action_log import (
    ActionLogBuilder,
    ActionLogBuilderError,
    InvalidTurnNumberError,
    InvalidTurnStateError,
)
from src.green.scenarios import (
    CharacterProfile,
    EvaluationCriterion,
    ResponseTiming,
    ScenarioConfig,
    ScenarioLoader,
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
)

__all__ = [
    # Action log builder
    "ActionLogBuilder",
    "ActionLogBuilderError",
    "InvalidTurnNumberError",
    "InvalidTurnStateError",
    # Scenario schema models
    "ResponseTiming",
    "CharacterProfile",
    "EvaluationCriterion",
    "ScenarioConfig",
    # Scenario loader utilities
    "ScenarioManager",
    "ScenarioLoader",
    "ScenarioNotFoundError",
    "ScenarioValidationError",
]
