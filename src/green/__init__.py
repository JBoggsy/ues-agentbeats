"""Green Agent package for AgentBeats competition.

The Green Agent orchestrates assessments, manages UES environments, and evaluates
Purple agents' performance. This package provides:

- Scenario management (loading, validation, configuration)
- Assessment orchestration (turn-based execution loop)
- Core infrastructure (LLM factory, action logging, message collection)
- Response generation (character simulation)
- Criteria evaluation (judging, scoring)

Subpackages:
    core: Infrastructure shared across components (LLM, action log, message collector)
    response: Character response generation system
    evaluation: Criteria evaluation and scoring
    scenarios: Scenario schema and loading utilities
    assessment: Assessment orchestration (in progress)

Example:
    >>> from pathlib import Path
    >>> from src.green.scenarios import ScenarioManager
    >>> manager = ScenarioManager(Path("scenarios"))
    >>> config = manager.load_scenario("email_triage_basic")
"""

from src.green.agent import GreenAgent
from src.green.core import (
    ActionLogBuilder,
    ActionLogBuilderError,
    InvalidTurnNumberError,
    InvalidTurnStateError,
    LLMConfig,
    LLMFactory,
    LLMProvider,
    UnsupportedModelError,
)
from src.green.evaluation import (
    CriteriaJudge,
    CriteriaJudgeError,
    EvaluationError,
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
    # GreenAgent orchestrator
    "GreenAgent",
    # Core infrastructure
    "ActionLogBuilder",
    "ActionLogBuilderError",
    "InvalidTurnNumberError",
    "InvalidTurnStateError",
    "LLMConfig",
    "LLMFactory",
    "LLMProvider",
    "UnsupportedModelError",
    # Criteria evaluation
    "CriteriaJudge",
    "CriteriaJudgeError",
    "EvaluationError",
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
