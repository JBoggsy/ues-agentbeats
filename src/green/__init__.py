"""Green Agent package for AgentBeats competition.

The Green Agent orchestrates assessments, manages UES environments, and evaluates
Purple agents' performance. This package provides:

- Scenario management (loading, validation, configuration)
- Assessment orchestration (turn-based execution loop)
- Action log building (tracking Purple agent actions)
- Response generation (character simulation)
- Criteria evaluation (judging, scoring)
- LLM configuration (model factory)
- Message collection (new message tracking)

Modules:
    scenarios: Scenario schema and loading utilities
    action_log: Action log building for assessments
    message_collector: New message collection from UES
    response_generator: Character response generation
    judge: Criteria evaluation and scoring
    llm_config: LLM factory for model instantiation

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
from src.green.judge import (
    CriteriaJudge,
    CriteriaJudgeError,
    EvaluationError,
)
from src.green.llm_config import (
    LLMConfig,
    LLMFactory,
    LLMProvider,
    UnsupportedModelError,
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
    # Criteria judge
    "CriteriaJudge",
    "CriteriaJudgeError",
    "EvaluationError",
    # LLM configuration
    "LLMConfig",
    "LLMFactory",
    "LLMProvider",
    "UnsupportedModelError",
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
