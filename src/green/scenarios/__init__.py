"""Scenario management for the Green Agent.

This module provides schema definitions and loading utilities for assessment
scenarios. Scenarios define:

- Characters to simulate and their response patterns
- Initial UES state (emails, calendar events, messages)
- Evaluation criteria for scoring Purple agents
- Timing parameters for the assessment

Key Classes:
    ScenarioConfig: Complete scenario configuration
    CharacterProfile: Profile for a simulated character
    EvaluationCriterion: Single evaluation criterion definition
    ScenarioManager: Loads and manages scenarios from disk

Key Types:
    EvalResult: Return type for programmatic evaluators
    AgentBeatsEvalContext: Context passed to evaluators during assessment
    EvaluatorFunc: Type alias for evaluator function signature
    EvaluatorRegistry: Type alias for evaluator ID -> function mapping

Example:
    >>> from src.green.scenarios import ScenarioManager, ScenarioConfig
    >>> manager = ScenarioManager(Path("scenarios"))
    >>> scenarios = manager.list_scenarios()
    >>> config = manager.get_scenario("email_triage_basic")
    >>> evaluators = manager.get_evaluators("email_triage_basic")
    >>> config.scenario_id
    'email_triage_basic'
"""

from src.green.scenarios.loader import (
    EvaluatorLoadError,
    ScenarioLoader,
    ScenarioManager,
    ScenarioNotFoundError,
    ScenarioValidationError,
)
from src.green.scenarios.schema import (
    AgentBeatsEvalContext,
    CharacterProfile,
    EvalResult,
    EvaluationCriterion,
    EvaluatorFunc,
    EvaluatorRegistry,
    ResponseTiming,
    ScenarioConfig,
)

__all__ = [
    # Schema models
    "ResponseTiming",
    "CharacterProfile",
    "EvaluationCriterion",
    "ScenarioConfig",
    # Evaluator types
    "EvalResult",
    "AgentBeatsEvalContext",
    "EvaluatorFunc",
    "EvaluatorRegistry",
    # Loader utilities
    "ScenarioManager",
    "ScenarioLoader",
    "ScenarioNotFoundError",
    "ScenarioValidationError",
    "EvaluatorLoadError",
]
