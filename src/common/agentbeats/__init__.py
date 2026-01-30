"""AgentBeats-specific helper code for Green and Purple agents.

This module provides AgentBeats-specific patterns, message types, and utilities
used by both Green Agent (evaluator) and Purple Agent (participant) implementations.

Modules:
    messages: Pydantic models for AgentBeats message payloads
    results: Pydantic models for assessment results artifacts
    updates: Task update models and emitter for logging/tracing
    config: Configuration models for Green and Purple agents
"""

from __future__ import annotations

from src.common.agentbeats.config import (
    AgentBeatsConfig,
    GreenAgentConfig,
    PurpleAgentConfig,
    merge_configs,
    validate_config,
)
from src.common.agentbeats.messages import (
    AssessmentCompleteMessage,
    AssessmentStartMessage,
    CalendarSummary,
    ChatSummary,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    SMSSummary,
    TurnCompleteMessage,
    TurnStartMessage,
    parse_message,
)
from src.common.agentbeats.results import (
    ALL_DIMENSIONS,
    ActionLogEntry,
    AssessmentResults,
    AssessmentStatus,
    CriterionResult,
    DimensionScore,
    OverallScore,
    RESULT_TYPE_REGISTRY,
    Scores,
    ScoringDimension,
    parse_result,
)
from src.common.agentbeats.updates import (
    # Update models (all emitted by Green Agent)
    ActionObservedUpdate,
    AssessmentCompletedUpdate,
    AssessmentStartedUpdate,
    CriterionEvaluatedUpdate,
    ErrorOccurredUpdate,
    EvaluationStartedUpdate,
    ResponsesGeneratedUpdate,
    ScenarioLoadedUpdate,
    SimulationAdvancedUpdate,
    TurnCompletedUpdate,
    TurnStartedUpdate,
    # Update emitter
    TaskUpdateEmitter,
    TaskUpdateType,
    # Update parsing
    UPDATE_TYPE_REGISTRY,
    parse_update,
)

__all__ = [
    # Message summary models
    "EmailSummary",
    "CalendarSummary",
    "SMSSummary",
    "ChatSummary",
    "InitialStateSummary",
    # Message models
    "AssessmentStartMessage",
    "TurnStartMessage",
    "TurnCompleteMessage",
    "AssessmentCompleteMessage",
    "EarlyCompletionMessage",
    # Message parsing utility
    "parse_message",
    # Result models
    "DimensionScore",
    "OverallScore",
    "Scores",
    "CriterionResult",
    "ActionLogEntry",
    "AssessmentResults",
    # Result types
    "ScoringDimension",
    "AssessmentStatus",
    "ALL_DIMENSIONS",
    "RESULT_TYPE_REGISTRY",
    # Result parsing utility
    "parse_result",
    # Update models (all emitted by Green Agent)
    "ActionObservedUpdate",
    "AssessmentCompletedUpdate",
    "AssessmentStartedUpdate",
    "CriterionEvaluatedUpdate",
    "ErrorOccurredUpdate",
    "EvaluationStartedUpdate",
    "ResponsesGeneratedUpdate",
    "ScenarioLoadedUpdate",
    "SimulationAdvancedUpdate",
    "TurnCompletedUpdate",
    "TurnStartedUpdate",
    # Update emitter and type
    "TaskUpdateEmitter",
    "TaskUpdateType",
    # Update registry and parsing
    "UPDATE_TYPE_REGISTRY",
    "parse_update",
    # Configuration models
    "AgentBeatsConfig",
    "GreenAgentConfig",
    "PurpleAgentConfig",
    # Configuration utilities
    "merge_configs",
    "validate_config",
]
