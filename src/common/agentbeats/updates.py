"""AgentBeats task update models and emitter.

This module provides Pydantic models for structured task updates that the Green
agent emits during assessments, plus a TaskUpdateEmitter helper class for convenience.

Task updates serve as logs/traces that provide visibility into assessment
progress. They are emitted via A2A TaskStatusUpdateEvent messages and can be
used for debugging, observability, and audit trails.

NOTE: All task updates are emitted by the Green agent. Purple agents do not
emit task updates directly - this simplifies Purple agent implementation and
ensures consistent logging under the Green agent's control.

Update Types:
    - AssessmentStartedUpdate: Assessment has begun
    - ScenarioLoadedUpdate: Scenario configuration loaded
    - TurnStartedUpdate: New turn has begun
    - TurnCompletedUpdate: Turn has completed
    - ResponsesGeneratedUpdate: Character responses generated
    - SimulationAdvancedUpdate: UES simulation time advanced
    - EvaluationStartedUpdate: Evaluation phase has begun
    - CriterionEvaluatedUpdate: Single criterion evaluated
    - AssessmentCompletedUpdate: Assessment has ended
    - ErrorOccurredUpdate: An error occurred
    - ActionObservedUpdate: Logs a Purple agent action (emitted by Green when
        processing TurnCompleteMessage)

Design Note:
    All update models include a `message_type` field with a fixed literal string
    value (prefixed with "update_") to enable easy parsing. Updates are meant to
    be embedded in A2A TaskStatusUpdateEvent messages.

Example:
    >>> from src.common.agentbeats.updates import TaskUpdateEmitter, ActionObservedUpdate
    >>> emitter = TaskUpdateEmitter(task_id="task-123", context_id="ctx-456")
    >>> update = emitter.action_observed(
    ...     action="email.send",
    ...     parameters={"to": ["alice@example.com"]},
    ...     success=True,
    ...     timestamp=datetime.now(tz=timezone.utc)
    ... )
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Union

from a2a.types import Message, TaskState, TaskStatusUpdateEvent, TaskStatus
from pydantic import BaseModel, ConfigDict, Field

from src.common.a2a.messages import create_data_message


# =============================================================================
# Update Type Enum
# =============================================================================


class TaskUpdateType(str, Enum):
    """Types of task updates for logging.

    These correspond to the `message_type` values in update models.
    All updates are emitted by the Green agent.
    """

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
    # Emitted when Green agent processes TurnCompleteMessage from Purple
    ACTION_OBSERVED = "update_action_observed"


# =============================================================================
# Green Agent Update Models
# =============================================================================


class AssessmentStartedUpdate(BaseModel):
    """Update emitted when an assessment begins.

    Attributes:
        message_type: Fixed identifier for this update type.
        assessment_id: Unique identifier for this assessment run.
        scenario_id: ID of the scenario being used.
        participant_url: A2A endpoint URL of the Purple agent.
        start_time: When the assessment started (simulation time).

    Example:
        >>> update = AssessmentStartedUpdate(
        ...     assessment_id="assess-123",
        ...     scenario_id="email_triage_basic",
        ...     participant_url="http://purple-agent:8001",
        ...     start_time=datetime.now(tz=timezone.utc)
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_assessment_started"] = "update_assessment_started"
    assessment_id: str = Field(..., description="Unique assessment identifier")
    scenario_id: str = Field(..., description="Scenario being used")
    participant_url: str = Field(..., description="Purple agent endpoint URL")
    start_time: datetime = Field(..., description="Assessment start time")


class ScenarioLoadedUpdate(BaseModel):
    """Update emitted when a scenario is loaded.

    Attributes:
        message_type: Fixed identifier for this update type.
        scenario_id: ID of the loaded scenario.
        scenario_name: Human-readable scenario name.
        criteria_count: Number of evaluation criteria.
        character_count: Number of simulated characters.

    Example:
        >>> update = ScenarioLoadedUpdate(
        ...     scenario_id="email_triage_basic",
        ...     scenario_name="Basic Email Triage",
        ...     criteria_count=5,
        ...     character_count=3
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_scenario_loaded"] = "update_scenario_loaded"
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    criteria_count: int = Field(..., ge=0, description="Number of evaluation criteria")
    character_count: int = Field(..., ge=0, description="Number of simulated characters")


class TurnStartedUpdate(BaseModel):
    """Update emitted when a new turn begins.

    Attributes:
        message_type: Fixed identifier for this update type.
        turn_number: The turn number (1-indexed).
        current_time: Current simulation time.
        events_pending: Number of pending events in UES.

    Example:
        >>> update = TurnStartedUpdate(
        ...     turn_number=3,
        ...     current_time=datetime.now(tz=timezone.utc),
        ...     events_pending=5
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_turn_started"] = "update_turn_started"
    turn_number: int = Field(..., ge=1, description="Turn number (1-indexed)")
    current_time: datetime = Field(..., description="Current simulation time")
    events_pending: int = Field(..., ge=0, description="Pending events in UES")


class TurnCompletedUpdate(BaseModel):
    """Update emitted when a turn completes.

    Attributes:
        message_type: Fixed identifier for this update type.
        turn_number: The turn number that completed.
        actions_taken: Number of actions the Purple agent took.
        time_advanced: ISO 8601 duration of time advanced.
        early_completion_requested: Whether Purple requested early completion.

    Example:
        >>> update = TurnCompletedUpdate(
        ...     turn_number=3,
        ...     actions_taken=4,
        ...     time_advanced="PT1H",
        ...     early_completion_requested=False
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_turn_completed"] = "update_turn_completed"
    turn_number: int = Field(..., ge=1, description="Turn number that completed")
    actions_taken: int = Field(..., ge=0, description="Number of actions taken")
    time_advanced: str = Field(..., description="Time advanced (ISO 8601 duration)")
    early_completion_requested: bool = Field(
        default=False, description="Whether early completion was requested"
    )


class ResponsesGeneratedUpdate(BaseModel):
    """Update emitted when character responses are generated.

    Attributes:
        message_type: Fixed identifier for this update type.
        turn_number: The turn for which responses were generated.
        responses_count: Number of responses generated.
        characters_involved: List of character names that responded.

    Example:
        >>> update = ResponsesGeneratedUpdate(
        ...     turn_number=2,
        ...     responses_count=3,
        ...     characters_involved=["Alice", "Bob"]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_responses_generated"] = "update_responses_generated"
    turn_number: int = Field(..., ge=1, description="Turn number")
    responses_count: int = Field(..., ge=0, description="Number of responses generated")
    characters_involved: list[str] = Field(
        default_factory=list, description="Characters that responded"
    )


class SimulationAdvancedUpdate(BaseModel):
    """Update emitted when UES simulation time is advanced.

    Attributes:
        message_type: Fixed identifier for this update type.
        previous_time: Time before advancement.
        new_time: Time after advancement.
        duration: ISO 8601 duration of advancement.
        events_processed: Number of events processed during advancement.

    Example:
        >>> update = SimulationAdvancedUpdate(
        ...     previous_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
        ...     new_time=datetime(2026, 1, 28, 10, 0, tzinfo=timezone.utc),
        ...     duration="PT1H",
        ...     events_processed=2
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_simulation_advanced"] = "update_simulation_advanced"
    previous_time: datetime = Field(..., description="Time before advancement")
    new_time: datetime = Field(..., description="Time after advancement")
    duration: str = Field(..., description="Duration advanced (ISO 8601)")
    events_processed: int = Field(..., ge=0, description="Events processed")


class EvaluationStartedUpdate(BaseModel):
    """Update emitted when the evaluation phase begins.

    Attributes:
        message_type: Fixed identifier for this update type.
        criteria_count: Total number of criteria to evaluate.
        dimensions: List of scoring dimensions being evaluated.

    Example:
        >>> update = EvaluationStartedUpdate(
        ...     criteria_count=5,
        ...     dimensions=["accuracy", "efficiency", "politeness"]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_evaluation_started"] = "update_evaluation_started"
    criteria_count: int = Field(..., ge=0, description="Number of criteria to evaluate")
    dimensions: list[str] = Field(..., description="Scoring dimensions being evaluated")


class CriterionEvaluatedUpdate(BaseModel):
    """Update emitted when a single criterion is evaluated.

    Attributes:
        message_type: Fixed identifier for this update type.
        criterion_id: ID of the evaluated criterion.
        criterion_name: Human-readable criterion name.
        dimension: The scoring dimension this criterion belongs to.
        score: Achieved score.
        max_score: Maximum possible score.
        evaluation_method: How the criterion was evaluated ("programmatic" or "llm").

    Example:
        >>> update = CriterionEvaluatedUpdate(
        ...     criterion_id="email_politeness",
        ...     criterion_name="Email Politeness",
        ...     dimension="politeness",
        ...     score=8,
        ...     max_score=10,
        ...     evaluation_method="llm"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_criterion_evaluated"] = "update_criterion_evaluated"
    criterion_id: str = Field(..., description="Criterion identifier")
    criterion_name: str = Field(..., description="Human-readable criterion name")
    dimension: str = Field(..., description="Scoring dimension")
    score: int = Field(..., ge=0, description="Achieved score")
    max_score: int = Field(..., ge=0, description="Maximum possible score")
    evaluation_method: Literal["programmatic", "llm"] = Field(
        ..., description="Evaluation method used"
    )


class AssessmentCompletedUpdate(BaseModel):
    """Update emitted when an assessment ends.

    Attributes:
        message_type: Fixed identifier for this update type.
        reason: Why the assessment ended.
        total_turns: Total number of turns taken.
        total_actions: Total number of actions taken.
        duration_seconds: Assessment duration in seconds.
        overall_score: Final overall score.
        max_score: Maximum possible score.

    Example:
        >>> update = AssessmentCompletedUpdate(
        ...     reason="scenario_complete",
        ...     total_turns=10,
        ...     total_actions=25,
        ...     duration_seconds=1234.5,
        ...     overall_score=85,
        ...     max_score=100
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_assessment_completed"] = "update_assessment_completed"
    reason: Literal[
        "scenario_complete",
        "early_completion",
        "timeout",
        "error",
    ] = Field(..., description="Why the assessment ended")
    total_turns: int = Field(..., ge=0, description="Total turns taken")
    total_actions: int = Field(..., ge=0, description="Total actions taken")
    duration_seconds: float = Field(..., ge=0, description="Duration in seconds")
    overall_score: int = Field(..., ge=0, description="Final overall score")
    max_score: int = Field(..., ge=0, description="Maximum possible score")


class ErrorOccurredUpdate(BaseModel):
    """Update emitted when an error occurs.

    Attributes:
        message_type: Fixed identifier for this update type.
        error_type: Category of error.
        error_message: Human-readable error description.
        recoverable: Whether the assessment can continue.
        context: Optional additional context about the error.

    Example:
        >>> update = ErrorOccurredUpdate(
        ...     error_type="timeout",
        ...     error_message="Purple agent did not respond within 300 seconds",
        ...     recoverable=False,
        ...     context={"turn_number": 5}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_error_occurred"] = "update_error_occurred"
    error_type: Literal[
        "timeout",
        "communication_error",
        "protocol_error",
        "ues_error",
        "evaluation_error",
        "internal_error",
    ] = Field(..., description="Error category")
    error_message: str = Field(..., description="Human-readable error description")
    recoverable: bool = Field(..., description="Whether assessment can continue")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional error context"
    )


# =============================================================================
# Action Observed Update (emitted by Green agent)
# =============================================================================


class ActionObservedUpdate(BaseModel):
    """Update emitted by the Green agent when logging a Purple agent action.

    The Green agent emits this update when processing a TurnCompleteMessage
    from the Purple agent. For each ActionLogEntry in the message, the Green
    agent emits one ActionObservedUpdate. This ensures all Purple agent actions
    are logged consistently without requiring Purple agents to emit updates.

    The fields mirror ActionLogEntry to ensure complete information capture.

    Attributes:
        message_type: Fixed identifier for this update type.
        turn_number: The turn during which the action was taken.
        timestamp: When the action was performed.
        action: Action identifier (e.g., "email.send", "calendar.create").
        parameters: Action-specific parameters.
        success: Whether the action succeeded.
        error_message: Error message if success=False.
        notes: Optional reasoning or explanation for the action.

    Example:
        >>> update = ActionObservedUpdate(
        ...     turn_number=2,
        ...     timestamp=datetime.now(tz=timezone.utc),
        ...     action="email.send",
        ...     parameters={"to": ["alice@example.com"], "subject": "Hello"},
        ...     success=True,
        ...     notes="Responding to Alice's question"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["update_action_observed"] = "update_action_observed"
    turn_number: int = Field(..., ge=1, description="Turn number (1-indexed)")
    timestamp: datetime = Field(..., description="When the action was performed")
    action: str = Field(
        ..., description="Action identifier (e.g., 'email.send', 'calendar.create')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action-specific parameters"
    )
    success: bool = Field(..., description="Whether the action succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if success=False"
    )
    notes: str | None = Field(
        default=None, description="Optional reasoning or explanation"
    )


# =============================================================================
# Update Parsing
# =============================================================================

# Union type for all AgentBeats update types
AgentBeatsUpdate = Union[
    AssessmentStartedUpdate,
    ScenarioLoadedUpdate,
    TurnStartedUpdate,
    TurnCompletedUpdate,
    ResponsesGeneratedUpdate,
    SimulationAdvancedUpdate,
    EvaluationStartedUpdate,
    CriterionEvaluatedUpdate,
    AssessmentCompletedUpdate,
    ErrorOccurredUpdate,
    ActionObservedUpdate,
]

# Mapping from message_type string to model class
UPDATE_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "update_assessment_started": AssessmentStartedUpdate,
    "update_scenario_loaded": ScenarioLoadedUpdate,
    "update_turn_started": TurnStartedUpdate,
    "update_turn_completed": TurnCompletedUpdate,
    "update_responses_generated": ResponsesGeneratedUpdate,
    "update_simulation_advanced": SimulationAdvancedUpdate,
    "update_evaluation_started": EvaluationStartedUpdate,
    "update_criterion_evaluated": CriterionEvaluatedUpdate,
    "update_assessment_completed": AssessmentCompletedUpdate,
    "update_error_occurred": ErrorOccurredUpdate,
    "update_action_observed": ActionObservedUpdate,
}


def parse_update(data: dict[str, Any]) -> AgentBeatsUpdate:
    """Parse a dictionary into the appropriate AgentBeats update type.

    Uses the `message_type` field to determine which model to instantiate.

    Args:
        data: Dictionary containing update data, must include `message_type`.

    Returns:
        The appropriate update model instance.

    Raises:
        ValueError: If `message_type` is missing or unrecognized.

    Example:
        >>> data = {"message_type": "update_turn_started", "turn_number": 1,
        ...         "current_time": "2026-01-28T12:00:00Z", "events_pending": 0}
        >>> update = parse_update(data)
        >>> isinstance(update, TurnStartedUpdate)
        True
        >>> update.turn_number
        1
    """
    message_type = data.get("message_type")
    if message_type is None:
        raise ValueError("Update data must include 'message_type' field")

    model_class = UPDATE_TYPE_REGISTRY.get(message_type)
    if model_class is None:
        valid_types = ", ".join(sorted(UPDATE_TYPE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown message_type '{message_type}'. Valid types: {valid_types}"
        )

    return model_class.model_validate(data)


# =============================================================================
# Task Update Emitter
# =============================================================================


class TaskUpdateEmitter:
    """Helper for emitting structured task updates as A2A events.

    This class simplifies the process of creating and emitting AgentBeats-style
    task updates. It holds the task_id and context_id, and provides typed
    convenience methods for each update type.

    The emitter creates TaskStatusUpdateEvent objects that can be yielded
    from an A2A executor or enqueued to an event queue.

    Attributes:
        task_id: The A2A task ID for all emitted updates.
        context_id: The A2A context ID for all emitted updates.

    Example:
        >>> emitter = TaskUpdateEmitter(task_id="task-123", context_id="ctx-456")
        >>> event = emitter.assessment_started(
        ...     assessment_id="assess-001",
        ...     scenario_id="email_triage_basic",
        ...     participant_url="http://purple:8001",
        ...     start_time=datetime.now(tz=timezone.utc)
        ... )
        >>> # event is a TaskStatusUpdateEvent ready to be yielded/enqueued
    """

    def __init__(self, task_id: str, context_id: str) -> None:
        """Initialize the emitter.

        Args:
            task_id: The A2A task ID for updates.
            context_id: The A2A context ID for updates.
        """
        self.task_id = task_id
        self.context_id = context_id

    def _create_event(
        self,
        update: AgentBeatsUpdate,
        state: TaskState = TaskState.working,
        final: bool = False,
    ) -> TaskStatusUpdateEvent:
        """Create a TaskStatusUpdateEvent from an update model.

        Args:
            update: The update model to embed in the event.
            state: The task state for the event.
            final: Whether this is a final update.

        Returns:
            A TaskStatusUpdateEvent with the update as a data message.
        """
        # Serialize update to dict and wrap in A2A message
        update_data = update.model_dump(mode="json")
        message = create_data_message(data=update_data)

        timestamp = datetime.now(timezone.utc).isoformat()

        return TaskStatusUpdateEvent(
            task_id=self.task_id,
            context_id=self.context_id,
            status=TaskStatus(
                state=state,
                message=message,
                timestamp=timestamp,
            ),
            final=final,
        )

    # =========================================================================
    # Green Agent Update Methods
    # =========================================================================

    def assessment_started(
        self,
        assessment_id: str,
        scenario_id: str,
        participant_url: str,
        start_time: datetime,
    ) -> TaskStatusUpdateEvent:
        """Emit an assessment started update.

        Args:
            assessment_id: Unique identifier for this assessment.
            scenario_id: ID of the scenario being used.
            participant_url: A2A endpoint URL of the Purple agent.
            start_time: When the assessment started.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = AssessmentStartedUpdate(
            assessment_id=assessment_id,
            scenario_id=scenario_id,
            participant_url=participant_url,
            start_time=start_time,
        )
        return self._create_event(update)

    def scenario_loaded(
        self,
        scenario_id: str,
        scenario_name: str,
        criteria_count: int,
        character_count: int,
    ) -> TaskStatusUpdateEvent:
        """Emit a scenario loaded update.

        Args:
            scenario_id: ID of the loaded scenario.
            scenario_name: Human-readable scenario name.
            criteria_count: Number of evaluation criteria.
            character_count: Number of simulated characters.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = ScenarioLoadedUpdate(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            criteria_count=criteria_count,
            character_count=character_count,
        )
        return self._create_event(update)

    def turn_started(
        self,
        turn_number: int,
        current_time: datetime,
        events_pending: int,
    ) -> TaskStatusUpdateEvent:
        """Emit a turn started update.

        Args:
            turn_number: The turn number (1-indexed).
            current_time: Current simulation time.
            events_pending: Number of pending events in UES.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = TurnStartedUpdate(
            turn_number=turn_number,
            current_time=current_time,
            events_pending=events_pending,
        )
        return self._create_event(update)

    def turn_completed(
        self,
        turn_number: int,
        actions_taken: int,
        time_advanced: str,
        early_completion_requested: bool = False,
    ) -> TaskStatusUpdateEvent:
        """Emit a turn completed update.

        Args:
            turn_number: The turn number that completed.
            actions_taken: Number of actions the Purple agent took.
            time_advanced: ISO 8601 duration of time advanced.
            early_completion_requested: Whether Purple requested early completion.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = TurnCompletedUpdate(
            turn_number=turn_number,
            actions_taken=actions_taken,
            time_advanced=time_advanced,
            early_completion_requested=early_completion_requested,
        )
        return self._create_event(update)

    def responses_generated(
        self,
        turn_number: int,
        responses_count: int,
        characters_involved: list[str] | None = None,
    ) -> TaskStatusUpdateEvent:
        """Emit a responses generated update.

        Args:
            turn_number: The turn for which responses were generated.
            responses_count: Number of responses generated.
            characters_involved: List of character names that responded.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = ResponsesGeneratedUpdate(
            turn_number=turn_number,
            responses_count=responses_count,
            characters_involved=characters_involved or [],
        )
        return self._create_event(update)

    def simulation_advanced(
        self,
        previous_time: datetime,
        new_time: datetime,
        duration: str,
        events_processed: int,
    ) -> TaskStatusUpdateEvent:
        """Emit a simulation advanced update.

        Args:
            previous_time: Time before advancement.
            new_time: Time after advancement.
            duration: ISO 8601 duration of advancement.
            events_processed: Number of events processed.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = SimulationAdvancedUpdate(
            previous_time=previous_time,
            new_time=new_time,
            duration=duration,
            events_processed=events_processed,
        )
        return self._create_event(update)

    def evaluation_started(
        self,
        criteria_count: int,
        dimensions: list[str],
    ) -> TaskStatusUpdateEvent:
        """Emit an evaluation started update.

        Args:
            criteria_count: Total number of criteria to evaluate.
            dimensions: List of scoring dimensions being evaluated.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = EvaluationStartedUpdate(
            criteria_count=criteria_count,
            dimensions=dimensions,
        )
        return self._create_event(update)

    def criterion_evaluated(
        self,
        criterion_id: str,
        criterion_name: str,
        dimension: str,
        score: int,
        max_score: int,
        evaluation_method: Literal["programmatic", "llm"],
    ) -> TaskStatusUpdateEvent:
        """Emit a criterion evaluated update.

        Args:
            criterion_id: ID of the evaluated criterion.
            criterion_name: Human-readable criterion name.
            dimension: The scoring dimension this criterion belongs to.
            score: Achieved score.
            max_score: Maximum possible score.
            evaluation_method: How the criterion was evaluated.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = CriterionEvaluatedUpdate(
            criterion_id=criterion_id,
            criterion_name=criterion_name,
            dimension=dimension,
            score=score,
            max_score=max_score,
            evaluation_method=evaluation_method,
        )
        return self._create_event(update)

    def assessment_completed(
        self,
        reason: Literal["scenario_complete", "early_completion", "timeout", "error"],
        total_turns: int,
        total_actions: int,
        duration_seconds: float,
        overall_score: int,
        max_score: int,
    ) -> TaskStatusUpdateEvent:
        """Emit an assessment completed update.

        Args:
            reason: Why the assessment ended.
            total_turns: Total number of turns taken.
            total_actions: Total number of actions taken.
            duration_seconds: Assessment duration in seconds.
            overall_score: Final overall score.
            max_score: Maximum possible score.

        Returns:
            A TaskStatusUpdateEvent with final=True.
        """
        update = AssessmentCompletedUpdate(
            reason=reason,
            total_turns=total_turns,
            total_actions=total_actions,
            duration_seconds=duration_seconds,
            overall_score=overall_score,
            max_score=max_score,
        )
        # Assessment completed is a terminal state
        return self._create_event(update, state=TaskState.completed, final=True)

    def error_occurred(
        self,
        error_type: Literal[
            "timeout",
            "communication_error",
            "protocol_error",
            "ues_error",
            "evaluation_error",
            "internal_error",
        ],
        error_message: str,
        recoverable: bool,
        context: dict[str, Any] | None = None,
    ) -> TaskStatusUpdateEvent:
        """Emit an error occurred update.

        Args:
            error_type: Category of error.
            error_message: Human-readable error description.
            recoverable: Whether the assessment can continue.
            context: Optional additional context about the error.

        Returns:
            A TaskStatusUpdateEvent (final if not recoverable).
        """
        update = ErrorOccurredUpdate(
            error_type=error_type,
            error_message=error_message,
            recoverable=recoverable,
            context=context,
        )
        # If not recoverable, this is a terminal error
        state = TaskState.working if recoverable else TaskState.failed
        final = not recoverable
        return self._create_event(update, state=state, final=final)

    # =========================================================================
    # Action Logging (called when processing TurnCompleteMessage)
    # =========================================================================

    def action_observed(
        self,
        turn_number: int,
        timestamp: datetime,
        action: str,
        parameters: dict[str, Any],
        success: bool,
        error_message: str | None = None,
        notes: str | None = None,
    ) -> TaskStatusUpdateEvent:
        """Emit an action observed update.

        The Green agent calls this method for each ActionLogEntry in a
        TurnCompleteMessage received from the Purple agent. This ensures
        all Purple agent actions are logged for debugging and audit.

        Args:
            turn_number: The turn during which the action was taken.
            timestamp: When the action was performed.
            action: Action identifier (e.g., "email.send", "calendar.create").
            parameters: Action-specific parameters.
            success: Whether the action succeeded.
            error_message: Error message if success=False.
            notes: Optional reasoning or explanation for the action.

        Returns:
            A TaskStatusUpdateEvent to be yielded/enqueued.
        """
        update = ActionObservedUpdate(
            turn_number=turn_number,
            timestamp=timestamp,
            action=action,
            parameters=parameters,
            success=success,
            error_message=error_message,
            notes=notes,
        )
        return self._create_event(update)
