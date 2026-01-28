"""AgentBeats assessment message models.

This module defines Pydantic models for all messages exchanged between Green
and Purple agents during an AgentBeats assessment. All models include a
`message_type` field with a fixed literal value to enable easy parsing.

Message Types:
    - Summary models: EmailSummary, CalendarSummary, SMSSummary, ChatSummary
    - InitialStateSummary: Aggregates all modality summaries
    - AssessmentStartMessage: Green -> Purple at assessment start
    - TurnStartMessage: Green -> Purple at each turn start
    - ActionLogEntry: Single action taken by Purple agent
    - TurnCompleteMessage: Purple -> Green when turn completes
    - AssessmentCompleteMessage: Green -> Purple when assessment ends
    - EarlyCompletionMessage: Purple -> Green to signal early completion

Design Note:
    All message models include a `message_type` field with a fixed literal string
    value. This allows agents to easily parse incoming messages by checking the
    `message_type` field rather than inferring the type from which fields are
    present. The `message_type` is set as a class-level default and should not
    be overridden.

Example:
    >>> from src.common.agentbeats.messages import TurnStartMessage, parse_message
    >>> msg = TurnStartMessage(
    ...     turn_number=1,
    ...     current_time=datetime.now(tz=timezone.utc),
    ...     events_processed=0
    ... )
    >>> msg.message_type
    'turn_start'
    >>> data = msg.model_dump(mode="json")
    >>> parsed = parse_message(data)
    >>> isinstance(parsed, TurnStartMessage)
    True
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Modality Summary Models
# =============================================================================


class EmailSummary(BaseModel):
    """Summary counts for email modality.

    Derived from UES email `get_snapshot()` output. The `unread` count is
    computed by summing `folders[*].unread_count` from the snapshot.

    Attributes:
        message_type: Fixed identifier for this message type.
        total_emails: Total number of emails across all folders.
        total_threads: Total number of email threads.
        unread: Number of unread emails.
        draft_count: Number of draft emails.

    Example:
        >>> summary = EmailSummary(
        ...     total_emails=42,
        ...     total_threads=15,
        ...     unread=5,
        ...     draft_count=2
        ... )
        >>> summary.message_type
        'email_summary'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["email_summary"] = "email_summary"
    total_emails: int = Field(..., ge=0, description="Total emails across all folders")
    total_threads: int = Field(..., ge=0, description="Total email threads")
    unread: int = Field(..., ge=0, description="Number of unread emails")
    draft_count: int = Field(..., ge=0, description="Number of draft emails")


class CalendarSummary(BaseModel):
    """Summary counts for calendar modality.

    Derived from UES calendar `get_snapshot()` and `get_compact_snapshot()` outputs.
    The `events_today` count requires `get_compact_snapshot(current_time)`.

    Attributes:
        message_type: Fixed identifier for this message type.
        event_count: Total number of calendar events.
        calendar_count: Number of calendars.
        events_today: Number of events scheduled for the current day.

    Example:
        >>> summary = CalendarSummary(
        ...     event_count=20,
        ...     calendar_count=3,
        ...     events_today=4
        ... )
        >>> summary.message_type
        'calendar_summary'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["calendar_summary"] = "calendar_summary"
    event_count: int = Field(..., ge=0, description="Total calendar events")
    calendar_count: int = Field(..., ge=0, description="Number of calendars")
    events_today: int = Field(..., ge=0, description="Events scheduled for today")


class SMSSummary(BaseModel):
    """Summary counts for SMS modality.

    Derived from UES SMS `get_snapshot()` output.

    Attributes:
        message_type: Fixed identifier for this message type.
        total_messages: Total number of SMS messages.
        total_conversations: Number of SMS conversations.
        unread: Number of unread SMS messages.

    Example:
        >>> summary = SMSSummary(
        ...     total_messages=100,
        ...     total_conversations=10,
        ...     unread=3
        ... )
        >>> summary.message_type
        'sms_summary'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["sms_summary"] = "sms_summary"
    total_messages: int = Field(..., ge=0, description="Total SMS messages")
    total_conversations: int = Field(..., ge=0, description="Number of conversations")
    unread: int = Field(..., ge=0, description="Number of unread messages")


class ChatSummary(BaseModel):
    """Summary counts for chat modality.

    Derived from UES chat `get_snapshot()` output. Note that chat has no
    "unread" concept as messages are user-assistant pairs.

    Attributes:
        message_type: Fixed identifier for this message type.
        total_messages: Total number of chat messages.
        conversation_count: Number of chat conversations.

    Example:
        >>> summary = ChatSummary(
        ...     total_messages=50,
        ...     conversation_count=1
        ... )
        >>> summary.message_type
        'chat_summary'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["chat_summary"] = "chat_summary"
    total_messages: int = Field(..., ge=0, description="Total chat messages")
    conversation_count: int = Field(..., ge=0, description="Number of conversations")


class InitialStateSummary(BaseModel):
    """Summary of initial UES state, aggregating all modality snapshots.

    This summary is included in AssessmentStartMessage to give Purple agents
    an overview of the environment state without needing to query each modality.

    Attributes:
        message_type: Fixed identifier for this message type.
        email: Summary of email modality state.
        calendar: Summary of calendar modality state.
        sms: Summary of SMS modality state.
        chat: Summary of chat modality state.

    Example:
        >>> state = InitialStateSummary(
        ...     email=EmailSummary(total_emails=42, total_threads=15, unread=5, draft_count=2),
        ...     calendar=CalendarSummary(event_count=20, calendar_count=3, events_today=4),
        ...     sms=SMSSummary(total_messages=100, total_conversations=10, unread=3),
        ...     chat=ChatSummary(total_messages=50, conversation_count=1)
        ... )
        >>> state.message_type
        'initial_state_summary'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["initial_state_summary"] = "initial_state_summary"
    email: EmailSummary = Field(..., description="Email modality summary")
    calendar: CalendarSummary = Field(..., description="Calendar modality summary")
    sms: SMSSummary = Field(..., description="SMS modality summary")
    chat: ChatSummary = Field(..., description="Chat modality summary")


# =============================================================================
# Assessment Flow Messages
# =============================================================================


class AssessmentStartMessage(BaseModel):
    """Message sent from Green agent to Purple agent at assessment start.

    This message provides the Purple agent with everything needed to connect
    to the UES environment and understand the assessment context.

    Attributes:
        message_type: Fixed identifier for this message type.
        ues_url: URL of the UES server to connect to.
        api_key: API key for authenticating with UES.
        assessment_instructions: Instructions describing what the Purple agent
            should accomplish during the assessment.
        current_time: The current simulated time in UES.
        initial_state_summary: Summary of the initial UES state.

    Example:
        >>> msg = AssessmentStartMessage(
        ...     ues_url="http://localhost:8080",
        ...     api_key="secret-key",
        ...     assessment_instructions="Triage all unread emails.",
        ...     current_time=datetime.now(tz=timezone.utc),
        ...     initial_state_summary=state  # InitialStateSummary object
        ... )
        >>> msg.message_type
        'assessment_start'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["assessment_start"] = "assessment_start"
    ues_url: str = Field(..., description="UES server URL")
    api_key: str = Field(..., description="UES API key")
    assessment_instructions: str = Field(
        ..., description="Instructions for the assessment"
    )
    current_time: datetime = Field(..., description="Current simulated time")
    initial_state_summary: InitialStateSummary = Field(
        ..., description="Initial UES state summary"
    )


class TurnStartMessage(BaseModel):
    """Message sent from Green agent to Purple agent at each turn start.

    After the initial assessment start, the Green agent sends this message
    at the beginning of each turn to update the Purple agent on the current
    state.

    Attributes:
        message_type: Fixed identifier for this message type.
        turn_number: The current turn number (1-indexed).
        current_time: The current simulated time in UES.
        events_processed: Number of simulated events processed since last turn.

    Example:
        >>> msg = TurnStartMessage(
        ...     turn_number=3,
        ...     current_time=datetime.now(tz=timezone.utc),
        ...     events_processed=5
        ... )
        >>> msg.message_type
        'turn_start'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["turn_start"] = "turn_start"
    turn_number: int = Field(..., ge=1, description="Current turn number (1-indexed)")
    current_time: datetime = Field(..., description="Current simulated time")
    events_processed: int = Field(
        ..., ge=0, description="Events processed since last turn"
    )


class ActionLogEntry(BaseModel):
    """Single action taken by Purple agent during a turn.

    Purple agents report these in TurnCompleteMessage to document what actions
    they performed. The Green agent uses these to build the assessment action
    log and for evaluation purposes.

    Attributes:
        message_type: Fixed identifier for this message type.
        timestamp: When the action was performed.
        action: Action identifier (e.g., "email.send", "calendar.create").
        parameters: Action-specific parameters.
        success: Whether the action succeeded.
        error_message: Error message if success=False.

    Example:
        >>> entry = ActionLogEntry(
        ...     timestamp=datetime.now(tz=timezone.utc),
        ...     action="email.send",
        ...     parameters={"to": ["alice@example.com"], "subject": "Hello"},
        ...     success=True
        ... )
        >>> entry.message_type
        'action_log_entry'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["action_log_entry"] = "action_log_entry"
    timestamp: datetime = Field(..., description="When the action was performed")
    action: str = Field(
        ...,
        description="Action identifier (e.g., 'email.send', 'calendar.create')",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action-specific parameters"
    )
    success: bool = Field(..., description="Whether the action succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if success=False"
    )


class TurnCompleteMessage(BaseModel):
    """Message sent from Purple agent to Green agent when a turn completes.

    Purple agents report all actions taken during the turn. The Green agent
    uses this to build the assessment action log directly, rather than
    reconstructing it from UES event history.

    Attributes:
        message_type: Fixed identifier for this message type.
        actions: List of actions performed during this turn.
        notes: Optional reasoning or transparency notes.
        time_step: Requested time advance (ISO 8601 duration, default "PT1H").

    Example:
        >>> msg = TurnCompleteMessage(
        ...     actions=[
        ...         ActionLogEntry(
        ...             timestamp=datetime.now(tz=timezone.utc),
        ...             action="email.archive",
        ...             parameters={"email_id": "123"},
        ...             success=True
        ...         )
        ...     ],
        ...     notes="Archived spam email.",
        ...     time_step="PT30M"
        ... )
        >>> msg.message_type
        'turn_complete'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["turn_complete"] = "turn_complete"
    actions: list[ActionLogEntry] = Field(
        default_factory=list, description="Actions performed this turn"
    )
    notes: str | None = Field(
        default=None, description="Optional reasoning or transparency notes"
    )
    time_step: str = Field(
        default="PT1H", description="Requested time advance (ISO 8601 duration)"
    )


class AssessmentCompleteMessage(BaseModel):
    """Message sent from Green agent to Purple agent when assessment ends.

    This message notifies the Purple agent that the assessment has concluded
    and provides the reason for completion.

    Attributes:
        message_type: Fixed identifier for this message type.
        reason: The reason the assessment ended.

    Example:
        >>> msg = AssessmentCompleteMessage(reason="scenario_complete")
        >>> msg.message_type
        'assessment_complete'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["assessment_complete"] = "assessment_complete"
    reason: Literal[
        "scenario_complete",
        "early_completion",
        "timeout",
        "error",
    ] = Field(..., description="Reason the assessment ended")


class EarlyCompletionMessage(BaseModel):
    """Message sent from Purple agent to Green agent to signal early completion.

    Purple agents can send this message when they believe they have completed
    all assessment goals before the turn limit is reached.

    Attributes:
        message_type: Fixed identifier for this message type.
        reason: Optional explanation for why early completion is requested.

    Example:
        >>> msg = EarlyCompletionMessage(reason="All emails triaged and calendar updated.")
        >>> msg.message_type
        'early_completion'
    """

    model_config = ConfigDict(frozen=True)

    message_type: Literal["early_completion"] = "early_completion"
    reason: str | None = Field(
        default=None, description="Reason for early completion request"
    )


# =============================================================================
# Message Parsing
# =============================================================================

# Union type for all AgentBeats message types
AgentBeatsMessage = Union[
    EmailSummary,
    CalendarSummary,
    SMSSummary,
    ChatSummary,
    InitialStateSummary,
    AssessmentStartMessage,
    TurnStartMessage,
    ActionLogEntry,
    TurnCompleteMessage,
    AssessmentCompleteMessage,
    EarlyCompletionMessage,
]

# Mapping from message_type string to model class
MESSAGE_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "email_summary": EmailSummary,
    "calendar_summary": CalendarSummary,
    "sms_summary": SMSSummary,
    "chat_summary": ChatSummary,
    "initial_state_summary": InitialStateSummary,
    "assessment_start": AssessmentStartMessage,
    "turn_start": TurnStartMessage,
    "action_log_entry": ActionLogEntry,
    "turn_complete": TurnCompleteMessage,
    "assessment_complete": AssessmentCompleteMessage,
    "early_completion": EarlyCompletionMessage,
}


def parse_message(data: dict[str, Any]) -> AgentBeatsMessage:
    """Parse a dictionary into the appropriate AgentBeats message type.

    Uses the `message_type` field to determine which model to instantiate.

    Args:
        data: Dictionary containing message data, must include `message_type`.

    Returns:
        The appropriate message model instance.

    Raises:
        ValueError: If `message_type` is missing or unrecognized.

    Example:
        >>> data = {"message_type": "turn_start", "turn_number": 1,
        ...         "current_time": "2026-01-28T12:00:00Z", "events_processed": 0}
        >>> msg = parse_message(data)
        >>> isinstance(msg, TurnStartMessage)
        True
        >>> msg.turn_number
        1
    """
    message_type = data.get("message_type")
    if message_type is None:
        raise ValueError("Message data must include 'message_type' field")

    model_class = MESSAGE_TYPE_REGISTRY.get(message_type)
    if model_class is None:
        valid_types = ", ".join(sorted(MESSAGE_TYPE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown message_type '{message_type}'. Valid types: {valid_types}"
        )

    return model_class.model_validate(data)
