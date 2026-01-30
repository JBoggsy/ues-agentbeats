"""Response generation data models.

This module defines data models used by the ResponseGenerator for creating
character responses during assessments. These models represent:

- Scheduled responses to be injected into UES
- LLM output structures for decision-making
- Internal context objects for response generation

Classes:
    ScheduledResponse: A response scheduled for injection into UES.
    ShouldRespondResult: LLM output for should-respond decisions.
    CalendarRSVPResult: LLM output for calendar RSVP decisions.
    ThreadContext: Prepared thread history for LLM prompts.
    MessageContext: Context for processing a single message.

Design Notes:
    - ScheduledResponse is a dataclass (not Pydantic) for simplicity
    - LLM result models use Pydantic for structured output parsing
    - ThreadContext includes both raw and formatted thread data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from ues.client import CalendarEvent, Email, SMSMessage


# =============================================================================
# Response Types
# =============================================================================

ResponseModality = Literal["email", "sms", "calendar"]
CalendarRSVPStatus = Literal["accepted", "declined", "tentative"]


# =============================================================================
# Scheduled Response Model
# =============================================================================


@dataclass
class ScheduledResponse:
    """A response scheduled for injection into UES.

    This dataclass holds all information needed to schedule a character's
    response in UES. The GreenAgent uses these to inject character responses
    at the appropriate simulation times.

    Attributes:
        modality: The response modality ("email", "sms", or "calendar").
        character_name: Name of the character sending the response.
        character_email: Email address of the character (for email/calendar).
        character_phone: Phone number of the character (for SMS).
        scheduled_time: When the response should be delivered in simulation time.
        content: The response content (body text for email/SMS).
        original_message_id: ID of the message being responded to.
        thread_id: Thread ID for threading (email/SMS).
        subject: Email subject line (for email responses only).
        recipients: List of recipient addresses/numbers.
        cc_recipients: CC recipients (for email only).
        in_reply_to: Message ID being replied to (for email threading).
        references: Message reference chain (for email threading).
        event_id: Calendar event ID (for calendar RSVPs).
        rsvp_status: RSVP response status (for calendar RSVPs).
        rsvp_comment: Optional comment with RSVP.

    Example:
        >>> response = ScheduledResponse(
        ...     modality="email",
        ...     character_name="Alice Chen",
        ...     character_email="alice.chen@company.com",
        ...     scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
        ...     content="Thanks for the update!",
        ...     original_message_id="msg-123",
        ...     thread_id="thread-456",
        ...     subject="Re: Project Update",
        ...     recipients=["user@example.com"],
        ... )
    """

    # Required fields
    modality: ResponseModality
    character_name: str
    scheduled_time: datetime

    # Contact information (at least one required based on modality)
    character_email: str | None = None
    character_phone: str | None = None

    # Response content
    content: str | None = None  # Body text for email/SMS

    # Message threading
    original_message_id: str | None = None
    thread_id: str | None = None

    # Email-specific fields
    subject: str | None = None
    recipients: list[str] = field(default_factory=list)
    cc_recipients: list[str] = field(default_factory=list)
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)

    # Calendar-specific fields
    event_id: str | None = None
    rsvp_status: CalendarRSVPStatus | None = None
    rsvp_comment: str | None = None

    def __post_init__(self) -> None:
        """Validate the response based on modality."""
        if self.modality == "email":
            if not self.character_email:
                raise ValueError("Email responses require character_email")
            if not self.content:
                raise ValueError("Email responses require content")
            if not self.recipients:
                raise ValueError("Email responses require at least one recipient")
        elif self.modality == "sms":
            if not self.character_phone:
                raise ValueError("SMS responses require character_phone")
            if not self.content:
                raise ValueError("SMS responses require content")
            if not self.recipients:
                raise ValueError("SMS responses require at least one recipient")
        elif self.modality == "calendar":
            if not self.character_email:
                raise ValueError("Calendar responses require character_email")
            if not self.event_id:
                raise ValueError("Calendar responses require event_id")
            if not self.rsvp_status:
                raise ValueError("Calendar responses require rsvp_status")


# =============================================================================
# LLM Result Models
# =============================================================================


class ShouldRespondResult(BaseModel):
    """Result from the should-respond LLM check.

    This model is used to parse structured output from the LLM when
    determining whether a character should respond to a message.

    Attributes:
        should_respond: Whether the character should send a response.
        reasoning: Brief explanation of the decision.

    Example:
        >>> result = ShouldRespondResult(
        ...     should_respond=True,
        ...     reasoning="The message asks a direct question that requires an answer."
        ... )
    """

    should_respond: bool = Field(
        ...,
        description="Whether the character should send a response to this message",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why the character should or should not respond",
    )


class CalendarRSVPResult(BaseModel):
    """Result from the calendar RSVP LLM decision.

    This model is used to parse structured output from the LLM when
    determining how a character should respond to a calendar invitation.

    Attributes:
        status: The RSVP status (accepted, declined, or tentative).
        comment: Optional comment to include with the RSVP.
        reasoning: Explanation of the decision (for logging/debugging).

    Example:
        >>> result = CalendarRSVPResult(
        ...     status="accepted",
        ...     comment="Looking forward to it!",
        ...     reasoning="The character generally accepts team meetings."
        ... )
    """

    status: CalendarRSVPStatus = Field(
        ...,
        description="RSVP response status: accepted, declined, or tentative",
    )
    comment: str | None = Field(
        default=None,
        description="Optional brief comment to include with the RSVP response",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the character would respond this way",
    )


# =============================================================================
# Internal Context Models
# =============================================================================


@dataclass
class ThreadContext:
    """Prepared thread history for LLM prompts.

    This dataclass holds thread history in multiple formats for use in
    LLM prompts. It includes both the raw messages and a formatted string
    representation suitable for prompt injection.

    Attributes:
        messages: The raw message objects in chronological order (oldest first).
        formatted_history: Pre-formatted string representation for prompts.
        summary: Optional summary of older messages (when thread is truncated).
        total_message_count: Total messages in thread (before truncation).
        included_message_count: Number of messages included in formatted_history.

    Example:
        >>> context = ThreadContext(
        ...     messages=[email1, email2, email3],
        ...     formatted_history="[10:00] Alice: Hi!\\n[10:05] Bob: Hello!",
        ...     total_message_count=10,
        ...     included_message_count=3,
        ...     summary="Earlier: Alice and Bob discussed the project timeline.",
        ... )
    """

    messages: list[Email | SMSMessage]
    formatted_history: str
    total_message_count: int = 0
    included_message_count: int = 0
    summary: str | None = None


@dataclass
class MessageContext:
    """Context for processing a single message.

    This dataclass holds all context needed when processing a message to
    determine if characters should respond and generate responses.

    Attributes:
        message: The message being processed.
        modality: The modality of the message ("email" or "sms").
        sender_address: Email address or phone number of the sender.
        all_recipients: All recipient addresses/numbers (to + cc for email).
        thread_context: Prepared thread history (if available).

    Example:
        >>> context = MessageContext(
        ...     message=email,
        ...     modality="email",
        ...     sender_address="alice@example.com",
        ...     all_recipients={"bob@example.com", "carol@example.com"},
        ...     thread_context=thread_ctx,
        ... )
    """

    message: Email | SMSMessage
    modality: Literal["email", "sms"]
    sender_address: str
    all_recipients: set[str] = field(default_factory=set)
    thread_context: ThreadContext | None = None


@dataclass
class CalendarEventContext:
    """Context for processing a calendar event.

    This dataclass holds context for a calendar event when determining
    how characters should respond to invitations.

    Attributes:
        event: The calendar event being processed.
        organizer_email: Email of the event organizer.
        all_attendees: Set of all attendee email addresses.

    Example:
        >>> context = CalendarEventContext(
        ...     event=calendar_event,
        ...     organizer_email="alice@example.com",
        ...     all_attendees={"bob@example.com", "carol@example.com"},
        ... )
    """

    event: CalendarEvent
    organizer_email: str
    all_attendees: set[str] = field(default_factory=set)
