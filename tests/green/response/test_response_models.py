"""Tests for response generation data models.

Tests cover:
- ScheduledResponse validation and field requirements by modality
- ShouldRespondResult Pydantic parsing
- CalendarRSVPResult Pydantic parsing
- ThreadContext and MessageContext dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.green.response.models import (
    CalendarEventContext,
    CalendarRSVPResult,
    MessageContext,
    ScheduledResponse,
    ShouldRespondResult,
    ThreadContext,
)


# =============================================================================
# Mock Objects for Testing
# =============================================================================


@dataclass
class MockEmail:
    """Mock Email object for testing."""

    message_id: str = "msg-1"
    thread_id: str = "thread-1"
    from_address: str = "sender@example.com"
    subject: str = "Test Subject"


@dataclass
class MockSMSMessage:
    """Mock SMSMessage object for testing."""

    message_id: str = "sms-1"
    thread_id: str = "sms-thread-1"
    from_number: str = "+15551234567"


@dataclass
class MockCalendarEvent:
    """Mock CalendarEvent object for testing."""

    event_id: str = "evt-1"
    title: str = "Test Event"
    organizer: str = "organizer@example.com"


# =============================================================================
# ScheduledResponse Tests
# =============================================================================


class TestScheduledResponseEmail:
    """Tests for ScheduledResponse with email modality."""

    def test_valid_email_response(self) -> None:
        """Test creating a valid email response."""
        response = ScheduledResponse(
            modality="email",
            character_name="Alice Chen",
            character_email="alice@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            content="Thanks for the update!",
            original_message_id="msg-123",
            thread_id="thread-456",
            subject="Re: Project Update",
            recipients=["user@example.com"],
        )

        assert response.modality == "email"
        assert response.character_name == "Alice Chen"
        assert response.character_email == "alice@example.com"
        assert response.content == "Thanks for the update!"
        assert response.recipients == ["user@example.com"]

    def test_email_response_with_cc(self) -> None:
        """Test email response with CC recipients."""
        response = ScheduledResponse(
            modality="email",
            character_name="Alice Chen",
            character_email="alice@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            content="Looping in the team.",
            recipients=["user@example.com"],
            cc_recipients=["team@example.com", "manager@example.com"],
        )

        assert response.cc_recipients == ["team@example.com", "manager@example.com"]

    def test_email_response_with_threading(self) -> None:
        """Test email response with full threading information."""
        response = ScheduledResponse(
            modality="email",
            character_name="Alice Chen",
            character_email="alice@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            content="Got it!",
            recipients=["user@example.com"],
            in_reply_to="msg-original",
            references=["msg-1", "msg-2", "msg-original"],
        )

        assert response.in_reply_to == "msg-original"
        assert response.references == ["msg-1", "msg-2", "msg-original"]

    def test_email_response_requires_character_email(self) -> None:
        """Test that email responses require character_email."""
        with pytest.raises(ValueError, match="Email responses require character_email"):
            ScheduledResponse(
                modality="email",
                character_name="Alice Chen",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                content="Hello!",
                recipients=["user@example.com"],
            )

    def test_email_response_requires_content(self) -> None:
        """Test that email responses require content."""
        with pytest.raises(ValueError, match="Email responses require content"):
            ScheduledResponse(
                modality="email",
                character_name="Alice Chen",
                character_email="alice@example.com",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                recipients=["user@example.com"],
            )

    def test_email_response_requires_recipients(self) -> None:
        """Test that email responses require at least one recipient."""
        with pytest.raises(
            ValueError, match="Email responses require at least one recipient"
        ):
            ScheduledResponse(
                modality="email",
                character_name="Alice Chen",
                character_email="alice@example.com",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                content="Hello!",
            )


class TestScheduledResponseSMS:
    """Tests for ScheduledResponse with SMS modality."""

    def test_valid_sms_response(self) -> None:
        """Test creating a valid SMS response."""
        response = ScheduledResponse(
            modality="sms",
            character_name="Bob Smith",
            character_phone="+15551234567",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            content="On my way!",
            original_message_id="sms-123",
            thread_id="sms-thread-456",
            recipients=["+15559876543"],
        )

        assert response.modality == "sms"
        assert response.character_name == "Bob Smith"
        assert response.character_phone == "+15551234567"
        assert response.content == "On my way!"
        assert response.recipients == ["+15559876543"]

    def test_sms_response_requires_character_phone(self) -> None:
        """Test that SMS responses require character_phone."""
        with pytest.raises(ValueError, match="SMS responses require character_phone"):
            ScheduledResponse(
                modality="sms",
                character_name="Bob Smith",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                content="Hello!",
                recipients=["+15559876543"],
            )

    def test_sms_response_requires_content(self) -> None:
        """Test that SMS responses require content."""
        with pytest.raises(ValueError, match="SMS responses require content"):
            ScheduledResponse(
                modality="sms",
                character_name="Bob Smith",
                character_phone="+15551234567",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                recipients=["+15559876543"],
            )

    def test_sms_response_requires_recipients(self) -> None:
        """Test that SMS responses require at least one recipient."""
        with pytest.raises(
            ValueError, match="SMS responses require at least one recipient"
        ):
            ScheduledResponse(
                modality="sms",
                character_name="Bob Smith",
                character_phone="+15551234567",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                content="Hello!",
            )


class TestScheduledResponseCalendar:
    """Tests for ScheduledResponse with calendar modality."""

    def test_valid_calendar_response_accepted(self) -> None:
        """Test creating a valid calendar RSVP (accepted)."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Carol Davis",
            character_email="carol@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            event_id="evt-123",
            rsvp_status="accepted",
            rsvp_comment="Looking forward to it!",
        )

        assert response.modality == "calendar"
        assert response.character_name == "Carol Davis"
        assert response.event_id == "evt-123"
        assert response.rsvp_status == "accepted"
        assert response.rsvp_comment == "Looking forward to it!"

    def test_valid_calendar_response_declined(self) -> None:
        """Test creating a valid calendar RSVP (declined)."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Carol Davis",
            character_email="carol@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            event_id="evt-123",
            rsvp_status="declined",
        )

        assert response.rsvp_status == "declined"
        assert response.rsvp_comment is None

    def test_valid_calendar_response_tentative(self) -> None:
        """Test creating a valid calendar RSVP (tentative)."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Carol Davis",
            character_email="carol@example.com",
            scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
            event_id="evt-123",
            rsvp_status="tentative",
            rsvp_comment="Need to check my schedule.",
        )

        assert response.rsvp_status == "tentative"

    def test_calendar_response_requires_character_email(self) -> None:
        """Test that calendar responses require character_email."""
        with pytest.raises(
            ValueError, match="Calendar responses require character_email"
        ):
            ScheduledResponse(
                modality="calendar",
                character_name="Carol Davis",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                event_id="evt-123",
                rsvp_status="accepted",
            )

    def test_calendar_response_requires_event_id(self) -> None:
        """Test that calendar responses require event_id."""
        with pytest.raises(ValueError, match="Calendar responses require event_id"):
            ScheduledResponse(
                modality="calendar",
                character_name="Carol Davis",
                character_email="carol@example.com",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                rsvp_status="accepted",
            )

    def test_calendar_response_requires_rsvp_status(self) -> None:
        """Test that calendar responses require rsvp_status."""
        with pytest.raises(ValueError, match="Calendar responses require rsvp_status"):
            ScheduledResponse(
                modality="calendar",
                character_name="Carol Davis",
                character_email="carol@example.com",
                scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
                event_id="evt-123",
            )


# =============================================================================
# ShouldRespondResult Tests
# =============================================================================


class TestShouldRespondResult:
    """Tests for ShouldRespondResult Pydantic model."""

    def test_parse_should_respond_true(self) -> None:
        """Test parsing a positive should-respond result."""
        result = ShouldRespondResult(
            should_respond=True,
            reasoning="The message asks a direct question.",
        )

        assert result.should_respond is True
        assert result.reasoning == "The message asks a direct question."

    def test_parse_should_respond_false(self) -> None:
        """Test parsing a negative should-respond result."""
        result = ShouldRespondResult(
            should_respond=False,
            reasoning="The message is a simple acknowledgment that doesn't require a reply.",
        )

        assert result.should_respond is False
        assert "acknowledgment" in result.reasoning

    def test_parse_from_dict(self) -> None:
        """Test parsing from a dictionary (simulating LLM JSON output)."""
        data = {
            "should_respond": True,
            "reasoning": "The sender is asking for information.",
        }
        result = ShouldRespondResult.model_validate(data)

        assert result.should_respond is True
        assert result.reasoning == "The sender is asking for information."

    def test_parse_from_json_string(self) -> None:
        """Test parsing from a JSON string."""
        json_str = '{"should_respond": false, "reasoning": "No response needed."}'
        result = ShouldRespondResult.model_validate_json(json_str)

        assert result.should_respond is False

    def test_requires_should_respond(self) -> None:
        """Test that should_respond is required."""
        with pytest.raises(ValidationError) as exc_info:
            ShouldRespondResult(reasoning="Some reasoning")  # type: ignore

        assert "should_respond" in str(exc_info.value)

    def test_requires_reasoning(self) -> None:
        """Test that reasoning is required."""
        with pytest.raises(ValidationError) as exc_info:
            ShouldRespondResult(should_respond=True)  # type: ignore

        assert "reasoning" in str(exc_info.value)

    def test_serialize_to_dict(self) -> None:
        """Test serializing to dictionary."""
        result = ShouldRespondResult(
            should_respond=True,
            reasoning="Test reasoning",
        )

        data = result.model_dump()
        assert data == {
            "should_respond": True,
            "reasoning": "Test reasoning",
        }


# =============================================================================
# CalendarRSVPResult Tests
# =============================================================================


class TestCalendarRSVPResult:
    """Tests for CalendarRSVPResult Pydantic model."""

    def test_parse_accepted(self) -> None:
        """Test parsing an accepted RSVP result."""
        result = CalendarRSVPResult(
            status="accepted",
            comment="Looking forward to it!",
            reasoning="The character typically accepts team meetings.",
        )

        assert result.status == "accepted"
        assert result.comment == "Looking forward to it!"
        assert "team meetings" in result.reasoning

    def test_parse_declined(self) -> None:
        """Test parsing a declined RSVP result."""
        result = CalendarRSVPResult(
            status="declined",
            reasoning="The character has a conflict.",
        )

        assert result.status == "declined"
        assert result.comment is None

    def test_parse_tentative(self) -> None:
        """Test parsing a tentative RSVP result."""
        result = CalendarRSVPResult(
            status="tentative",
            comment="Need to check with my manager.",
            reasoning="The character is uncertain about availability.",
        )

        assert result.status == "tentative"

    def test_parse_from_dict(self) -> None:
        """Test parsing from a dictionary (simulating LLM JSON output)."""
        data = {
            "status": "accepted",
            "comment": "Great!",
            "reasoning": "Character always accepts.",
        }
        result = CalendarRSVPResult.model_validate(data)

        assert result.status == "accepted"
        assert result.comment == "Great!"

    def test_requires_status(self) -> None:
        """Test that status is required."""
        with pytest.raises(ValidationError) as exc_info:
            CalendarRSVPResult(reasoning="Some reasoning")  # type: ignore

        assert "status" in str(exc_info.value)

    def test_requires_reasoning(self) -> None:
        """Test that reasoning is required."""
        with pytest.raises(ValidationError) as exc_info:
            CalendarRSVPResult(status="accepted")  # type: ignore

        assert "reasoning" in str(exc_info.value)

    def test_invalid_status(self) -> None:
        """Test that invalid status values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CalendarRSVPResult(
                status="maybe",  # type: ignore
                reasoning="Invalid status",
            )

        assert "status" in str(exc_info.value).lower()

    def test_comment_is_optional(self) -> None:
        """Test that comment is optional."""
        result = CalendarRSVPResult(
            status="declined",
            reasoning="Can't make it.",
        )

        assert result.comment is None


# =============================================================================
# ThreadContext Tests
# =============================================================================


class TestThreadContext:
    """Tests for ThreadContext dataclass."""

    def test_create_empty_context(self) -> None:
        """Test creating an empty thread context."""
        context = ThreadContext(
            messages=[],
            formatted_history="",
        )

        assert context.messages == []
        assert context.formatted_history == ""
        assert context.total_message_count == 0
        assert context.included_message_count == 0
        assert context.summary is None

    def test_create_with_messages(self) -> None:
        """Test creating a thread context with messages."""
        email1 = MockEmail(message_id="msg-1")
        email2 = MockEmail(message_id="msg-2")

        context = ThreadContext(
            messages=[email1, email2],  # type: ignore
            formatted_history="[10:00] Alice: Hi!\n[10:05] Bob: Hello!",
            total_message_count=2,
            included_message_count=2,
        )

        assert len(context.messages) == 2
        assert "Alice" in context.formatted_history

    def test_create_with_summary(self) -> None:
        """Test creating a thread context with truncation summary."""
        context = ThreadContext(
            messages=[MockEmail()],  # type: ignore
            formatted_history="[11:00] Recent message",
            total_message_count=15,
            included_message_count=5,
            summary="Earlier discussion covered project timeline and budget.",
        )

        assert context.total_message_count == 15
        assert context.included_message_count == 5
        assert context.summary is not None
        assert "timeline" in context.summary


# =============================================================================
# MessageContext Tests
# =============================================================================


class TestMessageContext:
    """Tests for MessageContext dataclass."""

    def test_create_email_context(self) -> None:
        """Test creating a message context for email."""
        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="alice@example.com",
            all_recipients={"bob@example.com", "carol@example.com"},
        )

        assert context.modality == "email"
        assert context.sender_address == "alice@example.com"
        assert len(context.all_recipients) == 2
        assert context.thread_context is None

    def test_create_sms_context(self) -> None:
        """Test creating a message context for SMS."""
        sms = MockSMSMessage()
        context = MessageContext(
            message=sms,  # type: ignore
            modality="sms",
            sender_address="+15551234567",
            all_recipients={"+15559876543"},
        )

        assert context.modality == "sms"
        assert context.sender_address == "+15551234567"

    def test_create_with_thread_context(self) -> None:
        """Test creating a message context with thread history."""
        thread_ctx = ThreadContext(
            messages=[],
            formatted_history="Previous messages...",
            total_message_count=5,
            included_message_count=3,
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="alice@example.com",
            thread_context=thread_ctx,
        )

        assert context.thread_context is not None
        assert context.thread_context.total_message_count == 5

    def test_default_recipients_empty(self) -> None:
        """Test that all_recipients defaults to empty set."""
        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="alice@example.com",
        )

        assert context.all_recipients == set()


# =============================================================================
# CalendarEventContext Tests
# =============================================================================


class TestCalendarEventContext:
    """Tests for CalendarEventContext dataclass."""

    def test_create_event_context(self) -> None:
        """Test creating a calendar event context."""
        event = MockCalendarEvent()
        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="organizer@example.com",
            all_attendees={"alice@example.com", "bob@example.com"},
        )

        assert context.organizer_email == "organizer@example.com"
        assert len(context.all_attendees) == 2
        assert "alice@example.com" in context.all_attendees

    def test_default_attendees_empty(self) -> None:
        """Test that all_attendees defaults to empty set."""
        event = MockCalendarEvent()
        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="organizer@example.com",
        )

        assert context.all_attendees == set()
