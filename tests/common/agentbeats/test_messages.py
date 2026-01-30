"""Tests for AgentBeats assessment message models.

Tests cover:
- Message model creation and validation
- Serialization/deserialization (round-trip)
- message_type field immutability
- parse_message utility function
- Edge cases and validation errors
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.common.agentbeats.messages import (
    AgentBeatsMessage,
    AssessmentCompleteMessage,
    AssessmentStartMessage,
    CalendarSummary,
    ChatSummary,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    MESSAGE_TYPE_REGISTRY,
    SMSSummary,
    TurnCompleteMessage,
    TurnStartMessage,
    parse_message,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_email_summary() -> EmailSummary:
    """Create a sample EmailSummary for testing."""
    return EmailSummary(
        total_emails=42,
        total_threads=15,
        unread=5,
        draft_count=2,
    )


@pytest.fixture
def sample_calendar_summary() -> CalendarSummary:
    """Create a sample CalendarSummary for testing."""
    return CalendarSummary(
        event_count=20,
        calendar_count=3,
        events_today=4,
    )


@pytest.fixture
def sample_sms_summary() -> SMSSummary:
    """Create a sample SMSSummary for testing."""
    return SMSSummary(
        total_messages=100,
        total_conversations=10,
        unread=3,
    )


@pytest.fixture
def sample_chat_summary() -> ChatSummary:
    """Create a sample ChatSummary for testing."""
    return ChatSummary(
        total_messages=50,
        conversation_count=1,
    )


@pytest.fixture
def sample_initial_state(
    sample_email_summary: EmailSummary,
    sample_calendar_summary: CalendarSummary,
    sample_sms_summary: SMSSummary,
    sample_chat_summary: ChatSummary,
) -> InitialStateSummary:
    """Create a sample InitialStateSummary for testing."""
    return InitialStateSummary(
        email=sample_email_summary,
        calendar=sample_calendar_summary,
        sms=sample_sms_summary,
        chat=sample_chat_summary,
    )


@pytest.fixture
def sample_timestamp() -> datetime:
    """Create a sample timezone-aware timestamp."""
    return datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# EmailSummary Tests
# =============================================================================


class TestEmailSummary:
    """Tests for EmailSummary model."""

    def test_create_email_summary(self, sample_email_summary: EmailSummary):
        """Test creating an EmailSummary with valid data."""
        assert sample_email_summary.message_type == "email_summary"
        assert sample_email_summary.total_emails == 42
        assert sample_email_summary.total_threads == 15
        assert sample_email_summary.unread == 5
        assert sample_email_summary.draft_count == 2

    def test_message_type_is_literal(self):
        """Test that message_type is always 'email_summary'."""
        summary = EmailSummary(
            total_emails=0, total_threads=0, unread=0, draft_count=0
        )
        assert summary.message_type == "email_summary"

    def test_message_type_cannot_be_overridden(self):
        """Test that providing a different message_type is rejected."""
        with pytest.raises(ValidationError):
            EmailSummary(
                message_type="wrong_type",  # type: ignore
                total_emails=0,
                total_threads=0,
                unread=0,
                draft_count=0,
            )

    def test_negative_values_rejected(self):
        """Test that negative counts are rejected."""
        with pytest.raises(ValidationError):
            EmailSummary(
                total_emails=-1,
                total_threads=0,
                unread=0,
                draft_count=0,
            )

    def test_serialization_round_trip(self, sample_email_summary: EmailSummary):
        """Test JSON serialization and deserialization."""
        data = sample_email_summary.model_dump(mode="json")
        restored = EmailSummary.model_validate(data)
        assert restored == sample_email_summary

    def test_model_is_frozen(self, sample_email_summary: EmailSummary):
        """Test that the model is immutable."""
        with pytest.raises(ValidationError):
            sample_email_summary.total_emails = 100  # type: ignore


# =============================================================================
# CalendarSummary Tests
# =============================================================================


class TestCalendarSummary:
    """Tests for CalendarSummary model."""

    def test_create_calendar_summary(self, sample_calendar_summary: CalendarSummary):
        """Test creating a CalendarSummary with valid data."""
        assert sample_calendar_summary.message_type == "calendar_summary"
        assert sample_calendar_summary.event_count == 20
        assert sample_calendar_summary.calendar_count == 3
        assert sample_calendar_summary.events_today == 4

    def test_message_type_is_literal(self):
        """Test that message_type is always 'calendar_summary'."""
        summary = CalendarSummary(
            event_count=0, calendar_count=0, events_today=0
        )
        assert summary.message_type == "calendar_summary"

    def test_serialization_round_trip(self, sample_calendar_summary: CalendarSummary):
        """Test JSON serialization and deserialization."""
        data = sample_calendar_summary.model_dump(mode="json")
        restored = CalendarSummary.model_validate(data)
        assert restored == sample_calendar_summary


# =============================================================================
# SMSSummary Tests
# =============================================================================


class TestSMSSummary:
    """Tests for SMSSummary model."""

    def test_create_sms_summary(self, sample_sms_summary: SMSSummary):
        """Test creating an SMSSummary with valid data."""
        assert sample_sms_summary.message_type == "sms_summary"
        assert sample_sms_summary.total_messages == 100
        assert sample_sms_summary.total_conversations == 10
        assert sample_sms_summary.unread == 3

    def test_message_type_is_literal(self):
        """Test that message_type is always 'sms_summary'."""
        summary = SMSSummary(
            total_messages=0, total_conversations=0, unread=0
        )
        assert summary.message_type == "sms_summary"

    def test_serialization_round_trip(self, sample_sms_summary: SMSSummary):
        """Test JSON serialization and deserialization."""
        data = sample_sms_summary.model_dump(mode="json")
        restored = SMSSummary.model_validate(data)
        assert restored == sample_sms_summary


# =============================================================================
# ChatSummary Tests
# =============================================================================


class TestChatSummary:
    """Tests for ChatSummary model."""

    def test_create_chat_summary(self, sample_chat_summary: ChatSummary):
        """Test creating a ChatSummary with valid data."""
        assert sample_chat_summary.message_type == "chat_summary"
        assert sample_chat_summary.total_messages == 50
        assert sample_chat_summary.conversation_count == 1

    def test_message_type_is_literal(self):
        """Test that message_type is always 'chat_summary'."""
        summary = ChatSummary(total_messages=0, conversation_count=0)
        assert summary.message_type == "chat_summary"

    def test_serialization_round_trip(self, sample_chat_summary: ChatSummary):
        """Test JSON serialization and deserialization."""
        data = sample_chat_summary.model_dump(mode="json")
        restored = ChatSummary.model_validate(data)
        assert restored == sample_chat_summary


# =============================================================================
# InitialStateSummary Tests
# =============================================================================


class TestInitialStateSummary:
    """Tests for InitialStateSummary model."""

    def test_create_initial_state(self, sample_initial_state: InitialStateSummary):
        """Test creating an InitialStateSummary with valid data."""
        assert sample_initial_state.message_type == "initial_state_summary"
        assert sample_initial_state.email.message_type == "email_summary"
        assert sample_initial_state.calendar.message_type == "calendar_summary"
        assert sample_initial_state.sms.message_type == "sms_summary"
        assert sample_initial_state.chat.message_type == "chat_summary"

    def test_nested_summaries_preserved(
        self,
        sample_initial_state: InitialStateSummary,
        sample_email_summary: EmailSummary,
    ):
        """Test that nested summaries maintain their values."""
        assert sample_initial_state.email == sample_email_summary

    def test_serialization_round_trip(self, sample_initial_state: InitialStateSummary):
        """Test JSON serialization and deserialization."""
        data = sample_initial_state.model_dump(mode="json")
        restored = InitialStateSummary.model_validate(data)
        assert restored == sample_initial_state
        # Verify nested types are correct
        assert isinstance(restored.email, EmailSummary)
        assert isinstance(restored.calendar, CalendarSummary)
        assert isinstance(restored.sms, SMSSummary)
        assert isinstance(restored.chat, ChatSummary)


# =============================================================================
# AssessmentStartMessage Tests
# =============================================================================


class TestAssessmentStartMessage:
    """Tests for AssessmentStartMessage model."""

    def test_create_assessment_start(
        self,
        sample_initial_state: InitialStateSummary,
        sample_timestamp: datetime,
    ):
        """Test creating an AssessmentStartMessage with valid data."""
        msg = AssessmentStartMessage(
            ues_url="http://localhost:8080",
            api_key="secret-key",
            assessment_instructions="Triage all unread emails.",
            current_time=sample_timestamp,
            initial_state_summary=sample_initial_state,
        )
        assert msg.message_type == "assessment_start"
        assert msg.ues_url == "http://localhost:8080"
        assert msg.api_key == "secret-key"
        assert msg.assessment_instructions == "Triage all unread emails."
        assert msg.current_time == sample_timestamp
        assert msg.initial_state_summary == sample_initial_state

    def test_serialization_round_trip(
        self,
        sample_initial_state: InitialStateSummary,
        sample_timestamp: datetime,
    ):
        """Test JSON serialization and deserialization."""
        msg = AssessmentStartMessage(
            ues_url="http://localhost:8080",
            api_key="secret-key",
            assessment_instructions="Test instructions",
            current_time=sample_timestamp,
            initial_state_summary=sample_initial_state,
        )
        data = msg.model_dump(mode="json")
        restored = AssessmentStartMessage.model_validate(data)
        # Compare field by field since datetime serialization may differ
        assert restored.message_type == msg.message_type
        assert restored.ues_url == msg.ues_url
        assert restored.api_key == msg.api_key
        assert restored.initial_state_summary == msg.initial_state_summary


# =============================================================================
# TurnStartMessage Tests
# =============================================================================


class TestTurnStartMessage:
    """Tests for TurnStartMessage model."""

    def test_create_turn_start(self, sample_timestamp: datetime):
        """Test creating a TurnStartMessage with valid data."""
        msg = TurnStartMessage(
            turn_number=1,
            current_time=sample_timestamp,
            events_processed=0,
        )
        assert msg.message_type == "turn_start"
        assert msg.turn_number == 1
        assert msg.current_time == sample_timestamp
        assert msg.events_processed == 0

    def test_turn_number_must_be_positive(self, sample_timestamp: datetime):
        """Test that turn_number must be >= 1."""
        with pytest.raises(ValidationError):
            TurnStartMessage(
                turn_number=0,
                current_time=sample_timestamp,
                events_processed=0,
            )

    def test_events_processed_cannot_be_negative(self, sample_timestamp: datetime):
        """Test that events_processed cannot be negative."""
        with pytest.raises(ValidationError):
            TurnStartMessage(
                turn_number=1,
                current_time=sample_timestamp,
                events_processed=-1,
            )

    def test_serialization_round_trip(self, sample_timestamp: datetime):
        """Test JSON serialization and deserialization."""
        msg = TurnStartMessage(
            turn_number=5,
            current_time=sample_timestamp,
            events_processed=10,
        )
        data = msg.model_dump(mode="json")
        restored = TurnStartMessage.model_validate(data)
        assert restored.turn_number == msg.turn_number
        assert restored.events_processed == msg.events_processed


# =============================================================================
# TurnCompleteMessage Tests
# =============================================================================


class TestTurnCompleteMessage:
    """Tests for TurnCompleteMessage model."""

    def test_create_turn_complete_minimal(self):
        """Test creating a TurnCompleteMessage with minimal data."""
        msg = TurnCompleteMessage()
        assert msg.message_type == "turn_complete"
        assert msg.notes is None
        assert msg.time_step == "PT1H"

    def test_create_turn_complete_with_notes(self):
        """Test creating a TurnCompleteMessage with notes."""
        msg = TurnCompleteMessage(
            notes="Processed one email.",
            time_step="PT30M",
        )
        assert msg.notes == "Processed one email."
        assert msg.time_step == "PT30M"

    def test_serialization_round_trip(self):
        """Test JSON serialization and deserialization."""
        msg = TurnCompleteMessage(
            notes="Test notes",
            time_step="PT2H",
        )
        data = msg.model_dump(mode="json")
        restored = TurnCompleteMessage.model_validate(data)
        assert restored.notes == msg.notes
        assert restored.time_step == msg.time_step


# =============================================================================
# AssessmentCompleteMessage Tests
# =============================================================================


class TestAssessmentCompleteMessage:
    """Tests for AssessmentCompleteMessage model."""

    def test_create_scenario_complete(self):
        """Test creating an AssessmentCompleteMessage for scenario_complete."""
        msg = AssessmentCompleteMessage(reason="scenario_complete")
        assert msg.message_type == "assessment_complete"
        assert msg.reason == "scenario_complete"

    def test_all_reason_values(self):
        """Test all valid reason values."""
        valid_reasons = ["scenario_complete", "early_completion", "timeout", "error"]
        for reason in valid_reasons:
            msg = AssessmentCompleteMessage(reason=reason)  # type: ignore
            assert msg.reason == reason

    def test_invalid_reason_rejected(self):
        """Test that invalid reasons are rejected."""
        with pytest.raises(ValidationError):
            AssessmentCompleteMessage(reason="invalid_reason")  # type: ignore

    def test_serialization_round_trip(self):
        """Test JSON serialization and deserialization."""
        msg = AssessmentCompleteMessage(reason="early_completion")
        data = msg.model_dump(mode="json")
        restored = AssessmentCompleteMessage.model_validate(data)
        assert restored == msg


# =============================================================================
# EarlyCompletionMessage Tests
# =============================================================================


class TestEarlyCompletionMessage:
    """Tests for EarlyCompletionMessage model."""

    def test_create_early_completion_minimal(self):
        """Test creating an EarlyCompletionMessage with no reason."""
        msg = EarlyCompletionMessage()
        assert msg.message_type == "early_completion"
        assert msg.reason is None

    def test_create_early_completion_with_reason(self):
        """Test creating an EarlyCompletionMessage with a reason."""
        msg = EarlyCompletionMessage(reason="All tasks completed.")
        assert msg.reason == "All tasks completed."

    def test_serialization_round_trip(self):
        """Test JSON serialization and deserialization."""
        msg = EarlyCompletionMessage(reason="Done early")
        data = msg.model_dump(mode="json")
        restored = EarlyCompletionMessage.model_validate(data)
        assert restored == msg


# =============================================================================
# parse_message Tests
# =============================================================================


class TestParseMessage:
    """Tests for the parse_message utility function."""

    def test_parse_email_summary(self):
        """Test parsing EmailSummary from dict."""
        data = {
            "message_type": "email_summary",
            "total_emails": 42,
            "total_threads": 15,
            "unread": 5,
            "draft_count": 2,
        }
        msg = parse_message(data)
        assert isinstance(msg, EmailSummary)
        assert msg.total_emails == 42

    def test_parse_turn_start_message(self):
        """Test parsing TurnStartMessage from dict."""
        data = {
            "message_type": "turn_start",
            "turn_number": 3,
            "current_time": "2026-01-28T12:00:00Z",
            "events_processed": 5,
        }
        msg = parse_message(data)
        assert isinstance(msg, TurnStartMessage)
        assert msg.turn_number == 3

    def test_parse_assessment_complete(self):
        """Test parsing AssessmentCompleteMessage from dict."""
        data = {
            "message_type": "assessment_complete",
            "reason": "timeout",
        }
        msg = parse_message(data)
        assert isinstance(msg, AssessmentCompleteMessage)
        assert msg.reason == "timeout"

    def test_parse_missing_message_type(self):
        """Test that missing message_type raises ValueError."""
        data = {"total_emails": 42}
        with pytest.raises(ValueError, match="must include 'message_type'"):
            parse_message(data)

    def test_parse_unknown_message_type(self):
        """Test that unknown message_type raises ValueError."""
        data = {"message_type": "unknown_type"}
        with pytest.raises(ValueError, match="Unknown message_type 'unknown_type'"):
            parse_message(data)

    def test_parse_all_registered_types(self, sample_timestamp: datetime):
        """Test that all registered types can be parsed."""
        # Create sample data for each message type
        samples: dict[str, dict] = {
            "email_summary": {
                "message_type": "email_summary",
                "total_emails": 0,
                "total_threads": 0,
                "unread": 0,
                "draft_count": 0,
            },
            "calendar_summary": {
                "message_type": "calendar_summary",
                "event_count": 0,
                "calendar_count": 0,
                "events_today": 0,
            },
            "sms_summary": {
                "message_type": "sms_summary",
                "total_messages": 0,
                "total_conversations": 0,
                "unread": 0,
            },
            "chat_summary": {
                "message_type": "chat_summary",
                "total_messages": 0,
                "conversation_count": 0,
            },
            "turn_start": {
                "message_type": "turn_start",
                "turn_number": 1,
                "current_time": sample_timestamp.isoformat(),
                "events_processed": 0,
            },
            "turn_complete": {
                "message_type": "turn_complete",
            },
            "assessment_complete": {
                "message_type": "assessment_complete",
                "reason": "scenario_complete",
            },
            "early_completion": {
                "message_type": "early_completion",
            },
        }

        for message_type, data in samples.items():
            msg = parse_message(data)
            expected_class = MESSAGE_TYPE_REGISTRY[message_type]
            assert isinstance(msg, expected_class), (
                f"Expected {expected_class.__name__} for {message_type}"
            )


# =============================================================================
# Message Type Registry Tests
# =============================================================================


class TestMessageTypeRegistry:
    """Tests for the MESSAGE_TYPE_REGISTRY."""

    def test_all_message_types_registered(self):
        """Test that all expected message types are in the registry."""
        expected_types = {
            "email_summary",
            "calendar_summary",
            "sms_summary",
            "chat_summary",
            "initial_state_summary",
            "assessment_start",
            "turn_start",
            "turn_complete",
            "assessment_complete",
            "early_completion",
        }
        assert set(MESSAGE_TYPE_REGISTRY.keys()) == expected_types

    def test_registry_values_are_model_classes(self):
        """Test that all registry values are Pydantic model classes."""
        from pydantic import BaseModel

        for message_type, model_class in MESSAGE_TYPE_REGISTRY.items():
            assert issubclass(model_class, BaseModel), (
                f"{message_type} is not a BaseModel subclass"
            )
