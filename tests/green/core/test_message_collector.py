"""Tests for the NewMessageCollector module.

Tests cover:
- NewMessages dataclass properties and methods
- NewMessageCollector lifecycle (initialize, collect, reset)
- Error handling (not initialized)
- Edge cases (empty results, calendar-only changes, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from src.green.core.message_collector import (
    CollectorNotInitializedError,
    MessageCollectorError,
    NewMessageCollector,
    NewMessages,
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================


@dataclass
class MockEmail:
    """Mock Email object for testing."""

    message_id: str
    thread_id: str
    from_address: str
    to_addresses: list[str]
    subject: str
    body_text: str
    received_at: datetime


@dataclass
class MockSMSMessage:
    """Mock SMSMessage object for testing."""

    message_id: str
    thread_id: str
    from_number: str
    to_numbers: list[str]
    body: str
    direction: str
    sent_at: datetime


@dataclass
class MockCalendarEvent:
    """Mock CalendarEvent object for testing."""

    event_id: str
    calendar_id: str
    title: str
    start: datetime
    end: datetime
    attendees: list[dict[str, Any]]


@dataclass
class MockEmailQueryResponse:
    """Mock EmailQueryResponse for testing."""

    emails: list[MockEmail]


@dataclass
class MockSMSQueryResponse:
    """Mock SMSQueryResponse for testing."""

    messages: list[MockSMSMessage]


@dataclass
class MockCalendarQueryResponse:
    """Mock CalendarQueryResponse for testing."""

    events: list[MockCalendarEvent]


def make_mock_email(
    message_id: str = "msg-1",
    thread_id: str = "thread-1",
    from_address: str = "sender@example.com",
    to_addresses: list[str] | None = None,
    subject: str = "Test Subject",
    body_text: str = "Test body",
    received_at: datetime | None = None,
) -> MockEmail:
    """Create a mock email for testing."""
    return MockEmail(
        message_id=message_id,
        thread_id=thread_id,
        from_address=from_address,
        to_addresses=to_addresses or ["recipient@example.com"],
        subject=subject,
        body_text=body_text,
        received_at=received_at or datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
    )


def make_mock_sms(
    message_id: str = "sms-1",
    thread_id: str = "sms-thread-1",
    from_number: str = "+15551234567",
    to_numbers: list[str] | None = None,
    body: str = "Test SMS",
    direction: str = "incoming",
    sent_at: datetime | None = None,
) -> MockSMSMessage:
    """Create a mock SMS message for testing."""
    return MockSMSMessage(
        message_id=message_id,
        thread_id=thread_id,
        from_number=from_number,
        to_numbers=to_numbers or ["+15559876543"],
        body=body,
        direction=direction,
        sent_at=sent_at or datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
    )


def make_mock_calendar_event(
    event_id: str = "evt-1",
    calendar_id: str = "cal-1",
    title: str = "Test Event",
    start: datetime | None = None,
    end: datetime | None = None,
    attendees: list[dict[str, Any]] | None = None,
) -> MockCalendarEvent:
    """Create a mock calendar event for testing."""
    return MockCalendarEvent(
        event_id=event_id,
        calendar_id=calendar_id,
        title=title,
        start=start or datetime(2026, 1, 29, 14, 0, tzinfo=timezone.utc),
        end=end or datetime(2026, 1, 29, 15, 0, tzinfo=timezone.utc),
        attendees=attendees or [],
    )


@pytest.fixture
def mock_client() -> AsyncMock:
    """Create a mock AsyncUESClient."""
    client = AsyncMock()

    # Set up email client mock
    client.email = AsyncMock()
    client.email.query = AsyncMock(
        return_value=MockEmailQueryResponse(emails=[])
    )

    # Set up SMS client mock
    client.sms = AsyncMock()
    client.sms.query = AsyncMock(
        return_value=MockSMSQueryResponse(messages=[])
    )

    # Set up calendar client mock
    client.calendar = AsyncMock()
    client.calendar.query = AsyncMock(
        return_value=MockCalendarQueryResponse(events=[])
    )

    return client


@pytest.fixture
def collector(mock_client: AsyncMock) -> NewMessageCollector:
    """Create a NewMessageCollector with mock client."""
    return NewMessageCollector(mock_client)


@pytest.fixture
def now() -> datetime:
    """Return a fixed datetime for testing."""
    return datetime(2026, 1, 29, 10, 0, 0, tzinfo=timezone.utc)


# =============================================================================
# NewMessages Tests
# =============================================================================


class TestNewMessages:
    """Tests for the NewMessages dataclass."""

    def test_empty_by_default(self) -> None:
        """NewMessages should have empty lists by default."""
        messages = NewMessages()
        assert messages.emails == []
        assert messages.sms_messages == []
        assert messages.calendar_events == []

    def test_total_count_empty(self) -> None:
        """total_count should be 0 for empty NewMessages."""
        messages = NewMessages()
        assert messages.total_count == 0

    def test_total_count_with_items(self) -> None:
        """total_count should sum all message lists."""
        messages = NewMessages(
            emails=[make_mock_email("e1"), make_mock_email("e2")],
            sms_messages=[make_mock_sms("s1")],
            calendar_events=[make_mock_calendar_event("c1"), make_mock_calendar_event("c2")],
        )
        assert messages.total_count == 5

    def test_individual_counts(self) -> None:
        """Individual count properties should return correct values."""
        messages = NewMessages(
            emails=[make_mock_email("e1"), make_mock_email("e2")],
            sms_messages=[make_mock_sms("s1")],
            calendar_events=[],
        )
        assert messages.email_count == 2
        assert messages.sms_count == 1
        assert messages.calendar_count == 0

    def test_is_empty_true(self) -> None:
        """is_empty should return True when all lists are empty."""
        messages = NewMessages()
        assert messages.is_empty() is True

    def test_is_empty_false_with_email(self) -> None:
        """is_empty should return False when emails exist."""
        messages = NewMessages(emails=[make_mock_email()])
        assert messages.is_empty() is False

    def test_is_empty_false_with_sms(self) -> None:
        """is_empty should return False when SMS messages exist."""
        messages = NewMessages(sms_messages=[make_mock_sms()])
        assert messages.is_empty() is False

    def test_is_empty_false_with_calendar(self) -> None:
        """is_empty should return False when calendar events exist."""
        messages = NewMessages(calendar_events=[make_mock_calendar_event()])
        assert messages.is_empty() is False

    def test_repr_empty(self) -> None:
        """__repr__ should show zeros for empty NewMessages."""
        messages = NewMessages()
        assert repr(messages) == "NewMessages(emails=0, sms=0, calendar=0)"

    def test_repr_with_items(self) -> None:
        """__repr__ should show correct counts."""
        messages = NewMessages(
            emails=[make_mock_email()],
            sms_messages=[make_mock_sms(), make_mock_sms()],
            calendar_events=[],
        )
        assert repr(messages) == "NewMessages(emails=1, sms=2, calendar=0)"


# =============================================================================
# NewMessageCollector Initialization Tests
# =============================================================================


class TestCollectorInitialization:
    """Tests for NewMessageCollector initialization."""

    def test_not_initialized_by_default(
        self, collector: NewMessageCollector
    ) -> None:
        """Collector should not be initialized by default."""
        assert collector.is_initialized is False

    def test_last_check_time_none_by_default(
        self, collector: NewMessageCollector
    ) -> None:
        """last_check_time should be None before initialization."""
        assert collector.last_check_time is None

    def test_seen_calendar_count_zero_by_default(
        self, collector: NewMessageCollector
    ) -> None:
        """seen_calendar_event_count should be 0 before initialization."""
        assert collector.seen_calendar_event_count == 0

    async def test_initialize_sets_flag(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """initialize() should set is_initialized to True."""
        await collector.initialize(now)
        assert collector.is_initialized is True

    async def test_initialize_sets_last_check_time(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """initialize() should set last_check_time."""
        await collector.initialize(now)
        assert collector.last_check_time == now

    async def test_initialize_captures_calendar_events(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """initialize() should capture existing calendar event IDs."""
        existing_events = [
            make_mock_calendar_event("evt-1"),
            make_mock_calendar_event("evt-2"),
            make_mock_calendar_event("evt-3"),
        ]
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=existing_events
        )

        await collector.initialize(now)

        assert collector.seen_calendar_event_count == 3

    async def test_initialize_queries_calendar(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """initialize() should query calendar for existing events."""
        await collector.initialize(now)

        mock_client.calendar.query.assert_called_once()

    async def test_reinitialize_resets_state(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """Calling initialize() again should reset state."""
        # First initialization with some events
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("evt-1")]
        )
        await collector.initialize(now)
        assert collector.seen_calendar_event_count == 1

        # Second initialization with different events
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("evt-2"), make_mock_calendar_event("evt-3")]
        )
        new_time = now + timedelta(hours=1)
        await collector.initialize(new_time)

        assert collector.seen_calendar_event_count == 2
        assert collector.last_check_time == new_time


# =============================================================================
# NewMessageCollector Collection Tests
# =============================================================================


class TestCollectorCollection:
    """Tests for NewMessageCollector.collect() method."""

    async def test_collect_raises_if_not_initialized(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """collect() should raise CollectorNotInitializedError if not initialized."""
        with pytest.raises(CollectorNotInitializedError) as exc_info:
            await collector.collect(now)

        assert "not initialized" in str(exc_info.value).lower()

    async def test_collect_returns_new_messages(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should return NewMessages container."""
        await collector.initialize(now)
        result = await collector.collect(now + timedelta(hours=1))

        assert isinstance(result, NewMessages)

    async def test_collect_queries_email_with_received_after(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should query emails with received_after filter."""
        await collector.initialize(now)

        new_time = now + timedelta(hours=1)
        await collector.collect(new_time)

        mock_client.email.query.assert_called_with(
            received_after=now,
            sort_order="asc",
        )

    async def test_collect_queries_sms_with_sent_after(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should query SMS with sent_after filter."""
        await collector.initialize(now)

        new_time = now + timedelta(hours=1)
        await collector.collect(new_time)

        mock_client.sms.query.assert_called_with(
            sent_after=now,
            sort_order="asc",
        )

    async def test_collect_returns_new_emails(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should return new emails."""
        await collector.initialize(now)

        new_emails = [
            make_mock_email("email-1"),
            make_mock_email("email-2"),
        ]
        mock_client.email.query.return_value = MockEmailQueryResponse(
            emails=new_emails
        )

        result = await collector.collect(now + timedelta(hours=1))

        assert result.email_count == 2
        assert result.emails[0].message_id == "email-1"
        assert result.emails[1].message_id == "email-2"

    async def test_collect_returns_new_sms(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should return new SMS messages."""
        await collector.initialize(now)

        new_sms = [make_mock_sms("sms-1")]
        mock_client.sms.query.return_value = MockSMSQueryResponse(messages=new_sms)

        result = await collector.collect(now + timedelta(hours=1))

        assert result.sms_count == 1
        assert result.sms_messages[0].message_id == "sms-1"

    async def test_collect_identifies_new_calendar_events(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should identify new calendar events by comparing IDs."""
        # Initialize with one existing event
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("existing-1")]
        )
        await collector.initialize(now)

        # Now there are two events (one existing, one new)
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[
                make_mock_calendar_event("existing-1"),
                make_mock_calendar_event("new-1"),
            ]
        )

        result = await collector.collect(now + timedelta(hours=1))

        assert result.calendar_count == 1
        assert result.calendar_events[0].event_id == "new-1"

    async def test_collect_updates_last_check_time(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should update last_check_time."""
        await collector.initialize(now)

        new_time = now + timedelta(hours=1)
        await collector.collect(new_time)

        assert collector.last_check_time == new_time

    async def test_collect_updates_seen_calendar_ids(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should update seen calendar event IDs."""
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(events=[])
        await collector.initialize(now)

        # Add new events
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("evt-1"), make_mock_calendar_event("evt-2")]
        )
        await collector.collect(now + timedelta(hours=1))

        assert collector.seen_calendar_event_count == 2

    async def test_sequential_collections(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """Sequential collect() calls should work correctly."""
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(events=[])
        await collector.initialize(now)

        # First collection: 1 email
        mock_client.email.query.return_value = MockEmailQueryResponse(
            emails=[make_mock_email("email-1")]
        )
        result1 = await collector.collect(now + timedelta(hours=1))
        assert result1.email_count == 1

        # Second collection: 2 more emails
        mock_client.email.query.return_value = MockEmailQueryResponse(
            emails=[make_mock_email("email-2"), make_mock_email("email-3")]
        )
        result2 = await collector.collect(now + timedelta(hours=2))
        assert result2.email_count == 2

        # Verify received_after was updated between calls
        calls = mock_client.email.query.call_args_list
        assert calls[-2].kwargs["received_after"] == now  # First collect
        assert calls[-1].kwargs["received_after"] == now + timedelta(hours=1)  # Second collect


# =============================================================================
# NewMessageCollector Reset Tests
# =============================================================================


class TestCollectorReset:
    """Tests for NewMessageCollector.reset() method."""

    async def test_reset_clears_initialization(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """reset() should clear is_initialized flag."""
        await collector.initialize(now)
        assert collector.is_initialized is True

        collector.reset()

        assert collector.is_initialized is False

    async def test_reset_clears_last_check_time(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """reset() should clear last_check_time."""
        await collector.initialize(now)
        assert collector.last_check_time is not None

        collector.reset()

        assert collector.last_check_time is None

    async def test_reset_clears_seen_calendar_ids(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """reset() should clear seen calendar event IDs."""
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("evt-1")]
        )
        await collector.initialize(now)
        assert collector.seen_calendar_event_count == 1

        collector.reset()

        assert collector.seen_calendar_event_count == 0

    async def test_collect_fails_after_reset(
        self, collector: NewMessageCollector, now: datetime
    ) -> None:
        """collect() should fail after reset() until re-initialized."""
        await collector.initialize(now)
        collector.reset()

        with pytest.raises(CollectorNotInitializedError):
            await collector.collect(now)

    async def test_reinitialize_after_reset(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """Should be able to reinitialize after reset."""
        await collector.initialize(now)
        collector.reset()

        new_time = now + timedelta(hours=1)
        await collector.initialize(new_time)

        assert collector.is_initialized is True
        assert collector.last_check_time == new_time


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    async def test_collect_with_no_new_messages(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should return empty NewMessages when nothing is new."""
        await collector.initialize(now)

        result = await collector.collect(now + timedelta(hours=1))

        assert result.is_empty() is True
        assert result.total_count == 0

    async def test_collect_with_only_calendar_changes(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should work when only calendar has changes."""
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(events=[])
        await collector.initialize(now)

        # No new emails or SMS, but new calendar event
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[make_mock_calendar_event("new-evt")]
        )

        result = await collector.collect(now + timedelta(hours=1))

        assert result.email_count == 0
        assert result.sms_count == 0
        assert result.calendar_count == 1

    async def test_deleted_calendar_events_not_reported_as_new(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """Deleted calendar events should not cause issues."""
        # Start with two events
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[
                make_mock_calendar_event("evt-1"),
                make_mock_calendar_event("evt-2"),
            ]
        )
        await collector.initialize(now)

        # One event deleted, one new added
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=[
                make_mock_calendar_event("evt-1"),  # Still exists
                make_mock_calendar_event("evt-3"),  # New
            ]
        )

        result = await collector.collect(now + timedelta(hours=1))

        # Only evt-3 is new
        assert result.calendar_count == 1
        assert result.calendar_events[0].event_id == "evt-3"

    async def test_many_new_calendar_events(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """collect() should handle many new calendar events."""
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(events=[])
        await collector.initialize(now)

        # Add 100 new events
        many_events = [make_mock_calendar_event(f"evt-{i}") for i in range(100)]
        mock_client.calendar.query.return_value = MockCalendarQueryResponse(
            events=many_events
        )

        result = await collector.collect(now + timedelta(hours=1))

        assert result.calendar_count == 100

    async def test_email_list_is_mutable_copy(
        self, collector: NewMessageCollector, mock_client: AsyncMock, now: datetime
    ) -> None:
        """Returned email list should be a copy, not the original."""
        await collector.initialize(now)

        original_emails = [make_mock_email("email-1")]
        mock_client.email.query.return_value = MockEmailQueryResponse(
            emails=original_emails
        )

        result = await collector.collect(now + timedelta(hours=1))

        # Modify the returned list
        result.emails.append(make_mock_email("email-2"))

        # Original should be unchanged
        assert len(original_emails) == 1


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_message_collector_error_is_exception(self) -> None:
        """MessageCollectorError should be an Exception."""
        assert issubclass(MessageCollectorError, Exception)

    def test_collector_not_initialized_error_inheritance(self) -> None:
        """CollectorNotInitializedError should inherit from MessageCollectorError."""
        assert issubclass(CollectorNotInitializedError, MessageCollectorError)

    def test_collector_not_initialized_error_message(self) -> None:
        """CollectorNotInitializedError should accept custom message."""
        error = CollectorNotInitializedError("Custom error message")
        assert str(error) == "Custom error message"
