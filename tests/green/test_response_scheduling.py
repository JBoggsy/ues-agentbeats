"""Tests for GreenAgent response scheduling methods.

This module provides unit tests for the response scheduling methods of the
``GreenAgent`` class:

    - ``_schedule_response``: Dispatcher for modality-specific scheduling
    - ``_schedule_email_response``: Schedule email responses via UES
    - ``_schedule_sms_response``: Schedule SMS responses via UES
    - ``_schedule_calendar_response``: Schedule calendar RSVP responses via UES

Tests use mock UES client instances to verify correct API calls are made
without requiring a running UES server.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.green.agent import GreenAgent
from src.green.response.models import ScheduledResponse


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ues_client() -> MagicMock:
    """Create a mock AsyncUESClient with all modality sub-clients."""
    client = MagicMock()

    # Email sub-client
    client.email = MagicMock()
    client.email.receive = AsyncMock()

    # SMS sub-client
    client.sms = MagicMock()
    client.sms.receive = AsyncMock()

    # Calendar sub-client
    client.calendar = MagicMock()
    client.calendar.respond_to_event = AsyncMock()

    return client


@pytest.fixture
def green_agent(mock_ues_client: MagicMock) -> GreenAgent:
    """Create a GreenAgent with a mocked UES client.

    This fixture creates a GreenAgent by bypassing __init__ and directly
    setting the ues_client attribute to a mock.
    """
    # Create a minimal GreenAgent without calling __init__
    agent = object.__new__(GreenAgent)
    agent.ues_client = mock_ues_client
    return agent


@pytest.fixture
def email_response() -> ScheduledResponse:
    """Create a sample email ScheduledResponse."""
    return ScheduledResponse(
        modality="email",
        character_name="Alice Chen",
        character_email="alice.chen@company.com",
        scheduled_time=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
        content="Thanks for the update!",
        original_message_id="msg-123",
        thread_id="thread-456",
        subject="Re: Project Update",
        recipients=["user@example.com"],
        cc_recipients=["bob@company.com"],
        in_reply_to="<msg-122@example.com>",
        references=["<msg-121@example.com>", "<msg-122@example.com>"],
    )


@pytest.fixture
def sms_response() -> ScheduledResponse:
    """Create a sample SMS ScheduledResponse."""
    return ScheduledResponse(
        modality="sms",
        character_name="Bob Smith",
        character_phone="+15551234567",
        scheduled_time=datetime(2026, 1, 29, 11, 0, tzinfo=timezone.utc),
        content="Got it, thanks!",
        original_message_id="sms-456",
        thread_id="thread-789",
        recipients=["+15559876543"],
    )


@pytest.fixture
def calendar_response() -> ScheduledResponse:
    """Create a sample calendar RSVP ScheduledResponse."""
    return ScheduledResponse(
        modality="calendar",
        character_name="Carol Davis",
        character_email="carol.davis@company.com",
        scheduled_time=datetime(2026, 1, 29, 12, 0, tzinfo=timezone.utc),
        event_id="evt-789",
        rsvp_status="accepted",
        rsvp_comment="Looking forward to it!",
    )


# =============================================================================
# _schedule_response Tests
# =============================================================================


class TestScheduleResponse:
    """Tests for the _schedule_response dispatcher method."""

    @pytest.mark.asyncio
    async def test_dispatches_email_modality(
        self,
        green_agent: GreenAgent,
        email_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """Email modality dispatches to _schedule_email_response."""
        await green_agent._schedule_response(email_response)

        mock_ues_client.email.receive.assert_called_once()
        mock_ues_client.sms.receive.assert_not_called()
        mock_ues_client.calendar.respond_to_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_sms_modality(
        self,
        green_agent: GreenAgent,
        sms_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """SMS modality dispatches to _schedule_sms_response."""
        await green_agent._schedule_response(sms_response)

        mock_ues_client.sms.receive.assert_called_once()
        mock_ues_client.email.receive.assert_not_called()
        mock_ues_client.calendar.respond_to_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_calendar_modality(
        self,
        green_agent: GreenAgent,
        calendar_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """Calendar modality dispatches to _schedule_calendar_response."""
        await green_agent._schedule_response(calendar_response)

        mock_ues_client.calendar.respond_to_event.assert_called_once()
        mock_ues_client.email.receive.assert_not_called()
        mock_ues_client.sms.receive.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_unknown_modality(
        self,
        green_agent: GreenAgent,
    ) -> None:
        """Unknown modality raises ValueError."""
        # Create a response with an invalid modality by bypassing validation
        response = object.__new__(ScheduledResponse)
        response.modality = "fax"  # type: ignore[assignment]
        response.character_name = "Test"
        response.scheduled_time = datetime.now(tz=timezone.utc)

        with pytest.raises(ValueError, match="Unknown modality: fax"):
            await green_agent._schedule_response(response)


# =============================================================================
# _schedule_email_response Tests
# =============================================================================


class TestScheduleEmailResponse:
    """Tests for the _schedule_email_response method."""

    @pytest.mark.asyncio
    async def test_calls_email_receive_with_correct_args(
        self,
        green_agent: GreenAgent,
        email_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """Email receive is called with all required arguments."""
        await green_agent._schedule_email_response(email_response)

        mock_ues_client.email.receive.assert_called_once_with(
            from_address="alice.chen@company.com",
            to_addresses=["user@example.com"],
            subject="Re: Project Update",
            body_text="Thanks for the update!",
            cc_addresses=["bob@company.com"],
            thread_id="thread-456",
            in_reply_to="<msg-122@example.com>",
            references=["<msg-121@example.com>", "<msg-122@example.com>"],
            sent_at=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_handles_empty_optional_fields(
        self,
        green_agent: GreenAgent,
        mock_ues_client: MagicMock,
    ) -> None:
        """Handles None values for optional fields correctly."""
        response = ScheduledResponse(
            modality="email",
            character_name="Alice Chen",
            character_email="alice@company.com",
            scheduled_time=datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
            content="Hello!",
            recipients=["user@example.com"],
            # No optional fields set
        )

        await green_agent._schedule_email_response(response)

        mock_ues_client.email.receive.assert_called_once_with(
            from_address="alice@company.com",
            to_addresses=["user@example.com"],
            subject="",
            body_text="Hello!",
            cc_addresses=None,
            thread_id=None,
            in_reply_to=None,
            references=None,
            sent_at=datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc),
        )


# =============================================================================
# _schedule_sms_response Tests
# =============================================================================


class TestScheduleSmsResponse:
    """Tests for the _schedule_sms_response method."""

    @pytest.mark.asyncio
    async def test_calls_sms_receive_with_correct_args(
        self,
        green_agent: GreenAgent,
        sms_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """SMS receive is called with all required arguments."""
        await green_agent._schedule_sms_response(sms_response)

        mock_ues_client.sms.receive.assert_called_once_with(
            from_number="+15551234567",
            to_numbers=["+15559876543"],
            body="Got it, thanks!",
            replied_to_message_id="sms-456",
            sent_at=datetime(2026, 1, 29, 11, 0, tzinfo=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_handles_no_reply_id(
        self,
        green_agent: GreenAgent,
        mock_ues_client: MagicMock,
    ) -> None:
        """Handles None reply message ID correctly."""
        response = ScheduledResponse(
            modality="sms",
            character_name="Bob Smith",
            character_phone="+15551234567",
            scheduled_time=datetime(2026, 1, 29, 11, 0, tzinfo=timezone.utc),
            content="Starting a new conversation",
            recipients=["+15559876543"],
            # No original_message_id
        )

        await green_agent._schedule_sms_response(response)

        mock_ues_client.sms.receive.assert_called_once_with(
            from_number="+15551234567",
            to_numbers=["+15559876543"],
            body="Starting a new conversation",
            replied_to_message_id=None,
            sent_at=datetime(2026, 1, 29, 11, 0, tzinfo=timezone.utc),
        )


# =============================================================================
# _schedule_calendar_response Tests
# =============================================================================


class TestScheduleCalendarResponse:
    """Tests for the _schedule_calendar_response method."""

    @pytest.mark.asyncio
    async def test_calls_respond_to_event_with_correct_args(
        self,
        green_agent: GreenAgent,
        calendar_response: ScheduledResponse,
        mock_ues_client: MagicMock,
    ) -> None:
        """Calendar respond_to_event is called with all required arguments."""
        await green_agent._schedule_calendar_response(calendar_response)

        mock_ues_client.calendar.respond_to_event.assert_called_once_with(
            event_id="evt-789",
            attendee_email="carol.davis@company.com",
            response="accepted",
            comment="Looking forward to it!",
        )

    @pytest.mark.asyncio
    async def test_handles_declined_rsvp(
        self,
        green_agent: GreenAgent,
        mock_ues_client: MagicMock,
    ) -> None:
        """Handles declined RSVP status correctly."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Dave Wilson",
            character_email="dave@company.com",
            scheduled_time=datetime(2026, 1, 29, 12, 0, tzinfo=timezone.utc),
            event_id="evt-999",
            rsvp_status="declined",
            rsvp_comment="Sorry, I have a conflict.",
        )

        await green_agent._schedule_calendar_response(response)

        mock_ues_client.calendar.respond_to_event.assert_called_once_with(
            event_id="evt-999",
            attendee_email="dave@company.com",
            response="declined",
            comment="Sorry, I have a conflict.",
        )

    @pytest.mark.asyncio
    async def test_handles_tentative_without_comment(
        self,
        green_agent: GreenAgent,
        mock_ues_client: MagicMock,
    ) -> None:
        """Handles tentative RSVP without comment correctly."""
        response = ScheduledResponse(
            modality="calendar",
            character_name="Eve Brown",
            character_email="eve@company.com",
            scheduled_time=datetime(2026, 1, 29, 12, 0, tzinfo=timezone.utc),
            event_id="evt-888",
            rsvp_status="tentative",
            # No comment
        )

        await green_agent._schedule_calendar_response(response)

        mock_ues_client.calendar.respond_to_event.assert_called_once_with(
            event_id="evt-888",
            attendee_email="eve@company.com",
            response="tentative",
            comment=None,
        )
