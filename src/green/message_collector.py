"""New message collector for Green agent response generation.

This module provides a collector class that queries UES modality states for
new messages since the last check. Unlike querying the event log, this approach
gives direct access to full message objects with all fields needed for response
generation (including `message_id` and `thread_id`).

The collector supports three modalities:
- **Email**: Queries by `received_after` timestamp
- **SMS**: Queries by `sent_after` timestamp  
- **Calendar**: Compares event IDs (no creation-time filter available)

Classes:
    NewMessages: Container for collected messages from all modalities.
    NewMessageCollector: Collects new messages from UES since a timestamp.
    MessageCollectorError: Base exception for collector errors.
    CollectorNotInitializedError: Raised when collect() is called before initialize().

Example:
    >>> from datetime import datetime, timezone
    >>> from ues.client import AsyncUESClient
    >>> from src.green.message_collector import NewMessageCollector
    >>>
    >>> async with AsyncUESClient() as client:
    ...     collector = NewMessageCollector(client)
    ...     
    ...     # Initialize at assessment start
    ...     current_time = datetime(2026, 1, 29, 9, 0, tzinfo=timezone.utc)
    ...     await collector.initialize(current_time)
    ...     
    ...     # After time advances, collect new messages
    ...     new_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)
    ...     new_messages = await collector.collect(new_time)
    ...     
    ...     # Process new messages
    ...     for email in new_messages.emails:
    ...         print(f"New email: {email.subject}")
    ...     for sms in new_messages.sms_messages:
    ...         print(f"New SMS from {sms.from_number}")

Design Rationale:
    The previous approach of querying UES events had two problems:
    1. Events may not include `message_id` or `thread_id` for emails/SMS
    2. Most events aren't response-triggering, making event scanning wasteful
    
    By querying modality states directly, we:
    - Get full message objects with all fields
    - Only retrieve items that could trigger responses
    - Have reliable threading information for context retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ues.client import AsyncUESClient, CalendarEvent, Email, SMSMessage


# =============================================================================
# Exceptions
# =============================================================================


class MessageCollectorError(Exception):
    """Base exception for message collector errors.

    Attributes:
        message: Human-readable error description.
    """

    pass


class CollectorNotInitializedError(MessageCollectorError):
    """Raised when collect() is called before initialize().

    The collector must be initialized with a baseline time and calendar
    event IDs before collecting new messages.

    Attributes:
        message: Human-readable error description.
    """

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NewMessages:
    """Container for new messages collected from UES modalities.

    This dataclass holds the results of a collection operation, grouping
    new items by modality type. All lists are empty by default.

    Attributes:
        emails: New emails received since last check.
        sms_messages: New SMS messages since last check.
        calendar_events: New calendar events since last check.

    Example:
        >>> messages = NewMessages()
        >>> messages.is_empty()
        True
        >>> messages.emails.append(some_email)
        >>> messages.total_count
        1
    """

    emails: list[Email] = field(default_factory=list)
    sms_messages: list[SMSMessage] = field(default_factory=list)
    calendar_events: list[CalendarEvent] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Return total count of all new messages across modalities.

        Returns:
            Sum of emails, SMS messages, and calendar events.
        """
        return len(self.emails) + len(self.sms_messages) + len(self.calendar_events)

    @property
    def email_count(self) -> int:
        """Return count of new emails.

        Returns:
            Number of new emails collected.
        """
        return len(self.emails)

    @property
    def sms_count(self) -> int:
        """Return count of new SMS messages.

        Returns:
            Number of new SMS messages collected.
        """
        return len(self.sms_messages)

    @property
    def calendar_count(self) -> int:
        """Return count of new calendar events.

        Returns:
            Number of new calendar events collected.
        """
        return len(self.calendar_events)

    def is_empty(self) -> bool:
        """Check if no new messages were collected.

        Returns:
            True if all message lists are empty, False otherwise.
        """
        return self.total_count == 0

    def __repr__(self) -> str:
        """Return string representation showing counts.

        Returns:
            String like "NewMessages(emails=2, sms=1, calendar=0)".
        """
        return (
            f"NewMessages(emails={self.email_count}, "
            f"sms={self.sms_count}, "
            f"calendar={self.calendar_count})"
        )


# =============================================================================
# Collector Class
# =============================================================================


class NewMessageCollector:
    """Collects new messages from UES modalities since a timestamp.

    This class queries UES modality states for new emails, SMS messages,
    and calendar events that have appeared since the last collection.
    It maintains internal state to track what's been seen.

    The collector uses different strategies per modality:
    - **Email**: Query with `received_after` filter
    - **SMS**: Query with `sent_after` filter
    - **Calendar**: Compare current event IDs against previously seen IDs
      (no creation-time filter is available for calendar)

    Lifecycle:
        1. Create with UES client
        2. Call `initialize()` to set baseline (records time, captures event IDs)
        3. After each time advance, call `collect()` to get new messages
        4. Call `reset()` to clear state for a new assessment

    Attributes:
        is_initialized: Whether the collector has been initialized.
        last_check_time: The timestamp of the last collection (or initialization).

    Example:
        >>> collector = NewMessageCollector(ues_client)
        >>> await collector.initialize(start_time)
        >>> 
        >>> # ... time advances, Purple takes actions ...
        >>> 
        >>> new_messages = await collector.collect(current_time)
        >>> for email in new_messages.emails:
        ...     # Generate response if needed
        ...     pass

    Thread Safety:
        This class is NOT thread-safe. Each GreenAgent should have its own
        collector instance.
    """

    def __init__(self, client: AsyncUESClient) -> None:
        """Initialize a new message collector.

        Args:
            client: Async UES client for querying modality states.
        """
        self._client = client
        self._last_check_time: datetime | None = None
        self._seen_calendar_event_ids: set[str] = set()
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Check if the collector has been initialized.

        Returns:
            True if initialize() has been called successfully.
        """
        return self._initialized

    @property
    def last_check_time(self) -> datetime | None:
        """Return the timestamp of the last check.

        Returns:
            The datetime of the last collection, or None if not initialized.
        """
        return self._last_check_time

    @property
    def seen_calendar_event_count(self) -> int:
        """Return count of calendar events seen so far.

        Returns:
            Number of unique calendar event IDs tracked.
        """
        return len(self._seen_calendar_event_ids)

    async def initialize(self, current_time: datetime) -> None:
        """Initialize collector with baseline state.

        Sets the baseline time for email/SMS queries and captures all
        existing calendar event IDs. Must be called before `collect()`.

        Args:
            current_time: Current simulation time (from UES time modality).
                Should be timezone-aware.

        Note:
            If called multiple times, resets state and re-initializes.
            The calendar query fetches ALL events to establish the baseline.
        """
        self._last_check_time = current_time

        # Capture all existing calendar events to establish baseline
        calendar_result = await self._client.calendar.query()
        self._seen_calendar_event_ids = {e.event_id for e in calendar_result.events}

        self._initialized = True

    async def collect(self, current_time: datetime) -> NewMessages:
        """Collect new messages since last check.

        Queries each modality for items created/received after the last
        check time. Updates internal state to track what's been seen.

        The method queries:
        - Emails with `received_after=last_check_time`
        - SMS with `sent_after=last_check_time`
        - Calendar events, comparing IDs against previously seen ones

        Args:
            current_time: Current simulation time. Will become the new
                `last_check_time` after this call.

        Returns:
            NewMessages container with new emails, SMS, and calendar events.
            Results are sorted oldest-first within each modality.

        Raises:
            CollectorNotInitializedError: If called before initialize().

        Note:
            This method updates internal state. Calling it twice with the
            same `current_time` will return empty results on the second call
            (for email/SMS) because `last_check_time` gets updated.
        """
        if not self._initialized:
            raise CollectorNotInitializedError(
                "NewMessageCollector not initialized. Call initialize() first."
            )

        new_messages = NewMessages()

        # Query for new emails (received after last check)
        email_result = await self._client.email.query(
            received_after=self._last_check_time,
            sort_order="asc",  # Oldest first for processing order
        )
        new_messages.emails = list(email_result.emails)

        # Query for new SMS messages (sent after last check)
        sms_result = await self._client.sms.query(
            sent_after=self._last_check_time,
            sort_order="asc",
        )
        new_messages.sms_messages = list(sms_result.messages)

        # Query calendar and find new events by comparing IDs
        calendar_result = await self._client.calendar.query()
        current_event_ids = {e.event_id for e in calendar_result.events}
        new_event_ids = current_event_ids - self._seen_calendar_event_ids

        new_messages.calendar_events = [
            e for e in calendar_result.events if e.event_id in new_event_ids
        ]

        # Update state for next collection
        self._last_check_time = current_time
        self._seen_calendar_event_ids = current_event_ids

        return new_messages

    def reset(self) -> None:
        """Reset collector state for a new assessment.

        Clears all internal state including:
        - Last check time
        - Seen calendar event IDs
        - Initialization flag

        After calling reset(), initialize() must be called again before
        collect() can be used.
        """
        self._last_check_time = None
        self._seen_calendar_event_ids = set()
        self._initialized = False
