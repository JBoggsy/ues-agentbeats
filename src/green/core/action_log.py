"""Action log builder for Green agent assessments.

This module provides a helper class for building the assessment action log from
UES event history. The builder tracks turn numbers and converts UES EventResponse
objects to ActionLogEntry objects that include turn context.

The action log only includes events from the Purple agent (identified by agent_id).
Events from the Green agent (character responses, scheduled events) are tracked
separately for observability but not included in the assessment action log.

Classes:
    ActionLogBuilder: Builder for assessment action logs from UES events.

Example:
    >>> from datetime import datetime, timezone
    >>> from src.green.action_log import ActionLogBuilder
    >>>
    >>> builder = ActionLogBuilder(purple_agent_id="purple-agent-123")
    >>> builder.start_turn(1)
    >>>
    >>> # After time advances, process executed events from UES
    >>> events = [...]  # EventResponse objects from client.events.list_events()
    >>> builder.add_events_from_turn(events)
    >>> builder.end_turn()
    >>>
    >>> log = builder.get_log()
    >>> len(log)  # Only Purple agent events are in the log
    3
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.common.agentbeats.results import ActionLogEntry


class ActionLogBuilderError(Exception):
    """Base exception for ActionLogBuilder errors.

    Attributes:
        message: Human-readable error message.
    """

    pass


class InvalidTurnStateError(ActionLogBuilderError):
    """Raised when an operation is performed in an invalid turn state.

    This error is raised when:
    - `add_events_from_turn()` is called without an active turn
    - `start_turn()` is called while a turn is already active
    - `end_turn()` is called without an active turn

    Attributes:
        message: Human-readable error message.
    """

    pass


class InvalidTurnNumberError(ActionLogBuilderError):
    """Raised when a turn number is invalid.

    This error is raised when:
    - `start_turn()` is called with a turn number less than 1
    - `start_turn()` is called with a turn number not equal to current_turn + 1

    Attributes:
        message: Human-readable error message.
        expected: The expected turn number.
        actual: The actual turn number provided.
    """

    def __init__(self, message: str, expected: int, actual: int) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            expected: The expected turn number.
            actual: The actual turn number provided.
        """
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class ActionLogBuilder:
    """Builder for assessment action logs from UES events.

    This class accumulates action log entries across turns by processing UES
    events. It filters events to only include those from the Purple agent
    (identified by agent_id) for the assessment action log.

    The builder follows a strict lifecycle:
    1. Call `start_turn(turn_number)` to begin a new turn
    2. Call `add_events_from_turn(events)` with executed UES events
    3. Call `end_turn()` to complete the turn
    4. Repeat steps 1-3 for additional turns
    5. Call `get_log()` to retrieve all Purple agent entries

    Use `reset()` to clear all entries and start fresh for a new assessment.

    Attributes:
        purple_agent_id: The agent_id used to identify Purple agent events.
        current_turn: The current turn number (0 if no turn has started).
        is_turn_active: Whether a turn is currently active.

    Example:
        >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
        >>> builder.current_turn
        0
        >>> builder.is_turn_active
        False
        >>> builder.start_turn(1)
        >>> builder.is_turn_active
        True
        >>> builder.current_turn
        1
    """

    def __init__(self, purple_agent_id: str) -> None:
        """Initialize an action log builder.

        Args:
            purple_agent_id: The agent_id used to identify Purple agent events
                in UES. Only events with this agent_id will be included in the
                assessment action log.
        """
        self._purple_agent_id = purple_agent_id
        self._entries: list[ActionLogEntry] = []
        self._current_turn: int = 0
        self._is_turn_active: bool = False
        self._actions_in_current_turn: int = 0
        # Track all events (including Green agent) for statistics
        self._total_events_processed: int = 0
        self._green_events_count: int = 0

    @property
    def purple_agent_id(self) -> str:
        """Return the Purple agent ID used for filtering events.

        Returns:
            The agent_id string used to identify Purple agent events.
        """
        return self._purple_agent_id

    @property
    def current_turn(self) -> int:
        """Return the current turn number.

        Returns 0 if no turn has started yet. During an active turn, returns the
        turn number passed to `start_turn()`. After `end_turn()`, returns the
        last completed turn number.

        Returns:
            The current turn number (0 if no turn has been started).
        """
        return self._current_turn

    @property
    def is_turn_active(self) -> bool:
        """Return whether a turn is currently active.

        A turn is active between `start_turn()` and `end_turn()` calls.

        Returns:
            True if a turn is active, False otherwise.
        """
        return self._is_turn_active

    def start_turn(self, turn_number: int) -> None:
        """Start a new turn.

        Turn numbers must be sequential starting from 1. This method must be
        called before adding any events for the turn.

        Args:
            turn_number: The turn number to start (must be current_turn + 1).

        Raises:
            InvalidTurnStateError: If a turn is already active.
            InvalidTurnNumberError: If turn_number is not valid (must be
                current_turn + 1 and >= 1).

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.start_turn(1)
            >>> builder.current_turn
            1
            >>> builder.is_turn_active
            True
        """
        if self._is_turn_active:
            raise InvalidTurnStateError(
                f"Cannot start turn {turn_number}: turn {self._current_turn} "
                "is already active. Call end_turn() first."
            )

        expected_turn = self._current_turn + 1
        if turn_number < 1:
            raise InvalidTurnNumberError(
                f"Turn number must be >= 1, got {turn_number}",
                expected=expected_turn,
                actual=turn_number,
            )
        if turn_number != expected_turn:
            raise InvalidTurnNumberError(
                f"Turn numbers must be sequential. Expected {expected_turn}, "
                f"got {turn_number}",
                expected=expected_turn,
                actual=turn_number,
            )

        self._current_turn = turn_number
        self._is_turn_active = True
        self._actions_in_current_turn = 0

    def end_turn(self) -> int:
        """End the current turn.

        This method must be called after all events for the turn have been
        added.

        Returns:
            The number of Purple agent actions added during this turn.

        Raises:
            InvalidTurnStateError: If no turn is currently active.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.start_turn(1)
            >>> actions_count = builder.end_turn()
            >>> actions_count
            0
            >>> builder.is_turn_active
            False
        """
        if not self._is_turn_active:
            raise InvalidTurnStateError(
                "Cannot end turn: no turn is currently active. "
                "Call start_turn() first."
            )

        self._is_turn_active = False
        return self._actions_in_current_turn

    def add_events_from_turn(
        self,
        events: list[dict[str, Any]],
    ) -> tuple[list[ActionLogEntry], list[dict[str, Any]]]:
        """Add events from UES to the current turn.

        Processes a list of UES EventResponse objects (as dicts) and adds
        Purple agent events to the action log. Returns both the Purple agent
        entries added and the Green agent events (for observability).

        Args:
            events: List of UES EventResponse objects as dictionaries. Each
                should have at least: event_id, scheduled_time, modality,
                status, executed_at, data, and optionally agent_id.

        Returns:
            A tuple of (purple_entries, green_events) where:
                - purple_entries: ActionLogEntry objects added to the log
                - green_events: Event dicts from the Green agent (not logged)

        Raises:
            InvalidTurnStateError: If no turn is currently active.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.start_turn(1)
            >>> events = [
            ...     {"event_id": "e1", "modality": "email", "status": "executed",
            ...      "executed_at": "2026-01-28T10:00:00Z", "data": {"operation": "send"},
            ...      "agent_id": "purple-123"},
            ...     {"event_id": "e2", "modality": "email", "status": "executed",
            ...      "executed_at": "2026-01-28T10:05:00Z", "data": {"operation": "receive"},
            ...      "agent_id": "green-456"},
            ... ]
            >>> purple, green = builder.add_events_from_turn(events)
            >>> len(purple)  # Only Purple agent event
            1
            >>> len(green)  # Green agent event returned separately
            1
        """
        if not self._is_turn_active:
            raise InvalidTurnStateError(
                "Cannot add events: no turn is currently active. "
                "Call start_turn() first."
            )

        purple_entries: list[ActionLogEntry] = []
        green_events: list[dict[str, Any]] = []

        for event in events:
            self._total_events_processed += 1
            agent_id = event.get("agent_id")

            if agent_id == self._purple_agent_id:
                # Purple agent event - add to action log
                entry = self._convert_event_to_entry(event)
                self._entries.append(entry)
                purple_entries.append(entry)
                self._actions_in_current_turn += 1
            else:
                # Green agent or system event - track but don't log
                green_events.append(event)
                self._green_events_count += 1

        return purple_entries, green_events

    def _convert_event_to_entry(self, event: dict[str, Any]) -> ActionLogEntry:
        """Convert a UES EventResponse dict to an ActionLogEntry.

        Args:
            event: UES EventResponse as a dictionary.

        Returns:
            ActionLogEntry with turn context.
        """
        # Extract action from modality and data
        modality = event.get("modality", "unknown")
        data = event.get("data", {})
        action_type = data.get("operation", "unknown")
        action = f"{modality}.{action_type}"

        # Determine success from status
        status = event.get("status", "unknown")
        success = status == "executed"
        error_message = event.get("error_message") if not success else None

        # Get timestamp from executed_at or scheduled_time
        timestamp_str = event.get("executed_at") or event.get("scheduled_time")
        if isinstance(timestamp_str, str):
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        elif isinstance(timestamp_str, datetime):
            timestamp = timestamp_str
        else:
            # Fallback to current time if no timestamp available
            from datetime import timezone

            timestamp = datetime.now(tz=timezone.utc)

        # Extract parameters from data (excluding 'operation' key)
        parameters = {k: v for k, v in data.items() if k != "operation"}

        return ActionLogEntry(
            turn=self._current_turn,
            timestamp=timestamp,
            action=action,
            parameters=parameters,
            success=success,
            error_message=error_message,
        )

    def get_log(self) -> list[ActionLogEntry]:
        """Return a copy of the action log.

        Returns a new list containing all Purple agent action log entries.
        The returned list is a shallow copy; the entries themselves are
        immutable (frozen).

        Returns:
            List of all ActionLogEntry objects from the Purple agent.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> log = builder.get_log()
            >>> len(log)
            0
        """
        return list(self._entries)

    def get_total_actions(self) -> int:
        """Return the total number of Purple agent actions logged.

        Returns:
            Total number of Purple agent actions across all turns.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_total_actions()
            0
        """
        return len(self._entries)

    def get_successful_actions(self) -> int:
        """Return the number of successful Purple agent actions.

        Returns:
            Number of actions where success=True.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_successful_actions()
            0
        """
        return sum(1 for entry in self._entries if entry.success)

    def get_failed_actions(self) -> int:
        """Return the number of failed Purple agent actions.

        Returns:
            Number of actions where success=False.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_failed_actions()
            0
        """
        return sum(1 for entry in self._entries if not entry.success)

    def get_actions_by_turn(self, turn_number: int) -> list[ActionLogEntry]:
        """Return all Purple agent actions for a specific turn.

        Args:
            turn_number: The turn number to filter by.

        Returns:
            List of Purple agent actions taken during the specified turn.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_actions_by_turn(1)
            []
        """
        return [entry for entry in self._entries if entry.turn == turn_number]

    def get_actions_by_type(self, action_type: str) -> list[ActionLogEntry]:
        """Return all Purple agent actions of a specific type.

        Args:
            action_type: The action type to filter by (e.g., "email.send").

        Returns:
            List of Purple agent actions matching the specified type.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_actions_by_type("email.send")
            []
        """
        return [entry for entry in self._entries if entry.action == action_type]

    def get_total_events_processed(self) -> int:
        """Return the total number of events processed (all agents).

        Returns:
            Total events processed including both Purple and Green agents.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_total_events_processed()
            0
        """
        return self._total_events_processed

    def get_green_events_count(self) -> int:
        """Return the number of Green agent events processed.

        Returns:
            Number of events from the Green agent (not in action log).

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.get_green_events_count()
            0
        """
        return self._green_events_count

    def reset(self) -> None:
        """Reset the builder for a new assessment.

        Clears all entries and resets the turn counter to 0. Any active turn
        is also ended. The purple_agent_id is preserved.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> builder.start_turn(1)
            >>> builder.reset()
            >>> builder.current_turn
            0
            >>> builder.is_turn_active
            False
            >>> builder.get_total_actions()
            0
        """
        self._entries.clear()
        self._current_turn = 0
        self._is_turn_active = False
        self._actions_in_current_turn = 0
        self._total_events_processed = 0
        self._green_events_count = 0

    def __len__(self) -> int:
        """Return the total number of Purple agent actions logged.

        Returns:
            Total number of Purple agent actions.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> len(builder)
            0
        """
        return len(self._entries)

    def __repr__(self) -> str:
        """Return a string representation of the builder.

        Returns:
            String representation showing current state.

        Example:
            >>> builder = ActionLogBuilder(purple_agent_id="purple-123")
            >>> repr(builder)
            'ActionLogBuilder(turn=0, actions=0, active=False)'
        """
        return (
            f"ActionLogBuilder(turn={self._current_turn}, "
            f"actions={len(self._entries)}, active={self._is_turn_active})"
        )
