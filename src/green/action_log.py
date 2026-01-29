"""Action log builder for Green agent assessments.

This module provides a helper class for building the assessment action log from
Purple agent turn reports. The builder tracks turn numbers and converts
`ActionLogEntry` objects (from TurnCompleteMessage) to `ActionLogEntryWithTurn`
objects that include turn context.

The ActionLogBuilder has no dependencies on other Green agent modules, making
it easy to test and reuse.

Classes:
    ActionLogBuilder: Builder for assessment action logs.

Example:
    >>> from datetime import datetime, timezone
    >>> from src.common.agentbeats.messages import ActionLogEntry
    >>> from src.green.action_log import ActionLogBuilder
    >>>
    >>> builder = ActionLogBuilder()
    >>> builder.start_turn(1)
    >>>
    >>> entry = ActionLogEntry(
    ...     timestamp=datetime.now(tz=timezone.utc),
    ...     action="email.send",
    ...     parameters={"to": ["alice@example.com"]},
    ...     success=True
    ... )
    >>> builder.add_action(entry)
    >>> builder.end_turn()
    >>>
    >>> log = builder.get_log()
    >>> len(log)
    1
    >>> log[0].turn
    1
"""

from __future__ import annotations

from src.common.agentbeats.messages import ActionLogEntry
from src.common.agentbeats.results import ActionLogEntryWithTurn


class ActionLogBuilderError(Exception):
    """Base exception for ActionLogBuilder errors.

    Attributes:
        message: Human-readable error message.
    """

    pass


class InvalidTurnStateError(ActionLogBuilderError):
    """Raised when an operation is performed in an invalid turn state.

    This error is raised when:
    - `add_action()` is called without an active turn
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
    """Builder for assessment action logs.

    This class accumulates action log entries across turns, tracking the current
    turn number and converting ActionLogEntry objects to ActionLogEntryWithTurn
    objects. The builder ensures proper turn sequencing and provides methods to
    retrieve statistics about the logged actions.

    The builder follows a strict lifecycle:
    1. Call `start_turn(turn_number)` to begin a new turn
    2. Call `add_action(entry)` for each action in the turn (zero or more)
    3. Call `end_turn()` to complete the turn
    4. Repeat steps 1-3 for additional turns
    5. Call `get_log()` to retrieve all entries

    Use `reset()` to clear all entries and start fresh for a new assessment.

    Attributes:
        current_turn: The current turn number (0 if no turn has started).
        is_turn_active: Whether a turn is currently active.

    Example:
        >>> builder = ActionLogBuilder()
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

    def __init__(self) -> None:
        """Initialize an empty action log builder."""
        self._entries: list[ActionLogEntryWithTurn] = []
        self._current_turn: int = 0
        self._is_turn_active: bool = False
        self._actions_in_current_turn: int = 0

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
        called before adding any actions for the turn.

        Args:
            turn_number: The turn number to start (must be current_turn + 1).

        Raises:
            InvalidTurnStateError: If a turn is already active.
            InvalidTurnNumberError: If turn_number is not valid (must be
                current_turn + 1 and >= 1).

        Example:
            >>> builder = ActionLogBuilder()
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

        This method must be called after all actions for the turn have been
        added.

        Returns:
            The number of actions added during this turn.

        Raises:
            InvalidTurnStateError: If no turn is currently active.

        Example:
            >>> builder = ActionLogBuilder()
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

    def add_action(self, entry: ActionLogEntry) -> ActionLogEntryWithTurn:
        """Add an action to the current turn.

        Converts the ActionLogEntry to an ActionLogEntryWithTurn by adding the
        current turn number.

        Args:
            entry: The action log entry to add.

        Returns:
            The converted ActionLogEntryWithTurn object.

        Raises:
            InvalidTurnStateError: If no turn is currently active.

        Example:
            >>> from datetime import datetime, timezone
            >>> from src.common.agentbeats.messages import ActionLogEntry
            >>> builder = ActionLogBuilder()
            >>> builder.start_turn(1)
            >>> entry = ActionLogEntry(
            ...     timestamp=datetime.now(tz=timezone.utc),
            ...     action="email.read",
            ...     parameters={"email_id": "123"},
            ...     success=True
            ... )
            >>> result = builder.add_action(entry)
            >>> result.turn
            1
            >>> result.action
            'email.read'
        """
        if not self._is_turn_active:
            raise InvalidTurnStateError(
                "Cannot add action: no turn is currently active. "
                "Call start_turn() first."
            )

        entry_with_turn = ActionLogEntryWithTurn(
            turn=self._current_turn,
            timestamp=entry.timestamp,
            action=entry.action,
            parameters=entry.parameters,
            success=entry.success,
            error_message=entry.error_message,
        )

        self._entries.append(entry_with_turn)
        self._actions_in_current_turn += 1

        return entry_with_turn

    def add_actions(self, entries: list[ActionLogEntry]) -> list[ActionLogEntryWithTurn]:
        """Add multiple actions to the current turn.

        Convenience method that calls `add_action()` for each entry. All entries
        are added with the current turn number.

        Args:
            entries: List of action log entries to add.

        Returns:
            List of converted ActionLogEntryWithTurn objects.

        Raises:
            InvalidTurnStateError: If no turn is currently active.

        Example:
            >>> from datetime import datetime, timezone
            >>> from src.common.agentbeats.messages import ActionLogEntry
            >>> builder = ActionLogBuilder()
            >>> builder.start_turn(1)
            >>> entries = [
            ...     ActionLogEntry(
            ...         timestamp=datetime.now(tz=timezone.utc),
            ...         action="email.read",
            ...         parameters={},
            ...         success=True
            ...     ),
            ...     ActionLogEntry(
            ...         timestamp=datetime.now(tz=timezone.utc),
            ...         action="email.archive",
            ...         parameters={},
            ...         success=True
            ...     ),
            ... ]
            >>> results = builder.add_actions(entries)
            >>> len(results)
            2
        """
        return [self.add_action(entry) for entry in entries]

    def get_log(self) -> list[ActionLogEntryWithTurn]:
        """Return a copy of the action log.

        Returns a new list containing all action log entries. The returned list
        is a shallow copy; the entries themselves are immutable (frozen).

        Returns:
            List of all ActionLogEntryWithTurn objects added so far.

        Example:
            >>> builder = ActionLogBuilder()
            >>> log = builder.get_log()
            >>> len(log)
            0
        """
        return list(self._entries)

    def get_total_actions(self) -> int:
        """Return the total number of actions logged across all turns.

        Returns:
            Total number of actions.

        Example:
            >>> builder = ActionLogBuilder()
            >>> builder.get_total_actions()
            0
        """
        return len(self._entries)

    def get_successful_actions(self) -> int:
        """Return the number of successful actions.

        Returns:
            Number of actions where success=True.

        Example:
            >>> builder = ActionLogBuilder()
            >>> builder.get_successful_actions()
            0
        """
        return sum(1 for entry in self._entries if entry.success)

    def get_failed_actions(self) -> int:
        """Return the number of failed actions.

        Returns:
            Number of actions where success=False.

        Example:
            >>> builder = ActionLogBuilder()
            >>> builder.get_failed_actions()
            0
        """
        return sum(1 for entry in self._entries if not entry.success)

    def get_actions_by_turn(self, turn_number: int) -> list[ActionLogEntryWithTurn]:
        """Return all actions for a specific turn.

        Args:
            turn_number: The turn number to filter by.

        Returns:
            List of actions taken during the specified turn.

        Example:
            >>> builder = ActionLogBuilder()
            >>> builder.get_actions_by_turn(1)
            []
        """
        return [entry for entry in self._entries if entry.turn == turn_number]

    def get_actions_by_type(self, action_type: str) -> list[ActionLogEntryWithTurn]:
        """Return all actions of a specific type.

        Args:
            action_type: The action type to filter by (e.g., "email.send").

        Returns:
            List of actions matching the specified type.

        Example:
            >>> builder = ActionLogBuilder()
            >>> builder.get_actions_by_type("email.send")
            []
        """
        return [entry for entry in self._entries if entry.action == action_type]

    def reset(self) -> None:
        """Reset the builder for a new assessment.

        Clears all entries and resets the turn counter to 0. Any active turn
        is also ended.

        Example:
            >>> from datetime import datetime, timezone
            >>> from src.common.agentbeats.messages import ActionLogEntry
            >>> builder = ActionLogBuilder()
            >>> builder.start_turn(1)
            >>> entry = ActionLogEntry(
            ...     timestamp=datetime.now(tz=timezone.utc),
            ...     action="email.read",
            ...     parameters={},
            ...     success=True
            ... )
            >>> builder.add_action(entry)
            ActionLogEntryWithTurn(...)
            >>> builder.get_total_actions()
            1
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

    def __len__(self) -> int:
        """Return the total number of actions logged.

        Returns:
            Total number of actions.

        Example:
            >>> builder = ActionLogBuilder()
            >>> len(builder)
            0
        """
        return len(self._entries)

    def __repr__(self) -> str:
        """Return a string representation of the builder.

        Returns:
            String representation showing current state.

        Example:
            >>> builder = ActionLogBuilder()
            >>> repr(builder)
            'ActionLogBuilder(turn=0, actions=0, active=False)'
        """
        return (
            f"ActionLogBuilder(turn={self._current_turn}, "
            f"actions={len(self._entries)}, active={self._is_turn_active})"
        )
