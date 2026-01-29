"""Tests for src.green.action_log module.

This module provides comprehensive tests for the ActionLogBuilder class,
covering:
- Basic builder operations (start_turn, add_action, end_turn)
- Turn sequencing and validation
- Error handling for invalid states
- Statistics and filtering methods
- Reset functionality
- Edge cases and special scenarios
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.common.agentbeats.messages import ActionLogEntry
from src.common.agentbeats.results import ActionLogEntryWithTurn
from src.green.action_log import (
    ActionLogBuilder,
    ActionLogBuilderError,
    InvalidTurnNumberError,
    InvalidTurnStateError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def builder() -> ActionLogBuilder:
    """Create a fresh ActionLogBuilder instance."""
    return ActionLogBuilder()


@pytest.fixture
def now() -> datetime:
    """Return a timezone-aware datetime for testing."""
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_entry(now: datetime) -> ActionLogEntry:
    """Create a sample ActionLogEntry."""
    return ActionLogEntry(
        timestamp=now,
        action="email.read",
        parameters={"email_id": "123"},
        success=True,
    )


@pytest.fixture
def failed_entry(now: datetime) -> ActionLogEntry:
    """Create a sample failed ActionLogEntry."""
    return ActionLogEntry(
        timestamp=now,
        action="email.send",
        parameters={"to": ["invalid"]},
        success=False,
        error_message="Invalid recipient",
    )


def make_entry(
    now: datetime,
    action: str = "email.read",
    success: bool = True,
    error_message: str | None = None,
    offset_minutes: int = 0,
    **params: str,
) -> ActionLogEntry:
    """Helper to create ActionLogEntry with custom attributes."""
    return ActionLogEntry(
        timestamp=now + timedelta(minutes=offset_minutes),
        action=action,
        parameters=dict(params) if params else {},
        success=success,
        error_message=error_message,
    )


# =============================================================================
# Initial State Tests
# =============================================================================


class TestInitialState:
    """Tests for ActionLogBuilder initial state."""

    def test_initial_current_turn_is_zero(self, builder: ActionLogBuilder) -> None:
        """Builder should start with current_turn = 0."""
        assert builder.current_turn == 0

    def test_initial_is_turn_active_is_false(self, builder: ActionLogBuilder) -> None:
        """Builder should start with no active turn."""
        assert builder.is_turn_active is False

    def test_initial_log_is_empty(self, builder: ActionLogBuilder) -> None:
        """Builder should start with an empty log."""
        assert builder.get_log() == []

    def test_initial_total_actions_is_zero(self, builder: ActionLogBuilder) -> None:
        """Builder should start with zero total actions."""
        assert builder.get_total_actions() == 0

    def test_initial_len_is_zero(self, builder: ActionLogBuilder) -> None:
        """Builder should start with len() = 0."""
        assert len(builder) == 0

    def test_initial_repr(self, builder: ActionLogBuilder) -> None:
        """Builder repr should show initial state."""
        assert repr(builder) == "ActionLogBuilder(turn=0, actions=0, active=False)"


# =============================================================================
# Turn Management Tests
# =============================================================================


class TestStartTurn:
    """Tests for start_turn() method."""

    def test_start_turn_updates_current_turn(self, builder: ActionLogBuilder) -> None:
        """start_turn() should update current_turn."""
        builder.start_turn(1)
        assert builder.current_turn == 1

    def test_start_turn_activates_turn(self, builder: ActionLogBuilder) -> None:
        """start_turn() should set is_turn_active to True."""
        builder.start_turn(1)
        assert builder.is_turn_active is True

    def test_start_turn_must_start_at_one(self, builder: ActionLogBuilder) -> None:
        """start_turn() must start at turn 1."""
        with pytest.raises(InvalidTurnNumberError) as exc_info:
            builder.start_turn(0)
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 0
        assert "Turn number must be >= 1" in str(exc_info.value)

    def test_start_turn_rejects_negative(self, builder: ActionLogBuilder) -> None:
        """start_turn() should reject negative turn numbers."""
        with pytest.raises(InvalidTurnNumberError) as exc_info:
            builder.start_turn(-1)
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == -1

    def test_start_turn_must_be_sequential(self, builder: ActionLogBuilder) -> None:
        """start_turn() must follow sequential order."""
        builder.start_turn(1)
        builder.end_turn()
        with pytest.raises(InvalidTurnNumberError) as exc_info:
            builder.start_turn(3)  # Should be 2
        assert exc_info.value.expected == 2
        assert exc_info.value.actual == 3
        assert "must be sequential" in str(exc_info.value)

    def test_start_turn_while_active_raises_error(
        self, builder: ActionLogBuilder
    ) -> None:
        """start_turn() should raise error if turn is already active."""
        builder.start_turn(1)
        with pytest.raises(InvalidTurnStateError) as exc_info:
            builder.start_turn(2)
        assert "already active" in str(exc_info.value)
        assert "Call end_turn() first" in str(exc_info.value)

    def test_start_turn_preserves_current_turn_on_error(
        self, builder: ActionLogBuilder
    ) -> None:
        """start_turn() should not change state on error."""
        builder.start_turn(1)
        builder.end_turn()
        original_turn = builder.current_turn
        with pytest.raises(InvalidTurnNumberError):
            builder.start_turn(5)
        assert builder.current_turn == original_turn

    def test_start_turn_updates_repr(self, builder: ActionLogBuilder) -> None:
        """start_turn() should update repr to show active state."""
        builder.start_turn(1)
        assert repr(builder) == "ActionLogBuilder(turn=1, actions=0, active=True)"


class TestEndTurn:
    """Tests for end_turn() method."""

    def test_end_turn_deactivates_turn(self, builder: ActionLogBuilder) -> None:
        """end_turn() should set is_turn_active to False."""
        builder.start_turn(1)
        builder.end_turn()
        assert builder.is_turn_active is False

    def test_end_turn_preserves_current_turn(self, builder: ActionLogBuilder) -> None:
        """end_turn() should preserve current_turn value."""
        builder.start_turn(1)
        builder.end_turn()
        assert builder.current_turn == 1

    def test_end_turn_returns_action_count(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """end_turn() should return the number of actions in the turn."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        builder.add_action(sample_entry)
        count = builder.end_turn()
        assert count == 2

    def test_end_turn_returns_zero_for_empty_turn(
        self, builder: ActionLogBuilder
    ) -> None:
        """end_turn() should return 0 for turn with no actions."""
        builder.start_turn(1)
        count = builder.end_turn()
        assert count == 0

    def test_end_turn_without_active_turn_raises_error(
        self, builder: ActionLogBuilder
    ) -> None:
        """end_turn() should raise error if no turn is active."""
        with pytest.raises(InvalidTurnStateError) as exc_info:
            builder.end_turn()
        assert "no turn is currently active" in str(exc_info.value)
        assert "Call start_turn() first" in str(exc_info.value)

    def test_end_turn_twice_raises_error(self, builder: ActionLogBuilder) -> None:
        """end_turn() called twice should raise error."""
        builder.start_turn(1)
        builder.end_turn()
        with pytest.raises(InvalidTurnStateError):
            builder.end_turn()

    def test_end_turn_allows_new_turn(self, builder: ActionLogBuilder) -> None:
        """After end_turn(), a new turn can be started."""
        builder.start_turn(1)
        builder.end_turn()
        builder.start_turn(2)
        assert builder.current_turn == 2
        assert builder.is_turn_active is True


# =============================================================================
# Add Action Tests
# =============================================================================


class TestAddAction:
    """Tests for add_action() method."""

    def test_add_action_returns_entry_with_turn(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should return ActionLogEntryWithTurn."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert isinstance(result, ActionLogEntryWithTurn)

    def test_add_action_sets_turn_number(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should set the correct turn number."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert result.turn == 1

    def test_add_action_preserves_timestamp(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should preserve the original timestamp."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert result.timestamp == sample_entry.timestamp

    def test_add_action_preserves_action(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should preserve the action type."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert result.action == sample_entry.action

    def test_add_action_preserves_parameters(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should preserve the parameters."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert result.parameters == sample_entry.parameters

    def test_add_action_preserves_success(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should preserve the success flag."""
        builder.start_turn(1)
        result = builder.add_action(sample_entry)
        assert result.success == sample_entry.success

    def test_add_action_preserves_error_message(
        self, builder: ActionLogBuilder, failed_entry: ActionLogEntry
    ) -> None:
        """add_action() should preserve the error message."""
        builder.start_turn(1)
        result = builder.add_action(failed_entry)
        assert result.error_message == failed_entry.error_message

    def test_add_action_without_active_turn_raises_error(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should raise error if no turn is active."""
        with pytest.raises(InvalidTurnStateError) as exc_info:
            builder.add_action(sample_entry)
        assert "no turn is currently active" in str(exc_info.value)

    def test_add_action_after_end_turn_raises_error(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() after end_turn() should raise error."""
        builder.start_turn(1)
        builder.end_turn()
        with pytest.raises(InvalidTurnStateError):
            builder.add_action(sample_entry)

    def test_add_action_increments_count(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """add_action() should increment total action count."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        assert builder.get_total_actions() == 1
        builder.add_action(sample_entry)
        assert builder.get_total_actions() == 2

    def test_add_action_correct_turn_for_later_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Actions in later turns should have correct turn numbers."""
        builder.start_turn(1)
        builder.end_turn()

        builder.start_turn(2)
        entry = make_entry(now)
        result = builder.add_action(entry)
        assert result.turn == 2


class TestAddActions:
    """Tests for add_actions() method."""

    def test_add_actions_returns_list(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """add_actions() should return a list of ActionLogEntryWithTurn."""
        builder.start_turn(1)
        entries = [make_entry(now), make_entry(now, offset_minutes=1)]
        results = builder.add_actions(entries)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ActionLogEntryWithTurn) for r in results)

    def test_add_actions_empty_list(self, builder: ActionLogBuilder) -> None:
        """add_actions() with empty list should return empty list."""
        builder.start_turn(1)
        results = builder.add_actions([])
        assert results == []

    def test_add_actions_all_have_same_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """All actions added via add_actions() should have same turn."""
        builder.start_turn(1)
        entries = [
            make_entry(now, action="action1"),
            make_entry(now, action="action2", offset_minutes=1),
            make_entry(now, action="action3", offset_minutes=2),
        ]
        results = builder.add_actions(entries)
        assert all(r.turn == 1 for r in results)

    def test_add_actions_without_active_turn_raises_error(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """add_actions() should raise error if no turn is active."""
        entries = [make_entry(now)]
        with pytest.raises(InvalidTurnStateError):
            builder.add_actions(entries)


# =============================================================================
# Get Log Tests
# =============================================================================


class TestGetLog:
    """Tests for get_log() method."""

    def test_get_log_returns_copy(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """get_log() should return a new list (not internal reference)."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        log1 = builder.get_log()
        log2 = builder.get_log()
        assert log1 is not log2
        assert log1 == log2

    def test_get_log_modification_does_not_affect_builder(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """Modifying returned log should not affect builder."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        log = builder.get_log()
        log.clear()
        assert builder.get_total_actions() == 1

    def test_get_log_returns_all_entries(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_log() should return all entries across all turns."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="action1"))
        builder.end_turn()

        builder.start_turn(2)
        builder.add_action(make_entry(now, action="action2"))
        builder.add_action(make_entry(now, action="action3"))
        builder.end_turn()

        log = builder.get_log()
        assert len(log) == 3
        assert log[0].action == "action1"
        assert log[1].action == "action2"
        assert log[2].action == "action3"

    def test_get_log_entries_are_immutable(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """Entries in log should be frozen (immutable)."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        log = builder.get_log()
        # ActionLogEntryWithTurn has frozen=True, so modification should raise
        with pytest.raises(Exception):  # pydantic ValidationError
            log[0].turn = 999  # type: ignore[misc]


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_total_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_total_actions() should count all actions."""
        builder.start_turn(1)
        builder.add_action(make_entry(now))
        builder.add_action(make_entry(now))
        builder.end_turn()

        builder.start_turn(2)
        builder.add_action(make_entry(now))
        builder.end_turn()

        assert builder.get_total_actions() == 3

    def test_get_successful_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_successful_actions() should count only successful actions."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, success=True))
        builder.add_action(make_entry(now, success=False, error_message="error"))
        builder.add_action(make_entry(now, success=True))
        builder.end_turn()

        assert builder.get_successful_actions() == 2

    def test_get_failed_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_failed_actions() should count only failed actions."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, success=True))
        builder.add_action(make_entry(now, success=False, error_message="e1"))
        builder.add_action(make_entry(now, success=False, error_message="e2"))
        builder.end_turn()

        assert builder.get_failed_actions() == 2

    def test_statistics_sum_to_total(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Successful + failed should equal total."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, success=True))
        builder.add_action(make_entry(now, success=False, error_message="e"))
        builder.add_action(make_entry(now, success=True))
        builder.add_action(make_entry(now, success=False, error_message="e"))
        builder.add_action(make_entry(now, success=True))
        builder.end_turn()

        total = builder.get_total_actions()
        successful = builder.get_successful_actions()
        failed = builder.get_failed_actions()
        assert successful + failed == total

    def test_len_equals_total_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """len(builder) should equal get_total_actions()."""
        builder.start_turn(1)
        builder.add_action(make_entry(now))
        builder.add_action(make_entry(now))
        builder.end_turn()

        assert len(builder) == builder.get_total_actions()


# =============================================================================
# Filtering Tests
# =============================================================================


class TestFiltering:
    """Tests for filtering methods."""

    def test_get_actions_by_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_turn() should return only actions for that turn."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="t1a1"))
        builder.add_action(make_entry(now, action="t1a2"))
        builder.end_turn()

        builder.start_turn(2)
        builder.add_action(make_entry(now, action="t2a1"))
        builder.end_turn()

        turn1_actions = builder.get_actions_by_turn(1)
        assert len(turn1_actions) == 2
        assert all(a.turn == 1 for a in turn1_actions)
        assert turn1_actions[0].action == "t1a1"
        assert turn1_actions[1].action == "t1a2"

        turn2_actions = builder.get_actions_by_turn(2)
        assert len(turn2_actions) == 1
        assert turn2_actions[0].action == "t2a1"

    def test_get_actions_by_turn_nonexistent(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_turn() for nonexistent turn should return empty."""
        builder.start_turn(1)
        builder.add_action(make_entry(now))
        builder.end_turn()

        assert builder.get_actions_by_turn(99) == []

    def test_get_actions_by_type(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_type() should return only matching actions."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="email.read"))
        builder.add_action(make_entry(now, action="email.send"))
        builder.add_action(make_entry(now, action="email.read"))
        builder.add_action(make_entry(now, action="calendar.create"))
        builder.end_turn()

        email_reads = builder.get_actions_by_type("email.read")
        assert len(email_reads) == 2
        assert all(a.action == "email.read" for a in email_reads)

    def test_get_actions_by_type_nonexistent(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_type() for nonexistent type should return empty."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="email.read"))
        builder.end_turn()

        assert builder.get_actions_by_type("nonexistent") == []

    def test_get_actions_by_type_across_turns(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_type() should work across multiple turns."""
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="email.send"))
        builder.end_turn()

        builder.start_turn(2)
        builder.add_action(make_entry(now, action="email.send"))
        builder.end_turn()

        sends = builder.get_actions_by_type("email.send")
        assert len(sends) == 2
        assert sends[0].turn == 1
        assert sends[1].turn == 2


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for reset() method."""

    def test_reset_clears_entries(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """reset() should clear all entries."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        builder.end_turn()

        builder.reset()
        assert builder.get_log() == []
        assert builder.get_total_actions() == 0

    def test_reset_clears_current_turn(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """reset() should set current_turn to 0."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        builder.end_turn()

        builder.reset()
        assert builder.current_turn == 0

    def test_reset_deactivates_turn(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """reset() should deactivate any active turn."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        # Don't end turn, reset while active

        builder.reset()
        assert builder.is_turn_active is False

    def test_reset_allows_starting_fresh(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """After reset(), can start turn 1 again."""
        builder.start_turn(1)
        builder.add_action(make_entry(now))
        builder.end_turn()
        builder.start_turn(2)
        builder.end_turn()

        builder.reset()

        # Should be able to start at turn 1 again
        builder.start_turn(1)
        assert builder.current_turn == 1

    def test_reset_empty_builder(self, builder: ActionLogBuilder) -> None:
        """reset() on empty builder should be safe."""
        builder.reset()
        assert builder.current_turn == 0
        assert builder.is_turn_active is False
        assert len(builder) == 0

    def test_reset_updates_repr(
        self, builder: ActionLogBuilder, sample_entry: ActionLogEntry
    ) -> None:
        """reset() should update repr to initial state."""
        builder.start_turn(1)
        builder.add_action(sample_entry)
        builder.reset()
        assert repr(builder) == "ActionLogBuilder(turn=0, actions=0, active=False)"


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_exception_hierarchy(self) -> None:
        """All custom exceptions should inherit from ActionLogBuilderError."""
        assert issubclass(InvalidTurnStateError, ActionLogBuilderError)
        assert issubclass(InvalidTurnNumberError, ActionLogBuilderError)

    def test_invalid_turn_number_error_attributes(self) -> None:
        """InvalidTurnNumberError should have expected and actual attributes."""
        error = InvalidTurnNumberError("test message", expected=5, actual=10)
        assert error.expected == 5
        assert error.actual == 10
        assert "test message" in str(error)


# =============================================================================
# Integration / Multi-Turn Scenario Tests
# =============================================================================


class TestMultiTurnScenarios:
    """Integration tests for multi-turn scenarios."""

    def test_complete_assessment_flow(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Test a complete multi-turn assessment flow."""
        # Turn 1: Purple reads and archives emails
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="email.read", email_id="e1"))
        builder.add_action(make_entry(now, action="email.archive", email_id="e1"))
        builder.add_action(make_entry(now, action="email.read", email_id="e2"))
        actions_turn_1 = builder.end_turn()
        assert actions_turn_1 == 3

        # Turn 2: Purple sends a reply
        builder.start_turn(2)
        builder.add_action(
            make_entry(
                now,
                action="email.send",
                to="alice@example.com",
                subject="Reply",
            )
        )
        actions_turn_2 = builder.end_turn()
        assert actions_turn_2 == 1

        # Turn 3: Calendar operations
        builder.start_turn(3)
        builder.add_action(make_entry(now, action="calendar.list"))
        builder.add_action(
            make_entry(
                now,
                action="calendar.create",
                success=False,
                error_message="Conflict",
            )
        )
        actions_turn_3 = builder.end_turn()
        assert actions_turn_3 == 2

        # Verify final state
        assert builder.current_turn == 3
        assert builder.get_total_actions() == 6
        assert builder.get_successful_actions() == 5
        assert builder.get_failed_actions() == 1
        assert len(builder.get_actions_by_turn(1)) == 3
        assert len(builder.get_actions_by_turn(2)) == 1
        assert len(builder.get_actions_by_turn(3)) == 2

    def test_empty_turns_allowed(self, builder: ActionLogBuilder) -> None:
        """Empty turns (no actions) should be allowed."""
        builder.start_turn(1)
        builder.end_turn()

        builder.start_turn(2)
        builder.end_turn()

        assert builder.current_turn == 2
        assert builder.get_total_actions() == 0

    def test_reset_and_restart(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Test reset followed by new assessment."""
        # First assessment
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="first_assessment"))
        builder.end_turn()

        builder.reset()

        # Second assessment
        builder.start_turn(1)
        builder.add_action(make_entry(now, action="second_assessment"))
        builder.end_turn()

        log = builder.get_log()
        assert len(log) == 1
        assert log[0].action == "second_assessment"
        assert log[0].turn == 1

    def test_many_actions_single_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Test turn with many actions."""
        builder.start_turn(1)
        for i in range(100):
            builder.add_action(make_entry(now, action=f"action_{i}"))
        count = builder.end_turn()

        assert count == 100
        assert builder.get_total_actions() == 100

    def test_log_preserves_order(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Log should preserve chronological order of actions."""
        builder.start_turn(1)
        for i in range(5):
            builder.add_action(make_entry(now, action=f"t1_action_{i}"))
        builder.end_turn()

        builder.start_turn(2)
        for i in range(5):
            builder.add_action(make_entry(now, action=f"t2_action_{i}"))
        builder.end_turn()

        log = builder.get_log()
        expected_order = [f"t1_action_{i}" for i in range(5)] + [
            f"t2_action_{i}" for i in range(5)
        ]
        actual_order = [entry.action for entry in log]
        assert actual_order == expected_order


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_action_with_empty_parameters(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Actions can have empty parameters dict."""
        builder.start_turn(1)
        entry = ActionLogEntry(
            timestamp=now,
            action="email.list",
            parameters={},
            success=True,
        )
        result = builder.add_action(entry)
        assert result.parameters == {}

    def test_action_with_none_error_message_on_failure(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Failed action can have None error_message."""
        builder.start_turn(1)
        entry = ActionLogEntry(
            timestamp=now,
            action="email.send",
            parameters={},
            success=False,
            error_message=None,
        )
        result = builder.add_action(entry)
        assert result.success is False
        assert result.error_message is None

    def test_action_with_complex_parameters(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Actions can have complex nested parameters."""
        params = {
            "to": ["a@example.com", "b@example.com"],
            "cc": [],
            "body": {"text": "Hello", "html": "<p>Hello</p>"},
            "metadata": {"priority": 1, "tags": ["urgent", "meeting"]},
        }
        builder.start_turn(1)
        entry = ActionLogEntry(
            timestamp=now,
            action="email.send",
            parameters=params,
            success=True,
        )
        result = builder.add_action(entry)
        assert result.parameters == params

    def test_repr_with_many_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """repr should handle many actions."""
        builder.start_turn(1)
        for _ in range(1000):
            builder.add_action(make_entry(now))
        builder.end_turn()

        assert repr(builder) == "ActionLogBuilder(turn=1, actions=1000, active=False)"
