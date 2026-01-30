"""Tests for src.green.core.action_log module.

This module provides comprehensive tests for the ActionLogBuilder class,
which builds an action log from UES event history, filtering for events
attributed to the Purple agent.

Tests cover:
- Basic event processing via add_events_from_turn()
- Purple vs Green event filtering based on agent_id
- Event conversion to ActionLogEntry models
- Statistics and filtering methods
- Reset functionality
- Edge cases and special scenarios
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.common.agentbeats.results import ActionLogEntry
from src.green.core.action_log import (
    ActionLogBuilder,
    ActionLogBuilderError,
)


# =============================================================================
# Fixtures
# =============================================================================


PURPLE_AGENT_ID = "purple-agent-123"
GREEN_AGENT_ID = "green-agent-456"


@pytest.fixture
def builder() -> ActionLogBuilder:
    """Create a fresh ActionLogBuilder instance with Purple agent ID."""
    return ActionLogBuilder(purple_agent_id=PURPLE_AGENT_ID)


@pytest.fixture
def now() -> datetime:
    """Return a timezone-aware datetime for testing."""
    return datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def make_ues_event(
    event_id: str,
    modality: str,
    action: str,
    agent_id: str | None,
    executed_at: datetime,
    status: str = "executed",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a UES EventResponse dict for testing.
    
    Args:
        event_id: Unique event identifier.
        modality: Event modality (email, sms, calendar, etc.).
        action: Action type (send, receive, create, etc.).
        agent_id: ID of agent that triggered the event (None for pre-scheduled).
        executed_at: When the event was executed.
        status: Event status (executed, failed, etc.).
        data: Additional event data (parameters).
        
    Returns:
        A dict matching UES EventResponse structure.
    """
    return {
        "event_id": event_id,
        "modality": modality,
        "status": status,
        "executed_at": executed_at.isoformat(),
        "agent_id": agent_id,
        "data": {
            "action": action,
            **(data or {}),
        },
    }


# =============================================================================
# Initial State Tests
# =============================================================================


class TestInitialState:
    """Tests for ActionLogBuilder initial state."""

    def test_initial_current_turn_is_zero(self, builder: ActionLogBuilder) -> None:
        """Builder should start with current_turn = 0."""
        assert builder.current_turn == 0

    def test_initial_log_is_empty(self, builder: ActionLogBuilder) -> None:
        """Builder should start with empty log."""
        assert builder.get_log() == []

    def test_initial_total_actions_is_zero(self, builder: ActionLogBuilder) -> None:
        """Builder should start with zero actions."""
        assert builder.get_total_actions() == 0

    def test_initial_green_events_count_is_zero(
        self, builder: ActionLogBuilder
    ) -> None:
        """Builder should track zero Green events initially."""
        assert builder._green_events_count == 0

    def test_initial_total_events_processed_is_zero(
        self, builder: ActionLogBuilder
    ) -> None:
        """Builder should have processed zero events initially."""
        assert builder._total_events_processed == 0


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Tests for ActionLogBuilder constructor."""

    def test_stores_purple_agent_id(self) -> None:
        """Constructor should store the purple_agent_id."""
        builder = ActionLogBuilder(purple_agent_id="my-purple-id")
        assert builder._purple_agent_id == "my-purple-id"

    def test_requires_purple_agent_id(self) -> None:
        """Constructor should require purple_agent_id argument."""
        with pytest.raises(TypeError):
            ActionLogBuilder()  # type: ignore


# =============================================================================
# add_events_from_turn Tests
# =============================================================================


class TestAddEventsFromTurn:
    """Tests for add_events_from_turn method."""

    def test_empty_events_list(self, builder: ActionLogBuilder) -> None:
        """Processing empty events list returns empty results."""
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=[])
        builder.end_turn()
        assert purple_entries == []
        assert green_events == []
        assert builder.current_turn == 1

    def test_increments_current_turn(self, builder: ActionLogBuilder) -> None:
        """Processing events should update current_turn."""
        builder.start_turn(1)
        builder.add_events_from_turn(events=[])
        builder.end_turn()
        assert builder.current_turn == 1
        builder.start_turn(2)
        builder.add_events_from_turn(events=[])
        builder.end_turn()
        assert builder.current_turn == 2

    def test_filters_purple_events(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Events with Purple agent_id should be returned as purple_entries."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                data={"to": ["alice@example.com"]},
            ),
        ]
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert len(purple_entries) == 1
        assert green_events == []
        assert purple_entries[0].action == "email.send"

    def test_filters_green_events(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Events with Green agent_id should be returned as green_events."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="receive",
                agent_id=GREEN_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries == []
        assert len(green_events) == 1

    def test_filters_scenario_events(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Events with no agent_id (pre-scheduled) should be returned as green_events."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="receive",
                agent_id=None,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries == []
        assert len(green_events) == 1

    def test_mixed_events(self, builder: ActionLogBuilder, now: datetime) -> None:
        """Mixed events should be correctly separated."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-2",
                modality="email",
                action="receive",
                agent_id=GREEN_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-3",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-4",
                modality="calendar",
                action="invite",
                agent_id=None,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert len(purple_entries) == 2
        assert len(green_events) == 2

    def test_updates_statistics(self, builder: ActionLogBuilder, now: datetime) -> None:
        """Processing events should update internal statistics."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-2",
                modality="email",
                action="receive",
                agent_id=GREEN_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-3",
                modality="sms",
                action="receive",
                agent_id=None,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder._total_events_processed == 3
        assert builder._green_events_count == 2
        assert builder.get_total_actions() == 1


# =============================================================================
# Event Conversion Tests
# =============================================================================


class TestEventConversion:
    """Tests for converting UES events to ActionLogEntry."""

    def test_converts_action_format(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event action should be formatted as 'modality.action'."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].action == "email.send"

    def test_extracts_timestamp(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event executed_at should become entry timestamp."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].timestamp == now

    def test_extracts_parameters(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event data should become entry parameters (excluding 'action')."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                data={"to": ["alice@example.com"], "subject": "Hello"},
            ),
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].parameters == {
            "to": ["alice@example.com"],
            "subject": "Hello",
        }

    def test_determines_success_from_status(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Entry success should be True if status is 'executed'."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="executed",
            ),
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].success is True

    def test_determines_failure_from_status(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Entry success should be False if status is not 'executed'."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="failed",
            ),
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].success is False

    def test_sets_turn_number(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Entry should have the turn number from the current turn."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.end_turn()
        builder.start_turn(2)
        builder.end_turn()
        builder.start_turn(3)
        builder.end_turn()
        builder.start_turn(4)
        builder.end_turn()
        builder.start_turn(5)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].turn == 5


# =============================================================================
# get_log Tests
# =============================================================================


class TestGetLog:
    """Tests for get_log method."""

    def test_returns_all_purple_entries(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_log should return all Purple entries across turns."""
        events1 = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        events2 = [
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events1)
        builder.end_turn()
        builder.start_turn(2)
        builder.add_events_from_turn(events=events2)
        builder.end_turn()
        log = builder.get_log()
        assert len(log) == 2
        assert log[0].action == "email.send"
        assert log[1].action == "sms.send"

    def test_returns_copy_not_reference(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_log should return a copy, not the internal list."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        log1 = builder.get_log()
        log2 = builder.get_log()
        assert log1 is not log2
        assert log1 == log2


# =============================================================================
# Statistics Methods Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_total_actions(self, builder: ActionLogBuilder, now: datetime) -> None:
        """get_total_actions should return count of Purple actions."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-3",
                modality="email",
                action="receive",
                agent_id=GREEN_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder.get_total_actions() == 2

    def test_get_successful_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_successful_actions should return count of successful Purple actions."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="executed",
            ),
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="failed",
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder.get_successful_actions() == 1

    def test_get_failed_actions(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_failed_actions should return count of failed Purple actions."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="executed",
            ),
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="failed",
            ),
            make_ues_event(
                event_id="evt-3",
                modality="email",
                action="reply",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
                status="error",
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder.get_failed_actions() == 2


# =============================================================================
# Filtering Methods Tests
# =============================================================================


class TestFiltering:
    """Tests for filtering methods."""

    def test_get_actions_by_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_turn should return actions for a specific turn."""
        events1 = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        events2 = [
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-3",
                modality="email",
                action="reply",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events1)
        builder.end_turn()
        builder.start_turn(2)
        builder.add_events_from_turn(events=events2)
        builder.end_turn()
        
        turn1_actions = builder.get_actions_by_turn(1)
        turn2_actions = builder.get_actions_by_turn(2)
        
        assert len(turn1_actions) == 1
        assert turn1_actions[0].action == "email.send"
        assert len(turn2_actions) == 2

    def test_get_actions_by_turn_nonexistent(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_turn should return empty list for nonexistent turn."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder.get_actions_by_turn(99) == []

    def test_get_actions_by_type(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_type should return actions matching the type."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-2",
                modality="sms",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-3",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        
        email_sends = builder.get_actions_by_type("email.send")
        sms_sends = builder.get_actions_by_type("sms.send")
        
        assert len(email_sends) == 2
        assert len(sms_sends) == 1

    def test_get_actions_by_type_nonexistent(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """get_actions_by_type should return empty list for nonexistent type."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert builder.get_actions_by_type("calendar.create") == []


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for reset method."""

    def test_clears_all_entries(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """reset should clear all logged entries."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.reset()
        assert builder.get_log() == []

    def test_resets_current_turn(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """reset should set current_turn back to 0."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.start_turn(2)
        builder.end_turn()
        builder.start_turn(3)
        builder.end_turn()
        builder.start_turn(4)
        builder.end_turn()
        builder.start_turn(5)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.reset()
        assert builder.current_turn == 0

    def test_resets_statistics(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """reset should clear all statistics."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
            make_ues_event(
                event_id="evt-2",
                modality="email",
                action="receive",
                agent_id=GREEN_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.reset()
        assert builder.get_total_actions() == 0
        assert builder._green_events_count == 0
        assert builder._total_events_processed == 0

    def test_preserves_purple_agent_id(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """reset should preserve the purple_agent_id."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.reset()
        assert builder._purple_agent_id == PURPLE_AGENT_ID

    def test_can_add_after_reset(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Should be able to add events after reset."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="send",
                agent_id=PURPLE_AGENT_ID,
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        builder.reset()
        builder.start_turn(1)
        builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert len(builder.get_log()) == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_event_with_empty_data(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event with empty data dict should have empty parameters."""
        events = [
            {
                "event_id": "evt-1",
                "modality": "email",
                "status": "executed",
                "executed_at": now.isoformat(),
                "agent_id": PURPLE_AGENT_ID,
                "data": {"action": "read"},
            },
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].parameters == {}

    def test_event_with_no_data_key(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event without data key should be handled gracefully."""
        events = [
            {
                "event_id": "evt-1",
                "modality": "email",
                "status": "executed",
                "executed_at": now.isoformat(),
                "agent_id": PURPLE_AGENT_ID,
            },
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        # Should default to "unknown" action and empty parameters
        assert purple_entries[0].action == "email.unknown"
        assert purple_entries[0].parameters == {}

    def test_events_with_empty_string_agent_id(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Event with empty string agent_id should be treated as green_event."""
        events = [
            make_ues_event(
                event_id="evt-1",
                modality="email",
                action="receive",
                agent_id="",
                executed_at=now,
            ),
        ]
        builder.start_turn(1)
        purple_entries, green_events = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries == []
        assert len(green_events) == 1

    def test_handles_iso_timestamp_with_z_suffix(
        self, builder: ActionLogBuilder, now: datetime
    ) -> None:
        """Should handle ISO timestamps with Z suffix."""
        events = [
            {
                "event_id": "evt-1",
                "modality": "email",
                "status": "executed",
                "executed_at": "2025-01-15T10:00:00Z",
                "agent_id": PURPLE_AGENT_ID,
                "data": {"action": "send"},
            },
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        assert purple_entries[0].timestamp == now

    def test_handles_iso_timestamp_with_offset(
        self, builder: ActionLogBuilder
    ) -> None:
        """Should handle ISO timestamps with timezone offset."""
        events = [
            {
                "event_id": "evt-1",
                "modality": "email",
                "status": "executed",
                "executed_at": "2025-01-15T05:00:00-05:00",
                "agent_id": PURPLE_AGENT_ID,
                "data": {"action": "send"},
            },
        ]
        builder.start_turn(1)
        purple_entries, _ = builder.add_events_from_turn(events=events)
        builder.end_turn()
        # 05:00 -05:00 = 10:00 UTC
        expected = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert purple_entries[0].timestamp == expected
