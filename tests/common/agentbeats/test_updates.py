"""Tests for AgentBeats task update models and emitter.

This module tests:
- All update Pydantic models (serialization, validation, message_type immutability)
- The parse_update function
- The TaskUpdateEmitter class and all its methods
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from a2a.types import TaskState

from src.common.agentbeats.updates import (
    # Update models
    ActionObservedUpdate,
    AssessmentCompletedUpdate,
    AssessmentStartedUpdate,
    CriterionEvaluatedUpdate,
    ErrorOccurredUpdate,
    EvaluationStartedUpdate,
    ResponsesGeneratedUpdate,
    ScenarioLoadedUpdate,
    SimulationAdvancedUpdate,
    TaskUpdateEmitter,
    TaskUpdateType,
    TurnCompletedUpdate,
    TurnStartedUpdate,
    # Parsing
    UPDATE_TYPE_REGISTRY,
    parse_update,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_datetime() -> datetime:
    """Return a sample datetime for tests."""
    return datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def emitter() -> TaskUpdateEmitter:
    """Return a TaskUpdateEmitter for tests."""
    return TaskUpdateEmitter(task_id="task-123", context_id="ctx-456")


# =============================================================================
# TaskUpdateType Enum Tests
# =============================================================================


class TestTaskUpdateType:
    """Tests for TaskUpdateType enum."""

    def test_all_update_types_defined(self) -> None:
        """Verify all expected update types are defined."""
        expected = {
            "ASSESSMENT_STARTED",
            "SCENARIO_LOADED",
            "TURN_STARTED",
            "TURN_COMPLETED",
            "RESPONSES_GENERATED",
            "SIMULATION_ADVANCED",
            "EVALUATION_STARTED",
            "CRITERION_EVALUATED",
            "ASSESSMENT_COMPLETED",
            "ERROR_OCCURRED",
            "ACTION_OBSERVED",
        }
        actual = {t.name for t in TaskUpdateType}
        assert actual == expected

    def test_values_have_update_prefix(self) -> None:
        """All values should have 'update_' prefix."""
        for t in TaskUpdateType:
            assert t.value.startswith("update_")


# =============================================================================
# Green Agent Update Model Tests
# =============================================================================


class TestAssessmentStartedUpdate:
    """Tests for AssessmentStartedUpdate model."""

    def test_create_with_required_fields(self, sample_datetime: datetime) -> None:
        """Can create update with all required fields."""
        update = AssessmentStartedUpdate(
            assessment_id="assess-123",
            scenario_id="email_triage_basic",
            participant_url="http://purple:8001",
            start_time=sample_datetime,
        )
        assert update.assessment_id == "assess-123"
        assert update.scenario_id == "email_triage_basic"
        assert update.participant_url == "http://purple:8001"
        assert update.start_time == sample_datetime

    def test_message_type_is_fixed(self, sample_datetime: datetime) -> None:
        """message_type should be fixed to 'update_assessment_started'."""
        update = AssessmentStartedUpdate(
            assessment_id="assess-123",
            scenario_id="test",
            participant_url="http://test:8001",
            start_time=sample_datetime,
        )
        assert update.message_type == "update_assessment_started"

    def test_serialization_round_trip(self, sample_datetime: datetime) -> None:
        """Update should serialize and deserialize correctly."""
        update = AssessmentStartedUpdate(
            assessment_id="assess-123",
            scenario_id="email_triage_basic",
            participant_url="http://purple:8001",
            start_time=sample_datetime,
        )
        data = update.model_dump(mode="json")
        restored = AssessmentStartedUpdate.model_validate(data)
        assert restored == update

    def test_model_is_frozen(self, sample_datetime: datetime) -> None:
        """Model should be immutable (frozen)."""
        update = AssessmentStartedUpdate(
            assessment_id="assess-123",
            scenario_id="test",
            participant_url="http://test:8001",
            start_time=sample_datetime,
        )
        with pytest.raises(Exception):  # ValidationError for frozen model
            update.assessment_id = "new-id"


class TestScenarioLoadedUpdate:
    """Tests for ScenarioLoadedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = ScenarioLoadedUpdate(
            scenario_id="email_triage_basic",
            scenario_name="Basic Email Triage",
            criteria_count=5,
            character_count=3,
        )
        assert update.scenario_id == "email_triage_basic"
        assert update.scenario_name == "Basic Email Triage"
        assert update.criteria_count == 5
        assert update.character_count == 3

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = ScenarioLoadedUpdate(
            scenario_id="test",
            scenario_name="Test",
            criteria_count=0,
            character_count=0,
        )
        assert update.message_type == "update_scenario_loaded"

    def test_counts_must_be_non_negative(self) -> None:
        """Counts must be >= 0."""
        with pytest.raises(ValueError):
            ScenarioLoadedUpdate(
                scenario_id="test",
                scenario_name="Test",
                criteria_count=-1,
                character_count=0,
            )


class TestTurnStartedUpdate:
    """Tests for TurnStartedUpdate model."""

    def test_create_with_required_fields(self, sample_datetime: datetime) -> None:
        """Can create update with all required fields."""
        update = TurnStartedUpdate(
            turn_number=3,
            current_time=sample_datetime,
            events_pending=5,
        )
        assert update.turn_number == 3
        assert update.current_time == sample_datetime
        assert update.events_pending == 5

    def test_message_type_is_fixed(self, sample_datetime: datetime) -> None:
        """message_type should be fixed."""
        update = TurnStartedUpdate(
            turn_number=1,
            current_time=sample_datetime,
            events_pending=0,
        )
        assert update.message_type == "update_turn_started"

    def test_turn_number_must_be_positive(self, sample_datetime: datetime) -> None:
        """turn_number must be >= 1."""
        with pytest.raises(ValueError):
            TurnStartedUpdate(
                turn_number=0,
                current_time=sample_datetime,
                events_pending=0,
            )


class TestTurnCompletedUpdate:
    """Tests for TurnCompletedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = TurnCompletedUpdate(
            turn_number=3,
            actions_taken=4,
            time_advanced="PT1H",
        )
        assert update.turn_number == 3
        assert update.actions_taken == 4
        assert update.time_advanced == "PT1H"
        assert update.early_completion_requested is False

    def test_early_completion_flag(self) -> None:
        """Can set early_completion_requested."""
        update = TurnCompletedUpdate(
            turn_number=5,
            actions_taken=2,
            time_advanced="PT30M",
            early_completion_requested=True,
        )
        assert update.early_completion_requested is True

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = TurnCompletedUpdate(
            turn_number=1,
            actions_taken=0,
            time_advanced="PT1H",
        )
        assert update.message_type == "update_turn_completed"


class TestResponsesGeneratedUpdate:
    """Tests for ResponsesGeneratedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = ResponsesGeneratedUpdate(
            turn_number=2,
            responses_count=3,
            characters_involved=["Alice", "Bob"],
        )
        assert update.turn_number == 2
        assert update.responses_count == 3
        assert update.characters_involved == ["Alice", "Bob"]

    def test_characters_defaults_to_empty_list(self) -> None:
        """characters_involved defaults to empty list."""
        update = ResponsesGeneratedUpdate(
            turn_number=1,
            responses_count=0,
        )
        assert update.characters_involved == []

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = ResponsesGeneratedUpdate(
            turn_number=1,
            responses_count=0,
        )
        assert update.message_type == "update_responses_generated"


class TestSimulationAdvancedUpdate:
    """Tests for SimulationAdvancedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        t1 = datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 28, 10, 0, tzinfo=timezone.utc)
        update = SimulationAdvancedUpdate(
            previous_time=t1,
            new_time=t2,
            duration="PT1H",
            events_processed=2,
        )
        assert update.previous_time == t1
        assert update.new_time == t2
        assert update.duration == "PT1H"
        assert update.events_processed == 2

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        t = datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc)
        update = SimulationAdvancedUpdate(
            previous_time=t,
            new_time=t,
            duration="PT0S",
            events_processed=0,
        )
        assert update.message_type == "update_simulation_advanced"


class TestEvaluationStartedUpdate:
    """Tests for EvaluationStartedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = EvaluationStartedUpdate(
            criteria_count=5,
            dimensions=["accuracy", "efficiency", "politeness"],
        )
        assert update.criteria_count == 5
        assert update.dimensions == ["accuracy", "efficiency", "politeness"]

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = EvaluationStartedUpdate(
            criteria_count=0,
            dimensions=[],
        )
        assert update.message_type == "update_evaluation_started"


class TestCriterionEvaluatedUpdate:
    """Tests for CriterionEvaluatedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = CriterionEvaluatedUpdate(
            criterion_id="email_politeness",
            criterion_name="Email Politeness",
            dimension="politeness",
            score=8,
            max_score=10,
            evaluation_method="llm",
        )
        assert update.criterion_id == "email_politeness"
        assert update.criterion_name == "Email Politeness"
        assert update.dimension == "politeness"
        assert update.score == 8
        assert update.max_score == 10
        assert update.evaluation_method == "llm"

    def test_evaluation_method_values(self) -> None:
        """evaluation_method must be 'programmatic' or 'llm'."""
        update_prog = CriterionEvaluatedUpdate(
            criterion_id="test",
            criterion_name="Test",
            dimension="accuracy",
            score=5,
            max_score=5,
            evaluation_method="programmatic",
        )
        assert update_prog.evaluation_method == "programmatic"

        with pytest.raises(ValueError):
            CriterionEvaluatedUpdate(
                criterion_id="test",
                criterion_name="Test",
                dimension="accuracy",
                score=5,
                max_score=5,
                evaluation_method="invalid",  # type: ignore
            )

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = CriterionEvaluatedUpdate(
            criterion_id="test",
            criterion_name="Test",
            dimension="accuracy",
            score=0,
            max_score=0,
            evaluation_method="llm",
        )
        assert update.message_type == "update_criterion_evaluated"


class TestAssessmentCompletedUpdate:
    """Tests for AssessmentCompletedUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = AssessmentCompletedUpdate(
            reason="scenario_complete",
            total_turns=10,
            total_actions=25,
            duration_seconds=1234.5,
            overall_score=85,
            max_score=100,
        )
        assert update.reason == "scenario_complete"
        assert update.total_turns == 10
        assert update.total_actions == 25
        assert update.duration_seconds == 1234.5
        assert update.overall_score == 85
        assert update.max_score == 100

    def test_reason_values(self) -> None:
        """reason must be one of the allowed values."""
        for reason in ["scenario_complete", "early_completion", "timeout", "error"]:
            update = AssessmentCompletedUpdate(
                reason=reason,  # type: ignore
                total_turns=1,
                total_actions=1,
                duration_seconds=1.0,
                overall_score=0,
                max_score=0,
            )
            assert update.reason == reason

        with pytest.raises(ValueError):
            AssessmentCompletedUpdate(
                reason="invalid",  # type: ignore
                total_turns=1,
                total_actions=1,
                duration_seconds=1.0,
                overall_score=0,
                max_score=0,
            )

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = AssessmentCompletedUpdate(
            reason="scenario_complete",
            total_turns=0,
            total_actions=0,
            duration_seconds=0.0,
            overall_score=0,
            max_score=0,
        )
        assert update.message_type == "update_assessment_completed"


class TestErrorOccurredUpdate:
    """Tests for ErrorOccurredUpdate model."""

    def test_create_with_required_fields(self) -> None:
        """Can create update with all required fields."""
        update = ErrorOccurredUpdate(
            error_type="timeout",
            error_message="Purple agent did not respond within 300 seconds",
            recoverable=False,
        )
        assert update.error_type == "timeout"
        assert update.error_message == "Purple agent did not respond within 300 seconds"
        assert update.recoverable is False
        assert update.context is None

    def test_with_context(self) -> None:
        """Can include context dict."""
        update = ErrorOccurredUpdate(
            error_type="communication_error",
            error_message="Connection refused",
            recoverable=True,
            context={"turn_number": 5, "retry_count": 2},
        )
        assert update.context == {"turn_number": 5, "retry_count": 2}

    def test_error_type_values(self) -> None:
        """error_type must be one of the allowed values."""
        valid_types = [
            "timeout",
            "communication_error",
            "protocol_error",
            "ues_error",
            "evaluation_error",
            "internal_error",
        ]
        for error_type in valid_types:
            update = ErrorOccurredUpdate(
                error_type=error_type,  # type: ignore
                error_message="test",
                recoverable=True,
            )
            assert update.error_type == error_type

    def test_message_type_is_fixed(self) -> None:
        """message_type should be fixed."""
        update = ErrorOccurredUpdate(
            error_type="internal_error",
            error_message="test",
            recoverable=True,
        )
        assert update.message_type == "update_error_occurred"


# =============================================================================
# Action Observed Update Tests (emitted by Green agent)
# =============================================================================


class TestActionObservedUpdate:
    """Tests for ActionObservedUpdate model.

    Note: ActionObservedUpdate is emitted by the Green agent when processing
    TurnCompleteMessage from Purple agents, not by Purple agents directly.
    """

    def test_create_with_required_fields(self, sample_datetime: datetime) -> None:
        """Can create update with all required fields."""
        update = ActionObservedUpdate(
            turn_number=2,
            timestamp=sample_datetime,
            action="email.send",
            parameters={"to": ["alice@example.com"], "subject": "Hello"},
            success=True,
        )
        assert update.turn_number == 2
        assert update.timestamp == sample_datetime
        assert update.action == "email.send"
        assert update.parameters == {"to": ["alice@example.com"], "subject": "Hello"}
        assert update.success is True
        assert update.error_message is None
        assert update.notes is None

    def test_with_error_message(self, sample_datetime: datetime) -> None:
        """Can include error_message for failed actions."""
        update = ActionObservedUpdate(
            turn_number=3,
            timestamp=sample_datetime,
            action="calendar.create",
            parameters={"title": "Meeting"},
            success=False,
            error_message="Calendar not found",
        )
        assert update.success is False
        assert update.error_message == "Calendar not found"

    def test_with_notes(self, sample_datetime: datetime) -> None:
        """Can include notes for reasoning transparency."""
        update = ActionObservedUpdate(
            turn_number=1,
            timestamp=sample_datetime,
            action="email.archive",
            parameters={"email_id": "123"},
            success=True,
            notes="Archiving spam email to clean inbox",
        )
        assert update.notes == "Archiving spam email to clean inbox"

    def test_message_type_is_fixed(self, sample_datetime: datetime) -> None:
        """message_type should be fixed."""
        update = ActionObservedUpdate(
            turn_number=1,
            timestamp=sample_datetime,
            action="test.action",
            parameters={},
            success=True,
        )
        assert update.message_type == "update_action_observed"

    def test_parameters_defaults_to_empty_dict(
        self, sample_datetime: datetime
    ) -> None:
        """parameters defaults to empty dict."""
        update = ActionObservedUpdate(
            turn_number=1,
            timestamp=sample_datetime,
            action="test.action",
            success=True,
        )
        assert update.parameters == {}

    def test_serialization_round_trip(self, sample_datetime: datetime) -> None:
        """Update should serialize and deserialize correctly."""
        update = ActionObservedUpdate(
            turn_number=2,
            timestamp=sample_datetime,
            action="email.send",
            parameters={"to": ["bob@example.com"]},
            success=True,
            notes="Test note",
        )
        data = update.model_dump(mode="json")
        restored = ActionObservedUpdate.model_validate(data)
        assert restored == update

    def test_includes_all_action_log_entry_fields(
        self, sample_datetime: datetime
    ) -> None:
        """ActionObservedUpdate should include all fields from ActionLogEntry."""
        # This test verifies the requirement that ActionObservedUpdate
        # captures all details from an ActionLogEntry
        update = ActionObservedUpdate(
            turn_number=1,
            timestamp=sample_datetime,
            action="email.send",
            parameters={"to": ["alice@example.com"]},
            success=False,
            error_message="Network error",
            notes="Attempted to send follow-up",
        )

        # Verify all ActionLogEntry fields are present
        assert hasattr(update, "timestamp")
        assert hasattr(update, "action")
        assert hasattr(update, "parameters")
        assert hasattr(update, "success")
        assert hasattr(update, "error_message")

        # Plus additional context fields
        assert hasattr(update, "turn_number")
        assert hasattr(update, "notes")


# =============================================================================
# Parse Update Tests
# =============================================================================


class TestParseUpdate:
    """Tests for parse_update function."""

    def test_parse_assessment_started(self) -> None:
        """Can parse AssessmentStartedUpdate."""
        data = {
            "message_type": "update_assessment_started",
            "assessment_id": "assess-123",
            "scenario_id": "test",
            "participant_url": "http://test:8001",
            "start_time": "2026-01-28T12:00:00Z",
        }
        update = parse_update(data)
        assert isinstance(update, AssessmentStartedUpdate)
        assert update.assessment_id == "assess-123"

    def test_parse_action_observed(self) -> None:
        """Can parse ActionObservedUpdate."""
        data = {
            "message_type": "update_action_observed",
            "turn_number": 2,
            "timestamp": "2026-01-28T12:00:00Z",
            "action": "email.send",
            "parameters": {"to": ["alice@example.com"]},
            "success": True,
        }
        update = parse_update(data)
        assert isinstance(update, ActionObservedUpdate)
        assert update.action == "email.send"

    def test_parse_all_update_types(self) -> None:
        """Can parse all registered update types."""
        # Verify registry has all expected types
        assert len(UPDATE_TYPE_REGISTRY) == 11

        # Verify all enum values are in registry
        for update_type in TaskUpdateType:
            assert update_type.value in UPDATE_TYPE_REGISTRY

    def test_missing_message_type_raises(self) -> None:
        """Missing message_type raises ValueError."""
        data = {"assessment_id": "test"}
        with pytest.raises(ValueError, match="must include 'message_type'"):
            parse_update(data)

    def test_unknown_message_type_raises(self) -> None:
        """Unknown message_type raises ValueError."""
        data = {"message_type": "update_unknown"}
        with pytest.raises(ValueError, match="Unknown message_type"):
            parse_update(data)


# =============================================================================
# TaskUpdateEmitter Tests
# =============================================================================


class TestTaskUpdateEmitter:
    """Tests for TaskUpdateEmitter class."""

    def test_init(self, emitter: TaskUpdateEmitter) -> None:
        """Emitter initializes with task_id and context_id."""
        assert emitter.task_id == "task-123"
        assert emitter.context_id == "ctx-456"

    def test_assessment_started(
        self, emitter: TaskUpdateEmitter, sample_datetime: datetime
    ) -> None:
        """assessment_started creates correct event."""
        event = emitter.assessment_started(
            assessment_id="assess-001",
            scenario_id="email_triage_basic",
            participant_url="http://purple:8001",
            start_time=sample_datetime,
        )

        assert event.task_id == "task-123"
        assert event.context_id == "ctx-456"
        assert event.status.state == TaskState.working
        assert event.final is False

        # Verify embedded update data
        assert event.status.message is not None
        parts = event.status.message.parts
        assert len(parts) == 1
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_assessment_started"
        assert data_part.data["assessment_id"] == "assess-001"

    def test_scenario_loaded(self, emitter: TaskUpdateEmitter) -> None:
        """scenario_loaded creates correct event."""
        event = emitter.scenario_loaded(
            scenario_id="email_triage_basic",
            scenario_name="Basic Email Triage",
            criteria_count=5,
            character_count=3,
        )

        assert event.status.state == TaskState.working
        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_scenario_loaded"
        assert data_part.data["criteria_count"] == 5

    def test_turn_started(
        self, emitter: TaskUpdateEmitter, sample_datetime: datetime
    ) -> None:
        """turn_started creates correct event."""
        event = emitter.turn_started(
            turn_number=3,
            current_time=sample_datetime,
            events_pending=5,
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_turn_started"
        assert data_part.data["turn_number"] == 3

    def test_turn_completed(self, emitter: TaskUpdateEmitter) -> None:
        """turn_completed creates correct event."""
        event = emitter.turn_completed(
            turn_number=3,
            actions_taken=4,
            time_advanced="PT1H",
            early_completion_requested=True,
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_turn_completed"
        assert data_part.data["early_completion_requested"] is True

    def test_responses_generated(self, emitter: TaskUpdateEmitter) -> None:
        """responses_generated creates correct event."""
        event = emitter.responses_generated(
            turn_number=2,
            responses_count=3,
            characters_involved=["Alice", "Bob"],
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_responses_generated"
        assert data_part.data["characters_involved"] == ["Alice", "Bob"]

    def test_responses_generated_default_characters(
        self, emitter: TaskUpdateEmitter
    ) -> None:
        """responses_generated defaults characters to empty list."""
        event = emitter.responses_generated(
            turn_number=1,
            responses_count=0,
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["characters_involved"] == []

    def test_simulation_advanced(self, emitter: TaskUpdateEmitter) -> None:
        """simulation_advanced creates correct event."""
        t1 = datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 28, 10, 0, tzinfo=timezone.utc)

        event = emitter.simulation_advanced(
            previous_time=t1,
            new_time=t2,
            duration="PT1H",
            events_processed=2,
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_simulation_advanced"
        assert data_part.data["events_processed"] == 2

    def test_evaluation_started(self, emitter: TaskUpdateEmitter) -> None:
        """evaluation_started creates correct event."""
        event = emitter.evaluation_started(
            criteria_count=5,
            dimensions=["accuracy", "efficiency"],
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_evaluation_started"
        assert data_part.data["dimensions"] == ["accuracy", "efficiency"]

    def test_criterion_evaluated(self, emitter: TaskUpdateEmitter) -> None:
        """criterion_evaluated creates correct event."""
        event = emitter.criterion_evaluated(
            criterion_id="email_politeness",
            criterion_name="Email Politeness",
            dimension="politeness",
            score=8,
            max_score=10,
            evaluation_method="llm",
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_criterion_evaluated"
        assert data_part.data["score"] == 8
        assert data_part.data["evaluation_method"] == "llm"

    def test_assessment_completed(self, emitter: TaskUpdateEmitter) -> None:
        """assessment_completed creates terminal event."""
        event = emitter.assessment_completed(
            reason="scenario_complete",
            total_turns=10,
            total_actions=25,
            duration_seconds=1234.5,
            overall_score=85,
            max_score=100,
        )

        # Should be final and completed state
        assert event.status.state == TaskState.completed
        assert event.final is True

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_assessment_completed"
        assert data_part.data["overall_score"] == 85

    def test_error_occurred_recoverable(self, emitter: TaskUpdateEmitter) -> None:
        """error_occurred with recoverable=True is not final."""
        event = emitter.error_occurred(
            error_type="communication_error",
            error_message="Connection refused",
            recoverable=True,
            context={"retry_count": 1},
        )

        assert event.status.state == TaskState.working
        assert event.final is False

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["context"] == {"retry_count": 1}

    def test_error_occurred_not_recoverable(self, emitter: TaskUpdateEmitter) -> None:
        """error_occurred with recoverable=False is final."""
        event = emitter.error_occurred(
            error_type="timeout",
            error_message="Purple agent timed out",
            recoverable=False,
        )

        assert event.status.state == TaskState.failed
        assert event.final is True

    def test_action_observed(
        self, emitter: TaskUpdateEmitter, sample_datetime: datetime
    ) -> None:
        """action_observed creates correct event with all fields."""
        event = emitter.action_observed(
            turn_number=2,
            timestamp=sample_datetime,
            action="email.send",
            parameters={"to": ["alice@example.com"], "subject": "Hello"},
            success=True,
            notes="Responding to Alice",
        )

        assert event.status.state == TaskState.working
        assert event.final is False

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["message_type"] == "update_action_observed"
        assert data_part.data["turn_number"] == 2
        assert data_part.data["action"] == "email.send"
        assert data_part.data["parameters"] == {
            "to": ["alice@example.com"],
            "subject": "Hello",
        }
        assert data_part.data["success"] is True
        assert data_part.data["notes"] == "Responding to Alice"

    def test_action_observed_with_error(
        self, emitter: TaskUpdateEmitter, sample_datetime: datetime
    ) -> None:
        """action_observed captures error_message for failed actions."""
        event = emitter.action_observed(
            turn_number=3,
            timestamp=sample_datetime,
            action="calendar.create",
            parameters={"title": "Meeting"},
            success=False,
            error_message="Calendar not found",
        )

        parts = event.status.message.parts
        data_part = parts[0].root
        assert data_part.data["success"] is False
        assert data_part.data["error_message"] == "Calendar not found"

    def test_events_have_timestamps(
        self, emitter: TaskUpdateEmitter, sample_datetime: datetime
    ) -> None:
        """All emitted events should have timestamps."""
        event = emitter.turn_started(
            turn_number=1,
            current_time=sample_datetime,
            events_pending=0,
        )

        assert event.status.timestamp is not None
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(event.status.timestamp.replace("Z", "+00:00"))
