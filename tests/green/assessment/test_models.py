"""Tests for assessment data models.

Tests cover:
- TurnResult creation and defaults
- EndOfTurnResult creation
- Field access for both models
"""

from __future__ import annotations

from src.green.assessment.models import EndOfTurnResult, TurnResult


# =============================================================================
# Tests for TurnResult
# =============================================================================


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Should create TurnResult with required fields only."""
        result = TurnResult(
            turn_number=1,
            actions_taken=3,
            time_step="PT5M",
            events_processed=7,
            early_completion=False,
        )
        assert result.turn_number == 1
        assert result.actions_taken == 3
        assert result.time_step == "PT5M"
        assert result.events_processed == 7
        assert result.early_completion is False
        assert result.notes is None
        assert result.error is None

    def test_create_with_all_fields(self) -> None:
        """Should create TurnResult with all fields including optionals."""
        result = TurnResult(
            turn_number=5,
            actions_taken=10,
            time_step="PT10M",
            events_processed=15,
            early_completion=True,
            notes="Purple signaled completion.",
            error=None,
        )
        assert result.turn_number == 5
        assert result.actions_taken == 10
        assert result.time_step == "PT10M"
        assert result.events_processed == 15
        assert result.early_completion is True
        assert result.notes == "Purple signaled completion."
        assert result.error is None

    def test_create_with_error(self) -> None:
        """Should create TurnResult with an error message."""
        result = TurnResult(
            turn_number=2,
            actions_taken=0,
            time_step="PT5M",
            events_processed=0,
            early_completion=False,
            error="Purple agent timed out.",
        )
        assert result.error == "Purple agent timed out."
        assert result.actions_taken == 0

    def test_zero_actions_and_events(self) -> None:
        """Should allow zero actions and events (idle turn)."""
        result = TurnResult(
            turn_number=3,
            actions_taken=0,
            time_step="PT1M",
            events_processed=0,
            early_completion=False,
        )
        assert result.actions_taken == 0
        assert result.events_processed == 0

    def test_notes_default_is_none(self) -> None:
        """Notes should default to None when not provided."""
        result = TurnResult(
            turn_number=1,
            actions_taken=1,
            time_step="PT5M",
            events_processed=2,
            early_completion=False,
        )
        assert result.notes is None

    def test_error_default_is_none(self) -> None:
        """Error should default to None when not provided."""
        result = TurnResult(
            turn_number=1,
            actions_taken=1,
            time_step="PT5M",
            events_processed=2,
            early_completion=False,
        )
        assert result.error is None


# =============================================================================
# Tests for EndOfTurnResult
# =============================================================================


class TestEndOfTurnResult:
    """Tests for EndOfTurnResult dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Should create EndOfTurnResult with all fields."""
        result = EndOfTurnResult(
            actions_taken=5,
            total_events=12,
            responses_generated=3,
        )
        assert result.actions_taken == 5
        assert result.total_events == 12
        assert result.responses_generated == 3

    def test_zero_values(self) -> None:
        """Should allow all-zero values (no activity in turn)."""
        result = EndOfTurnResult(
            actions_taken=0,
            total_events=0,
            responses_generated=0,
        )
        assert result.actions_taken == 0
        assert result.total_events == 0
        assert result.responses_generated == 0

    def test_no_responses_generated(self) -> None:
        """Should allow turns with actions but no responses."""
        result = EndOfTurnResult(
            actions_taken=4,
            total_events=8,
            responses_generated=0,
        )
        assert result.actions_taken == 4
        assert result.responses_generated == 0

    def test_high_volume_values(self) -> None:
        """Should handle high-volume scenarios."""
        result = EndOfTurnResult(
            actions_taken=100,
            total_events=500,
            responses_generated=50,
        )
        assert result.actions_taken == 100
        assert result.total_events == 500
        assert result.responses_generated == 50


# =============================================================================
# Tests for package-level imports
# =============================================================================


class TestAssessmentPackageImports:
    """Tests that models are accessible via the assessment package."""

    def test_import_turn_result_from_package(self) -> None:
        """TurnResult should be importable from assessment package."""
        from src.green.assessment import TurnResult as TR

        assert TR is TurnResult

    def test_import_end_of_turn_result_from_package(self) -> None:
        """EndOfTurnResult should be importable from assessment package."""
        from src.green.assessment import EndOfTurnResult as EOTR

        assert EOTR is EndOfTurnResult
