"""Tests for AgentBeats assessment results models.

Tests cover:
- Result model creation and validation
- Score validation (score <= max_score)
- Scores consistency validation (overall matches sum of dimensions)
- AssessmentResults validation (actions_taken matches action_log)
- Serialization/deserialization (round-trip)
- message_type field values
- parse_result utility function
- Percentage calculation properties
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.common.agentbeats.results import (
    ALL_DIMENSIONS,
    ActionLogEntryWithTurn,
    AssessmentResults,
    CriterionResult,
    DimensionScore,
    OverallScore,
    RESULT_TYPE_REGISTRY,
    Scores,
    parse_result,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_timestamp() -> datetime:
    """Create a sample timezone-aware timestamp."""
    return datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_dimension_score() -> DimensionScore:
    """Create a sample DimensionScore for testing."""
    return DimensionScore(score=8, max_score=10)


@pytest.fixture
def sample_overall_score() -> OverallScore:
    """Create a sample OverallScore for testing."""
    return OverallScore(score=35, max_score=50)


@pytest.fixture
def sample_scores() -> Scores:
    """Create a sample Scores for testing."""
    return Scores(
        overall=OverallScore(score=35, max_score=50),
        dimensions={
            "accuracy": DimensionScore(score=18, max_score=20),
            "efficiency": DimensionScore(score=17, max_score=30),
        },
    )


@pytest.fixture
def sample_criterion_result() -> CriterionResult:
    """Create a sample CriterionResult for testing."""
    return CriterionResult(
        criterion_id="email_politeness",
        name="Email Politeness",
        dimension="politeness",
        score=8,
        max_score=10,
        explanation="Most emails were polite, but one was too curt.",
        details={"per_email_scores": {"email_1": 10, "email_2": 6}},
    )


@pytest.fixture
def sample_action_entry(sample_timestamp: datetime) -> ActionLogEntryWithTurn:
    """Create a sample ActionLogEntryWithTurn for testing."""
    return ActionLogEntryWithTurn(
        turn=3,
        timestamp=sample_timestamp,
        action="email.send",
        parameters={"to": ["alice@example.com"], "subject": "Hello"},
        success=True,
    )


@pytest.fixture
def sample_assessment_results(
    sample_timestamp: datetime,
) -> AssessmentResults:
    """Create a sample AssessmentResults for testing."""
    action_entry = ActionLogEntryWithTurn(
        turn=1,
        timestamp=sample_timestamp,
        action="email.archive",
        parameters={"email_id": "123"},
        success=True,
    )
    criterion = CriterionResult(
        criterion_id="accuracy_1",
        name="Action Accuracy",
        dimension="accuracy",
        score=8,
        max_score=10,
        explanation="Good accuracy overall.",
    )
    scores = Scores(
        overall=OverallScore(score=8, max_score=10),
        dimensions={"accuracy": DimensionScore(score=8, max_score=10)},
    )
    return AssessmentResults(
        assessment_id="assess-123",
        scenario_id="email_triage_basic",
        participant="purple-agent-1",
        status="completed",
        duration_seconds=1234.5,
        turns_taken=5,
        actions_taken=1,
        scores=scores,
        criteria_results=[criterion],
        action_log=[action_entry],
    )


# =============================================================================
# DimensionScore Tests
# =============================================================================


class TestDimensionScore:
    """Tests for DimensionScore model."""

    def test_create_dimension_score(self, sample_dimension_score: DimensionScore):
        """Test creating a DimensionScore with valid data."""
        assert sample_dimension_score.message_type == "dimension_score"
        assert sample_dimension_score.score == 8
        assert sample_dimension_score.max_score == 10

    def test_percentage_calculation(self):
        """Test percentage property calculation."""
        score = DimensionScore(score=8, max_score=10)
        assert score.percentage == 80.0

    def test_percentage_zero_max(self):
        """Test percentage returns 0 when max_score is 0."""
        score = DimensionScore(score=0, max_score=0)
        assert score.percentage == 0.0

    def test_percentage_full_score(self):
        """Test percentage at 100%."""
        score = DimensionScore(score=10, max_score=10)
        assert score.percentage == 100.0

    def test_score_cannot_exceed_max(self):
        """Test that score cannot exceed max_score."""
        with pytest.raises(ValidationError, match="cannot exceed max_score"):
            DimensionScore(score=15, max_score=10)

    def test_negative_score_rejected(self):
        """Test that negative scores are rejected."""
        with pytest.raises(ValidationError):
            DimensionScore(score=-1, max_score=10)

    def test_negative_max_score_rejected(self):
        """Test that negative max_score is rejected."""
        with pytest.raises(ValidationError):
            DimensionScore(score=0, max_score=-1)

    def test_serialization_round_trip(self, sample_dimension_score: DimensionScore):
        """Test JSON serialization and deserialization."""
        data = sample_dimension_score.model_dump(mode="json")
        restored = DimensionScore.model_validate(data)
        assert restored == sample_dimension_score

    def test_model_is_frozen(self, sample_dimension_score: DimensionScore):
        """Test that the model is immutable."""
        with pytest.raises(ValidationError):
            sample_dimension_score.score = 5  # type: ignore


# =============================================================================
# OverallScore Tests
# =============================================================================


class TestOverallScore:
    """Tests for OverallScore model."""

    def test_create_overall_score(self, sample_overall_score: OverallScore):
        """Test creating an OverallScore with valid data."""
        assert sample_overall_score.message_type == "overall_score"
        assert sample_overall_score.score == 35
        assert sample_overall_score.max_score == 50

    def test_percentage_calculation(self):
        """Test percentage property calculation."""
        score = OverallScore(score=35, max_score=50)
        assert score.percentage == 70.0

    def test_score_cannot_exceed_max(self):
        """Test that score cannot exceed max_score."""
        with pytest.raises(ValidationError, match="cannot exceed max_score"):
            OverallScore(score=60, max_score=50)

    def test_serialization_round_trip(self, sample_overall_score: OverallScore):
        """Test JSON serialization and deserialization."""
        data = sample_overall_score.model_dump(mode="json")
        restored = OverallScore.model_validate(data)
        assert restored == sample_overall_score


# =============================================================================
# Scores Tests
# =============================================================================


class TestScores:
    """Tests for Scores model."""

    def test_create_scores(self, sample_scores: Scores):
        """Test creating Scores with valid data."""
        assert sample_scores.message_type == "scores"
        assert sample_scores.overall.score == 35
        assert len(sample_scores.dimensions) == 2

    def test_overall_must_match_dimension_sum(self):
        """Test that overall score must match sum of dimension scores."""
        with pytest.raises(ValidationError, match="must equal sum of dimension scores"):
            Scores(
                overall=OverallScore(score=100, max_score=100),  # Wrong total (35 expected)
                dimensions={
                    "accuracy": DimensionScore(score=18, max_score=20),
                    "efficiency": DimensionScore(score=17, max_score=30),
                },
            )

    def test_overall_max_must_match_dimension_sum(self):
        """Test that overall max_score must match sum of dimension max_scores."""
        with pytest.raises(
            ValidationError, match="must equal sum of dimension max_scores"
        ):
            Scores(
                overall=OverallScore(score=35, max_score=100),  # Wrong max
                dimensions={
                    "accuracy": DimensionScore(score=18, max_score=20),
                    "efficiency": DimensionScore(score=17, max_score=30),
                },
            )

    def test_empty_dimensions(self):
        """Test Scores with no dimensions."""
        scores = Scores(
            overall=OverallScore(score=0, max_score=0),
            dimensions={},
        )
        assert scores.overall.score == 0

    def test_serialization_round_trip(self, sample_scores: Scores):
        """Test JSON serialization and deserialization."""
        data = sample_scores.model_dump(mode="json")
        restored = Scores.model_validate(data)
        assert restored.overall == sample_scores.overall
        assert restored.dimensions == sample_scores.dimensions


# =============================================================================
# CriterionResult Tests
# =============================================================================


class TestCriterionResult:
    """Tests for CriterionResult model."""

    def test_create_criterion_result(self, sample_criterion_result: CriterionResult):
        """Test creating a CriterionResult with valid data."""
        assert sample_criterion_result.message_type == "criterion_result"
        assert sample_criterion_result.criterion_id == "email_politeness"
        assert sample_criterion_result.dimension == "politeness"
        assert sample_criterion_result.score == 8
        assert sample_criterion_result.max_score == 10

    def test_percentage_calculation(self, sample_criterion_result: CriterionResult):
        """Test percentage property calculation."""
        assert sample_criterion_result.percentage == 80.0

    def test_all_valid_dimensions(self):
        """Test that all valid dimensions are accepted."""
        for dimension in ALL_DIMENSIONS:
            result = CriterionResult(
                criterion_id=f"test_{dimension}",
                name="Test Criterion",
                dimension=dimension,
                score=5,
                max_score=10,
                explanation="Test explanation",
            )
            assert result.dimension == dimension

    def test_invalid_dimension_rejected(self):
        """Test that invalid dimensions are rejected."""
        with pytest.raises(ValidationError):
            CriterionResult(
                criterion_id="test",
                name="Test",
                dimension="invalid_dimension",  # type: ignore
                score=5,
                max_score=10,
                explanation="Test",
            )

    def test_score_cannot_exceed_max(self):
        """Test that score cannot exceed max_score."""
        with pytest.raises(ValidationError, match="cannot exceed max_score"):
            CriterionResult(
                criterion_id="test",
                name="Test",
                dimension="accuracy",
                score=15,
                max_score=10,
                explanation="Test",
            )

    def test_max_score_must_be_positive(self):
        """Test that max_score must be at least 1."""
        with pytest.raises(ValidationError):
            CriterionResult(
                criterion_id="test",
                name="Test",
                dimension="accuracy",
                score=0,
                max_score=0,
                explanation="Test",
            )

    def test_details_optional(self):
        """Test that details field is optional."""
        result = CriterionResult(
            criterion_id="test",
            name="Test",
            dimension="accuracy",
            score=5,
            max_score=10,
            explanation="Test",
        )
        assert result.details is None

    def test_serialization_round_trip(self, sample_criterion_result: CriterionResult):
        """Test JSON serialization and deserialization."""
        data = sample_criterion_result.model_dump(mode="json")
        restored = CriterionResult.model_validate(data)
        assert restored.criterion_id == sample_criterion_result.criterion_id
        assert restored.details == sample_criterion_result.details


# =============================================================================
# ActionLogEntryWithTurn Tests
# =============================================================================


class TestActionLogEntryWithTurn:
    """Tests for ActionLogEntryWithTurn model."""

    def test_create_action_entry(self, sample_action_entry: ActionLogEntryWithTurn):
        """Test creating an ActionLogEntryWithTurn with valid data."""
        assert sample_action_entry.message_type == "action_log_entry_with_turn"
        assert sample_action_entry.turn == 3
        assert sample_action_entry.action == "email.send"
        assert sample_action_entry.success is True

    def test_turn_must_be_positive(self, sample_timestamp: datetime):
        """Test that turn number must be >= 1."""
        with pytest.raises(ValidationError):
            ActionLogEntryWithTurn(
                turn=0,
                timestamp=sample_timestamp,
                action="test",
                success=True,
            )

    def test_failed_action_with_error(self, sample_timestamp: datetime):
        """Test creating a failed action with error message."""
        entry = ActionLogEntryWithTurn(
            turn=1,
            timestamp=sample_timestamp,
            action="email.send",
            parameters={"to": ["invalid"]},
            success=False,
            error_message="Invalid email address",
        )
        assert entry.success is False
        assert entry.error_message == "Invalid email address"

    def test_parameters_default_to_empty_dict(self, sample_timestamp: datetime):
        """Test that parameters defaults to empty dict."""
        entry = ActionLogEntryWithTurn(
            turn=1,
            timestamp=sample_timestamp,
            action="email.read",
            success=True,
        )
        assert entry.parameters == {}

    def test_serialization_round_trip(
        self, sample_action_entry: ActionLogEntryWithTurn
    ):
        """Test JSON serialization and deserialization."""
        data = sample_action_entry.model_dump(mode="json")
        restored = ActionLogEntryWithTurn.model_validate(data)
        assert restored.turn == sample_action_entry.turn
        assert restored.action == sample_action_entry.action


# =============================================================================
# AssessmentResults Tests
# =============================================================================


class TestAssessmentResults:
    """Tests for AssessmentResults model."""

    def test_create_assessment_results(
        self, sample_assessment_results: AssessmentResults
    ):
        """Test creating AssessmentResults with valid data."""
        assert sample_assessment_results.message_type == "assessment_results"
        assert sample_assessment_results.assessment_id == "assess-123"
        assert sample_assessment_results.status == "completed"
        assert sample_assessment_results.turns_taken == 5
        assert sample_assessment_results.actions_taken == 1

    def test_all_valid_statuses(self, sample_timestamp: datetime):
        """Test all valid status values."""
        valid_statuses = ["completed", "failed", "timeout"]
        for status in valid_statuses:
            scores = Scores(
                overall=OverallScore(score=0, max_score=0),
                dimensions={},
            )
            results = AssessmentResults(
                assessment_id="test",
                scenario_id="test",
                participant="test",
                status=status,  # type: ignore
                duration_seconds=0,
                turns_taken=0,
                actions_taken=0,
                scores=scores,
                criteria_results=[],
                action_log=[],
            )
            assert results.status == status

    def test_invalid_status_rejected(self, sample_timestamp: datetime):
        """Test that invalid status is rejected."""
        scores = Scores(
            overall=OverallScore(score=0, max_score=0),
            dimensions={},
        )
        with pytest.raises(ValidationError):
            AssessmentResults(
                assessment_id="test",
                scenario_id="test",
                participant="test",
                status="invalid",  # type: ignore
                duration_seconds=0,
                turns_taken=0,
                actions_taken=0,
                scores=scores,
                criteria_results=[],
                action_log=[],
            )

    def test_actions_taken_must_match_log_length(self, sample_timestamp: datetime):
        """Test that actions_taken must match action_log length."""
        scores = Scores(
            overall=OverallScore(score=0, max_score=0),
            dimensions={},
        )
        action_entry = ActionLogEntryWithTurn(
            turn=1,
            timestamp=sample_timestamp,
            action="test",
            success=True,
        )
        with pytest.raises(ValidationError, match="must match action_log length"):
            AssessmentResults(
                assessment_id="test",
                scenario_id="test",
                participant="test",
                status="completed",
                duration_seconds=0,
                turns_taken=1,
                actions_taken=5,  # Mismatch!
                scores=scores,
                criteria_results=[],
                action_log=[action_entry],  # Only 1 entry
            )

    def test_criterion_dimension_must_exist_in_scores(self, sample_timestamp: datetime):
        """Test that criterion dimensions must exist in scores.dimensions."""
        scores = Scores(
            overall=OverallScore(score=10, max_score=10),
            dimensions={"accuracy": DimensionScore(score=10, max_score=10)},
        )
        criterion = CriterionResult(
            criterion_id="test",
            name="Test",
            dimension="politeness",  # Not in scores.dimensions!
            score=5,
            max_score=10,
            explanation="Test",
        )
        with pytest.raises(ValidationError, match="not in scores.dimensions"):
            AssessmentResults(
                assessment_id="test",
                scenario_id="test",
                participant="test",
                status="completed",
                duration_seconds=0,
                turns_taken=0,
                actions_taken=0,
                scores=scores,
                criteria_results=[criterion],
                action_log=[],
            )

    def test_empty_results(self):
        """Test creating results with no actions or criteria."""
        scores = Scores(
            overall=OverallScore(score=0, max_score=0),
            dimensions={},
        )
        results = AssessmentResults(
            assessment_id="test",
            scenario_id="test",
            participant="test",
            status="timeout",
            duration_seconds=300.0,
            turns_taken=0,
            actions_taken=0,
            scores=scores,
            criteria_results=[],
            action_log=[],
        )
        assert results.actions_taken == 0
        assert len(results.action_log) == 0

    def test_serialization_round_trip(
        self, sample_assessment_results: AssessmentResults
    ):
        """Test JSON serialization and deserialization."""
        data = sample_assessment_results.model_dump(mode="json")
        restored = AssessmentResults.model_validate(data)
        assert restored.assessment_id == sample_assessment_results.assessment_id
        assert restored.status == sample_assessment_results.status
        assert len(restored.action_log) == len(sample_assessment_results.action_log)


# =============================================================================
# parse_result Tests
# =============================================================================


class TestParseResult:
    """Tests for the parse_result utility function."""

    def test_parse_dimension_score(self):
        """Test parsing DimensionScore from dict."""
        data = {
            "message_type": "dimension_score",
            "score": 8,
            "max_score": 10,
        }
        result = parse_result(data)
        assert isinstance(result, DimensionScore)
        assert result.score == 8

    def test_parse_overall_score(self):
        """Test parsing OverallScore from dict."""
        data = {
            "message_type": "overall_score",
            "score": 35,
            "max_score": 50,
        }
        result = parse_result(data)
        assert isinstance(result, OverallScore)
        assert result.score == 35

    def test_parse_criterion_result(self):
        """Test parsing CriterionResult from dict."""
        data = {
            "message_type": "criterion_result",
            "criterion_id": "test",
            "name": "Test Criterion",
            "dimension": "accuracy",
            "score": 5,
            "max_score": 10,
            "explanation": "Test explanation",
        }
        result = parse_result(data)
        assert isinstance(result, CriterionResult)
        assert result.criterion_id == "test"

    def test_parse_missing_message_type(self):
        """Test that missing message_type raises ValueError."""
        data = {"score": 8, "max_score": 10}
        with pytest.raises(ValueError, match="must include 'message_type'"):
            parse_result(data)

    def test_parse_unknown_message_type(self):
        """Test that unknown message_type raises ValueError."""
        data = {"message_type": "unknown_type"}
        with pytest.raises(ValueError, match="Unknown message_type 'unknown_type'"):
            parse_result(data)

    def test_parse_all_registered_types(self, sample_timestamp: datetime):
        """Test that all registered types can be parsed."""
        # Create sample data for each result type
        samples: dict[str, dict] = {
            "dimension_score": {
                "message_type": "dimension_score",
                "score": 5,
                "max_score": 10,
            },
            "overall_score": {
                "message_type": "overall_score",
                "score": 5,
                "max_score": 10,
            },
            "scores": {
                "message_type": "scores",
                "overall": {"message_type": "overall_score", "score": 5, "max_score": 10},
                "dimensions": {
                    "accuracy": {
                        "message_type": "dimension_score",
                        "score": 5,
                        "max_score": 10,
                    }
                },
            },
            "criterion_result": {
                "message_type": "criterion_result",
                "criterion_id": "test",
                "name": "Test",
                "dimension": "accuracy",
                "score": 5,
                "max_score": 10,
                "explanation": "Test",
            },
            "action_log_entry_with_turn": {
                "message_type": "action_log_entry_with_turn",
                "turn": 1,
                "timestamp": sample_timestamp.isoformat(),
                "action": "test",
                "success": True,
            },
        }

        for message_type, data in samples.items():
            result = parse_result(data)
            expected_class = RESULT_TYPE_REGISTRY[message_type]
            assert isinstance(result, expected_class), (
                f"Expected {expected_class.__name__} for {message_type}"
            )


# =============================================================================
# Result Type Registry Tests
# =============================================================================


class TestResultTypeRegistry:
    """Tests for the RESULT_TYPE_REGISTRY."""

    def test_all_result_types_registered(self):
        """Test that all expected result types are in the registry."""
        expected_types = {
            "dimension_score",
            "overall_score",
            "scores",
            "criterion_result",
            "action_log_entry_with_turn",
            "assessment_results",
        }
        assert set(RESULT_TYPE_REGISTRY.keys()) == expected_types

    def test_registry_values_are_model_classes(self):
        """Test that all registry values are Pydantic model classes."""
        from pydantic import BaseModel

        for message_type, model_class in RESULT_TYPE_REGISTRY.items():
            assert issubclass(model_class, BaseModel), (
                f"{message_type} is not a BaseModel subclass"
            )


# =============================================================================
# ALL_DIMENSIONS Tests
# =============================================================================


class TestAllDimensions:
    """Tests for the ALL_DIMENSIONS constant."""

    def test_all_dimensions_list(self):
        """Test that ALL_DIMENSIONS contains all expected dimensions."""
        expected = {
            "accuracy",
            "instruction_following",
            "efficiency",
            "safety",
            "politeness",
        }
        assert set(ALL_DIMENSIONS) == expected

    def test_dimensions_are_strings(self):
        """Test that all dimensions are strings."""
        for dim in ALL_DIMENSIONS:
            assert isinstance(dim, str)
