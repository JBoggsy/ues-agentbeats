"""Tests for scenario schema models.

Tests cover:
- Model creation with valid data
- Field validation (required fields, patterns, constraints)
- Model validation (cross-field validation)
- Serialization/deserialization (round-trip)
- Computed properties
- Helper methods
- ISO 8601 duration parsing
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    ResponseTiming,
    ScenarioConfig,
    parse_iso8601_duration,
    validate_iso8601_duration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_response_timing() -> ResponseTiming:
    """Create a sample ResponseTiming for testing."""
    return ResponseTiming(base_delay="PT30M", variance="PT10M")


@pytest.fixture
def sample_character(sample_response_timing: ResponseTiming) -> CharacterProfile:
    """Create a sample CharacterProfile for testing."""
    return CharacterProfile(
        name="Alice Chen",
        role="Manager",
        personality="Professional but friendly. Prefers concise communication.",
        email="alice.chen@company.com",
        response_timing=sample_response_timing,
    )


@pytest.fixture
def sample_character_with_phone(sample_response_timing: ResponseTiming) -> CharacterProfile:
    """Create a sample CharacterProfile with phone for testing."""
    return CharacterProfile(
        name="Bob Smith",
        role="Friend",
        personality="Casual and humorous. Enjoys long conversations.",
        phone="+1-555-123-4567",
        response_timing=sample_response_timing,
    )


@pytest.fixture
def sample_criterion() -> EvaluationCriterion:
    """Create a sample EvaluationCriterion for testing."""
    return EvaluationCriterion(
        criterion_id="email_response_accuracy",
        name="Email Response Accuracy",
        description="Evaluates whether sent emails address the original query.",
        dimension="accuracy",
        max_score=10,
        evaluator_id="check_email_response_content",
        params={"required_keywords": ["confirmation", "schedule"]},
    )


@pytest.fixture
def sample_criterion_llm() -> EvaluationCriterion:
    """Create a sample EvaluationCriterion with LLM evaluation for testing."""
    return EvaluationCriterion(
        criterion_id="email_politeness",
        name="Email Politeness",
        description="Evaluates the politeness of email responses.",
        dimension="politeness",
        max_score=10,
        evaluation_prompt="Rate the politeness of this email response on a scale of 0-10.",
    )


@pytest.fixture
def sample_scenario(
    sample_character: CharacterProfile,
    sample_criterion: EvaluationCriterion,
) -> ScenarioConfig:
    """Create a sample ScenarioConfig for testing."""
    return ScenarioConfig(
        scenario_id="email_triage_basic",
        name="Basic Email Triage",
        description="Handle incoming emails appropriately.",
        start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
        default_time_step="PT1H",
        user_prompt="Please triage my inbox and respond to urgent emails.",
        user_character="alice",
        characters={"alice": sample_character},
        initial_state={"email": {"folders": []}},
        criteria=[sample_criterion],
    )


# =============================================================================
# ISO 8601 Duration Parsing Tests
# =============================================================================


class TestISO8601Duration:
    """Tests for ISO 8601 duration parsing utilities."""

    def test_parse_hours(self):
        """Test parsing hour durations."""
        assert parse_iso8601_duration("PT1H") == timedelta(hours=1)
        assert parse_iso8601_duration("PT12H") == timedelta(hours=12)
        assert parse_iso8601_duration("PT24H") == timedelta(hours=24)

    def test_parse_minutes(self):
        """Test parsing minute durations."""
        assert parse_iso8601_duration("PT30M") == timedelta(minutes=30)
        assert parse_iso8601_duration("PT1M") == timedelta(minutes=1)
        assert parse_iso8601_duration("PT90M") == timedelta(minutes=90)

    def test_parse_seconds(self):
        """Test parsing second durations."""
        assert parse_iso8601_duration("PT30S") == timedelta(seconds=30)
        assert parse_iso8601_duration("PT1.5S") == timedelta(seconds=1.5)

    def test_parse_combined_time(self):
        """Test parsing combined time durations."""
        assert parse_iso8601_duration("PT1H30M") == timedelta(hours=1, minutes=30)
        assert parse_iso8601_duration("PT2H15M30S") == timedelta(
            hours=2, minutes=15, seconds=30
        )

    def test_parse_days(self):
        """Test parsing day durations."""
        assert parse_iso8601_duration("P1D") == timedelta(days=1)
        assert parse_iso8601_duration("P7D") == timedelta(days=7)

    def test_parse_days_and_time(self):
        """Test parsing day and time durations."""
        assert parse_iso8601_duration("P1DT12H") == timedelta(days=1, hours=12)
        assert parse_iso8601_duration("P2DT6H30M") == timedelta(
            days=2, hours=6, minutes=30
        )

    def test_parse_months_years(self):
        """Test parsing month and year durations (approximate)."""
        # Years are approximated as 365 days
        assert parse_iso8601_duration("P1Y") == timedelta(days=365)
        # Months are approximated as 30 days
        assert parse_iso8601_duration("P1M") == timedelta(days=30)
        assert parse_iso8601_duration("P1Y1M") == timedelta(days=395)

    def test_parse_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
            parse_iso8601_duration("1H")  # Missing P/T prefix
        with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
            parse_iso8601_duration("invalid")
        with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
            parse_iso8601_duration("")

    def test_parse_zero_duration(self):
        """Test that zero durations are valid (PT produces zero timedelta)."""
        # "PT" with no components is technically valid ISO 8601 (zero duration)
        assert parse_iso8601_duration("PT") == timedelta(0)
        assert parse_iso8601_duration("P") == timedelta(0)
        assert parse_iso8601_duration("PT0S") == timedelta(0)

    def test_validate_duration_valid(self):
        """Test validate_iso8601_duration with valid values."""
        assert validate_iso8601_duration("PT1H") == "PT1H"
        assert validate_iso8601_duration("P1D") == "P1D"
        assert validate_iso8601_duration("P1DT12H30M") == "P1DT12H30M"

    def test_validate_duration_invalid(self):
        """Test validate_iso8601_duration with invalid values."""
        with pytest.raises(ValueError, match="Invalid ISO 8601 duration"):
            validate_iso8601_duration("invalid")


# =============================================================================
# ResponseTiming Tests
# =============================================================================


class TestResponseTiming:
    """Tests for ResponseTiming model."""

    def test_create_valid(self, sample_response_timing: ResponseTiming):
        """Test creating a ResponseTiming with valid data."""
        assert sample_response_timing.base_delay == "PT30M"
        assert sample_response_timing.variance == "PT10M"

    def test_base_delay_timedelta_property(self, sample_response_timing: ResponseTiming):
        """Test the base_delay_timedelta property."""
        assert sample_response_timing.base_delay_timedelta == timedelta(minutes=30)

    def test_variance_timedelta_property(self, sample_response_timing: ResponseTiming):
        """Test the variance_timedelta property."""
        assert sample_response_timing.variance_timedelta == timedelta(minutes=10)

    def test_invalid_base_delay_format(self):
        """Test that invalid base_delay format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseTiming(base_delay="invalid", variance="PT10M")
        assert "Invalid ISO 8601 duration" in str(exc_info.value)

    def test_invalid_variance_format(self):
        """Test that invalid variance format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseTiming(base_delay="PT30M", variance="bad")
        assert "Invalid ISO 8601 duration" in str(exc_info.value)

    def test_frozen(self, sample_response_timing: ResponseTiming):
        """Test that ResponseTiming is immutable."""
        with pytest.raises(ValidationError):
            sample_response_timing.base_delay = "PT1H"  # type: ignore

    def test_serialization_roundtrip(self, sample_response_timing: ResponseTiming):
        """Test serialization and deserialization."""
        data = sample_response_timing.model_dump()
        restored = ResponseTiming.model_validate(data)
        assert restored == sample_response_timing

    def test_json_serialization(self, sample_response_timing: ResponseTiming):
        """Test JSON serialization."""
        json_str = sample_response_timing.model_dump_json()
        restored = ResponseTiming.model_validate_json(json_str)
        assert restored == sample_response_timing


# =============================================================================
# CharacterProfile Tests
# =============================================================================


class TestCharacterProfile:
    """Tests for CharacterProfile model."""

    def test_create_with_email(self, sample_character: CharacterProfile):
        """Test creating a CharacterProfile with email."""
        assert sample_character.name == "Alice Chen"
        assert sample_character.role == "Manager"
        assert sample_character.email == "alice.chen@company.com"
        assert sample_character.phone is None

    def test_create_with_phone(
        self, sample_character_with_phone: CharacterProfile
    ):
        """Test creating a CharacterProfile with phone."""
        assert sample_character_with_phone.name == "Bob Smith"
        assert sample_character_with_phone.phone == "+1-555-123-4567"
        assert sample_character_with_phone.email is None

    def test_create_with_both_contact_methods(
        self, sample_response_timing: ResponseTiming
    ):
        """Test creating a CharacterProfile with both email and phone."""
        character = CharacterProfile(
            name="Carol Davis",
            role="Colleague",
            personality="Balanced and helpful.",
            email="carol@example.com",
            phone="+1-555-999-0000",
            response_timing=sample_response_timing,
        )
        assert character.email == "carol@example.com"
        assert character.phone == "+1-555-999-0000"

    def test_create_without_contact_method_fails(
        self, sample_response_timing: ResponseTiming
    ):
        """Test that CharacterProfile requires at least one contact method."""
        with pytest.raises(ValidationError) as exc_info:
            CharacterProfile(
                name="Nobody",
                role="Ghost",
                personality="Doesn't exist.",
                response_timing=sample_response_timing,
            )
        assert "at least one contact method" in str(exc_info.value)

    def test_optional_fields(self, sample_response_timing: ResponseTiming):
        """Test optional fields."""
        character = CharacterProfile(
            name="Test",
            role="Tester",
            personality="Testing.",
            email="test@test.com",
            response_timing=sample_response_timing,
            special_instructions="Always respond in all caps.",
            config={"price_list": {"item_a": 100}},
        )
        assert character.special_instructions == "Always respond in all caps."
        assert character.config == {"price_list": {"item_a": 100}}

    def test_name_empty_fails(self, sample_response_timing: ResponseTiming):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CharacterProfile(
                name="",
                role="Test",
                personality="Test.",
                email="test@test.com",
                response_timing=sample_response_timing,
            )
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_name_too_long_fails(self, sample_response_timing: ResponseTiming):
        """Test that overly long name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CharacterProfile(
                name="A" * 201,
                role="Test",
                personality="Test.",
                email="test@test.com",
                response_timing=sample_response_timing,
            )
        assert "String should have at most 200 characters" in str(exc_info.value)

    def test_frozen(self, sample_character: CharacterProfile):
        """Test that CharacterProfile is immutable."""
        with pytest.raises(ValidationError):
            sample_character.name = "New Name"  # type: ignore

    def test_serialization_roundtrip(self, sample_character: CharacterProfile):
        """Test serialization and deserialization."""
        data = sample_character.model_dump()
        restored = CharacterProfile.model_validate(data)
        assert restored == sample_character

    def test_json_serialization(self, sample_character: CharacterProfile):
        """Test JSON serialization."""
        json_str = sample_character.model_dump_json()
        restored = CharacterProfile.model_validate_json(json_str)
        assert restored == sample_character


# =============================================================================
# EvaluationCriterion Tests
# =============================================================================


class TestEvaluationCriterion:
    """Tests for EvaluationCriterion model."""

    def test_create_with_evaluator_id(self, sample_criterion: EvaluationCriterion):
        """Test creating a criterion with evaluator_id."""
        assert sample_criterion.criterion_id == "email_response_accuracy"
        assert sample_criterion.evaluator_id == "check_email_response_content"
        assert sample_criterion.evaluation_prompt is None
        assert sample_criterion.params == {"required_keywords": ["confirmation", "schedule"]}

    def test_create_with_evaluation_prompt(
        self, sample_criterion_llm: EvaluationCriterion
    ):
        """Test creating a criterion with evaluation_prompt."""
        assert sample_criterion_llm.criterion_id == "email_politeness"
        assert sample_criterion_llm.evaluator_id is None
        assert "Rate the politeness" in sample_criterion_llm.evaluation_prompt  # type: ignore

    def test_create_with_both_methods(self):
        """Test creating a criterion with both evaluation methods."""
        criterion = EvaluationCriterion(
            criterion_id="hybrid_eval",
            name="Hybrid Evaluation",
            description="Uses both programmatic and LLM evaluation.",
            dimension="accuracy",
            max_score=20,
            evaluator_id="basic_check",
            evaluation_prompt="Rate the accuracy.",
        )
        assert criterion.evaluator_id == "basic_check"
        assert criterion.evaluation_prompt == "Rate the accuracy."

    def test_create_without_evaluation_method_fails(self):
        """Test that criterion requires at least one evaluation method."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationCriterion(
                criterion_id="no_method",
                name="No Method",
                description="Has no evaluation method.",
                dimension="accuracy",
                max_score=10,
            )
        assert "at least one evaluation method" in str(exc_info.value)

    def test_all_dimensions(self):
        """Test that all valid dimensions are accepted."""
        dimensions = [
            "accuracy",
            "instruction_following",
            "efficiency",
            "safety",
            "politeness",
        ]
        for dim in dimensions:
            criterion = EvaluationCriterion(
                criterion_id=f"test_{dim}",
                name=f"Test {dim}",
                description=f"Tests {dim}.",
                dimension=dim,  # type: ignore
                max_score=10,
                evaluator_id="test_eval",
            )
            assert criterion.dimension == dim

    def test_invalid_dimension_fails(self):
        """Test that invalid dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationCriterion(
                criterion_id="bad_dim",
                name="Bad Dimension",
                description="Has invalid dimension.",
                dimension="invalid",  # type: ignore
                max_score=10,
                evaluator_id="test",
            )
        assert "dimension" in str(exc_info.value).lower()

    def test_criterion_id_pattern(self):
        """Test that criterion_id must match pattern."""
        # Valid IDs
        valid_ids = ["test", "test_criterion", "a1", "my_test_123"]
        for id_ in valid_ids:
            criterion = EvaluationCriterion(
                criterion_id=id_,
                name="Test",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="test",
            )
            assert criterion.criterion_id == id_

        # Invalid IDs
        invalid_ids = ["", "123test", "Test", "test-criterion", "test.criterion"]
        for id_ in invalid_ids:
            with pytest.raises(ValidationError):
                EvaluationCriterion(
                    criterion_id=id_,
                    name="Test",
                    description="Test.",
                    dimension="accuracy",
                    max_score=10,
                    evaluator_id="test",
                )

    def test_max_score_bounds(self):
        """Test max_score bounds (1-100)."""
        # Valid max_score
        for score in [1, 10, 50, 100]:
            criterion = EvaluationCriterion(
                criterion_id="test",
                name="Test",
                description="Test.",
                dimension="accuracy",
                max_score=score,
                evaluator_id="test",
            )
            assert criterion.max_score == score

        # Invalid max_score
        for score in [0, -1, 101, 1000]:
            with pytest.raises(ValidationError):
                EvaluationCriterion(
                    criterion_id="test",
                    name="Test",
                    description="Test.",
                    dimension="accuracy",
                    max_score=score,
                    evaluator_id="test",
                )

    def test_frozen(self, sample_criterion: EvaluationCriterion):
        """Test that EvaluationCriterion is immutable."""
        with pytest.raises(ValidationError):
            sample_criterion.max_score = 20  # type: ignore

    def test_serialization_roundtrip(self, sample_criterion: EvaluationCriterion):
        """Test serialization and deserialization."""
        data = sample_criterion.model_dump()
        restored = EvaluationCriterion.model_validate(data)
        assert restored == sample_criterion

    def test_json_serialization(self, sample_criterion: EvaluationCriterion):
        """Test JSON serialization."""
        json_str = sample_criterion.model_dump_json()
        restored = EvaluationCriterion.model_validate_json(json_str)
        assert restored == sample_criterion


# =============================================================================
# ScenarioConfig Tests
# =============================================================================


class TestScenarioConfig:
    """Tests for ScenarioConfig model."""

    def test_create_valid(self, sample_scenario: ScenarioConfig):
        """Test creating a ScenarioConfig with valid data."""
        assert sample_scenario.scenario_id == "email_triage_basic"
        assert sample_scenario.name == "Basic Email Triage"
        assert len(sample_scenario.characters) == 1
        assert len(sample_scenario.criteria) == 1

    def test_scenario_id_pattern(self, sample_character: CharacterProfile):
        """Test that scenario_id must match pattern."""
        base_data = {
            "name": "Test",
            "description": "Test scenario.",
            "start_time": datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
            "end_time": datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
            "default_time_step": "PT1H",
            "user_prompt": "Test prompt for the scenario.",
            "user_character": "alice",
            "characters": {"alice": sample_character},
            "initial_state": {},
            "criteria": [
                EvaluationCriterion(
                    criterion_id="test",
                    name="Test",
                    description="Test.",
                    dimension="accuracy",
                    max_score=10,
                    evaluator_id="test",
                )
            ],
        }

        # Valid IDs
        for id_ in ["test", "test_scenario", "email_triage_v2"]:
            config = ScenarioConfig(scenario_id=id_, **base_data)
            assert config.scenario_id == id_

        # Invalid IDs
        for id_ in ["", "123test", "Test", "test-scenario"]:
            with pytest.raises(ValidationError):
                ScenarioConfig(scenario_id=id_, **base_data)

    def test_start_time_before_end_time(self, sample_character: CharacterProfile):
        """Test that start_time must be before end_time."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),  # Before start
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion],
            )
        assert "start_time must be before end_time" in str(exc_info.value)

    def test_equal_start_end_time_fails(self, sample_character: CharacterProfile):
        """Test that equal start and end times fail."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )
        same_time = datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc)

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=same_time,
                end_time=same_time,
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion],
            )
        assert "start_time must be before end_time" in str(exc_info.value)

    def test_timezone_required(self, sample_character: CharacterProfile):
        """Test that datetimes must be timezone-aware."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0),  # No timezone
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion],
            )
        assert "timezone-aware" in str(exc_info.value)

    def test_unique_criterion_ids(self, sample_character: CharacterProfile):
        """Test that criterion IDs must be unique."""
        criterion1 = EvaluationCriterion(
            criterion_id="duplicate_id",
            name="First",
            description="First criterion.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test1",
        )
        criterion2 = EvaluationCriterion(
            criterion_id="duplicate_id",
            name="Second",
            description="Second criterion.",
            dimension="efficiency",
            max_score=10,
            evaluator_id="test2",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion1, criterion2],
            )
        assert "Duplicate criterion IDs" in str(exc_info.value)

    def test_unique_character_emails(self, sample_response_timing: ResponseTiming):
        """Test that character emails must be unique."""
        character1 = CharacterProfile(
            name="Alice",
            role="Manager",
            personality="Test.",
            email="same@example.com",
            response_timing=sample_response_timing,
        )
        character2 = CharacterProfile(
            name="Bob",
            role="Employee",
            personality="Test.",
            email="same@example.com",  # Duplicate
            response_timing=sample_response_timing,
        )
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": character1, "bob": character2},
                initial_state={},
                criteria=[criterion],
            )
        assert "Duplicate character emails" in str(exc_info.value)

    def test_unique_character_phones(self, sample_response_timing: ResponseTiming):
        """Test that character phone numbers must be unique."""
        character1 = CharacterProfile(
            name="Alice",
            role="Manager",
            personality="Test.",
            phone="+1-555-123-4567",
            response_timing=sample_response_timing,
        )
        character2 = CharacterProfile(
            name="Bob",
            role="Employee",
            personality="Test.",
            phone="+1-555-123-4567",  # Duplicate
            response_timing=sample_response_timing,
        )
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": character1, "bob": character2},
                initial_state={},
                criteria=[criterion],
            )
        assert "Duplicate character phones" in str(exc_info.value)

    def test_at_least_one_character_required(self):
        """Test that at least one character is required."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="nobody",
                characters={},  # Empty
                initial_state={},
                criteria=[criterion],
            )
        assert "at least 1" in str(exc_info.value).lower()

    def test_at_least_one_criterion_required(
        self, sample_character: CharacterProfile
    ):
        """Test that at least one criterion is required."""
        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="alice",
                characters={"alice": sample_character},
                initial_state={},
                criteria=[],  # Empty
            )
        assert "at least 1" in str(exc_info.value).lower()

    def test_default_time_step_property(self, sample_scenario: ScenarioConfig):
        """Test the default_time_step_timedelta property."""
        assert sample_scenario.default_time_step_timedelta == timedelta(hours=1)

    def test_duration_property(self, sample_scenario: ScenarioConfig):
        """Test the duration property."""
        assert sample_scenario.duration == timedelta(hours=8)

    def test_get_character_by_email(self, sample_scenario: ScenarioConfig):
        """Test get_character_by_email method."""
        character = sample_scenario.get_character_by_email("alice.chen@company.com")
        assert character is not None
        assert character.name == "Alice Chen"

        # Non-existent email
        assert sample_scenario.get_character_by_email("nobody@example.com") is None

    def test_get_character_by_phone(
        self, sample_character_with_phone: CharacterProfile
    ):
        """Test get_character_by_phone method."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )
        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            description="Test scenario.",
            start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
            default_time_step="PT1H",
            user_prompt="Test prompt.",
            user_character="bob",
            characters={"bob": sample_character_with_phone},
            initial_state={},
            criteria=[criterion],
        )

        character = scenario.get_character_by_phone("+1-555-123-4567")
        assert character is not None
        assert character.name == "Bob Smith"

        # Non-existent phone
        assert scenario.get_character_by_phone("+1-999-999-9999") is None

    def test_get_criteria_by_dimension(
        self, sample_character: CharacterProfile
    ):
        """Test get_criteria_by_dimension method."""
        criteria = [
            EvaluationCriterion(
                criterion_id="accuracy_1",
                name="Accuracy 1",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="accuracy_2",
                name="Accuracy 2",
                description="Test.",
                dimension="accuracy",
                max_score=15,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="efficiency_1",
                name="Efficiency 1",
                description="Test.",
                dimension="efficiency",
                max_score=20,
                evaluator_id="test",
            ),
        ]
        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            description="Test scenario.",
            start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
            default_time_step="PT1H",
            user_prompt="Test prompt.",
            user_character="alice",
            characters={"alice": sample_character},
            initial_state={},
            criteria=criteria,
        )

        accuracy_criteria = scenario.get_criteria_by_dimension("accuracy")
        assert len(accuracy_criteria) == 2
        assert all(c.dimension == "accuracy" for c in accuracy_criteria)

        efficiency_criteria = scenario.get_criteria_by_dimension("efficiency")
        assert len(efficiency_criteria) == 1

        safety_criteria = scenario.get_criteria_by_dimension("safety")
        assert len(safety_criteria) == 0

    def test_get_total_max_score(self, sample_character: CharacterProfile):
        """Test get_total_max_score method."""
        criteria = [
            EvaluationCriterion(
                criterion_id="c1",
                name="C1",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="c2",
                name="C2",
                description="Test.",
                dimension="efficiency",
                max_score=20,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="c3",
                name="C3",
                description="Test.",
                dimension="safety",
                max_score=15,
                evaluator_id="test",
            ),
        ]
        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            description="Test scenario.",
            start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
            default_time_step="PT1H",
            user_prompt="Test prompt.",
            user_character="alice",
            characters={"alice": sample_character},
            initial_state={},
            criteria=criteria,
        )

        assert scenario.get_total_max_score() == 45

    def test_get_max_score_by_dimension(self, sample_character: CharacterProfile):
        """Test get_max_score_by_dimension method."""
        criteria = [
            EvaluationCriterion(
                criterion_id="a1",
                name="A1",
                description="Test.",
                dimension="accuracy",
                max_score=10,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="a2",
                name="A2",
                description="Test.",
                dimension="accuracy",
                max_score=5,
                evaluator_id="test",
            ),
            EvaluationCriterion(
                criterion_id="e1",
                name="E1",
                description="Test.",
                dimension="efficiency",
                max_score=20,
                evaluator_id="test",
            ),
        ]
        scenario = ScenarioConfig(
            scenario_id="test",
            name="Test",
            description="Test scenario.",
            start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
            default_time_step="PT1H",
            user_prompt="Test prompt.",
            user_character="alice",
            characters={"alice": sample_character},
            initial_state={},
            criteria=criteria,
        )

        scores = scenario.get_max_score_by_dimension()
        assert scores["accuracy"] == 15
        assert scores["efficiency"] == 20
        assert "safety" not in scores

    def test_early_completion_conditions(self, sample_scenario: ScenarioConfig):
        """Test early_completion_conditions optional field."""
        assert sample_scenario.early_completion_conditions is None

        # Create scenario with early completion conditions
        new_scenario = ScenarioConfig(
            **{
                **sample_scenario.model_dump(),
                "early_completion_conditions": ["all_emails_processed", "inbox_empty"],
            }
        )
        assert new_scenario.early_completion_conditions == [
            "all_emails_processed",
            "inbox_empty",
        ]

    def test_user_character_required(self, sample_character: CharacterProfile):
        """Test that user_character is a required field."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                # user_character is missing
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion],
            )
        assert "user_character" in str(exc_info.value).lower()

    def test_user_character_must_exist_in_characters(
        self, sample_character: CharacterProfile
    ):
        """Test that user_character must reference a valid character key."""
        criterion = EvaluationCriterion(
            criterion_id="test",
            name="Test",
            description="Test.",
            dimension="accuracy",
            max_score=10,
            evaluator_id="test",
        )

        with pytest.raises(ValidationError) as exc_info:
            ScenarioConfig(
                scenario_id="test",
                name="Test",
                description="Test scenario.",
                start_time=datetime(2026, 1, 28, 9, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 28, 17, 0, tzinfo=timezone.utc),
                default_time_step="PT1H",
                user_prompt="Test prompt.",
                user_character="nonexistent",  # Not in characters dict
                characters={"alice": sample_character},
                initial_state={},
                criteria=[criterion],
            )
        assert "user_character" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        assert "not found in characters" in str(exc_info.value)

    def test_get_user_character_profile(self, sample_scenario: ScenarioConfig):
        """Test get_user_character_profile method."""
        user_profile = sample_scenario.get_user_character_profile()
        assert user_profile is not None
        assert user_profile.name == "Alice Chen"
        assert user_profile == sample_scenario.characters["alice"]

    def test_frozen(self, sample_scenario: ScenarioConfig):
        """Test that ScenarioConfig is immutable."""
        with pytest.raises(ValidationError):
            sample_scenario.name = "New Name"  # type: ignore

    def test_serialization_roundtrip(self, sample_scenario: ScenarioConfig):
        """Test serialization and deserialization."""
        data = sample_scenario.model_dump()
        restored = ScenarioConfig.model_validate(data)
        assert restored == sample_scenario

    def test_json_serialization(self, sample_scenario: ScenarioConfig):
        """Test JSON serialization."""
        json_str = sample_scenario.model_dump_json()
        restored = ScenarioConfig.model_validate_json(json_str)
        assert restored == sample_scenario

    def test_json_mode_serialization(self, sample_scenario: ScenarioConfig):
        """Test JSON mode serialization (for file export)."""
        data = sample_scenario.model_dump(mode="json")
        # JSON mode should serialize datetime to string
        assert isinstance(data["start_time"], str)
        assert isinstance(data["end_time"], str)
        # Should be parseable back
        restored = ScenarioConfig.model_validate(data)
        assert restored == sample_scenario


# =============================================================================
# EvalResult Tests
# =============================================================================


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_valid(self):
        """Test creating a valid EvalResult with score and max_score."""
        from src.green.scenarios.schema import EvalResult

        result = EvalResult(
            score=17,
            max_score=20,
            explanation="Good performance overall",
        )
        assert result.score == 17
        assert result.max_score == 20
        assert result.explanation == "Good performance overall"
        assert result.details is None

    def test_create_with_details(self):
        """Test creating EvalResult with details."""
        from src.green.scenarios.schema import EvalResult

        result = EvalResult(
            score=5,
            max_score=10,
            explanation="Partial success",
            details={"items_processed": 5, "total_items": 10},
        )
        assert result.details == {"items_processed": 5, "total_items": 10}

    def test_score_can_be_any_positive_value(self):
        """Test that score can be any positive value (not normalized 0-1)."""
        from src.green.scenarios.schema import EvalResult

        # Score of 85 out of 100
        result_large = EvalResult(score=85, max_score=100, explanation="Large scale")
        assert result_large.score == 85
        assert result_large.max_score == 100

        # Score of 0
        result_zero = EvalResult(score=0, max_score=10, explanation="Failed")
        assert result_zero.score == 0

        # Score equal to max
        result_max = EvalResult(score=10, max_score=10, explanation="Perfect")
        assert result_max.score == 10

    def test_fractional_scores(self):
        """Test that fractional scores work correctly."""
        from src.green.scenarios.schema import EvalResult

        result = EvalResult(score=7.5, max_score=10, explanation="Partial credit")
        assert result.score == 7.5
        assert result.max_score == 10

    def test_mutable(self):
        """Test that EvalResult is mutable (not frozen)."""
        from src.green.scenarios.schema import EvalResult

        result = EvalResult(score=5, max_score=10, explanation="Test")

        # Should be able to modify attributes
        result.score = 8
        assert result.score == 8

        result.explanation = "Updated"
        assert result.explanation == "Updated"

    def test_equality(self):
        """Test equality comparison."""
        from src.green.scenarios.schema import EvalResult

        result1 = EvalResult(score=5, max_score=10, explanation="Test")
        result2 = EvalResult(score=5, max_score=10, explanation="Test")
        result3 = EvalResult(score=6, max_score=10, explanation="Test")

        assert result1 == result2
        assert result1 != result3

    def test_details_default_is_none(self):
        """Test that details defaults to None, not empty dict."""
        from src.green.scenarios.schema import EvalResult

        result = EvalResult(score=10, max_score=10, explanation="Perfect")
        assert result.details is None


# =============================================================================
# AgentBeatsEvalContext Tests
# =============================================================================


class TestAgentBeatsEvalContext:
    """Tests for AgentBeatsEvalContext dataclass."""

    @pytest.fixture
    def mock_async_ues_client(self):
        """Create a mock AsyncUESClient."""
        from unittest.mock import AsyncMock, MagicMock

        client = MagicMock()
        client.email = MagicMock()
        client.email.get_state = AsyncMock(return_value={"folders": []})
        client.sms = MagicMock()
        client.sms.get_state = AsyncMock(return_value={"threads": []})
        client.calendar = MagicMock()
        client.calendar.get_state = AsyncMock(return_value={"events": []})
        client.chat = MagicMock()
        client.chat.get_state = AsyncMock(return_value={"channels": []})
        client.time = MagicMock()
        client.time.get_state = AsyncMock(
            return_value=MagicMock(
                current_time=datetime(2026, 1, 28, 12, 0, tzinfo=timezone.utc)
            )
        )
        return client

    @pytest.fixture
    def sample_action_log(self):
        """Create a sample action log with ActionLogEntryWithTurn objects."""
        from unittest.mock import MagicMock

        # Create mock ActionLogEntryWithTurn objects
        entry1 = MagicMock()
        entry1.action = "read_email"
        entry1.turn = 1
        entry2 = MagicMock()
        entry2.action = "send_email"
        entry2.turn = 2
        return [entry1, entry2]

    def test_create_valid(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test creating a valid AgentBeatsEvalContext."""
        from src.green.scenarios.schema import AgentBeatsEvalContext

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={"scenario_id": "test", "name": "Test Scenario"},
            action_log=sample_action_log,
            initial_state={"email": {"folders": []}},
            final_state={"email": {"folders": [{"name": "Archive"}]}},
            user_prompt="Please triage my inbox.",
        )

        assert ctx.client is mock_async_ues_client
        assert ctx.scenario_config["scenario_id"] == "test"
        assert ctx.action_log == sample_action_log
        assert ctx.initial_state == {"email": {"folders": []}}
        assert ctx.final_state == {"email": {"folders": [{"name": "Archive"}]}}
        assert ctx.user_prompt == "Please triage my inbox."

    def test_action_log_mutable(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test that action_log can be extended (not frozen)."""
        from src.green.scenarios.schema import AgentBeatsEvalContext
        from unittest.mock import MagicMock

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={"scenario_id": "test"},
            action_log=sample_action_log,
            initial_state={},
            final_state={},
            user_prompt="Test prompt",
        )

        # Context is mutable (not frozen dataclass)
        entry3 = MagicMock()
        entry3.action = "archive_email"
        entry3.turn = 3
        ctx.action_log.append(entry3)
        assert len(ctx.action_log) == 3

    @pytest.mark.asyncio
    async def test_get_state_email(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test get_state method for email modality."""
        from src.green.scenarios.schema import AgentBeatsEvalContext

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={},
            action_log=sample_action_log,
            initial_state={},
            final_state={},
            user_prompt="Test",
        )

        state = await ctx.get_state("email")
        assert state == {"folders": []}
        mock_async_ues_client.email.get_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_all_modalities(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test get_state method for all supported modalities."""
        from src.green.scenarios.schema import AgentBeatsEvalContext

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={},
            action_log=sample_action_log,
            initial_state={},
            final_state={},
            user_prompt="Test",
        )

        # Test all supported modalities
        await ctx.get_state("email")
        await ctx.get_state("sms")
        await ctx.get_state("calendar")
        await ctx.get_state("chat")

        mock_async_ues_client.email.get_state.assert_called_once()
        mock_async_ues_client.sms.get_state.assert_called_once()
        mock_async_ues_client.calendar.get_state.assert_called_once()
        mock_async_ues_client.chat.get_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_unknown_modality_raises(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test get_state raises ValueError for unknown modality."""
        from src.green.scenarios.schema import AgentBeatsEvalContext

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={},
            action_log=sample_action_log,
            initial_state={},
            final_state={},
            user_prompt="Test",
        )

        with pytest.raises(ValueError, match="Unknown modality: unknown"):
            await ctx.get_state("unknown")

    @pytest.mark.asyncio
    async def test_get_time(
        self,
        mock_async_ues_client,
        sample_action_log,
    ):
        """Test get_time method returns simulation time."""
        from src.green.scenarios.schema import AgentBeatsEvalContext

        ctx = AgentBeatsEvalContext(
            client=mock_async_ues_client,
            scenario_config={},
            action_log=sample_action_log,
            initial_state={},
            final_state={},
            user_prompt="Test",
        )

        current_time = await ctx.get_time()
        assert current_time == datetime(2026, 1, 28, 12, 0, tzinfo=timezone.utc)
        mock_async_ues_client.time.get_state.assert_called_once()


# =============================================================================
# Type Alias Tests
# =============================================================================


class TestTypeAliases:
    """Tests for type alias definitions."""

    def test_evaluator_func_type(self):
        """Test that EvaluatorFunc type is properly defined."""
        from collections.abc import Awaitable, Callable

        from src.green.scenarios.schema import (
            AgentBeatsEvalContext,
            EvalResult,
            EvaluatorFunc,
        )

        # EvaluatorFunc should be a Callable type
        # This is a compile-time check, but we can verify the alias exists
        assert EvaluatorFunc is not None

    def test_evaluator_registry_type(self):
        """Test that EvaluatorRegistry type is properly defined."""
        from src.green.scenarios.schema import EvaluatorFunc, EvaluatorRegistry

        # EvaluatorRegistry should be dict[str, EvaluatorFunc]
        assert EvaluatorRegistry is not None

    def test_example_evaluator_matches_type(self):
        """Test that a simple evaluator matches the type signature."""
        import asyncio

        from src.green.scenarios.schema import (
            AgentBeatsEvalContext,
            EvalResult,
            EvaluatorFunc,
        )

        async def sample_evaluator(
            ctx: AgentBeatsEvalContext,
            params: dict,
        ) -> EvalResult:
            return EvalResult(score=10, max_score=10, explanation="OK")

        # The function should match the type signature
        assert asyncio.iscoroutinefunction(sample_evaluator)
        assert callable(sample_evaluator)
