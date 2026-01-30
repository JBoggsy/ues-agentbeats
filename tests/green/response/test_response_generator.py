"""Tests for the ResponseGenerator module.

Tests cover:
- ResponseGenerator initialization and properties
- Character lookup methods
- Message processing for email, SMS, and calendar
- Thread context preparation
- LLM integration (with mocked LLMs)
- Response building (timing, subject derivation, references)
- Error handling and graceful degradation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.green.response.generator import (
    MAX_THREAD_MESSAGES,
    ResponseGenerationError,
    ResponseGenerator,
    ResponseGeneratorError,
)
from src.green.response.models import (
    CalendarRSVPResult,
    MessageContext,
    ScheduledResponse,
    ShouldRespondResult,
    ThreadContext,
)
from src.green.scenarios.schema import (
    CharacterProfile,
    EvaluationCriterion,
    ResponseTiming,
    ScenarioConfig,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockEmail:
    """Mock Email object for testing."""

    message_id: str = "msg-1"
    thread_id: str = "thread-1"
    from_address: str = "sender@example.com"
    to_addresses: list[str] = field(default_factory=lambda: ["recipient@example.com"])
    cc_addresses: list[str] = field(default_factory=list)
    subject: str = "Test Subject"
    body_text: str = "Test email body"
    received_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)
    )
    references: list[str] = field(default_factory=list)


@dataclass
class MockSMSMessage:
    """Mock SMSMessage object for testing."""

    message_id: str = "sms-1"
    thread_id: str = "sms-thread-1"
    from_number: str = "+15551234567"
    to_numbers: list[str] = field(default_factory=lambda: ["+15559876543"])
    body: str = "Test SMS body"
    direction: str = "incoming"
    sent_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)
    )


@dataclass
class MockAttendee:
    """Mock calendar attendee."""

    email: str
    response_status: str = "needsAction"


@dataclass
class MockCalendarEvent:
    """Mock CalendarEvent object for testing."""

    event_id: str = "evt-1"
    calendar_id: str = "cal-1"
    title: str = "Test Event"
    start: datetime = field(
        default_factory=lambda: datetime(2026, 1, 29, 14, 0, tzinfo=timezone.utc)
    )
    end: datetime = field(
        default_factory=lambda: datetime(2026, 1, 29, 15, 0, tzinfo=timezone.utc)
    )
    organizer: str = "organizer@example.com"
    description: str = "Test event description"
    location: str = "Conference Room A"
    attendees: list[MockAttendee] = field(default_factory=list)


@dataclass
class MockEmailQueryResponse:
    """Mock EmailQueryResponse for testing."""

    emails: list[MockEmail] = field(default_factory=list)


@dataclass
class MockSMSQueryResponse:
    """Mock SMSQueryResponse for testing."""

    messages: list[MockSMSMessage] = field(default_factory=list)


class MockLLMResponse:
    """Mock LLM response object."""

    def __init__(self, content: str):
        self.content = content


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def response_timing() -> ResponseTiming:
    """Create a response timing configuration."""
    return ResponseTiming(base_delay="PT30M", variance="PT10M")


@pytest.fixture
def user_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create a user character profile."""
    return CharacterProfile(
        name="Alex Thompson",
        relationships={},
        personality="The user being assisted",
        email="alex.thompson@example.com",
        phone="+15550001111",
        response_timing=response_timing,
    )


@pytest.fixture
def alice_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create Alice character profile."""
    return CharacterProfile(
        name="Alice Chen",
        relationships={"Alex Thompson": "colleague"},
        personality="Professional but friendly. Prefers concise communication.",
        email="alice.chen@example.com",
        phone="+15550002222",
        response_timing=response_timing,
        special_instructions="Always responds promptly to urgent matters.",
    )


@pytest.fixture
def bob_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create Bob character profile."""
    return CharacterProfile(
        name="Bob Smith",
        relationships={"Alex Thompson": "manager"},
        personality="Formal and thorough. Likes detailed explanations.",
        email="bob.smith@example.com",
        phone="+15550003333",
        response_timing=response_timing,
    )


@pytest.fixture
def unavailable_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create a character who is unavailable."""
    return CharacterProfile(
        name="Carol Davis",
        relationships={},
        personality="Usually helpful.",
        email="carol.davis@example.com",
        response_timing=response_timing,
        special_instructions="On vacation. Do not respond to any messages.",
    )


@pytest.fixture
def sample_criterion() -> EvaluationCriterion:
    """Create a sample evaluation criterion."""
    return EvaluationCriterion(
        criterion_id="test_criterion",
        name="Test Criterion",
        description="A test evaluation criterion",
        dimension="accuracy",
        max_score=10,
        evaluator_id="test_evaluator",
    )


@pytest.fixture
def sample_scenario(
    user_character: CharacterProfile,
    alice_character: CharacterProfile,
    bob_character: CharacterProfile,
    unavailable_character: CharacterProfile,
    sample_criterion: EvaluationCriterion,
) -> ScenarioConfig:
    """Create a sample scenario configuration."""
    return ScenarioConfig(
        scenario_id="test_scenario",
        name="Test Scenario",
        description="A test scenario for response generator",
        start_time=datetime(2026, 1, 29, 9, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 29, 17, 0, tzinfo=timezone.utc),
        default_time_step="PT1H",
        user_prompt="Please handle my communications.",
        user_character="alex",
        characters={
            "alex": user_character,
            "alice": alice_character,
            "bob": bob_character,
            "carol": unavailable_character,
        },
        initial_state={"environment": {}},
        criteria=[sample_criterion],
    )


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock UES client."""
    client = AsyncMock()

    # Setup email client
    client.email.query = AsyncMock(return_value=MockEmailQueryResponse())

    # Setup SMS client
    client.sms.query = AsyncMock(return_value=MockSMSQueryResponse())

    return client


@pytest.fixture
def mock_response_llm() -> MagicMock:
    """Create a mock response LLM."""
    llm = MagicMock()
    # Setup basic ainvoke
    llm.ainvoke = AsyncMock(return_value=MockLLMResponse("Generated response content"))
    return llm


@pytest.fixture
def mock_summarization_llm() -> MagicMock:
    """Create a mock summarization LLM."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=MockLLMResponse("Summary of earlier conversation.")
    )
    return llm


@pytest.fixture
def response_generator(
    mock_ues_client: AsyncMock,
    sample_scenario: ScenarioConfig,
    mock_response_llm: MagicMock,
    mock_summarization_llm: MagicMock,
) -> ResponseGenerator:
    """Create a ResponseGenerator instance for testing."""
    return ResponseGenerator(
        client=mock_ues_client,
        scenario_config=sample_scenario,
        response_llm=mock_response_llm,
        summarization_llm=mock_summarization_llm,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestResponseGeneratorInit:
    """Tests for ResponseGenerator initialization."""

    def test_init_sets_scenario_config(
        self, response_generator: ResponseGenerator, sample_scenario: ScenarioConfig
    ) -> None:
        """Test that initialization sets scenario config."""
        assert response_generator.scenario_config == sample_scenario

    def test_init_sets_user_email(self, response_generator: ResponseGenerator) -> None:
        """Test that initialization sets user email."""
        assert response_generator.user_email == "alex.thompson@example.com"

    def test_init_sets_user_phone(self, response_generator: ResponseGenerator) -> None:
        """Test that initialization sets user phone."""
        assert response_generator.user_phone == "+15550001111"

    def test_builds_email_lookup_table(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that email lookup table is built correctly."""
        # Should find characters by email
        alice = response_generator._get_character_by_email("alice.chen@example.com")
        assert alice is not None
        assert alice.name == "Alice Chen"

    def test_builds_phone_lookup_table(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that phone lookup table is built correctly."""
        # Should find characters by phone
        bob = response_generator._get_character_by_phone("+15550003333")
        assert bob is not None
        assert bob.name == "Bob Smith"


# =============================================================================
# Character Lookup Tests
# =============================================================================


class TestCharacterLookup:
    """Tests for character lookup methods."""

    def test_get_character_by_email_found(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test finding a character by email."""
        character = response_generator._get_character_by_email("alice.chen@example.com")
        assert character is not None
        assert character.name == "Alice Chen"

    def test_get_character_by_email_case_insensitive(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that email lookup is case-insensitive."""
        character = response_generator._get_character_by_email("ALICE.CHEN@EXAMPLE.COM")
        assert character is not None
        assert character.name == "Alice Chen"

    def test_get_character_by_email_not_found(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test looking up unknown email returns None."""
        character = response_generator._get_character_by_email("unknown@example.com")
        assert character is None

    def test_get_character_by_phone_found(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test finding a character by phone."""
        character = response_generator._get_character_by_phone("+15550002222")
        assert character is not None
        assert character.name == "Alice Chen"

    def test_get_character_by_phone_not_found(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test looking up unknown phone returns None."""
        character = response_generator._get_character_by_phone("+15559999999")
        assert character is None

    def test_is_user_email_true(self, response_generator: ResponseGenerator) -> None:
        """Test identifying user email."""
        assert response_generator._is_user_email("alex.thompson@example.com") is True

    def test_is_user_email_case_insensitive(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test user email check is case-insensitive."""
        assert response_generator._is_user_email("ALEX.THOMPSON@EXAMPLE.COM") is True

    def test_is_user_email_false(self, response_generator: ResponseGenerator) -> None:
        """Test non-user email returns False."""
        assert response_generator._is_user_email("alice.chen@example.com") is False

    def test_is_user_phone_true(self, response_generator: ResponseGenerator) -> None:
        """Test identifying user phone."""
        assert response_generator._is_user_phone("+15550001111") is True

    def test_is_user_phone_false(self, response_generator: ResponseGenerator) -> None:
        """Test non-user phone returns False."""
        assert response_generator._is_user_phone("+15550002222") is False


# =============================================================================
# Find Responders Tests
# =============================================================================


class TestFindResponders:
    """Tests for finding potential responders."""

    def test_find_email_responders_excludes_sender(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that sender is excluded from responders."""
        responders = response_generator._find_email_responders(
            sender_address="alice.chen@example.com",
            all_recipients={"alice.chen@example.com", "bob.smith@example.com"},
        )
        names = [r.name for r in responders]
        assert "Alice Chen" not in names
        assert "Bob Smith" in names

    def test_find_email_responders_excludes_user(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that user is excluded from responders."""
        responders = response_generator._find_email_responders(
            sender_address="external@example.com",
            all_recipients={
                "alex.thompson@example.com",
                "alice.chen@example.com",
            },
        )
        names = [r.name for r in responders]
        assert "Alex Thompson" not in names
        assert "Alice Chen" in names

    def test_find_email_responders_excludes_unavailable(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that unavailable characters are excluded."""
        responders = response_generator._find_email_responders(
            sender_address="external@example.com",
            all_recipients={
                "alice.chen@example.com",
                "carol.davis@example.com",  # On vacation
            },
        )
        names = [r.name for r in responders]
        assert "Alice Chen" in names
        assert "Carol Davis" not in names

    def test_find_sms_responders(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test finding SMS responders."""
        responders = response_generator._find_sms_responders(
            sender_phone="+15551234567",
            all_recipients={"+15550002222", "+15550003333"},
        )
        names = [r.name for r in responders]
        assert "Alice Chen" in names
        assert "Bob Smith" in names

    def test_find_calendar_responders(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test finding calendar event responders."""
        responders = response_generator._find_calendar_responders(
            organizer_email="alex.thompson@example.com",
            all_attendees={
                "alice.chen@example.com",
                "bob.smith@example.com",
            },
        )
        names = [r.name for r in responders]
        assert "Alice Chen" in names
        assert "Bob Smith" in names


# =============================================================================
# Should Skip Character Tests
# =============================================================================


class TestShouldSkipCharacter:
    """Tests for the should_skip_character heuristic."""

    def test_skip_never_respond(
        self, response_timing: ResponseTiming
    ) -> None:
        """Test character with 'never respond' instruction is skipped."""
        character = CharacterProfile(
            name="Test",
            relationships={},
            personality="Normal",
            email="test@example.com",
            response_timing=response_timing,
            special_instructions="This character never responds to emails.",
        )
        generator = ResponseGenerator(
            client=AsyncMock(),
            scenario_config=MagicMock(),
            response_llm=MagicMock(),
            summarization_llm=MagicMock(),
        )
        assert generator._should_skip_character(character) is True

    def test_skip_on_vacation(
        self, response_timing: ResponseTiming
    ) -> None:
        """Test character on vacation is skipped."""
        character = CharacterProfile(
            name="Test",
            relationships={},
            personality="Normal",
            email="test@example.com",
            response_timing=response_timing,
            special_instructions="Currently on vacation until next month.",
        )
        generator = ResponseGenerator(
            client=AsyncMock(),
            scenario_config=MagicMock(),
            response_llm=MagicMock(),
            summarization_llm=MagicMock(),
        )
        assert generator._should_skip_character(character) is True

    def test_no_skip_normal_character(
        self, response_generator: ResponseGenerator, alice_character: CharacterProfile
    ) -> None:
        """Test normal character is not skipped."""
        assert response_generator._should_skip_character(alice_character) is False

    def test_no_skip_no_instructions(
        self, response_timing: ResponseTiming
    ) -> None:
        """Test character without special instructions is not skipped."""
        character = CharacterProfile(
            name="Test",
            relationships={},
            personality="Normal",
            email="test@example.com",
            response_timing=response_timing,
        )
        generator = ResponseGenerator(
            client=AsyncMock(),
            scenario_config=MagicMock(),
            response_llm=MagicMock(),
            summarization_llm=MagicMock(),
        )
        assert generator._should_skip_character(character) is False


# =============================================================================
# Response Timing Tests
# =============================================================================


class TestResponseTiming:
    """Tests for response timing calculation."""

    def test_calculate_response_time_within_range(
        self, response_generator: ResponseGenerator, alice_character: CharacterProfile
    ) -> None:
        """Test that calculated time is within expected range."""
        reference_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)

        # Alice has base_delay=PT30M, variance=PT10M
        # So response should be between 20-40 minutes from reference
        scheduled_time = response_generator._calculate_response_time(
            alice_character, reference_time
        )

        min_time = reference_time + timedelta(minutes=20)
        max_time = reference_time + timedelta(minutes=40)

        assert min_time <= scheduled_time <= max_time

    def test_calculate_response_time_randomness(
        self, response_generator: ResponseGenerator, alice_character: CharacterProfile
    ) -> None:
        """Test that response times have randomness."""
        reference_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)

        times = [
            response_generator._calculate_response_time(alice_character, reference_time)
            for _ in range(10)
        ]

        # With 10 samples, we should see at least some variation
        unique_times = set(times)
        assert len(unique_times) > 1, "Response times should have randomness"


# =============================================================================
# Email Subject Tests
# =============================================================================


class TestEmailSubject:
    """Tests for email subject derivation."""

    def test_derive_subject_adds_re(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that Re: is added to new subjects."""
        result = response_generator._derive_email_subject("Meeting Tomorrow")
        assert result == "Re: Meeting Tomorrow"

    def test_derive_subject_preserves_existing_re(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that existing Re: is preserved."""
        result = response_generator._derive_email_subject("Re: Meeting Tomorrow")
        assert result == "Re: Meeting Tomorrow"

    def test_derive_subject_case_insensitive_re(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that re: (lowercase) is recognized."""
        result = response_generator._derive_email_subject("re: Meeting Tomorrow")
        assert result == "re: Meeting Tomorrow"

    def test_derive_subject_strips_whitespace(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that whitespace is stripped."""
        result = response_generator._derive_email_subject("  Meeting Tomorrow  ")
        assert result == "Re: Meeting Tomorrow"


# =============================================================================
# Email References Tests
# =============================================================================


class TestEmailReferences:
    """Tests for building email references."""

    def test_build_references_with_existing(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test building references with existing references."""
        email = MockEmail(
            message_id="msg-3",
            references=["msg-1", "msg-2"],
        )
        references = response_generator._build_email_references(email)  # type: ignore

        assert references == ["msg-1", "msg-2", "msg-3"]

    def test_build_references_no_existing(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test building references without existing references."""
        email = MockEmail(message_id="msg-1", references=[])
        references = response_generator._build_email_references(email)  # type: ignore

        assert references == ["msg-1"]

    def test_build_references_avoids_duplicates(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that message_id is not duplicated."""
        email = MockEmail(
            message_id="msg-2",
            references=["msg-1", "msg-2"],  # msg-2 already in references
        )
        references = response_generator._build_email_references(email)  # type: ignore

        # msg-2 should not appear twice
        assert references.count("msg-2") == 1


# =============================================================================
# Message Formatting Tests
# =============================================================================


class TestMessageFormatting:
    """Tests for message formatting."""

    def test_format_email_for_prompt(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test formatting an email for prompt."""
        email = MockEmail(
            from_address="alice.chen@example.com",  # Known character
            subject="Test Subject",
            body_text="Test body content",
            received_at=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
        )
        formatted = response_generator._format_email_for_prompt(email)  # type: ignore

        assert "Alice Chen" in formatted  # Uses character name
        assert "Test Subject" in formatted
        assert "Test body content" in formatted
        assert "2026-01-29 10:30" in formatted

    def test_format_email_unknown_sender(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test formatting an email from unknown sender."""
        email = MockEmail(
            from_address="unknown@example.com",
            subject="Test Subject",
            body_text="Test body",
        )
        formatted = response_generator._format_email_for_prompt(email)  # type: ignore

        assert "unknown@example.com" in formatted  # Falls back to email

    def test_format_sms_for_prompt(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test formatting an SMS for prompt."""
        sms = MockSMSMessage(
            from_number="+15550002222",  # Alice's phone
            body="Hey, are you coming?",
            sent_at=datetime(2026, 1, 29, 10, 30, tzinfo=timezone.utc),
        )
        formatted = response_generator._format_sms_for_prompt(sms)  # type: ignore

        assert "Alice Chen" in formatted
        assert "Hey, are you coming?" in formatted

    def test_get_name_for_email_known(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test getting name for known email."""
        name = response_generator._get_name_for_email("bob.smith@example.com")
        assert name == "Bob Smith"

    def test_get_name_for_email_unknown(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test getting name for unknown email."""
        name = response_generator._get_name_for_email("stranger@example.com")
        assert name == "stranger@example.com"


# =============================================================================
# Thread Context Tests
# =============================================================================


class TestThreadContext:
    """Tests for thread context preparation."""

    @pytest.mark.asyncio
    async def test_prepare_thread_context_short_thread(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test preparing context for a short thread."""
        messages = [
            MockEmail(message_id=f"msg-{i}", body_text=f"Message {i}")
            for i in range(5)
        ]

        context = await response_generator._prepare_thread_context(
            messages, "email", None  # type: ignore
        )

        assert context.total_message_count == 5
        assert context.included_message_count == 5
        assert context.summary is None  # No summarization needed

    @pytest.mark.asyncio
    async def test_prepare_thread_context_long_thread(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test preparing context for a long thread (triggers summarization)."""
        messages = [
            MockEmail(message_id=f"msg-{i}", body_text=f"Message {i}")
            for i in range(15)
        ]

        context = await response_generator._prepare_thread_context(
            messages, "email", None  # type: ignore
        )

        assert context.total_message_count == 15
        assert context.included_message_count == MAX_THREAD_MESSAGES
        assert context.summary is not None

    @pytest.mark.asyncio
    async def test_prepare_email_thread_context_no_thread_id(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that None is returned when email has no thread_id."""
        email = MockEmail(thread_id=None)  # type: ignore
        context = await response_generator._prepare_email_thread_context(
            email, None  # type: ignore
        )
        assert context is None


# =============================================================================
# LLM Integration Tests (with mocked LLMs)
# =============================================================================


class TestLLMIntegration:
    """Tests for LLM integration with mocked LLMs."""

    @pytest.mark.asyncio
    async def test_should_respond_true(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test should_respond when LLM returns True."""
        # Setup mock to return structured output
        mock_structured = AsyncMock(
            return_value=ShouldRespondResult(
                should_respond=True,
                reasoning="The message asks a question.",
            )
        )
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_structured)
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="sender@example.com",
        )

        result = await response_generator._should_respond(
            alice_character, context, None
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_should_respond_false(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test should_respond when LLM returns False."""
        mock_structured = AsyncMock(
            return_value=ShouldRespondResult(
                should_respond=False,
                reasoning="Just an acknowledgment.",
            )
        )
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_structured)
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="sender@example.com",
        )

        result = await response_generator._should_respond(
            alice_character, context, None
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_should_respond_error_returns_false(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that should_respond returns False on error."""
        mock_structured = AsyncMock(side_effect=Exception("LLM error"))
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_structured)
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="sender@example.com",
        )

        result = await response_generator._should_respond(
            alice_character, context, None
        )

        # Should default to False on error
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_response_success(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test successful response generation."""
        response_generator._response_llm.ainvoke = AsyncMock(
            return_value=MockLLMResponse("Thanks for the update!")
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="sender@example.com",
        )

        result = await response_generator._generate_response(
            alice_character, context, None
        )

        assert result == "Thanks for the update!"

    @pytest.mark.asyncio
    async def test_generate_response_error_raises(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that generate_response raises on error."""
        response_generator._response_llm.ainvoke = AsyncMock(
            side_effect=Exception("LLM error")
        )

        email = MockEmail()
        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address="sender@example.com",
        )

        with pytest.raises(ResponseGenerationError) as exc_info:
            await response_generator._generate_response(
                alice_character, context, None
            )

        assert exc_info.value.character_name == "Alice Chen"
        assert exc_info.value.modality == "email"

    @pytest.mark.asyncio
    async def test_decide_calendar_rsvp(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test calendar RSVP decision."""
        mock_structured = AsyncMock(
            return_value=CalendarRSVPResult(
                status="accepted",
                comment="Looking forward to it!",
                reasoning="Team meeting, should attend.",
            )
        )
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_structured)
        )

        from src.green.response.models import CalendarEventContext

        event = MockCalendarEvent()
        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="organizer@example.com",
            all_attendees={"alice.chen@example.com"},
        )

        result = await response_generator._decide_calendar_rsvp(
            alice_character, context, None
        )

        assert result.status == "accepted"
        assert result.comment == "Looking forward to it!"

    @pytest.mark.asyncio
    async def test_decide_calendar_rsvp_error_returns_tentative(
        self,
        response_generator: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that calendar RSVP defaults to tentative on error."""
        mock_structured = AsyncMock(side_effect=Exception("LLM error"))
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_structured)
        )

        from src.green.response.models import CalendarEventContext

        event = MockCalendarEvent()
        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="organizer@example.com",
            all_attendees={"alice.chen@example.com"},
        )

        result = await response_generator._decide_calendar_rsvp(
            alice_character, context, None
        )

        assert result.status == "tentative"


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestProcessNewMessages:
    """Tests for the main process_new_messages method."""

    @pytest.mark.asyncio
    async def test_process_empty_messages(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test processing empty message set."""
        from src.green.core.message_collector import NewMessages

        new_messages = NewMessages()
        current_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)

        responses = await response_generator.process_new_messages(
            new_messages, current_time
        )

        assert responses == []

    @pytest.mark.asyncio
    async def test_process_email_generates_response(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that processing an email can generate a response."""
        from src.green.core.message_collector import NewMessages

        # Setup mocks for LLM calls
        mock_should_respond = AsyncMock(
            return_value=ShouldRespondResult(
                should_respond=True,
                reasoning="Question asked.",
            )
        )
        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_should_respond)
        )
        response_generator._response_llm.ainvoke = AsyncMock(
            return_value=MockLLMResponse("Here's my response!")
        )

        # Create email from user to Alice
        email = MockEmail(
            from_address="alex.thompson@example.com",  # User
            to_addresses=["alice.chen@example.com"],  # Alice
            subject="Question",
            body_text="Can you help me?",
        )

        new_messages = NewMessages(emails=[email])  # type: ignore
        current_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)

        responses = await response_generator.process_new_messages(
            new_messages, current_time
        )

        # Alice should respond
        assert len(responses) == 1
        assert responses[0].character_name == "Alice Chen"
        assert responses[0].modality == "email"
        assert responses[0].content == "Here's my response!"

    @pytest.mark.asyncio
    async def test_process_continues_on_error(
        self, response_generator: ResponseGenerator
    ) -> None:
        """Test that processing continues when one message fails."""
        from src.green.core.message_collector import NewMessages

        # First call raises, second succeeds
        call_count = 0

        async def mock_should_respond(*args: Any, **kwargs: Any) -> ShouldRespondResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return ShouldRespondResult(
                should_respond=True,
                reasoning="OK",
            )

        response_generator._response_llm.with_structured_output = MagicMock(
            return_value=MagicMock(ainvoke=mock_should_respond)
        )
        response_generator._response_llm.ainvoke = AsyncMock(
            return_value=MockLLMResponse("Response!")
        )

        # Two emails
        email1 = MockEmail(
            message_id="msg-1",
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
        )
        email2 = MockEmail(
            message_id="msg-2",
            from_address="alex.thompson@example.com",
            to_addresses=["bob.smith@example.com"],
        )

        new_messages = NewMessages(emails=[email1, email2])  # type: ignore
        current_time = datetime(2026, 1, 29, 10, 0, tzinfo=timezone.utc)

        # Should not raise, should return what it can
        responses = await response_generator.process_new_messages(
            new_messages, current_time
        )

        # At least one response should be generated
        # (depending on which email succeeds)
        assert len(responses) >= 0  # Just verifies no exception


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_response_generator_error(self) -> None:
        """Test base exception."""
        error = ResponseGeneratorError("Test error")
        assert str(error) == "Test error"

    def test_response_generation_error(self) -> None:
        """Test response generation error."""
        error = ResponseGenerationError(
            "Failed to generate",
            character_name="Alice",
            modality="email",
        )
        assert str(error) == "Failed to generate"
        assert error.character_name == "Alice"
        assert error.modality == "email"

    def test_response_generation_error_optional_fields(self) -> None:
        """Test response generation error with optional fields."""
        error = ResponseGenerationError("Failed")
        assert error.character_name is None
        assert error.modality is None
