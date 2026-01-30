"""Integration tests for ResponseGenerator with real LLMs.

These tests use actual LLM calls to verify the response generation logic
works correctly with real models. They are marked with pytest markers to
allow selective execution:

- ``pytest -m ollama`` - Run only Ollama tests
- ``pytest -m openai`` - Run only OpenAI tests  
- ``pytest -m integration`` - Run all integration tests

Tests are automatically skipped if:
- The required environment variables are not set (OPENAI_API_KEY for OpenAI)
- The Ollama server is not running (for Ollama tests)

Environment variables can be set in a ``.env`` file in the project root:

    OPENAI_API_KEY=sk-...

Example:
    # Run Ollama tests only
    uv run pytest tests/green/test_response_generator_integration.py -m ollama -v

    # Run OpenAI tests (requires OPENAI_API_KEY in .env or environment)
    uv run pytest tests/green/test_response_generator_integration.py -m openai -v

    # Run all integration tests
    uv run pytest tests/green/test_response_generator_integration.py -m integration -v
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from dotenv import load_dotenv

from src.green.llm_config import LLMFactory
from src.green.response_generator import ResponseGenerator
from src.green.response_models import (
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
# Load environment variables from .env file
# =============================================================================

# Find project root (where .env file should be)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

# Load .env file if it exists
load_dotenv(_ENV_FILE)


# =============================================================================
# Skip Conditions
# =============================================================================


def ollama_available() -> bool:
    """Check if Ollama server is running and accessible."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def openai_key_available() -> bool:
    """Check if OpenAI API key is available (from .env or environment)."""
    return bool(os.environ.get("OPENAI_API_KEY"))


# Pytest markers for conditional skipping
requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama server not running at localhost:11434",
)

requires_openai = pytest.mark.skipif(
    not openai_key_available(),
    reason="OPENAI_API_KEY environment variable not set",
)


# =============================================================================
# Mock Objects (same as unit tests, for UES client mocking)
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
    """Create Alice character profile - friendly colleague."""
    return CharacterProfile(
        name="Alice Chen",
        relationships={"Alex Thompson": "colleague and friend"},
        personality=(
            "Alice is a friendly software engineer who enjoys helping others. "
            "She's known for her clear communication style and positive attitude. "
            "She works on the backend team and is very knowledgeable about APIs."
        ),
        email="alice.chen@example.com",
        phone="+15550002222",
        response_timing=response_timing,
        special_instructions="Alice tends to respond promptly to work questions.",
    )


@pytest.fixture
def bob_character(response_timing: ResponseTiming) -> CharacterProfile:
    """Create Bob character profile - formal manager."""
    return CharacterProfile(
        name="Bob Smith",
        relationships={"Alex Thompson": "direct report"},
        personality=(
            "Bob is Alex's manager. He's professional and thorough in his "
            "communications. He appreciates detailed updates and clear action items. "
            "Bob has 15 years of experience in software engineering management."
        ),
        email="bob.smith@example.com",
        phone="+15550003333",
        response_timing=response_timing,
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
        },
        initial_state={"environment": {}},
        criteria=[sample_criterion],
    )


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock UES client."""
    client = AsyncMock()
    client.email.query = AsyncMock(return_value=MockEmailQueryResponse())
    client.sms.query = AsyncMock(return_value=MockSMSQueryResponse())
    return client


# =============================================================================
# Ollama Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.ollama
@requires_ollama
class TestOllamaIntegration:
    """Integration tests using local Ollama model (gemma3:12b)."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an Ollama LLM instance."""
        return LLMFactory.create("ollama/gemma3:12b", temperature=0.3)

    @pytest.fixture
    def response_generator_ollama(
        self,
        mock_ues_client: AsyncMock,
        sample_scenario: ScenarioConfig,
        ollama_llm,
    ) -> ResponseGenerator:
        """Create a ResponseGenerator with Ollama LLM."""
        return ResponseGenerator(
            client=mock_ues_client,
            scenario_config=sample_scenario,
            response_llm=ollama_llm,
            summarization_llm=ollama_llm,
        )

    @pytest.mark.asyncio
    async def test_should_respond_to_question(
        self,
        response_generator_ollama: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM correctly identifies a question that needs response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="Question about API",
            body_text=(
                "Hi Alice,\n\n"
                "I'm working on the new feature and I'm stuck on something. "
                "Could you explain how the authentication endpoint works? "
                "I need to understand the token refresh flow.\n\n"
                "Thanks,\nAlex"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        result = await response_generator_ollama._should_respond(
            alice_character, context, None
        )

        # Alice should respond to a direct question about her area
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_respond_to_fyi(
        self,
        response_generator_ollama: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM correctly identifies FYI message not needing response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="FYI: Deployment complete",
            body_text=(
                "Hi Alice,\n\n"
                "Just wanted to let you know the deployment to staging "
                "completed successfully. No action needed.\n\n"
                "Best,\nAlex"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        result = await response_generator_ollama._should_respond(
            alice_character, context, None
        )

        # FYI messages typically don't need a response
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_response_content(
        self,
        response_generator_ollama: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM generates appropriate in-character response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="Quick question",
            body_text=(
                "Hey Alice,\n\n"
                "What time is the standup meeting tomorrow?\n\n"
                "Thanks!"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        response = await response_generator_ollama._generate_response(
            alice_character, context, None
        )

        # Response should exist and be reasonable
        assert response is not None
        assert len(response) > 10  # Should be more than just a word
        assert len(response) < 2000  # Should be reasonable length
        # Friendly Alice should have warm tone
        assert any(
            greeting in response.lower()
            for greeting in ["hi", "hey", "hello", "sure", "of course", "happy"]
        ) or "standup" in response.lower()

    @pytest.mark.asyncio
    async def test_calendar_rsvp_decision(
        self,
        response_generator_ollama: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM makes reasonable calendar RSVP decision."""
        from src.green.response_models import CalendarEventContext

        event = MockCalendarEvent(
            title="Team Planning Session",
            description="Weekly planning session to discuss upcoming sprint tasks.",
            start=datetime(2026, 1, 30, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 1, 30, 11, 0, tzinfo=timezone.utc),
            organizer="bob.smith@example.com",
            attendees=[MockAttendee(email="alice.chen@example.com")],
        )

        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="bob.smith@example.com",
            all_attendees={"alice.chen@example.com"},
        )

        result = await response_generator_ollama._decide_calendar_rsvp(
            alice_character, context, None
        )

        # Should return a valid RSVP status
        assert result.status in ["accepted", "declined", "tentative"]
        assert result.reasoning is not None
        assert len(result.reasoning) > 10


# =============================================================================
# OpenAI Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.openai
@requires_openai
class TestOpenAIIntegration:
    """Integration tests using OpenAI gpt-4o-mini."""

    @pytest.fixture
    def openai_llm(self):
        """Create an OpenAI LLM instance."""
        return LLMFactory.create("gpt-4o-mini", temperature=0.3)

    @pytest.fixture
    def response_generator_openai(
        self,
        mock_ues_client: AsyncMock,
        sample_scenario: ScenarioConfig,
        openai_llm,
    ) -> ResponseGenerator:
        """Create a ResponseGenerator with OpenAI LLM."""
        return ResponseGenerator(
            client=mock_ues_client,
            scenario_config=sample_scenario,
            response_llm=openai_llm,
            summarization_llm=openai_llm,
        )

    @pytest.mark.asyncio
    async def test_should_respond_to_question(
        self,
        response_generator_openai: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM correctly identifies a question that needs response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="Question about API",
            body_text=(
                "Hi Alice,\n\n"
                "I'm working on the new feature and I'm stuck on something. "
                "Could you explain how the authentication endpoint works? "
                "I need to understand the token refresh flow.\n\n"
                "Thanks,\nAlex"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        result = await response_generator_openai._should_respond(
            alice_character, context, None
        )

        # Alice should respond to a direct question about her area
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_respond_to_fyi(
        self,
        response_generator_openai: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM correctly identifies FYI message not needing response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="FYI: Deployment complete",
            body_text=(
                "Hi Alice,\n\n"
                "Just wanted to let you know the deployment to staging "
                "completed successfully. No action needed.\n\n"
                "Best,\nAlex"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        result = await response_generator_openai._should_respond(
            alice_character, context, None
        )

        # FYI messages typically don't need a response
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_response_content(
        self,
        response_generator_openai: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM generates appropriate in-character response."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["alice.chen@example.com"],
            subject="Quick question",
            body_text=(
                "Hey Alice,\n\n"
                "What time is the standup meeting tomorrow?\n\n"
                "Thanks!"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        response = await response_generator_openai._generate_response(
            alice_character, context, None
        )

        # Response should exist and be reasonable
        assert response is not None
        assert len(response) > 10  # Should be more than just a word
        assert len(response) < 2000  # Should be reasonable length

    @pytest.mark.asyncio
    async def test_calendar_rsvp_decision(
        self,
        response_generator_openai: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that LLM makes reasonable calendar RSVP decision."""
        from src.green.response_models import CalendarEventContext

        event = MockCalendarEvent(
            title="Team Planning Session",
            description="Weekly planning session to discuss upcoming sprint tasks.",
            start=datetime(2026, 1, 30, 10, 0, tzinfo=timezone.utc),
            end=datetime(2026, 1, 30, 11, 0, tzinfo=timezone.utc),
            organizer="bob.smith@example.com",
            attendees=[MockAttendee(email="alice.chen@example.com")],
        )

        context = CalendarEventContext(
            event=event,  # type: ignore
            organizer_email="bob.smith@example.com",
            all_attendees={"alice.chen@example.com"},
        )

        result = await response_generator_openai._decide_calendar_rsvp(
            alice_character, context, None
        )

        # Should return a valid RSVP status
        assert result.status in ["accepted", "declined", "tentative"]
        assert result.reasoning is not None
        assert len(result.reasoning) > 10

    @pytest.mark.asyncio
    async def test_sms_response_informal_tone(
        self,
        response_generator_openai: ResponseGenerator,
        alice_character: CharacterProfile,
    ) -> None:
        """Test that SMS responses have appropriate informal tone."""
        sms = MockSMSMessage(
            from_number="+15550001111",  # Alex's phone
            to_numbers=["+15550002222"],  # Alice's phone
            body="Hey, running 5 min late to lunch. Go ahead and order for me?",
        )

        context = MessageContext(
            message=sms,  # type: ignore
            modality="sms",
            sender_address=sms.from_number,
            all_recipients=set(sms.to_numbers),
        )

        response = await response_generator_openai._generate_response(
            alice_character, context, None
        )

        # SMS responses should be shorter and more casual
        assert response is not None
        assert len(response) < 500  # SMS should be concise
        # Should not have formal email structure
        assert "Dear" not in response
        assert "Sincerely" not in response

    @pytest.mark.asyncio
    async def test_manager_response_formal_tone(
        self,
        response_generator_openai: ResponseGenerator,
        bob_character: CharacterProfile,
    ) -> None:
        """Test that manager character uses appropriate formal tone."""
        email = MockEmail(
            from_address="alex.thompson@example.com",
            to_addresses=["bob.smith@example.com"],
            subject="Vacation Request",
            body_text=(
                "Hi Bob,\n\n"
                "I'd like to request time off from February 15-20 for "
                "a family vacation. I've ensured all my projects will be "
                "covered during that time.\n\n"
                "Let me know if this works.\n\n"
                "Thanks,\nAlex"
            ),
        )

        context = MessageContext(
            message=email,  # type: ignore
            modality="email",
            sender_address=email.from_address,
            all_recipients=set(email.to_addresses),
        )

        response = await response_generator_openai._generate_response(
            bob_character, context, None
        )

        # Manager response should exist and be professional
        assert response is not None
        assert len(response) > 20
        # Should address the vacation request topic
        assert any(
            word in response.lower()
            for word in ["vacation", "time off", "request", "approved", "dates", "february"]
        )


# =============================================================================
# Thread Summarization Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.openai
@requires_openai
class TestThreadSummarization:
    """Tests for thread summarization with real LLMs."""

    @pytest.fixture
    def openai_llm(self):
        """Create an OpenAI LLM instance."""
        return LLMFactory.create("gpt-4o-mini", temperature=0.2)

    @pytest.fixture
    def response_generator(
        self,
        mock_ues_client: AsyncMock,
        sample_scenario: ScenarioConfig,
        openai_llm,
    ) -> ResponseGenerator:
        """Create a ResponseGenerator with OpenAI LLM."""
        return ResponseGenerator(
            client=mock_ues_client,
            scenario_config=sample_scenario,
            response_llm=openai_llm,
            summarization_llm=openai_llm,
        )

    @pytest.mark.asyncio
    async def test_summarize_long_thread(
        self,
        response_generator: ResponseGenerator,
    ) -> None:
        """Test summarization of a long email thread."""
        # Create a realistic long thread
        thread_emails = [
            MockEmail(
                message_id=f"msg-{i}",
                from_address="alex.thompson@example.com" if i % 2 == 0 else "alice.chen@example.com",
                subject="Re: Project Timeline Discussion",
                body_text=f"Message {i}: {'Proposing timeline change' if i % 2 == 0 else 'Discussing requirements'}",
                received_at=datetime(2026, 1, 29, 9 + i, 0, tzinfo=timezone.utc),
            )
            for i in range(12)  # More than MAX_THREAD_MESSAGES
        ]

        context = await response_generator._prepare_thread_context(
            thread_emails,  # type: ignore
            "email",
            None,
        )

        # Should have summarized earlier messages
        assert context is not None
        assert context.summary is not None
        assert len(context.summary) > 20
        assert context.total_message_count == 12
        assert context.included_message_count < 12  # Not all messages included
