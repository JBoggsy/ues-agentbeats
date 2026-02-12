"""Tests for GreenAgent Purple communication methods.

Tests cover:
- _send_and_wait_purple: sending messages and parsing responses
- _extract_response_data: extracting DataPart from A2A responses
- _send_assessment_start: sending assessment start messages
- _send_assessment_complete: sending assessment complete messages
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import (
    Artifact,
    DataPart,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

from src.common.agentbeats.messages import (
    AssessmentCompleteMessage,
    AssessmentStartMessage,
    CalendarSummary,
    ChatSummary,
    EarlyCompletionMessage,
    EmailSummary,
    InitialStateSummary,
    SMSSummary,
    TurnCompleteMessage,
    TurnStartMessage,
)
from src.green.agent import GreenAgent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ues_client() -> AsyncMock:
    """Create a mock UES client for testing."""
    client = AsyncMock()
    # Mock time.get_state to return a valid time state
    time_state = MagicMock()
    time_state.current_time = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)
    client.time.get_state = AsyncMock(return_value=time_state)
    return client


@pytest.fixture
def mock_purple_client() -> AsyncMock:
    """Create a mock A2A client wrapper for Purple agent."""
    client = AsyncMock()
    return client


@pytest.fixture
def green_agent(mock_ues_client: AsyncMock) -> GreenAgent:
    """Create a GreenAgent with mocked dependencies for testing.

    This bypasses __init__ by creating an uninitialized instance and
    setting the required attributes directly.
    """
    # Create instance without calling __init__
    agent = object.__new__(GreenAgent)
    agent.ues_client = mock_ues_client
    agent.ues_port = 8100
    return agent


@pytest.fixture
def sample_initial_summary() -> InitialStateSummary:
    """Create a sample initial state summary for testing."""
    return InitialStateSummary(
        email=EmailSummary(total_emails=10, total_threads=5, unread=3, draft_count=1),
        calendar=CalendarSummary(event_count=5, calendar_count=2, events_today=2),
        sms=SMSSummary(total_messages=20, total_conversations=3, unread=2),
        chat=ChatSummary(total_messages=15, conversation_count=1),
    )


@pytest.fixture
def sample_scenario() -> MagicMock:
    """Create a sample scenario config for testing."""
    scenario = MagicMock()
    scenario.user_prompt = "Please triage all unread emails."
    return scenario


# =============================================================================
# Tests for _extract_response_data
# =============================================================================


class TestExtractResponseData:
    """Tests for the _extract_response_data method."""

    def test_extract_from_task_with_artifacts(
        self, green_agent: GreenAgent
    ) -> None:
        """Test extraction from Task with artifacts containing DataPart."""
        data = {"message_type": "turn_complete", "notes": "Done!"}
        task = Task(
            id="task-123",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[
                Artifact(
                    artifactId="art-1",
                    parts=[Part(root=DataPart(data=data))],
                )
            ],
        )

        result = green_agent._extract_response_data(task)
        assert result == data

    def test_extract_from_task_with_history(
        self, green_agent: GreenAgent
    ) -> None:
        """Test extraction from Task with history messages."""
        data = {"message_type": "early_completion", "reason": "All done"}
        task = Task(
            id="task-123",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
            history=[
                Message(
                    role=Role.user,
                    parts=[Part(root=DataPart(data={"message_type": "turn_start"}))],
                    message_id="msg-1",
                ),
                Message(
                    role=Role.agent,
                    parts=[Part(root=DataPart(data=data))],
                    message_id="msg-2",
                ),
            ],
        )

        result = green_agent._extract_response_data(task)
        assert result == data

    def test_extract_from_message(self, green_agent: GreenAgent) -> None:
        """Test extraction from Message with DataPart."""
        data = {"message_type": "turn_complete", "time_step": "PT30M"}
        message = Message(
            role=Role.agent,
            parts=[Part(root=DataPart(data=data))],
            message_id="msg-1",
        )

        result = green_agent._extract_response_data(message)
        assert result == data

    def test_extract_raises_for_empty_task(
        self, green_agent: GreenAgent
    ) -> None:
        """Test that ValueError is raised for Task without DataPart."""
        task = Task(
            id="task-123",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
        )

        with pytest.raises(ValueError, match="No DataPart found"):
            green_agent._extract_response_data(task)

    def test_extract_raises_for_empty_message(
        self, green_agent: GreenAgent
    ) -> None:
        """Test that ValueError is raised for Message without DataPart."""
        message = Message(
            role=Role.agent,
            parts=[],
            message_id="msg-1",
        )

        with pytest.raises(ValueError, match="No DataPart found"):
            green_agent._extract_response_data(message)

    def test_extract_raises_for_unknown_type(
        self, green_agent: GreenAgent
    ) -> None:
        """Test that ValueError is raised for unknown response types."""
        response = {"some": "dict"}

        with pytest.raises(ValueError, match="Unable to extract DataPart"):
            green_agent._extract_response_data(response)


# =============================================================================
# Tests for _send_and_wait_purple
# =============================================================================


class TestSendAndWaitPurple:
    """Tests for the _send_and_wait_purple method."""

    @pytest.mark.asyncio
    async def test_send_and_wait_returns_turn_complete(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test sending a message and receiving TurnCompleteMessage."""
        response_data = {
            "message_type": "turn_complete",
            "notes": "Processed emails",
            "time_step": "PT1H",
        }
        mock_task = Task(
            id="task-1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[Artifact(artifactId="art-1", parts=[Part(root=DataPart(data=response_data))])],
        )
        mock_purple_client.send_message = AsyncMock(return_value=mock_task)

        message = TurnStartMessage(
            turn_number=1,
            current_time=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            events_processed=0,
        )

        result = await green_agent._send_and_wait_purple(
            purple_client=mock_purple_client,
            message=message,
            timeout=30.0,
        )

        assert isinstance(result, TurnCompleteMessage)
        assert result.notes == "Processed emails"
        assert result.time_step == "PT1H"
        mock_purple_client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_and_wait_returns_early_completion(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test sending a message and receiving EarlyCompletionMessage."""
        response_data = {
            "message_type": "early_completion",
            "reason": "All tasks completed",
        }
        mock_task = Task(
            id="task-1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[Artifact(artifactId="art-2", parts=[Part(root=DataPart(data=response_data))])],
        )
        mock_purple_client.send_message = AsyncMock(return_value=mock_task)

        message = TurnStartMessage(
            turn_number=3,
            current_time=datetime(2026, 2, 9, 15, 0, 0, tzinfo=timezone.utc),
            events_processed=10,
        )

        result = await green_agent._send_and_wait_purple(
            purple_client=mock_purple_client,
            message=message,
            timeout=30.0,
        )

        assert isinstance(result, EarlyCompletionMessage)
        assert result.reason == "All tasks completed"

    @pytest.mark.asyncio
    async def test_send_and_wait_raises_on_timeout(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test that TimeoutError is raised when Purple doesn't respond."""

        async def slow_response(*args: Any, **kwargs: Any) -> Task:
            await asyncio.sleep(10)  # Long delay
            return Task(
                id="task-1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.completed),
            )

        mock_purple_client.send_message = slow_response

        message = TurnStartMessage(
            turn_number=1,
            current_time=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            events_processed=0,
        )

        with pytest.raises(asyncio.TimeoutError):
            await green_agent._send_and_wait_purple(
                purple_client=mock_purple_client,
                message=message,
                timeout=0.1,  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_send_and_wait_raises_on_unexpected_message_type(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test that ValueError is raised for unexpected message types."""
        response_data = {
            "message_type": "unknown_type",
            "data": "something",
        }
        mock_task = Task(
            id="task-1",
            context_id="ctx-1",
            status=TaskStatus(state=TaskState.completed),
            artifacts=[Artifact(artifactId="art-3", parts=[Part(root=DataPart(data=response_data))])],
        )
        mock_purple_client.send_message = AsyncMock(return_value=mock_task)

        message = TurnStartMessage(
            turn_number=1,
            current_time=datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc),
            events_processed=0,
        )

        with pytest.raises(ValueError, match="Unexpected message type"):
            await green_agent._send_and_wait_purple(
                purple_client=mock_purple_client,
                message=message,
                timeout=30.0,
            )


# =============================================================================
# Tests for _send_assessment_start
# =============================================================================


class TestSendAssessmentStart:
    """Tests for the _send_assessment_start method."""

    @pytest.mark.asyncio
    async def test_send_assessment_start_calls_send_data(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
        sample_scenario: MagicMock,
        sample_initial_summary: InitialStateSummary,
    ) -> None:
        """Test that assessment start sends data with correct structure."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_start(
            purple_client=mock_purple_client,
            scenario=sample_scenario,
            initial_summary=sample_initial_summary,
            ues_url="http://127.0.0.1:8100",
            api_key="secret-key-123",
        )

        mock_purple_client.send_data.assert_called_once()
        call_kwargs = mock_purple_client.send_data.call_args.kwargs
        assert call_kwargs["blocking"] is False

        data = call_kwargs["data"]
        assert data["message_type"] == "assessment_start"
        assert data["ues_url"] == "http://127.0.0.1:8100"
        assert data["api_key"] == "secret-key-123"
        assert data["assessment_instructions"] == "Please triage all unread emails."
        assert "initial_state_summary" in data

    @pytest.mark.asyncio
    async def test_send_assessment_start_includes_current_time(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
        mock_ues_client: AsyncMock,
        sample_scenario: MagicMock,
        sample_initial_summary: InitialStateSummary,
    ) -> None:
        """Test that assessment start includes current simulation time."""
        mock_purple_client.send_data = AsyncMock()
        expected_time = datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)

        await green_agent._send_assessment_start(
            purple_client=mock_purple_client,
            scenario=sample_scenario,
            initial_summary=sample_initial_summary,
            ues_url="http://127.0.0.1:8100",
            api_key="secret-key",
        )

        data = mock_purple_client.send_data.call_args.kwargs["data"]
        assert "current_time" in data
        # Time is serialized as ISO 8601 string
        assert expected_time.isoformat().startswith(
            data["current_time"][:19]
        ) or data["current_time"].startswith("2026-02-09")


# =============================================================================
# Tests for _send_assessment_complete
# =============================================================================


class TestSendAssessmentComplete:
    """Tests for the _send_assessment_complete method."""

    @pytest.mark.asyncio
    async def test_send_assessment_complete_scenario_complete(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test sending assessment complete with scenario_complete reason."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_complete(
            purple_client=mock_purple_client,
            reason="scenario_complete",
        )

        mock_purple_client.send_data.assert_called_once()
        call_kwargs = mock_purple_client.send_data.call_args.kwargs
        assert call_kwargs["blocking"] is False

        data = call_kwargs["data"]
        assert data["message_type"] == "assessment_complete"
        assert data["reason"] == "scenario_complete"

    @pytest.mark.asyncio
    async def test_send_assessment_complete_early_completion(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test sending assessment complete with early_completion reason."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_complete(
            purple_client=mock_purple_client,
            reason="early_completion",
        )

        data = mock_purple_client.send_data.call_args.kwargs["data"]
        assert data["reason"] == "early_completion"

    @pytest.mark.asyncio
    async def test_send_assessment_complete_max_turns_maps_to_timeout(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test that max_turns_reached maps to timeout reason."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_complete(
            purple_client=mock_purple_client,
            reason="max_turns_reached",
        )

        data = mock_purple_client.send_data.call_args.kwargs["data"]
        assert data["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_send_assessment_complete_cancelled_maps_to_timeout(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test that cancelled maps to timeout reason."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_complete(
            purple_client=mock_purple_client,
            reason="cancelled",
        )

        data = mock_purple_client.send_data.call_args.kwargs["data"]
        assert data["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_send_assessment_complete_error_reason(
        self,
        green_agent: GreenAgent,
        mock_purple_client: AsyncMock,
    ) -> None:
        """Test sending assessment complete with error reason."""
        mock_purple_client.send_data = AsyncMock()

        await green_agent._send_assessment_complete(
            purple_client=mock_purple_client,
            reason="error",
        )

        data = mock_purple_client.send_data.call_args.kwargs["data"]
        assert data["reason"] == "error"
